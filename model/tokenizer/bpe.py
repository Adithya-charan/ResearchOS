"""Byte-level Byte Pair Encoding tokenizer, implemented from scratch.

Design
------
* Text is Unicode-NFC normalised, split into "pieces" by a fixed regex
  (letters / single digits / punctuation / whitespace), and each piece is
  converted to UTF-8 bytes. Merges never cross piece boundaries.
* IDs 0..5 are reserved special tokens, IDs 6..261 are the 256 raw bytes, and
  every learned merge gets the next ID in order. Because every byte has a token,
  `encode` can never fail and `<UNK>` is never produced for text input.
* Training is deterministic: pieces are processed in sorted order and ties are
  broken by the lowest (left_id, right_id) pair.
* `fingerprint` hashes everything that defines the ID space. Checkpoints store
  it so a changed tokenizer can never be silently paired with old weights.
"""
from __future__ import annotations

import hashlib
import heapq
import json
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from model.common import atomic_write_json, atomic_write_text, canonical_json, sha256_bytes

FORMAT_VERSION = 1
SPECIAL_TOKENS: tuple[str, ...] = ("<PAD>", "<UNK>", "<BOS>", "<EOS>", "<DOC>", "<SEP>")
PAD_ID, UNK_ID, BOS_ID, EOS_ID, DOC_ID, SEP_ID = range(len(SPECIAL_TOKENS))
NUM_SPECIAL = len(SPECIAL_TOKENS)
BYTE_OFFSET = NUM_SPECIAL  # token id of byte 0x00
BASE_VOCAB_SIZE = NUM_SPECIAL + 256
NORMALIZATION = "NFC"

# letters | single digit | punctuation/symbols/underscore | trailing-space-aware whitespace | anything else.
# Each alternative may take one leading space so " word" is a single piece.
PRETOKENIZE_PATTERN = r" ?[^\W\d_]+| ?\d| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+|."
_PRETOKENIZE_RE = re.compile(PRETOKENIZE_PATTERN, re.DOTALL)
_SPECIAL_RE = re.compile("(" + "|".join(re.escape(t) for t in SPECIAL_TOKENS) + ")")
_SPECIAL_TO_ID = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
_CACHE_LIMIT = 500_000


def normalize(text: str) -> str:
    return unicodedata.normalize(NORMALIZATION, text)


def pretokenize(text: str) -> list[str]:
    """Split already-normalised text into pieces (legacy "v1"). Concatenating the result restores `text`."""
    return _PRETOKENIZE_RE.findall(text)


# "v2": identical to v1 except that combining marks (Unicode categories Mn/Mc/Me: Telugu and other Indic vowel signs,
# viramas, Arabic/Hebrew points, diacritics) and the zero-width (non-)joiners count as part of a word. Under v1 a Telugu
# word is shredded at every vowel sign, so no merge can ever span a consonant and its vowel sign.
PRETOKENIZERS = ("v1", "v2")
DEFAULT_PRETOKENIZER = "v1"  # kept so every tokenizer fingerprint and checkpoint written before P8 stays valid
_V2_RE: re.Pattern[str] | None = None


def _v2_regex() -> re.Pattern[str]:
    global _V2_RE
    if _V2_RE is None:
        ranges: list[tuple[int, int]] = []
        start = prev = None
        for cp in list(range(0x300, 0x30000)) + list(range(0xE0000, 0xE0200)):
            if unicodedata.category(chr(cp)) in ("Mn", "Mc", "Me") or cp in (0x200C, 0x200D):
                if start is None:
                    start = prev = cp
                elif cp == prev + 1:
                    prev = cp
                else:
                    ranges.append((start, prev))
                    start = prev = cp
        if start is not None:
            ranges.append((start, prev))
        marks = "".join(f"\\U{a:08X}-\\U{b:08X}" for a, b in ranges)
        pattern = (
            r" ?(?:[^\W\d_]|[" + marks + r"])+| ?\d| ?(?:[^\s\w" + marks + r"]|_)+|\s+(?!\S)|\s+|."
        )
        _V2_RE = re.compile(pattern, re.DOTALL)
    return _V2_RE


def pretokenize_with(name: str, text: str) -> list[str]:
    if name == "v1":
        return _PRETOKENIZE_RE.findall(text)
    if name == "v2":
        return _v2_regex().findall(text)
    raise ValueError(f"unknown pretokenizer {name!r}; expected one of {PRETOKENIZERS}")


def corpus_fingerprint(texts: Iterable[str]) -> dict:
    """Order-sensitive SHA-256 over a text corpus (normalised, length-prefixed so boundaries matter) plus its size."""
    h = hashlib.sha256()
    docs = chars = 0
    for t in texts:
        n = normalize(t)
        b = n.encode("utf-8", errors="replace")
        h.update(len(b).to_bytes(8, "big"))
        h.update(b)
        docs += 1
        chars += len(n)
    return {"sha256": h.hexdigest(), "num_documents": docs, "num_chars": chars}


class BPETokenizer:
    def __init__(self, merges: Sequence[tuple[int, int]], pretokenizer: str = DEFAULT_PRETOKENIZER):
        if pretokenizer not in PRETOKENIZERS:
            raise ValueError(f"unknown pretokenizer {pretokenizer!r}; expected one of {PRETOKENIZERS}")
        self.pretokenizer = pretokenizer
        #: set by `train` (corpus fingerprint of the training texts) or restored from manifest.json by `load`
        self.training_info: dict | None = None
        self.merges: list[tuple[int, int]] = [(int(a), int(b)) for a, b in merges]
        for rank, (a, b) in enumerate(self.merges):
            limit = BASE_VOCAB_SIZE + rank
            if not (NUM_SPECIAL <= a < limit and NUM_SPECIAL <= b < limit):
                raise ValueError(f"merge #{rank} {(a, b)} references an undefined token")
        self._ranks = {pair: r for r, pair in enumerate(self.merges)}
        if len(self._ranks) != len(self.merges):
            raise ValueError("duplicate merge rules")
        self.vocab_size = BASE_VOCAB_SIZE + len(self.merges)
        self._token_bytes = self._build_token_bytes()
        self._cache: dict[str, tuple[int, ...]] = {}

    # ------------------------------------------------------------------ training
    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int,
        min_frequency: int = 2,
        progress: Callable[[int, int], None] | None = None,
        pretokenizer: str = DEFAULT_PRETOKENIZER,
    ) -> BPETokenizer:
        if vocab_size < BASE_VOCAB_SIZE:
            raise ValueError(f"vocab_size must be >= {BASE_VOCAB_SIZE}")
        piece_freq: Counter[str] = Counter()
        fp = hashlib.sha256()
        n_docs = n_chars = 0
        for text in texts:
            norm = normalize(text)
            raw = norm.encode("utf-8", errors="replace")
            fp.update(len(raw).to_bytes(8, "big"))
            fp.update(raw)
            n_docs += 1
            n_chars += len(norm)
            piece_freq.update(pretokenize_with(pretokenizer, norm))

        pieces = sorted(piece_freq)  # deterministic order
        words: list[list[int]] = [[BYTE_OFFSET + b for b in p.encode("utf-8", errors="replace")] for p in pieces]
        freqs: list[int] = [piece_freq[p] for p in pieces]

        pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        pair_words: dict[tuple[int, int], set[int]] = defaultdict(set)
        for wi, word in enumerate(words):
            for pair in zip(word, word[1:]):
                pair_counts[pair] += freqs[wi]
                pair_words[pair].add(wi)

        heap = [(-c, p) for p, c in pair_counts.items()]
        heapq.heapify(heap)
        merges: list[tuple[int, int]] = []
        target = vocab_size - BASE_VOCAB_SIZE

        while len(merges) < target and heap:
            best = None
            while heap:
                neg_count, pair = heapq.heappop(heap)
                if pair_counts.get(pair, 0) == -neg_count and -neg_count > 0:
                    best = (pair, -neg_count)
                    break
            if best is None or best[1] < min_frequency:
                break
            pair, _ = best
            new_id = BASE_VOCAB_SIZE + len(merges)
            merges.append(pair)

            changed: set[tuple[int, int]] = set()
            for wi in sorted(pair_words[pair]):
                word, f = words[wi], freqs[wi]
                merged: list[int] = []
                i = 0
                hit = False
                while i < len(word):
                    if i + 1 < len(word) and word[i] == pair[0] and word[i + 1] == pair[1]:
                        merged.append(new_id)
                        i += 2
                        hit = True
                    else:
                        merged.append(word[i])
                        i += 1
                if not hit:
                    continue
                for p in zip(word, word[1:]):
                    pair_counts[p] -= f
                    changed.add(p)
                for p in zip(merged, merged[1:]):
                    pair_counts[p] += f
                    pair_words[p].add(wi)
                    changed.add(p)
                words[wi] = merged
            for p in changed:
                c = pair_counts.get(p, 0)
                if c > 0:
                    heapq.heappush(heap, (-c, p))
            pair_words.pop(pair, None)
            if progress is not None and len(merges) % 100 == 0:
                progress(len(merges), target)
        tok = cls(merges, pretokenizer=pretokenizer)
        tok.training_info = {
            "corpus_sha256": fp.hexdigest(),
            "num_documents": n_docs,
            "num_chars": n_chars,
            "requested_vocab_size": vocab_size,
            "min_frequency": min_frequency,
        }
        return tok

    # ------------------------------------------------------------------ encode / decode
    def _build_token_bytes(self) -> list[bytes]:
        table: list[bytes] = [b""] * NUM_SPECIAL
        table += [bytes([b]) for b in range(256)]
        for a, b in self.merges:
            table.append(table[a] + table[b])
        return table

    def _encode_piece(self, piece: str) -> tuple[int, ...]:
        cached = self._cache.get(piece)
        if cached is not None:
            return cached
        # errors="replace": a lone surrogate (invalid Unicode) becomes '?' instead of crashing.
        ids = [BYTE_OFFSET + b for b in piece.encode("utf-8", errors="replace")]
        ranks = self._ranks
        while len(ids) > 1:
            best_rank = None
            best_pair = None
            for pair in zip(ids, ids[1:]):
                r = ranks.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_pair = r, pair
            if best_pair is None:
                break
            new_id = BASE_VOCAB_SIZE + best_rank
            out: list[int] = []
            i = 0
            while i < len(ids):
                if i + 1 < len(ids) and ids[i] == best_pair[0] and ids[i + 1] == best_pair[1]:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ids = out
        result = tuple(ids)
        if len(self._cache) >= _CACHE_LIMIT:
            self._cache.clear()
        self._cache[piece] = result
        return result

    def encode_ordinary(self, text: str) -> list[int]:
        ids: list[int] = []
        for piece in pretokenize_with(self.pretokenizer, normalize(text)):
            ids.extend(self._encode_piece(piece))
        return ids

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
        allow_special: bool = False,
    ) -> list[int]:
        """Encode text. Special-token strings inside `text` are treated as ordinary
        text unless `allow_special=True`, so untrusted documents cannot inject
        control tokens."""
        ids: list[int] = [BOS_ID] if add_bos else []
        if allow_special:
            for part in _SPECIAL_RE.split(normalize(text)):
                if part in _SPECIAL_TO_ID:
                    ids.append(_SPECIAL_TO_ID[part])
                elif part:
                    ids.extend(self.encode_ordinary(part))
        else:
            ids.extend(self.encode_ordinary(text))
        if add_eos:
            ids.append(EOS_ID)
        return ids

    def decode(self, ids: Iterable[int], *, skip_special: bool = True) -> str:
        out = bytearray()
        for i in ids:
            i = int(i)
            if i < 0 or i >= self.vocab_size:
                raise ValueError(f"token id {i} out of range (vocab_size={self.vocab_size})")
            if i < NUM_SPECIAL:
                if not skip_special:
                    out += SPECIAL_TOKENS[i].encode("utf-8")
            else:
                out += self._token_bytes[i]
        return out.decode("utf-8", errors="replace")

    def batch_encode(
        self,
        texts: Iterable[str],
        *,
        add_bos: bool = False,
        add_eos: bool = False,
        allow_special: bool = False,
        max_length: int | None = None,
        pad: bool = False,
    ) -> list[list[int]]:
        """Encode many texts. `max_length` truncates (a requested trailing EOS is kept); `pad=True` right-pads
        every row with PAD_ID to the longest row (or `max_length`)."""
        rows: list[list[int]] = []
        for t in texts:
            ids = self.encode(t, add_bos=add_bos, add_eos=False, allow_special=allow_special)
            if max_length is not None:
                keep = max_length - (1 if add_eos else 0)
                if keep < 0:
                    raise ValueError("max_length is too small for the requested special tokens")
                ids = ids[:keep]
            if add_eos:
                ids.append(EOS_ID)
            rows.append(ids)
        if pad and rows:
            width = max_length if max_length is not None else max(len(r) for r in rows)
            rows = [r + [PAD_ID] * (width - len(r)) for r in rows]
        return rows

    def batch_decode(self, batch: Iterable[Iterable[int]], *, skip_special: bool = True) -> list[str]:
        return [self.decode(ids, skip_special=skip_special) for ids in batch]

    def token_bytes(self, token_id: int) -> bytes:
        return self._token_bytes[token_id]

    # ------------------------------------------------------------------ statistics
    def vocab_statistics(self, sample_texts: Iterable[str] | None = None) -> dict:
        """Descriptive statistics of the vocabulary and, if `sample_texts` is given, of its behaviour on that text.
        Measured quantities only; nothing here is a quality claim."""
        lengths = [len(self._token_bytes[i]) for i in range(BASE_VOCAB_SIZE, self.vocab_size)]
        hist: Counter[int] = Counter(lengths)
        stats: dict = {
            "vocab_size": self.vocab_size,
            "special_tokens": len(SPECIAL_TOKENS),
            "byte_tokens": 256,
            "merged_tokens": len(self.merges),
            "merged_token_bytes": {
                "min": min(lengths) if lengths else None,
                "max": max(lengths) if lengths else None,
                "mean": round(statistics.fmean(lengths), 4) if lengths else None,
                "histogram": {str(k): hist[k] for k in sorted(hist)},
            },
            "pretokenizer": self.pretokenizer,
        }
        if sample_texts is not None:
            n_tok = n_chars = n_bytes = n_docs = 0
            used: Counter[int] = Counter()
            for t in sample_texts:
                ids = self.encode_ordinary(t)
                used.update(ids)
                norm = normalize(t)
                n_tok += len(ids)
                n_chars += len(norm)
                n_bytes += len(norm.encode("utf-8", errors="replace"))
                n_docs += 1
            stats["sample"] = {
                "documents": n_docs,
                "tokens": n_tok,
                "chars": n_chars,
                "chars_per_token": round(n_chars / n_tok, 4) if n_tok else None,
                "bytes_per_token": round(n_bytes / n_tok, 4) if n_tok else None,
                "distinct_tokens_used": len(used),
                "vocab_coverage": round(len(used) / self.vocab_size, 4),
                "unk_tokens": used.get(UNK_ID, 0),
            }
        return stats

    # ------------------------------------------------------------------ identity / persistence
    def _definition(self) -> dict:
        d = {
            "format_version": FORMAT_VERSION,
            "special_tokens": list(SPECIAL_TOKENS),
            "normalization": NORMALIZATION,
            "pretokenize_pattern": PRETOKENIZE_PATTERN,
            "merges": [list(m) for m in self.merges],
        }
        if self.pretokenizer != "v1":
            d["pretokenizer"] = self.pretokenizer  # v1 omits the key so pre-P8 fingerprints are unchanged
        return d

    @property
    def fingerprint(self) -> str:
        """SHA-256 over everything that defines the token-ID space."""
        return sha256_bytes(canonical_json(self._definition()).encode("utf-8"))

    def save(
        self,
        directory: str | Path,
        *,
        extra_config: dict | None = None,
        sample_texts: Iterable[str] | None = None,
    ) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        merges_txt = "#version: researchos-bpe-%d\n" % FORMAT_VERSION
        merges_txt += "".join(f"{a} {b}\n" for a, b in self.merges)
        atomic_write_text(d / "merges.txt", merges_txt)
        atomic_write_json(
            d / "vocab.json",
            {
                "special_tokens": _SPECIAL_TO_ID,
                "byte_offset": BYTE_OFFSET,
                "vocab_size": self.vocab_size,
                "tokens_hex": {str(i): self._token_bytes[i].hex() for i in range(NUM_SPECIAL, self.vocab_size)},
            },
        )
        config = {
            "format_version": FORMAT_VERSION,
            "vocab_size": self.vocab_size,
            "num_merges": len(self.merges),
            "normalization": NORMALIZATION,
            "pretokenize_pattern": PRETOKENIZE_PATTERN,
            "special_tokens": list(SPECIAL_TOKENS),
            "pretokenizer": self.pretokenizer,
        }
        if extra_config:
            config["training"] = extra_config
        atomic_write_json(d / "tokenizer_config.json", config)
        atomic_write_text(d / "tokenizer_version.txt", f"researchos-bpe-v{FORMAT_VERSION}\n{self.fingerprint}\n")
        atomic_write_json(d / "vocab_statistics.json", self.vocab_statistics(sample_texts))
        # manifest.json is written last and lists the sha-256 of every other file, so a partial or edited
        # tokenizer directory is detected by `verify_manifest` / `load`.
        files = ["merges.txt", "vocab.json", "tokenizer_config.json", "tokenizer_version.txt", "vocab_statistics.json"]
        manifest = {
            "tokenizer_name": "researchos-bpe",
            "tokenizer_version": f"researchos-bpe-v{FORMAT_VERSION}",
            "fingerprint": self.fingerprint,
            "vocab_size": self.vocab_size,
            "num_merges": len(self.merges),
            "pretokenizer": self.pretokenizer,
            "normalization": NORMALIZATION,
            "special_tokens": _SPECIAL_TO_ID,
            "training_corpus": self.training_info,
            "files": {f: sha256_bytes((d / f).read_bytes()) for f in files},
        }
        atomic_write_json(d / "manifest.json", manifest)
        return d

    @staticmethod
    def verify_manifest(directory: str | Path) -> list[str]:
        """Return a list of problems (empty = every file matches manifest.json)."""
        d = Path(directory)
        mp = d / "manifest.json"
        if not mp.exists():
            return ["manifest.json is missing"]
        try:
            manifest = json.loads(mp.read_text(encoding="utf-8"))
        except ValueError as e:
            return [f"manifest.json is not valid JSON: {e}"]
        problems = []
        for name, digest in manifest.get("files", {}).items():
            f = d / name
            if not f.exists():
                problems.append(f"{name} is missing")
            elif sha256_bytes(f.read_bytes()) != digest:
                problems.append(f"{name} does not match its manifest hash")
        return problems

    @classmethod
    def load(cls, directory: str | Path) -> BPETokenizer:
        d = Path(directory)
        lines = (d / "merges.txt").read_text(encoding="utf-8").splitlines()
        if not lines or not lines[0].startswith("#version: researchos-bpe-"):
            raise ValueError(f"{d / 'merges.txt'} is not a RESEARCHOS merges file")
        merges = []
        for line in lines[1:]:
            if line.strip():
                a, b = line.split()
                merges.append((int(a), int(b)))
        pretokenizer = DEFAULT_PRETOKENIZER
        cfg_path = d / "tokenizer_config.json"
        if cfg_path.exists():
            pretokenizer = json.loads(cfg_path.read_text(encoding="utf-8")).get("pretokenizer", DEFAULT_PRETOKENIZER)
        tok = cls(merges, pretokenizer=pretokenizer)
        expected = (d / "tokenizer_version.txt").read_text(encoding="utf-8").split()[-1]
        if tok.fingerprint != expected:
            raise ValueError("tokenizer fingerprint mismatch: merges.txt does not match tokenizer_version.txt")
        if (d / "manifest.json").exists():  # tokenizers saved before P8 have none and still load
            problems = cls.verify_manifest(d)
            if problems:
                raise ValueError("tokenizer manifest check failed: " + "; ".join(problems))
            tok.training_info = json.loads((d / "manifest.json").read_text(encoding="utf-8")).get("training_corpus")
        return tok

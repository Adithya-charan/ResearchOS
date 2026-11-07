"""Write the task-specific evaluation sets in datasets/evaluation/task_sets/ (deterministic; run once, outputs are committed).

All text below was written for this project (licence: CC0-1.0). References, authors and numbers are FICTIONAL - they exist
only to exercise formats and vocabulary and must never be read as real sources. These sets are small on purpose: they are
smoke-level probes, not benchmarks, and they say nothing about performance on real scientific text.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.common import sha256_file  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "datasets" / "evaluation" / "task_sets"

SETS: dict[str, dict] = {
    "general_lm": {
        "kind": "text",
        "description": "Plain English passages with no scientific vocabulary.",
        "items": [
            "The market opens early on Saturday mornings, and the street fills with people carrying baskets.",
            "She walked along the river until the path turned toward the old stone bridge.",
            "After the rain stopped, the children ran outside to look for puddles.",
            "The train was late, so we waited in the small cafe near the platform.",
            "He kept the letters in a wooden box beside the window.",
            "In autumn the hills turn orange and the air smells of smoke from distant chimneys.",
            "The teacher asked everyone to open their books to the second chapter.",
            "They painted the fence white and planted roses along the garden wall.",
        ],
    },
    "scientific_vocabulary": {
        "kind": "text",
        "description": "Sentences dense in scientific terms (invented statements, not facts).",
        "items": [
            "The electrolyte exhibited high ionic conductivity at elevated temperature.",
            "Spectroscopic analysis revealed a shift in the absorption peak after annealing.",
            "The regression coefficient was statistically significant after correcting for multiple comparisons.",
            "Sample viscosity decreased monotonically as the shear rate increased.",
            "The catalyst retained its activity over fifty cycles without measurable degradation.",
            "Calibration of the detector was repeated before each measurement session.",
            "The control group received a placebo while the treatment group received the compound.",
            "Phase transitions were identified by differential scanning calorimetry.",
        ],
    },
    "research_question_completion": {
        "kind": "completion",
        "description": "Prefix is the start of a research question; only the continuation is scored.",
        "items": [
            {"prefix": "Research question: How does temperature affect", "continuation": " the ionic conductivity of the material?"},
            {"prefix": "Research question: Does increasing the sample size reduce", "continuation": " the variance of the estimate?"},
            {"prefix": "Research question: What is the relationship between", "continuation": " training data quality and model accuracy?"},
            {"prefix": "Research question: Can the proposed method generalize to", "continuation": " conditions that were not seen during calibration?"},
            {"prefix": "Research question: Which factors explain the difference in", "continuation": " reported outcomes between the two studies?"},
            {"prefix": "Research question: To what extent does the measurement protocol influence", "continuation": " the reproducibility of the results?"},
        ],
    },
    "abstract_continuation": {
        "kind": "completion",
        "description": "Prefix is the opening of an invented abstract; the closing sentence is scored.",
        "items": [
            {"prefix": "We study the effect of annealing on thin films. Samples were prepared at five temperatures and measured in triplicate.",
             "continuation": " Results indicate that conductivity increases with temperature, but replication is required."},
            {"prefix": "This work evaluates a screening method for candidate materials. The method was applied to forty samples.",
             "continuation": " The findings are limited to the conditions tested and should not be generalized."},
            {"prefix": "We compare two measurement protocols on the same instrument. Each protocol was repeated twenty times.",
             "continuation": " The protocols differed in mean value, and the difference exceeded the stated uncertainty."},
            {"prefix": "A survey of published methods was conducted. Only studies with reported uncertainties were included.",
             "continuation": " We conclude that reporting practices vary widely across the literature."},
        ],
    },
    "citation_format": {
        "kind": "text",
        "description": "Sentences with citation markers. Every author, year and number is fictional; scores format handling only.",
        "items": [
            "Conductivity increased with temperature (Doe et al., 2019).",
            "This agrees with earlier measurements [1] and with the later replication [2].",
            "As reported by Roe and Poe (2021), the effect was small but consistent.",
            "Several groups observed the same trend [3, 4, 5].",
            "The protocol follows Smith (2018), with the modifications described in Lee et al. (2020).",
            "See Table 2 and Figure 3 for the full results [6].",
        ],
    },
    "structured_output": {
        "kind": "text",
        "description": "Claim/status/evidence blocks in the format the platform's agents emit.",
        "items": [
            "Claim C1: conductivity increased with temperature.\nStatus: supported.\nEvidence: doc-1, page 3.\n",
            "Claim C2: the effect did not replicate.\nStatus: contradicted.\nEvidence: doc-2, page 7.\n",
            "Claim C3: the sample size was sufficient.\nStatus: insufficient evidence.\nEvidence: none found.\n",
            "Claim C4: the method generalizes.\nStatus: unverified.\nEvidence: doc-4, page 2.\n",
            "Hypothesis H1: annealing raises conductivity.\nTest: measure five temperatures in triplicate.\nCriterion: slope above zero.\n",
            "Limitation L1: only one instrument was used.\nImpact: results may not transfer.\n",
        ],
    },
    "research_terminology": {
        "kind": "text",
        "description": "Short definitions of research terms.",
        "items": [
            "A confounder is a variable that influences both the exposure and the outcome.",
            "Replication means repeating a study to check whether the original result holds.",
            "A control group is a baseline against which the treatment group is compared.",
            "Statistical power is the probability of detecting an effect when one exists.",
            "A systematic review follows a pre-registered protocol to collect and appraise evidence.",
            "Selection bias occurs when the sample does not represent the population of interest.",
        ],
    },
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "license": "CC0-1.0",
        "authorship": "written for the RESEARCHOS project; all references, authors and numbers are fictional",
        "purpose": "smoke-level task probes; held out from every training corpus; NOT a benchmark",
        "sets": {},
    }
    for name, spec in SETS.items():
        path = OUT / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for i, item in enumerate(spec["items"]):
                rec = {"id": f"{name}-{i:02d}", "kind": spec["kind"]}
                rec.update(item if isinstance(item, dict) else {"text": item})
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        manifest["sets"][name] = {
            "file": path.name, "kind": spec["kind"], "description": spec["description"],
            "items": len(spec["items"]), "sha256": sha256_file(path),
        }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(SETS)} task sets to {OUT}")


if __name__ == "__main__":
    main()

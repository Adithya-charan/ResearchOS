/** Server-Sent Events over fetch (so the HttpOnly cookie is sent and Last-Event-ID can be controlled), with de-duplication,
 *  bounded reconnect with backoff, and clean teardown. It never fabricates events: it only forwards what the server sends. */
import { BASE } from "./api";

export interface SseMessage { id?: number; event: string; data: unknown }
export type SseStatus = "connecting" | "open" | "reconnecting" | "closed" | "ended";

export interface SseOptions {
  path: string;                       // e.g. /runs/{id}/stream
  onMessage: (m: SseMessage) => void;
  onStatus?: (s: SseStatus, info?: string) => void;
  afterId?: number;                   // resume point
  maxRetries?: number;
  fetchImpl?: typeof fetch;           // injectable for tests
}

/** Parse a buffer into complete SSE blocks; returns the rest. Exported for unit tests. */
export function parseBlocks(buf: string): { blocks: SseMessage[]; rest: string } {
  const blocks: SseMessage[] = [];
  let rest = buf.replace(/\r\n/g, "\n");
  let i: number;
  while ((i = rest.indexOf("\n\n")) >= 0) {
    const raw = rest.slice(0, i); rest = rest.slice(i + 2);
    let event = "message", id: number | undefined; const data: string[] = [];
    for (const line of raw.split("\n")) {
      if (line.startsWith(":")) continue; // comment / keepalive
      const k = line.indexOf(":"); const field = k < 0 ? line : line.slice(0, k); const value = k < 0 ? "" : line.slice(k + 1).replace(/^ /, "");
      if (field === "event") event = value; else if (field === "data") data.push(value); else if (field === "id" && /^\d+$/.test(value)) id = Number(value);
    }
    if (!data.length) continue;
    let parsed: unknown = data.join("\n");
    try { parsed = JSON.parse(parsed as string); } catch { /* keep text */ }
    blocks.push({ id, event, data: parsed });
  }
  return { blocks, rest };
}

export function connectSse(o: SseOptions): () => void {
  const f = o.fetchImpl ?? fetch;
  const ctl = new AbortController();
  let last = o.afterId ?? 0;
  let closed = false;
  const seen = new Set<number>();
  const max = o.maxRetries ?? 8;

  (async () => {
    let attempt = 0;
    while (!closed) {
      o.onStatus?.(attempt === 0 ? "connecting" : "reconnecting");
      try {
        const res = await f(BASE + o.path + (o.path.includes("?") ? "&" : "?") + `after=${last}`, {
          credentials: "same-origin", signal: ctl.signal, headers: { Accept: "text/event-stream", "Last-Event-ID": String(last) },
        });
        if (!res.ok || !res.body) { if (res.status >= 400 && res.status < 500) { o.onStatus?.("closed", `HTTP ${res.status}`); return; } throw new Error(`HTTP ${res.status}`); }
        o.onStatus?.("open"); attempt = 0;
        const reader = res.body.getReader(); const dec = new TextDecoder(); let buf = ""; let reconnect = false;
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          buf += dec.decode(value, { stream: true });
          const { blocks, rest } = parseBlocks(buf); buf = rest;
          for (const b of blocks) {
            if (b.id !== undefined) { if (seen.has(b.id)) continue; seen.add(b.id); if (b.id > last) last = b.id; }
            if (b.event === "end") {
              const reason = (b.data as { reason?: string } | null)?.reason;
              if (reason === "max_duration") { reconnect = true; break; } // server asks us to resume from Last-Event-ID
              o.onStatus?.("ended"); return;
            }
            o.onMessage(b);
          }
          if (reconnect) break;
        }
      } catch (e) {
        if (closed || (e as { name?: string })?.name === "AbortError") return;
      }
      if (closed) return;
      if (++attempt > max) { o.onStatus?.("closed", "too many reconnect attempts"); return; }
      await new Promise((r) => setTimeout(r, Math.min(15_000, 500 * 2 ** attempt)));
    }
  })();

  return () => { closed = true; ctl.abort(); o.onStatus?.("closed"); };
}

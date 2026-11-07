import { describe, expect, it, vi } from "vitest";
import { connectSse, parseBlocks, type SseMessage } from "./sse";

const stream = (chunks: string[]) => new Response(new ReadableStream({ start(c) { chunks.forEach((x) => c.enqueue(new TextEncoder().encode(x))); c.close(); } }), { status: 200 });

describe("SSE parser", () => {
  it("parses ids, events and JSON data, ignores keep-alive comments, and returns the incomplete tail", () => {
    const { blocks, rest } = parseBlocks(': keepalive\n\nid: 4\nevent: state\ndata: {"a":1}\n\nid: 5\ndata: {"b"');
    expect(blocks).toEqual([{ id: 4, event: "state", data: { a: 1 } }]);
    expect(rest).toBe('id: 5\ndata: {"b"');
  });
  it("handles CRLF and joins multi-line data", () => {
    const { blocks } = parseBlocks("id: 1\r\ndata: one\r\ndata: two\r\n\r\n");
    expect(blocks[0].data).toBe("one\ntwo");
  });
});

describe("SSE client", () => {
  it("de-duplicates repeated ids, resumes from Last-Event-ID after a drop, and stops on an end event", async () => {
    const seen: SseMessage[] = [];
    const calls: { url: string; last: string | undefined }[] = [];
    const responses = [
      stream(['id: 1\ndata: {"type":"a"}\n\nid: 2\ndata: {"type":"b"}\n\nid: 2\ndata: {"type":"dup"}\n\n']),
      stream(['id: 3\ndata: {"type":"c"}\n\nevent: end\ndata: {"reason":"finished"}\n\n']),
    ];
    const fetchImpl = vi.fn(async (url: RequestInfo | URL, init?: RequestInit) => { calls.push({ url: String(url), last: (init?.headers as Record<string, string>)["Last-Event-ID"] }); return responses.shift()!; });
    const status: string[] = [];
    await new Promise<void>((resolve) => {
      connectSse({ path: "/runs/r1/stream", fetchImpl: fetchImpl as unknown as typeof fetch, maxRetries: 3,
        onMessage: (m) => seen.push(m), onStatus: (s) => { status.push(s); if (s === "ended") resolve(); } });
    });
    expect(seen.map((m) => m.id)).toEqual([1, 2, 3]); // the duplicate id 2 was dropped
    expect(calls[1].last).toBe("2");
    expect(calls[1].url).toContain("after=2");
    expect(status).toContain("reconnecting");
  });

  it("stops without retrying on a 4xx and cleans up on close", async () => {
    const fetchImpl = vi.fn(async () => new Response("no", { status: 403 }));
    const status: string[] = [];
    const stop = connectSse({ path: "/runs/x/stream", fetchImpl: fetchImpl as unknown as typeof fetch, onMessage: () => undefined, onStatus: (s) => status.push(s) });
    await vi.waitFor(() => expect(status).toContain("closed"));
    expect(fetchImpl).toHaveBeenCalledTimes(1);
    stop();
  });
});

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, api, getCsrf, messageFrom, onUnauthorized, setCsrf, withQuery } from "./api";

const json = (status: number, body: unknown) => new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });

describe("api client", () => {
  const fetchMock = vi.fn();
  beforeEach(() => { fetchMock.mockReset(); vi.stubGlobal("fetch", fetchMock); setCsrf("tok-123"); });
  afterEach(() => { vi.unstubAllGlobals(); vi.useRealTimers(); });

  it("sends the CSRF header on mutations but not on reads, and always same-origin credentials", async () => {
    fetchMock.mockImplementation(async () => json(200, {}));
    await api.get("/projects");
    await api.post("/projects", { title: "x" });
    const [get, post] = fetchMock.mock.calls;
    expect((get[1].headers as Record<string, string>)["X-CSRF-Token"]).toBeUndefined();
    expect((post[1].headers as Record<string, string>)["X-CSRF-Token"]).toBe("tok-123");
    expect(get[1].credentials).toBe("same-origin");
    expect(post[1].credentials).toBe("same-origin");
  });

  it("keeps the token in memory only (never in browser storage)", () => {
    setCsrf("secret");
    expect(getCsrf()).toBe("secret");
    expect(JSON.stringify({ ...localStorage, ...sessionStorage })).not.toContain("secret");
  });

  it("retries a safe GET once after a server error, but never retries a POST", async () => {
    vi.useFakeTimers();
    fetchMock.mockResolvedValueOnce(json(503, {})).mockResolvedValueOnce(json(200, { ok: 1 }));
    const p = api.get("/projects");
    await vi.advanceTimersByTimeAsync(400);
    await expect(p).resolves.toEqual({ ok: 1 });
    expect(fetchMock).toHaveBeenCalledTimes(2);

    fetchMock.mockReset(); fetchMock.mockImplementation(async () => json(503, {}));
    await expect(api.post("/projects", {})).rejects.toBeInstanceOf(ApiError);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("does not retry client errors and surfaces the server message", async () => {
    fetchMock.mockImplementation(async () => json(409, { detail: "project has no searchable documents" }));
    await expect(api.get("/x")).rejects.toMatchObject({ status: 409, message: "project has no searchable documents" });
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("notifies listeners on 401 for data calls but not for /auth/ calls", async () => {
    const fn = vi.fn(); const off = onUnauthorized(fn);
    fetchMock.mockImplementation(async () => json(401, { detail: "no" }));
    await expect(api.get("/projects")).rejects.toBeInstanceOf(ApiError);
    expect(fn).toHaveBeenCalledTimes(1);
    await expect(api.get("/auth/me", { retries: 0 })).rejects.toBeInstanceOf(ApiError);
    expect(fn).toHaveBeenCalledTimes(1);
    off();
  });

  it("turns validation arrays and unknown statuses into short messages", () => {
    expect(messageFrom(422, { detail: [{ msg: "field required" }, { msg: "too short" }] })).toBe("field required; too short");
    expect(messageFrom(500, null)).toMatch(/server hit an error/i);
    expect(messageFrom(429, null)).toMatch(/quota/i);
  });

  it("builds query strings without empty values", () => {
    expect(withQuery("/search", { q: "a b", k: 5, author: "", year: undefined, hq: false })).toBe("/search?q=a+b&k=5&hq=false");
    expect(withQuery("/x")).toBe("/x");
  });
});

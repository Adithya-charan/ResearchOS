/** Typed API client. Same-origin, cookie-credentialed; CSRF header on every mutation; timeouts; cancellation; safe GET retry only. */
import { z } from "zod";

export const BASE = ""; // same origin (served under /app by the backend; proxied by Vite in development)
export const DEFAULT_TIMEOUT_MS = 30_000;
export const UPLOAD_TIMEOUT_MS = 180_000;

export class ApiError extends Error {
  constructor(public status: number, message: string, public detail?: unknown) {
    super(message);
    this.name = "ApiError";
  }
}

const SessionSchema = z.object({
  user: z.object({ id: z.string(), email: z.string(), display_name: z.string() }),
  csrf_token: z.string(),
  expires_at: z.string(),
});
export type Session = z.infer<typeof SessionSchema>;

let csrfToken = "";
export const setCsrf = (t: string) => { csrfToken = t; };
export const getCsrf = () => csrfToken;

type Listener = () => void;
const unauthorizedListeners = new Set<Listener>();
export const onUnauthorized = (fn: Listener) => { unauthorizedListeners.add(fn); return () => { unauthorizedListeners.delete(fn); }; };

/** Turn FastAPI error bodies into one short, safe message (never a stack trace; server detail strings are plain text). */
export function messageFrom(status: number, body: unknown): string {
  const d = (body as { detail?: unknown } | null)?.detail;
  if (typeof d === "string") return d;
  if (Array.isArray(d)) return d.map((x) => (x && typeof x === "object" && "msg" in x ? String((x as { msg: unknown }).msg) : "invalid input")).join("; ");
  if (status === 401) return "Your session has ended. Please sign in again.";
  if (status === 403) return "You do not have permission to do that.";
  if (status === 404) return "Not found.";
  if (status === 429) return "Too many requests or a quota was reached. Try again shortly.";
  if (status >= 500) return "The server hit an error. Try again.";
  return `Request failed (${status}).`;
}

export interface RequestOptions {
  signal?: AbortSignal;
  timeoutMs?: number;
  /** Safe-to-retry reads only. Mutations are never retried automatically. */
  retries?: number;
  query?: Record<string, string | number | boolean | undefined | null>;
}

export function withQuery(path: string, query?: RequestOptions["query"]): string {
  if (!query) return path;
  const p = new URLSearchParams();
  for (const [k, v] of Object.entries(query)) if (v !== undefined && v !== null && v !== "") p.set(k, String(v));
  const s = p.toString();
  return s ? `${path}?${s}` : path;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function once(method: string, path: string, body: unknown, opts: RequestOptions, raw?: FormData): Promise<Response> {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(new DOMException("timeout", "TimeoutError")), opts.timeoutMs ?? DEFAULT_TIMEOUT_MS);
  const onAbort = () => ctl.abort(opts.signal?.reason);
  opts.signal?.addEventListener("abort", onAbort);
  try {
    const headers: Record<string, string> = {};
    let payload: BodyInit | undefined;
    if (raw) payload = raw;
    else if (body !== undefined) { headers["Content-Type"] = "application/json"; payload = JSON.stringify(body); }
    if (method !== "GET" && method !== "HEAD") headers["X-CSRF-Token"] = csrfToken;
    return await fetch(BASE + withQuery(path, opts.query), { method, headers, body: payload, credentials: "same-origin", signal: ctl.signal });
  } catch (e) {
    if (e instanceof DOMException && e.name === "TimeoutError") throw new ApiError(0, "The request timed out.");
    if (opts.signal?.aborted) throw e;
    if (ctl.signal.aborted && (ctl.signal.reason as DOMException | undefined)?.name === "TimeoutError") throw new ApiError(0, "The request timed out.");
    throw new ApiError(0, "Could not reach the server. Check your connection.");
  } finally {
    clearTimeout(timer);
    opts.signal?.removeEventListener("abort", onAbort);
  }
}

async function request<T>(method: string, path: string, body?: unknown, opts: RequestOptions = {}, raw?: FormData, parse: "json" | "text" | "blob" = "json"): Promise<T> {
  const retries = method === "GET" ? (opts.retries ?? 1) : 0; // never auto-retry a non-idempotent call
  let attempt = 0;
  for (;;) {
    let res: Response;
    try {
      res = await once(method, path, body, opts, raw);
    } catch (e) {
      if (attempt < retries && !(opts.signal?.aborted) && e instanceof ApiError && e.status === 0) { attempt++; await sleep(300 * attempt); continue; }
      throw e;
    }
    if (res.status >= 500 && attempt < retries) { attempt++; await sleep(300 * attempt); continue; }
    if (!res.ok) {
      let data: unknown = null;
      try { data = await res.json(); } catch { /* non-JSON error body */ }
      if (res.status === 401 && !path.startsWith("/auth/")) unauthorizedListeners.forEach((f) => f());
      throw new ApiError(res.status, messageFrom(res.status, data), data);
    }
    if (res.status === 204) return undefined as T;
    if (parse === "text") return (await res.text()) as T;
    if (parse === "blob") return (await res.blob()) as T;
    return (await res.json()) as T;
  }
}

export const api = {
  get: <T>(path: string, opts?: RequestOptions) => request<T>("GET", path, undefined, opts),
  post: <T>(path: string, body?: unknown, opts?: RequestOptions) => request<T>("POST", path, body ?? {}, opts),
  put: <T>(path: string, body?: unknown, opts?: RequestOptions) => request<T>("PUT", path, body ?? {}, opts),
  patch: <T>(path: string, body?: unknown, opts?: RequestOptions) => request<T>("PATCH", path, body ?? {}, opts),
  del: <T = void>(path: string, opts?: RequestOptions) => request<T>("DELETE", path, undefined, opts),
  text: (path: string, opts?: RequestOptions) => request<string>("GET", path, undefined, opts, undefined, "text"),
  blob: (path: string, opts?: RequestOptions) => request<Blob>("GET", path, undefined, opts, undefined, "blob"),
  upload: <T>(path: string, form: FormData, opts?: RequestOptions) => request<T>("POST", path, undefined, { timeoutMs: UPLOAD_TIMEOUT_MS, ...opts }, form),
};

/** Session calls validate their response so a malformed reply can never be treated as a signed-in user. */
export async function fetchSession(signal?: AbortSignal): Promise<Session> {
  const s = SessionSchema.parse(await api.get("/auth/me", { signal, retries: 0 }));
  setCsrf(s.csrf_token);
  return s;
}
export async function login(email: string, password: string): Promise<Session> {
  const s = SessionSchema.parse(await api.post("/auth/login", { email, password }));
  setCsrf(s.csrf_token);
  return s;
}
export async function register(email: string, password: string, display_name: string): Promise<Session> {
  const s = SessionSchema.parse(await api.post("/auth/register", { email, password, display_name }));
  setCsrf(s.csrf_token);
  return s;
}
export async function logout(): Promise<void> {
  try { await api.post("/auth/logout"); } finally { setCsrf(""); }
}

/** Save a Blob/text as a file download without navigating. */
export function saveFile(name: string, data: Blob | string, type = "text/plain") {
  const blob = typeof data === "string" ? new Blob([data], { type }) : data;
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = name; document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

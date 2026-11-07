import { useQueryClient } from "@tanstack/react-query";
import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { fetchSession, login as apiLogin, logout as apiLogout, onUnauthorized, register as apiRegister, ApiError, type Session } from "./lib/api";

type State = { status: "loading" } | { status: "anon"; reason?: string } | { status: "authed"; session: Session };
interface Ctx {
  state: State;
  user: Session["user"] | null;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, name: string) => Promise<void>;
  logout: () => Promise<void>;
}
const AuthCtx = createContext<Ctx | null>(null);
export const PROJECT_KEY = "researchos.project"; // the selected project id (not a credential); cleared on sign-out and on account change

export function AuthProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<State>({ status: "loading" });
  const qc = useQueryClient();

  const reset = useCallback((reason?: string) => {
    qc.clear(); // never show one account's cached data to the next
    try { localStorage.removeItem(PROJECT_KEY); } catch { /* storage unavailable */ }
    setState({ status: "anon", reason });
  }, [qc]);

  useEffect(() => {
    const ctl = new AbortController();
    fetchSession(ctl.signal).then((session) => setState({ status: "authed", session }))
      .catch((e) => { if (!ctl.signal.aborted) setState({ status: "anon", reason: e instanceof ApiError && e.status !== 401 ? e.message : undefined }); });
    return () => ctl.abort();
  }, []);

  useEffect(() => onUnauthorized(() => { setState((s) => { if (s.status === "authed") { qc.clear(); return { status: "anon", reason: "Your session expired. Please sign in again." }; } return s; }); }), [qc]);

  const value = useMemo<Ctx>(() => ({
    state, user: state.status === "authed" ? state.session.user : null,
    async login(email, password) { const s = await apiLogin(email, password); qc.clear(); setState({ status: "authed", session: s }); },
    async register(email, password, name) { const s = await apiRegister(email, password, name); qc.clear(); setState({ status: "authed", session: s }); },
    async logout() { try { await apiLogout(); } finally { reset("You have been signed out."); } },
  }), [state, qc, reset]);
  return <AuthCtx.Provider value={value}>{children}</AuthCtx.Provider>;
}
export function useAuth() { const c = useContext(AuthCtx); if (!c) throw new Error("useAuth outside AuthProvider"); return c; }

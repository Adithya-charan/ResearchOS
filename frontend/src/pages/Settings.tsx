import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { useAuth } from "../auth";
import { Q } from "../components/common";
import { Card, CardTitle, KV, Notice, PageHeader, Select } from "../components/ui";
import { api } from "../lib/api";
import { agentName, fmtDate } from "../lib/format";
import type { AgentsResponse } from "../lib/types";

const THEME_KEY = "researchos.theme";

export default function Settings() {
  const { state, user } = useAuth();
  const [theme, setTheme] = useState<string>(() => { try { return localStorage.getItem(THEME_KEY) === "dark" ? "dark" : "light"; } catch { return "light"; } });
  const agents = useQuery({ queryKey: ["agents"], queryFn: ({ signal }) => api.get<AgentsResponse>("/agents", { signal }), staleTime: 300_000 });
  const health = useQuery({ queryKey: ["models-health"], queryFn: ({ signal }) => api.get<{ status: string; system_models: string[]; training: { enabled: boolean } }>("/models/health", { signal }), retry: false });
  const apply = (v: string) => { setTheme(v); document.documentElement.dataset.theme = v; try { localStorage.setItem(THEME_KEY, v); } catch { /* storage unavailable */ } };
  return (
    <div className="space-y-4">
      <PageHeader title="Settings" subtitle="Only settings that can actually be saved are shown here." />
      <div className="grid gap-4 lg:grid-cols-2">
        <Card><CardTitle>Account</CardTitle>
          <KV items={[["Name", user?.display_name ?? "-"], ["Email", user?.email ?? "-"], ["Session expires", state.status === "authed" ? fmtDate(state.session.expires_at) : "-"]]} />
          <p className="mt-3 text-xs text-muted">Changing the password or profile is not available: the backend has no endpoint for it yet.</p></Card>
        <Card><CardTitle>Appearance</CardTitle>
          <label htmlFor="set-theme" className="mb-1 block text-sm font-medium">Theme</label>
          <Select id="set-theme" value={theme} onChange={(e) => apply(e.target.value)}><option value="light">Light</option><option value="dark">Dark</option></Select>
          <p className="mt-2 text-xs text-muted">Saved in this browser only.</p></Card>
        <Card><CardTitle>Agents</CardTitle>
          <Q q={agents} label="agents">{(a) => <><p className="mb-2 text-xs text-muted">Workflow: {a.workflow}. Read-only.</p><ul className="space-y-1 text-sm">{a.agents.map((x) => <li key={x.agent_type}><b>{agentName(x.agent_type)}</b> <span className="text-muted">— timeout {x.spec.timeout_s ?? "-"} s, retries {x.spec.max_retries ?? "-"}</span></li>)}</ul></>}</Q></Card>
        <Card><CardTitle>Model status</CardTitle>
          {health.isError ? <p className="text-sm text-muted">Custom model status is not available.</p> : <Q q={health} label="model status">{(h) => <KV items={[["Service", h.status], ["Registered models", h.system_models.length], ["Training", h.training.enabled ? "enabled" : "disabled"]]} />}</Q>}</Card>
      </div>
      <Notice>Security: you are signed in with an HttpOnly session cookie; no token is kept in browser storage. Changes to data go through CSRF-protected requests.</Notice>
    </div>
  );
}

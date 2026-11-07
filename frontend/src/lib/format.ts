export const fmtDate = (s?: string | null) => (s ? new Date(s.endsWith("Z") || /[+-]\d\d:\d\d$/.test(s) ? s : s + "Z").toLocaleString() : "-");
export const fmtDay = (s: string) => new Date(s).toLocaleDateString(undefined, { month: "short", day: "numeric" });
export const fmtNum = (n: number | null | undefined, d = 0) => (n == null || Number.isNaN(n) ? "-" : n.toLocaleString(undefined, { maximumFractionDigits: d }));
export const fmtBytes = (n?: number | null) => (n == null ? "-" : n > 2 ** 30 ? `${(n / 2 ** 30).toFixed(2)} GiB` : n > 2 ** 20 ? `${(n / 2 ** 20).toFixed(1)} MiB` : `${(n / 1024).toFixed(1)} KiB`);
export const fmtDuration = (s?: number | null) => (s == null ? "-" : s < 1 ? `${Math.round(s * 1000)} ms` : s < 60 ? `${s.toFixed(1)} s` : `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`);
export const titleCase = (s: string) => s.replace(/[_-]+/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

/** Labels for claim statuses. Always lexical: never worded as proof. */
export const CLAIM_LABEL: Record<string, string> = {
  SUPPORTED: "Lexically supported by corpus evidence",
  PARTIALLY_SUPPORTED: "Partially supported (lexical)",
  CONTRADICTED: "Contradicted by corpus text (lexical)",
  INSUFFICIENT_EVIDENCE: "Insufficient evidence in the corpus",
  UNVERIFIED: "Not yet verified",
};
export const CLAIM_SHORT: Record<string, string> = { SUPPORTED: "Supported", PARTIALLY_SUPPORTED: "Partially supported", CONTRADICTED: "Contradicted", INSUFFICIENT_EVIDENCE: "Insufficient evidence", UNVERIFIED: "Unverified" };

/** Agent display names (the backend `agent_type` ids). */
export const AGENT_NAMES: Record<string, string> = {
  supervisor: "Supervisor", discovery: "Literature Discovery", critic: "Evidence Critic", literature: "Literature Analysis", gap: "Research Gap",
  hypothesis: "Hypothesis", experiment_design: "Experiment Design", verifier: "Verification", reviewer: "Critical Reviewer", synthesis: "Synthesis", experiment: "Experiment",
};
export const agentName = (t: string) => AGENT_NAMES[t] ?? titleCase(t);

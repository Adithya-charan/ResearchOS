/** Response types, written from the real API payloads (see docs/FRONTEND_API_INTEGRATION.md). Optional fields are those the API may omit. */
export interface Project { id: string; title: string; question: string; description: string; created_at: string; updated_at: string }

export interface RunState {
  status: string; lifecycle?: string; stop_reason?: string | null; control?: string; elapsed_s?: number;
  task_counts?: Record<string, number>; model_tokens_used?: number; active: boolean;
  job?: { status: string; attempts: number; max_attempts: number; error: string } | null;
}
export interface Run { id: string; project_id: string; question: string; workflow: string; created_at: string; state: RunState }

export interface AgentSpec { agent_type: string; role: string; responsibilities: string[]; input_keys: string[]; output_keys: string[]; timeout_s: number | null; max_retries: number | null; evaluation_criteria?: string[]; memory_writes?: string[] }
export interface AgentInfo { agent_type: string; spec: AgentSpec; [k: string]: unknown }
export interface AgentsResponse { workflow: string; agents: AgentInfo[] }

export interface Task {
  seq: number; id: string; run_id: string; parent_task_id: string | null; agent_type: string; title: string; depends_on: string[];
  status: string; iteration: number; retry_count: number; error: string | null; requires_approval: boolean;
  created_at: string; started_at: string | null; completed_at: string | null; summary?: string; output_status?: string;
}
export interface Execution { id: number; task_id: string; agent_type: string; attempt: number; status: string; started_at: string; finished_at: string | null; duration_s: number | null; error: string | null }
export interface TaskDetail extends Task { input?: Record<string, unknown>; output?: AgentOutput | null; executions?: Execution[]; logs?: EngineEvent[] }
export interface AgentOutput {
  agent_type: string; status: string; summary: string; findings: string[]; uncertainties: string[]; model_notes?: unknown[];
  evidence: { document_id: string; page_number: number | null; passage_id: string }[];
  data: Record<string, unknown>; proposed_tasks?: unknown[]; task_id: string;
}
export interface EngineEvent { id: number; run_id: string; task_id: string | null; type: string; payload: Record<string, unknown>; created_at: string }

export interface DocumentRec {
  id: string; filename: string; file_type: string; size_bytes: number; title: string; authors: string[]; year: number | null; doi: string | null;
  pages: number; source: string; license: string; allowed_use: string[]; extraction_status: string; quality?: { status: string; issues: string[] };
  warnings: string[]; created_at: string; publication_date: string | null; url: string | null; document_type: string | null; ingestion_status?: string;
  suspected_prompt_injection?: unknown[]; untrusted_content_notice?: string;
}
export interface Passage { passage_id: string; page_number: number | null; section: string; quality: string; text: string }
export interface Citation { document_id: string; chunk_id?: string; title: string; excerpt: string; page_number: number | null; section?: string; score?: number; url?: string | null; doi?: string | null; source?: string }
export interface SearchHit { document_id: string; title: string; year: number | null; passage_id?: string; page_number: number | null; section?: string; quality?: string; score: number; matched_terms?: string[]; matched_phrases?: string[]; text?: string; excerpt?: string; citation?: Citation; best_passage?: unknown }
export interface SearchResponse { query: string; level: string; method: string; hits: SearchHit[]; [k: string]: unknown }
export interface CorpusStats { documents: number; passages: number; indexed_documents: number; indexed_passages: number; searchable_documents?: number; low_quality_passages?: number }

export type ClaimStatus = "SUPPORTED" | "PARTIALLY_SUPPORTED" | "CONTRADICTED" | "INSUFFICIENT_EVIDENCE" | "UNVERIFIED";
export const CLAIM_STATUSES: ClaimStatus[] = ["SUPPORTED", "PARTIALLY_SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE", "UNVERIFIED"];
export interface ClaimEvidence { id: number; role: string; directness: string; document_id: string; passage_id: string; page_number: number | null; quote: string; retrieval_method: string; retrieval_score: number | null; coverage: number; citation_valid: boolean; reason: string; origin: string }
export interface Claim {
  id: string; project_id: string; run_id: string | null; claim_key: string | null; text: string; claim_type: string; origin_agent: string; origin_task_id: string | null;
  source: string; status: ClaimStatus; strength: { level?: string; basis?: string; note?: string; independent_documents?: number }; created_at: string; updated_at: string;
  evidence_counts: { supporting: number; contradicting: number; contextual: number };
  explanation?: string[]; evidence?: ClaimEvidence[]; verifications?: number; scope_note?: string;
}
export interface ClaimList { total: number; claims: Claim[] }
export interface ClaimVerification { id: number; claim_id: string; round: number; status: ClaimStatus; strength: Claim["strength"]; explanation: string[]; evidence: Partial<ClaimEvidence>[]; verifier: string; requested_by: string; created_at: string }

export interface NoveltyAssessment {
  statement: string; statement_kind: string; label: string; reasons: string[];
  similar_existing_research: NoveltyDoc[]; closest_documents: NoveltyDoc[]; overlapping_research_questions: unknown[];
  differences_from_closest: { statement_terms_not_found_in_closest_documents: string[]; note: string };
  search_coverage: { corpus_documents: number; corpus_passages: number; queries: string[]; documents_with_retrieved_passages: number; passages_read: number };
  unsearched: string[]; limitations: string[]; disclaimer: string; excluded_source_documents: unknown[];
}
export interface NoveltyDoc { document_id: string; title: string; year: number | null; doi: string | null; best_coverage: number; passage_id: string; page_number: number | null; quote: string; matched_terms: string[] }
export interface NoveltyRow { id: string; project_id: string; run_id: string | null; claim_id: string | null; statement: string; label: string; created_at: string; assessment: NoveltyAssessment }

export interface ReportSummary { id: string; run_id: string; title: string; status: string; version: number; review: string; updated_at: string }
export interface ReportFull { id: string; project_id: string; run_id: string; title: string; version: number; created_at: string; updated_at: string; meta: Record<string, unknown>; content_intact: boolean; content: Record<string, unknown>; editable?: Record<string, unknown> }
export interface ReportBlock { text: string; kind: string; epistemic?: "evidence" | "fact" | "inference" | "absence"; status: ClaimStatus | null; strength?: string | null; claim_key?: string | null; citations: { document_id: string; page_number: number | null; passage_id: string; quote: string; source_text?: string; resolvable: boolean }[] }
export interface ReportSection { id: string; number: number; title: string; blocks: ReportBlock[] }
export interface ReportQuality {
  statement_mix: Record<string, number>; statement_meaning: Record<string, string>;
  evidence_coverage: { claims: number; by_status: Record<string, number>; supported_or_partial: number; fraction_supported_or_partial: number | null; claims_not_fully_supported: number; documents_cited: number; blocks_with_citation: number };
  contradictions: number; limitations_listed: number; recommendations_listed: number; assumptions: string; reading_note: string;
}
export interface CitationIssue { section: number; block: number; claim_key: string | null; passage_id: string | null; document_id: string | null; issue: string; detail: string }
export interface CitationIntegrity {
  citations_checked: number; citations_valid: number; citations_with_issues: number; integrity_rate: number | null; documents_cited: number;
  claim_blocks: number; claim_blocks_with_a_valid_citation: number;
  unsupported_claims: { section: number; block: number; claim_key: string | null; status: string | null; text: string; reason: string }[];
  issues: CitationIssue[]; warnings: CitationIssue[]; by_issue: Record<string, number>; ok: boolean; scope: string;
}
export interface StructuredReport { report_id: string; version: number; content_sha256: string; citation_integrity?: CitationIntegrity; structured: { title: string; sections: ReportSection[]; claim_counts: Record<string, number>; unresolved_claims: unknown[]; verification_rounds: number; rules?: string; quality?: ReportQuality }; verification: { claims_checked: number; counts: Record<string, number>; needs_followup: number; round: number }[] }

export interface GapEvidence { document_id: string; passage_id: string; page_number: number | null; quote: string }
export interface LibraryGap { id: string; type: string; basis: "observed" | "hypothesis"; statement: string; evidence: GapEvidence[]; documents: string[]; caveat?: string; verifier_reason?: string; themes?: string[] }
export interface LibraryGapReport {
  scope: string; limitations: string[]; documents: number; gaps: LibraryGap[]; note?: string; excluded_not_quotable: number; traceability_problems: string[];
  themes: { frequent: { theme: string; documents: string[]; document_count: number }[]; single_document: { theme: string; document: string; title: string }[] };
  follow_ups: { for_gap: string; basis: string; suggestion: string; documents: string[] }[];
  counts: { observed: number; hypothesis: number; by_type: Record<string, number> };
}

export interface RunReliability {
  run_id: string; run_status: string; finished: boolean; degraded: boolean; report_available: boolean; tasks: number; note: string; guarantees: string[];
  agents: Record<string, { tasks: number; succeeded: number; failed: number; skipped: number; cancelled: number; rejected: number; attempts: number; retries: number; timeouts: number; abandoned_attempts: number; duration_s: number }>;
  failures: { task_id: string; agent_type: string; title: string; status: string; error: string | null; attempts: number; retries_used: number; timed_out: boolean }[];
  skipped: { task_id: string; agent_type: string; title: string; reason: string | null; caused_by: { task_id: string; agent_type: string; status: string }[] }[];
  partial_results: { task_id: string; agent_type: string; title: string }[];
  retried_tasks: { task_id: string; agent_type: string; retries_used: number }[];
}
export interface ReportVersion { version: number; editor_id: string; summary: string; at: string; editable: Record<string, unknown> }

export interface Approval { kind: "task" | "experiment"; id: string | number; run_id?: string; project_id: string; title: string; agent_type?: string; task_id?: string; decision: string | null; comment?: string; requested_at: string; decided_at?: string | null; reviewer?: string | null; review_hint?: string; code_sha256?: string }
export interface ExperimentSummary { id: string; project_id: string; name: string; status: string; created_at: string; updated_at: string; approval: string }
export interface ExperimentFull extends ExperimentSummary { spec: Record<string, unknown>; spec_sha256: string; approval_record: Record<string, unknown> | null; error: string | null; manifest: Record<string, unknown> | null; reproduction: unknown; review: { status: string; [k: string]: unknown }; limitations: string[]; integrity?: { ok: boolean; problems: string[] } }

export interface Overview {
  projects: number; runs: number; runs_scanned: number; active_runs: number; completed_runs: number; pending_approvals: number; documents: number;
  claims: Record<ClaimStatus, number>; claims_total: number; reports: number; run_status: Record<string, number>; task_status: Record<string, number>;
  runs_per_day: { date: string; runs: number }[]; note: string;
  recent_runs: { id: string; project_id: string; question: string; created_at: string; status: string; active: boolean }[];
  recent_projects: { id: string; title: string; question: string; updated_at: string }[];
}
export interface AuditEntry { id: number; at: string; action: string; target: string; detail: Record<string, unknown> }
export interface AuditResponse { total: number; entries: AuditEntry[]; actions: string[] }

export interface LabOverview {
  hardware: Record<string, unknown>; models: { model_id: string; state: string; capability: string; stage_reached: number }[]; lifecycle_counts: Record<string, number>;
  selection: { provider: string; model_id: string | null; mode: string | null; allow_experimental: boolean }; training_enabled: boolean;
  inference: { device: string; loaded: unknown[] }; datasets: number; is_admin: boolean; notice: string;
}

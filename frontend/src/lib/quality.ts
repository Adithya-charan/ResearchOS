import type { CitationIntegrity, CitationIssue } from "./types";

/** Block key used by the report viewer is `${sectionIndex}-${blockIndex}`; the API numbers sections from 1. */
export const blockKey = (section: number, block: number) => `${section - 1}-${block}`;

/** Citation problems grouped by the block they belong to, so a block can show exactly what is wrong with its own citations. */
export function issuesByBlock(ci: CitationIntegrity | undefined): Map<string, CitationIssue[]> {
  const m = new Map<string, CitationIssue[]>();
  for (const i of ci?.issues ?? []) {
    const k = blockKey(i.section, i.block);
    m.set(k, [...(m.get(k) ?? []), i]);
  }
  return m;
}

/** Claim blocks that carry a status but no valid citation (the API's unsupported_claims), keyed the same way. */
export function unsupportedBlocks(ci: CitationIntegrity | undefined): Set<string> {
  return new Set((ci?.unsupported_claims ?? []).map((u) => blockKey(u.section, u.block)));
}

/** A rate as a percentage, or a dash when there is nothing to measure (never 0% or 100% for an empty set). */
export const pct = (r: number | null | undefined): string => (r == null ? "n/a" : `${(r * 100).toFixed(r === 1 || r === 0 ? 0 : 1)}%`);

export const ISSUE_LABEL: Record<string, string> = {
  no_passage_id: "no passage named", passage_missing: "passage not found", document_mismatch: "wrong document", page_mismatch: "wrong page",
  quote_not_in_passage: "quote not in passage", source_changed: "source text changed",
};
export const BASIS_LABEL: Record<string, string> = { observed: "Observed in the library", hypothesis: "Hypothesis (a suggestion, not a finding)" };

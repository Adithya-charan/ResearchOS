import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { QualityPanel } from "./QualityPanel";
import type { CitationIntegrity, ReportQuality } from "../lib/types";

const quality = (o: Partial<ReportQuality["evidence_coverage"]> = {}): ReportQuality => ({
  statement_mix: { evidence: 4, fact: 6, inference: 2, absence: 1 },
  statement_meaning: { evidence: "quoted from a stored passage", fact: "computed from run data", inference: "drafted and untested", absence: "nothing found" },
  evidence_coverage: { claims: 5, by_status: { SUPPORTED: 3, PARTIALLY_SUPPORTED: 1, CONTRADICTED: 1, INSUFFICIENT_EVIDENCE: 0, UNVERIFIED: 0 }, supported_or_partial: 4, fraction_supported_or_partial: 0.8, claims_not_fully_supported: 1, documents_cited: 3, blocks_with_citation: 7, ...o },
  contradictions: 1, limitations_listed: 5, recommendations_listed: 2, assumptions: "No explicit assumptions are recorded.", reading_note: "Coverage is not a measure of how much was read.",
});
const integrity = (o: Partial<CitationIntegrity> = {}): CitationIntegrity => ({
  citations_checked: 10, citations_valid: 10, citations_with_issues: 0, integrity_rate: 1, documents_cited: 3, claim_blocks: 5, claim_blocks_with_a_valid_citation: 5,
  unsupported_claims: [], issues: [], warnings: [], by_issue: {}, ok: true, scope: "Does not check that the passage supports the claim.", ...o,
});

describe("QualityPanel", () => {
  it("shows the real numbers and the caveats that go with them", () => {
    render(<QualityPanel quality={quality()} integrity={integrity()} />);
    expect(screen.getByText(/4 of 5 \(80\.0%\)/)).toBeVisible();
    expect(screen.getByText("all citations resolve")).toBeVisible();
    expect(screen.getByText(/not a measure of how much was read/i)).toBeVisible();
    expect(screen.getByText(/does not check that the passage supports the claim/i)).toBeVisible();
  });
  it("says 'none to measure' instead of a percentage when there are no claims or citations", () => {
    render(<QualityPanel quality={quality({ claims: 0, supported_or_partial: 0, fraction_supported_or_partial: null })} integrity={integrity({ citations_checked: 0, citations_valid: 0, integrity_rate: null, claim_blocks: 0, claim_blocks_with_a_valid_citation: 0 })} />);
    expect(screen.getByText("none to measure")).toBeVisible();
    expect(screen.getByText("none to check")).toBeVisible();
    expect(screen.getByText("no citations")).toBeVisible();
    expect(screen.queryByText(/%/)).toBeNull();
  });
  it("lists each citation problem with a text label and warns about unsupported claims", () => {
    render(<QualityPanel quality={quality()} integrity={integrity({
      citations_valid: 8, citations_with_issues: 2, integrity_rate: 0.8, ok: false, by_issue: { passage_missing: 1, page_mismatch: 1 },
      issues: [{ section: 10, block: 0, claim_key: "k", passage_id: "d:p9", document_id: "d", issue: "passage_missing", detail: "the cited passage does not exist" },
        { section: 11, block: 2, claim_key: null, passage_id: "d:p1", document_id: "d", issue: "page_mismatch", detail: "the cited page differs" }],
      unsupported_claims: [{ section: 10, block: 0, claim_key: "k", status: "SUPPORTED", text: "Capacity fell 22 percent", reason: "no valid citation" }] })} />);
    expect(screen.getByText("2 with issues")).toBeVisible();
    const problems = screen.getByText("Citation problems").closest("details") as HTMLElement;
    expect(within(problems).getByText(/the cited passage does not exist/)).toBeInTheDocument();
    expect(within(problems).getByText(/the cited page differs/)).toBeInTheDocument();
    expect(screen.getByRole("note")).toHaveTextContent(/no valid citation/i);
  });
  it("renders nothing when the API sent neither summary", () => {
    const { container } = render(<QualityPanel />);
    expect(container).toBeEmptyDOMElement();
  });
});

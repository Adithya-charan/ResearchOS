import { describe, expect, it } from "vitest";
import { blockKey, issuesByBlock, pct, unsupportedBlocks } from "./quality";
import type { CitationIntegrity } from "./types";

const ci = (o: Partial<CitationIntegrity>): CitationIntegrity => ({
  citations_checked: 0, citations_valid: 0, citations_with_issues: 0, integrity_rate: null, documents_cited: 0, claim_blocks: 0, claim_blocks_with_a_valid_citation: 0,
  unsupported_claims: [], issues: [], warnings: [], by_issue: {}, ok: true, scope: "", ...o,
});

describe("report quality helpers", () => {
  it("API section numbers (from 1) line up with the viewer's zero-based block keys", () => {
    expect(blockKey(1, 0)).toBe("0-0");
    expect(blockKey(13, 4)).toBe("12-4");
  });
  it("groups citation issues by the block they belong to", () => {
    const m = issuesByBlock(ci({ issues: [
      { section: 10, block: 2, claim_key: "k", passage_id: "a", document_id: "d", issue: "passage_missing", detail: "x" },
      { section: 10, block: 2, claim_key: "k", passage_id: "b", document_id: "d", issue: "page_mismatch", detail: "y" },
      { section: 11, block: 0, claim_key: null, passage_id: "c", document_id: "d", issue: "quote_not_in_passage", detail: "z" }] }));
    expect(m.get("9-2")?.map((i) => i.issue)).toEqual(["passage_missing", "page_mismatch"]);
    expect(m.get("10-0")).toHaveLength(1);
    expect(issuesByBlock(undefined).size).toBe(0);
  });
  it("finds unsupported claim blocks", () => {
    const s = unsupportedBlocks(ci({ unsupported_claims: [{ section: 10, block: 1, claim_key: "k", status: "SUPPORTED", text: "t", reason: "r" }] }));
    expect(s.has("9-1")).toBe(true);
    expect(s.has("9-0")).toBe(false);
  });
  it("never shows a rate for an empty set as if it were measured", () => {
    expect(pct(null)).toBe("n/a");
    expect(pct(undefined)).toBe("n/a");
    expect(pct(1)).toBe("100%");
    expect(pct(0)).toBe("0%");
    expect(pct(0.6667)).toBe("66.7%");
  });
});

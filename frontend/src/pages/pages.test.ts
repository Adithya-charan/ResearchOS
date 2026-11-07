import { describe, expect, it } from "vitest";
import { layout } from "./Agents";
import { stageOf } from "./Experiments";
import { ALLOWED_EXT, MAX_UPLOAD_BYTES, validateUpload } from "./Library";
import { credentialsSchema, registerSchema } from "./Login";
import { projectSchema } from "./Projects";
import { citationLabel } from "./Reports";
import { runSchema, stagesOf } from "./Workspace";
import type { Task } from "../lib/types";

const task = (o: Partial<Task>): Task => ({ seq: 1, id: "t1", run_id: "r", parent_task_id: null, agent_type: "supervisor", title: "t", depends_on: [], status: "succeeded", iteration: 0, retry_count: 0, error: null, requires_approval: false, created_at: "", started_at: null, completed_at: null, ...o });

describe("form validation", () => {
  it("login requires a valid email and a password", () => {
    expect(credentialsSchema.safeParse({ email: "nope", password: "" }).success).toBe(false);
    expect(credentialsSchema.safeParse({ email: "a@b.org", password: "x" }).success).toBe(true);
  });
  it("register enforces the minimum password length", () => {
    expect(registerSchema.safeParse({ email: "a@b.org", password: "short" }).success).toBe(false);
    expect(registerSchema.safeParse({ email: "a@b.org", password: "long-enough-pw" }).success).toBe(true);
  });
  it("project needs a title and a real question", () => {
    expect(projectSchema.safeParse({ title: "", question: "why?" }).success).toBe(false);
    expect(projectSchema.safeParse({ title: "T", question: "How does X affect Y?" }).success).toBe(true);
  });
  it("run limits are bounded and coerced from form strings", () => {
    expect(runSchema.safeParse({ question: "How does X affect Y?", max_runtime_s: "300", max_tasks: "40" }).success).toBe(true);
    expect(runSchema.safeParse({ question: "How does X affect Y?", max_runtime_s: "1", max_tasks: "40" }).success).toBe(false);
  });
});

describe("upload pre-checks", () => {
  it("accepts the allowed types and rejects others, empty and oversize files", () => {
    for (const ext of ALLOWED_EXT) expect(validateUpload({ name: `a${ext.toUpperCase()}`, size: 10 })).toBeNull();
    expect(validateUpload({ name: "a.exe", size: 10 })).toMatch(/unsupported/i);
    expect(validateUpload({ name: "a.pdf", size: 0 })).toMatch(/empty/i);
    expect(validateUpload({ name: "a.pdf", size: MAX_UPLOAD_BYTES + 1 })).toMatch(/larger/i);
  });
});

describe("run progress and graph helpers", () => {
  it("stagesOf groups tasks per agent in planning order and reports the most urgent status", () => {
    const stages = stagesOf([task({ seq: 1, id: "a", agent_type: "supervisor" }), task({ seq: 2, id: "b", agent_type: "discovery", status: "running" }), task({ seq: 3, id: "c", agent_type: "discovery", status: "succeeded" }), task({ seq: 4, id: "d", agent_type: "critic", status: "failed" })]);
    expect(stages.map((s) => s.agent)).toEqual(["supervisor", "discovery", "critic"]);
    expect(stages[1]).toMatchObject({ status: "running", done: 1, total: 2 });
    expect(stages[2].status).toBe("failed");
  });
  it("layout derives dependency depth from depends_on only", () => {
    const { depth } = layout([task({ id: "a" }), task({ id: "b", depends_on: ["a"] }), task({ id: "c", depends_on: ["a", "b"] }), task({ id: "d", depends_on: ["missing"] })]);
    expect(depth).toMatchObject({ a: 0, b: 1, c: 2, d: 0 });
  });
  it("layout survives a dependency cycle", () => {
    const { depth } = layout([task({ id: "a", depends_on: ["b"] }), task({ id: "b", depends_on: ["a"] })]);
    expect(Object.keys(depth)).toHaveLength(2);
  });
});

describe("experiment stage and citations", () => {
  it("separates proposed, awaiting approval, approved and executed states", () => {
    expect(stageOf({ status: "draft", approval: "none" })).toBe("draft");
    expect(stageOf({ status: "draft", approval: "pending" })).toBe("awaiting_approval");
    expect(stageOf({ status: "draft", approval: "approved" })).toBe("approved");
    expect(stageOf({ status: "completed", approval: "approved" })).toBe("completed");
    expect(stageOf({ status: "failed", approval: "approved" })).toBe("failed");
  });
  it("formats a citation from document, passage and page", () => {
    expect(citationLabel({ document_id: "doc-1", passage_id: "doc-1:p000", page_number: 3, quote: "", resolvable: true })).toBe("doc-1 · doc-1:p000 · p.3");
    expect(citationLabel({ document_id: "doc-1", passage_id: "", page_number: null, quote: "", resolvable: false })).toBe("doc-1");
  });
});

import { fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";
import { Badge, ClaimBadge, ConfirmDialog, Highlight, Segmented, StatusBadge, Tabs, ToastProvider, useToast } from "./ui";
import { Markdown } from "./Markdown";

describe("status indicators", () => {
  it("StatusBadge always carries text, never colour alone", () => {
    render(<><StatusBadge status="awaiting_approval" /><StatusBadge status="failed" /></>);
    expect(screen.getByText("Awaiting Approval")).toBeInTheDocument();
    expect(screen.getByText("Failed")).toBeInTheDocument();
  });
  it("ClaimBadge uses lexical wording in its tooltip, not truth wording", () => {
    render(<ClaimBadge status="SUPPORTED" />);
    expect(screen.getByText("Supported").closest("span")).toHaveAttribute("title", expect.stringMatching(/lexically supported/i));
  });
  it("Badge renders children", () => { render(<Badge>hello</Badge>); expect(screen.getByText("hello")).toBeVisible(); });
});

describe("Highlight", () => {
  it("marks every matched term case-insensitively and escapes regex characters", () => {
    const { container } = render(<Highlight text="Conductivity rose; conductivity (C++) held." terms={["conductivity", "c++"]} />);
    expect(Array.from(container.querySelectorAll("mark")).map((m) => m.textContent)).toEqual(["Conductivity", "conductivity", "C++"]);
  });
  it("renders plain text when there are no usable terms", () => {
    const { container } = render(<Highlight text="nothing" terms={["a", ""]} />);
    expect(container.querySelector("mark")).toBeNull();
  });
});

describe("Tabs and Segmented", () => {
  it("Tabs move with arrow keys and mark the selected tab", () => {
    const Demo = () => { const [v, setV] = useState<"a" | "b">("a"); return <Tabs label="demo" value={v} onChange={setV} tabs={[{ id: "a", label: "A" }, { id: "b", label: "B" }]} />; };
    render(<Demo />);
    const a = screen.getByRole("tab", { name: "A" });
    expect(a).toHaveAttribute("aria-selected", "true");
    fireEvent.keyDown(a, { key: "ArrowRight" });
    expect(screen.getByRole("tab", { name: "B" })).toHaveAttribute("aria-selected", "true");
  });
  it("Segmented is a radio group", () => {
    const on = vi.fn();
    render(<Segmented label="Level" value="x" onChange={on} options={[{ id: "x", label: "X" }, { id: "y", label: "Y" }]} />);
    fireEvent.click(screen.getByRole("radio", { name: "Y" }));
    expect(on).toHaveBeenCalledWith("y");
  });
});

describe("ConfirmDialog", () => {
  it("does nothing until confirmed and is cancellable", () => {
    const ok = vi.fn(); const close = vi.fn();
    render(<ConfirmDialog open onOpenChange={close} title="Delete?" description="Gone for good." confirmLabel="Delete it" onConfirm={ok} />);
    expect(ok).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(close).toHaveBeenCalledWith(false);
    fireEvent.click(screen.getByRole("button", { name: "Delete it" }));
    expect(ok).toHaveBeenCalledTimes(1);
  });
});

describe("toasts", () => {
  it("announce through a polite live region and can be dismissed", () => {
    const Fire = () => { const t = useToast(); return <button onClick={() => t("ok", "Saved.")}>go</button>; };
    render(<ToastProvider><Fire /></ToastProvider>);
    fireEvent.click(screen.getByText("go"));
    expect(screen.getByText("Saved.").closest("[aria-live]")).toHaveAttribute("aria-live", "polite");
    fireEvent.click(screen.getByLabelText("Dismiss notification"));
    expect(screen.queryByText("Saved.")).toBeNull();
  });
});

describe("Markdown safety", () => {
  it("drops script tags and javascript: links from untrusted text", () => {
    const { container } = render(<Markdown>{"hello <script>alert(1)</script> [x](javascript:alert(1)) [ok](https://example.org)"}</Markdown>);
    expect(container.querySelector("script")).toBeNull();
    const hrefs = Array.from(container.querySelectorAll("a")).map((a) => a.getAttribute("href"));
    expect(hrefs.some((h) => h?.startsWith("javascript:"))).toBe(false);
    expect(container.querySelector('a[href="https://example.org"]')).toHaveAttribute("rel", expect.stringContaining("noopener"));
  });
});

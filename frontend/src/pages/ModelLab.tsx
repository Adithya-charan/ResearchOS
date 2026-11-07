import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { ApiError, api } from "../lib/api";
import { Q } from "../components/common";
import { Badge, Card, CardTitle, EmptyState, KV, Modal, Notice, PageHeader, Table } from "../components/ui";
import { fmtBytes, titleCase } from "../lib/format";
import type { LabOverview } from "../lib/types";

interface ProviderInfo { selection: { provider: string; model_id: string | null; mode: string | null; allow_experimental: boolean }; effective: { provider: string; reason?: string; [k: string]: unknown }; error: string | null; never_uses_hosted_api: boolean; note: string }

function ModelDetail({ id, onClose }: { id: string; onClose: () => void }) {
  const q = useQuery({ queryKey: ["lab-model", id], queryFn: ({ signal }) => api.get<Record<string, unknown>>(`/modellab/models/${id}`, { signal }) });
  const ev = useQuery({ queryKey: ["lab-eval", id], queryFn: ({ signal }) => api.get<Record<string, unknown>>(`/modellab/models/${id}/evaluation`, { signal }), retry: false });
  return (
    <Modal open onOpenChange={(o) => !o && onClose()} title={`Model ${id}`}>
      <Q q={q} label="model">{(m) => <details open><summary className="cursor-pointer font-medium">Registered record</summary><pre tabIndex={0} className="mt-1 max-h-64 overflow-auto rounded bg-surface-2 p-2 text-xs">{JSON.stringify(m, null, 2)}</pre></details>}</Q>
      <div className="mt-3"><h3 className="mb-1 text-sm font-semibold">Evaluation</h3>
        {ev.isLoading ? <p className="text-sm text-muted">Loading…</p> : ev.isError ? <p className="text-sm text-muted">No evaluation has been recorded for this model.</p> : <pre tabIndex={0} className="max-h-64 overflow-auto rounded bg-surface-2 p-2 text-xs">{JSON.stringify(ev.data, null, 2)}</pre>}</div>
    </Modal>
  );
}

export default function ModelLab() {
  const [open, setOpen] = useState<string | null>(null);
  const q = useQuery({ queryKey: ["lab-overview"], queryFn: ({ signal }) => api.get<LabOverview>("/modellab/overview", { signal }), retry: false });
  const provider = useQuery({ queryKey: ["lab-provider"], queryFn: ({ signal }) => api.get<ProviderInfo>("/modellab/provider", { signal }), retry: false });
  const bench = useQuery({ queryKey: ["lab-bench"], queryFn: ({ signal }) => api.get<unknown>("/modellab/hardware-benchmarks", { signal }), retry: false });
  const unavailable = q.error instanceof ApiError && [403, 404].includes(q.error.status);
  return (
    <div className="space-y-4">
      <PageHeader title="Model Lab" subtitle="The from-scratch Transformer: registered models, lifecycle state, hardware and provider selection. Read-only here." />
      {unavailable ? <EmptyState title="Custom model training has not been configured yet." hint="The Model Lab API is not available to this account or deployment." /> : (
        <Q q={q} label="Model Lab">{(o) => (
          <div className="space-y-4">
            <Notice tone="warn">{o.notice}</Notice>
            <div className="grid gap-4 lg:grid-cols-2">
              <Card><CardTitle>Status</CardTitle><KV items={[["Training", o.training_enabled ? "enabled on this server" : "disabled on this server (no training can be started)"], ["Inference device", o.inference.device], ["Loaded models", o.inference.loaded.length], ["Datasets", o.datasets], ["Lifecycle", Object.entries(o.lifecycle_counts).map(([k, v]) => `${titleCase(k)}: ${v}`).join(", ") || "none"]]} /></Card>
              <Card><CardTitle>Hardware</CardTitle>
                <KV items={[["GPU", String(o.hardware.gpu_name ?? "none detected")], ["GPU memory", fmtBytes(o.hardware.gpu_total_bytes as number | undefined)], ["CUDA available", String(o.hardware.cuda_available)], ["RAM available", `${fmtBytes(o.hardware.ram_available_bytes as number | undefined)} of ${fmtBytes(o.hardware.ram_total_bytes as number | undefined)}`], ["PyTorch", String(o.hardware.torch ?? "-")], ["bf16", String(o.hardware.bf16_supported ?? "-")]]} /></Card>
            </div>
            <Card><CardTitle>Registered models</CardTitle>
              {o.models.length === 0 ? <EmptyState title="No models registered" hint="Nothing has been trained or registered." /> : (
                <Table caption="Registered models" head={["Model", "Lifecycle state", "Capability", "Stage reached", ""]}>{o.models.map((m) => (
                  <tr key={m.model_id}><td className="px-3 py-2 font-medium">{m.model_id}</td><td className="px-3 py-2"><Badge tone={m.state === "VALIDATED" ? "ok" : "neutral"}>{titleCase(m.state)}</Badge></td><td className="px-3 py-2">{titleCase(m.capability)}</td><td className="px-3 py-2">{m.stage_reached}</td><td className="px-3 py-2"><button className="text-accent underline" onClick={() => setOpen(m.model_id)}>Details</button></td></tr>))}</Table>)}
              <p className="mt-2 text-xs text-muted">Models that are only “trained” have not passed validation and are not used by research runs.</p></Card>
            <Card><CardTitle>Agent text provider</CardTitle>
              <Q q={provider} label="provider">{(p) => (<div className="space-y-2 text-sm"><KV items={[["Selected", p.selection.provider], ["Effective", p.effective.provider], ["Model", p.selection.model_id ?? "-"], ["Hosted API used", p.never_uses_hosted_api ? "never" : "unknown"]]} />{p.effective.reason && <p className="text-muted">{p.effective.reason}</p>}{p.error && <p className="text-warn">{p.error}</p>}<p className="text-xs text-muted">{p.note}</p></div>)}</Q></Card>
            <Card><CardTitle>Recorded hardware benchmarks</CardTitle>
              {bench.isLoading ? <p className="text-sm text-muted">Loading…</p> : bench.isError ? <p className="text-sm text-muted">No benchmarks available.</p> : <pre tabIndex={0} className="max-h-64 overflow-auto rounded bg-surface-2 p-2 text-xs">{JSON.stringify(bench.data, null, 2)}</pre>}</Card>
            <p className="text-xs text-muted">Training jobs are started from the API or the standalone Lab console at <a className="text-accent underline" href="/lab">/lab</a>; this page never displays loss curves or checkpoints that do not exist.</p>
            {open && <ModelDetail id={open} onClose={() => setOpen(null)} />}
          </div>)}</Q>
      )}
    </div>
  );
}

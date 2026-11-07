/* RESEARCHOS Model Lab. Plain JavaScript, no build step, no third-party code. Every number shown comes from an API response; nothing is
   simulated. Text from the server is only ever inserted as text nodes, never parsed as markup. */
"use strict";

const state = { csrf: "", user: null, models: [], modelId: localStorage.getItem("lab.model") || "", timers: [] };

/* ------------------------------------------------------------------ helpers */
function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === false || v == null) continue;
    if (k === "class") el.className = v;
    else if (k === "style") el.style.cssText = v; // CSSOM, allowed by the CSP (a style="" attribute would not be)
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) if (kid != null && kid !== false) el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  return el;
}
const fmt = (n, d = 4) => (n == null || Number.isNaN(n) ? "-" : typeof n === "number" ? (Number.isInteger(n) ? n.toLocaleString() : n.toFixed(d)) : String(n));
const bytes = (n) => (n == null ? "-" : n > 2 ** 30 ? (n / 2 ** 30).toFixed(2) + " GiB" : n > 2 ** 20 ? (n / 2 ** 20).toFixed(1) + " MiB" : (n / 1024).toFixed(1) + " KiB");
const chip = (text, kind) => h("span", { class: "chip " + (kind || "") }, text);
const stateChip = (s) => chip(s, { VALIDATED: "ok", DEPLOYED: "ok", TRAINED: "warn", FAILED: "bad", ARCHIVED: "" }[s] || "");

function kv(obj) {
  const dl = h("dl", { class: "kv" });
  for (const [k, v] of Object.entries(obj)) dl.append(h("dt", {}, k), h("dd", {}, v == null ? "-" : v.nodeType ? v : String(v)));
  return dl;
}
function table(caption, head, rows) {
  return h("div", { class: "scroll" }, h("table", {}, h("caption", {}, caption),
    h("thead", {}, h("tr", {}, head.map((x) => h("th", { scope: "col" }, x)))),
    h("tbody", {}, rows.map((r) => h("tr", {}, r.map((c) => h("td", {}, c == null ? "-" : c.nodeType ? c : String(c))))))));
}
const loading = (what) => h("div", { class: "state", role: "status" }, "Loading " + what + "…");
const empty = (msg) => h("div", { class: "state" }, msg);
function failed(err, retry) {
  return h("div", { class: "state error", role: "alert" }, "Failed: " + (err && err.message ? err.message : err), " ",
    retry ? h("button", { onclick: retry }, "Retry") : null);
}

async function api(method, path, body) {
  const opts = { method, credentials: "same-origin", headers: {} };
  if (body !== undefined) { opts.headers["Content-Type"] = "application/json"; opts.body = JSON.stringify(body); }
  if (method !== "GET") opts.headers["X-CSRF-Token"] = state.csrf;
  const res = await fetch(path, opts);
  if (res.status === 204) return null;
  let data = null;
  try { data = await res.json(); } catch (e) { /* non-JSON error body */ }
  if (!res.ok) {
    const detail = data && data.detail;
    const msg = typeof detail === "string" ? detail : Array.isArray(detail) ? detail.map((d) => d.msg).join("; ") : "HTTP " + res.status;
    const err = new Error(msg); err.status = res.status; throw err;
  }
  return data;
}

function clearTimers() { state.timers.forEach(clearInterval); state.timers = []; }
function every(ms, fn) { const t = setInterval(fn, ms); state.timers.push(t); }

/* a section renders into `root`; `guard` shows loading / failure around an async loader */
async function guard(root, what, load, render) {
  const run = async () => {
    root.replaceChildren(loading(what));
    try { root.replaceChildren(render(await load())); } catch (e) { root.replaceChildren(failed(e, run)); }
  };
  await run();
}

function chart(title, series, opts = {}) {
  const W = 800, H = 220, P = 36, ns = "http://www.w3.org/2000/svg";
  const pts = series.flatMap((s) => s.points);
  if (!pts.length) return empty("No " + title + " data yet.");
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = opts.zero ? 0 : Math.min(...ys), y1 = Math.max(...ys);
  const sx = (x) => P + ((x - x0) / (x1 - x0 || 1)) * (W - 2 * P), sy = (y) => H - P - ((y - y0) / (y1 - y0 || 1)) * (H - 2 * P);
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("class", "chart"); svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", title + ": " + series.map((s) => `${s.name} from ${fmt(s.points[0][1], 3)} to ${fmt(s.points.at(-1)[1], 3)}`).join("; "));
  const add = (tag, a, text) => { const e = document.createElementNS(ns, tag); for (const k in a) e.setAttribute(k, a[k]); if (text) e.textContent = text; svg.append(e); };
  add("line", { x1: P, y1: H - P, x2: W - P, y2: H - P, stroke: "currentColor", opacity: ".4" });
  add("line", { x1: P, y1: P, x2: P, y2: H - P, stroke: "currentColor", opacity: ".4" });
  add("text", { x: P, y: 14, fill: "currentColor", "font-size": 12 }, `${fmt(y1, 3)}`);
  add("text", { x: P, y: H - 8, fill: "currentColor", "font-size": 12 }, `${fmt(y0, 3)}  (step ${fmt(x0)} → ${fmt(x1)})`);
  const colours = ["#1a56db", "#c2410c", "#0b6b3a"];
  series.forEach((s, i) => add("polyline", { points: s.points.map((p) => `${sx(p[0])},${sy(p[1])}`).join(" "), fill: "none", stroke: colours[i % 3], "stroke-width": 2 }));
  return h("div", {}, svg, h("div", { class: "legend muted" }, series.map((s, i) => h("span", {}, h("span", { style: "color:" + colours[i % 3] }, "● "), s.name))));
}

/* ------------------------------------------------------------------ shared: model picker */
async function refreshModels() {
  const ov = await api("GET", "/modellab/overview");
  state.models = ov.models;
  if (!state.models.find((m) => m.model_id === state.modelId)) state.modelId = state.models.length ? state.models[0].model_id : "";
  return ov;
}
function modelPicker(onChange) {
  if (!state.models.length) return empty("No trained model exists yet. Prepare a dataset and start a training run (sections 3, 5, 6).");
  const sel = h("select", { id: "model-pick", "aria-label": "Model" }, state.models.map((m) =>
    h("option", { value: m.model_id, selected: m.model_id === state.modelId }, `${m.model_id} - ${m.state}`)));
  sel.addEventListener("change", () => { state.modelId = sel.value; localStorage.setItem("lab.model", sel.value); onChange(); });
  return h("div", { class: "row" }, h("label", { for: "model-pick" }, "Model"), sel);
}

/* ------------------------------------------------------------------ sections */
const SECTIONS = [
  ["overview", "1. Model overview"], ["tokenizer", "2. Tokenizer"], ["datasets", "3. Datasets"], ["architecture", "4. Architecture"],
  ["config", "5. Training configuration"], ["runs", "6. Training runs"], ["metrics", "7. Live training metrics"],
  ["checkpoints", "8. Checkpoints"], ["evaluation", "9. Evaluation"], ["playground", "10. Inference playground"],
  ["registry", "11. Model registry"], ["agents", "12. Agent integration"],
];

async function secOverview(root) {
  await guard(root, "overview", async () => ({ ov: await refreshModels(), bench: await api("GET", "/modellab/hardware-benchmarks") }), ({ ov, bench }) => {
    const hw = ov.hardware;
    return h("div", {}, h("h2", {}, "Model overview"), h("p", { class: "lead" }, ov.notice),
      h("div", { class: "grid" },
        h("div", { class: "card" }, h("h3", {}, "Hardware (this server, right now)"), kv({
          Platform: hw.platform, Python: hw.python, Torch: hw.torch, CPUs: hw.cpu_count, "RAM free / total": `${bytes(hw.ram_available_bytes)} / ${bytes(hw.ram_total_bytes)}`,
          GPU: hw.cuda_available ? `${hw.gpu_name} (${bytes(hw.gpu_free_bytes)} free of ${bytes(hw.gpu_total_bytes)}, bf16 ${hw.bf16_supported})` : "none (CPU only)" })),
        h("div", { class: "card" }, h("h3", {}, "Platform"), kv({
          "Training enabled": ov.training_enabled, "Inference device": ov.inference.device, "Loaded models": ov.inference.loaded.length,
          Datasets: ov.datasets, "Administrator": ov.is_admin, "Agent provider": ov.selection.provider })),
        h("div", { class: "card" }, h("h3", {}, "Lifecycle"), Object.keys(ov.lifecycle_counts).length
          ? kv(ov.lifecycle_counts) : empty("No models yet."))),
      h("h3", {}, "Models"), ov.models.length ? table("Models you can see", ["Model", "State", "Capability", "Measured integration stage"],
        ov.models.map((m) => [m.model_id, stateChip(m.state), m.capability, m.stage_reached])) : empty("No models yet."),
      h("h3", {}, "Measured hardware benchmarks"), bench.status === "not_run" ? empty(bench.note) : h("pre", {}, JSON.stringify(bench, null, 2)));
  });
}

async function secTokenizer(root) {
  await guard(root, "tokenizer information", () => api("GET", "/modellab/datasets"), (d) => {
    const prepared = d.datasets.filter((x) => x.status === "prepared");
    const box = h("div", {});
    const show = async (id) => {
      box.replaceChildren(loading("tokenizer"));
      try {
        const t = await api("GET", `/modellab/datasets/${id}/tokenizer`);
        const m = t.manifest, s = t.statistics;
        box.replaceChildren(h("div", { class: "card" }, kv({ Name: m.tokenizer_name, Version: m.tokenizer_version, Fingerprint: m.fingerprint, "Vocabulary size": m.vocab_size,
          Merges: m.num_merges, "Pre-tokenizer": m.pretokenizer, Normalization: m.normalization, "Training corpus (sha256)": m.training_corpus && m.training_corpus.corpus_sha256,
          "Training documents": m.training_corpus && m.training_corpus.num_documents, "Trained on": t.trained_on })),
        s ? h("div", { class: "card" }, h("h3", {}, "Vocabulary statistics"), kv({ "Merged tokens": s.merged_tokens, "Token bytes (min / mean / max)": `${s.merged_token_bytes.min} / ${fmt(s.merged_token_bytes.mean, 2)} / ${s.merged_token_bytes.max}`,
          ...(s.sample ? { "Sample chars per token": s.sample.chars_per_token, "Sample distinct tokens used": s.sample.distinct_tokens_used, "Sample unknown tokens": s.sample.unk_tokens } : {}) }),
          table("Merged token length histogram (bytes → count)", ["Bytes", "Tokens"], Object.entries(s.merged_token_bytes.histogram))) : null);
      } catch (e) { box.replaceChildren(failed(e)); }
    };
    const pick = h("select", { id: "tok-ds", "aria-label": "Dataset" }, prepared.map((x) => h("option", { value: x.id }, `${x.name} (${x.id})`)));
    pick.addEventListener("change", () => show(pick.value));
    if (prepared.length) show(prepared[0].id);
    return h("div", {}, h("h2", {}, "Tokenizer"), h("p", { class: "lead" }, "Byte-level BPE trained from scratch on the training split only. Every figure is read from the tokenizer's manifest."),
      prepared.length ? h("div", { class: "row" }, h("label", { for: "tok-ds" }, "Dataset"), pick) : empty("No prepared dataset yet. Upload documents and press Prepare in section 3: preparing trains the tokenizer."), box);
  });
}

async function secDatasets(root) {
  const render = (d) => {
    const msg = h("div", { role: "status", class: "muted" });
    const name = h("input", { id: "ds-name", required: true, maxlength: 80 }), lic = h("select", { id: "ds-lic" }, d.licenses.map((l) => h("option", { value: l }, l)));
    const attr = h("input", { id: "ds-attr", maxlength: 300 }), rights = h("input", { type: "checkbox", id: "ds-rights" });
    const files = h("input", { type: "file", id: "ds-files", multiple: true, accept: ".txt,.md,text/plain,text/markdown" });
    const paste = h("textarea", { id: "ds-paste", placeholder: "Or paste one document here" });
    const submit = h("button", { class: "primary", type: "submit" }, "Upload dataset");
    const form = h("form", { class: "card" }, h("h3", {}, "New dataset"),
      h("p", { class: "muted" }, d.policy),
      h("label", { for: "ds-name" }, "Name"), name, h("label", { for: "ds-lic" }, "License"), lic,
      h("label", { for: "ds-attr" }, "Attribution (source / author; required for licensed material)"), attr,
      h("label", { for: "ds-files" }, "Text files (.txt / .md)"), files, h("label", { for: "ds-paste" }, "Pasted text"), paste,
      h("label", {}, rights, "I confirm I have the right to train a model on these documents"), h("div", { class: "row" }, submit), msg);
    form.addEventListener("submit", async (ev) => {
      ev.preventDefault(); submit.disabled = true; msg.textContent = "Uploading…";
      try {
        const documents = [];
        for (const f of files.files) documents.push({ title: f.name, text: await f.text() });
        if (paste.value.trim()) documents.push({ title: "pasted", text: paste.value });
        await api("POST", "/modellab/datasets", { name: name.value, license: lic.value, attribution: attr.value, rights_confirmed: rights.checked, documents });
        await secDatasets(root);
      } catch (e) { msg.replaceChildren(failed(e)); submit.disabled = false; }
    });
    const act = async (fn) => { try { await fn(); await secDatasets(root); } catch (e) { alert(e.message); } };
    const rows = d.datasets.map((x) => [x.name, x.license, x.documents, fmt(x.chars), x.status + (x.error ? ": " + x.error : ""), x.stats.tokens ? fmt(x.stats.tokens.train) : "-",
      h("div", { class: "row" },
        h("button", { onclick: () => act(() => api("POST", `/modellab/datasets/${x.id}/prepare`, { vocab_size: 1024, pretokenizer: "v2", min_chars: 100 })), disabled: x.status === "preparing" }, "Prepare"),
        h("button", { class: "danger", onclick: () => confirm("Delete this dataset and its files?") && act(() => api("DELETE", `/modellab/datasets/${x.id}`)) }, "Delete"))]);
    if (d.datasets.some((x) => x.status === "preparing")) every(2500, () => secDatasets(root));
    return h("div", {}, h("h2", {}, "Dataset management"), h("p", { class: "lead" }, "Your own text only. Nothing is downloaded or scraped. Each upload records its license, attribution and rights confirmation in a hashed manifest."),
      d.datasets.length ? table("Your datasets", ["Name", "License", "Documents", "Characters", "Status", "Train tokens", "Actions"], rows) : empty("No datasets yet. Upload documents below."), form);
  };
  clearTimers();
  await guard(root, "datasets", () => api("GET", "/modellab/datasets"), render);
}

async function secArchitecture(root) {
  clearTimers();
  await guard(root, "architecture", async () => ({ profiles: await api("GET", "/modellab/profiles"), ov: await refreshModels() }), ({ profiles }) => {
    const box = h("div", {});
    const show = async () => {
      if (!state.modelId) { box.replaceChildren(empty("Select or train a model to see its configuration.")); return; }
      box.replaceChildren(loading("model configuration"));
      try {
        const d = await api("GET", `/modellab/models/${state.modelId}`);
        const c = d.config;
        box.replaceChildren(c ? h("div", { class: "card" }, h("h3", {}, "Selected model"), kv({ ...c.model, Parameters: c.parameters, "Weights tied": c.model.tie_embeddings,
          Precision: c.precision, "Lineage root": c.lineage_root, "Estimated training memory (estimate, not measured)": bytes(c.memory_estimate.total) })) : empty("This model has no readable configuration yet."));
      } catch (e) { box.replaceChildren(failed(e, show)); }
    };
    show();
    return h("div", {}, h("h2", {}, "Model architecture"), h("p", { class: "lead" }, "Decoder-only Transformer, written from scratch, random initialisation only."),
      modelPicker(show), box, h("h3", {}, "Hardware profiles"),
      table("Profiles. 'Fits by estimate' is a planning figure from the memory estimator; only the hardware benchmark in section 1 is a measurement.",
        ["Profile", "Parameters", "Target hardware", "Fits here (estimate)", "Context", "d_model × layers"],
        profiles.map((p) => [p.name, fmt(p.parameters), p.target_hardware, p.fit_check.fits_by_estimate ? `yes (${bytes(p.fit_check.estimated_bytes)} of ${bytes(p.fit_check.budget_bytes)} free on ${p.fit_check.device})` : `no (${bytes(p.fit_check.estimated_bytes)} needed, ${bytes(p.fit_check.budget_bytes)} free on ${p.fit_check.device})`,
          p.model.max_seq_len, `${p.model.d_model} × ${p.model.n_layers}`])));
  });
}

async function secConfig(root) {
  clearTimers();
  await guard(root, "training configuration", async () => ({ cfg: await api("GET", "/training/configs"), ds: await api("GET", "/modellab/datasets") }), ({ cfg, ds }) => {
    const msg = h("div", { role: "status" });
    const base = h("select", { id: "tc-base" }, cfg.configs.map((c) => h("option", { value: c.name }, c.name))), dsel = h("select", { id: "tc-ds" },
      h("option", { value: "" }, "operator corpus (default)"), ds.datasets.filter((x) => x.status === "prepared").map((x) => h("option", { value: x.id }, x.name)));
    const nm = h("input", { id: "tc-name", maxlength: 80, value: "my run" }), steps = h("input", { id: "tc-steps", type: "number", min: 1, max: cfg.max_steps, placeholder: "config default" });
    const lr = h("input", { id: "tc-lr", type: "number", step: "any", min: 0, max: 0.1, placeholder: "config default" }), go = h("button", { class: "primary", type: "submit", disabled: !cfg.enabled }, "Start training");
    const form = h("form", { class: "card" }, h("h3", {}, "Start a training run"),
      cfg.enabled ? null : h("p", { class: "state error" }, "Training is disabled on this server (the operator enables it with RESEARCHOS_TRAINING_ENABLED=true)."),
      h("label", { for: "tc-name" }, "Run name"), nm, h("label", { for: "tc-base" }, "Base configuration"), base, h("label", { for: "tc-ds" }, "Dataset"), dsel,
      h("label", { for: "tc-steps" }, `Max steps (server limit ${cfg.max_steps})`), steps, h("label", { for: "tc-lr" }, "Learning rate"), lr, h("div", { class: "row" }, go), msg);
    form.addEventListener("submit", async (ev) => {
      ev.preventDefault(); go.disabled = true;
      const body = { name: nm.value, base_config: base.value };
      if (dsel.value) body.dataset_id = dsel.value;
      if (steps.value) body.max_steps = Number(steps.value);
      if (lr.value) body.learning_rate = Number(lr.value);
      try { const j = await api("POST", "/training/jobs", body); msg.replaceChildren(h("div", { class: "state" }, "Started " + j.id + ". See section 6.")); }
      catch (e) { msg.replaceChildren(failed(e)); }
      go.disabled = !cfg.enabled;
    });
    return h("div", {}, h("h2", {}, "Training configuration"), h("p", { class: "lead" }, cfg.note),
      table("Operator-controlled base configurations", ["Name", "Description", "Context", "d_model", "Layers", "Device"], cfg.configs.map((c) => [c.name, c.description, c.model.max_seq_len, c.model.d_model, c.model.n_layers, c.device])), form);
  });
}

async function secRuns(root) {
  clearTimers();
  const render = (jobs) => {
    const act = async (id, verb) => { try { await api("POST", `/training/jobs/${id}/${verb}`); await secRuns(root); } catch (e) { alert(e.message); } };
    if (jobs.some((j) => ["running", "pausing", "cancelling"].includes(j.status))) every(3000, () => secRuns(root));
    return h("div", {}, h("h2", {}, "Training runs"), h("p", { class: "lead" }, "Each run is its own process; pause and cancel are cooperative and checkpoint first."),
      jobs.length ? table("Your training runs", ["Run", "Name", "Status", "Step", "Last loss", "Checkpoints", "Actions"], jobs.map((j) => [j.id, j.name,
        h("span", {}, chip(j.status, j.status === "succeeded" ? "ok" : j.status === "failed" ? "bad" : j.status === "cancelled" ? "warn" : ""), j.error ? h("div", { class: "muted" }, j.error.slice(0, 160)) : null),
        `${fmt(j.step)} / ${fmt(j.max_steps)}`, fmt(j.last_loss), j.checkpoints,
        h("div", { class: "row" }, h("button", { onclick: () => { state.modelId = j.id; localStorage.setItem("lab.model", j.id); location.hash = "metrics"; } }, "Metrics"),
          j.status === "running" ? h("button", { onclick: () => act(j.id, "pause") }, "Pause") : null,
          ["running", "paused", "pausing"].includes(j.status) ? h("button", { class: "danger", onclick: () => act(j.id, "cancel") }, "Cancel") : null,
          j.resumable ? h("button", { onclick: () => act(j.id, "resume") }, "Resume") : null)])) : empty("No training runs yet. Start one in section 5."));
  };
  await guard(root, "training runs", () => api("GET", "/training/jobs"), render);
}

async function secMetrics(root) {
  clearTimers();
  await guard(root, "training runs", () => api("GET", "/training/jobs"), (jobs) => {
    if (!jobs.length) return empty("No training runs yet, so there are no metrics.");
    let id = jobs.find((j) => j.id === state.modelId) ? state.modelId : jobs[0].id;
    const box = h("div", {}), sel = h("select", { id: "mj", "aria-label": "Training run" }, jobs.map((j) => h("option", { value: j.id, selected: j.id === id }, `${j.name} (${j.status})`)));
    const draw = async () => {
      try {
        const [job, m] = await Promise.all([api("GET", `/training/jobs/${id}`), api("GET", `/training/jobs/${id}/metrics`)]);
        const tr = m.train.map((x) => [x.step, x.loss]), ev = m.eval.map((x) => [x.step, x.val_loss]), last = m.train.at(-1), lastEval = m.eval.at(-1);
        box.replaceChildren(h("div", { class: "card" }, kv({ Status: job.status, Step: `${fmt(job.step)} / ${fmt(job.max_steps)}`, "Latest train loss": fmt(last && last.loss),
          "Latest validation loss": fmt(lastEval && lastEval.val_loss), "Validation perplexity": fmt(lastEval && lastEval.val_perplexity, 2), "Tokens / second": fmt(last && last.tokens_per_sec, 0),
          "Peak GPU memory (MiB)": fmt(last && last.gpu_mem_mb, 0), "Gradient norm": fmt(last && last.grad_norm, 3), "Learning rate": last && last.lr })),
          h("h3", {}, "Loss"), chart("loss", [{ name: "train (batch)", points: tr }, { name: "validation", points: ev }]), h("p", { class: "muted" }, m.note));
        if (!["running", "pausing", "cancelling"].includes(job.status)) clearTimers();
      } catch (e) { box.replaceChildren(failed(e, draw)); }
    };
    sel.addEventListener("change", () => { id = sel.value; clearTimers(); draw(); every(3000, draw); });
    draw(); every(3000, draw);
    return h("div", {}, h("h2", {}, "Live training metrics"), h("p", { class: "lead" }, "Read from the run's metrics file; refreshes while the run is active."),
      h("div", { class: "row" }, h("label", { for: "mj" }, "Run"), sel), box);
  });
}

async function secCheckpoints(root) {
  clearTimers();
  await guard(root, "checkpoints", refreshModels, () => {
    const box = h("div", {});
    const show = async () => {
      if (!state.modelId) { box.replaceChildren(empty("No model selected.")); return; }
      box.replaceChildren(loading("checkpoints"));
      try {
        const cks = await api("GET", `/models/${state.modelId}/checkpoints`);
        const verify = async (n, cell) => { cell.textContent = "verifying…"; try { const v = await api("GET", `/models/${state.modelId}/checkpoints/${n}/verify`); cell.textContent = v.ok ? "OK: hash and lineage verified" : "FAILED: " + v.problem; } catch (e) { cell.textContent = "error: " + e.message; } };
        const del = async (n) => { if (!confirm(`Delete checkpoint ${n}? This cannot be undone.`)) return; try { await api("DELETE", `/modellab/models/${state.modelId}/checkpoints/${n}`); show(); } catch (e) { alert(e.message); } };
        box.replaceChildren(cks.length ? table("Checkpoints", ["Name", "Step", "Validation loss", "Size", "Flags", "Integrity", "Actions"], cks.map((c) => {
          const cell = h("span", {}, "not checked");
          return [c.name, c.step, fmt(c.val_loss), bytes(c.size_bytes), [c.is_best ? chip("best", "ok") : null, c.is_latest ? chip("latest") : null], cell,
            h("div", { class: "row" }, h("button", { onclick: () => verify(c.name, cell) }, "Verify"),
              state.modelId.startsWith("trn_") ? h("button", { class: "danger", disabled: c.is_best || c.is_latest, onclick: () => del(c.name) }, "Delete") : null)];
        })) : empty("This model has no checkpoints."));
      } catch (e) { box.replaceChildren(failed(e, show)); }
    };
    show();
    return h("div", {}, h("h2", {}, "Checkpoint manager"), h("p", { class: "lead" }, "Weights are stored as safetensors with a recorded SHA-256; Verify re-hashes the file. Best and latest cannot be deleted."), modelPicker(show), box);
  });
}

async function secEvaluation(root) {
  clearTimers();
  await guard(root, "evaluation", refreshModels, () => {
    const box = h("div", {});
    const show = async () => {
      if (!state.modelId) { box.replaceChildren(empty("No model selected.")); return; }
      try {
        const ev = await api("GET", `/modellab/models/${state.modelId}/evaluation`);
        const run = h("button", { class: "primary", disabled: ev.state === "running", onclick: async () => { try { await api("POST", `/modellab/models/${state.modelId}/evaluate`); every(2500, show); show(); } catch (e) { alert(e.message); } } }, ev.state === "running" ? "Evaluating…" : "Run evaluation");
        const kids = [h("div", { class: "row" }, run, chip(ev.state))];
        if (ev.state === "running") every(2500, show); else if (ev.state !== "running") clearTimers();
        if (ev.state === "failed") kids.push(failed(ev.error));
        const r = ev.report;
        if (!r) kids.push(empty(ev.state === "running" ? "Evaluation is running…" : "No evaluation has been run for this model."));
        else {
          const v = r.splits.validation, t = r.splits.test;
          kids.push(h("div", { class: "card" }, h("h3", {}, "Headline"), kv({ "Beats random-initialised baseline": r.beats_random_baseline == null ? "-" : String(r.beats_random_baseline),
            "Validation loss (model / random baseline)": v ? `${fmt(v.model.loss)} / ${fmt(v.random_baseline.loss)}` : "-", "Validation perplexity (model / baseline)": v ? `${fmt(v.model.perplexity, 2)} / ${fmt(v.random_baseline.perplexity, 2)}` : "-",
            "Token accuracy (model / baseline)": v ? `${fmt(v.model.token_accuracy)} / ${fmt(v.random_baseline.token_accuracy)}` : "-", "Held-out test loss (model / baseline)": t ? `${fmt(t.model.loss)} / ${fmt(t.random_baseline.loss)}` : "-",
            "Memorisation risk": r.memorization_risk ? "FLAGGED - high overlap with training text" : "not flagged", Parameters: fmt(r.parameters), "Checkpoint size": bytes(r.checkpoint_bytes) })),
            h("div", { class: "card" }, h("h3", {}, "Inference speed (measured here)"), r.inference ? kv({ "Cached tokens/s": fmt(r.inference.cached.tokens_per_second, 1), "Uncached tokens/s": fmt(r.inference.uncached.tokens_per_second, 1),
              "ms per token (cached)": fmt(r.inference.cached.ms_per_token, 2), "Time to first token (ms)": fmt(r.inference.cached.time_to_first_token_ms, 1), "Cache speedup": fmt(r.inference.cache_speedup, 2) + "×",
              "Peak GPU memory": bytes(r.inference.peak_gpu_bytes) }) : empty("not measured")),
            table("Task probes (tiny, smoke-level sets)", ["Task", "Items", "Loss (model)", "Loss (baseline)", "Token accuracy (model)", "Train overlap"],
              Object.entries(r.tasks).map(([k, x]) => [k, x.model.items, fmt(x.model.loss), x.random_baseline ? fmt(x.random_baseline.loss) : "-", fmt(x.model.token_accuracy), x.model.train_overlap && x.model.train_overlap.checked ? fmt(x.model.train_overlap.overlap_fraction, 3) : "-"])),
            h("div", { class: "card" }, h("h3", {}, "Caveats (always apply)"), h("ul", {}, r.caveats.map((c) => h("li", {}, c))), h("p", { class: "muted" }, "Comparison with other models: " + r.comparison_with_other_models)));
        }
        box.replaceChildren(...kids);
      } catch (e) { box.replaceChildren(failed(e, show)); }
    };
    show();
    return h("div", {}, h("h2", {}, "Model evaluation"), h("p", { class: "lead" }, "Scored on the model's own held-out validation and test splits, next to a random-initialised model of the same architecture."), modelPicker(show), box);
  });
}

async function secPlayground(root) {
  clearTimers();
  await guard(root, "models", refreshModels, () => {
    const out = h("pre", { "aria-live": "polite", id: "pg-out" }, ""), info = h("div", { class: "muted", role: "status" });
    const prompt = h("textarea", { id: "pg-prompt", maxlength: 4000 }, "Title: "), n = h("input", { id: "pg-n", type: "number", min: 1, max: 256, value: 64 });
    const temp = h("input", { id: "pg-t", type: "number", step: "0.1", min: 0, max: 2, value: 0.8 }), go = h("button", { class: "primary", type: "button" }, "Generate"), stop = h("button", { type: "button", disabled: true }, "Stop");
    let ctl = null;
    stop.addEventListener("click", () => ctl && ctl.abort());
    go.addEventListener("click", async () => {
      if (!state.modelId) { info.textContent = "No model selected."; return; }
      ctl = new AbortController(); go.disabled = true; stop.disabled = false; out.textContent = ""; info.textContent = "Generating…";
      try {
        const res = await fetch(`/models/${state.modelId}/generate/stream`, { method: "POST", credentials: "same-origin", signal: ctl.signal,
          headers: { "Content-Type": "application/json", "X-CSRF-Token": state.csrf }, body: JSON.stringify({ prompt: prompt.value, max_new_tokens: Number(n.value), temperature: Number(temp.value) }) });
        if (!res.ok) { const d = await res.json().catch(() => ({})); throw new Error(d.detail || "HTTP " + res.status); }
        const reader = res.body.getReader(), dec = new TextDecoder(); let buf = "";
        for (;;) {
          const { value, done } = await reader.read(); if (done) break;
          buf += dec.decode(value, { stream: true });
          let i; while ((i = buf.indexOf("\n\n")) >= 0) {
            const block = buf.slice(0, i); buf = buf.slice(i + 2);
            const ev = /^event: (.+)$/m.exec(block), data = /^data: (.+)$/m.exec(block); if (!ev || !data) continue;
            const d = JSON.parse(data[1]);
            if (ev[1] === "token") out.textContent += d.text;
            else if (ev[1] === "done") info.textContent = `Finished (${d.finish_reason}): ${d.new_tokens} tokens. ${d.notice}`;
            else if (ev[1] === "error") throw new Error(d.message);
          }
        }
      } catch (e) { info.replaceChildren(e.name === "AbortError" ? empty("Cancelled.") : failed(e)); }
      go.disabled = false; stop.disabled = true;
    });
    return h("div", {}, h("h2", {}, "Inference playground"), h("p", { class: "lead" }, "Tokens stream from the local model. The output is an unverified sample, not evidence."), modelPicker(() => {}),
      h("label", { for: "pg-prompt" }, "Prompt"), prompt, h("label", { for: "pg-n" }, "Max new tokens"), n, h("label", { for: "pg-t" }, "Temperature (0 = greedy)"), temp,
      h("div", { class: "row" }, go, stop), info, h("h3", {}, "Output"), out);
  });
}

const NEXT_NOTES = { DEPLOYED: "Deploying needs an administrator.", VALIDATED: "Needs a successful evaluation that beats the random baseline." };
async function secRegistry(root) {
  clearTimers();
  await guard(root, "registry", refreshModels, () => {
    const box = h("div", {});
    const show = async () => {
      if (!state.modelId) { box.replaceChildren(empty("No model selected.")); return; }
      box.replaceChildren(loading("lifecycle"));
      try {
        const lc = await api("GET", `/modellab/models/${state.modelId}/lifecycle`);
        const move = async (target) => { const note = prompt(`Move to ${target}? Optional note:`, ""); if (note === null) return; try { await api("POST", `/modellab/models/${state.modelId}/lifecycle`, { target, note }); await refreshModels(); show(); } catch (e) { alert(e.message); } };
        box.replaceChildren(h("div", { class: "card" }, h("h3", {}, "State: ", stateChip(lc.state)), kv({ "Measured agent-integration stage": lc.stage_reached, "Last refusal": lc.evidence && lc.evidence.last_refusal }),
          h("div", { class: "row" }, lc.allowed_next.filter((s) => !["TRAINING", "TRAINED", "INITIALIZED", "FAILED"].includes(s)).map((s) => h("button", { title: NEXT_NOTES[s] || "", onclick: () => move(s) }, "→ " + s)))),
          h("h3", {}, "Evidence behind the current state"), lc.evidence && Object.keys(lc.evidence).length ? h("pre", {}, JSON.stringify(lc.evidence, null, 2)) : empty("No evidence recorded."),
          h("h3", {}, "History"), table("State changes", ["When", "From", "To", "By", "Note"], lc.history.slice().reverse().map((x) => [x.at, x.from, x.to, x.by, x.note])));
      } catch (e) { box.replaceChildren(failed(e, show)); }
    };
    show();
    return h("div", {}, h("h2", {}, "Model registry"), h("p", { class: "lead" }, "INITIALIZED → TRAINING → TRAINED → VALIDATED → DEPLOYED (ARCHIVED, FAILED). States follow real artifacts; the server refuses a state the evidence does not support."), modelPicker(show), box);
  });
}

async function secAgents(root) {
  clearTimers();
  await guard(root, "agent integration", async () => ({ p: await api("GET", "/modellab/provider"), ov: await refreshModels() }), ({ p }) => {
    const msg = h("div", { role: "status" });
    const prov = h("select", { id: "ag-prov" }, h("option", { value: "deterministic" }, "deterministic (no language model)"), h("option", { value: "custom_transformer", selected: p.selection.provider === "custom_transformer" }, "custom Transformer"));
    const mid = h("select", { id: "ag-model" }, state.models.map((m) => h("option", { value: m.model_id, selected: m.model_id === p.selection.model_id }, `${m.model_id} (${m.state})`)));
    const mode = h("select", { id: "ag-mode" }, ["experimental", "production"].map((x) => h("option", { value: x, selected: x === p.selection.mode }, x)));
    const exp = h("input", { type: "checkbox", id: "ag-exp", checked: p.selection.allow_experimental }), save = h("button", { class: "primary", type: "submit" }, "Save selection");
    const form = h("form", { class: "card" }, h("h3", {}, "Provider selection"), h("label", { for: "ag-prov" }, "Provider"), prov, h("label", { for: "ag-model" }, "Model"), mid, h("label", { for: "ag-mode" }, "Mode"), mode,
      h("label", {}, exp, "Allow explicit experimental use (output stays an untrusted draft note)"), h("div", { class: "row" }, save), msg);
    form.addEventListener("submit", async (ev) => {
      ev.preventDefault(); save.disabled = true;
      const body = { provider: prov.value, mode: mode.value, allow_experimental: exp.checked }; if (prov.value === "custom_transformer") body.model_id = mid.value;
      try { await api("PUT", "/modellab/selection", body); await secAgents(root); } catch (e) { msg.replaceChildren(failed(e)); save.disabled = false; }
    });
    const eff = p.effective;
    return h("div", {}, h("h2", {}, "Agent integration status"), h("p", { class: "lead" }, p.note),
      h("div", { class: "card" }, h("h3", {}, "Effective provider"), kv({ Provider: eff.provider, Capability: eff.capability, Mode: eff.mode, Model: eff.model_id, "Lifecycle state": eff.lifecycle_state, "Measured stage": eff.stage_reached,
        "Hosted API ever used": p.never_uses_hosted_api ? "never" : "?", Fallback: eff.fallback_reason, Notes: (eff.notes || []).join(" | ") })), p.error ? failed(p.error) : null,
      form, h("h3", {}, "Integration stages"), table("A stage is only reached by measurement (section 9 runs the checks)", ["Stage", "Requirement"], Object.entries(p.stages)),
      h("h3", {}, "Roles that may attach untrusted model notes"), p.roles.length ? h("p", {}, p.roles.join(", ")) : empty("None: agents run on deterministic software only."));
  });
}

const RENDER = { overview: secOverview, tokenizer: secTokenizer, datasets: secDatasets, architecture: secArchitecture, config: secConfig, runs: secRuns, metrics: secMetrics,
  checkpoints: secCheckpoints, evaluation: secEvaluation, playground: secPlayground, registry: secRegistry, agents: secAgents };

/* ------------------------------------------------------------------ shell, sign-in */
function renderNav() {
  const cur = location.hash.slice(1) || "overview", nav = document.getElementById("nav");
  nav.replaceChildren(...SECTIONS.map(([id, label]) => h("a", { href: "#" + id, "aria-current": id === cur ? "page" : false }, label)));
}
async function route() {
  clearTimers(); renderNav();
  const id = location.hash.slice(1) || "overview", main = document.getElementById("main");
  const f = RENDER[id] || secOverview;
  main.replaceChildren(h("div", { id: "section" })); await f(document.getElementById("section"));
}

function signIn(msg) {
  const main = document.getElementById("main"), em = h("input", { id: "em", type: "email", autocomplete: "username", required: true }), pw = h("input", { id: "pw", type: "password", autocomplete: "current-password", required: true });
  const err = h("div", { role: "alert" }), form = h("form", { class: "card" }, h("h2", {}, "Sign in"), msg ? h("p", { class: "muted" }, msg) : null, h("label", { for: "em" }, "Email"), em, h("label", { for: "pw" }, "Password"), pw,
    h("div", { class: "row" }, h("button", { class: "primary", type: "submit" }, "Sign in"), h("button", { type: "button", id: "reg" }, "Create account")), err);
  const go = async (path) => { try { const r = await api("POST", path, { email: em.value, password: pw.value }); state.csrf = r.csrf_token; state.user = r.user; start(); } catch (e) { err.replaceChildren(failed(e)); } };
  form.addEventListener("submit", (ev) => { ev.preventDefault(); go("/auth/login"); }); form.querySelector("#reg").addEventListener("click", () => go("/auth/register"));
  document.getElementById("nav").replaceChildren(); main.replaceChildren(form);
}

async function start() {
  document.getElementById("who").replaceChildren(state.user.email + " ", h("button", { onclick: async () => { await api("POST", "/auth/logout"); state.user = null; document.getElementById("who").replaceChildren(); signIn("Signed out."); } }, "Sign out"));
  window.onhashchange = route; await route();
}

(async function boot() {
  try { const r = await api("GET", "/auth/me"); state.csrf = r.csrf_token; state.user = r.user; await start(); }
  catch (e) { signIn(e.status === 401 ? "" : "Could not reach the API: " + e.message); }
})();

/* FlowTrace dashboard — dependency-free (works offline). */
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => Array.from(el.querySelectorAll(s));
const api = async (path, opts = {}) => {
  const r = await fetch(path, opts);
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
};
const post = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
const fmtBytes = (b) => { if (!b && b !== 0) return "–"; const u = ["B", "KB", "MB", "GB", "TB"]; let i = 0; while (Math.abs(b) >= 1024 && i < 4) { b /= 1024; i++; } return b.toFixed(b < 10 ? 2 : 1) + " " + u[i]; };
const fmt = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v)) ? "–" : (Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(2) : (+v).toPrecision(d));
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
function b64f32(s) { const bin = atob(s); const buf = new ArrayBuffer(bin.length); const u8 = new Uint8Array(buf); for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i); return new Float32Array(buf); }
function b64i32(s) { const bin = atob(s); const buf = new ArrayBuffer(bin.length); const u8 = new Uint8Array(buf); for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i); return new Int32Array(buf); }

/* ------------------------------------------------------------------ strings & theme */
const store = { get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* private mode: not remembered */ } } };
const LANG = "en";
const t = (k) => ((window.FT_I18N || {})[LANG] || {})[k] || ((window.FT_I18N || {}).en || {})[k] || k;
function applyI18n() {
  const root = document.documentElement;
  root.lang = LANG; root.dir = "ltr";
  $$("[data-i18n]").forEach((el) => { el.textContent = t(el.dataset.i18n); });
  $$("[data-i18n-aria]").forEach((el) => el.setAttribute("aria-label", t(el.dataset.i18nAria)));
  $$("[data-i18n-placeholder]").forEach((el) => el.setAttribute("placeholder", t(el.dataset.i18nPlaceholder)));
  const pill = $("#statusPill"); if (pill.dataset.status) pill.textContent = statusText(pill.dataset.status, pill.dataset.label);
  updateThemeButton();
}
function statusText(st, label) { return t("st_" + st) + (label && st === "running" ? ` · ${label}` : ""); }
const THEMES = ["auto", "light", "dark"];
function currentTheme() { return document.documentElement.dataset.theme || "auto"; }
function updateThemeButton() {
  const th = currentTheme(), b = $("#themeBtn");
  const key = th === "light" ? "themeLight" : th === "dark" ? "themeDark" : "themeAuto";
  b.setAttribute("aria-label", t(key)); b.title = t(key); b.textContent = th === "light" ? "☀" : th === "dark" ? "☾" : "◐";
}
function cycleTheme() {
  const next = THEMES[(THEMES.indexOf(currentTheme()) + 1) % THEMES.length];
  if (next === "auto") delete document.documentElement.dataset.theme; else document.documentElement.dataset.theme = next;
  store.set("ft-theme", next); updateThemeButton(); redrawAll();
}
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
function redrawAll() { draw(); drawResiduals(); drawMonitor(); }

/* ------------------------------------------------------------------ colormaps */
const CMAPS = {
  turbo(x) {
    x = Math.min(1, Math.max(0, x));
    const x2 = x * x, x3 = x2 * x, x4 = x3 * x, x5 = x4 * x;
    const r = 0.13572138 + 4.61539260 * x - 42.66032258 * x2 + 132.13108234 * x3 - 152.94239396 * x4 + 59.28637943 * x5;
    const g = 0.09140261 + 2.19418839 * x + 4.84296658 * x2 - 14.18503333 * x3 + 4.27729857 * x4 + 2.82956604 * x5;
    const b = 0.10667330 + 12.64194608 * x - 60.58204836 * x2 + 110.36276771 * x3 - 89.90310912 * x4 + 27.34824973 * x5;
    return [r, g, b];
  },
  viridis(t) {
    t = Math.min(1, Math.max(0, t));
    const c = [[0.2777273272234177, 0.005407344544966578, 0.3340998053353061], [0.1050930431085774, 1.404613529898575, 1.384590162594685],
      [-0.3308618287255563, 0.214847559468213, 0.09509516302823659], [-4.634230498983486, -5.799100973351585, -19.33244095627987],
      [6.228269936347081, 14.17993336680509, 56.69055260068105], [4.776384997670288, -13.74514537774601, -65.35303263337234],
      [-5.435455855934631, 4.645852612178535, 26.3124352495832]];
    return [0, 1, 2].map((k) => { let v = c[6][k]; for (let i = 5; i >= 0; i--) v = c[i][k] + t * v; return v; });
  },
  rainbow(t) { // Fluent-like blue -> cyan -> green -> yellow -> red
    t = Math.min(1, Math.max(0, t));
    const h = (1 - t) * 240 / 60, x = 1 - Math.abs((h % 2) - 1);
    const s = [[1, x, 0], [x, 1, 0], [0, 1, x], [0, x, 1], [x, 0, 1]][Math.min(4, Math.floor(h))];
    return s;
  },
  coolwarm(t) {
    t = Math.min(1, Math.max(0, t));
    const a = [59, 76, 192], m = [221, 221, 221], b = [180, 4, 38];
    const lerp = (p, q, u) => p.map((v, i) => (v + (q[i] - v) * u) / 255);
    return t < 0.5 ? lerp(a, m, t * 2) : lerp(m, b, (t - 0.5) * 2);
  },
  gray(t) { t = Math.min(1, Math.max(0, t)); return [t, t, t]; },
};
const LUT = {};
function lut(name) {
  if (!LUT[name]) {
    const a = new Uint8ClampedArray(256 * 3);
    for (let i = 0; i < 256; i++) { const c = CMAPS[name](i / 255); a[3 * i] = c[0] * 255; a[3 * i + 1] = c[1] * 255; a[3 * i + 2] = c[2] * 255; }
    LUT[name] = a;
  }
  return LUT[name];
}

/* ------------------------------------------------------------------ state */
const S = {
  source: "structured", presets: [], system: null, meshId: null, meshInfo: null, geometry: null, frame: null,
  res: { keys: [], series: {}, x: [], labels: [] }, mon: [], monName: "", resCount: 0, job: null, view: null,
  cellPaths: null, cellCentres: null,
};

/* ------------------------------------------------------------------ UI wiring */
function selectTab(b, focus) {
  $$("#tabs button").forEach((x) => {
    const on = x === b;
    x.classList.toggle("active", on); x.setAttribute("aria-selected", on ? "true" : "false"); x.tabIndex = on ? 0 : -1;
  });
  $$(".tab-body > section").forEach((s) => s.classList.toggle("active", s.dataset.tab === b.dataset.tab));
  if (focus) b.focus();
}
function initTabs() {
  const tabs = $$("#tabs button");
  tabs.forEach((b, i) => {
    b.addEventListener("click", () => selectTab(b));
    b.addEventListener("keydown", (e) => {
      const rtl = document.documentElement.dir === "rtl";
      let j = null;
      if (e.key === (rtl ? "ArrowLeft" : "ArrowRight")) j = (i + 1) % tabs.length;
      else if (e.key === (rtl ? "ArrowRight" : "ArrowLeft")) j = (i - 1 + tabs.length) % tabs.length;
      else if (e.key === "Home") j = 0; else if (e.key === "End") j = tabs.length - 1;
      if (j !== null) { e.preventDefault(); selectTab(tabs[j], true); }
    });
  });
  $$("#sourceSeg button").forEach((b) => b.addEventListener("click", () => setSource(b.dataset.src)));
}
function setSource(src) {
  S.source = src;
  $$("#sourceSeg button").forEach((x) => { x.classList.toggle("active", x.dataset.src === src); x.setAttribute("aria-pressed", x.dataset.src === src ? "true" : "false"); });
  $("#structuredCase").classList.toggle("hidden", src !== "structured");
  $("#unstructuredCase").classList.toggle("hidden", src !== "unstructured");
  updateSolverCards();
}
function presetKind() {
  const p = S.presets.find((x) => x.id === $("#presetSelect").value);
  return p ? p.kind : "structured";
}
function updateSolverCards() {
  const src = S.source, uPreset = src === "structured" && presetKind() === "unstructured";
  const pressureBased = src === "unstructured" || uPreset || $("#mode").value === "steady-simplec";
  $("#structSolverCard").classList.toggle("hidden", src !== "structured" || uPreset);
  $("#gridCard").classList.toggle("hidden", uPreset);
  $("#unstructSolverCard").classList.toggle("hidden", !pressureBased);
  $("#modelsCard").classList.toggle("hidden", !pressureBased || uPreset);
}

function renderPresetParams() {
  const p = S.presets.find((x) => x.id === $("#presetSelect").value);
  if (!p) return;
  $("#presetDesc").textContent = p.description;
  const box = $("#presetParams");
  box.innerHTML = "";
  const labels = { re: "Reynolds number", ra: "Rayleigh number", pr: "Prandtl number", n: "Cells per side", nx: "Cells in x", ny: "Cells in y",
    n_per_d: "Cells per diameter", n_per_h: "Cells per step height", length: "Channel length", aspect: "Aspect ratio",
    re_tau: "Friction Reynolds number Re_τ", yplus1: "First-cell y+", re_l: "Reynolds number per unit length",
    tau: "Optical thickness τ", t_hot: "Hot wall T (K)", t_cold: "Cold wall T (K)", dt: "Time step", t_end: "End time",
    diffusivity: "Species diffusivity D" };
  for (const [k, v] of Object.entries(p.params)) {
    const l = document.createElement("label");
    l.className = "field";
    l.innerHTML = `${esc(labels[k] || k)}<input type="number" step="any" data-param="${k}" value="${v}">`;
    box.appendChild(l);
  }
  const d = p.defaults;
  if (p.kind === "unstructured") {
    $("#maxIter").value = d.max_iter; $("#uTol").value = d.tol;
    $("#algorithm").value = d.algorithm; $("#uScheme").value = d.scheme;
    $("#relaxU").value = d.relax_u; $("#relaxP").value = d.relax_p ?? "";
  } else {
    $("#tEnd").value = d.t_end ?? "";
    $("#maxSteps").value = d.max_steps;
    $("#steadyTol").value = d.steady_tol ?? "";
    $("#scheme").value = d.scheme;
    $("#cfl").value = d.cfl;
  }
  $("#caseTitle").textContent = p.title;
  updateSolverCards();
}

async function loadSystem() {
  const sys = await api("/api/system");
  S.system = sys;
  const p = sys.profile;
  const g = p.best_gpu;
  const rows = [
    ["OS", p.os], ["CPU", p.cpu_model], ["Cores", `${p.physical_cores} physical / ${p.logical_cores} logical`],
    ["SIMD", (p.simd || []).join(" ") || "–"], ["RAM", `${fmtBytes(p.ram_available)} free of ${fmtBytes(p.ram_total)}`],
    ["Disk (work dir)", `${fmtBytes(p.disk_free)} free`], ["Power", p.on_battery === null ? "n/a" : (p.on_battery ? "battery" : "AC")],
    ["Backends", sys.backends.join(", ")],
    ["Libraries", Object.entries(p.libraries).map(([k, v]) => `${k} ${v || "—"}`).join(" · ")],
  ];
  $("#sysTable").innerHTML = rows.map(([k, v]) => `<tr><td>${esc(k)}</td><td>${esc(v)}</td></tr>`).join("");
  $("#gpuCount").textContent = `${p.gpus.length} found`;
  $("#gpuBox").innerHTML = p.gpus.length ? p.gpus.map((x) => `
    <table class="kv"><tr><td>Name</td><td>${esc(x.name)} ${x.usable ? '<span class="tag gpu">usable</span>' : '<span class="tag">driver only</span>'}</td></tr>
    <tr><td>VRAM</td><td>${fmtBytes(x.vram_free)} free / ${fmtBytes(x.vram_total)}</td></tr>
    <tr><td>Compute capability</td><td>${esc(x.compute_capability || "?")} · ${x.multiprocessors || "?"} SMs</td></tr>
    <tr><td>Tensor / RT cores</td><td>${x.tensor_cores ? "yes" : "no"} / ${x.rt_cores ? "yes" : "no"}</td></tr>
    <tr><td>FP64 : FP32</td><td>1 : ${Math.round(1 / x.fp64_rate)}</td></tr></table>`).join("")
    : esc(t("noGpu"));
  $("#sysNotes").innerHTML = (p.notes || []).map((n) => `• ${esc(n)}`).join("<br>") || "–";
  const be = $("#backend");
  be.innerHTML = '<option value="auto">Auto (measure & choose)</option>' + sys.backends.map((b) => `<option value="${b}">${b}</option>`).join("");
  if (sys.bench) renderBench(sys.bench);
  const pf = sys.pyfluent;
  $("#pyfluentBox").textContent = pf.available ? `${t("pyfluentYes")} (PyFluent ${pf.version})` : t("pyfluentNo");
}

function renderBench(b) {
  const r = b.results || {};
  const keys = Object.keys(r);
  if (!keys.length) return;
  const rows = [["Memory bandwidth", "bandwidth_gbs", "GB/s"], ["GEMM FP64", "gemm_gflops_fp64", "GFLOP/s"], ["GEMM FP32", "gemm_gflops_fp32", "GFLOP/s"],
    ["DCT 512²", "dct512_ms", "ms"], ["5-pt stencil", "stencil_mlups", "Mcell/s"], ["Launch latency", "launch_us", "µs"], ["Host→GPU", "h2d_gbs", "GB/s"], ["GPU→Host", "d2h_gbs", "GB/s"]];
  $("#benchBox").innerHTML = `<table class="grid"><tr><th></th>${keys.map((k) => `<th>${k}</th>`).join("")}</tr>` +
    rows.map(([lab, key, unit]) => `<tr><td>${lab} <span class="muted">${unit}</span></td>${keys.map((k) => `<td class="num">${r[k][key] !== undefined ? fmt(r[k][key], 4) : (r[k].error ? "err" : "–")}</td>`).join("")}</tr>`).join("") + "</table>";
}

async function loadPresets() {
  const r = await api("/api/presets");
  const sp = r.presets.map((p) => ({ ...p, kind: "structured" }));
  const up = (r.unstructured_presets || []).map((p) => ({ ...p, kind: "unstructured" }));
  S.presets = sp.concat(up);
  const opt = (p) => `<option value="${p.id}">${esc(p.title)}</option>`;
  $("#presetSelect").innerHTML = `<optgroup label="Structured (MAC, GPU-ready)">${sp.map(opt).join("")}</optgroup>` +
    (up.length ? `<optgroup label="Fluent-style (SIMPLEC, unstructured)">${up.map(opt).join("")}</optgroup>` : "");
  $("#presetSelect").addEventListener("change", renderPresetParams);
  renderPresetParams();
}

/* ------------------------------------------------------------------ mesh upload */
function initMeshUpload() {
  const dz = $("#dropzone"), fi = $("#meshFile");
  dz.addEventListener("click", () => fi.click());
  dz.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fi.click(); } });
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("drag"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("drag"));
  dz.addEventListener("drop", (e) => { e.preventDefault(); dz.classList.remove("drag"); if (e.dataTransfer.files[0]) uploadMesh(e.dataTransfer.files[0]); });
  fi.addEventListener("change", () => fi.files[0] && uploadMesh(fi.files[0]));
}
async function uploadMesh(file) {
  $("#meshSummary").textContent = `${t("reading")} ${file.name} …`;
  try {
    const buf = await file.arrayBuffer();
    const r = await fetch("/api/mesh", { method: "POST", headers: { "X-Filename": file.name }, body: buf });
    const info = await r.json();
    if (!r.ok) throw new Error(info.error);
    await renderMeshInfo(info);
  } catch (e) { $("#meshSummary").innerHTML = `<span style="color:var(--err)">${esc(e.message)}</span>`; }
}
async function renderMeshInfo(info) {
  S.meshId = info.id; S.meshInfo = info;
  $("#meshSummary").innerHTML = `<b>${esc(info.name)}</b> · ${info.dim}D · ${info.cells.toLocaleString()} cells · ${info.faces.toLocaleString()} faces · volume ${fmt(info.volume, 5)} · ${info.read_seconds}s` +
    (info.repaired_faces ? ` · <span style="color:var(--warn)">${info.repaired_faces} left-handed faces repaired</span>` : "");
  const q = info.quality;
  if (q) {
    const warn = (q.warnings || []).map((w) => `<div style="color:${w.startsWith("CRITICAL") ? "var(--err)" : "var(--warn)"}">${esc(w)}</div>`).join("");
    $("#meshQuality").innerHTML = `<b>${esc(t("quality"))}</b>: ${esc(t("oq"))} ${fmt(q.min_orthogonal_quality, 3)} · ${esc(t("skew"))} ${fmt(q.max_skewness, 3)} · ${esc(t("ar"))} ${fmt(q.max_aspect_ratio, 3)}` +
      (warn || ` · <span style="color:var(--ok)">${esc(t("qualityOk"))}</span>`);
  }
  if (info.unit_hint) {
    $("#meshQuality").innerHTML += `<div style="color:var(--warn)">Units: ${esc(info.unit_hint.why)}. With "Mesh length unit" on auto-detect the mesh is scaled from ${esc(info.unit_hint.unit)} to metres. If the file really is in metres, choose "m (as in the file)": the Reynolds number differs by orders of magnitude between the two.</div>`;
  }
  renderZoneTable(info);
  $("#zoneCard").classList.remove("hidden");
  const solids = info.solid_zones || [];
  $("#solidCard").classList.toggle("hidden", !solids.length);
  $("#solidTable").innerHTML = solids.map((z) => `<tr data-solid="${esc(z)}"><td><b>${esc(z)}</b><div class="row" style="margin-top:4px">
      <label class="grow"><span class="sr-only">${esc(z)} ${esc(t("conductivityOf"))}</span><input type="number" step="any" data-k="k" value="1" title="${esc(t("conductivityOf"))}"></label>
      <label class="grow"><span class="sr-only">${esc(z)} ${esc(t("source"))}</span><input type="number" step="any" data-k="q" value="0" title="${esc(t("source"))}"></label></div></td></tr>`).join("");
  if (solids.length) $("#uEnergy").checked = true;
  $("#propsCard").classList.remove("hidden");
  $("#caseTitle").textContent = info.name;
  if (info.dim === 2) {
    S.geometry = parseGeometry(await api(`/api/mesh/${info.id}`));
    S.geom3 = null; S.cellPaths = null; S.frame = null; draw();
  } else {
    // 3-D: a picture of the boundary surfaces (sampled face centres, one colour per zone)
    const g = await api(`/api/mesh/${info.id}`);
    S.geometry = null; S.frame = null;
    S.geom3 = { names: g.names, types: g.types, pts: b64f32(g.pts), zid: b64i32(g.zid), n: g.n, yaw: -0.6, pitch: 0.35, zoom: 1 };
    fitMesh3D(S.geom3); $("#empty").classList.add("hidden"); draw();
  }
}
function fitMesh3D(g) {
  // frame the walls (the body) when there are any, otherwise everything
  const walls = g.types.map((ty, i) => (ty === "wall" ? i : -1)).filter((i) => i >= 0);
  const use = (z) => (walls.length ? walls.includes(z) : true);
  const lo = [1e300, 1e300, 1e300], hi = [-1e300, -1e300, -1e300];
  for (let i = 0; i < g.n; i++) if (use(g.zid[i])) for (let d = 0; d < 3; d++) {
    const v = g.pts[3 * i + d]; if (v < lo[d]) lo[d] = v; if (v > hi[d]) hi[d] = v;
  }
  g.c = lo.map((v, d) => 0.5 * (v + hi[d]));
  g.r = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], 1e-12) * 0.5 * 1.3;
}
const ZONE_HUES = [18, 200, 140, 45, 280, 350, 170, 320, 90, 235];
const KIND_COLOR = { inlet: "#4aa3df", outlet: "#e07b53", pressure: "#e07b53", wall: "#b0aea5", symmetry: "#7fb685", periodic: "#c9788a" };
function hiKind() {
  const z = S.hiZone; if (!z) return null;
  const tr = document.querySelector(`#zoneTable tr[data-zone="${CSS.escape(z)}"]`);
  const sel = tr && tr.querySelector("select[data-k=kind]");
  return sel ? sel.value : null;
}
function drawArrow(ctx, x, y, dx, dy, col) {
  const L = 26, hx = x + dx * L, hy = y + dy * L;
  ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 2.4;
  ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(hx, hy); ctx.stroke();
  const a = Math.atan2(dy, dx), h = 8;
  ctx.beginPath(); ctx.moveTo(hx, hy); ctx.lineTo(hx - h * Math.cos(a - 0.45), hy - h * Math.sin(a - 0.45));
  ctx.lineTo(hx - h * Math.cos(a + 0.45), hy - h * Math.sin(a + 0.45)); ctx.closePath(); ctx.fill();
}
function drawZoneHighlight2D(ctx, g, vr, x0, y0) {
  if (!S.hiZone || !g.segs || !g.zones.length) return;
  const zi = g.zones.findIndex((z) => z.name === S.hiZone); if (zi < 0) return;
  const kind = hiKind() || "wall", col = KIND_COLOR[kind] || "#e07b53";
  const P = (n) => [vr.x + (g.nodes[2 * n] - x0) * vr.s, vr.y + vr.h - (g.nodes[2 * n + 1] - y0) * vr.s];
  const cx = vr.x + vr.w / 2, cy = vr.y + vr.h / 2;
  ctx.save(); ctx.lineCap = "round";
  ctx.strokeStyle = "rgba(0,0,0,.35)"; ctx.lineWidth = 1;
  const mids = [];
  for (let k = 0; k + 2 < g.segs.length; k += 3) {
    const [ax, ay] = P(g.segs[k]), [bx, by] = P(g.segs[k + 1]);
    if (g.segs[k + 2] === zi) { mids.push([(ax + bx) / 2, (ay + by) / 2, bx - ax, by - ay]); continue; }
    ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
  }
  ctx.strokeStyle = col; ctx.lineWidth = 5;
  for (let k = 0, i = 0; k + 2 < g.segs.length; k += 3) {
    if (g.segs[k + 2] !== zi) continue;
    const [ax, ay] = P(g.segs[k]), [bx, by] = P(g.segs[k + 1]);
    ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke(); i++;
  }
  if (mids.length && (kind === "inlet" || kind === "outlet" || kind === "pressure")) {
    const step = Math.max(1, Math.floor(mids.length / 5));
    for (let i = Math.floor(step / 2); i < mids.length; i += step) {
      const [mx, my, tx, ty] = mids[i]; const len = Math.hypot(tx, ty) || 1;
      let nx = -ty / len, ny = tx / len;
      if (nx * (cx - mx) + ny * (cy - my) < 0) { nx = -nx; ny = -ny; } // pointing into the domain
      if (kind === "inlet") drawArrow(ctx, mx - nx * 30, my - ny * 30, nx, ny, col); else drawArrow(ctx, mx, my, -nx, -ny, col);
    }
  }
  const m = mids[Math.floor(mids.length / 2)];
  if (m) {
    const label = `${S.hiZone} · ${kind}`;
    ctx.font = "600 12px system-ui, sans-serif"; const w = ctx.measureText(label).width + 14;
    const lx = Math.max(6, Math.min(ctx.canvas.clientWidth - w - 6, m[0] - w / 2)), ly = Math.max(6, Math.min(ctx.canvas.clientHeight - 26, m[1] - 40));
    ctx.fillStyle = col; ctx.beginPath(); ctx.roundRect(lx, ly, w, 22, 6); ctx.fill();
    ctx.fillStyle = "#1f1e1d"; ctx.fillText(label, lx + 7, ly + 15);
  }
  ctx.restore();
}
function drawMesh3D(ctx, W, H, g) {
  const cy = Math.cos(g.yaw), sy = Math.sin(g.yaw), cp = Math.cos(g.pitch), sp = Math.sin(g.pitch);
  const sc = 0.5 * Math.min(W, H) / g.r * g.zoom;
  const hidden = g.hidden || new Set();
  const order = g.names.map((_, i) => i);
  const hz = S.hiZone ? g.names.indexOf(S.hiZone) : -1;
  for (const z of order) {
    if (hidden.has(z)) continue;
    ctx.globalAlpha = hz >= 0 && z !== hz ? 0.18 : 1;
    ctx.fillStyle = hz === z ? (KIND_COLOR[hiKind() || "wall"] || "#e07b53") : `hsl(${ZONE_HUES[z % ZONE_HUES.length]} 70% 60%)`;
    const sz = hz === z ? 3.2 : 1.6;
    for (let i = 0; i < g.n; i++) {
      if (g.zid[i] !== z) continue;
      const x = g.pts[3 * i] - g.c[0], y = g.pts[3 * i + 1] - g.c[1], zz = g.pts[3 * i + 2] - g.c[2];
      const x1 = cy * x + sy * zz, z1 = -sy * x + cy * zz;
      const y2 = cp * y - sp * z1;
      const px = W / 2 + sc * x1, py = H / 2 - sc * y2;
      if (px < -2 || px > W + 2 || py < -2 || py > H + 2) continue;
      ctx.fillRect(px, py, sz, sz);
    }
  }
  ctx.globalAlpha = 1;
  ctx.font = "12px system-ui, sans-serif";
  g.names.forEach((nm, z) => {
    ctx.fillStyle = hidden.has(z) ? cssVar("--line") : `hsl(${ZONE_HUES[z % ZONE_HUES.length]} 70% 60%)`;
    ctx.fillText("■ " + nm + (g.types[z] ? " (" + g.types[z] + ")" : ""), 10, 18 + 16 * z);
  });
  ctx.fillStyle = cssVar("--muted");
  ctx.fillText("drag: rotate · wheel: zoom · double-click: reset · click a legend entry: show/hide", 10, H - 10);
}
function bindMesh3DControls() {
  const cv = $("#view"); let drag = null;
  cv.addEventListener("pointerdown", (e) => { if (S.geom3 && !S.frame) { drag = [e.clientX, e.clientY]; cv.setPointerCapture(e.pointerId); } });
  cv.addEventListener("pointermove", (e) => {
    if (!drag || !S.geom3) return;
    S.geom3.yaw += (e.clientX - drag[0]) * 0.01; S.geom3.pitch += (e.clientY - drag[1]) * 0.01;
    drag = [e.clientX, e.clientY]; draw();
  });
  cv.addEventListener("pointerup", () => { drag = null; });
  cv.addEventListener("wheel", (e) => {
    if (!S.geom3 || S.frame) return; e.preventDefault();
    S.geom3.zoom *= Math.exp(-e.deltaY * 0.001); draw();
  }, { passive: false });
  cv.addEventListener("dblclick", () => { if (S.geom3) { S.geom3.zoom = 1; S.geom3.yaw = -0.6; S.geom3.pitch = 0.35; draw(); } });
  cv.addEventListener("click", (e) => {
    const g = S.geom3; if (!g || S.frame) return;
    const r = cv.getBoundingClientRect(), z = Math.floor((e.clientY - r.top - 8) / 16);
    if (e.clientX - r.left < 220 && z >= 0 && z < g.names.length) { g.hidden = g.hidden || new Set(); g.hidden.has(z) ? g.hidden.delete(z) : g.hidden.add(z); draw(); }
  });
}
function renderZoneTable(info) {
  const t = $("#zoneTable");
  const kinds = ["wall", "inlet", "outlet", "pressure", "symmetry", "periodic"];
  const dim = info.dim;
  t.innerHTML = info.zones.filter((z) => z.kind === "face" && z.type !== "interior").map((z) => {
    const b = info.boundaries[z.name] || { kind: "wall" };
    return `<tr data-zone="${esc(z.name)}"><td>
      <div class="row" style="justify-content:space-between"><b>${esc(z.name)}</b><span class="muted">${esc(z.type)} · ${z.count} faces</span></div>
      <div class="row" style="margin-top:4px">
        <select data-k="kind" style="width:96px">${kinds.map((k) => `<option ${k === b.kind ? "selected" : ""}>${k}</option>`).join("")}</select>
        <input type="text" data-k="val" title="inlet: normal speed or u,v${dim === 3 ? ",w" : ""}; wall: wall velocity u,v${dim === 3 ? ",w" : ""}; outlet: gauge pressure" placeholder="speed / u,v / p" value="${b.kind === "inlet" ? (b.normal_speed ?? 1) : ""}">
        <input type="text" data-k="therm" title="T=value (fixed temperature) or q=value (heat flux into the domain)" placeholder="T=… / q=…" value="">
      </div></td></tr>`;
  }).join("");
}
function zoneSpecs() {
  const out = {};
  $$("#zoneTable tr[data-zone]").forEach((tr) => {
    const kind = $("select[data-k=kind]", tr).value;
    const val = $("input[data-k=val]", tr).value.trim();
    const th = $("input[data-k=therm]", tr).value.trim();
    const spec = { kind };
    if (val) {
      const nums = val.split(/[ ,;]+/).map(Number).filter((x) => !Number.isNaN(x));
      if (kind === "inlet") { if (nums.length === 1) spec.normal_speed = nums[0]; else spec.velocity = nums; }
      else if (kind === "wall") { if (nums.length > 1) spec.velocity = nums; }
      else if (kind === "outlet" || kind === "pressure") spec.pressure = nums[0] || 0;
    }
    if (th) {
      const m = th.match(/(?:^|\s)([TtQq])\s*=\s*([-+0-9.eE]+)/);
      if (m) { spec.thermal = m[1].toLowerCase() === "t" ? "temperature" : "flux"; spec.value = +m[2]; }
      else if (!Number.isNaN(+th)) { spec.thermal = "temperature"; spec.value = +th; }
      const e = th.match(/(?:^|\s)e\s*=\s*([0-9.]+)/);
      if (e) spec.emissivity = +e[1];
      for (const mm of th.matchAll(/(?:^|\s)([A-Za-z][\w\-]*)\s*=\s*([-+0-9.eE]+)/g)) {
        if (!["T", "t", "q", "Q", "e"].includes(mm[1])) { spec.species = spec.species || {}; spec.species[mm[1]] = +mm[2]; }
      }
    }
    out[tr.dataset.zone] = spec;
  });
  return out;
}

/* ------------------------------------------------------------------ run */
function modelSettings() {
  const vec = (txt) => txt.trim() ? txt.split(/[ ,;]+/).map(Number).filter((x) => !Number.isNaN(x)) : [0, 0, 0];
  return {
    turbulence: $("#uTurb").value, turb_intensity: +$("#turbI").value, turb_viscosity_ratio: +$("#turbR").value,
    turb_scheme: $("#turbScheme").value, radiation: $("#radModel").value, absorption: +$("#radA").value,
    periodic_pressure_gradient: vec($("#perG").value), unsteady: $("#unsteady").checked, dt: +$("#uDt").value,
    t_end: +$("#uTend").value, inner_iterations: +$("#uInner").value, time_scheme: $("#uTimeScheme").value,
    species: $("#species").value.split(/[ ,;]+/).filter((x) => x), species_diffusivity: +$("#speciesD").value,
  };
}
function readParams() {
  const params = {};
  $$("#presetParams input").forEach((i) => { params[i.dataset.param] = Number.isInteger(+i.value) && !i.value.includes(".") && !/e/i.test(i.value) ? parseInt(i.value, 10) : +i.value; });
  return params;
}
function buildSpec() {
  const backend = $("#backend").value, precision = $("#precision").value;
  if (S.source === "structured" && presetKind() === "unstructured") {
    return {
      kind: "unstructured", preset: $("#presetSelect").value, params: readParams(), checkpoint_seconds: +$("#ckpt").value || 0,
      field: $("#fieldSelect").value || null,
      overrides: { backend, max_iter: +$("#maxIter").value, tol: +$("#uTol").value, algorithm: $("#algorithm").value,
        scheme: $("#uScheme").value, relax_u: +$("#relaxU").value,
        relax_p: $("#relaxP").value.trim() === "" ? null : +$("#relaxP").value },
    };
  }
  if (S.source === "structured") {
    const params = {};
    $$("#presetParams input").forEach((i) => { params[i.dataset.param] = Number.isInteger(+i.value) && !i.value.includes(".") && !/e/i.test(i.value) ? parseInt(i.value, 10) : +i.value; });
    const tEnd = $("#tEnd").value.trim(), tol = $("#steadyTol").value.trim();
    return {
      kind: "structured", preset: $("#presetSelect").value, params, mode: $("#mode").value, fmg: $("#fmg").checked,
      checkpoint_seconds: +$("#ckpt").value || 0, field: $("#fieldSelect").value || null,
      max_iter: +$("#maxIter").value, tol: +$("#uTol").value,
      overrides: { backend, precision, scheme: $("#scheme").value, cfl: +$("#cfl").value, poisson: $("#poisson").value,
        max_steps: +$("#maxSteps").value, t_end: tEnd === "" ? null : +tEnd, steady_tol: tol === "" ? null : +tol },
    };
  }
  if (!S.meshId) throw new Error(t("needMesh"));
  const solids = {};
  $$("#solidTable tr[data-solid]").forEach((tr) => { solids[tr.dataset.solid] = { conductivity: +$("input[data-k=k]", tr).value, source: +$("input[data-k=q]", tr).value }; });
  return {
    kind: "unstructured", mesh_id: S.meshId, checkpoint_seconds: +$("#ckpt").value || 0,
    auto: $("#autoOpts").checked, mesh_units: $("#meshUnits").value, certify: +$("#certify").value || 0,
    case: {
      name: S.meshInfo.name, boundaries: zoneSpecs(), density: +$("#uDensity").value, viscosity: +$("#uViscosity").value,
      energy: $("#uEnergy").checked || Object.keys(solids).length > 0 || $("#radModel").value !== "off",
      conductivity: +$("#uConductivity").value, cp: +$("#uCp").value, solids, ...modelSettings(),
      algorithm: $("#algorithm").value, scheme: $("#uScheme").value, relax_u: +$("#relaxU").value, relax_p: $("#relaxP").value.trim() === "" ? null : +$("#relaxP").value,
      max_iter: +$("#maxIter").value, tol: +$("#uTol").value, backend, precision: precision === "fp32" ? "fp32" : "fp64",
    },
  };
}
async function run() {
  try {
    const spec = buildSpec();
    resetRunView();
    await post("/api/run", spec);
  } catch (e) { logLine({ level: "error", msg: e.message }); }
}
function resetRunView() {
  S.res = { keys: [], series: {}, x: [], labels: [] }; S.mon = []; S.resCount = 0;
  $("#console").innerHTML = ""; $("#validation").innerHTML = `<div class="muted">${esc(t("running"))}</div>`;
  $("#planTable").innerHTML = `<tr><td class="muted">${esc(t("measuring"))}</td></tr>`;
  drawResiduals(); drawMonitor();
}

/* ------------------------------------------------------------------ events */
function connectEvents() {
  const es = new EventSource("/api/events");
  es.addEventListener("status", (e) => onStatus(JSON.parse(e.data)));
  es.addEventListener("plan", (e) => onPlan(JSON.parse(e.data)));
  es.addEventListener("residuals", (e) => onResiduals(JSON.parse(e.data)));
  es.addEventListener("frame", (e) => onFrame(JSON.parse(e.data)));
  es.addEventListener("geometry", (e) => { S.geometry = parseGeometry(JSON.parse(e.data)); S.cellPaths = null; });
  es.addEventListener("log", (e) => logLine(JSON.parse(e.data)));
  es.addEventListener("progress", (e) => onProgress(JSON.parse(e.data)));
  es.onerror = () => { /* the browser reconnects automatically */ };
}
function onStatus(ev) {
  const st = ev.status;
  const pill = $("#statusPill");
  pill.dataset.status = st; pill.dataset.label = ev.label || "";
  pill.textContent = statusText(st, ev.label);
  pill.className = "pill " + st;
  const busy = st === "running" || st === "planning";
  if (st === "planning") showLoading(0, "Reading the case"); else hideLoading();
  $("#runBtn").disabled = busy; $("#stopBtn").disabled = !busy;
  $("#reportBtn").disabled = !["converged", "finished", "stopped"].includes(st);
  if (ev.name) $("#caseTitle").textContent = ev.name;
  if (ev.result) onResult(ev.result);
  if (st === "error") logLine({ level: "error", msg: ev.error });
}
function onPlan(ev) {
  const p = ev.plan;
  $("#planBackend").textContent = `${p.backend} · ${p.precision}`;
  const meas = p.measured_ms_per_step || {};
  let html = p.placements.map((x) => `<tr><td>${esc(x.stage)}<div class="muted">${esc(x.reason)}</div></td><td><span class="tag ${/GPU/i.test(x.device) ? "gpu" : "cpu"}">${esc(x.device)}</span><div class="muted">${esc(x.method)}</div></td></tr>`).join("");
  if (Object.keys(meas).length) html = `<tr><td colspan="2" class="muted">Measured ms/step: ${Object.entries(meas).map(([k, v]) => `<b style="color:var(--text)">${k}</b> ${v}`).join(" · ")}</td></tr>` + html;
  $("#planTable").innerHTML = html;
  const mem = (p.memory || {})[p.backend];
  if (mem) {
    const frac = mem.available ? Math.min(1, mem.required / mem.available) : 0;
    $("#memBox").innerHTML = `<div class="row muted" style="justify-content:space-between"><span>Memory budget (${p.device})</span><span class="mono">${fmtBytes(mem.required)} / ${fmtBytes(mem.available)}</span></div><div class="membar"><div style="width:${(frac * 100).toFixed(1)}%"></div></div>`;
  }
}
const RES_COLORS = { "x-velocity": "#ef4444", "y-velocity": "#10b981", "z-velocity": "#3b82f6", energy: "#f59e0b", k: "#a855f7", omega: "#ec4899", epsilon: "#ec4899" };
const resColor = (k) => (k === "continuity" ? cssVar("--text") : RES_COLORS[k]) || "#6366f1";
function onResiduals(ev) {
  const R = S.res;
  S.resCount++;
  R.x.push(S.resCount);
  if (!R.labels.length || R.labels[R.labels.length - 1].label !== ev.label) R.labels.push({ label: ev.label, at: S.resCount });
  for (const [k, v] of Object.entries(ev.scaled)) {
    if (!R.series[k]) { R.series[k] = []; R.keys.push(k); }
    R.series[k][S.resCount - 1] = v;
  }
  $("#sStep").textContent = ev.step.toLocaleString();
  $("#sTime").textContent = ev.dt ? fmt(ev.t, 4) : "–";
  $("#sDt").textContent = ev.dt ? fmt(ev.dt, 3) : "–";
  $("#sRate").textContent = fmt(ev.steps_per_s, 3);
  $("#sMcell").textContent = fmt(ev.mcells_per_s, 3);
  let mname = "", mval = null;
  if (ev.probe) { mname = "probe v"; mval = ev.probe[1]; }
  else if (ev.nusselt !== undefined) { mname = "Nusselt (hot wall)"; mval = ev.nusselt; }
  else if (ev.ke !== undefined) { mname = "kinetic energy"; mval = ev.ke; }
  else if (ev.linear) { mname = "p' AMG-PCG iterations"; mval = ev.linear.pressure_iters; }
  if (mname) { S.monName = mname; S.mon.push(mval); $("#sExtraK").textContent = mname.split(" ")[0]; $("#sExtra").textContent = fmt(mval, 4); $("#monName").textContent = mname; }
  $("#resInfo").textContent = Object.entries(ev.scaled).map(([k, v]) => `${k[0]}:${v.toExponential(1)}`).join(" ");
  if (ev.timers) drawTimers(ev.timers);
  scheduleCharts();
}
let chartPending = false;
function scheduleCharts() { if (chartPending) return; chartPending = true; requestAnimationFrame(() => { chartPending = false; drawResiduals(); drawMonitor(); }); }
function onFrame(ev) {
  S.frame = ev;
  if (ev.mesh === "structured") ev.values = b64f32(ev.data), ev.u = b64f32(ev.u), ev.v = b64f32(ev.v);
  else { ev.values = b64f32(ev.data); if (ev.u && ev.v) { ev.u = b64f32(ev.u); ev.v = b64f32(ev.v); } }
  const sel = $("#fieldSelect");
  if (sel.options.length !== ev.fields.length || Array.from(sel.options).some((o, i) => o.value !== ev.fields[i])) {
    sel.innerHTML = ev.fields.map((f) => `<option value="${f}">${f}</option>`).join("");
  }
  sel.value = ev.field;
  $("#empty").classList.add("hidden");
  $("#frameInfo").textContent = ev.mesh === "structured" ? `${ev.grid[0]}×${ev.grid[1]} · step ${ev.step}` : `${ev.cells.toLocaleString()} cells · it ${ev.step}`;
  draw();
}
function onResult(res) {
  const v = res.validation;
  if (v && v.checks) {
    $("#validation").innerHTML = v.checks.map((c) => `<div class="check"><span class="badge ${c.pass ? "pass" : "check"}">${c.pass ? "PASS" : "CHECK"}</span><span>${esc(c.name)}${c.reference !== undefined && c.reference !== null ? ` <span class="muted">(ref ${c.reference})</span>` : ""}</span><span class="val">${fmt(c.value, 4)} <span class="muted">${esc(c.unit || "")}</span></span></div>`).join("");
  } else if (res.mass_balance) {
    const mb = res.mass_balance; const net = Object.values(mb).reduce((a, b) => a + b, 0);
    $("#validation").innerHTML = `<div class="check"><span class="badge ${Math.abs(net) < 1e-6 * Math.max(1e-30, ...Object.values(mb).map(Math.abs)) + 1e-9 ? "pass" : "check"}">MASS</span><span>Net boundary mass flow</span><span class="val">${net.toExponential(2)}</span></div>` +
      Object.entries(mb).map(([k, v]) => `<div class="check"><span class="muted">${esc(k)}</span><span class="val">${fmt(v, 5)}</span></div>`).join("") +
      `<div class="check"><span class="badge ${res.converged ? "pass" : "check"}">${res.converged ? "CONV" : "ITER"}</span><span>${res.iterations} iterations</span></div>`;
  }
  if (res.continuity_local_max !== undefined) {
    $("#validation").insertAdjacentHTML("beforeend", `<div class="check"><span class="muted">Local continuity (max cell imbalance / mean face flux)</span><span class="val">${Number(res.continuity_local_max).toExponential(2)}</span></div>`);
  }
  const cert = res.certificate;
  if (cert && cert.quantities) {
    const weak = (q) => q.status === "oscillating" || q.status === "not converging" || q.status === "slow" || q.status === "irregular";
    const rows = Object.entries(cert.quantities).map(([n, q]) => `<tr><td>${esc(n)}</td><td class="num">${fmt(q.value, 5)}</td><td class="num">± ${fmt(q.band, 2)}</td>` +
      `<td class="num">${(100 * q.relative).toPrecision(2)} %</td><td><span class="badge ${weak(q) ? "check" : "pass"}">${esc(q.status)}</span>${q.order !== null && q.order !== undefined ? ` <span class="muted">order ${fmt(q.order, 3)}</span>` : ""}</td></tr>`).join("");
    const fe = cert.velocity_field_error;
    $("#validation").insertAdjacentHTML("beforeend", `<div style="margin-top:8px"><b>${esc(t("certTitle"))}</b> <span class="muted">${cert.cells.map((c) => c.toLocaleString()).join(" → ")} cells</span></div>` +
      `<table class="grid"><tr><th>Result (finest mesh)</th><th>Value</th><th>Error band</th><th>%</th><th>Mesh convergence</th></tr>${rows}</table>` +
      `<div class="desc">Velocity field of the coarser mesh: mean error ${(100 * fe.mean).toPrecision(2)} %, worst cell ${(100 * fe.max).toPrecision(2)} % of the largest speed. The field “mesh-error-estimate” shows where.</div>` +
      cert.notes.map((n) => `<div class="desc">${esc(n)}</div>`).join(""));
  }
  const wr = res.wall_report;
  if (wr && Object.keys(wr).length) {
    const rows = Object.entries(wr).map(([z, r]) => `<tr><td>${esc(z)}</td><td class="num">${fmt(r.force_total[0], 4)}</td><td class="num">${fmt(r.force_total[1], 4)}</td>` +
      `<td class="num">${fmt(r.cd, 4)}</td><td class="num">${fmt(r.cf_avg, 4)}</td><td class="num">${fmt(r.yplus_avg, 3)} / ${fmt(r.yplus_max, 3)}</td>` +
      `<td class="num">${r.heat_flow_conv !== undefined ? fmt(r.heat_flow_conv, 4) : "–"}${r.heat_flow_rad !== undefined ? " + " + fmt(r.heat_flow_rad, 4) + " rad" : ""}</td></tr>`).join("");
    $("#validation").insertAdjacentHTML("beforeend", `<div style="margin-top:8px"><b>Wall report</b></div><table class="grid"><tr><th>Wall</th><th>Fx</th><th>Fy</th><th>Cd</th><th>Cf avg</th><th>y+ avg/max</th><th>Q</th></tr>${rows}</table>`);
  }
  const pm = res.periodic_mass_flow;
  if (pm && Object.keys(pm).length) {
    $("#validation").insertAdjacentHTML("beforeend", Object.entries(pm).map(([k, v]) => `<div class="check"><span class="muted">periodic ${esc(k)}</span><span class="val">${fmt(v, 6)}</span></div>`).join(""));
  }
  if (res.timers) drawTimers(res.timers);
}
function logLine(ev) {
  const c = $("#console");
  const div = document.createElement("div");
  const m = ev.msg || "";
  div.className = ev.level === "error" ? "error" : ev.level === "warn" ? "warn" : m.startsWith("[plan]") ? "plan" : m.startsWith("[validation]") ? "val" : "";
  div.textContent = m;
  c.appendChild(div);
  while (c.childNodes.length > 400) c.removeChild(c.firstChild);
  c.scrollTop = c.scrollHeight;
}
const TIMER_COLORS = ["#e07b53", "#7aa6c2", "#7fb685", "#e3b45a", "#c9788a", "#8f8bb5", "#b9b29a"];
function drawTimers(t) {
  S.timers = t;
  const entries = Object.entries(t).filter(([, v]) => v > 0);
  const tot = entries.reduce((a, [, v]) => a + v, 0) || 1;
  $("#timerBar").innerHTML = entries.map(([k, v], i) => `<div title="${k} ${(100 * v / tot).toFixed(1)}%" style="width:${100 * v / tot}%;background:${TIMER_COLORS[i % 7]}"></div>`).join("");
  $("#timerLegend").innerHTML = entries.map(([k, v], i) => `<span><i style="background:${TIMER_COLORS[i % 7]}"></i>${esc(k)} ${(100 * v / tot).toFixed(0)}%</span>`).join("");
}

/* ------------------------------------------------------------------ charts */
function setupCanvas(cv) {
  const r = cv.getBoundingClientRect(); const dpr = window.devicePixelRatio || 1;
  const w = Math.max(10, Math.round(r.width * dpr)), h = Math.max(10, Math.round(r.height * dpr));
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
  const ctx = cv.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return [ctx, r.width, r.height];
}
function drawResiduals() {
  const cv = $("#resChart"); const [ctx, W, H] = setupCanvas(cv);
  ctx.clearRect(0, 0, W, H);
  const R = S.res, n = S.resCount;
  const pad = { l: 40, r: 8, t: 8, b: 18 };
  let lo = Infinity, hi = -Infinity;
  for (const k of R.keys) for (const v of R.series[k]) if (v > 0) { const l = Math.log10(v); lo = Math.min(lo, l); hi = Math.max(hi, l); }
  if (!isFinite(lo)) { lo = -6; hi = 0; }
  lo = Math.floor(Math.max(lo, -16)); hi = Math.ceil(Math.min(Math.max(hi, lo + 1), 4));
  const X = (i) => pad.l + (W - pad.l - pad.r) * (n > 1 ? i / (n - 1) : 0);
  const Y = (l) => pad.t + (H - pad.t - pad.b) * (hi - l) / (hi - lo);
  ctx.font = "10px ui-monospace, monospace"; ctx.fillStyle = cssVar("--muted"); ctx.strokeStyle = cssVar("--chart-grid"); ctx.lineWidth = 1;
  for (let e = lo; e <= hi; e++) { const y = Y(e); ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke(); ctx.fillText(`1e${e}`, 4, y + 3); }
  ctx.strokeStyle = cssVar("--line"); ctx.setLineDash([3, 3]);
  for (const lab of R.labels) { if (lab.at > 1) { const x = X(lab.at - 1); ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, H - pad.b); ctx.stroke(); ctx.fillText(lab.label, x + 3, H - 5); } }
  ctx.setLineDash([]);
  for (const k of R.keys) {
    ctx.strokeStyle = resColor(k); ctx.lineWidth = 1.4; ctx.beginPath();
    let started = false;
    R.series[k].forEach((v, i) => { if (!(v > 0)) return; const x = X(i), y = Y(Math.log10(v)); if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y); });
    ctx.stroke();
  }
  $("#resLegend").innerHTML = R.keys.map((k) => `<span><i style="background:${resColor(k)}"></i>${esc(k)}</span>`).join("");
}
function drawMonitor() {
  const cv = $("#monChart"); const [ctx, W, H] = setupCanvas(cv);
  ctx.clearRect(0, 0, W, H);
  const d = S.mon.filter((v) => v !== null && v !== undefined && isFinite(v));
  if (d.length < 2) return;
  let lo = Math.min(...d), hi = Math.max(...d); if (hi === lo) { hi += 1e-12; lo -= 1e-12; }
  ctx.strokeStyle = cssVar("--accent"); ctx.lineWidth = 1.5; ctx.beginPath();
  d.forEach((v, i) => { const x = 4 + (W - 8) * i / (d.length - 1), y = 6 + (H - 12) * (hi - v) / (hi - lo); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  ctx.stroke();
  ctx.fillStyle = cssVar("--muted"); ctx.font = "10px ui-monospace, monospace";
  ctx.fillText(fmt(hi, 4), 6, 12); ctx.fillText(fmt(lo, 4), 6, H - 4);
}

/* ------------------------------------------------------------------ viewport */
function parseGeometry(g) {
  if (!g || !g.nodes) return null;
  return { nodes: b64f32(g.nodes), ptr: b64i32(g.ptr), loops: b64i32(g.loops), bbox: g.bbox, cells: g.cells,
    zones: g.zones || [], segs: g.segs ? b64i32(g.segs) : null };
}
function viewRect(W, H, lx, ly) {
  const m = { l: 24, r: 96, t: 24, b: 24 };
  const aw = W - m.l - m.r, ah = H - m.t - m.b;
  const s = Math.min(aw / lx, ah / ly);
  const w = lx * s, h = ly * s;
  return { x: m.l + (aw - w) / 2, y: m.t + (ah - h) / 2, w, h, s };
}
const off = document.createElement("canvas");
function draw() {
  const cv = $("#view"); const [ctx, W, H] = setupCanvas(cv);
  ctx.clearRect(0, 0, W, H);
  const f = S.frame;
  const cmap = lut($("#cmapSelect").value);
  if (f && f.mesh === "structured") drawStructured(ctx, W, H, f, cmap);
  else if (S.geometry) drawUnstructured(ctx, W, H, f, cmap);
  else if (S.geom3) drawMesh3D(ctx, W, H, S.geom3);
  drawColorbar(f, cmap);
  if (f) $("#view").setAttribute("aria-label", `${t("fieldImage")} ${f.field}: ${fmt(f.min, 4)} … ${fmt(f.max, 4)}`);
}
function drawStructured(ctx, W, H, f, cmap) {
  const { nx, ny, lx, ly } = f;
  const vr = viewRect(W, H, lx, ly); S.view = { ...vr, lx, ly, mesh: "structured" };
  off.width = nx; off.height = ny;
  const octx = off.getContext("2d"); const img = octx.createImageData(nx, ny);
  const lo = f.min, hi = f.max, inv = hi > lo ? 255 / (hi - lo) : 0;
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const v = f.values[j * nx + i]; const p = 4 * ((ny - 1 - j) * nx + i);
    if (Number.isNaN(v)) { img.data[p] = 40; img.data[p + 1] = 46; img.data[p + 2] = 56; img.data[p + 3] = 255; continue; }
    const k = 3 * Math.max(0, Math.min(255, Math.round((v - lo) * inv)));
    img.data[p] = cmap[k]; img.data[p + 1] = cmap[k + 1]; img.data[p + 2] = cmap[k + 2]; img.data[p + 3] = 255;
  }
  octx.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = $("#smooth").checked;
  ctx.drawImage(off, vr.x, vr.y, vr.w, vr.h);
  ctx.strokeStyle = "#3a4b62"; ctx.lineWidth = 1; ctx.strokeRect(vr.x, vr.y, vr.w, vr.h);
  if ($("#showStream").checked) drawStreamlines(ctx, f, vr);
  if ($("#showVectors").checked) drawVectors(ctx, f, vr);
}
function sampleVel(f, gx, gy) {
  const nx = f.vnx, ny = f.vny;
  const x = gx - 0.5, y = gy - 0.5;
  let i = Math.floor(x), j = Math.floor(y);
  if (i < -1 || j < -1 || i > nx - 1 || j > ny - 1) return null;
  i = Math.max(0, Math.min(nx - 2, i)); j = Math.max(0, Math.min(ny - 2, j));
  const tx = Math.max(0, Math.min(1, x - i)), ty = Math.max(0, Math.min(1, y - j));
  const id = (a, b) => b * nx + a;
  const u00 = f.u[id(i, j)], u10 = f.u[id(i + 1, j)], u01 = f.u[id(i, j + 1)], u11 = f.u[id(i + 1, j + 1)];
  const v00 = f.v[id(i, j)], v10 = f.v[id(i + 1, j)], v01 = f.v[id(i, j + 1)], v11 = f.v[id(i + 1, j + 1)];
  if ([u00, u10, u01, u11].some(Number.isNaN)) return null;
  const u = (u00 * (1 - tx) + u10 * tx) * (1 - ty) + (u01 * (1 - tx) + u11 * tx) * ty;
  const v = (v00 * (1 - tx) + v10 * tx) * (1 - ty) + (v01 * (1 - tx) + v11 * tx) * ty;
  return [u, v];
}
function drawStreamlines(ctx, f, vr) {
  const nx = f.vnx, ny = f.vny;
  const sx = vr.w / nx, sy = vr.h / ny; // px per velocity cell
  const occN = 2, occ = new Uint8Array(nx * occN * ny * occN);
  const ar = (f.lx / nx) / (f.ly / ny); // physical cell aspect
  ctx.strokeStyle = "rgba(255,255,255,0.55)"; ctx.lineWidth = 0.9;
  const seeds = []; const ns = 26;
  for (let a = 0; a < ns; a++) for (let b = 0; b < ns; b++) seeds.push([(a + 0.5) / ns * nx, (b + 0.5) / ns * ny]);
  for (const [sx0, sy0] of seeds) {
    const ox = Math.floor(sx0 * occN), oy = Math.floor(sy0 * occN);
    if (occ[oy * nx * occN + ox]) continue;
    for (const dir of [1, -1]) {
      let x = sx0, y = sy0; const pts = [[x, y]]; const mine = [];
      for (let k = 0; k < 400; k++) {
        const a = sampleVel(f, x, y); if (!a) break;
        let [u, v] = a; u /= ar; const m = Math.hypot(u, v); if (m < 1e-12) break;
        const h = 0.35 * dir;
        const xm = x + 0.5 * h * u / m, ym = y + 0.5 * h * v / m;
        const b = sampleVel(f, xm, ym); if (!b) break;
        let [u2, v2] = b; u2 /= ar; const m2 = Math.hypot(u2, v2); if (m2 < 1e-12) break;
        x += h * u2 / m2; y += h * v2 / m2;
        if (x < 0 || y < 0 || x > nx || y > ny) break;
        const cx = Math.floor(x * occN), cy = Math.floor(y * occN), ci = cy * nx * occN + cx;
        if (occ[ci] && !mine.includes(ci)) break;
        if (!mine.includes(ci)) { mine.push(ci); if (mine.length > 6) occ[mine[mine.length - 7]] = 1; }
        pts.push([x, y]);
      }
      mine.forEach((ci) => { occ[ci] = 1; });
      if (pts.length < 4) continue;
      ctx.beginPath();
      pts.forEach(([px, py], i) => { const X = vr.x + px * sx, Y = vr.y + vr.h - py * sy; i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); });
      ctx.stroke();
    }
  }
}
function drawVectors(ctx, f, vr) {
  const nx = f.vnx, ny = f.vny; const step = Math.max(1, Math.round(Math.max(nx, ny) / 28));
  let umax = 1e-30; for (let k = 0; k < f.u.length; k++) { const m = Math.hypot(f.u[k], f.v[k]); if (m > umax) umax = m; }
  const L = step * Math.min(vr.w / nx, vr.h / ny) * 0.9;
  ctx.strokeStyle = "rgba(255,255,255,0.85)"; ctx.fillStyle = "rgba(255,255,255,0.85)"; ctx.lineWidth = 1;
  for (let j = Math.floor(step / 2); j < ny; j += step) for (let i = Math.floor(step / 2); i < nx; i += step) {
    const u = f.u[j * nx + i], v = f.v[j * nx + i]; if (Number.isNaN(u)) continue;
    const m = Math.hypot(u, v) / umax; if (m < 1e-3) continue;
    const X = vr.x + (i + 0.5) * vr.w / nx, Y = vr.y + vr.h - (j + 0.5) * vr.h / ny;
    const dx = u / umax * L, dy = -v / umax * L; const ex = X + dx, ey = Y + dy;
    ctx.beginPath(); ctx.moveTo(X, Y); ctx.lineTo(ex, ey); ctx.stroke();
    const a = Math.atan2(dy, dx), hl = Math.max(2.5, 0.3 * Math.hypot(dx, dy));
    ctx.beginPath(); ctx.moveTo(ex, ey); ctx.lineTo(ex - hl * Math.cos(a - 0.45), ey - hl * Math.sin(a - 0.45)); ctx.lineTo(ex - hl * Math.cos(a + 0.45), ey - hl * Math.sin(a + 0.45)); ctx.closePath(); ctx.fill();
  }
}
function drawUnstructured(ctx, W, H, f, cmap) {
  const g = S.geometry; const [x0, y0, x1, y1] = g.bbox;
  const vr = viewRect(W, H, x1 - x0, y1 - y0); S.view = { ...vr, mesh: "unstructured", x0, y0 };
  const key = `${W}x${H}`;
  if (!S.cellPaths || S.cellPaths.key !== key) {
    const paths = new Array(g.cells), cent = new Float32Array(2 * g.cells);
    for (let c = 0; c < g.cells; c++) {
      const p = new Path2D(); let cx = 0, cy = 0; const s = g.ptr[c], e = g.ptr[c + 1];
      for (let k = s; k < e; k++) {
        const n = g.loops[k]; const X = vr.x + (g.nodes[2 * n] - x0) * vr.s, Y = vr.y + vr.h - (g.nodes[2 * n + 1] - y0) * vr.s;
        k === s ? p.moveTo(X, Y) : p.lineTo(X, Y); cx += g.nodes[2 * n]; cy += g.nodes[2 * n + 1];
      }
      p.closePath(); paths[c] = p; cent[2 * c] = cx / (e - s); cent[2 * c + 1] = cy / (e - s);
    }
    S.cellPaths = { key, paths }; S.cellCentres = cent;
  }
  const edges = $("#showEdges").checked;
  const lo = f ? f.min : 0, hi = f ? f.max : 1, inv = hi > lo ? 255 / (hi - lo) : 0;
  ctx.lineWidth = 0.5;
  if (f && f.values.length !== g.cells) f = null; // a frame of another refinement level, before its geometry arrived
  if (f && $("#smooth").checked) {
    // node-interpolated shading: every cell as a fan of triangles (centroid, corner, corner)
    const dpr = window.devicePixelRatio || 1, q = Math.min(dpr, Math.sqrt(2.5e6 / Math.max(1, W * H)));
    const rw = Math.max(1, Math.round(W * q)), rh = Math.max(1, Math.round(H * q));
    const [img] = rasterCells(g, [f.values], rw, rh, (x) => (vr.x + (x - x0) * vr.s) * rw / W, (y) => (vr.y + vr.h - (y - y0) * vr.s) * rh / H);
    off.width = rw; off.height = rh;
    const octx = off.getContext("2d"); const id = octx.createImageData(rw, rh); const d = id.data;
    for (let i = 0; i < rw * rh; i++) {
      const v = img[i]; if (v !== v) continue;
      const k = 3 * Math.max(0, Math.min(255, Math.round((v - lo) * inv)));
      d[4 * i] = cmap[k]; d[4 * i + 1] = cmap[k + 1]; d[4 * i + 2] = cmap[k + 2]; d[4 * i + 3] = 255;
    }
    octx.putImageData(id, 0, 0);
    ctx.imageSmoothingEnabled = true; ctx.drawImage(off, 0, 0, W, H);
    if (edges) { ctx.strokeStyle = "rgba(0,0,0,0.35)"; for (let c = 0; c < g.cells; c++) ctx.stroke(S.cellPaths.paths[c]); }
  } else for (let c = 0; c < g.cells; c++) {
    if (f) { const k = 3 * Math.max(0, Math.min(255, Math.round((f.values[c] - lo) * inv))); ctx.fillStyle = `rgb(${cmap[k]},${cmap[k + 1]},${cmap[k + 2]})`; }
    else ctx.fillStyle = cssVar("--panel-2");
    ctx.fill(S.cellPaths.paths[c]);
    if (edges || !f) { ctx.strokeStyle = f ? "rgba(0,0,0,0.35)" : cssVar("--muted"); ctx.stroke(S.cellPaths.paths[c]); }
  }
  if (!f) $("#empty").classList.add("hidden");
  const wantS = $("#showStream").checked, wantV = $("#showVectors").checked;
  if (f && f.u && f.u.length === g.cells && (wantS || wantV)) {
    // velocity on a regular grid over the bounding box (NaN outside the domain), then as for structured cases
    const lx = x1 - x0, ly = y1 - y0, m = 150 / Math.max(lx, ly);
    const vnx = Math.max(8, Math.round(lx * m)), vny = Math.max(8, Math.round(ly * m));
    const [u, v] = rasterCells(g, [f.u, f.v], vnx, vny, (x) => (x - x0) / lx * vnx, (y) => (y - y0) / ly * vny);
    const vf = { vnx, vny, lx, ly, u, v };
    if (wantS) drawStreamlines(ctx, vf, vr);
    if (wantV) drawVectors(ctx, vf, vr);
  }
  drawZoneHighlight2D(ctx, g, vr, x0, y0);
}
/* Cell values -> node values (mean of the cells around a node) -> linear interpolation over each cell's
   triangle fan, sampled at the centres of a W x H grid. X, Y map mesh coordinates to grid coordinates.
   Returns one Float32Array per channel, NaN outside the mesh. */
function rasterCells(g, chans, W, H, X, Y) {
  const nn = g.nodes.length / 2, nch = chans.length;
  const cnt = new Float32Array(nn); const nodeV = chans.map(() => new Float32Array(nn));
  for (let c = 0; c < g.cells; c++) for (let k = g.ptr[c]; k < g.ptr[c + 1]; k++) { const n = g.loops[k]; cnt[n]++; for (let q = 0; q < nch; q++) nodeV[q][n] += chans[q][c]; }
  for (let n = 0; n < nn; n++) if (cnt[n]) for (let q = 0; q < nch; q++) nodeV[q][n] /= cnt[n];
  const px = new Float32Array(nn), py = new Float32Array(nn);
  for (let n = 0; n < nn; n++) { px[n] = X(g.nodes[2 * n]); py[n] = Y(g.nodes[2 * n + 1]); }
  const out = chans.map(() => new Float32Array(W * H).fill(NaN));
  const eps = -1e-3;
  for (let c = 0; c < g.cells; c++) {
    const s = g.ptr[c], e = g.ptr[c + 1]; let cx = 0, cy = 0;
    for (let k = s; k < e; k++) { cx += px[g.loops[k]]; cy += py[g.loops[k]]; }
    cx /= e - s; cy /= e - s;
    for (let k = s; k < e; k++) {
      const a = g.loops[k], b = g.loops[k + 1 < e ? k + 1 : s];
      const ax = px[a], ay = py[a], bx = px[b], by = py[b];
      const den = (ay - by) * (cx - bx) + (bx - ax) * (cy - by); if (Math.abs(den) < 1e-12) continue;
      const i0 = Math.max(0, Math.floor(Math.min(cx, ax, bx) - 0.5)), i1 = Math.min(W - 1, Math.ceil(Math.max(cx, ax, bx) - 0.5));
      const j0 = Math.max(0, Math.floor(Math.min(cy, ay, by) - 0.5)), j1 = Math.min(H - 1, Math.ceil(Math.max(cy, ay, by) - 0.5));
      for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) {
        const x = i + 0.5, y = j + 0.5;
        const w0 = ((ay - by) * (x - bx) + (bx - ax) * (y - by)) / den; if (w0 < eps) continue;
        const w1 = ((by - cy) * (x - bx) + (cx - bx) * (y - by)) / den; if (w1 < eps) continue;
        const w2 = 1 - w0 - w1; if (w2 < eps) continue;
        for (let q = 0; q < nch; q++) out[q][j * W + i] = w0 * chans[q][c] + w1 * nodeV[q][a] + w2 * nodeV[q][b];
      }
    }
  }
  return out;
}
function drawColorbar(f, cmap) {
  const cv = $("#colorbar"); const [ctx, W, H] = setupCanvas(cv);
  ctx.clearRect(0, 0, W, H);
  if (!f) return;
  const bw = 14, x = 4;
  for (let y = 0; y < H - 20; y++) { const k = 3 * Math.round(255 * (1 - y / (H - 21))); ctx.fillStyle = `rgb(${cmap[k]},${cmap[k + 1]},${cmap[k + 2]})`; ctx.fillRect(x, 10 + y, bw, 1); }
  ctx.strokeStyle = cssVar("--line"); ctx.strokeRect(x, 10, bw, H - 20);
  ctx.fillStyle = cssVar("--text"); ctx.font = "10.5px ui-monospace, monospace";
  for (let i = 0; i <= 4; i++) { const v = f.max - (f.max - f.min) * i / 4; ctx.fillText(fmt(v, 3), x + bw + 4, 14 + (H - 24) * i / 4); }
}
function initHover() {
  const vp = $("#viewport"), hv = $("#hover");
  vp.addEventListener("mousemove", (e) => {
    const f = S.frame, v = S.view; if (!f || !v) { hv.style.display = "none"; return; }
    const r = vp.getBoundingClientRect(); const px = e.clientX - r.left, py = e.clientY - r.top;
    if (px < v.x || py < v.y || px > v.x + v.w || py > v.y + v.h) { hv.style.display = "none"; return; }
    let txt;
    if (v.mesh === "structured") {
      const fx = (px - v.x) / v.w, fy = 1 - (py - v.y) / v.h;
      const i = Math.min(f.nx - 1, Math.floor(fx * f.nx)), j = Math.min(f.ny - 1, Math.floor(fy * f.ny));
      txt = `x=${fmt(fx * f.lx, 4)} y=${fmt(fy * f.ly, 4)}  ${f.field}=${fmt(f.values[j * f.nx + i], 5)}`;
    } else if (S.cellCentres) {
      const X = v.x0 + (px - v.x) / v.s, Y = v.y0 + (v.y + v.h - py) / v.s; const c = S.cellCentres; let best = 0, bd = Infinity;
      for (let k = 0; k < c.length / 2; k++) { const d = (c[2 * k] - X) ** 2 + (c[2 * k + 1] - Y) ** 2; if (d < bd) { bd = d; best = k; } }
      txt = `x=${fmt(X, 4)} y=${fmt(Y, 4)}  cell ${best}  ${f.field}=${fmt(f.values[best], 5)}`;
    }
    hv.textContent = txt; hv.style.display = "block";
  });
  vp.addEventListener("mouseleave", () => { hv.style.display = "none"; });
}

/* ------------------------------------------------------------------ projects & report */
function uiState() {
  const params = {};
  $$("#presetParams input").forEach((i) => { params[i.dataset.param] = i.value; });
  const vals = {};
  ["tEnd", "maxSteps", "steadyTol", "mode", "scheme", "cfl", "poisson", "algorithm", "uScheme", "relaxU", "relaxP", "maxIter", "uTol",
    "backend", "precision", "ckpt", "uDensity", "uViscosity", "uConductivity", "uCp", "uTurb"].forEach((id) => { vals[id] = $("#" + id).value; });
  ["fmg", "uEnergy"].forEach((id) => { vals[id] = $("#" + id).checked; });
  return { source: S.source, preset: $("#presetSelect").value, params, vals, zones: S.source === "unstructured" ? zoneSpecs() : null };
}
async function refreshProjects() {
  const { projects } = await api("/api/projects");
  $("#projectList").innerHTML = projects.length ? projects.map((p) => `<tr><td><b>${esc(p.name)}</b><div class="muted">${esc(p.title || "")} · ${esc(p.kind || "")} · ${esc(t("saved"))} ${esc(p.saved || "")}${p.has_result ? " · ✓" : ""}</div></td>
      <td style="text-align:end"><button class="btn small" type="button" data-open="${esc(p.name)}">${esc(t("open"))}</button></td></tr>`).join("")
    : `<tr><td class="muted">${esc(t("noProjects"))}</td></tr>`;
  $$("[data-open]").forEach((b) => b.addEventListener("click", () => openProject(b.dataset.open)));
}
async function saveProject() {
  const name = $("#projectName").value.trim();
  if (!name) { $("#projectMsg").textContent = t("needName"); return; }
  try {
    let spec = null;
    try { spec = buildSpec(); } catch (e) { spec = { kind: S.source }; }
    const r = await post("/api/projects", { name, spec, ui: uiState(), title: $("#caseTitle").textContent });
    $("#projectMsg").textContent = `${t("projectSaved")} ${r.name}`;
    refreshProjects();
  } catch (e) { $("#projectMsg").textContent = e.message; }
}
async function openProject(name) {
  try {
    const r = await api(`/api/projects/${encodeURIComponent(name)}`);
    const ui = r.project.ui || {};
    setSource(ui.source || "structured");
    if (ui.preset) { $("#presetSelect").value = ui.preset; renderPresetParams(); }
    for (const [k, v] of Object.entries(ui.params || {})) { const el = $(`#presetParams input[data-param="${k}"]`); if (el) el.value = v; }
    for (const [k, v] of Object.entries(ui.vals || {})) { const el = $("#" + k); if (!el) continue; if (el.type === "checkbox") el.checked = !!v; else el.value = v; }
    if (r.mesh) {
      await renderMeshInfo(r.mesh);
      for (const [zone, spec] of Object.entries(ui.zones || {})) {
        const tr = $(`#zoneTable tr[data-zone="${CSS.escape(zone)}"]`); if (!tr) continue;
        $("select[data-k=kind]", tr).value = spec.kind;
        const val = spec.normal_speed ?? (spec.velocity ? spec.velocity.join(",") : (spec.pressure ?? ""));
        $("input[data-k=val]", tr).value = val;
        if (spec.thermal && spec.thermal !== "adiabatic") $("input[data-k=therm]", tr).value = `${spec.thermal === "flux" ? "q" : "T"}=${spec.value}`;
      }
    }
    $("#projectMsg").textContent = `${t("projectOpened")} ${r.project.name}`;
    $("#caseTitle").textContent = r.project.title || r.project.name;
    $("#projectsDialog").close();
  } catch (e) { $("#projectMsg").textContent = e.message; }
}
async function exportReport() {
  try {
    let image = null;
    try { image = $("#view").toDataURL("image/png"); } catch (e) { image = null; }
    const r = await fetch("/api/report", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ image }) });
    if (!r.ok) { const j = await r.json().catch(() => ({})); throw new Error(j.error || r.statusText); }
    const blob = await r.blob(); const a = document.createElement("a");
    const m = (r.headers.get("Content-Disposition") || "").match(/filename="([^"]+)"/);
    a.href = URL.createObjectURL(blob); a.download = m ? m[1] : "flowtrace_report.html"; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  } catch (e) { logLine({ level: "error", msg: `${t("reportFailed")} ${e.message}` }); }
}

/* ------------------------------------------------------------------ boot */
async function restoreState() {
  const st = await api("/api/state");
  for (const ev of st.log) logLine(ev);
  for (const ev of st.residuals) onResiduals(ev);
  if (st.job) {
    onStatus({ status: st.job.status, result: st.job.result });
    if (st.job.plan) onPlan({ plan: st.job.plan });
  }
}
function init() {
  applyI18n();
  initTabs(); initMeshUpload(); initHover(); bindMesh3DControls(); initZoneHighlight(); initBigCharts();
  $("#themeBtn").addEventListener("click", cycleTheme);
  $("#projectsBtn").addEventListener("click", () => { $("#projectMsg").textContent = ""; refreshProjects().catch((e) => { $("#projectMsg").textContent = e.message; }); $("#projectsDialog").showModal(); });
  $("#saveProjectBtn").addEventListener("click", saveProject);
  $("#reportBtn").addEventListener("click", exportReport);
  try { window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redrawAll); } catch (e) { /* old browsers */ }
  $("#runBtn").addEventListener("click", run);
  $("#stopBtn").addEventListener("click", () => post("/api/stop"));
  $("#mode").addEventListener("change", () => setSource(S.source));
  $("#presetSelect").addEventListener("change", updateSolverCards);
  $("#fieldSelect").addEventListener("change", (e) => post("/api/view", { field: e.target.value }));
  ["#cmapSelect", "#showStream", "#showVectors", "#showEdges", "#smooth"].forEach((s) => $(s).addEventListener("change", draw));
  $("#benchBtn").addEventListener("click", async () => {
    $("#benchBtn").disabled = true; $("#benchBox").textContent = t("benchRunning");
    try { renderBench(await post("/api/benchmark")); } catch (e) { $("#benchBox").textContent = e.message; }
    $("#benchBtn").disabled = false;
  });
  $("#estimateBtn").addEventListener("click", async () => {
    try {
      const spec = buildSpec();
      const r = await post("/api/estimate", spec.kind === "structured" ? spec : { kind: "unstructured", mesh_id: spec.mesh_id, energy: spec.case.energy });
      $("#estimateOut").innerHTML = Object.entries(r).filter(([k]) => k !== "cells").map(([k, v]) => `<b>${k}</b> (${v.device}): ${fmtBytes(v.required)} of ${fmtBytes(v.available)} ${v.fits ? '<span style="color:var(--ok)">fits</span>' : '<span style="color:var(--err)">does not fit</span>'}`).join("<br>") + `<br>${(r.cells || 0).toLocaleString()} cells`;
    } catch (e) { $("#estimateOut").textContent = e.message; }
  });
  $$("[data-export]").forEach((b) => b.addEventListener("click", async () => {
    const r = await fetch(`/api/export/${b.dataset.export}`);
    if (!r.ok) { const j = await r.json().catch(() => ({})); logLine({ level: "warn", msg: `export: ${j.error || r.statusText}` }); return; }
    const blob = await r.blob(); const a = document.createElement("a");
    const cd = r.headers.get("Content-Disposition") || ""; const m = cd.match(/filename="([^"]+)"/);
    a.href = URL.createObjectURL(blob); a.download = m ? m[1] : `flowtrace.${b.dataset.export}`; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  }));
  window.addEventListener("resize", () => { S.cellPaths = null; draw(); drawResiduals(); drawMonitor(); });
  setSource("structured");
  Promise.all([loadSystem(), loadPresets()]).then(restoreState).catch((e) => logLine({ level: "error", msg: e.message }));
  connectEvents();
  draw(); drawResiduals();
}
document.addEventListener("DOMContentLoaded", init);


/* ------------------------------------------------------------------ loading screen */
let loadShown = 0;
function showLoading(frac, label) {
  const el = $("#loading"); el.classList.remove("hidden");
  const pct = Math.max(loadShown, Math.round(100 * frac)); // never backwards
  loadShown = pct;
  $("#loadFill").style.width = pct + "%"; $("#loadPct").textContent = pct + "%"; $("#loadBar").setAttribute("aria-valuenow", String(pct));
  $("#loadLabel").textContent = label || "";
}
function hideLoading() { loadShown = 0; $("#loading").classList.add("hidden"); }
function onProgress(ev) { if (!$("#loading").classList.contains("hidden") || ev.fraction < 1) showLoading(ev.fraction, ev.label); }

/* ------------------------------------------------------------------ zone highlight on the picture */
function initZoneHighlight() {
  const tbl = $("#zoneTable"); if (!tbl) return;
  const set = (name) => { if (S.hiZone !== name) { S.hiZone = name; draw(); } };
  tbl.addEventListener("mouseover", (e) => { const tr = e.target.closest("tr[data-zone]"); if (tr) set(tr.dataset.zone); });
  tbl.addEventListener("mouseleave", () => set(null));
  tbl.addEventListener("focusin", (e) => { const tr = e.target.closest("tr[data-zone]"); if (tr) set(tr.dataset.zone); });
  tbl.addEventListener("focusout", () => set(null));
  tbl.addEventListener("change", () => { if (S.hiZone) draw(); });
}

/* ------------------------------------------------------------------ full-screen charts */
const fmtE = (v) => (v === null || v === undefined || !isFinite(v) ? "–" : (Math.abs(v) >= 1e-3 && Math.abs(v) < 1e5 ? (+v.toPrecision(5)).toString() : v.toExponential(3)));
function lineChart(cv, o) {
  // o: {series:[{name,color,data}], logY, range:[i0,i1], hover, yLabel}
  const [ctx, W, H] = setupCanvas(cv);
  ctx.clearRect(0, 0, W, H);
  const pad = { l: 70, r: 20, t: 18, b: 40 };
  const n = Math.max(...o.series.map((s) => s.data.length), 1);
  const [i0, i1] = o.range || [0, n - 1];
  const tv = (v) => (o.logY ? Math.log10(v) : v), ok = (v) => v !== null && v !== undefined && isFinite(v) && (!o.logY || v > 0);
  let lo = Infinity, hi = -Infinity;
  for (const s of o.series) for (let i = Math.max(0, Math.floor(i0)); i <= Math.min(s.data.length - 1, Math.ceil(i1)); i++) if (ok(s.data[i])) { const v = tv(s.data[i]); lo = Math.min(lo, v); hi = Math.max(hi, v); }
  if (!isFinite(lo)) { lo = o.logY ? -6 : 0; hi = o.logY ? 0 : 1; }
  if (o.logY) { lo = Math.floor(Math.max(lo, -16)); hi = Math.ceil(Math.max(hi, lo + 1)); } else { const d = (hi - lo) || Math.abs(hi) * 0.1 || 1; lo -= d * 0.08; hi += d * 0.08; }
  const X = (i) => pad.l + (W - pad.l - pad.r) * ((i - i0) / Math.max(i1 - i0, 1)), Y = (v) => pad.t + (H - pad.t - pad.b) * (hi - v) / (hi - lo);
  ctx.font = "12px ui-monospace, monospace"; ctx.lineWidth = 1;
  const nt = o.logY ? hi - lo : 6;
  for (let k = 0; k <= nt; k++) {
    const v = o.logY ? lo + k : lo + (hi - lo) * k / nt, y = Y(v);
    ctx.strokeStyle = cssVar("--chart-grid"); ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
    ctx.fillStyle = cssVar("--muted"); ctx.textAlign = "right"; ctx.fillText(o.logY ? `1e${v}` : fmtE(v), pad.l - 8, y + 4);
  }
  ctx.textAlign = "center";
  for (let k = 0; k <= 8; k++) {
    const i = i0 + (i1 - i0) * k / 8, x = X(i);
    ctx.strokeStyle = cssVar("--chart-grid"); ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, H - pad.b); ctx.stroke();
    ctx.fillStyle = cssVar("--muted"); ctx.fillText(String(Math.round(i) + 1), x, H - pad.b + 18);
  }
  ctx.fillStyle = cssVar("--text"); ctx.fillText(o.xLabel || t("iteration"), pad.l + (W - pad.l - pad.r) / 2, H - 6);
  ctx.save(); ctx.translate(14, pad.t + (H - pad.t - pad.b) / 2); ctx.rotate(-Math.PI / 2); ctx.fillText(o.yLabel || "", 0, 0); ctx.restore();
  ctx.save(); ctx.beginPath(); ctx.rect(pad.l, pad.t, W - pad.l - pad.r, H - pad.t - pad.b); ctx.clip();
  for (const s of o.series) {
    ctx.strokeStyle = s.color; ctx.lineWidth = 2; ctx.beginPath(); let on = false;
    for (let i = Math.max(0, Math.floor(i0) - 1); i <= Math.min(s.data.length - 1, Math.ceil(i1) + 1); i++) {
      if (!ok(s.data[i])) continue; const x = X(i), y = Y(tv(s.data[i])); on ? ctx.lineTo(x, y) : ctx.moveTo(x, y); on = true;
    }
    ctx.stroke();
  }
  ctx.restore();
  if (o.hover !== undefined && o.hover !== null) {
    const x = X(o.hover); ctx.strokeStyle = cssVar("--accent"); ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, H - pad.b); ctx.stroke(); ctx.setLineDash([]);
    for (const s of o.series) if (ok(s.data[o.hover])) { ctx.fillStyle = s.color; ctx.beginPath(); ctx.arc(x, Y(tv(s.data[o.hover])), 4, 0, 7); ctx.fill(); }
  }
  return { pad, W, H, n, i0, i1, X };
}
const BIG = { kind: null, range: null, geom: null, hover: null };
function bigSeries() {
  if (BIG.kind === "res") return S.res.keys.map((k) => ({ name: k, color: resColor(k), data: S.res.series[k] }));
  if (BIG.kind === "mon") return [{ name: S.monName || t("monitor"), color: cssVar("--accent"), data: S.mon }];
  return [];
}
function statsRow(s, log) {
  const d = s.data.filter((v) => v !== null && v !== undefined && isFinite(v) && (!log || v > 0));
  if (!d.length) return null;
  const first = d[0], last = d[d.length - 1];
  return { first, last, min: Math.min(...d), max: Math.max(...d), mean: d.reduce((a, b) => a + b, 0) / d.length, orders: log ? Math.log10(first / last) : null, n: d.length };
}
function renderBig() {
  const cv = $("#bigChart"), ser = bigSeries(), log = BIG.kind === "res";
  const n = Math.max(...ser.map((s) => s.data.length), 1);
  if (!BIG.range) BIG.range = [0, Math.max(n - 1, 1)];
  BIG.geom = lineChart(cv, { series: ser, logY: log, range: BIG.range, hover: BIG.hover, yLabel: log ? "scaled residual" : (S.monName || "") });
  const rows = ser.map((s) => ({ s, st: statsRow(s, log) })).filter((r) => r.st);
  $("#bigSide").innerHTML = `<h4>${esc(t("series"))}</h4><table><tr><th>${esc(t("series"))}</th><th>${esc(t("first"))}</th><th>${esc(t("last"))}</th><th>${esc(t("min"))}</th><th>${esc(t("max"))}</th>${log ? `<th>${esc(t("orders"))}</th>` : `<th>${esc(t("mean"))}</th>`}</tr>`
    + rows.map(({ s, st }) => `<tr><td><span class="sw" style="background:${s.color}"></span>${esc(s.name)}</td><td class="n">${fmtE(st.first)}</td><td class="n">${fmtE(st.last)}</td><td class="n">${fmtE(st.min)}</td><td class="n">${fmtE(st.max)}</td><td class="n">${log ? st.orders.toFixed(2) : fmtE(st.mean)}</td></tr>`).join("") + `</table>`
    + `<h4>${esc(t("iteration"))}</h4><div class="mono">${n.toLocaleString()} · ${esc($("#statusPill").textContent)}</div>`;
}
function openBig(kind) {
  BIG.kind = kind; BIG.range = null; BIG.hover = null;
  const dlg = $("#chartDialog");
  $("#chartTitle").textContent = t(kind === "res" ? "chartResiduals" : kind === "mon" ? "chartMonitor" : "chartPerf");
  $("#chartHint").textContent = kind === "perf" ? "" : t("chartHintRes");
  $(".big-plot").classList.toggle("hidden", kind === "perf");
  $(".big-body").style.gridTemplateColumns = kind === "perf" ? "1fr" : "";
  if (!dlg.open) dlg.showModal();
  if (kind === "perf") renderPerf(); else renderBig();
}
function renderPerf() {
  const entries = Object.entries(S.timers || {}).filter(([, v]) => v > 0).sort((a, b) => b[1] - a[1]);
  const tot = entries.reduce((a, [, v]) => a + v, 0) || 1;
  $("#bigSide").innerHTML = `<h4>${esc(t("chartPerf"))}</h4><table style="max-width:880px"><tr><th>${esc(t("stage"))}</th><th>${esc(t("seconds"))}</th><th>${esc(t("share"))}</th><th style="width:45%"></th></tr>`
    + entries.map(([k, v], i) => `<tr><td><span class="sw" style="background:${TIMER_COLORS[i % 7]}"></span>${esc(k)}</td><td class="n">${v.toFixed(3)}</td><td class="n">${(100 * v / tot).toFixed(1)}%</td><td><div class="bar" style="width:${100 * v / tot}%;background:${TIMER_COLORS[i % 7]}"></div></td></tr>`).join("")
    + `</table><h4>${esc(t("step"))}</h4><div class="mono">${esc($("#sStep").textContent)} · ${esc($("#sRate").textContent)} steps/s · ${esc($("#sMcell").textContent)} Mcell/s</div><h4>${esc(t("plan"))}</h4><table style="max-width:880px">${$("#planTable").innerHTML}</table>`;
}
function initBigCharts() {
  const open = (kind) => () => openBig(kind);
  $("#resChart").addEventListener("click", open("res")); $("#monChart").addEventListener("click", open("mon")); $("#perfCard").addEventListener("click", open("perf"));
  for (const [id, kind] of [["resChart", "res"], ["monChart", "mon"], ["perfCard", "perf"]]) $("#" + id).addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openBig(kind); } });
  $("#chartClose").addEventListener("click", () => $("#chartDialog").close());
  const cv = $("#bigChart"), tip = $("#bigTip");
  const idxAt = (e) => { const g = BIG.geom, r = cv.getBoundingClientRect(); if (!g) return null; const f = (e.clientX - r.left - g.pad.l) / (g.W - g.pad.l - g.pad.r); return f < 0 || f > 1 ? null : Math.round(g.i0 + f * (g.i1 - g.i0)); };
  cv.addEventListener("pointermove", (e) => {
    const i = idxAt(e); BIG.hover = i; renderBig();
    if (i === null) { tip.classList.add("hidden"); return; }
    const rows = bigSeries().filter((s) => s.data[i] !== undefined && s.data[i] !== null).map((s) => `<div><span class="sw" style="background:${s.color};display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:6px"></span>${esc(s.name)}: <b>${fmtE(s.data[i])}</b></div>`).join("");
    tip.innerHTML = `<div>${esc(t("iteration"))} <b>${i + 1}</b></div>${rows}`; tip.classList.remove("hidden");
    const pr = $(".big-plot").getBoundingClientRect(), cr = cv.getBoundingClientRect();
    tip.style.left = Math.min(e.clientX - pr.left + 16, pr.width - tip.offsetWidth - 10) + "px"; tip.style.top = Math.max(8, e.clientY - pr.top - 12) + "px";
  });
  cv.addEventListener("pointerleave", () => { BIG.hover = null; tip.classList.add("hidden"); renderBig(); });
  cv.addEventListener("wheel", (e) => {
    e.preventDefault(); const g = BIG.geom; if (!g) return;
    const i = idxAt(e) ?? (g.i0 + g.i1) / 2, k = Math.exp(e.deltaY * 0.0015), n = Math.max(g.n - 1, 1);
    let a = i - (i - g.i0) * k, b = i + (g.i1 - i) * k;
    if (b - a < 4) return; BIG.range = [Math.max(0, a), Math.min(n, b)]; renderBig();
  }, { passive: false });
  cv.addEventListener("dblclick", () => { BIG.range = null; renderBig(); });
  window.addEventListener("resize", () => { if ($("#chartDialog").open && BIG.kind !== "perf") renderBig(); });
}

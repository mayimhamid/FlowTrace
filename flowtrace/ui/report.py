"""Self-contained, print-ready technical report of a finished run.

The report is one HTML file (no scripts, no external files): a cover with the headline numbers,
the computational set-up (mesh figure with the boundary zones in colour, boundary conditions,
models and properties), the numerical methods, the execution plan with a timing breakdown, the
results (colour contour figures, residual history, wall and conservation charts), verification and
validation checks, a discussion and conclusions written from the measured numbers, the limitations,
the reproducibility record and the references. Every figure is inline SVG, so no plotting library is
needed and the file prints cleanly (A4, one section per page break).

Nothing is invented: a section whose data the run does not have (no wall zone, no energy equation,
a 3-D mesh without a snapshot) says so or is left out.
"""

# ruff: noqa: E501  (inline SVG and HTML templates)

from __future__ import annotations

import base64
import html as _html
import json
import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

KIND_COLOUR = {"wall": "#8d8a80", "inlet": "#2b7bbb", "outlet": "#c6613f", "pressure": "#d9822b",
               "symmetry": "#4f9d69", "periodic": "#a8557a", "velocity-inlet": "#2b7bbb", "pressure-outlet": "#c6613f",
               "pressure-far-field": "#d9822b", "interior": "#cccccc"}
SERIES_COLOURS = ["#c6613f", "#2b7bbb", "#4f9d69", "#8e5bb5", "#d9a21b", "#2aa5a0", "#a8557a", "#6b6a64"]
ESC = lambda x: _html.escape(str(x))  # noqa: E731


def turbo(t: float) -> str:
    """Google's polynomial approximation of the Turbo colormap, t in [0, 1] -> '#rrggbb'."""
    t = min(max(float(t), 0.0), 1.0)
    r = 0.13572138 + t * (4.61539260 + t * (-42.66032258 + t * (132.13108234 + t * (-152.94239396 + t * 59.28637943))))
    g = 0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333 + t * (4.27729857 + t * 2.82956604))))
    b = 0.10667330 + t * (12.64194608 + t * (-60.58204836 + t * (110.36276771 + t * (-89.90310912 + t * (27.34824973)))))
    return "#%02x%02x%02x" % tuple(int(round(255 * min(max(c, 0.0), 1.0))) for c in (r, g, b))


def _num(v, digits: int = 4) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not math.isfinite(f):
        return "n/a"
    if f == 0:
        return "0"
    return f"{f:.{digits}g}" if 1e-3 <= abs(f) < 1e6 else f"{f:.{max(digits - 1, 1)}e}"


# ------------------------------------------------------------------ figures
def _cell_polygons(mesh) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ptr, loops = mesh.cell_polygons_2d()
    return mesh.nodes[:, :2], np.asarray(ptr), np.asarray(loops)


def _fit(lo, hi, width, max_h=420, pad=10):
    lx, ly = max(hi[0] - lo[0], 1e-30), max(hi[1] - lo[1], 1e-30)
    s = (width - 2 * pad) / lx
    h = ly * s + 2 * pad
    if h > max_h:
        s *= (max_h - 2 * pad) / (h - 2 * pad)
        h = max_h
    w = lx * s + 2 * pad
    return s, w, h, pad


def svg_mesh_zones(mesh, zone_kinds: Dict[str, str], width: int = 620) -> str:
    """2-D mesh drawn with every boundary zone in the colour of its kind, and a legend."""
    nodes, ptr, loops = _cell_polygons(mesh)
    lo, hi = nodes.min(0), nodes.max(0)
    s, w, h, pad = _fit(lo, hi, width)
    P = lambda n: (pad + (nodes[n, 0] - lo[0]) * s, h - pad - (nodes[n, 1] - lo[1]) * s)  # noqa: E731
    out = [f'<svg viewBox="0 0 {w + 190:.0f} {h:.0f}" role="img" aria-label="Mesh and boundary zones" class="fig">']
    n_cells = len(ptr) - 1
    step = 1 if n_cells <= 12000 else int(math.ceil(n_cells / 12000))
    d = []
    for c in range(0, n_cells, step):
        pts = [P(loops[k]) for k in range(ptr[c], ptr[c + 1])]
        d.append("M" + "L".join(f"{x:.1f},{y:.1f}" for x, y in pts) + "Z")
    out.append(f'<path d="{"".join(d)}" fill="#f4f1ea" stroke="#b9b29a" stroke-width="0.35"/>')
    legend = []
    for z in mesh.boundary_zones():
        f = np.nonzero((mesh.neighbour < 0) & (mesh.face_zone == z.id))[0]
        if not f.size:
            continue
        kind = zone_kinds.get(z.name, z.type)
        col = KIND_COLOUR.get(kind, KIND_COLOUR.get(z.type, "#555"))
        a0, a1 = mesh.face_nodes[mesh.face_ptr[f]], mesh.face_nodes[mesh.face_ptr[f] + 1]
        seg = "".join(f"M{P(i)[0]:.1f},{P(i)[1]:.1f}L{P(j)[0]:.1f},{P(j)[1]:.1f}" for i, j in zip(a0, a1))
        out.append(f'<path d="{seg}" stroke="{col}" stroke-width="3.2" stroke-linecap="round" fill="none"/>')
        legend.append((z.name, kind, col, int(f.size)))
    y0 = 24
    for k, (name, kind, col, cnt) in enumerate(legend):
        y = y0 + 36 * k
        out.append(f'<rect x="{w + 8:.0f}" y="{y - 9}" width="14" height="14" rx="3" fill="{col}"/>'
                   f'<text x="{w + 28:.0f}" y="{y + 2}" font-size="12" fill="#1f1e1d">{ESC(name)}</text>'
                   f'<text x="{w + 28:.0f}" y="{y + 15}" font-size="10" fill="#6f6c63">{ESC(kind)} · {cnt} faces</text>')
    out.append("</svg>")
    return "".join(out)


def svg_surface_points(mesh, zone_kinds: Dict[str, str], width: int = 300) -> str:
    """Three orthographic views of a 3-D mesh's boundary (sampled face centres, coloured by zone)."""
    from .server import surface_preview

    g = surface_preview(mesh, per_zone=6000)
    pts = np.frombuffer(base64.b64decode(g["pts"]), np.float32).reshape(-1, 3)
    zid = np.frombuffer(base64.b64decode(g["zid"]), np.int32)
    lo, hi = pts.min(0), pts.max(0)
    cols = [KIND_COLOUR.get(zone_kinds.get(nm, ty), KIND_COLOUR.get(ty, "#555")) for nm, ty in zip(g["names"], g["types"])]
    panels, x_off = [], 0
    hmax = 0
    for (a, b, title) in ((0, 1, "x–y"), (0, 2, "x–z"), (2, 1, "z–y")):
        s, w, h, pad = _fit((lo[a], lo[b]), (hi[a], hi[b]), width, max_h=300)
        hmax = max(hmax, h)
        body = "".join(f'<circle cx="{x_off + pad + (p[a] - lo[a]) * s:.1f}" cy="{h - pad - (p[b] - lo[b]) * s:.1f}" r="0.9" fill="{cols[z]}"/>'
                       for p, z in zip(pts, zid))
        panels.append(f'<text x="{x_off + 6}" y="14" font-size="11" fill="#6f6c63">{title}</text>{body}')
        x_off += w + 6
    legend = "".join(f'<rect x="{x_off + 8:.0f}" y="{12 + 20 * k}" width="12" height="12" rx="3" fill="{c}"/>'
                     f'<text x="{x_off + 26:.0f}" y="{23 + 20 * k}" font-size="11" fill="#1f1e1d">{ESC(nm)}</text>'
                     for k, (nm, c) in enumerate(zip(g["names"], cols)))
    return (f'<svg viewBox="0 0 {x_off + 170:.0f} {hmax:.0f}" role="img" aria-label="Boundary surfaces" class="fig">'
            + "".join(panels) + legend + "</svg>")


def svg_field(mesh, values: np.ndarray, name: str, unit: str = "", width: int = 560) -> str:
    """2-D cell field as coloured polygons with a colour bar."""
    nodes, ptr, loops = _cell_polygons(mesh)
    lo, hi = nodes.min(0), nodes.max(0)
    s, w, h, pad = _fit(lo, hi, width)
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(v)
    vmin, vmax = (float(v[ok].min()), float(v[ok].max())) if ok.any() else (0.0, 1.0)
    span = (vmax - vmin) or 1.0
    P = lambda n: (pad + (nodes[n, 0] - lo[0]) * s, h - pad - (nodes[n, 1] - lo[1]) * s)  # noqa: E731
    out = [f'<svg viewBox="0 0 {w + 110:.0f} {max(h, 190):.0f}" role="img" aria-label="{ESC(name)} field" class="fig">']
    for c in range(len(ptr) - 1):
        col = turbo((v[c] - vmin) / span) if ok[c] else "#999"
        pts = "L".join(f"{P(loops[k])[0]:.1f},{P(loops[k])[1]:.1f}" for k in range(ptr[c], ptr[c + 1]))
        out.append(f'<path d="M{pts}Z" fill="{col}" stroke="{col}" stroke-width="0.4"/>')
    bx, by, bh = w + 20, 20, max(h, 190) - 50
    for k in range(60):
        out.append(f'<rect x="{bx}" y="{by + bh * k / 60:.1f}" width="16" height="{bh / 60 + 0.6:.1f}" fill="{turbo(1 - k / 59)}"/>')
    out.append(f'<rect x="{bx}" y="{by}" width="16" height="{bh}" fill="none" stroke="#555" stroke-width="0.6"/>')
    for k in range(5):
        val = vmax - span * k / 4
        out.append(f'<text x="{bx + 22}" y="{by + bh * k / 4 + 4:.1f}" font-size="10" fill="#1f1e1d">{_num(val, 3)}</text>')
    out.append(f'<text x="{bx}" y="{by - 7}" font-size="10.5" font-weight="600" fill="#1f1e1d">{ESC(unit)}</text></svg>')
    return "".join(out)


def svg_residuals(history: List[dict], width: int = 720, height: int = 340) -> str:
    """Residual history, log scale, one colour per equation, axis titles and the orders of magnitude dropped."""
    if not history:
        return "<p>No residual history.</p>"
    keys = list(history[-1].get("scaled", {}).keys())
    pts = {k: [(i, ev.get("scaled", {}).get(k)) for i, ev in enumerate(history)] for k in keys}
    vals = [math.log10(v) for k in keys for _, v in pts[k] if v and v > 0 and math.isfinite(v)]
    if not vals:
        return "<p>No residual history.</p>"
    lo, hi = math.floor(min(vals)), math.ceil(max(vals))
    hi = max(hi, lo + 1)
    n = max(len(history) - 1, 1)
    L, R, T, B = 62, 16, 14, 44
    X = lambda i: L + (width - L - R) * i / n  # noqa: E731
    Y = lambda lv: T + (height - T - B) * (hi - lv) / (hi - lo)  # noqa: E731
    out = [f'<svg viewBox="0 0 {width} {height + 24 * ((len(keys) + 3) // 4)}" role="img" aria-label="Residual history" class="fig">']
    for e in range(lo, hi + 1):
        out.append(f'<line x1="{L}" x2="{width - R}" y1="{Y(e):.1f}" y2="{Y(e):.1f}" stroke="#e6e2d6"/>'
                   f'<text x="{L - 6}" y="{Y(e) + 4:.1f}" font-size="11" text-anchor="end" fill="#6f6c63">1e{e}</text>')
    for k in range(6):
        i = n * k / 5
        out.append(f'<line x1="{X(i):.1f}" x2="{X(i):.1f}" y1="{T}" y2="{height - B}" stroke="#efece3"/>'
                   f'<text x="{X(i):.1f}" y="{height - B + 16}" font-size="11" text-anchor="middle" fill="#6f6c63">{int(round(i)) + 1}</text>')
    out.append(f'<text x="{(L + width - R) / 2:.0f}" y="{height - 6}" font-size="12" text-anchor="middle" fill="#1f1e1d">Iteration</text>'
               f'<text transform="translate(14 {(T + height - B) / 2:.0f}) rotate(-90)" font-size="12" text-anchor="middle" fill="#1f1e1d">Scaled residual</text>')
    for c, k in enumerate(keys):
        d = " ".join(f"{X(i):.1f},{Y(math.log10(v)):.1f}" for i, v in pts[k] if v and v > 0 and math.isfinite(v))
        col = SERIES_COLOURS[c % len(SERIES_COLOURS)]
        out.append(f'<polyline fill="none" stroke="{col}" stroke-width="1.7" points="{d}"/>')
        lx, ly = L + 6 + (c % 4) * 165, height + 12 + 22 * (c // 4)
        first = next((v for _, v in pts[k] if v and v > 0), None)
        last = next((v for _, v in reversed(pts[k]) if v and v > 0), None)
        drop = f" · {math.log10(first / last):.1f} orders" if first and last else ""
        out.append(f'<rect x="{lx}" y="{ly - 9}" width="12" height="4" fill="{col}"/>'
                   f'<text x="{lx + 17}" y="{ly}" font-size="11" fill="#1f1e1d">{ESC(k)}{drop}</text>')
    out.append("</svg>")
    return "".join(out)


def svg_bars(items: Sequence[Tuple[str, float]], unit: str = "", width: int = 560, signed: bool = False) -> str:
    """Horizontal bar chart (signed: bars grow left and right of a zero line)."""
    items = [(a, float(b)) for a, b in items if b is not None and math.isfinite(float(b))]
    if not items:
        return ""
    m = max(abs(v) for _, v in items) or 1.0
    row, L = 26, 150
    h = 12 + row * len(items) + 18
    zero = L + (width - L - 90) / 2 if signed else L
    span = (width - L - 90) / (2 if signed else 1)
    out = [f'<svg viewBox="0 0 {width} {h}" role="img" class="fig">']
    for i, (name, v) in enumerate(items):
        y = 10 + row * i
        w = span * abs(v) / m
        x = zero if v >= 0 or not signed else zero - w
        col = SERIES_COLOURS[i % len(SERIES_COLOURS)] if not signed else ("#2b7bbb" if v < 0 else "#c6613f")
        out.append(f'<text x="{L - 8}" y="{y + 13}" font-size="12" text-anchor="end" fill="#1f1e1d">{ESC(name)}</text>'
                   f'<rect x="{x:.1f}" y="{y}" width="{max(w, 1):.1f}" height="17" rx="3" fill="{col}"/>'
                   f'<text x="{(x + w + 6) if v >= 0 else (x - 6):.1f}" y="{y + 13}" font-size="11" '
                   f'text-anchor="{"start" if v >= 0 else "end"}" fill="#1f1e1d">{_num(v, 4)} {ESC(unit)}</text>')
    if signed:
        out.append(f'<line x1="{zero:.1f}" x2="{zero:.1f}" y1="4" y2="{h - 14}" stroke="#555" stroke-width="0.8"/>')
    out.append("</svg>")
    return "".join(out)


def svg_stacked(timers: Dict[str, float], width: int = 700) -> str:
    entries = sorted(((k, float(v)) for k, v in timers.items() if v and v > 0), key=lambda kv: -kv[1])
    tot = sum(v for _, v in entries)
    if not entries or tot <= 0:
        return ""
    out = [f'<svg viewBox="0 0 {width} {60 + 20 * ((len(entries) + 2) // 3)}" role="img" class="fig">']
    x = 0.0
    for i, (_, v) in enumerate(entries):
        w = width * v / tot
        out.append(f'<rect x="{x:.1f}" y="6" width="{max(w - 1, 0.5):.1f}" height="30" fill="{SERIES_COLOURS[i % len(SERIES_COLOURS)]}"/>')
        if w > 40:
            out.append(f'<text x="{x + w / 2:.1f}" y="26" font-size="11" text-anchor="middle" fill="#fff">{100 * v / tot:.0f}%</text>')
        x += w
    for i, (k, v) in enumerate(entries):
        lx, ly = (i % 3) * (width / 3), 58 + 20 * (i // 3)
        out.append(f'<rect x="{lx:.0f}" y="{ly - 9}" width="11" height="11" rx="2" fill="{SERIES_COLOURS[i % len(SERIES_COLOURS)]}"/>'
                   f'<text x="{lx + 16:.0f}" y="{ly}" font-size="11" fill="#1f1e1d">{ESC(k)} {v:.3g} s</text>')
    out.append("</svg>")
    return "".join(out)


# ------------------------------------------------------------------ the report
CSS = """
:root{--ink:#1f1e1d;--muted:#6f6c63;--line:#e3e0d4;--accent:#c6613f;--bg:#faf9f5;--panel:#fff}
*{box-sizing:border-box}body{font:14px/1.6 'Segoe UI',system-ui,Roboto,sans-serif;color:var(--ink);background:var(--bg);margin:0}
.page{max-width:980px;margin:0 auto;padding:32px 36px;background:var(--panel)}
h1{font:600 34px/1.2 Georgia,'Iowan Old Style',serif;margin:0 0 6px}h2{font:600 21px Georgia,serif;margin:34px 0 10px;padding-bottom:6px;border-bottom:2px solid var(--accent)}
h3{font:600 16px Georgia,serif;margin:20px 0 6px}p{margin:8px 0}.m{color:var(--muted)}.small{font-size:12px}
.cover{border-top:6px solid var(--accent);padding-top:18px;margin-bottom:8px}
.badge{display:inline-block;border-radius:999px;padding:2px 11px;font-size:12px;font-weight:600;color:#fff}
.ok{background:#3f7d4a}.warn{background:#c98a1b}.bad{background:#b3372f}.neutral{background:#6f6c63}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:16px 0}
.card{border:1px solid var(--line);border-radius:10px;padding:10px 12px;background:var(--bg)}
.card .k{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px}.card .v{font:600 22px Georgia,serif;margin-top:2px}.card .s{font-size:11px;color:var(--muted)}
table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}th,td{border-bottom:1px solid var(--line);padding:5px 8px;text-align:left;vertical-align:top}
th{background:#f3f0e7;font-weight:600}td.n{font-family:Consolas,monospace;text-align:right;white-space:nowrap}
.fig{width:100%;height:auto;max-height:560px;background:#fff;border:1px solid var(--line);border-radius:8px}
figure{margin:14px 0}figcaption{font-size:12.5px;color:var(--muted);margin-top:5px}figcaption b{color:var(--ink)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}pre{background:#f6f3ea;border:1px solid var(--line);padding:10px;overflow:auto;font-size:11.5px;border-radius:6px}
.chip{display:inline-block;border-radius:5px;padding:1px 8px;font-size:11.5px;color:#fff;font-weight:600}
ol.refs{font-size:12.5px;padding-left:22px}
@media print{body{background:#fff}.page{padding:0;max-width:none}h2{break-after:avoid}figure,table,.cards{break-inside:avoid}.pb{break-before:page}@page{size:A4;margin:16mm}}
@media (max-width:720px){.grid2{grid-template-columns:1fr}}
"""

REFERENCES = {
    "simple": "Patankar, S. V. and Spalding, D. B. (1972). A calculation procedure for heat, mass and momentum transfer in "
              "three-dimensional parabolic flows. Int. J. Heat Mass Transfer 15, 1787–1806.",
    "simplec": "Van Doormaal, J. P. and Raithby, G. D. (1984). Enhancements of the SIMPLE method for predicting incompressible "
               "fluid flows. Numerical Heat Transfer 7, 147–163.",
    "rhie": "Rhie, C. M. and Chow, W. L. (1983). Numerical study of the turbulent flow past an airfoil with trailing edge "
            "separation. AIAA J. 21, 1525–1532.",
    "sst": "Menter, F. R. (1994). Two-equation eddy-viscosity turbulence models for engineering applications. AIAA J. 32, 1598–1605.",
    "ke": "Shih, T.-H., Liou, W. W., Shabbir, A., Yang, Z. and Zhu, J. (1995). A new k-epsilon eddy viscosity model for high "
          "Reynolds number turbulent flows. Computers & Fluids 24, 227–238.",
    "fvm": "Moukalled, F., Mangani, L. and Darwish, M. (2016). The Finite Volume Method in Computational Fluid Dynamics. Springer.",
    "mac": "Harlow, F. H. and Welch, J. E. (1965). Numerical calculation of time-dependent viscous incompressible flow of fluid "
           "with free surface. Phys. Fluids 8, 2182–2189.",
    "chorin": "Chorin, A. J. (1968). Numerical solution of the Navier–Stokes equations. Math. Comp. 22, 745–762.",
    "p1": "Modest, M. F. (2013). Radiative Heat Transfer, 3rd ed. Academic Press.",
}


def _badge(ok: Optional[bool], yes="pass", no="check") -> str:
    if ok is None:
        return "<span class='badge neutral'>n/a</span>"
    return f"<span class='badge {'ok' if ok else 'warn'}'>{yes if ok else no}</span>"


def _kv(rows: Sequence[Tuple[str, object]], head: Optional[Tuple[str, str]] = None) -> str:
    h = f"<tr><th>{ESC(head[0])}</th><th>{ESC(head[1])}</th></tr>" if head else ""
    return "<table>" + h + "".join(f"<tr><th style='width:42%'>{ESC(k)}</th><td>{v if isinstance(v, str) and v.startswith('<') else ESC(v)}</td></tr>"
                                   for k, v in rows) + "</table>"


def build_report(app, job, image: Optional[str] = None) -> str:
    """The report of ``job`` (HTML text). ``app`` supplies the machine profile and the event history."""
    from .server import _diagnostics_table, _jsonable, _version

    res = _jsonable(job.result or {})
    plan = job.plan or {}
    prof = app.profile()
    solver = job.solver
    case = getattr(job, "case", None)
    mesh = getattr(job, "mesh", None)
    spec = dict(job.spec)
    spec.pop("field", None)
    mine = [ev for ev in app.hub.residuals if ev.get("job") == job.id]
    n_fig, n_tab = [0], [0]

    def fig(svg: str, caption: str) -> str:
        if not svg:
            return ""
        n_fig[0] += 1
        return f"<figure>{svg}<figcaption><b>Figure {n_fig[0]}.</b> {caption}</figcaption></figure>"

    def tcap(caption: str) -> str:
        n_tab[0] += 1
        return f"<p class='small m'><b>Table {n_tab[0]}.</b> {caption}</p>"

    name = getattr(case, "name", None) or spec.get("preset") or "FlowTrace case"
    converged = bool(res.get("converged"))
    iters = res.get("iterations") or res.get("steps") or 0
    elapsed = (job.finished_at or time.time()) - (job.started or time.time()) if getattr(job, "started", None) else None
    status = job.status
    cells = getattr(mesh, "n_cells", None) or (getattr(case, "cells", None) if case is not None else None)
    parts: List[str] = []

    # ---- cover and key numbers
    sb = {"converged": "ok", "finished": "neutral", "stopped": "warn", "error": "bad"}.get(status, "neutral")
    parts.append(f"<div class='cover'><div class='m small'>FlowTrace technical report</div><h1>{ESC(name)}</h1>"
                 f"<p class='m'>Job {ESC(job.id)} · FlowTrace {ESC(_version())} · {ESC(time.strftime('%d %B %Y, %H:%M'))} · "
                 f"status <span class='badge {sb}'>{ESC(status)}</span></p></div>")
    mb = res.get("mass_balance") or {}
    mass_rel = None
    if mb:
        scale = max(max(abs(v) for v in mb.values()), 1e-30)
        mass_rel = abs(sum(mb.values())) / scale
    drop = None
    if mine and mine[-1].get("scaled"):
        k0 = "continuity" if "continuity" in mine[-1]["scaled"] else next(iter(mine[-1]["scaled"]))
        a = next((ev["scaled"][k0] for ev in mine if ev["scaled"].get(k0, 0) > 0), None)
        b = mine[-1]["scaled"].get(k0)
        if a and b and b > 0:
            drop = math.log10(a / b)
    cards = [("Cells", _num(cells) if cells else "–", f"{getattr(mesh, 'dim', '')}-D" if mesh is not None else ""),
             ("Iterations / steps", _num(iters), "converged" if converged else "not converged"),
             ("Wall time", f"{elapsed:.1f} s" if elapsed else "–", plan.get("backend", "")),
             ("Continuity drop", f"{drop:.1f}" if drop is not None else "–", "orders of magnitude"),
             ("Mass imbalance", _num(mass_rel, 2) if mass_rel is not None else "–", "relative, global")]
    parts.append("<div class='cards'>" + "".join(f"<div class='card'><div class='k'>{ESC(k)}</div><div class='v'>{ESC(v)}</div>"
                                                 f"<div class='s'>{ESC(s)}</div></div>" for k, v, s in cards) + "</div>")

    # ---- 1. problem and set-up
    parts.append("<h2>1. Problem description and computational set-up</h2>")
    zone_kinds = {n: b.kind for n, b in (getattr(case, "boundaries", None) or {}).items() if hasattr(b, "kind")}
    if mesh is not None:
        q = None
        try:
            from ..mesh.quality import mesh_quality

            q = mesh_quality(mesh) if mesh.n_cells <= 3_000_000 else None
        except Exception:
            q = None
        rows = [("Dimension", f"{mesh.dim}-D"), ("Cells", _num(mesh.n_cells)), ("Faces", _num(mesh.n_faces)),
                ("Nodes", _num(mesh.n_nodes)), ("Boundary zones", ", ".join(z.name for z in mesh.boundary_zones()))]
        if q is not None:
            rows += [("Minimum orthogonal quality", _num(q.min_orthogonal_quality)), ("Maximum skewness", _num(q.max_skewness)),
                     ("Maximum aspect ratio", _num(q.max_aspect_ratio)), ("Maximum non-orthogonality", f"{_num(q.max_non_orthogonality_deg)}°"),
                     ("Total volume (area in 2-D)", _num(q.total_volume))]
        parts.append(_kv(rows) + tcap("Mesh statistics."))
        try:
            svg = svg_mesh_zones(mesh, zone_kinds) if mesh.dim == 2 else svg_surface_points(mesh, zone_kinds)
            parts.append(fig(svg, "Computational mesh with the boundary zones coloured by condition type "
                                  + ("(every cell drawn)." if mesh.dim == 2 else "(sampled boundary-face centres, three projections).")))
        except Exception as exc:  # a picture must never break the report
            parts.append(f"<p class='m small'>Mesh figure not available ({ESC(type(exc).__name__)}).</p>")
    elif case is not None and hasattr(case, "nx"):
        parts.append(_kv([("Grid", f"{case.nx} × {case.ny}"), ("Cells", _num(case.cells)), ("Domain", f"{case.lx} × {case.ly}")])
                     + tcap("Structured (staggered MAC) grid."))
    bcs = getattr(case, "boundaries", None) if case is not None else None
    if bcs:
        rows = []
        for zname, b in bcs.items():
            kind = getattr(b, "kind", "?")
            vals = []
            if getattr(b, "normal_speed", None) is not None:
                vals.append(f"normal speed {_num(b.normal_speed)} m/s")
            if getattr(b, "velocity", None):
                vals.append(f"velocity {[round(float(x), 6) for x in b.velocity]}")
            if kind in ("outlet", "pressure"):
                vals.append(f"gauge pressure {_num(b.pressure)} Pa")
            th = getattr(b, "thermal", "adiabatic")
            if kind == "wall" and th != "adiabatic":
                vals.append(f"{th} {_num(b.value)}")
            elif kind in ("inlet", "pressure") and getattr(b, "value", 0):
                vals.append(f"T = {_num(b.value)} K")
            rows.append(f"<tr><td>{ESC(zname)}</td><td><span class='chip' style='background:{KIND_COLOUR.get(kind, '#555')}'>{ESC(kind)}</span></td>"
                        f"<td>{ESC('; '.join(vals) or 'default')}</td></tr>")
        parts.append("<table><tr><th>Zone</th><th>Condition</th><th>Specification</th></tr>" + "".join(rows) + "</table>"
                     + tcap("Boundary conditions."))
    if case is not None and hasattr(case, "viscosity"):
        mu, rho = case.viscosity, case.density
        rows = [("Density ρ", f"{_num(rho)} kg/m³"), ("Dynamic viscosity μ", f"{_num(mu)} Pa·s"),
                ("Turbulence model", case.turbulence), ("Energy equation", "on" if case.energy else "off")]
        if case.energy:
            rows += [("Conductivity k", f"{_num(case.conductivity)} W/(m·K)"), ("Specific heat cp", f"{_num(case.cp)} J/(kg·K)")]
        if getattr(case, "radiation", "off") != "off":
            rows.append(("Radiation", "P-1 (absorption " + _num(case.absorption) + " 1/m)"))
        if getattr(case, "species", None):
            rows.append(("Species", ", ".join(case.species)))
        speeds = [abs(b.normal_speed) for b in (bcs or {}).values() if getattr(b, "normal_speed", None)]
        if speeds and mu:
            rows.append(("Largest inlet speed / ρU/μ", f"{_num(max(speeds))} m/s · {_num(rho * max(speeds) / mu)} 1/m"))
        parts.append(_kv(rows) + tcap("Fluid properties and models."))
    elif case is not None and hasattr(case, "nu"):
        parts.append(_kv([("Kinematic viscosity ν", _num(case.nu)), ("Energy equation", "on" if case.energy else "off")])
                     + tcap("Fluid properties."))

    # ---- 2. numerical methods
    parts.append("<h2>2. Numerical methods</h2>")
    refs = ["fvm"]
    if case is not None and hasattr(case, "viscosity"):
        algo = getattr(solver, "algo", None) or case.algorithm
        refs += ["simple", "rhie"] + (["simplec"] if "SIMPLEC" in (algo, case.algorithm) else [])
        txt = (f"The incompressible Navier–Stokes equations are discretised with a cell-centred, collocated finite-volume method on "
               f"the unstructured mesh. Pressure–velocity coupling uses the <b>{ESC(algo)}</b> algorithm; face mass fluxes follow Rhie–Chow "
               f"interpolation. Convection is <b>{ESC(case.scheme)}</b> for momentum"
               + (f" and <b>{ESC(case.scalar_convection)}</b> for energy and species"
                  if case.scalar_convection != case.scheme else "")
               + "; gradients are least-squares, and the non-orthogonal part of the diffusive flux is treated explicitly. "
               "The pressure correction is solved with an algebraic multigrid preconditioned conjugate gradient method.")
        if case.turbulence == "k-omega-sst":
            refs.append("sst")
            txt += " Turbulence is closed with the k–ω SST model (Menter, 1994) with wall functions."
        elif case.turbulence.startswith("k-epsilon"):
            refs.append("ke")
            txt += " Turbulence is closed with the realizable k–ε model (Shih et al., 1995) with standard wall functions."
        if getattr(case, "radiation", "off") == "p1":
            refs.append("p1")
            txt += " Radiative transfer uses the P-1 spherical-harmonics approximation."
        parts.append(f"<p>{txt}</p>")
        relax_p = getattr(solver, "relax_p", case.relax_p)
        rows = [("Pressure–velocity coupling", algo), ("Momentum under-relaxation", _num(getattr(solver, "relax_u", case.relax_u))),
                ("Pressure under-relaxation", _num(relax_p)), ("Convergence target (scaled residual)", _num(case.tol)),
                ("Maximum iterations", _num(case.max_iter)), ("Precision", case.precision)]
        guard = getattr(solver, "guard_events", None)
        if guard:
            rows.append(("Stability guard", f"{len(guard)} automatic recoveries (last: {ESC(guard[-1])})"))
        parts.append(_kv(rows) + tcap("Solver controls (values in effect at the end of the run)."))
    else:
        refs += ["mac", "chorin"]
        parts.append("<p>The incompressible Navier–Stokes equations are solved on a staggered marker-and-cell grid (Harlow and Welch, 1965) "
                     "with a projection method (Chorin, 1968); the pressure Poisson equation is solved with the method named in the "
                     "execution plan below.</p>")

    # ---- 3. execution
    parts.append("<h2 class='pb'>3. Computing environment and execution plan</h2>")
    parts.append(_kv([("CPU", prof.cpu_model), ("GPU", prof.best_gpu.name if prof.best_gpu else "none"), ("OS", prof.os),
                      ("Backend / precision", f"{plan.get('backend', '–')} / {plan.get('precision', '–')}")])
                 + tcap("Machine and the backend chosen by the planner."))
    pl = plan.get("placements", [])
    if pl:
        parts.append("<h3>Execution plan</h3><table><tr><th>Stage</th><th>Device</th><th>Method</th><th>Reason</th></tr>"
                     + "".join(f"<tr><td>{ESC(p.get('stage'))}</td><td>{ESC(p.get('device'))}</td><td>{ESC(p.get('method'))}</td>"
                               f"<td>{ESC(p.get('reason'))}</td></tr>" for p in pl) + "</table>")
    else:
        parts.append("<h3>Execution plan</h3><p class='m'>No plan was recorded for this run.</p>")
    parts.append(fig(svg_stacked(res.get("timers") or {}), "Where the wall time went, by solver stage."))

    # ---- 4. results
    parts.append("<h2 class='pb'>4. Results</h2>")
    shown = 0
    if solver is not None and mesh is not None and getattr(mesh, "dim", 0) == 2 and mesh.n_cells <= 60000 and hasattr(solver, "fields"):
        try:
            fields = solver.fields()
            order = ["velocity-magnitude", "pressure", "temperature", "turb-kinetic-energy", "x-velocity", "y-velocity"]
            units = {"velocity-magnitude": "m/s", "x-velocity": "m/s", "y-velocity": "m/s", "pressure": "Pa", "temperature": "K",
                     "turb-kinetic-energy": "m²/s²"}
            for fname in [f for f in order if f in fields][:4]:
                parts.append(fig(svg_field(mesh, np.asarray(fields[fname]), fname, units.get(fname, "")),
                                 f"Converged {ESC(fname)} field (cell values, Turbo colour map; range "
                                 f"{_num(float(np.nanmin(fields[fname])))} … {_num(float(np.nanmax(fields[fname])))} {units.get(fname, '')})."))
                shown += 1
        except Exception as exc:
            parts.append(f"<p class='m small'>Field figures not available ({ESC(type(exc).__name__)}: {ESC(exc)}).</p>")
    if isinstance(image, str) and image.startswith("data:image/png;base64,"):
        parts.append(fig(f"<img class='fig' alt='Field snapshot' src='{image}'>", "Field shown in the dashboard when the report was exported."))
        shown += 1
    if not shown:
        parts.append("<p class='m'>No field figure is available for this run (3-D meshes are shown through the dashboard snapshot).</p>")
    parts.append(fig(svg_residuals(mine), "Convergence history: scaled residual of every solved equation, with the orders of magnitude each dropped."))
    if res.get("wall_report"):
        wr = res["wall_report"]
        parts.append("<h3>Wall report</h3><table><tr><th>Zone</th><th>Force (total) [N or N/m]</th><th>Cd</th><th>Cl</th><th>y⁺ max</th><th>Cf avg</th></tr>"
                     + "".join(f"<tr><td>{ESC(z)}</td><td class='n'>{ESC([round(v, 6) for v in r['force_total']])}</td><td class='n'>{r['cd']:.5g}</td>"
                               f"<td class='n'>{r['cl']:.5g}</td><td class='n'>{r['yplus_max']:.3g}</td><td class='n'>{r['cf_avg']:.4g}</td></tr>"
                               for z, r in wr.items()) + "</table>")
        parts.append(fig(svg_bars([(z, float(np.linalg.norm(r["force_total"]))) for z, r in wr.items()], "N", signed=False),
                         "Magnitude of the total force on each wall zone."))

    # ---- 5. verification
    parts.append("<h2 class='pb'>5. Verification and validation</h2>")
    checks = (res.get("validation") or {}).get("checks") or []
    if checks:
        parts.append("<table><tr><th>Check</th><th>Value</th><th>Reference</th><th>Result</th></tr>"
                     + "".join(f"<tr><td>{ESC(c.get('name'))}</td><td class='n'>{ESC(_num(c.get('value')))} {ESC(c.get('unit', ''))}</td>"
                               f"<td class='n'>{ESC(c.get('reference', '–'))}</td><td>{_badge(bool(c.get('pass')))}</td></tr>" for c in checks)
                     + "</table>" + tcap("Checks run automatically on the finished solution."))
    else:
        parts.append("<p class='m'>No automatic checks were recorded.</p>")
    if mb:
        extra = {}
        if "continuity_local_max" in res:
            extra["local continuity: max cell imbalance / (sum of |face mass fluxes| per cell)"] = res["continuity_local_max"]
        parts.append("<h3>Mass balance</h3>" + _kv([(k, _num(v)) for k, v in {**mb, "net": sum(mb.values()), **extra}.items()],
                                                   ("Zone", "Mass flow [kg/s]")))
        parts.append(fig(svg_bars(list(mb.items()), "kg/s", signed=True), "Mass flow through every boundary zone (negative = into the domain)."))
    diag = res.get("diagnostics") or {}
    if diag.get("available"):
        parts.append("<h3>Solver diagnostics (last iteration)</h3>" + _diagnostics_table(diag, ESC))

    # ---- 6. discussion
    parts.append("<h2>6. Discussion and conclusions</h2>")
    pts = []
    if converged:
        pts.append(f"The solution <b>converged</b> to the requested tolerance in {_num(iters)} iterations"
                   + (f" ({elapsed:.1f} s on the {ESC(plan.get('backend', 'selected'))} backend)." if elapsed else "."))
    else:
        pts.append(f"The run ended with status <b>{ESC(status)}</b> after {_num(iters)} iterations <b>without meeting the convergence target</b>; "
                   "the fields below are the last iterate and must not be used as a converged solution.")
    if drop is not None:
        pts.append(f"The continuity residual fell by {drop:.1f} orders of magnitude.")
    if mass_rel is not None:
        pts.append(f"The global mass imbalance is {_num(mass_rel, 2)} of the largest boundary flow"
                   + (", which is within round-off." if mass_rel < 1e-6 else "."))
    bad = [c for c in checks if not c.get("pass")]
    if checks:
        pts.append(f"{len(checks) - len(bad)} of {len(checks)} automatic checks passed" + (": " + "; ".join(ESC(c.get("name")) for c in bad) if bad else "."))
    guard = getattr(solver, "guard_events", None)
    if guard:
        pts.append(f"The stability guard intervened {len(guard)} time(s) (rollback and lower relaxation); the final settings are listed in Table "
                   f"{n_tab[0]}.")
    parts.append("<ul>" + "".join(f"<li>{p}</li>" for p in pts) + "</ul>")
    parts.append("<h3>Limitations</h3><ul><li>Verification here means conservation and consistency checks on this solution. No comparison with "
                 "experiment or with another solver is part of this report unless one is listed in Section 5.</li>"
                 "<li>Mesh independence was not assessed in this run: repeat on a refined mesh before drawing quantitative conclusions.</li>"
                 "<li>Steady RANS results are time-averaged solutions of a model; turbulence-model error is not quantified.</li></ul>")

    # ---- 7. reproducibility
    parts.append("<h2 class='pb'>7. Reproducibility record</h2>")
    parts.append(f"<p>FlowTrace {ESC(_version())}. The complete job specification follows; saving it as a project reproduces the run.</p>"
                 "<pre>" + ESC(json.dumps(_jsonable(spec), indent=1)) + "</pre>")
    log = [ev.get("msg", "") for ev in list(app.hub.log) if ev.get("job") == job.id][-80:]
    parts.append(f"<h3>Log of job {ESC(job.id)}</h3><pre>" + ESC("\n".join(log)) + "</pre>")

    # ---- references
    parts.append("<h2>References</h2><ol class='refs'>" + "".join(f"<li>{ESC(REFERENCES[r])}</li>" for r in dict.fromkeys(refs)) + "</ol>")
    parts.append("<p class='m small'>Ansys and Fluent are trademarks of ANSYS, Inc. FlowTrace is an independent project, not affiliated with "
                 "or endorsed by ANSYS, Inc.</p>")
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'><title>FlowTrace report — " + ESC(name) + "</title>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'><style>" + CSS + "</style></head><body><div class='page'>"
            + "".join(parts) + "</div></body></html>")

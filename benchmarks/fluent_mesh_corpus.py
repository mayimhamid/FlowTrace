"""Read every Fluent mesh/case file given (files or folders) and check that it is ready to solve.

For each file: format, read time, cells/faces, orientation repairs, mesh quality, and (up to
``--solve-limit`` cells) a few SIMPLEC iterations with the default boundary conditions, which
must stay finite. Writes a Markdown table (``--out``) and exits non-zero if any file failed.

    python benchmarks/fluent_mesh_corpus.py examples "C:/path/to/meshes" --out benchmarks/results/fluent_corpus.md
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np

EXT = (".msh", ".cas", ".gz", ".h5", ".txt")


def files_of(paths):
    for p in paths:
        if os.path.isdir(p):
            for f in sorted(glob.glob(os.path.join(p, "**", "*"), recursive=True)):
                if f.lower().endswith(EXT) and not f.lower().endswith(".dat.h5"):
                    yield f
        else:
            yield p


def check(path: str, solve_limit: int) -> dict:
    from flowtrace.mesh.fluent_h5 import read_any_mesh
    from flowtrace.mesh.quality import mesh_quality
    from flowtrace.unstructured.case import UnstructuredCase, default_boundaries
    from flowtrace.unstructured.simple import SimpleSolver

    row = {"file": os.path.basename(path), "size_mb": os.path.getsize(path) / 2**20}
    t = time.perf_counter()
    mesh, fmt = read_any_mesh(path)
    row.update(format=fmt, read_s=time.perf_counter() - t, dim=mesh.dim, cells=mesh.n_cells, faces=mesh.n_faces,
               zones=len(mesh.boundary_zones()))
    q = mesh_quality(mesh)
    row.update(min_oq=q.min_orthogonal_quality, repaired=q.left_handed_faces_repaired, quality_ok=q.ok)
    if mesh.n_cells <= solve_limit:
        case = UnstructuredCase(boundaries=default_boundaries(mesh), max_iter=5, tol=0.0, backend="numpy")
        t = time.perf_counter()
        s = SimpleSolver(mesh, case)
        for _ in range(5):
            s.step()
        row["solve5_s"] = time.perf_counter() - t
        row["finite"] = bool(all(np.isfinite(v).all() for v in s.fields().values()))
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--solve-limit", type=int, default=400_000)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    rows, failed = [], 0
    for f in files_of(a.paths):
        try:
            r = check(f, a.solve_limit)
            r["result"] = "ok" if r.get("finite", True) else "non-finite"
        except Exception as exc:  # report and carry on: the point is to see every file
            r = {"file": os.path.basename(f), "result": f"FAILED: {type(exc).__name__}: {exc}"[:160]}
        failed += r["result"] != "ok"
        rows.append(r)
        print(r, flush=True)
    cols = ["file", "format", "size_mb", "read_s", "dim", "cells", "faces", "zones", "min_oq", "repaired", "solve5_s", "result"]
    fmt = lambda v: f"{v:.3g}" if isinstance(v, float) else ("" if v is None else str(v))  # noqa: E731
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(fmt(r.get(c)) for c in cols) + " |" for r in rows]
    text = "\n".join(md)
    print(text)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write("# Fluent mesh corpus\n\nRead + quality + 5 SIMPLEC iterations with default boundary conditions "
                     "(`benchmarks/fluent_mesh_corpus.py`).\n\n" + text + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

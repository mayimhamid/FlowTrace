"""The Fluent .msh/.cas reader against every encoding the file format allows.

The meshes are written here by an emitter independent of ``flowtrace.mesh.fluent_msh`` (straight from
the format specification), in every combination of: ASCII / single / double precision binary nodes,
ASCII / binary faces, uniform / mixed face zones, uniform / mixed cell declarations with ASCII or
binary type lists, node zones split in several sections, 39 / 45 zone headers, upper-case hex, CRLF
line endings, gzip, comments with parentheses and quotes, case-file settings and unknown ASCII and
binary sections, and adapted meshes (parent faces, a face tree, dead and parent cells). Each must
read back to exactly the same mesh.
"""

import gzip
import itertools
import struct

import numpy as np
import pytest

from flowtrace.mesh.fluent_msh import FluentMeshError, _faces_from_ints, read_fluent_mesh
from flowtrace.mesh.unstructured import (agglomerate_cells, cartesian_mesh_2d, compute_geometry, extrude_to_3d,
                                         set_cell_zone)


def _meshes():
    quad = cartesian_mesh_2d(7, 5, 2.0, 1.0, names=("inlet", "outlet", "bottom", "top"),
                             types=("velocity-inlet", "pressure-outlet", "wall", "symmetry"), warp=0.04)
    tri = cartesian_mesh_2d(6, 4, triangulate=True, warp=0.03)
    c = np.arange(36)
    poly2d = agglomerate_cells(cartesian_mesh_2d(6, 6), (c // 6 // 2) * 10 + (c % 6) // 2)
    hexes = extrude_to_3d(cartesian_mesh_2d(4, 3, warp=0.05), 3, 0.6)
    wedges = extrude_to_3d(cartesian_mesh_2d(3, 3, triangulate=True), 2, 0.5)  # tri and quad faces: mixed zones
    two_zones = set_cell_zone(cartesian_mesh_2d(6, 4), lambda x: x[:, 0] < 0.5, "solid-block", "solid")
    return {"quad": quad, "tri": tri, "poly2d": poly2d, "hex": hexes, "wedge": wedges, "zones": two_zones}


MESHES = _meshes()


def emit(mesh, nodes="ascii", faces="ascii", cells="ascii", mixed=False, names=45, upper=False, crlf=False,
         split_nodes=False, extras=False, parents=False, dead=None) -> bytes:
    """A Fluent mesh file for ``mesh`` written from the format specification (not by FlowTrace's writer)."""
    dim, nn, nc = mesh.dim, mesh.n_nodes, mesh.n_cells
    hx = (lambda v: format(int(v), "X")) if upper else (lambda v: format(int(v), "x"))
    nl = "\r\n" if crlf else "\n"
    out = []
    a = lambda txt: out.append(txt.replace("\n", nl).encode())  # noqa: E731
    a('(0 "written by the test emitter (with (parentheses) and \\"quotes\\")")\n')
    a(f"(2 {dim})\n")
    if extras:
        a('(1 "Fluent 2025 R1")\n(33 (10 10 1))\n(37 (\n(time-step 0.01)\n(model (k-epsilon) "nested ( paren")\n))\n')
        out.append(b"(2041 (1 2 3)(" + bytes(range(256)) + b")End of Binary Section 2041)\n")
    a(f"(10 (0 1 {hx(nn)} 0 {dim}))\n")
    blocks = [(0, nn)] if not split_nodes else [(0, nn // 2), (nn // 2, nn)]
    for zi, (lo, hi) in enumerate(blocks):
        zid = 1 + zi
        head = f"(0 1 {hx(lo + 1)} {hx(hi)} 1 {dim})".replace("(0 1 ", f"({hx(zid)} ")
        xyz = mesh.nodes[lo:hi, :dim]
        if nodes == "ascii":
            a(f"(10 {head}(\n" + "\n".join(" ".join(f"{v:.17g}" for v in r) for r in xyz) + "\n))\n")
        else:
            code, dt = ("2010", "<f4") if nodes == "f32" else ("3010", "<f8")
            out.append(f"({code} {head}(".encode() + xyz.astype(dt).tobytes() + f")End of Binary Section {code})\n".encode())
    # cells: zone per cell zone (dead cells get their own type-0 zone)
    cz = mesh.cell_zone.copy()
    ctype = {z.id: (1 if z.type == "fluid" else 17) for z in mesh.cell_zones()}
    if dead is not None:
        cz = cz.copy()
        cz[dead] = 99
        ctype[99] = 0
    a(f"(12 (0 1 {hx(nc)} 0 0))\n")
    # Fluent numbers cells zone by zone: renumber so that every zone is contiguous
    order = np.argsort(cz, kind="stable")
    new = np.empty(nc, dtype=np.int64)
    new[order] = np.arange(nc)
    for zid in sorted(set(cz.tolist())):
        idx = np.nonzero(cz[order] == zid)[0]
        lo, hi = idx[0] + 1, idx[-1] + 1
        etype = 0 if mixed or mesh.dim == 3 or zid == 99 else (3 if mesh.dim == 2 else 4)
        head = f"({hx(zid)} {hx(lo)} {hx(hi)} {hx(ctype[zid])} {hx(etype)})"
        if etype != 0:
            a(f"(12 {head})\n")
        elif cells == "ascii":
            a(f"(12 {head}(\n" + " ".join(["7"] * (hi - lo + 1)) + "\n))\n")
        else:
            out.append(f"(2012 {head}(".encode() + np.full(hi - lo + 1, 7, "<i4").tobytes() + b")End of Binary Section 2012)\n")
    # faces: Fluent's 3-D node order has its right-hand normal toward c0 (the reverse of FlowTrace's)
    ptr = mesh.face_ptr
    fnodes = [mesh.face_nodes[ptr[f]:ptr[f + 1]] for f in range(mesh.n_faces)]
    if dim == 3:
        fnodes = [n[::-1] for n in fnodes]
    zones = sorted(mesh.face_zones(), key=lambda z: (z.type != "interior", z.id))
    total = mesh.n_faces + (int((mesh.face_zone == zones[-1].id).sum()) if parents else 0)
    a(f"(13 (0 1 {hx(total)} 0 0))\n")
    first = 1
    bc = {"interior": 2, "wall": 3, "pressure-inlet": 4, "pressure-outlet": 5, "symmetry": 7, "velocity-inlet": 10}
    face_ids = {}

    def face_block(zid, ztype, flist, bctype):
        nonlocal first
        nf = len(flist)
        cnt = {len(fnodes[f]) for f in flist}
        ftype = 0 if (mixed or len(cnt) > 1) else cnt.pop()
        head = f"({hx(zid)} {hx(first)} {hx(first + nf - 1)} {hx(bctype)} {hx(ftype)})"
        recs = []
        for f in flist:
            nd = [int(v) + 1 for v in fnodes[f]]
            c0 = int(new[mesh.owner[f]]) + 1
            c1 = int(new[mesh.neighbour[f]]) + 1 if mesh.neighbour[f] >= 0 else 0
            recs.append(([len(nd)] if ftype == 0 else []) + nd + [c0, c1])
        if faces == "ascii":
            a(f"(13 {head}(\n" + "\n".join(" ".join(hx(v) for v in r) for r in recs) + "\n))\n")
        else:
            flat = np.array([v for r in recs for v in r], "<i4")
            out.append(f"(2013 {head}(".encode() + flat.tobytes() + b")End of Binary Section 2013)\n")
        for k, f in enumerate(flist):
            face_ids.setdefault(f, first + k)
        first += nf

    for z in zones:
        face_block(z.id, z.type, list(np.nonzero(mesh.face_zone == z.id)[0]), bc.get(z.type, 3))
    if parents:  # an adapted mesh: a parent face zone duplicating the last boundary zone, and its face tree
        last = list(np.nonzero(mesh.face_zone == zones[-1].id)[0])
        p0 = first
        face_block(98, "parent", last, 31)
        kids = "\n".join(f"1 {hx(face_ids[f])}" for f in last)
        a(f"(59 ({hx(p0)} {hx(first - 1)} 62 {hx(zones[-1].id)})(\n{kids}\n))\n")
    named = [(z.id, z.type, z.name) for z in zones] + [(z.id, z.type, z.name) for z in mesh.cell_zones()]
    if dead is not None:
        named.append((99, "dead", "dead-cells"))
    if parents:
        named.append((98, "parent", "parent-faces"))
    for zid, ztype, zname in named:
        if names == 45:
            a(f"(45 ({zid} {ztype} {zname})())\n")
        else:
            a(f"(39 ({zid} {ztype} {zname} 1)(\n(velocity (constant . 1))\n(nested (a (b c)))\n))\n")
    return b"".join(out)


def same(a, b):
    assert (a.dim, a.n_cells, a.n_faces) == (b.dim, b.n_cells, b.n_faces)
    ga, gb = compute_geometry(a), compute_geometry(b)
    assert gb.flipped_faces == 0  # every face oriented out of its owner, as written
    assert np.allclose(np.sort(ga.cell_volume), np.sort(gb.cell_volume), rtol=1e-6)
    za = sorted((z.name, z.type, z.count) for z in a.zones.values())
    zb = sorted((z.name, z.type, z.count) for z in b.zones.values())
    assert za == zb


ENCODINGS = list(itertools.product(["ascii", "f64"], ["ascii", "binary"], [False, True]))


@pytest.mark.parametrize("kind", sorted(MESHES))
@pytest.mark.parametrize("nodes,faces,mixed", ENCODINGS)
def test_every_encoding_reads_back_the_same_mesh(kind, nodes, faces, mixed):
    m = MESHES[kind]
    same(m, read_fluent_mesh(emit(m, nodes=nodes, faces=faces, cells="binary" if faces == "binary" else "ascii",
                                  mixed=mixed)))


@pytest.mark.parametrize("variant", ["f32", "split", "upper", "crlf", "names39", "extras", "gzip", "all"])
def test_format_variants(variant):
    m = MESHES["hex"]
    kw = {"f32": dict(nodes="f32"), "split": dict(split_nodes=True), "upper": dict(upper=True), "crlf": dict(crlf=True),
          "names39": dict(names=39), "extras": dict(extras=True), "gzip": {},
          "all": dict(nodes="f32", split_nodes=True, upper=True, crlf=True, names=39, extras=True, mixed=True)}[variant]
    raw = emit(m, **kw)
    if variant in ("gzip", "all"):
        raw = gzip.compress(raw)
    got = read_fluent_mesh(raw)
    if variant in ("f32", "all"):  # single precision nodes: volumes to ~1e-6
        assert got.n_cells == m.n_cells and np.allclose(np.sort(compute_geometry(got).cell_volume),
                                                        np.sort(compute_geometry(m).cell_volume), rtol=1e-5)
    else:
        same(m, got)


@pytest.mark.parametrize("kind", ["quad", "hex"])
def test_adapted_mesh_parent_faces_and_face_tree_are_dropped(kind):
    m = MESHES[kind]
    same(m, read_fluent_mesh(emit(m, parents=True)))
    same(m, read_fluent_mesh(emit(m, parents=True, faces="binary")))


def test_dead_cells_are_removed_and_their_faces_become_boundaries():
    m = MESHES["quad"]
    xc = compute_geometry(m).cell_centre
    dead = np.nonzero(xc[:, 0] < 0.6)[0]
    got = read_fluent_mesh(emit(m, dead=dead))
    keep = np.setdiff1d(np.arange(m.n_cells), dead)
    assert got.n_cells == keep.size and compute_geometry(got).flipped_faces == 0
    assert np.allclose(np.sort(compute_geometry(got).cell_volume), np.sort(compute_geometry(m).cell_volume[keep]))
    assert "dead-cells" not in {z.name for z in got.zones.values()}
    live = np.zeros(m.n_cells, bool)
    live[keep] = True
    nb = np.where(m.neighbour >= 0, live[np.maximum(m.neighbour, 0)], False)
    expected = int((live[m.owner] ^ nb).sum())  # faces with exactly one live cell
    assert (got.neighbour < 0).sum() == expected
    walls = got.zones[max(z.id for z in got.zones.values() if z.name == "wall-exposed")]
    assert walls.type == "wall" and walls.count > 0  # the cut through the interior became a wall zone


def test_polygonal_face_records_up_to_many_nodes():
    recs = [[5, 1, 2, 3, 4, 5, 7, 8], [3, 4, 5, 6, 8, 0], [9] + list(range(10, 19)) + [1, 2]]
    flat = np.array([v for r in recs for v in r])
    counts, nodes, c0, c1, used = _faces_from_ints(flat, 3, 0)
    assert counts.tolist() == [5, 3, 9] and used == flat.size
    assert nodes.tolist() == [1, 2, 3, 4, 5, 4, 5, 6] + list(range(10, 19))
    assert c0.tolist() == [7, 8, 1] and c1.tolist() == [8, 0, 2]


def test_truncated_and_garbage_files_fail_with_a_clear_error():
    raw = emit(MESHES["quad"], faces="binary")
    for bad in (raw[: len(raw) // 2], b"hello world", b"(2 2)(10 (0 1 4 0 2))"):
        with pytest.raises(FluentMeshError):
            read_fluent_mesh(bad)


def test_binary_struct_layout_is_little_endian_int32():
    """Fluent binary sections are little-endian 32-bit integers and IEEE floats: one face record checks it."""
    raw = emit(MESHES["quad"], faces="binary")
    i = raw.index(b"(2013 (")
    j = raw.index(b")(", i) + 2
    assert struct.unpack("<4i", raw[j:j + 16])[2] >= 1  # n0 n1 c0 c1 with c0 a 1-based cell

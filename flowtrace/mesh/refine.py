"""Uniform refinement of 2-D meshes of arbitrary polygons.

Every edge is split at its midpoint and every cell with k corners becomes k quadrilaterals
(corner, edge midpoint, cell centroid, edge midpoint): a quad gives the usual 4 quads, a
triangle 3, a polygon one per corner. The refined mesh covers exactly the same polygonal
domain with the same boundary zones, so two solutions differ by discretisation error only,
which is what a Richardson estimate needs.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .unstructured import UnstructuredMesh, Zone, _renumber_cells, compute_geometry, sort_faces_by_zone


def refine_2d(mesh: UnstructuredMesh) -> Tuple[UnstructuredMesh, np.ndarray]:
    """Return ``(fine mesh, parent)``; ``parent[i]`` is the cell of ``mesh`` that fine cell i lies in."""
    if mesh.dim != 2:
        raise NotImplementedError("uniform refinement is implemented for 2-D meshes")
    if mesh.periodic:
        raise NotImplementedError("refining a mesh with paired periodic zones is not supported")
    if np.any(np.diff(mesh.face_ptr) != 2):
        raise ValueError("a 2-D mesh must have two nodes per face")
    nn, nf, nc = mesh.n_nodes, mesh.n_faces, mesh.n_cells
    geo = compute_geometry(mesh)
    n0 = mesh.face_nodes[mesh.face_ptr[:-1]]
    n1 = mesh.face_nodes[mesh.face_ptr[:-1] + 1]
    mid = nn + np.arange(nf)
    cen = nn + nf + np.arange(nc)
    nodes = np.concatenate([mesh.nodes, 0.5 * (mesh.nodes[n0] + mesh.nodes[n1]), geo.cell_centre[:, :2]])

    # fine cell of (coarse cell, corner node): its position in the cell's node loop
    ptr, loops = mesh.cell_polygons_2d()
    cell_of = np.repeat(np.arange(nc), np.diff(ptr))
    keys = cell_of * (nn + 1) + loops
    order = np.argsort(keys)
    skeys = keys[order]

    def sub(c, v):
        k = c * (nn + 1) + v
        pos = np.searchsorted(skeys, k)
        if np.any(pos >= skeys.size) or np.any(skeys[np.minimum(pos, skeys.size - 1)] != k):
            raise ValueError("inconsistent cell connectivity (a face node is not a corner of its cell)")
        return order[pos]

    P, N = mesh.owner, mesh.neighbour
    inner = N >= 0
    Ni = N[inner]
    # each coarse face in two halves; the direction (owner on the left) is kept
    fa = np.stack([n0, mid], axis=1)
    fb = np.stack([mid, n1], axis=1)
    own_a, own_b = sub(P, n0), sub(P, n1)
    nei_a = np.full(nf, -1, dtype=np.int64)
    nei_b = np.full(nf, -1, dtype=np.int64)
    nei_a[inner] = sub(Ni, n0[inner])
    nei_b[inner] = sub(Ni, n1[inner])
    # new faces edge midpoint -> centroid; walking that way the corner behind the edge direction is on the left
    fo = np.stack([mid, cen[P]], axis=1)
    fn = np.stack([mid[inner], cen[Ni]], axis=1)
    interior = [z for z in mesh.face_zones() if z.type == "interior"]
    zones = {k: Zone(z.id, z.name, z.type, z.kind) for k, z in mesh.zones.items()}
    if interior:
        zin = interior[0].id
    else:
        zin = max(zones) + 1
        zones[zin] = Zone(zin, "interior-refined", "interior", "face")
    faces = np.concatenate([fa, fb, fo, fn]).astype(np.int64)
    owner = np.concatenate([own_a, own_b, sub(P, n0), sub(Ni, n1[inner])]).astype(np.int64)
    neigh = np.concatenate([nei_a, nei_b, sub(P, n1), sub(Ni, n0[inner])]).astype(np.int64)
    fzone = np.concatenate([mesh.face_zone, mesh.face_zone, np.full(nf + int(inner.sum()), zin)]).astype(np.int64)
    cz = mesh.cell_zone[cell_of]
    out = UnstructuredMesh(dim=2, nodes=nodes, face_ptr=np.arange(0, 2 * faces.shape[0] + 1, 2, dtype=np.int64),
                           face_nodes=faces.ravel(), owner=owner, neighbour=neigh, face_zone=fzone,
                           n_cells=int(loops.size), cell_zone=cz, zones=zones, name=mesh.name + "-refined")
    # _renumber_cells sorts cells by zone (stable): apply the same permutation to the parent map
    perm = np.argsort(cz, kind="stable")
    out = sort_faces_by_zone(_renumber_cells(out, cz, out.zones))
    return out, cell_of[perm]

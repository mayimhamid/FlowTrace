"""Uniform 2-D refinement and the mesh certificate (error bands from refined solves)."""

import os

import numpy as np
import pytest

from flowtrace.mesh.fluent_h5 import read_any_mesh
from flowtrace.mesh.refine import refine_2d
from flowtrace.mesh.unstructured import cartesian_mesh_2d, compute_geometry, extrude_to_3d
from flowtrace.runtime.engine import EventHub, Job
from flowtrace.unstructured.case import BoundarySpec, UnstructuredCase
from flowtrace.unstructured.certify import certify, error_band, format_certificate
from flowtrace.unstructured.simple import SimpleSolver

ELBOW = os.path.join(os.path.dirname(__file__), "..", "examples", "elbow-mix.msh.txt")


@pytest.mark.parametrize("make, factor", [(lambda: cartesian_mesh_2d(5, 4, 2.0, 1.0), 4),
                                          (lambda: cartesian_mesh_2d(4, 4, triangulate=True), 3),
                                          (lambda: read_any_mesh(ELBOW)[0], 4)])
def test_refine_2d_keeps_the_domain_the_zones_and_every_parent_volume(make, factor):
    mesh = make()
    fine, parent = refine_2d(mesh)
    g0, g1 = compute_geometry(mesh), compute_geometry(fine)
    assert fine.n_cells == factor * mesh.n_cells and g1.cell_volume.min() > 0
    assert np.allclose(np.bincount(parent, weights=g1.cell_volume), g0.cell_volume)
    for z in mesh.boundary_zones():  # every boundary zone: twice the faces, the same length
        f0 = np.nonzero((mesh.neighbour < 0) & (mesh.face_zone == z.id))[0]
        f1 = np.nonzero((fine.neighbour < 0) & (fine.face_zone == fine.zone_by_name(z.name).id))[0]
        assert f1.size == 2 * f0.size and g1.face_area[f1].sum() == pytest.approx(g0.face_area[f0].sum())
    with pytest.raises(NotImplementedError):
        refine_2d(extrude_to_3d(cartesian_mesh_2d(2, 2), 2, 1.0))


def test_error_band_follows_the_convergence_of_the_sequence():
    exact, r = 1.0, 2.0
    second = [exact + 0.1 * h ** 2 for h in (1.0, 0.5, 0.25)]
    b = error_band(second, r, 2.0)
    assert b["status"] == "asymptotic" and b["order"] == pytest.approx(2.0)
    assert abs(second[-1] - exact) <= b["band"] <= 2.0 * abs(second[-1] - exact)  # 1.25 x the Richardson error
    two = error_band(second[1:], r, 2.0)
    assert two["status"] == "two-level" and two["band"] == pytest.approx(3.0 * abs(second[-1] - exact))
    assert error_band([1.0, 1.2, 1.1], r, 2.0)["status"] == "oscillating"
    grow = error_band([1.0, 1.1, 1.3], r, 2.0)
    assert grow["status"] == "not converging" and grow["band"] == pytest.approx(0.6)
    assert error_band([1.0, 1.4, 1.7], r, 2.0)["status"] == "slow"
    assert error_band([2.0, 2.0, 2.0], r, 2.0) == {"value": 2.0, "band": 0.0, "order": None, "status": "converged"}


def _cavity(n):
    case = UnstructuredCase(boundaries={"top": BoundarySpec("wall", velocity=[1.0, 0.0, 0.0])}, viscosity=0.01,
                            max_iter=3000, tol=1e-7, backend="numpy")
    s = SimpleSolver(cartesian_mesh_2d(n, n), case)
    while not s.finished:
        s.step()
    return s


def test_certificate_band_covers_a_much_finer_solution_and_flags_the_singular_lid_force():
    cert = certify(_cavity(12), levels=3)
    ke = cert["quantities"]["mean kinetic energy"]
    assert cert["cells"] == [144, 576, 2304] and all(cert["converged"])
    assert ke["status"] == "asymptotic" and 1.3 < ke["order"] < 2.5
    ref = _cavity(96).fields()
    ke_ref = float(np.mean(0.5 * (ref["x-velocity"] ** 2 + ref["y-velocity"] ** 2)))  # uniform cells
    assert abs(ke["value"] - ke_ref) <= ke["band"]
    # the lid meets the side walls with a velocity jump: its shear force grows without bound as the mesh
    # is refined, and the certificate must not give it a small band
    lid = cert["quantities"]["force x on top"]
    assert lid["status"] != "asymptotic" and lid["relative"] > 0.2
    assert cert["error_map"].shape == (2304,) and cert["solver"].nc == 2304
    assert any("mean kinetic energy" in line for line in format_certificate(cert))


def test_job_with_certify_reports_the_certificate_on_the_refined_mesh(tmp_path):
    mesh = cartesian_mesh_2d(8, 8)
    case = {"boundaries": {"top": {"kind": "wall", "velocity": [1.0, 0.0, 0.0]}}, "viscosity": 0.01, "max_iter": 2000,
            "tol": 1e-6, "backend": "numpy", "lanes": "off"}
    job = Job({"kind": "unstructured", "mesh_id": "m", "case": case, "certify": 2}, EventHub(), str(tmp_path),
              meshes={"m": mesh})
    job.start()
    job.join(300)
    assert job.error is None and job.result["certificate"]["cells"] == [64, 256]
    assert job.solver.nc == 256 and job.mesh.n_cells == 256
    frame = job.unstructured_frame(job.solver, "mesh-error-estimate")
    assert frame["field"] == "mesh-error-estimate" and frame["cells"] == 256 and "u" in frame

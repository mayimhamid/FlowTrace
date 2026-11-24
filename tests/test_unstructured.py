import numpy as np
import pytest
from scipy.interpolate import griddata

from flowtrace.hardware.backend import get_backend
from flowtrace.mesh.unstructured import cartesian_mesh_2d, extrude_to_3d
from flowtrace.unstructured.case import BoundarySpec, UnstructuredCase, default_boundaries
from flowtrace.unstructured.simple import SimpleSolver
from flowtrace.validation.reference import compare_ghia

be = get_backend("numpy")


def run(s):
    while not s.finished:
        s.step()
    return s


def test_skew_symmetric_momentum_does_no_internal_work():
    mesh = cartesian_mesh_2d(5, 4, triangulate=True, jitter=0.15)
    solver = SimpleSolver(mesh, UnstructuredCase(scheme="skew-symmetric"), be)
    fv = solver.fv
    rng = np.random.default_rng(42)
    flux = rng.normal(size=fv.n_if)
    boundary_flux = np.zeros(fv.n_bf)
    assert np.linalg.norm(fv.net_outflow(flux, boundary_flux)) > 0.1
    diag, a_pn, a_np, _ = solver._coefficients(
        0.0, solver.m_dirvel, (flux, boundary_flux), skew_symmetric=True
    )
    operator = fv.matrix(diag, a_pn, a_np)
    velocity = rng.normal(size=(solver.nc, solver.dim))
    np.testing.assert_allclose(operator.toarray(), -operator.toarray().T, atol=1e-14)
    assert abs(np.sum(velocity * (operator @ velocity))) < 1e-13

    smooth_velocity = solver.cell_centres()
    linear_gradients = [np.broadcast_to(row, smooth_velocity.shape) for row in np.eye(solver.dim)]
    np.testing.assert_allclose(solver._skew_upwind_diffusion(smooth_velocity, linear_gradients, flux), 0.0)

    unresolved_gradients = [np.zeros_like(velocity) for _ in range(solver.dim)]
    diffusion = solver._skew_upwind_diffusion(velocity, unresolved_gradients, flux)
    diag, a_pn, a_np, _ = solver._coefficients(
        0.0, solver.m_dirvel, (flux, boundary_flux), skew_symmetric=True, upwind_diffusion=diffusion
    )
    dissipative_operator = fv.matrix(diag, a_pn, a_np)
    actual = np.sum(velocity * (dissipative_operator @ velocity))
    expected = np.sum(diffusion * np.sum((velocity[fv.N] - velocity[fv.P])**2, axis=1))
    assert actual == pytest.approx(expected)
    assert actual > 0.0


@pytest.mark.parametrize("time_scheme", ["euler", "bdf2"])
def test_transient_momentum_forcing_from_rest_in_3d(time_scheme):
    ny, nz = 8, 4
    mesh = extrude_to_3d(cartesian_mesh_2d(3, ny), nz, 1.0)
    rho, nu, dt = 2.5, 0.04, 0.025
    case = UnstructuredCase(
        scheme="skew-symmetric", density=rho, viscosity=rho * nu,
        periodic=[["left", "right"], ["bottom", "top"]],
        unsteady=True, time_scheme=time_scheme, dt=dt, t_end=4 * dt,
        inner_iterations=35, inner_tol=1e-10, pressure_tol=1e-7, lanes="off",
    )
    solver = SimpleSolver(mesh, case, be)
    np.testing.assert_array_equal(solver.U, 0.0)
    sampled_times = []

    def force(centres, time):
        sampled_times.append(time)
        source = np.zeros_like(centres)
        mode = np.sin(2 * np.pi * centres[:, 1]) * np.cos(2 * np.pi * centres[:, 2])
        source[:, 0] = rho * time * (mode + 0.25)
        return source

    solver.set_sources(momentum=force)
    run(solver)
    times = dt * np.arange(1, 5)
    np.testing.assert_allclose(sampled_times, times)
    np.testing.assert_allclose(solver.mom_source.sum(axis=0), 0.0, atol=1e-14)

    # Periodic in y and zero wall-normal derivative at the z symmetry planes.
    # The cell-centred Fourier mode has this exact finite-volume decay rate.
    decay = 4 * nu * (ny**2 * np.sin(np.pi / ny)**2 + nz**2 * np.sin(np.pi / nz)**2)
    previous = older = 0.0
    for step, time in enumerate(times):
        if step and time_scheme == "bdf2":
            amplitude = (dt * time + 2 * previous - 0.5 * older) / (1.5 + dt * decay)
        else:
            amplitude = (dt * time + previous) / (1 + dt * decay)
        previous, older = amplitude, previous
    _, y, z = solver.cell_centres().T
    np.testing.assert_allclose(solver.U[:, 0], amplitude * np.sin(2 * np.pi * y) * np.cos(2 * np.pi * z),
                               rtol=1e-8, atol=1e-11)
    np.testing.assert_allclose(solver.U[:, 1:], 0.0, atol=1e-11)
    np.testing.assert_allclose(solver.fv.net_outflow(solver.F, solver.Fb), 0.0, atol=1e-11)


def test_turbulent_pressure_is_absorbed_before_the_momentum_predictor():
    mesh = cartesian_mesh_2d(6, 4)
    case = UnstructuredCase(turbulence="k-omega-sst", density=2.0, max_iter=1)
    solver = SimpleSolver(mesh, case, be)
    solver.turb.k = 1.0 + solver.cell_centres()[:, 0]
    solver.p = -(2.0 / 3.0) * case.density * solver.turb.k
    solver.step()
    np.testing.assert_allclose(solver.U, 0.0, atol=1e-12)


def _ghia_err(s):
    xc, f = s.cell_centres(), s.fields()
    y = np.linspace(0.02, 0.98, 49)
    u = griddata(xc[:, :2], f["x-velocity"], np.column_stack([np.full_like(y, 0.5), y]))
    v = griddata(xc[:, :2], f["y-velocity"], np.column_stack([y, np.full_like(y, 0.5)]))
    return compare_ghia(100, y, u, y, v)


def test_simplec_cavity_skewed_triangles():
    m = cartesian_mesh_2d(32, 32, triangulate=True, jitter=0.2)
    c = UnstructuredCase(boundaries={"top": BoundarySpec("wall", velocity=[1.0, 0.0])}, viscosity=0.01,
                         max_iter=3000, tol=1e-6)
    s = run(SimpleSolver(m, c, be))
    assert s.converged
    e = _ghia_err(s)
    assert e["u_max_err"] < 0.03 and e["v_max_err"] < 0.03, e


def test_simple_algorithm_converges_too():
    m = cartesian_mesh_2d(24, 24)
    c = UnstructuredCase(boundaries={"top": BoundarySpec("wall", velocity=[1.0, 0.0])}, viscosity=0.01,
                         algorithm="SIMPLE", max_iter=4000, tol=1e-5)
    assert c.relax_p == 0.3
    s = run(SimpleSolver(m, c, be))
    assert s.converged


def test_channel_mass_balance_and_heat():
    m = cartesian_mesh_2d(60, 12, 5.0, 1.0, names=("inlet", "outlet", "bottom", "top"),
                          types=("velocity-inlet", "pressure-outlet", "wall", "wall"))
    bnd = default_boundaries(m)
    assert bnd["inlet"].kind == "inlet" and bnd["outlet"].kind == "outlet"
    bnd["inlet"] = BoundarySpec("inlet", normal_speed=1.0, value=0.0)
    bnd["bottom"] = BoundarySpec("wall", thermal="temperature", value=1.0)
    c = UnstructuredCase(boundaries=bnd, viscosity=0.02, energy=True, conductivity=0.02, max_iter=1500, tol=1e-6)
    s = run(SimpleSolver(m, c, be))
    assert s.converged
    mb = s.mass_balance()
    assert abs(mb["inlet"] + 1.0) < 1e-9 and abs(sum(mb.values())) < 1e-8
    T = s.fields()["temperature"]
    assert -1e-8 <= T.min() and T.max() <= 1.0 + 1e-6


def test_3d_extruded_cavity_is_two_dimensional():
    m = extrude_to_3d(cartesian_mesh_2d(16, 16), 2, 0.2)
    c = UnstructuredCase(boundaries={"top": BoundarySpec("wall", velocity=[1.0, 0.0, 0.0])}, viscosity=0.01,
                         max_iter=400, tol=1e-6)
    s = run(SimpleSolver(m, c, be))
    f = s.fields()
    assert np.abs(f["z-velocity"]).max() < 1e-8
    assert f["x-velocity"].min() < -0.15
    r = s.residuals()["scaled"]
    assert r["z-velocity"] < 1e-3


def test_wlsq_over_relaxed_flux_and_limiter_on_distorted_mesh():
    from flowtrace.unstructured.fvmesh import FVMesh

    mesh = cartesian_mesh_2d(9, 7, triangulate=True, warp=0.06)
    fv = FVMesh(mesh, be, cell_order="rcm")
    gradient = np.array([2.2, -1.4])
    phi, phib = fv.xc @ gradient + 3.0, fv.xfb @ gradient + 3.0
    reconstructed = fv.gradient(phi, phib)
    np.testing.assert_allclose(reconstructed, np.broadcast_to(gradient, reconstructed.shape), atol=5e-13)
    flux = fv.alpha * (phi[fv.N] - phi[fv.P]) + np.sum(fv.face_interp_vec(reconstructed) * fv.Tvec, axis=1)
    np.testing.assert_allclose(flux, fv.S @ gradient, atol=1e-13)
    constant = np.ones(fv.nc)
    flat = fv.venkat_limiter(constant, np.ones(fv.n_bf), np.zeros_like(reconstructed), 0.0)
    np.testing.assert_array_equal(flat, 1.0)
    limited = fv.venkat_limiter(phi, phib, reconstructed * 100.0, 0.0)
    assert np.all((limited >= 0.0) & (limited < 1.0))


def test_rcm_fluent_cell_data_roundtrip(tmp_path):
    from flowtrace.mesh.fluent_h5 import read_data_h5, write_data_h5
    from flowtrace.mesh.unstructured import compute_geometry

    mesh = cartesian_mesh_2d(9, 6)
    case = UnstructuredCase(cell_order="rcm", energy=True, conductivity=0.02,
                            boundaries={"top": BoundarySpec("wall", velocity=[1.0, 0.0]),
                                        "left": BoundarySpec("wall", thermal="temperature", value=1.0),
                                        "right": BoundarySpec("wall", thermal="temperature", value=0.0)},
                            max_iter=1200, tol=1e-8, lanes="off")
    solver = SimpleSolver(mesh, case, be)
    perm, inv = solver.fv.mesh.perm, solver.fv.mesh.inv_perm
    np.testing.assert_array_equal(inv[perm], np.arange(mesh.n_cells))
    assert np.any(perm != np.arange(mesh.n_cells))
    centres = compute_geometry(mesh).cell_centre
    np.testing.assert_array_equal(solver.cell_centres(), centres)
    solver.initialize({"temperature": centres[:, 0] + 2.0 * centres[:, 1]})
    np.testing.assert_array_equal(solver.fields(["temperature"])["temperature"], centres[:, 0] + 2.0 * centres[:, 1])
    run(solver)
    assert solver.converged
    fields = solver.fields(["temperature", "pressure", "x-velocity"])
    path = tmp_path / "rcm.dat.h5"
    write_data_h5(str(path), solver.mesh, fields)
    exported = read_data_h5(str(path), mesh.n_cells)
    for key, values in fields.items():
        np.testing.assert_array_equal(exported[key], values)
    np.testing.assert_array_equal(exported["temperature"], solver.T[inv])


def test_mixed_precision_recovers_true_fp64_residual():
    from scipy.sparse import diags
    from flowtrace.linalg.krylov import refined_krylov

    diagonal = np.geomspace(1.0, 1e5, 36)
    matrix = diags([-0.35 * diagonal[1:], diagonal, 0.15 * diagonal[:-1]], [-1, 0, 1], format="csr")
    exact = np.sin(np.arange(36))
    rhs = matrix @ exact
    result, info = refined_krylov(matrix, rhs, np, lambda x: x / diagonal.astype(x.dtype),
                                  None, 1e-11, 60)
    assert info.converged
    assert np.linalg.norm(rhs - matrix @ result) / np.linalg.norm(rhs) < 1e-11
    np.testing.assert_allclose(result, exact, atol=1e-7)


def test_convection_wall_conserves_analytic_slab_heat_flow():
    mesh = cartesian_mesh_2d(8, 3)
    case = UnstructuredCase(energy=True, conductivity=2.0, max_iter=500, tol=1e-14, lanes="off",
                            boundaries={
                                "left": BoundarySpec(thermal="temperature", value=1.0),
                                "right": BoundarySpec(thermal="convection", value=0.0, heat_transfer_coefficient=2.0),
                            })
    solver = run(SimpleSolver(mesh, case, be))
    assert solver.converged
    expected_flux = 1.0 / (1.0 / case.conductivity + 1.0 / 2.0)
    values = solver.heat_flow()
    np.testing.assert_allclose([values["left"], values["right"]], [expected_flux, -expected_flux], atol=1e-12, rtol=0)
    assert abs(sum(values.values())) < 1e-12
    exact = 1.0 - expected_flux * solver.cell_centres()[:, 0] / case.conductivity
    np.testing.assert_allclose(solver.fields()["temperature"], exact, atol=1e-12, rtol=0)


def test_consistent_rhie_chow_cavity_relaxation_invariance():
    mesh = cartesian_mesh_2d(16, 16)
    solutions = []
    for relax_u in (0.3, 0.7):
        case = UnstructuredCase(rhie_chow="consistent", viscosity=0.001, relax_u=relax_u,
                                pressure_tol=0.001, tol=1e-9, max_iter=4500, lanes="off",
                                boundaries={"top": BoundarySpec("wall", velocity=[1.0, 0.0])})
        solver = run(SimpleSolver(mesh, case, be))
        assert solver.converged
        solutions.append(solver.fields(["x-velocity", "y-velocity", "pressure"]))
    for name, values in solutions[0].items():
        np.testing.assert_allclose(values, solutions[1][name], rtol=0, atol=1e-6)


def test_open_heated_duct_closes_mass_and_energy_balances():
    mesh = cartesian_mesh_2d(18, 8, lx=4.0, names=("inlet", "outlet", "bottom", "top"),
                             types=("velocity-inlet", "pressure-outlet", "wall", "wall"))
    case = UnstructuredCase(energy=True, init_T=1.0, conductivity=0.03, viscosity=0.05,
                            pressure_tol=0.001, tol=1e-13, max_iter=2500, lanes="off",
                            boundaries={"inlet": BoundarySpec("inlet", normal_speed=1.0, value=1.0),
                                        "outlet": BoundarySpec("outlet"),
                                        "bottom": BoundarySpec(thermal="temperature", value=2.0)})
    solver = run(SimpleSolver(mesh, case, be))
    assert solver.converged
    for balance in (solver.mass_balance(), solver.heat_flow()):
        assert abs(sum(balance.values())) / abs(balance["inlet"]) < 1e-12

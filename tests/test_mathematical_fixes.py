import numpy as np
import pytest

from flowtrace.linalg.line_preconditioner import LineImplicitPreconditioner
from flowtrace.unstructured.simple import NonOrthogonalPressureCorrector
from flowtrace.compressible.solver import WeissSmithLowMachPreconditioner
from flowtrace.unstructured.turbulence import (
    spaldings_law_wall_shear,
    linearize_kepsilon_sources,
    assemble_and_relax_scalar_system,
)
from flowtrace.unstructured.certify import compute_gci_certificate


def test_line_implicit_preconditioner():
    n_cells = 8
    # Two lines of 4 cells each
    lines = [np.array([0, 1, 2, 3], dtype=np.int32), np.array([4, 5, 6, 7], dtype=np.int32)]
    prec = LineImplicitPreconditioner(n_cells, lines, backend="cpu")

    # Construct a tridiagonal CSR matrix representing -d^2/dy^2
    # Line 1: cells 0, 1, 2, 3; Line 2: cells 4, 5, 6, 7
    diag = np.full(n_cells, 2.0, dtype=np.float64)
    indptr = [0]
    indices = []
    data = []

    for i in range(n_cells):
        line_start = (i // 4) * 4
        line_end = line_start + 4
        # lower
        if i > line_start:
            indices.append(i - 1)
            data.append(-1.0)
        # diag
        indices.append(i)
        data.append(2.0)
        # upper
        if i < line_end - 1:
            indices.append(i + 1)
            data.append(-1.0)
        indptr.append(len(indices))

    csr_indptr = np.array(indptr, dtype=np.int32)
    csr_indices = np.array(indices, dtype=np.int32)
    csr_data = np.array(data, dtype=np.float64)

    prec.update(csr_indptr, csr_indices, csr_data, diag)

    # Test apply
    r = np.ones(n_cells, dtype=np.float64)
    z = prec.apply(r)
    assert z.shape == (n_cells,)
    assert not np.isnan(z).any()
    # Check that T @ z ≈ r
    # For line 1: T_k = [[2, -1, 0, 0], [-1, 2, -1, 0], [0, -1, 2, -1], [0, 0, -1, 2]]
    Tk = np.array([
        [2.0, -1.0, 0.0, 0.0],
        [-1.0, 2.0, -1.0, 0.0],
        [0.0, -1.0, 2.0, -1.0],
        [0.0, 0.0, -1.0, 2.0],
    ])
    z1 = z[:4]
    np.testing.assert_allclose(Tk @ z1, r[:4], rtol=1e-10)


def test_non_orthogonal_pressure_corrector():
    # 2 cells, 1 interior face with 45 degree skew
    owner = np.array([0], dtype=np.int64)
    neighbor = np.array([1], dtype=np.int64)
    Sf = np.array([[1.0, 1.0, 0.0]], dtype=np.float64)
    d_vec = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    weights = np.array([0.5], dtype=np.float64)
    n_cells = 2

    def grad_operator(p):
        g = np.zeros((n_cells, 3), dtype=np.float64)
        dp = p[1] - p[0]
        g[0, 0] = dp
        g[1, 0] = dp
        return g

    corr = NonOrthogonalPressureCorrector(
        owner, neighbor, Sf, d_vec, weights, n_cells, grad_operator, n_non_ortho_correctors=2
    )
    assert corr.alpha_f[0] > 0
    assert np.linalg.norm(corr.T_f[0]) > 0

    F_star = np.array([0.2], dtype=np.float64)
    rho_d_f = np.array([0.1], dtype=np.float64)

    def mock_amg(rhs, p0):
        # Solves diag * (p0 - p1) = rhs
        A_val = rho_d_f[0] * corr.alpha_f[0]
        p = np.zeros(2, dtype=np.float64)
        p[0] = -rhs[0] / (2.0 * A_val)
        p[1] = -p[0]
        return p

    p_prime, grad_p, F_corr = corr.solve_pressure_correction(F_star, rho_d_f, mock_amg)
    assert p_prime.shape == (n_cells,)
    assert grad_p.shape == (n_cells, 3)
    assert F_corr.shape == (1,)


def test_weiss_smith_low_mach_preconditioner():
    ws = WeissSmithLowMachPreconditioner(gamma=1.4)
    rho = np.array([1.2, 1.2])
    u = np.array([10.0, 0.1])  # 10 m/s and 0.1 m/s
    v = np.zeros(2)
    w = np.zeros(2)
    p = np.array([101325.0, 101325.0])
    mu = np.full(2, 1.8e-5)
    dx = np.full(2, 0.01)

    beta, c = ws.compute_beta(rho, u, v, w, p, mu, dx, u_ref_global=10.0)
    assert np.all(beta > 0)
    assert np.all(beta <= 1.0)
    assert beta[1] < beta[0]  # lower velocity has lower beta

    # Test precondition_residual
    U = np.zeros((2, 5))
    U[:, 0] = rho
    U[:, 1] = rho * u
    U[:, 4] = p / 0.4 + 0.5 * rho * u**2
    res = np.ones((2, 5))
    prec_res = ws.precondition_residual(U, res, beta)
    assert prec_res.shape == (2, 5)
    assert not np.isnan(prec_res).any()


def test_spaldings_law_wall_shear():
    u_tan = np.array([1.0, 10.0, 25.0])
    y_wall = np.array([1e-4, 1e-3, 1e-2])
    rho = np.full(3, 1.2)
    nu = np.full(3, 1.5e-5)

    tau_w, u_tau, y_plus = spaldings_law_wall_shear(u_tan, y_wall, rho, nu)
    assert np.all(tau_w > 0)
    assert np.all(u_tau > 0)
    assert np.all(y_plus > 0)
    # y+ should grow as y_wall grows
    assert y_plus[0] < y_plus[1] < y_plus[2]


def test_linearize_kepsilon_sources():
    rho = np.array([1.2, 1.2])
    nu = np.array([1.5e-5, 1.5e-5])
    k = np.array([0.1, 0.5])
    eps = np.array([1.0, 2.0])
    P_k = np.array([0.5, 1.5])
    S_mag = np.array([10.0, 20.0])

    Sc_k, Sp_k, Sc_eps, Sp_eps = linearize_kepsilon_sources(
        rho, nu, k, eps, P_k, S_mag, model="realizable"
    )
    # Patankar rule: Sc >= 0, Sp <= 0
    assert np.all(Sc_k >= 0)
    assert np.all(Sp_k <= 0)
    assert np.all(Sc_eps >= 0)
    assert np.all(Sp_eps <= 0)

    # Standard model
    Sc_k, Sp_k, Sc_eps, Sp_eps = linearize_kepsilon_sources(
        rho, nu, k, eps, P_k, S_mag, model="standard"
    )
    assert np.all(Sc_k >= 0)
    assert np.all(Sp_k <= 0)
    assert np.all(Sc_eps >= 0)
    assert np.all(Sp_eps <= 0)


def test_compute_gci_certificate():
    # Asymptotic second-order refinement: e ~ h^2
    # h1 = 1, h2 = 2, h3 = 4
    # f(h) = 10 + 2 * h^2 => f1 = 12, f2 = 18, f3 = 42
    f1, f2, f3 = 12.0, 18.0, 42.0
    r21, r32 = 2.0, 2.0
    report = compute_gci_certificate(f1, f2, f3, r21, r32, formal_order=2.0)
    assert abs(report.observed_order - 2.0) < 1e-4
    assert abs(report.extrapolated_value - 10.0) < 1e-4
    assert report.convergence_regime == "ASYMPTOTIC_MONOTONIC"

"""Backend-agnostic Krylov solvers (NumPy or CuPy arrays, callable operators).

Only the scalars needed for the iteration control (dot products, norms) are
transferred to the host; vectors stay on the device.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class KrylovInfo:
    iterations: int
    residual: float  # final relative residual ||b - A x|| / ||b||
    converged: bool


def _dot(xp, a, b):
    return float(xp.vdot(a, b).real)


def pcg(A: Callable, b, xp, M: Optional[Callable] = None, x0=None, tol: float = 1e-8,
        maxiter: int = 500, project: Optional[Callable] = None):
    """Preconditioned conjugate gradients for SPD (or consistent PSD) systems.

    ``project`` removes the null space (e.g. the constant for all-Neumann
    pressure) from residuals so that singular-but-consistent systems converge.
    """
    x = xp.zeros_like(b) if x0 is None else x0.copy()
    r = b - A(x) if x0 is not None else b.copy()
    if project is not None:
        r = project(r)
    bnorm = float(xp.linalg.norm(b)) or 1.0
    rnorm = float(xp.linalg.norm(r))
    if rnorm / bnorm < tol:
        return x, KrylovInfo(0, rnorm / bnorm, True)
    z = M(r) if M is not None else r
    if project is not None:
        z = project(z)
    p = z.copy()
    rz = _dot(xp, r, z)
    for k in range(1, maxiter + 1):
        Ap = A(p)
        pAp = _dot(xp, p, Ap)
        if pAp <= 0.0:
            return x, KrylovInfo(k, rnorm / bnorm, False)
        alpha = rz / pAp
        x += alpha * p
        r -= alpha * Ap
        rnorm = float(xp.linalg.norm(r))
        if rnorm / bnorm < tol:
            return x, KrylovInfo(k, rnorm / bnorm, True)
        z = M(r) if M is not None else r
        if project is not None:
            z = project(z)
        rz_new = _dot(xp, r, z)
        if rz == 0.0:  # breakdown (r.z underflowed: seen only on a diverging system); report, don't raise
            return x, KrylovInfo(k, rnorm / bnorm, False)
        beta = rz_new / rz
        rz = rz_new
        p = z + beta * p
    return x, KrylovInfo(maxiter, rnorm / bnorm, False)


def bicgstab(A: Callable, b, xp, M: Optional[Callable] = None, x0=None, tol: float = 1e-8,
             maxiter: int = 500, relative_to_initial: bool = False):
    """Right-preconditioned BiCGSTAB for non-symmetric systems (e.g. momentum).

    With ``relative_to_initial`` the tolerance applies to ||r|| / ||r0|| (the
    termination rule of segregated CFD solvers: reduce the residual of the
    current outer iteration by a fixed factor).
    """
    x = xp.zeros_like(b) if x0 is None else x0.copy()
    r = b - A(x) if x0 is not None else b.copy()
    rnorm = float(xp.linalg.norm(r))
    b_abs = float(xp.linalg.norm(b))
    bnorm = (rnorm if relative_to_initial else b_abs) or 1.0
    # an initial residual at round-off level cannot be reduced further; iterating on it
    # makes the recurrence coefficients overflow (seen when an outer iteration has converged)
    if rnorm / bnorm < tol or rnorm <= 1e-14 * b_abs:
        return x, KrylovInfo(0, rnorm / bnorm, True)
    r_hat = r.copy()
    rho = alpha = omega = 1.0
    v = xp.zeros_like(b)
    p = xp.zeros_like(b)
    for k in range(1, maxiter + 1):
        rho_new = _dot(xp, r_hat, r)
        if rho_new == 0.0:
            break
        beta = (rho_new / rho) * (alpha / omega)
        rho = rho_new
        p = r + beta * (p - omega * v)
        phat = M(p) if M is not None else p
        v = A(phat)
        denom = _dot(xp, r_hat, v)
        if denom == 0.0:
            break
        alpha = rho / denom
        s = r - alpha * v
        snorm = float(xp.linalg.norm(s))
        if snorm / bnorm < tol:
            x += alpha * phat
            return x, KrylovInfo(k, snorm / bnorm, True)
        shat = M(s) if M is not None else s
        t = A(shat)
        tt = _dot(xp, t, t)
        omega = _dot(xp, t, s) / tt if tt > 0 else 0.0
        x_new = x + alpha * phat + omega * shat
        r = s - omega * t
        rnorm = float(xp.linalg.norm(r))
        if not math.isfinite(rnorm):  # breakdown: keep the last finite iterate
            return x, KrylovInfo(k, float("nan"), False)
        x = x_new
        if rnorm / bnorm < tol or rnorm <= 1e-14 * b_abs:
            return x, KrylovInfo(k, rnorm / bnorm, True)
        if omega == 0.0:
            break
    return x, KrylovInfo(k, rnorm / bnorm, False)


def device_krylov(A, b, xp, M, x0, tol, maxiter, *, symmetric=False, project=None, relative_to_initial=False):
    """Krylov iterations with device scalars and a host convergence check per batch.

    Mask converged or broken-down recurrences so the remaining batch steps
    cannot alter their solution. The final residual is recomputed with A.
    """
    x = xp.zeros_like(b) if x0 is None else x0.copy()
    r = b - A(x) if x0 is not None else b.copy()
    if project is not None:
        r = project(r)
    dot = lambda a, b: xp.vdot(a, b).real
    norm2 = lambda v: dot(v, v)
    start2 = norm2(r)
    b2 = norm2(b)
    scale2 = start2 if relative_to_initial else b2
    scale2 = xp.where(scale2 > 0, scale2, 1.0)
    target = xp.maximum(tol**2 * scale2, 1e-28 * b2)
    active = start2 > target
    if not bool(active):
        return x, KrylovInfo(0, float(xp.sqrt(start2 / scale2)), bool(xp.isfinite(start2)))

    def ratio(numerator, denominator):
        good = xp.isfinite(numerator) & xp.isfinite(denominator) & (denominator != 0)
        return xp.where(good, numerator / xp.where(good, denominator, 1), 0), good

    count = xp.zeros((), dtype=xp.int32)
    if symmetric:
        z = M(r)
        if project is not None:
            z = project(z)
        direction = z.copy()
        rz = dot(r, z)
    else:
        shadow = r.copy()
        direction = xp.zeros_like(b)
        v = xp.zeros_like(b)
        rho = alpha = omega = xp.ones((), dtype=b.dtype)
    batch_size = 4
    for k in range(maxiter):
        count = count + active.astype(xp.int32)
        if symmetric:
            q = A(direction)
            dq = dot(direction, q)
            step, valid = ratio(rz, dq)
            live = active & valid & (dq > 0)
            correction = step * direction
            next_r = r - step * q
            continuation = rz != 0
        else:
            next_rho = dot(shadow, r)
            beta, valid_beta = ratio(next_rho * alpha, rho * omega)
            direction = r + beta * (direction - omega * v)
            y = M(direction)
            v = A(y)
            alpha, valid_alpha = ratio(next_rho, dot(shadow, v))
            short_r = r - alpha * v
            short = norm2(short_r) <= target
            z = M(short_r)
            q = A(z)
            omega, valid_omega = ratio(dot(q, short_r), norm2(q))
            live = active & valid_beta & valid_alpha & (next_rho != 0)
            live = live & (short | valid_omega)
            omega = xp.where(short, 0, omega)
            correction = alpha * y + omega * z
            next_r = short_r - omega * q
            rho = next_rho
            continuation = short | (omega != 0)
        residual2 = norm2(next_r)
        accepted = live & xp.isfinite(residual2)
        x = xp.where(accepted, x + correction, x)
        r = xp.where(accepted, next_r, r)
        active = accepted & continuation & (residual2 > target)
        if (k == 0 or (k + 1) % batch_size == 0 or k + 1 == maxiter) and not bool(active):
            break
        if symmetric:
            z = M(r)
            if project is not None:
                z = project(z)
            next_rz = dot(r, z)
            beta, valid_beta = ratio(next_rz, rz)
            direction = z + beta * direction
            active = active & valid_beta
            rz = next_rz
    true_r = b - A(x)
    if project is not None:
        true_r = project(true_r)
    true2 = norm2(true_r)
    return x, KrylovInfo(int(count), float(xp.sqrt(true2 / scale2)), bool(true2 <= target))


def refined_krylov(A, b, xp, M, x0, tol, maxiter, *, symmetric=False, project=None, relative_to_initial=False,
                  device=False):
    """FP32 inner solves with FP64 residuals, updates and FP64 recovery."""
    A32 = getattr(A, "csr", A).astype(xp.float32)

    def solve(operator, rhs, guess, target, limit):
        matvec = lambda v: operator @ v
        precondition = lambda v: M(v).astype(v.dtype, copy=False)
        if device:
            return device_krylov(matvec, rhs, xp, precondition, guess, target, limit, symmetric=symmetric,
                                 project=project, relative_to_initial=relative_to_initial)
        if symmetric:
            return pcg(matvec, rhs, xp, M=precondition, x0=guess, tol=target, maxiter=limit, project=project)
        return bicgstab(matvec, rhs, xp, M=precondition, x0=guess, tol=target, maxiter=limit,
                        relative_to_initial=relative_to_initial)

    x = xp.zeros_like(b) if x0 is None else x0.copy()
    residual = b - A @ x
    norm_b = float(xp.linalg.norm(b))
    initial = float(xp.linalg.norm(project(residual) if project is not None else residual))
    scale = (initial if relative_to_initial else norm_b) or 1.0
    target = max(tol * scale, 1e-14 * norm_b)
    previous = initial
    count = 0
    for _ in range(3):
        if previous <= target or count >= maxiter:
            break
        correction, info = solve(A32, residual.astype(xp.float32), None, max(tol * 0.25, 1e-4), maxiter - count)
        count += info.iterations
        candidate = x + correction.astype(x.dtype)
        residual = b - A @ candidate
        error = float(xp.linalg.norm(project(residual) if project is not None else residual))
        if not math.isfinite(error) or error >= 0.95 * previous:
            break
        x, previous = candidate, error
    if previous <= target:
        return x, KrylovInfo(count, previous / scale, True)
    # If the FP32 recurrence stalls, the FP64 solve keeps its normal iteration budget.
    x, info = solve(A, b, x, tol if not relative_to_initial else target / previous, maxiter)
    residual = b - A @ x
    error = float(xp.linalg.norm(project(residual) if project is not None else residual))
    return x, KrylovInfo(count + info.iterations, error / scale, error <= target)

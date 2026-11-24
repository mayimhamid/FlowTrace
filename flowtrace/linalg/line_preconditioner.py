"""Hybrid Line-Implicit / Directional Preconditioner for high-aspect-ratio cells.

Solves decoupled tridiagonal systems along wall-normal lines in structured
prisms/quads in boundary layers, and scales isotropic cells by inverse diagonal.
Supports both CPU (Numba / NumPy) and GPU (CuPy).
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange
    HAS_NUMBA = True
except Exception:
    HAS_NUMBA = False

try:
    import cupy as cp
    HAS_CUPY = True
except Exception:
    cp = None
    HAS_CUPY = False


if HAS_NUMBA:
    @njit(parallel=True, fastmath=True, cache=True)
    def _extract_tridiag_cpu(
        indptr: np.ndarray,
        indices: np.ndarray,
        data: np.ndarray,
        lines_idx: np.ndarray,
        line_lengths: np.ndarray,
        lower: np.ndarray,
        diag: np.ndarray,
        upper: np.ndarray,
    ) -> None:
        """Extracts tridiagonal coefficients from CSR matrix into padded SoA buffers."""
        n_lines, max_len = lines_idx.shape
        for k in prange(n_lines):
            length = line_lengths[k]
            for j in range(max_len):
                if j >= length:
                    lower[k, j] = 0.0
                    diag[k, j] = 1.0
                    upper[k, j] = 0.0
                    continue

                row = lines_idx[k, j]
                prev_cell = lines_idx[k, j - 1] if j > 0 else -1
                next_cell = lines_idx[k, j + 1] if j < length - 1 else -1

                l_val = 0.0
                d_val = 1.0
                u_val = 0.0

                for ptr in range(indptr[row], indptr[row + 1]):
                    col = indices[ptr]
                    val = data[ptr]
                    if col == row:
                        d_val = val
                    elif col == prev_cell:
                        l_val = val
                    elif col == next_cell:
                        u_val = val

                lower[k, j] = l_val
                diag[k, j] = d_val
                upper[k, j] = u_val


    @njit(parallel=True, fastmath=True, cache=True)
    def _thomas_solve_cpu(
        lower: np.ndarray,
        diag: np.ndarray,
        upper: np.ndarray,
        lines_idx: np.ndarray,
        line_lengths: np.ndarray,
        r: np.ndarray,
        z: np.ndarray,
        c_prime: np.ndarray,
        d_prime: np.ndarray,
    ) -> None:
        """Batched parallel Thomas algorithm for wall-normal lines on CPU."""
        n_lines = lines_idx.shape[0]
        for k in prange(n_lines):
            length = line_lengths[k]
            if length == 0:
                continue

            # Forward elimination
            c0 = lines_idx[k, 0]
            inv_d = 1.0 / diag[k, 0]
            c_prime[k, 0] = upper[k, 0] * inv_d
            d_prime[k, 0] = r[c0] * inv_d

            for j in range(1, length):
                cj = lines_idx[k, j]
                l_val = lower[k, j]
                denom = diag[k, j] - l_val * c_prime[k, j - 1]
                inv_denom = 1.0 / denom
                c_prime[k, j] = upper[k, j] * inv_denom if j < length - 1 else 0.0
                d_prime[k, j] = (r[cj] - l_val * d_prime[k, j - 1]) * inv_denom

            # Backward substitution
            last = length - 1
            z[lines_idx[k, last]] = d_prime[k, last]
            for j in range(last - 1, -1, -1):
                cj = lines_idx[k, j]
                z[cj] = d_prime[k, j] - c_prime[k, j] * z[lines_idx[k, j + 1]]

else:
    def _extract_tridiag_cpu(indptr, indices, data, lines_idx, line_lengths, lower, diag, upper):
        n_lines, max_len = lines_idx.shape
        for k in range(n_lines):
            length = line_lengths[k]
            for j in range(max_len):
                if j >= length:
                    lower[k, j] = 0.0
                    diag[k, j] = 1.0
                    upper[k, j] = 0.0
                    continue
                row = lines_idx[k, j]
                prev_cell = lines_idx[k, j - 1] if j > 0 else -1
                next_cell = lines_idx[k, j + 1] if j < length - 1 else -1
                l_val, d_val, u_val = 0.0, 1.0, 0.0
                for ptr in range(indptr[row], indptr[row + 1]):
                    col = indices[ptr]
                    val = data[ptr]
                    if col == row:
                        d_val = val
                    elif col == prev_cell:
                        l_val = val
                    elif col == next_cell:
                        u_val = val
                lower[k, j] = l_val
                diag[k, j] = d_val
                upper[k, j] = u_val

    def _thomas_solve_cpu(lower, diag, upper, lines_idx, line_lengths, r, z, c_prime, d_prime):
        n_lines = lines_idx.shape[0]
        for k in range(n_lines):
            length = line_lengths[k]
            if length == 0:
                continue
            c0 = lines_idx[k, 0]
            inv_d = 1.0 / diag[k, 0]
            c_prime[k, 0] = upper[k, 0] * inv_d
            d_prime[k, 0] = r[c0] * inv_d
            for j in range(1, length):
                cj = lines_idx[k, j]
                l_val = lower[k, j]
                denom = diag[k, j] - l_val * c_prime[k, j - 1]
                inv_denom = 1.0 / denom
                c_prime[k, j] = upper[k, j] * inv_denom if j < length - 1 else 0.0
                d_prime[k, j] = (r[cj] - l_val * d_prime[k, j - 1]) * inv_denom
            last = length - 1
            z[lines_idx[k, last]] = d_prime[k, last]
            for j in range(last - 1, -1, -1):
                cj = lines_idx[k, j]
                z[cj] = d_prime[k, j] - c_prime[k, j] * z[lines_idx[k, j + 1]]


_LINE_SOLVE_CUDA_KERNEL = None


def _get_cuda_kernel():
    global _LINE_SOLVE_CUDA_KERNEL
    if _LINE_SOLVE_CUDA_KERNEL is None and HAS_CUPY:
        _LINE_SOLVE_CUDA_KERNEL = cp.RawKernel(
            r"""
            extern "C" __global__
            void thomas_line_solve(
                const double* __restrict__ lower,
                const double* __restrict__ diag,
                const double* __restrict__ upper,
                const int* __restrict__ lines_idx,
                const int* __restrict__ line_lengths,
                const double* __restrict__ r,
                double* __restrict__ z,
                double* __restrict__ c_prime,
                double* __restrict__ d_prime,
                const int n_lines,
                const int max_len
            ) {
                int k = blockIdx.x * blockDim.x + threadIdx.x;
                if (k >= n_lines) return;

                int length = line_lengths[k];
                if (length <= 0) return;

                int base = k * max_len;
                int c0 = lines_idx[base];
                double inv_d = 1.0 / diag[base];
                c_prime[base] = upper[base] * inv_d;
                d_prime[base] = r[c0] * inv_d;

                for (int j = 1; j < length; ++j) {
                    int idx = base + j;
                    int cj = lines_idx[idx];
                    double l_val = lower[idx];
                    double denom = diag[idx] - l_val * c_prime[idx - 1];
                    double inv_denom = 1.0 / denom;
                    c_prime[idx] = (j < length - 1) ? (upper[idx] * inv_denom) : 0.0;
                    d_prime[idx] = (r[cj] - l_val * d_prime[idx - 1]) * inv_denom;
                }

                int last = length - 1;
                z[lines_idx[base + last]] = d_prime[base + last];
                for (int j = last - 1; j >= 0; --j) {
                    int idx = base + j;
                    z[lines_idx[idx]] = d_prime[idx] - c_prime[idx] * z[lines_idx[idx + 1]];
                }
            }
            """,
            "thomas_line_solve",
        )
    return _LINE_SOLVE_CUDA_KERNEL


class LineImplicitPreconditioner:
    """
    Hybrid Line-Implicit / Point-Jacobi Preconditioner for high-aspect-ratio
    boundary-layer grids on CPU (Numba) and GPU (CuPy).
    """

    def __init__(self, n_cells: int, lines_list: list[np.ndarray], backend: str = "cpu"):
        self.n_cells = n_cells
        self.backend = backend
        self.n_lines = len(lines_list)
        self.max_len = max((len(line) for line in lines_list), default=0)

        lines_idx = np.full((self.n_lines, self.max_len), -1, dtype=np.int32)
        line_lengths = np.zeros(self.n_lines, dtype=np.int32)
        in_line = np.zeros(n_cells, dtype=bool)

        for k, line in enumerate(lines_list):
            L = len(line)
            line_lengths[k] = L
            lines_idx[k, :L] = line
            in_line[line] = True

        iso_cells = np.where(~in_line)[0].astype(np.int32)

        xp = cp if (backend == "gpu" and HAS_CUPY) else np
        self.xp = xp
        self.lines_idx = xp.asarray(lines_idx)
        self.line_lengths = xp.asarray(line_lengths)
        self.iso_cells = xp.asarray(iso_cells)

        shape = (self.n_lines, self.max_len)
        self.lower = xp.zeros(shape, dtype=np.float64)
        self.diag = xp.ones(shape, dtype=np.float64)
        self.upper = xp.zeros(shape, dtype=np.float64)
        self.c_prime = xp.zeros(shape, dtype=np.float64)
        self.d_prime = xp.zeros(shape, dtype=np.float64)
        self.inv_diag_iso = xp.ones(len(iso_cells), dtype=np.float64)

    def update(self, csr_indptr, csr_indices, csr_data, diag_full) -> None:
        """Updates tridiagonal blocks and isotropic diagonal inverses from CSR matrix."""
        self.inv_diag_iso = 1.0 / diag_full[self.iso_cells]

        if self.n_lines == 0:
            return

        if self.backend == "cpu":
            _extract_tridiag_cpu(
                csr_indptr,
                csr_indices,
                csr_data,
                self.lines_idx,
                self.line_lengths,
                self.lower,
                self.diag,
                self.upper,
            )
        else:
            rows = self.lines_idx
            valid = rows >= 0
            safe_rows = cp.where(valid, rows, 0)
            self.diag = cp.where(valid, diag_full[safe_rows], 1.0)

            self.lower.fill(0.0)
            self.upper.fill(0.0)
            indptr_cpu = cp.asnumpy(csr_indptr) if hasattr(csr_indptr, "get") else np.asarray(csr_indptr)
            indices_cpu = cp.asnumpy(csr_indices) if hasattr(csr_indices, "get") else np.asarray(csr_indices)
            data_cpu = cp.asnumpy(csr_data) if hasattr(csr_data, "get") else np.asarray(csr_data)
            l_cpu = np.zeros((self.n_lines, self.max_len), dtype=np.float64)
            d_cpu = np.ones((self.n_lines, self.max_len), dtype=np.float64)
            u_cpu = np.zeros((self.n_lines, self.max_len), dtype=np.float64)
            _extract_tridiag_cpu(
                indptr_cpu,
                indices_cpu,
                data_cpu,
                cp.asnumpy(self.lines_idx) if hasattr(self.lines_idx, "get") else np.asarray(self.lines_idx),
                cp.asnumpy(self.line_lengths) if hasattr(self.line_lengths, "get") else np.asarray(self.line_lengths),
                l_cpu,
                d_cpu,
                u_cpu,
            )
            self.lower = cp.asarray(l_cpu)
            self.diag = cp.asarray(d_cpu)
            self.upper = cp.asarray(u_cpu)

    def apply(self, r):
        """Computes z = M^{-1} r."""
        z = self.xp.empty_like(r)
        z[self.iso_cells] = r[self.iso_cells] * self.inv_diag_iso

        if self.n_lines == 0:
            return z

        if self.backend == "cpu":
            _thomas_solve_cpu(
                self.lower,
                self.diag,
                self.upper,
                self.lines_idx,
                self.line_lengths,
                r,
                z,
                self.c_prime,
                self.d_prime,
            )
        else:
            threads = 128
            blocks = (self.n_lines + threads - 1) // threads
            kernel = _get_cuda_kernel()
            if kernel is not None:
                kernel(
                    (blocks,),
                    (threads,),
                    (
                        self.lower,
                        self.diag,
                        self.upper,
                        self.lines_idx,
                        self.line_lengths,
                        r,
                        z,
                        self.c_prime,
                        self.d_prime,
                        self.n_lines,
                        self.max_len,
                    ),
                )
        return z

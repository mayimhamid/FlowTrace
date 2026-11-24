r"""Device bridge: the one-way-coupled scalar equations run concurrently, each on its own hardware.

Why this is exact
-----------------
In a SIMPLE/SIMPLEC iteration without buoyancy, temperature (with P-1 radiation) and passive species
do not feed back into the flow: the momentum, pressure and turbulence equations of iteration n+1 never
read T or Y. The serial solver nevertheless waits for them every iteration, and on a heated turbulent
case they are 50-65 % of the iteration time (``benchmarks/device_bridge.py``).

A *lane* solves one of these equations on a device of its own while the flow solver carries on:

* ``energy``  (+ P-1 radiation): launched right after the pressure correction of iteration n with the
  inputs the serial solver uses at that point (corrected fluxes F_n, the effective diffusivity built
  from mu_t of iteration n-1, T_{n-1}). It overlaps the turbulence solve of iteration n and the
  gradients, momentum and pressure of iteration n+1.
* ``species``: launched after the turbulence solve of iteration n (its diffusivity uses mu_t of
  iteration n, as in the serial solver), overlapping the flow part of iteration n+1.

Every input is captured as an immutable snapshot in the main thread before the launch (the solver
rebinds its arrays, it never writes into them), so a lane computes exactly what the serial solver
computes. On the same device the fields are bit-identical to the serial solver's; on another device
they differ only by the floating-point summation order of that device's kernels.

A lane is collected right before it is launched again, and whenever anything outside the lane reads
T, Y or G (``fields()``, ``state_dict()``, a stability-guard rollback, a time-step advance, a convergence
test): see ``SimpleSolver._sync_lanes``. Residuals of a lane are one iteration behind the flow's in the
history; convergence is only declared after the lanes in flight have been collected and their own
residuals are below the tolerance too, so the converged state is consistent in every field.

The bridges
-----------
* CPU lane: a Python thread whose native (C++/OpenMP) kernels run on a share of the cores that is
  proportional to the lane's measured work (``ft_set_threads`` sets the OpenMP team of the calling
  thread only), while the flow solver keeps the remaining cores.
* GPU lane, flow also on the GPU: a non-blocking CUDA stream that waits on an event recorded after the
  flow solver produced the lane's inputs, so independent kernels of the two lanes interleave.
* Cross-device lane: a *shadow* solver for the scalar equations on the other device. Per launch only
  the face fluxes and the diffusivity cross the bus (O(faces) doubles); the scalar field stays resident
  on the lane's device and comes back once per iteration.

Which device each lane uses is not assumed: :class:`LaneTuner` measures the real iterations of the run
(serial first, then each candidate assignment for a few iterations, all of them useful work) and keeps
the fastest, or the serial solver when no assignment is at least ``MIN_GAIN`` faster.
"""

from __future__ import annotations

import copy
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import numpy as np

LANES = ("energy", "species")


# ---------------------------------------------------------------------------- eligibility
def eligible_lanes(solver) -> Tuple[List[str], Dict[str, str]]:
    """Lanes that may run concurrently with the flow, and the reason for each excluded equation."""
    c = solver.case
    ok, why = [], {}
    if solver.energy:
        if solver.external_energy:
            why["energy"] = "a conjugate solver owns the energy equation"
        elif any(c.buoyancy[: solver.dim]):
            why["energy"] = "buoyancy couples temperature back into momentum"
        else:
            ok.append("energy")
    if c.species:
        ok.append("species")
    return ok, why


def gpu_available() -> bool:
    try:
        from ..hardware.backend import get_backend

        be = get_backend("cupy")
        return not getattr(be.xp, "__fake__", False)
    except Exception:
        return False


# ---------------------------------------------------------------------------- transfers
def _to_device(v, gpu: bool):
    """Move an input (array, tuple of arrays, scalar, None) to the CPU (gpu=False) or the GPU."""
    if v is None or isinstance(v, (float, int, np.floating, np.integer)):
        return v
    if isinstance(v, tuple):
        return tuple(_to_device(a, gpu) for a in v)
    if isinstance(v, dict):
        return {k: _to_device(a, gpu) for k, a in v.items()}
    if gpu:
        import cupy as cp  # type: ignore

        return cp.asarray(v)
    return v.get() if hasattr(v, "get") and not isinstance(v, np.ndarray) else np.asarray(v)


def make_shadow(solver, device: str):
    """A solver for the scalar equations only, on ``device`` ('cpu' or 'gpu'): same mesh, boundary
    conditions and precision. It needs no turbulence model: the lanes receive the effective
    diffusivity from the main solver."""
    from .simple import SimpleSolver

    case = copy.deepcopy(solver.case)
    case.backend = "cupy" if device == "gpu" else "numpy"
    case.turbulence = "laminar"
    case.lanes = "off"
    case.periodic = []  # the main solver's mesh is already paired
    sh = SimpleSolver(solver.mesh, case)
    sh.guard_enabled = False
    return sh


# ---------------------------------------------------------------------------- one lane
class Lane:
    """One scalar equation solved on its own thread, device and (on the CPU) share of the cores."""

    def __init__(self, solver, name: str, device: str, threads: int = 0, shadow=None):
        self.main, self.name, self.device = solver, name, device
        self.gpu = device == "gpu"
        self.cross = device != solver.be.device
        if self.cross and shadow is None:
            shadow = make_shadow(solver, device)
        self.worker = shadow if self.cross else solver
        self.threads = int(threads)
        self.pool = ThreadPoolExecutor(1, thread_name_prefix=f"flowtrace-lane-{name}", initializer=self._init_thread)
        self.thread_ident = None
        self.stream = None
        if self.gpu:
            import cupy as cp  # type: ignore

            with cp.cuda.Device(self.worker.be.device_index):
                self.stream = cp.cuda.Stream(non_blocking=True)
        self.future = None
        self.dirty = True  # the worker's copy of the field must be refreshed from the main solver
        self.busy_seconds = 0.0
        self.launches = 0

    def _init_thread(self):
        self.thread_ident = threading.get_ident()
        if self.threads > 0 and not self.gpu:
            from .. import native

            native.set_threads(self.threads)

    @property
    def busy(self) -> bool:
        return self.future is not None

    # ------------------------------------------------------------------
    def launch(self, inputs: dict) -> None:
        assert self.future is None, "collect a lane before launching it again"
        state = None
        if self.cross and self.dirty:  # snapshot of the main solver's field, taken in the main thread
            m = self.main
            if self.name == "energy":
                state = {"T": m._T, "G": m.rad.G if m.rad is not None else None}
            else:
                state = {"Y": dict(m._Y)}
            self.dirty = False
        event = None
        if self.main.be.is_gpu:  # the inputs are produced on the main (legacy default) stream
            import cupy as cp  # type: ignore

            event = cp.cuda.Event(block=False, disable_timing=True)
            event.record(cp.cuda.Stream.null)
        self.future = self.pool.submit(self._run, inputs, state, event)
        self.launches += 1

    def _run(self, inputs, state, event):
        t0 = time.perf_counter()
        w = self.worker
        if self.gpu:
            ctx = self.stream
            if event is not None:
                self.stream.wait_event(event)
        else:
            import contextlib

            ctx = contextlib.nullcontext()
            if event is not None:  # inputs on the GPU, lane on the CPU: wait until they exist
                event.synchronize()
        with ctx:
            if self.cross:
                inputs = _to_device(inputs, self.gpu)
                if state is not None:
                    if self.name == "energy":
                        w._T = w.xp.asarray(_to_device(state["T"], self.gpu), dtype=w.dtype)
                        w._gT = None
                        if w.rad is not None and state["G"] is not None:
                            w.rad.G = w.xp.asarray(_to_device(state["G"], self.gpu), dtype=w.dtype)
                    else:
                        w._Y = {k: w.xp.asarray(_to_device(v, self.gpu), dtype=w.dtype) for k, v in state["Y"].items()}
            if self.name == "energy":
                T_new, res, cap = w._energy_kernel(w._T, inputs)
                if self.cross:
                    w._T = T_new
                out = {"T": T_new, "res": res, "capture": None if self.cross else cap,
                       "G": w.rad.G if w.rad is not None else None}
            else:
                Y_new, res = w._species_kernel(w._Y, inputs)
                if self.cross:
                    w._Y = Y_new
                out = {"Y": Y_new, "res": res}
            if self.cross:  # back to the main solver's device
                main_gpu = self.main.be.is_gpu
                out = {k: (_to_device(v, main_gpu) if k != "res" and k != "capture" else v) for k, v in out.items()}
            if self.gpu:
                self.stream.synchronize()
        self.busy_seconds += time.perf_counter() - t0
        return out

    def collect(self) -> Optional[dict]:
        if self.future is None:
            return None
        f, self.future = self.future, None
        return f.result()

    def close(self):
        try:
            self.collect()
        finally:
            self.pool.shutdown(wait=True)


# ---------------------------------------------------------------------------- all lanes of a solver
class Bridge:
    """The lanes of one solver and the bookkeeping that keeps the solver's state consistent."""

    def __init__(self, solver, assignment: Dict[str, str], threads: Optional[Dict[str, int]] = None,
                 shadows: Optional[dict] = None):
        self.s = solver
        self.assignment = dict(assignment)
        threads = threads or {}
        shadows = shadows if shadows is not None else {}
        self.lanes: Dict[str, Lane] = {}
        for name, dev in assignment.items():
            sh = None
            if dev != solver.be.device:
                if dev not in shadows:
                    shadows[dev] = make_shadow(solver, dev)
                sh = shadows[dev]
            self.lanes[name] = Lane(solver, name, dev, threads.get(name, 0), sh)
        # the flow's share of the cores; SimpleSolver.step applies it to its own thread for the step only
        self.main_threads = int(threads.get("flow", 0))
        self.residuals: Dict[str, float] = {}
        self.capture = None
        self.wait_seconds = 0.0

    def has(self, name: str) -> bool:
        return name in self.lanes

    @property
    def busy(self) -> bool:
        return any(l.busy for l in self.lanes.values())

    def in_lane_thread(self) -> bool:
        me = threading.get_ident()
        return any(l.thread_ident == me for l in self.lanes.values())

    def _apply(self, name: str, out: Optional[dict]) -> None:
        if out is None:
            return
        s = self.s
        if name == "energy":
            s._T = out["T"]
            if s.rad is not None and out.get("G") is not None:
                s.rad.G = out["G"]
            self.capture = out.get("capture")
        else:
            s._Y = out["Y"]
        self.residuals.update(out["res"])

    def collect(self, name: str) -> None:
        lane = self.lanes[name]
        if not lane.busy:
            return
        t0 = time.perf_counter()
        out = lane.collect()
        if self.s.be.is_gpu:
            # the previous field may still be read by queued work of the main stream; it is freed
            # (and its memory reused by the lane's stream) only after that work has finished
            self.s.xp.cuda.Stream.null.synchronize()
        self._apply(name, out)
        self.wait_seconds += time.perf_counter() - t0

    def cycle(self, name: str, inputs: dict) -> None:
        """Collect the lane's previous result, then launch it on ``inputs``."""
        self.collect(name)
        self.lanes[name].launch(inputs)

    def await_all(self) -> None:
        for name in self.lanes:
            self.collect(name)

    def mark_dirty(self, name: Optional[str] = None) -> None:
        for n, lane in self.lanes.items():
            if name is None or n == name:
                lane.dirty = True

    def close(self) -> None:
        self.await_all()
        for lane in self.lanes.values():
            lane.close()

    def report(self) -> dict:
        return {"lanes": {n: {"device": l.device, "cross_device": l.cross, "cpu_threads": l.threads or None,
                              "busy_s": round(l.busy_seconds, 3), "launches": l.launches}
                          for n, l in self.lanes.items()},
                "flow_cpu_threads": self.main_threads or None,
                "flow_wait_s": round(self.wait_seconds, 3)}


# ---------------------------------------------------------------------------- measured placement
class LaneTuner:
    """Measure the run's own iterations under each candidate configuration; keep the fastest.

    Phase 1 compares serial execution with lanes on the solver's chosen device.
    Each candidate runs for ``WINDOW`` measured iterations.
    Phase 2, core budget: the winner is re-timed with half the CPU threads, then a quarter, ... while
    that is at least ``BUDGET_GAIN`` faster. Small meshes are often fastest on few cores: an OpenMP
    region costs more than the work it splits, and concurrent lanes then also stop competing.

    Every timed iteration is a real iteration of the run (nothing is thrown away), and every
    configuration computes the same iterates (bit-identical on the same device with the same
    reduction order). Iteration 1 (AMG set-up, JIT) and the first iteration of every configuration
    (thread start, shadow set-up, kernel compilation) are not counted; medians are compared, so one
    AMG rebuild inside a window does not decide. Lanes are kept only if at least ``MIN_GAIN`` faster.
    """

    WINDOW = 5
    MIN_GAIN = 0.08
    BUDGET_GAIN = 0.03
    ABORT = 1.3  # a configuration whose two first timed iterations are both this much slower than the best is dropped
    MIN_ITERATIONS = 100  # shorter runs cannot pay for the trials

    def __init__(self, solver, lanes: List[str]):
        self.s = solver
        self.lanes = lanes
        self.phase = "baseline"
        self.samples: List[float] = []
        self.results: Dict[str, float] = {}
        self.queue: List[Tuple[str, Optional[Dict[str, str]], int]] = []
        self.current: Optional[Tuple[str, Optional[Dict[str, str]], int]] = None
        self.best: Optional[Tuple[str, Optional[Dict[str, str]], int, float]] = None
        self.shadows: dict = {}
        self.stage0: Dict[str, float] = {}
        self.decision: Optional[dict] = None
        self.started_at = solver.iteration
        from .. import native

        self.cores = native.max_threads() if native.available() else 0

    # ------------------------------------------------------------------
    def _placements(self) -> List[Tuple[str, Optional[Dict[str, str]], int]]:
        s = self.s
        home = s.be.device
        return [(f"lanes on the {home.upper()}", {name: home for name in self.lanes}, self.cores)]

    def _threads(self, assignment: Optional[Dict[str, str]], budget: int) -> Dict[str, int]:
        """Split ``budget`` CPU threads between the flow (if on the CPU) and the CPU lanes, in
        proportion to each one's measured serial work."""
        if budget <= 0:
            return {}
        work = {}
        if self.s.be.device == "cpu":
            work["flow"] = max(self.stage0.get("flow", 1.0), 1e-6)
        for n, dev in (assignment or {}).items():
            if dev == "cpu":
                work[n] = max(self.stage0.get(n, 1.0), 1e-6)
        if not work:
            return {}
        w = sum(work.values())
        out = {k: max(1, int(round(budget * v / w))) for k, v in work.items()}
        while sum(out.values()) > max(budget, len(out)):  # rounding must not oversubscribe the budget
            k = max(out, key=out.get)
            out[k] -= 1
        return out

    def _stage_times(self) -> Dict[str, float]:
        t = self.s.timers
        flow = sum(v for k, v in t.items() if k not in ("energy", "species"))
        return {"flow": flow, "energy": t.get("energy", 0.0), "species": t.get("species", 0.0)}

    def _apply(self, assignment: Optional[Dict[str, str]], budget: int) -> None:
        s = self.s
        threads = self._threads(assignment, budget)
        if assignment:
            s._set_bridge(Bridge(s, assignment, threads, self.shadows))
        else:
            s._set_bridge(None)
        s._flow_threads = threads.get("flow", 0)

    def _start(self, cand) -> None:
        label, assignment, budget = cand
        self.current = cand
        self.samples = []
        try:
            self._apply(assignment, budget)
        except Exception as exc:  # a device that cannot host a lane is skipped, never fatal
            self.results[label] = float("inf")
            self.s._lane_notes.append(f"{label}: unavailable ({exc})")
            self._next()

    def _next(self) -> None:
        if self.queue:
            self._start(self.queue.pop(0))
        elif self.phase == "placement":
            self._placement_done()
        else:
            self._finish()

    def _placement_done(self) -> None:
        base = self.results["serial"]
        trials = [(lab, a, b) for lab, a, b in self._tried if np.isfinite(self.results.get(lab, np.inf))]
        best = min(trials, key=lambda c: self.results[c[0]], default=None)
        if best is not None and self.results[best[0]] <= (1.0 - self.MIN_GAIN) * base:
            self.best = (*best, self.results[best[0]])
        else:
            self.best = ("serial", None, self.cores, base)
        self.phase = "budget"
        self._queue_budget()

    def _queue_budget(self) -> None:
        label, assignment, budget, _ = self.best
        consumers = len(self._threads(assignment, max(budget, 1)))  # 0: nothing runs on CPU threads
        nb = budget // 2
        if self.cores and consumers and nb >= consumers:
            base_label = label.split(" @ ")[0]
            self.queue = [(f"{base_label} @ {nb} CPU threads", assignment, nb)]
            self._tried.append(self.queue[0])
        self._next()

    def _finish(self) -> None:
        label, assignment, budget, t = self.best
        self._apply(assignment, budget if self.cores else 0)
        if assignment is None and budget >= self.cores:
            self.s._flow_threads = 0  # plain serial solver on its default threads
        base = self.results["serial"]
        self.decision = {
            "chosen": label,
            "assignment": assignment,
            "cpu_threads": self._threads(assignment, budget) if self.cores else None,
            "speedup": round(base / t, 3) if t > 0 else 1.0,
            "ms_per_iteration": {k: (round(1e3 * v, 2) if np.isfinite(v) else None) for k, v in self.results.items()},
            "reason": f"{label}: {1e3 * t:.1f} ms vs serial {1e3 * base:.1f} ms per iteration (median of {self.WINDOW})",
            "decided_at_iteration": self.s.iteration,
        }
        if not assignment or all(d == self.s.be.device for d in assignment.values()):
            self.shadows.clear()
        self.phase = "done"

    # ------------------------------------------------------------------
    def after_step(self, wall: float) -> None:
        if self.phase == "done":
            return
        s = self.s
        if s.iteration - self.started_at <= 1:  # first iteration: AMG set-up, JIT
            return
        self.samples.append(wall)
        if self.phase == "baseline":
            if len(self.samples) >= self.WINDOW:
                self.results["serial"] = statistics.median(self.samples)
                it = max(s.iteration, 1)
                self.stage0 = {k: v / it for k, v in self._stage_times().items()}
                self.queue = self._placements()
                self._tried = list(self.queue)
                self.phase = "placement"
                self._next()
            return
        label, assignment, budget = self.current
        ref = min([v for v in self.results.values() if np.isfinite(v)], default=float("inf"))
        if len(self.samples) == 3 and min(self.samples[1:]) > self.ABORT * ref:
            self.results[label] = statistics.median(self.samples[1:])  # clearly slower: stop timing it
            if self.phase == "budget":
                self.queue = []
            self._next()
            return
        if len(self.samples) < self.WINDOW + 1:  # the first sample of a configuration pays for its start-up
            return
        t = statistics.median(self.samples[1:])
        self.results[label] = t
        if self.phase == "budget":
            if t <= (1.0 - self.BUDGET_GAIN) * self.best[3]:
                self.best = (label, assignment, budget, t)
                self._queue_budget()
                return
            self.queue = []
        self._next()

"""What a Fluent case file says about the physics: models, material and boundary values.

A CFF ``.cas.h5`` carries, next to the mesh, the solver settings as Scheme text
(``/settings/Rampant Variables``, ``/settings/Thread Variables``). FlowTrace used to read the mesh
only, so a case set up in Fluent as a compressible nozzle was solved as incompressible flow with
default boundary values. :func:`read_case_settings` extracts what FlowTrace can act on, and
:func:`choose_solver` decides which of its solvers the case needs:

* ``compressible``: the fluid density follows the ideal-gas law, or a boundary is a pressure
  far field. Solved by the density-based solver (:mod:`flowtrace.compressible`).
* ``incompressible``: constant density. Solved by the pressure-based SIMPLE/SIMPLEC solver.

Only constant boundary values are read (profiles and UDFs are reported as not read). Gauge
pressures are returned as absolute ones (the operating pressure is added).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

_NUM = r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
R_UNIVERSAL = 8314.462618  # J/(kmol K)


def _const(seg: str, key: str) -> Optional[float]:
    """``(key (constant . v) ...)`` or ``(key . v)`` in a thread's settings."""
    m = re.search(r"\(%s \(constant \. %s\)" % (re.escape(key), _NUM), seg)
    if m is None:
        m = re.search(r"\(%s \. %s\)" % (re.escape(key), _NUM), seg)
    return float(m.group(1)) if m else None


def _flag(text: str, key: str) -> Optional[bool]:
    m = re.search(r"\(%s(?: \.)? (#t|#f)\)" % re.escape(key), text)
    return None if m is None else m.group(1) == "#t"


def _first_method(mat: str, prop: str) -> Tuple[Optional[str], Optional[float]]:
    """A material property lists every method Fluent knows; the first one is the active one."""
    m = re.search(r"\(%s \(([\w-]+) \. ([^()]*)\)" % re.escape(prop), mat)
    if m is None:
        return None, None
    try:
        return m.group(1), float(m.group(2))
    except ValueError:
        return m.group(1), None


def parse_settings(rampant: str, threads: str) -> dict:
    """Settings from the two Scheme texts of a case file (see the module docstring)."""
    out: dict = {"zones": {}, "notes": []}
    heads = list(re.finditer(r"\(\d+ \((\d+) ([\w-]+) ([^\s()]+) \d+\)\(", threads))
    fluid_material = None
    for i, h in enumerate(heads):
        seg = threads[h.end():heads[i + 1].start() if i + 1 < len(heads) else len(threads)]
        ztype, name = h.group(2), h.group(3)
        if ztype in ("fluid", "solid"):
            m = re.search(r"\(material \. \"?([\w-]+)\"?\)", seg)
            if ztype == "fluid" and m and fluid_material is None:
                fluid_material = m.group(1)
            continue
        if ztype == "interior":
            continue
        z = {"type": ztype}
        for key, label in (("p0", "p0"), ("t0", "T0"), ("p", "p"), ("t", "T"), ("vmag", "speed"), ("m", "mach"),
                           ("turb-intensity", "turb_intensity"), ("turb-viscosity-ratio", "turb_viscosity_ratio")):
            v = _const(seg, key)
            if v is not None:
                z[label] = v
        d = [_const(seg, k) for k in ("ni", "nj", "nk")]
        if _const(seg, "direction-spec") == 0 and d[0] is not None:  # 0: direction vector, 1: normal to boundary
            z["direction"] = [x or 0.0 for x in d]
        out["zones"][name] = z
    op = re.search(r"\(operating-pressure %s\)" % _NUM, rampant)
    p_op = float(op.group(1)) if op else 101325.0
    out["operating_pressure"] = p_op
    for z in out["zones"].values():
        for k in ("p0", "p"):
            if k in z:
                z[k] += p_op
    # material of the fluid zone
    mat: dict = {"name": fluid_material}
    if fluid_material:
        i = rampant.find("(%s fluid" % fluid_material)
        if i >= 0:
            depth, j = 0, i
            while j < len(rampant):  # the balanced s-expression of this material
                depth += rampant[j] == "("
                depth -= rampant[j] == ")"
                j += 1
                if depth == 0:
                    break
            text = rampant[i:j]
            law, rho = _first_method(text, "density")
            mat["density_law"] = law
            if rho is not None:
                mat["density"] = rho
            for prop, key in (("specific-heat", "cp"), ("viscosity", "viscosity"), ("thermal-conductivity", "conductivity"),
                              ("molecular-weight", "molecular_weight")):
                method, v = _first_method(text, prop)
                if v is not None and method == "constant":
                    mat[key] = v
                elif method is not None:
                    out["notes"].append(f"{prop} of '{fluid_material}' uses '{method}', not read: the constant default is used")
    out["material"] = mat
    f = lambda k: bool(_flag(rampant, k))  # noqa: E731
    if f("rp-inviscid?"):
        turb = "inviscid"
    elif f("kw-sst-on?") and f("rp-kw?"):
        turb = "k-omega-sst"
    elif f("rp-kw?"):
        turb = "k-omega"
    elif f("rp-ke?"):
        turb = "k-epsilon"
    elif f("rp-sa?"):
        turb = "spalart-allmaras"
    else:
        turb = "laminar"
    out["models"] = {"turbulence": turb, "energy": f("rp-energy?"), "unsteady": f("rp-unsteady?"),
                     "axisymmetric": f("rp-axi?"), "pressure_based": f("rp-seg?")}
    return out


def read_case_settings(path: str) -> Optional[dict]:
    """Settings of a CFF ``.cas.h5`` file, or None when the file has none (a mesh-only file)."""
    import h5py

    with h5py.File(path, "r") as fh:
        g = fh.get("settings")
        if g is None or "Thread Variables" not in g or "Rampant Variables" not in g:
            return None

        def text(name):
            v = g[name][0]
            return v.decode("latin1") if isinstance(v, bytes) else str(v)

        return parse_settings(text("Rampant Variables"), text("Thread Variables"))


def choose_solver(mesh, settings: Optional[dict]) -> Tuple[str, List[str]]:
    """('compressible' | 'incompressible', reasons) for a mesh and its case settings (may be None)."""
    reasons: List[str] = []
    types = {z.type for z in mesh.boundary_zones()}
    law = ((settings or {}).get("material") or {}).get("density_law")
    if law == "ideal-gas":
        reasons.append("the fluid density follows the ideal-gas law in the Fluent case")
    if "pressure-far-field" in types:
        reasons.append("a pressure-far-field boundary (defined only for compressible flow)")
    return ("compressible" if reasons else "incompressible"), reasons


def compressible_spec(mesh, settings: dict) -> Tuple[dict, List[str]]:
    """Keyword arguments of :class:`flowtrace.compressible.case.CompressibleCase` for these settings,
    and one note per decision."""
    from ..compressible.case import FLUENT_TO_COMP

    notes: List[str] = list(settings.get("notes", []))
    mat = settings.get("material") or {}
    models = settings.get("models") or {}
    cp = float(mat.get("cp", 1006.43))
    R = R_UNIVERSAL / float(mat.get("molecular_weight", 28.966))
    gas = {"gamma": cp / (cp - R), "R": R, "mu": float(mat.get("viscosity", 1.7894e-5))}
    if "conductivity" in mat and mat["conductivity"] > 0:
        gas["prandtl"] = gas["mu"] * cp / mat["conductivity"]
    notes.append(f"gas: gamma = {gas['gamma']:.4g}, R = {R:.5g} J/(kg K), mu = {gas['mu']:.4g} Pa s (from the case's material '{mat.get('name')}')")
    turb = models.get("turbulence", "laminar")
    spec: dict = {"gas": gas, "boundaries": {}}
    if turb == "inviscid":
        spec["viscous"] = False
    elif turb == "laminar":
        spec["viscous"] = True
    else:
        spec["turbulence"] = "k-omega-sst"
        if turb != "k-omega-sst":
            notes.append(f"turbulence model '{turb}' of the case is not available in the density-based solver: k-omega SST is used")
    has_axis = any(z.type == "axis" for z in mesh.boundary_zones())
    if mesh.dim == 2 and (models.get("axisymmetric") or has_axis):
        spec["axisymmetric"] = True
        notes.append("axisymmetric: " + ("set in the case" if models.get("axisymmetric") else "the mesh has an axis boundary"))
    p_ref, T_ref = [], []
    for z in mesh.boundary_zones():
        s = (settings.get("zones") or {}).get(z.name, {})
        kind = FLUENT_TO_COMP.get(z.type, "wall")
        bc = {"kind": kind}
        if kind == "pressure-inlet":
            bc.update({k: s[k] for k in ("p0", "T0", "p") if k in s})
            if "direction" in s:
                bc["direction"] = s["direction"]
            p_ref.append(bc.get("p", bc.get("p0", 101325.0)))
            T_ref.append(bc.get("T0", 300.0))
            notes.append(f"{z.name}: pressure inlet, p0 = {bc.get('p0', 101325.0):.6g} Pa, T0 = {bc.get('T0', 300.0):.5g} K")
        elif kind == "pressure-outlet":
            bc.update({k: s[k] for k in ("p",) if k in s})
            if "T0" in s:
                bc["T"] = s["T0"]  # Fluent: backflow total temperature
            notes.append(f"{z.name}: pressure outlet, p = {bc.get('p', 101325.0):.6g} Pa")
        elif kind == "far-field":
            bc.update({k: s[k] for k in ("p", "T", "mach") if k in s})
            if "direction" in s:
                bc["direction"] = s["direction"]
            p_ref.append(bc.get("p", 101325.0))
            T_ref.append(bc.get("T", 300.0))
            notes.append(f"{z.name}: far field, Mach {bc.get('mach', 0.0):.4g}, p = {bc.get('p', 101325.0):.6g} Pa, T = {bc.get('T', 300.0):.5g} K")
        elif kind == "velocity-inlet":
            if "speed" in s:
                bc["velocity"] = None  # set from the face normals by the caller
                bc["_speed"] = s["speed"]
            if "T" in s:
                bc["T"] = s["T"]
        for k in ("turb_intensity",):
            if k in s:
                bc[k] = s[k]
        if "turb_viscosity_ratio" in s:
            bc["turb_visc_ratio"] = s["turb_viscosity_ratio"]
        bc.pop("_speed", None)
        if bc.get("velocity", 0) is None:
            bc.pop("velocity")
        spec["boundaries"][z.name] = bc
    if p_ref:
        spec["init_p"], spec["init_T"] = max(p_ref), max(T_ref)
    return spec, notes


def describe(settings: Optional[dict], mesh) -> Dict[str, object]:
    """A short summary for the dashboard: which solver the case needs and the values it will use."""
    solver, reasons = choose_solver(mesh, settings)
    out: Dict[str, object] = {"solver": solver, "reasons": reasons, "has_settings": settings is not None}
    if settings is not None:
        out["models"] = settings.get("models")
        out["material"] = settings.get("material")
        out["zones"] = settings.get("zones")
    return out

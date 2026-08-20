#!/usr/bin/env python3
"""
mesh_any.py

Fully config-driven Gmsh mesher for the Ansys -> Palace pipeline.
No device-specific constants: every body name, role, and mesh size is
read from device.step + device_config.json (export_for_palace.py).

Design rules (in priority order):

  1. ROLES. Solids are classified from material + solve_inside, never
     from a boundary that merely touches one of their faces (that is the
     body_roles corruption in schema<=3 exports -- the cavity tagged
     "impedance_sheet"). Sheets are classified from the boundary whose
     assignment names them. A model body with no derivable role is a
     HARD ERROR: silent dropping is how physics is lost.

        solid, solve_inside True                    -> dielectric
        solid, solve_inside False, sigma >= 1e10
              or material contains "perfect"        -> pec_solid
        solid, solve_inside False, 0 < sigma < 1e10 -> conductor_solid
        sheet, boundary role junction/pec_sheet/
              impedance_sheet/conductivity_sheet    -> that role
        model == False or role exclude              -> removed

  2. SIZES. mesh_operations[*].assignment is authoritative (never the
     operation's NAME -- "cav_pin1mm" is assigned to the pin). A body
     with no operation gets a documented fallback:

        sheet:               narrowest in-plane dimension / 6
        dielectric solid:    min(lambda_material(target)/12,
                                 middle bbox dimension / 2)
        conductor/pec solid: smallest bbox dimension / 3

     (For this device the fallbacks reproduce pin=1.0 mm and RR=0.05 mm
     exactly; the cavity fallback is ~7 mm vs the historical hand-typed
     1.0 mm -- AMR owns bulk refinement, features own the sizes.)

  3. GRADING. Every sized surface group gets a Distance+Threshold field:
     SizeMin=s out to 3*s, ramping to min(250*s, global max) by 500*s.
     This generalizes the junction fix that removed the aspect-208
     slivers, and applies it automatically to any fine feature in any
     future design. Background field = Min(all).

     The plateau (DistMin) is 3*s, NOT 10*s: the uniform-size blanket
     around a sheet costs ~8.5*k*Area/s^2 tets (k = plateau in element
     lengths, both sides), so k=10 on a centimeter-scale 20 um ribbon
     is ~3M tets from ONE sheet, while k=3 keeps the sliver protection
     at the junction (where Area is tiny and the cost is nil) at ~1/3
     the volume cost. If kappa_max jumps after a change here, the
     plateau got too thin for some feature -- raise toward 5 before
     touching anything else.

  4. GROUPS. Deterministic attribute order: sorted dielectric names,
     PEC_exterior, sorted impedance boundary names, sorted
     Conductivity::<body>, sorted sheet groups. Downstream MUST read the
     groups JSON -- never assume numbers.

  5. OVERLAP ARBITRATION. Co-planar overlapping sheets (e.g. a filling
     strip drawn over the ground plane -- legal in HFSS, where both are
     just PerfE metal) fragment into surfaces claimed by BOTH bodies. A
     mesh face with two boundary owners is illegal in Palace ("A
     non-periodic face cannot have multiple boundary elements"). The
     smaller sheet (more specific feature, finer mesh size) keeps the
     shared surface; the larger one loses it. Both are boundary
     conditions on the same metal, so the physics is identical.

Usage:
    python3 mesh_any.py [--step device.step]
                        [--config device_config_sweep.json]
                        [--tag LABEL]        (names all outputs
                                              device_<LABEL>.msh/.vtk +
                                              device_groups_<LABEL>.json)
                        [--size BODY=MM ...] (per-body size override,
                                              beats every config tier;
                                              repeatable)
                        [--out-msh device.msh] [--out-vtk device.vtk]
                        [--out-groups device_groups.json]
                        [--tolerance 0.02] [--no-grading]
                        [--mesh-order 1|2]   (2 = curvilinear elements)

Sweep example (three meshes from ONE config, no file editing):
    python mesh_any.py --tag Pin1_0p8 --size Pin_1=0.8
    python mesh_any.py --tag Pin1_0p4 --size Pin_1=0.4
    python mesh_any.py --tag Pin1_0p2 --size Pin_1=0.2

Requires alongside it: step_bodies.py and palace_matchers.py (flag-free
matcher library). build_from_single_step.py is NOT needed.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import sys
import time

import gmsh

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from step_bodies import read_step_bodies, match_gmsh_entities
# Long-form geometric matchers live in a flag-free library extracted
# from the legacy mesher -- no ENABLE_* toggles, no device names.
import palace_matchers as matchers

C_MM_PER_S = 2.99792458e11  # speed of light in mm/s

SIGMA_PEC_THRESHOLD = 1.0e10   # S/m; at/above this a metal is ideal
CURVATURE_ELEMENTS_PER_2PI = 24
GRADE_SAMPLING = 200
# Uniform-size plateau around each sized surface, in element lengths
# (Threshold DistMin = GRADE_PLATEAU * size). Cost scales as
# ~8.5 * GRADE_PLATEAU * Area / size^2 tets per sheet, both sides --
# keep this SMALL. 3 protects junction-edge element quality; 10 (the
# historical value) wrapped every centimeter-scale ribbon in millions
# of feature-size tets.
GRADE_PLATEAU = 3.0


# ---------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------

def bbox(dim, tag):
    return gmsh.model.getBoundingBox(dim, tag)


def dims_of(box):
    return (box[3] - box[0], box[4] - box[1], box[5] - box[2])


def bbox_volume(box):
    dx, dy, dz = dims_of(box)
    return abs(dx * dy * dz)


def bbox_inside(inner, outer, tol):
    return all(inner[i] >= outer[i] - tol for i in range(3)) and \
           all(inner[i + 3] <= outer[i + 3] + tol for i in range(3))


def material_info(config, name):
    wanted = str(name or "").strip().lower()
    for known, info in config.get("materials", {}).items():
        if str(known).strip().lower() == wanted and isinstance(info, dict):
            return info
    return {}


def object_info(config, name):
    info = config.get("objects", {}).get(name, {})
    return info if isinstance(info, dict) else {}


# ---------------------------------------------------------------------
# 1. roles
# ---------------------------------------------------------------------

def derive_solid_role(name, config):
    """Material + solve_inside decide a solid's role. Boundary roles do
    NOT: a boundary on one face of the cavity does not make the cavity a
    sheet (schema<=3 body_roles corruption)."""
    entry = object_info(config, name)
    mat_name = str(entry.get("material") or "").strip()
    mat = material_info(config, mat_name)
    sigma = float(mat.get("conductivity_S_per_m") or 0.0)

    if entry.get("solve_inside", True):
        return "dielectric"
    if sigma >= SIGMA_PEC_THRESHOLD or "perfect" in mat_name.lower():
        return "pec_solid"
    if sigma > 0.0:
        return "conductor_solid"
    print(f"  WARNING: solid '{name}' has solve_inside=False but material "
          f"'{mat_name}' carries no conductivity -- treating as PEC")
    return "pec_solid"


def derive_sheet_role(name, config):
    """A sheet's role comes from the boundary whose assignment names it."""
    for bc_name, bc in config.get("boundaries", {}).items():
        if not isinstance(bc, dict):
            continue
        if name in (bc.get("assignment") or []):
            return bc.get("role"), bc_name
    return None, None


def classify_bodies(step_bodies_map, config):
    """
    Returns {name: {"kind": solid|sheet, "role": ..., "boundary": ...}}
    for every model body, removing excluded/non-model ones from concern.
    Hard-fails on an unclassifiable model body.
    """
    roles = {}
    for name, info in step_bodies_map.items():
        entry = object_info(config, name)
        cfg_role = str(config.get("body_roles", {}).get(name, "")).strip()

        if entry.get("model") is False or cfg_role == "exclude":
            roles[name] = {"kind": info["geo_type"], "role": "exclude",
                           "boundary": None}
            continue

        if info["geo_type"] == "solid":
            role = derive_solid_role(name, config)
            if cfg_role and cfg_role != role and cfg_role not in (
                    "dielectric", "pec_solid", "conductor_solid"):
                print(f"  NOTE: config body_roles['{name}'] = '{cfg_role}' "
                      f"looks like a face-level boundary leaked onto a "
                      f"solid; using derived role '{role}' instead")
            roles[name] = {"kind": "solid", "role": role, "boundary": None}
        else:
            role, bc_name = derive_sheet_role(name, config)
            if role is None:
                raise RuntimeError(
                    f"Model sheet '{name}' has no boundary condition in "
                    f"device_config.json and therefore no physics role. "
                    f"Assign one in HFSS (PerfE / Impedance / Lumped RLC) "
                    f"or set the body non-model. Refusing to guess.")
            roles[name] = {"kind": "sheet", "role": role, "boundary": bc_name}

    return roles


# ---------------------------------------------------------------------
# 2. sizes
# ---------------------------------------------------------------------

def _as_name_list(assignment):
    if isinstance(assignment, str):
        try:
            parsed = ast.literal_eval(assignment)
            assignment = parsed if isinstance(parsed, (list, tuple)) \
                else [parsed]
        except (ValueError, SyntaxError):
            assignment = [assignment]
    return [str(item) for item in (assignment or [])]


def sizes_from_operations(config):
    """Body -> size in mm, from mesh_operations ASSIGNMENTS (never the
    operation name; 'cav_pin1mm' is assigned to the pin)."""
    sizes = {}
    for op_name, op in config.get("mesh_operations", {}).items():
        size = op.get("size_mm")
        if size is None:
            continue
        for body in _as_name_list(op.get("assignment")):
            prev = sizes.get(body)
            sizes[body] = float(size) if prev is None \
                else min(prev, float(size))
            print(f"  size {body} = {sizes[body]:g} mm  (op '{op_name}')")
    return sizes


def sizes_from_ansys_stats(config):
    """
    Body -> size from HFSS's own Mesh Statistics (RMS edge length per
    body), exported by export_for_palace.py as "ansys_mesh_stats".

    These are what HFSS's adaptive refinement actually settled on for
    THIS design, so they beat any analytic fallback -- but they only
    exist if the design was solved in HFSS at least once. Where several
    setups were exported, the most-refined one (largest tet count) wins.
    Per-body aggregates lose within-body adaptivity; Palace AMR
    recovers that.
    """
    stats = config.get("ansys_mesh_stats") or {}
    best, best_n, best_setup = None, -1, None
    for setup, data in stats.items():
        bodies = (data or {}).get("bodies") or {}
        n = sum(int(b.get("num_tets") or 0) for b in bodies.values())
        if n > best_n:
            best, best_n, best_setup = bodies, n, setup
    out = {}
    if best:
        for body, rec in best.items():
            rms = rec.get("rms_edge_mm")
            if rms and float(rms) > 0.0:
                out[str(body)] = float(rms)
        if out:
            print(f"  HFSS mesh statistics available: setup "
                  f"'{best_setup}' ({best_n} tets)")
    return out


def fallback_size(name, step_info, role, config):
    d = sorted(abs(v) for v in step_info["size"])
    target_ghz = float(
        config.get("palace_solver", {}).get("target_freq_GHz") or 5.0)

    if role in ("junction", "pec_sheet", "impedance_sheet",
                "conductivity_sheet"):
        # narrowest in-plane dimension (middle of sorted dims; smallest
        # is the ~zero sheet thickness), six elements across.
        size = max(d[1] / 6.0, 1.0e-5)
        rule = "in-plane/6"
    elif role == "dielectric":
        eps = float(material_info(
            config, object_info(config, name).get("material")
        ).get("permittivity") or 1.0)
        lam = C_MM_PER_S / (target_ghz * 1.0e9) / math.sqrt(max(eps, 1.0))
        size = max(min(lam / 12.0, d[1] / 2.0), 1.0e-4)
        rule = "min(lambda/12, mid-dim/2)"
    else:  # pec_solid / conductor_solid
        size = max(d[0] / 3.0, 1.0e-4)
        rule = "smallest-dim/3"

    print(f"  size {name} = {size:g} mm  (fallback: {rule})")
    return size


# ---------------------------------------------------------------------
# 3. grading fields
# ---------------------------------------------------------------------

def add_grading_field(surface_tags, size):
    """Distance+Threshold: size out to GRADE_PLATEAU*size, ramp to
    250*size by 500*size. This is the generalized junction fix -- it is
    what removed the aspect-208 needles at the thin lead.

    The plateau is deliberately short (3 layers, not 10): every layer
    of uniform feature-size tets around a sheet costs ~8.5*Area/size^2
    elements, so a wide plateau on a large fine sheet (a mm-scale
    feedline ribbon at 20 um) is millions of tets for zero physics.
    The ramp slope beyond the plateau is unchanged (growth ratio ~1.5),
    which is what actually prevents slivers."""
    distance = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(
        distance, "SurfacesList", [int(t) for t in surface_tags])
    gmsh.model.mesh.field.setNumber(distance, "Sampling", GRADE_SAMPLING)

    threshold = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold, "InField", distance)
    gmsh.model.mesh.field.setNumber(threshold, "SizeMin", size)
    gmsh.model.mesh.field.setNumber(threshold, "SizeMax", 250.0 * size)
    gmsh.model.mesh.field.setNumber(threshold, "DistMin",
                                    GRADE_PLATEAU * size)
    gmsh.model.mesh.field.setNumber(threshold, "DistMax", 500.0 * size)
    # CRITICAL: without this, the field returns SizeMax (= 250*size)
    # EVERYWHERE beyond DistMax, and the Min-combined background then
    # caps the WHOLE domain at the finest feature's 250*size. For a
    # 0.2 um junction that is a 50 um ceiling over the entire cavity
    # -- a billion-tet mesh. StopAtDistMax makes the field inert past
    # DistMax so each feature's grading stays local.
    gmsh.model.mesh.field.setNumber(threshold, "StopAtDistMax", 1)
    return threshold


def resolve_mesher_settings(config, args):
    """
    Mesher toggles resolve CONFIG-first, CLI-override:

        device_config.json:  "mesher": {
            "mesh_order": 1|2,                  # 2 = curvilinear
            "curvature_elements_per_2pi": 24,
            "grading": true
        }

    Absent block or keys -> defaults (order 1, curvature 24, grading
    on). CLI flags (--mesh-order, --curvature-segments, --no-grading)
    beat the config for one-off experiments without editing files.
    """
    cfg = config.get("mesher") or {}
    order = args.mesh_order if args.mesh_order is not None \
        else int(cfg.get("mesh_order", 1) or 1)
    if order not in (1, 2):
        raise RuntimeError(f"mesher.mesh_order must be 1 or 2, got "
                           f"{order!r}")
    grading = bool(cfg.get("grading", True)) and not args.no_grading
    curvature = args.curvature_segments \
        if args.curvature_segments is not None \
        else float(cfg.get("curvature_elements_per_2pi",
                           CURVATURE_ELEMENTS_PER_2PI) or
                   CURVATURE_ELEMENTS_PER_2PI)
    return order, grading, curvature


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    p.add_argument("--step", default="device.step")
    p.add_argument("--config", default="device_config_sweep.json")
    p.add_argument("--out-msh", default="device.msh")
    p.add_argument("--out-vtk", default="device.vtk")
    p.add_argument("--out-groups", default="device_groups.json")
    p.add_argument("--tag", default=None,
                   help="Label stamped into every output filename: "
                        "device_<tag>.msh/.vtk and "
                        "device_groups_<tag>.json. Explicit --out-* "
                        "arguments still win over the tag. Use one tag "
                        "per sweep point so runs never overwrite each "
                        "other.")
    p.add_argument("--size", action="append", default=[],
                   metavar="BODY=MM",
                   help="Override the mesh size of one body in mm, "
                        "beating every config tier (operation / stats / "
                        "fallback). Repeatable: "
                        "--size Pin_1=0.4 --size cavity=2.0. Unknown "
                        "body names are a hard error, so a typo cannot "
                        "silently sweep nothing.")
    p.add_argument("--scale", type=float, default=1.0,
                   metavar="FACTOR",
                   help="Multiply EVERY resolved body size by this "
                        "factor (after the operation/stats/fallback "
                        "tiers). Element count in size-controlled "
                        "regions falls like 1/FACTOR^3: --scale 1.5 "
                        "cuts a fine-feature-dominated mesh to ~30%% "
                        "of its count. Exempt: junction-role bodies "
                        "(the port sheet stays fully resolved) and "
                        "bodies given an explicit CLI --size, which "
                        "are taken literally.")
    p.add_argument("--tolerance", type=float, default=0.02)
    p.add_argument("--boolean-tol", type=float, default=0.0,
                   metavar="MM",
                   help="Fuzzy tolerance for OCC boolean operations "
                        "(Geometry.ToleranceBoolean), in model units "
                        "(mm). 0 = exact booleans (default). Use a "
                        "small value (1e-4 = 0.1 um) when the mesher "
                        "dies with 'PLC Error: ... intersect at "
                        "point': edges from two drawn objects that "
                        "cross without sharing a vertex get snapped "
                        "together and split at the crossing during "
                        "the fragment step, so the surface meshes "
                        "become conformal and 3D boundary recovery "
                        "succeeds. Keep it well below the smallest "
                        "real feature (JJ leads ~1 um), or "
                        "legitimate small gaps get welded shut.")
    p.add_argument("--no-grading", action="store_true")
    p.add_argument("--threads", type=int, default=0,
                   help="OpenMP threads for gmsh (0 = all cores). "
                        "Parallelizes 2D surface meshing and the HXT "
                        "3D algorithm; OCC booleans also run their "
                        "internal parallel mode.")
    p.add_argument("--algo3d", choices=("delaunay", "hxt"),
                   default="delaunay",
                   help="3D meshing algorithm. 'hxt' is gmsh's "
                        "parallel Delaunay (Mesh.Algorithm3D=10), "
                        "typically 3-10x faster on multi-core "
                        "machines for large meshes; it respects the "
                        "same background size fields. If HXT ever "
                        "fails on a geometry (rare), fall back to "
                        "'delaunay'.")
    p.add_argument("--no-vtk", action="store_true",
                   help="Skip writing the .vtk copy of the mesh. The "
                        "vtk is only for ParaView inspection; at "
                        "millions of nodes it doubles the (slow, "
                        "ASCII) file-writing time for nothing Palace "
                        "uses.")
    p.add_argument("--mesh-order", type=int, default=None, choices=(1, 2),
                   help="Geometric element order; overrides the config. "
                        "2 = curvilinear (10-node tets whose edges follow "
                        "curved CAD surfaces; Palace reports 'Mesh "
                        "curvature order: 2'). DOF count is unchanged -- "
                        "that is set by Solver.Order, not mesh order.")
    p.add_argument("--curvature-segments", type=float, default=None,
                   help="Elements per 2*pi of surface curvature; "
                        "overrides the config. With --mesh-order 2 this "
                        "can drop to ~8 for the same fidelity.")
    args = p.parse_args()

    # --tag renames every output the user did not name explicitly.
    if args.tag:
        tag = str(args.tag).strip()
        if args.out_msh == p.get_default("out_msh"):
            args.out_msh = f"device_{tag}.msh"
        if args.out_vtk == p.get_default("out_vtk"):
            args.out_vtk = f"device_{tag}.vtk"
        if args.out_groups == p.get_default("out_groups"):
            args.out_groups = f"device_groups_{tag}.json"
        print(f"Output tag '{tag}': {args.out_msh}, {args.out_vtk}, "
              f"{args.out_groups}")

    # Parse --size BODY=MM overrides (validated against the STEP below).
    cli_sizes = {}
    for item in args.size:
        body, sep, value = str(item).partition("=")
        body = body.strip()
        try:
            parsed = float(value)
            if not sep or not body or parsed <= 0.0:
                raise ValueError
        except ValueError:
            raise SystemExit(
                f"--size expects BODY=MM with MM a positive number, "
                f"got {item!r}")
        cli_sizes[body] = parsed

    with open(args.config, encoding="utf-8") as fh:
        config = json.load(fh)

    mesh_order, grading_on, curvature_segments = \
        resolve_mesher_settings(config, args)
    print(f"Mesher settings: order={mesh_order} "
          f"({'curvilinear' if mesh_order > 1 else 'straight-sided'}), "
          f"curvature={curvature_segments:g}/2pi, "
          f"grading={'on' if grading_on else 'OFF'}")

    step_map = read_step_bodies(args.step)
    print(f"\nBodies in {args.step}:")
    for n, i in sorted(step_map.items()):
        dx, dy, dz = i["size"]
        print(f"  {n:<16}{i['geo_type']:<7}{i['material']:<20}"
              f"{dx:.4g} x {dy:.4g} x {dz:.4g}")

    roles = classify_bodies(step_map, config)
    print("\nDerived roles:")
    for n, r in sorted(roles.items()):
        via = f" (boundary '{r['boundary']}')" if r["boundary"] else ""
        print(f"  {n:<16}{r['kind']:<7}-> {r['role']}{via}")

    # -- normalize a working config so the imported matchers see the
    #    corrected roles (protects against schema<=3 body_roles noise)
    work = json.loads(json.dumps(config))
    strategy = {"pec_solid": "surface_pec",
                "conductor_solid": "surface_conductor",
                "dielectric": "volume_dielectric"}
    for n, r in roles.items():
        work.setdefault("body_roles", {})[n] = r["role"]
        if n in work.get("objects", {}):
            work["objects"][n]["domain_role"] = r["role"]
            if r["role"] in strategy:
                work["objects"][n]["modeling_strategy"] = strategy[r["role"]]

    # Sizing tiers: HFSS mesh operation > HFSS mesh statistics (RMS
    # edge of the adapted mesh) > analytic fallback.
    sizes = sizes_from_operations(work)
    stats_sizes = sizes_from_ansys_stats(work)
    for n, r in roles.items():
        if r["role"] == "exclude" or n in sizes:
            continue
        if n in stats_sizes:
            sizes[n] = stats_sizes[n]
            print(f"  size {n} = {sizes[n]:g} mm  (HFSS mesh stats, "
                  f"RMS edge)")
        else:
            sizes[n] = fallback_size(n, step_map[n], r["role"], work)

    # CLI --size overrides beat every tier. A typo must fail loudly:
    # sweeping a body that does not exist is a wasted HPC run.
    unknown = sorted(b for b in cli_sizes if b not in step_map)
    if unknown:
        raise RuntimeError(
            f"--size given for unknown body/bodies {unknown}; bodies in "
            f"this STEP are {sorted(step_map)}")
    for body, s in cli_sizes.items():
        sizes[body] = s
        print(f"  size {body} = {s:g} mm  (CLI --size override)")

    # Global coarsening/refinement knob. Applied LAST, to the fully
    # resolved sizes, so the relative resolution ratios the HFSS
    # adaptive passes converged to are preserved -- everything gets
    # uniformly coarser/finer together. Junction sheets are exempt
    # (they are tiny, cost nothing, and the lumped port needs them
    # resolved); explicit CLI --size values are exempt because the
    # user asked for those numbers literally.
    if args.scale != 1.0:
        if args.scale <= 0.0:
            raise RuntimeError(f"--scale must be positive, "
                               f"got {args.scale:g}")
        print(f"\nApplying global size scale x{args.scale:g} "
              f"(junction bodies and CLI --size overrides exempt):")
        for body in sorted(sizes):
            if body in cli_sizes:
                print(f"  size {body:<24} kept at "
                      f"{sizes[body]:g} mm (CLI override)")
                continue
            if roles.get(body, {}).get("role") == "junction":
                print(f"  size {body:<24} kept at "
                      f"{sizes[body]:g} mm (junction)")
                continue
            old = sizes[body]
            sizes[body] = old * args.scale
            print(f"  size {body:<24} {old:g} -> "
                  f"{sizes[body]:g} mm")

    dielectrics = sorted(
        (n for n, r in roles.items() if r["role"] == "dielectric"),
        key=lambda n: -bbox_volume(step_map[n]["bbox"]))
    pec_solids = sorted(n for n, r in roles.items()
                        if r["role"] == "pec_solid")
    conductor_solids = sorted(n for n, r in roles.items()
                              if r["role"] == "conductor_solid")
    sheets = sorted(n for n, r in roles.items()
                    if r["kind"] == "sheet" and r["role"] != "exclude")
    excluded = sorted(n for n, r in roles.items() if r["role"] == "exclude")

    if not dielectrics:
        raise RuntimeError("No dielectric (solve_inside) volume found -- "
                           "nothing to solve.")

    gmsh.initialize()
    try:
        # ---- parallelism ------------------------------------------------
        # OpenMP threads cover 2D surface meshing (independent surfaces
        # in parallel) and the HXT 3D algorithm; OCCParallel lets
        # OpenCASCADE run its boolean/fragment internals multithreaded.
        threads = args.threads if args.threads > 0 else (os.cpu_count() or 1)
        gmsh.option.setNumber("General.NumThreads", threads)
        try:
            # Turn gmsh 'Error:' log lines into exceptions. Without
            # this, a volume that fails to tet-mesh is just a log line
            # and generate(3) returns "successfully" with that volume
            # empty. (2 = throw an exception on error.)
            gmsh.option.setNumber("General.AbortOnError", 2)
        except Exception:
            pass
        gmsh.option.setNumber("Geometry.OCCParallel", 1)
        print(f"\ngmsh threads: {threads}; OCC parallel booleans: on")
        if args.boolean_tol > 0:
            # Fuzzy booleans: OCC treats geometry closer than this as
            # coincident during cut/fragment. Crossing edges from two
            # separately drawn sheets get a shared vertex inserted at
            # the crossing instead of surviving as a self-intersection
            # that kills 3D boundary recovery ("PLC Error").
            gmsh.option.setNumber("Geometry.ToleranceBoolean",
                                  args.boolean_tol)
            print(f"OCC fuzzy boolean tolerance: {args.boolean_tol:g} mm "
                  f"(crossing edges within this distance are snapped "
                  f"and split conformally)")
        gmsh.model.add("palace_device")
        occ = gmsh.model.occ
        try:
            gmsh.option.setNumber("Geometry.OCCImportLabels", 1)
        except Exception:
            pass
        occ.importShapes(args.step, highestDimOnly=False)
        occ.synchronize()

        # ---- match imported entities to names -----------------------
        vol_names, _ = match_gmsh_entities(gmsh, step_map, 3,
                                           tolerance=args.tolerance)
        sheet_names_map, _ = match_gmsh_entities(gmsh, step_map, 2,
                                                 tolerance=args.tolerance)
        # A body may span SEVERAL Gmsh volumes: DuplicateAlongLine
        # children and multi-lump united objects (bondwire rows) export
        # as same-named STEP solids, and step_bodies merges them into
        # one logical body with per-lump bboxes. Accept lists here.
        volumes_by_name = {}
        for tag, name in vol_names.items():
            volumes_by_name.setdefault(name, []).append(tag)
        for name, tags in sorted(volumes_by_name.items()):
            if len(tags) > 1:
                print(f"  multi-lump body '{name}': {len(tags)} volumes")
        sheet_tags_by_name = {}
        for tag, name in sheet_names_map.items():
            sheet_tags_by_name.setdefault(name, []).append(tag)

        # ---- fail EARLY if a required sheet matched nothing ---------
        # Before any expensive boolean work, and with enough printed
        # context to see WHY: the parsed STEP face boxes of the failing
        # body next to the nearest gmsh surfaces (and whatever name
        # those surfaces DID match). A distance slightly above the
        # tolerance points at a bbox-parsing overshoot in step_bodies
        # (splines/arcs); a huge distance means the face genuinely is
        # not in the export.
        missing_sheets = [n for n in sheets
                          if not sheet_tags_by_name.get(n)]
        if missing_sheets:
            for n in missing_sheets:
                fboxes = (step_map[n].get("face_bboxes")
                          or [step_map[n]["bbox"]])
                print(f"\nsheet body '{n}' matched no gmsh surface.")
                print(f"  parsed STEP face boxes ({len(fboxes)}):")
                for fb in fboxes:
                    print(f"    ({fb[0]:.4f}, {fb[1]:.4f}, {fb[2]:.4f})"
                          f" -> ({fb[3]:.4f}, {fb[4]:.4f}, {fb[5]:.4f})")
                ranked = []
                for _, tag in gmsh.model.getEntities(2):
                    gb = gmsh.model.getBoundingBox(2, tag)
                    dist = min(max(abs(gb[i] - fb[i]) for i in range(6))
                               for fb in fboxes)
                    ranked.append((dist, tag, gb))
                print(f"  nearest gmsh surfaces "
                      f"(tolerance is {args.tolerance} mm):")
                for dist, tag, gb in sorted(ranked)[:3]:
                    owner = sheet_names_map.get(tag)
                    owned = (f"matched to '{owner}'" if owner
                             else "unmatched")
                    print(f"    surface {tag} [{owned}]  "
                          f"({gb[0]:.4f}, {gb[1]:.4f}, {gb[2]:.4f}) -> "
                          f"({gb[3]:.4f}, {gb[4]:.4f}, {gb[5]:.4f})  "
                          f"distance {dist:.4f} mm")
            raise RuntimeError(
                f"sheet bodies matched no STEP surface: "
                f"{missing_sheets} -- see the box comparison above")

        # ---- remove excluded ----------------------------------------
        to_remove = []
        for n in excluded:
            for t in volumes_by_name.pop(n, []):
                to_remove.append((3, t))
            for t in sheet_tags_by_name.pop(n, []):
                to_remove.append((2, t))
        # sheets present in the STEP but not model bodies (solid faces
        # already matched by name are fine; unmatched ones are faces of
        # solids and are left alone)
        if to_remove:
            occ.remove(to_remove, recursive=True)
            occ.synchronize()
            print(f"\nRemoved excluded bodies: {excluded}")

        missing = [n for n in dielectrics + pec_solids + conductor_solids
                   if n not in volumes_by_name]
        if missing:
            raise RuntimeError(f"STEP volumes not matched: {missing}")

        # ---- record conductor faces BEFORE booleans -----------------
        # The legacy matcher expects ONE volume tag per name. Multi-lump
        # conductor solids (finite conductivity) would need per-lump
        # face recording -- not supported yet, and nothing in the lab's
        # designs does it (bondwire rows are PEC, not lossy). Fail
        # loudly rather than record only the first lump's faces.
        if conductor_solids:
            multi_lump_conductors = [
                n for n in conductor_solids
                if len(volumes_by_name.get(n, [])) > 1]
            if multi_lump_conductors:
                raise RuntimeError(
                    f"conductor (finite-sigma) solids with multiple "
                    f"same-named lumps are not supported: "
                    f"{multi_lump_conductors}. Separate Bodies / rename "
                    f"in Ansys, or make them perfect conductors.")
        first_volume_by_name = {n: tags[0]
                                for n, tags in volumes_by_name.items()}
        conductor_records = matchers.record_surface_conductor_geometry(
            work, first_volume_by_name) if conductor_solids else {}

        # ---- nested-dielectric cuts (outer minus inner) -------------
        current = {n: [(3, t) for t in volumes_by_name[n]]
                   for n in dielectrics}
        for i, outer in enumerate(dielectrics):
            inner_tools = [
                (3, t) for j in dielectrics[i + 1:]
                if bbox_inside(step_map[j]["bbox"], step_map[outer]["bbox"],
                               args.tolerance)
                for t in volumes_by_name[j]]
            if inner_tools:
                out, _ = occ.cut(current[outer], inner_tools,
                                 removeObject=True, removeTool=False)
                occ.synchronize()
                current[outer] = [dt for dt in out if dt[0] == 3]

        # ---- cut conductor + PEC solids out of every dielectric -----
        tools = [(3, t) for n in pec_solids + conductor_solids
                 for t in volumes_by_name[n]]
        flat, owners = [], []
        for n in dielectrics:
            for dt in current[n]:
                flat.append(dt)
                owners.append(n)
        if tools:
            _, cut_map = occ.cut(flat, tools, removeObject=True,
                                 removeTool=True)
            occ.synchronize()
            current = {n: [] for n in dielectrics}
            for idx, desc in enumerate(cut_map[:len(flat)]):
                current[owners[idx]].extend(
                    dt for dt in desc if dt[0] == 3)

        # ---- conformalize dielectric interfaces ---------------------
        flat, owners = [], []
        for n in dielectrics:
            for dt in current[n]:
                flat.append(dt)
                owners.append(n)
        if len(dielectrics) > 1 and len(flat) > 1:
            _, frag_map = occ.fragment(flat[:1], flat[1:],
                                       removeObject=True, removeTool=True)
            occ.synchronize()
            current = {n: [] for n in dielectrics}
            for idx, desc in enumerate(frag_map[:len(flat)]):
                current[owners[idx]].extend(
                    dt for dt in desc if dt[0] == 3)

        # ---- imprint sheets -----------------------------------------
        active_sheet_tags = [t for n in sheets
                             for t in sheet_tags_by_name.get(n, [])]
        for n in sheets:
            if not sheet_tags_by_name.get(n):
                raise RuntimeError(f"sheet body '{n}' matched no STEP "
                                   f"surface")
        flat, owners = [], []
        for n in dielectrics:
            for dt in current[n]:
                flat.append(dt)
                owners.append(n)
        sheet_descendants = {t: set() for t in active_sheet_tags}
        if active_sheet_tags:
            _, frag_map = occ.fragment(
                flat, [(2, t) for t in active_sheet_tags],
                removeObject=True, removeTool=True)
            occ.synchronize()
            current = {n: [] for n in dielectrics}
            for idx, desc in enumerate(frag_map):
                if idx < len(flat):
                    current[owners[idx]].extend(
                        dt for dt in desc if dt[0] == 3)
                else:
                    orig = active_sheet_tags[idx - len(flat)]
                    sheet_descendants[orig].update(
                        t for d, t in desc if d == 2)

        final_by_diel = {n: {t for _, t in current[n]} for n in dielectrics}
        all_vols = {t for _, t in gmsh.model.getEntities(3)}
        assigned = set().union(*final_by_diel.values())
        unowned = all_vols - assigned
        if unowned:
            raise RuntimeError(f"unassigned volume fragments: "
                               f"{sorted(unowned)}")

        final_sheets = {n: set() for n in sheets}
        for n in sheets:
            for orig in sheet_tags_by_name.get(n, []):
                final_sheets[n] |= sheet_descendants.get(orig, set())

        # ---- sheet adjacency ----------------------------------------
        print("\nSheet adjacency:")
        one_sided_contact = set()
        for n in sheets:
            keep = set()
            for s in sorted(final_sheets[n]):
                up, _ = gmsh.model.getAdjacencies(2, s)
                if len(up) == 2:
                    keep.add(s)
                elif len(up) == 1:
                    if roles[n]["role"] == "junction":
                        raise RuntimeError(
                            f"junction sheet '{n}' surface {s} is "
                            f"one-sided -- it would be an open port")
                    # a conductor sheet on a wall is PEC either way;
                    # keep it as its own named group
                    keep.add(s)
                    print(f"  {n}: surface {s} one-sided (kept in group)")
                else:
                    raise RuntimeError(f"sheet '{n}' surface {s}: "
                                       f"{len(up)} adjacent volumes")
            final_sheets[n] = keep

        # ---- deduplicate overlapping sheets -------------------------
        # Co-planar overlapping sheets (e.g. a filling strip drawn over
        # the ground plane -- legal in HFSS, where both are just PerfE
        # metal) fragment into surfaces claimed by BOTH bodies. A mesh
        # face with two boundary owners is illegal in Palace ("A
        # non-periodic face cannot have multiple boundary elements!").
        # Arbitrate: the smaller sheet (more specific feature, finer
        # mesh size) keeps the shared surface; the larger one loses it.
        # Both are boundary conditions on the same metal, so the
        # physics is identical either way.
        def _sheet_area(name):
            d = sorted(abs(v) for v in step_map[name]["size"])
            return d[1] * d[2]   # two largest dims; smallest ~ 0 thickness

        claimed = {}
        for n in sorted(sheets, key=_sheet_area):
            for s in sorted(final_sheets[n]):
                if s in claimed:
                    print(f"  overlap: surface {s} in both "
                          f"'{claimed[s]}' and '{n}' -- kept in "
                          f"'{claimed[s]}' (smaller body wins)")
                    final_sheets[n].discard(s)
                else:
                    claimed[s] = n

        all_sheet_surfs = set().union(*final_sheets.values()) \
            if final_sheets else set()

        # ---- face-level boundary matching ---------------------------
        # legacy matchers expect (config, final_cavity, final_chip, excl);
        # generalized: pass the union split as (all diel, empty)
        all_diel_tags = set().union(*final_by_diel.values())
        print("\nImpedance boundary matching:")
        # NOTE: the legacy matcher's owner check compares against the
        # literal names "cavity"/"chip"; pass the full dielectric set as
        # BOTH so a face owned by either (or any) dielectric matches.
        imped = matchers.match_impedance_surfaces(
            work, all_diel_tags, all_diel_tags, all_sheet_surfs)
        all_imped = set().union(*imped.values()) if imped else set()

        print("\nFinite-conductivity matching:")
        conductor_surfs = matchers.match_surface_conductor_surfaces(
            conductor_records, all_diel_tags, set(),
            all_sheet_surfs | all_imped) if conductor_records else {}
        all_cond = set().union(*conductor_surfs.values()) \
            if conductor_surfs else set()

        # ---- PEC exterior = every remaining one-sided surface -------
        pec_ext = set()
        for _, s in gmsh.model.getEntities(2):
            if s in all_sheet_surfs or s in all_imped or s in all_cond:
                continue
            up, _ = gmsh.model.getAdjacencies(2, s)
            if len(up) == 1:
                pec_ext.add(s)
            elif len(up) > 2:
                raise RuntimeError(f"non-manifold surface {s}")

        # ---- physical groups, deterministic order -------------------
        groups = {}
        for n in sorted(dielectrics):
            groups[n] = gmsh.model.addPhysicalGroup(
                3, sorted(final_by_diel[n]), name=n)
        groups["PEC_exterior"] = gmsh.model.addPhysicalGroup(
            2, sorted(pec_ext), name="PEC_exterior")
        for bn in sorted(imped):
            if imped[bn]:
                groups[bn] = gmsh.model.addPhysicalGroup(
                    2, sorted(imped[bn]), name=bn)
        for cn in sorted(conductor_surfs):
            if conductor_surfs[cn]:
                groups[f"Conductivity::{cn}"] = gmsh.model.addPhysicalGroup(
                    2, sorted(conductor_surfs[cn]),
                    name=f"Conductivity::{cn}")
        for n in sorted(sheets):
            if final_sheets[n]:
                groups[n] = gmsh.model.addPhysicalGroup(
                    2, sorted(final_sheets[n]), name=n)

        print("\nPhysical groups:")
        for n, a in groups.items():
            print(f"  {n:<28}-> attribute {a}")

        # ---- sizing + grading ---------------------------------------
        feature_sizes = {}   # {frozenset(surfaces): size}
        for n in sheets:
            if final_sheets[n]:
                feature_sizes[frozenset(final_sheets[n])] = sizes[n]
        for cn, surfs in conductor_surfs.items():
            if surfs:
                # NOTE: do not use sizes.get(cn, fallback_size(...)) --
                # Python evaluates (and prints) the fallback even when
                # the size already exists, leaving a lying log line.
                if cn in sizes:
                    feature_sizes[frozenset(surfs)] = sizes[cn]
                else:
                    feature_sizes[frozenset(surfs)] = fallback_size(
                        cn, step_map[cn], "conductor_solid", work)
        # PEC solids: their faces are inside pec_ext; size them via a
        # field on the faces recovered from their original bbox.
        # Per-LUMP boxes, not the union box: the union of a bondwire
        # row spans the whole row, and using it would capture unrelated
        # surfaces sitting between the wires.
        for n in pec_solids:
            lump_boxes = step_map[n].get("lump_bboxes") \
                or [step_map[n]["bbox"]]
            faces = {s for s in pec_ext
                     if any(bbox_inside(bbox(2, s), lb, args.tolerance)
                            for lb in lump_boxes)}
            if faces and n in sizes:
                feature_sizes[frozenset(faces)] = sizes[n]

        global_max = max(sizes[n] for n in dielectrics)
        global_min = min(feature_sizes.values(), default=global_max)

        gmsh.option.setNumber("Mesh.MeshSizeMin", global_min)
        gmsh.option.setNumber("Mesh.MeshSizeMax", global_max)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
        # Curvature-driven refinement is expressed in SEGMENTS PER
        # 2*pi, not millimetres, so --scale does not reach it through
        # the size dict. Left unscaled, the ~100 curved bondwire/pin
        # surfaces keep their fine rings and dominate a scaled mesh
        # (measured: SCALE=1.8 removed only 44% of elements instead of
        # the predicted 83% -- the survivors were curvature-driven).
        # A global coarsening must therefore divide the segment count
        # too. Floor of 6 keeps cylinders recognizably round; at mesh
        # order 2 the curved elements still follow the true surface.
        effective_curvature = curvature_segments
        if args.scale != 1.0 and curvature_segments:
            effective_curvature = max(curvature_segments / args.scale, 6.0)
            print(f"\nCurvature refinement scaled with --scale: "
                  f"{curvature_segments:g} -> {effective_curvature:g} "
                  f"segments per 2*pi")
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature",
                              effective_curvature)

        print("\nSizing:")
        print(f"  global: min {global_min:g}  max {global_max:g} mm")
        # Consolidate grading fields: ONE Distance+Threshold pair per
        # DISTINCT size value, not one per feature group. The mesher
        # evaluates the Min of ALL fields at every candidate point
        # during refinement, so field count is a direct multiplier on
        # sizing cost; a design with 60 bodies at 8 distinct sizes
        # needs 8 fields, not 60. Surfaces sharing a size share a
        # field -- the Distance field takes the union list.
        by_size = {}
        for surfs, s in feature_sizes.items():
            matchers.set_surface_point_size(sorted(surfs), s)
            by_size.setdefault(s, set()).update(surfs)
            print(f"  {len(surfs)} surface(s) at {s:g} mm"
                  + (f", graded to {min(250*s, global_max):g} mm"
                     if grading_on else ""))
        fields = []
        if grading_on:
            for s in sorted(by_size):
                fields.append(add_grading_field(sorted(by_size[s]), s))
            print(f"  grading fields: {len(fields)} (one per distinct "
                  f"size; {len(feature_sizes)} feature groups "
                  f"consolidated)")
        if fields:
            if len(fields) == 1:
                bg = fields[0]
            else:
                bg = gmsh.model.mesh.field.add("Min")
                gmsh.model.mesh.field.setNumbers(bg, "FieldsList", fields)
            gmsh.model.mesh.field.setAsBackgroundMesh(bg)

        if mesh_order > 1:
            # Curvilinear elements: nodes are added on curved CAD
            # surfaces so element edges follow the true geometry
            # (cylindrical pins) instead of chords.
            # HighOrderOptimize=1 (LOCAL per-element optimization).
            # Do NOT use 2/3 (elastic analogy): the elastic smoother
            # assembles one global PETSc system over ALL high-order
            # nodes, and beyond a few million elements its matrix
            # preallocation overflows the 32-bit PetscInt in gmsh's
            # bundled PETSc ("Integer too long for PetscInt"). The
            # local optimizer fixes distorted curved elements one at a
            # time with no global solve. Watch the gmsh log for any
            # 'negative Jacobian' report -- that is a broken mesh, do
            # not run Palace on it.
            gmsh.option.setNumber("Mesh.ElementOrder", mesh_order)
            gmsh.option.setNumber("Mesh.HighOrderOptimize", 1)
            print(f"\nCurvilinear meshing: element order "
                  f"{mesh_order}, high-order optimization: local")

        if args.algo3d == "hxt":
            # HXT: gmsh's parallel Delaunay kernel. Same background
            # size fields, multithreaded refinement.
            gmsh.option.setNumber("Mesh.Algorithm3D", 10)
            print(f"3D algorithm: HXT (parallel, {threads} threads)")

        t_gen = time.time()
        try:
            gmsh.model.mesh.generate(3)
        except Exception as exc:
            msg = str(exc)
            if "PLC" in msg or "intersect" in msg.lower():
                print(
                    "\n" + "=" * 66 +
                    "\n3D meshing failed on a boundary self-"
                    "intersection: two edges or\nfacets in the "
                    "imported geometry cross without sharing a "
                    "vertex, so\nno conformal tet mesh exists. This "
                    "is an input-geometry defect,\nnot a mesh-size "
                    "problem. In order of preference:\n"
                    "  1. Fix the drawing in HFSS. Run\n"
                    "       python find_crossing_bodies.py "
                    "<step> <x> <y> <z>\n"
                    "     with the point printed in the error above "
                    "to identify the\n     offending objects, then "
                    "unite/trim them so edges coincide.\n"
                    "  2. Re-run this mesher with --boolean-tol "
                    "1e-4 (0.1 um): OCC's\n     fuzzy booleans snap "
                    "the crossing edges together and split\n     "
                    "them at the crossing, making the surfaces "
                    "conformal.\n"
                    "  3. If it still fails, raise to 1e-3 -- but "
                    "stay well below\n     the smallest real gap in "
                    "the design.\n" + "=" * 66)
            raise
        print(f"\nMesh generation wall time: {time.time() - t_gen:.1f} s")

        # ---- validation ---------------------------------------------
        total = sum(len(t) for t in gmsh.model.mesh.getElements(3)[1])
        owned = 0
        empty_volumes = []
        for n in dielectrics:
            for v in gmsh.model.getEntitiesForPhysicalGroup(3, groups[n]):
                cnt = sum(len(t) for t in
                          gmsh.model.mesh.getElements(3, v)[1])
                if cnt == 0:
                    box = gmsh.model.getBoundingBox(3, v)
                    empty_volumes.append((n, v, box))
                owned += cnt
        # gmsh can FAIL to tet-mesh a volume, log an Error line, move on
        # to the next volume, and return from generate(3) as if all is
        # well -- leaving a mesh that is all skin and no flesh (500k
        # surface triangles, 1k tets; Vostok-3 2026-08-18). Palace
        # aborts on such a mesh (MFEM STable3D). Refuse to write it.
        if empty_volumes:
            print("\nVolumes with ZERO tetrahedra (3D meshing failed "
                  "for them -- scroll up for gmsh 'Error:' lines with "
                  "the reason):")
            for n, v, box in empty_volumes:
                print(f"  volume {v} of '{n}': "
                      f"({box[0]:.3f}, {box[1]:.3f}, {box[2]:.3f}) -> "
                      f"({box[3]:.3f}, {box[4]:.3f}, {box[5]:.3f})")
            raise RuntimeError(
                f"{len(empty_volumes)} volume(s) have no tets -- the "
                f"3D mesh is incomplete and Palace would abort on it. "
                f"The usual cause is corrupt/self-intersecting "
                f"geometry surviving the booleans (crossing sheet "
                f"edges, fuzzy-boolean damage). Fix the source "
                f"geometry; do not submit this mesh.")
        n_tris = sum(len(t) for t in gmsh.model.mesh.getElements(2)[1])
        if total < n_tris:
            raise RuntimeError(
                f"only {total} tets vs {n_tris} surface triangles -- "
                f"a healthy 3D mesh has far more tets than boundary "
                f"triangles. 3D meshing largely failed; see gmsh "
                f"'Error:' lines above. Do not submit this mesh.")
        print(f"\nMesh: {total} tets, "
              f"{len(gmsh.model.mesh.getNodes()[0])} nodes")
        if total != owned:
            raise RuntimeError(f"orphan tetrahedra: {owned}/{total}")

        # ---- GPU / AMR budget report --------------------------------
        # Live-calibrated on vanda 2xA40 (48 GB each): ~8M ND unknowns
        # is the hard VRAM ceiling (7.23M ran; 23M+ OOMed), and the
        # validated configs cap AMR MaxSize at 7M. Each AMR pass grows
        # the mesh ~1.5x, so an initial mesh supports
        #   floor( log(7M / initial_unknowns) / log(1.5) )
        # FULL refinement passes before hitting the cap. Unknowns/tet
        # measured on this pipeline's meshes: p=1 -> 1.31, p=2 -> 6.8.
        print("\nGPU / AMR budget (2xA40 node, 7M-unknown AMR cap, "
              "~1.5x growth per pass):")
        for p_order, per_tet in ((1, 1.31), (2, 6.8)):
            unk = total * per_tet
            if unk > 7.0e6:
                verdict = ("TOO BIG even without AMR -- will OOM "
                           "on the GPU node")
                passes = -1
            else:
                passes = int(math.floor(
                    math.log(7.0e6 / unk) / math.log(1.5)))
                if passes >= 3:
                    verdict = (f"fits {passes} full AMR passes "
                               f"-- GOOD for >=3 its")
                elif passes > 0:
                    verdict = (f"only {passes} full AMR pass(es) "
                               f"-- coarsen (raise SCALE) for 3")
                else:
                    verdict = ("fits with AMR OFF only "
                               "(MaxIts 0) -- coarsen for AMR")
            print(f"  Order {p_order}: ~{unk / 1e6:.2f}M unknowns "
                  f"-> {verdict}")
        print("  (targets: <=2.07M unknowns for 3 passes; that is "
              "~1.58M tets at Order 1, ~305k tets at Order 2)")
        for n in sheets:
            tris = matchers.count_2d_elements(sorted(final_sheets[n]))
            print(f"  {n:<20}{tris} triangles")
            if final_sheets[n] and tris == 0:
                raise RuntimeError(f"sheet '{n}' has surfaces but no "
                                   f"triangles")

        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        t_io = time.time()
        gmsh.write(args.out_msh)
        if args.no_vtk:
            print("  (.vtk skipped: --no-vtk)")
        else:
            gmsh.write(args.out_vtk)
        print(f"  mesh file write time: {time.time() - t_io:.1f} s")

        # Groups JSON: flat name->attribute map (backward compatible)
        # plus a "_meta" block telling the config writer what each group
        # IS, so it needs no hardcoded body or material names either.
        meta = {"format": 2, "mesh": os.path.basename(args.out_msh),
                "groups": {}}
        for n in sorted(dielectrics):
            meta["groups"][n] = {
                "dim": 3, "role": "dielectric",
                "material": object_info(work, n).get("material")}
        meta["groups"]["PEC_exterior"] = {"dim": 2, "role": "pec_exterior"}
        for bn in sorted(imped):
            if imped[bn]:
                meta["groups"][bn] = {
                    "dim": 2, "role": "impedance", "boundary": bn}
        for cn in sorted(conductor_surfs):
            if conductor_surfs[cn]:
                meta["groups"][f"Conductivity::{cn}"] = {
                    "dim": 2, "role": "conductor_surface", "body": cn,
                    "material": object_info(work, cn).get("material")}
        for n in sorted(sheets):
            if final_sheets[n]:
                meta["groups"][n] = {
                    "dim": 2, "role": roles[n]["role"],
                    "boundary": roles[n]["boundary"]}
        with open(args.out_groups, "w") as fh:
            json.dump({**groups, "_meta": meta}, fh, indent=2)
        print(f"\nWritten: {args.out_msh}, {args.out_vtk}, "
              f"{args.out_groups}")
        print("Downstream config generation MUST read the groups JSON.")

    finally:
        gmsh.finalize()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
write_palace_config.py

Fully general Palace eigenmode config writer for the Ansys -> Palace
pipeline. No device-specific names anywhere: every material, boundary,
and junction is resolved from

  1. device_groups.json  (mesh_any.py: name -> attribute, plus "_meta"
     describing what each group IS), and
  2. device_config.json  (export_for_palace.py: materials, boundaries,
     junction values, palace_solver settings).

Generality rules:

  * MATERIALS: one Domains.Materials entry per dielectric group, with
    permittivity/permeability/loss tangent from the body's own material
    record. No vacuum/sapphire special cases.
  * JUNCTIONS: one LumpedPort per junction group. L/C/R from the Lumped
    RLC boundary record. Direction is DERIVED from the junction sheet's
    exported face bbox (long in-plane axis) -- for the known device this
    reproduces the validated [0,1,0]. Printed prominently; --jj-direction
    overrides. No junction in the design -> no port, no error.
  * IMPEDANCE / CONDUCTIVITY: from the boundary record / the body's
    material record, matched to the group by name. A bulk-conductive
    object carrying an HFSS Impedance sheet is a hard error (copper is
    not 50 ohm/sq).
  * FULL-COVERAGE GATE: every mesh group must be consumed by exactly one
    material or boundary entry; an unconsumed group is fatal (silently
    dropped physics is how a junction once got shorted to PEC).
  * SOLVER: palace_solver from the config, with the settings validated
    on vanda 2026-08-04 as defaults: SuperLU + MGMaxLevels 1 +
    PartialAssemblyOrder 100 (direct solve on the fine level; AMS
    stalled at reduction factor 0.97), RefineNonlinear False (NLEPS
    diverges on v0.17.0-67; cost is sqrt(f/target) understated conductor
    loss above the target -- printed when relevant).

Works without "_meta" too (older mesher output): roles are then derived
from device_config.json the same way mesh_any.py derives them.

Usage:
    python write_palace_config.py
    python write_palace_config.py --mesh device.msh \
        --groups device_groups.json --device-config device_config.json \
        --output palace_config.json [--jj-direction DX DY DZ]
        [--refine-nonlinear] [--linear-type SuperLU|STRUMPACK|MUMPS|Default]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from typing import Any

DEFAULT_MESH = "device.msh"
DEFAULT_GROUPS = ("device_groups.json", "physical_groups.json")
DEFAULT_DEVICE_CONFIG = "device_config.json"
DEFAULT_OUTPUT = "palace_config.json"

# Validated on vanda (jobs 1285306/1285309, 2026-08-04): direct solve on
# the fine level; Palace's default AMS+GMRES stalled at reduction 0.97.
SOLVER_DEFAULTS = {
    "target_freq_GHz": 3.5,
    "n_modes": 5,
    "solver_order": 2,
    "eig_tol": 1.0e-5,
    "linear_tol": 1.0e-8,
    "eig_max_its": 500,
    "linear_max_its": 500,
    "amr_tol": 5.0e-4,
    "amr_max_its": 3,
    "amr_update_fraction": 0.7,
    "amr_max_size": 4_000_000,
    "linear_type": "SuperLU",
    "ksp_type": "Default",
    "mg_max_levels": 1,
    "partial_assembly_order": 100,
    "refine_nonlinear": False,
}

CONDUCTOR_SHEET_ROLES = {"conductivity_sheet"}
PEC_SHEET_ROLES = {"pec_sheet"}
JUNCTION_ROLES = {"junction"}
IMPEDANCE_ROLES = {"impedance", "impedance_sheet"}


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

def load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return data


def parse_numeric(value, default=None):
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    m = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
                  str(value))
    return float(m.group(0)) if m else default


def ci_record(records, wanted):
    wanted = str(wanted or "").strip().lower()
    for name, info in (records or {}).items():
        if str(name).strip().lower() == wanted:
            return info if isinstance(info, dict) else {}
    return {}


def unique_ints(values):
    return sorted({int(v) for v in values})


def split_groups(data):
    """Flat name->attribute map and the optional _meta block."""
    meta = data.get("_meta") if isinstance(data.get("_meta"), dict) else {}
    groups = {}
    for name, value in data.items():
        if isinstance(value, bool) or name == "_meta":
            continue
        if isinstance(value, int):
            groups[str(name)] = value
        elif isinstance(value, float) and value.is_integer():
            groups[str(name)] = int(value)
    if not groups:
        raise RuntimeError("no integer physical-group attributes found")
    return groups, (meta.get("groups") or {})


def derive_meta(groups, device):
    """
    Reconstruct group roles for older mesher output without _meta, using
    the same rules mesh_any.py uses to create the groups.
    """
    objects = device.get("objects", {})
    boundaries = device.get("boundaries", {})
    meta = {}
    for name in groups:
        if name == "PEC_exterior":
            meta[name] = {"dim": 2, "role": "pec_exterior"}
            continue
        if name.startswith("Conductivity::"):
            body = name[len("Conductivity::"):]
            meta[name] = {"dim": 2, "role": "conductor_surface",
                          "body": body,
                          "material": ci_record(objects, body).get(
                              "material")}
            continue
        if name in boundaries:
            role = str(boundaries[name].get("role") or "")
            meta[name] = {"dim": 2,
                          "role": "impedance" if role in IMPEDANCE_ROLES
                          else (role or "impedance"),
                          "boundary": name}
            continue
        obj = ci_record(objects, name)
        if obj.get("dimension") == 3:
            if obj.get("solve_inside", True):
                meta[name] = {"dim": 3, "role": "dielectric",
                              "material": obj.get("material")}
            else:
                raise RuntimeError(
                    f"group '{name}' is a solve_inside=False solid -- a "
                    f"conductor should appear as Conductivity::{name} or "
                    f"inside PEC_exterior, not as its own volume group")
            continue
        # a sheet body: role from the boundary that names it
        found = None
        for bc_name, bc in boundaries.items():
            if isinstance(bc, dict) and name in (bc.get("assignment")
                                                 or []):
                found = {"dim": 2, "role": str(bc.get("role") or ""),
                         "boundary": bc_name}
                break
        if found is None:
            raise RuntimeError(
                f"group '{name}' matches no object, boundary, or known "
                f"role in device_config.json -- cannot assign physics")
        meta[name] = found
    return meta


def derive_port_direction(bc_info):
    """
    Current direction across a junction sheet, from the exported face
    bbox: the LONG in-plane axis (current flows lead-to-lead along the
    sheet's length). Reproduces the validated [0,1,0] for the known
    device (jj 1um x 1.9um, long axis y). Only axis-aligned sheets are
    derivable; otherwise the caller must supply --jj-direction.
    """
    for face in bc_info.get("assignment_faces") or []:
        box = face.get("bbox_sample_mm")
        if not box or len(box) != 6:
            continue
        dims = [abs(box[3] - box[0]), abs(box[4] - box[1]),
                abs(box[5] - box[2])]
        axis = max(range(3), key=lambda i: dims[i])
        if dims[axis] <= 0.0:
            continue
        direction = [0.0, 0.0, 0.0]
        direction[axis] = 1.0
        return direction, f"long in-plane axis of exported face bbox " \
                          f"(dims {dims[0]:g} x {dims[1]:g} x " \
                          f"{dims[2]:g} mm)"
    return None, None


def conductive_assignments(bc_info, device):
    """Assigned objects whose materials are bulk conductors."""
    out = []
    for name in bc_info.get("assignment", []) or []:
        obj = ci_record(device.get("objects", {}), str(name))
        if not obj:
            continue
        mat_name = str(obj.get("material") or "").strip()
        sigma = parse_numeric(ci_record(device.get("materials", {}),
                                        mat_name)
                              .get("conductivity_S_per_m"))
        if sigma is not None and sigma > 0.0:
            out.append((str(name), mat_name, sigma))
    return out


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    ap.add_argument("--mesh", default=DEFAULT_MESH)
    ap.add_argument("--groups", default=None)
    ap.add_argument("--device-config", default=DEFAULT_DEVICE_CONFIG)
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--jj-direction", nargs=3, type=float, default=None,
                    metavar=("DX", "DY", "DZ"),
                    help="Override the derived junction current "
                         "direction (applies to ALL ports)")
    ap.add_argument("--refine-nonlinear", action="store_true",
                    help="Opt in to NLEPS refinement (diverges on "
                         "Palace v0.17.0-67; see session log)")
    ap.add_argument("--linear-type", default=None,
                    help="Override Solver.Linear.Type "
                         "(SuperLU/STRUMPACK/MUMPS/Default)")
    ap.add_argument("--device", choices=("cpu", "gpu"), default="cpu",
                    help="cpu: validated direct solver (SuperLU, full "
                         "assembly). gpu: validated iterative profile "
                         "(AMS + partial assembly + p-multigrid, "
                         "Solver.Device=GPU; cross-checked against the "
                         "direct solver to 6 decimals on 2026-08-12)")
    args = ap.parse_args()

    group_file = args.groups
    if group_file is None:
        for cand in DEFAULT_GROUPS:
            if os.path.isfile(cand):
                group_file = cand
                break
    if group_file is None or not os.path.isfile(group_file):
        raise FileNotFoundError(
            f"no physical-group map found (looked for {DEFAULT_GROUPS})")
    if not os.path.isfile(args.mesh):
        raise FileNotFoundError(f"mesh file not found: {args.mesh}")

    groups, meta = split_groups(load_json(group_file))
    device = load_json(args.device_config)
    if not meta:
        print("NOTE: groups file has no _meta block (older mesher); "
              "deriving group roles from device_config.json")
        meta = derive_meta(groups, device)
    missing_meta = [n for n in groups if n not in meta]
    if missing_meta:
        raise RuntimeError(f"groups without role metadata: {missing_meta}")

    materials_table = device.get("materials", {})
    boundaries_table = device.get("boundaries", {})

    print(f"Mesh:    {args.mesh}")
    print(f"Groups:  {group_file}")
    print(f"Config:  {args.device_config}\n")
    for n, a in sorted(groups.items(), key=lambda kv: kv[1]):
        print(f"  {n:<28} attr {a:<4} role={meta[n].get('role')}")

    # ---- solver settings -------------------------------------------
    solver = dict(SOLVER_DEFAULTS)
    cfg_solver = device.get("palace_solver") or {}
    # accept legacy key spellings
    aliases = {"eigen_tol": "eig_tol", "tol": "eig_tol"}
    for key, value in cfg_solver.items():
        solver[aliases.get(key, key)] = value
    if args.linear_type:
        solver["linear_type"] = args.linear_type
    if args.refine_nonlinear:
        solver["refine_nonlinear"] = True
    if args.device == "gpu":
        # Validated GPU/iterative profile (2026-08-12, v0.17.0-290 CUDA
        # build): AMS preconditioner + matrix-free partial assembly +
        # p-multigrid (p=1 coarse level direct-solved). Reproduced the
        # direct solver's eigenvalues to 6 decimals on the same mesh.
        if not args.linear_type:
            solver["linear_type"] = "Default"      # -> AMS for Maxwell
        solver["ksp_type"] = "GMRES"
        solver["partial_assembly_order"] = 1        # matrix-free at p=2
        solver["mg_max_levels"] = None              # Palace default MG
        # VRAM ruler: ~8M unknowns fit on one 2xA40 node; cap MaxSize at
        # 7M so the last AMR pass's overshoot cannot blow past it (the
        # 8M cap allowed 7.1M -> 9.5M and died in pass-6 postprocessing).
        if int(solver["amr_max_size"]) > 7_000_000:
            print(f"NOTE: --device gpu caps AMR MaxSize "
                  f"{solver['amr_max_size']} -> 7000000 (2xA40 VRAM)")
            solver["amr_max_size"] = 7_000_000
    target = float(solver["target_freq_GHz"])

    # ---- walk the groups -------------------------------------------
    materials_out = []
    pec_attrs = []
    ports = []
    impedance_out = []
    conductivity_out = []
    consumed = {}
    port_index = 1

    def consume(attr, kind):
        prev = consumed.get(attr)
        if prev:
            raise RuntimeError(
                f"attribute {attr} assigned to both {prev} and {kind}")
        consumed[attr] = kind

    for name, attr in sorted(groups.items(), key=lambda kv: kv[1]):
        info = meta[name]
        role = str(info.get("role") or "")

        if info.get("dim") == 3 or role == "dielectric":
            mat_name = str(info.get("material") or "").strip()
            mat = ci_record(materials_table, mat_name)
            if not mat_name:
                raise RuntimeError(f"dielectric group '{name}' has no "
                                   f"material name")
            if not mat:
                raise RuntimeError(
                    f"dielectric '{name}' uses material '{mat_name}' "
                    f"but device_config.json has no record for it")
            eps = parse_numeric(mat.get("permittivity"), 1.0)
            materials_out.append({
                "Attributes": [attr],
                "Permittivity": float(eps),
                "Permeability": float(
                    parse_numeric(mat.get("permeability"), 1.0)),
                "LossTan": float(
                    parse_numeric(mat.get("loss_tangent"), 0.0)),
            })
            consume(attr, f"Material({mat_name})")

        elif role in ("pec_exterior",) or role in PEC_SHEET_ROLES:
            pec_attrs.append(attr)
            consume(attr, "PEC")

        elif role in JUNCTION_ROLES:
            bc = ci_record(boundaries_table, info.get("boundary"))
            l_nh = parse_numeric(bc.get("L_nH"), device.get("L_JJ_nH"))
            if l_nh is None or l_nh <= 0.0:
                raise RuntimeError(
                    f"junction '{name}' has no positive inductance "
                    f"(L_nH={l_nh!r}); refusing an L=0 lumped port")
            c_ff = parse_numeric(bc.get("C_fF"), device.get("C_JJ_fF"))
            r_ohm = parse_numeric(bc.get("R_ohm"), 0.0) or 0.0
            if args.jj_direction is not None:
                direction = [float(v) for v in args.jj_direction]
                source = "--jj-direction override"
            else:
                direction, source = derive_port_direction(bc)
                if direction is None:
                    raise RuntimeError(
                        f"junction '{name}': no face bbox exported to "
                        f"derive a current direction -- pass "
                        f"--jj-direction DX DY DZ")
            print(f"\n  LumpedPort '{name}': direction {direction} "
                  f"({source}) -- VERIFY this is the lead-to-lead axis")
            ports.append({
                "Index": port_index,
                "Elements": [{"Attributes": [attr],
                              "Direction": direction}],
                "R": float(r_ohm),
                "L": float(l_nh) * 1.0e-9,
                "C": float(c_ff or 0.0) * 1.0e-15,
            })
            port_index += 1
            consume(attr, "LumpedPort")

        elif role in IMPEDANCE_ROLES:
            bc = ci_record(boundaries_table, info.get("boundary") or name)
            conductive = conductive_assignments(bc, device)
            # An impedance boundary on the FACE of a bulk conductor is
            # exactly the copper-as-50ohm mistake -- but the boundary's
            # "assignment" may legitimately name the owning dielectric
            # (face-level boundaries are recorded on the owner body).
            conductive = [c for c in conductive
                          if not ci_record(device.get("objects", {}),
                                           c[0]).get("solve_inside",
                                                     True)]
            if conductive:
                details = ", ".join(f"{o} ({m}, sigma={s:g} S/m)"
                                    for o, m, s in conductive)
                raise RuntimeError(
                    f"impedance boundary '{name}' is assigned to a "
                    f"bulk conductor: {details}. Export that metal as a "
                    f"Conductivity:: group instead of Rs ohm/sq.")
            rs = parse_numeric(bc.get("Rs_ohm_per_sq"),
                               parse_numeric(
                                   (bc.get("props") or {}).get(
                                       "Resistance"), 0.0)) or 0.0
            xs = parse_numeric(bc.get("Xs_ohm_per_sq"),
                               parse_numeric(
                                   (bc.get("props") or {}).get(
                                       "Reactance"), 0.0)) or 0.0
            entry = {"Attributes": [attr], "Rs": float(rs)}
            if abs(xs) > 0.0:
                omega = 2.0 * math.pi * target * 1.0e9
                if xs > 0.0:
                    entry["Ls"] = xs / omega
                else:
                    entry["Cs"] = -1.0 / (omega * xs)
                print(f"  WARNING: '{name}' reactance {xs} ohm/sq "
                      f"converted at {target} GHz")
            impedance_out.append(entry)
            consume(attr, "Impedance")

        elif role == "conductor_surface":
            mat_name = str(info.get("material") or "").strip()
            mat = ci_record(materials_table, mat_name)
            sigma = parse_numeric(mat.get("conductivity_S_per_m"))
            if sigma is None or sigma <= 0.0:
                raise RuntimeError(
                    f"conductor group '{name}' material '{mat_name}' "
                    f"has no positive conductivity")
            conductivity_out.append({
                "Attributes": [attr],
                "Conductivity": float(sigma),
                "Permeability": float(
                    parse_numeric(mat.get("permeability"), 1.0)),
            })
            consume(attr, "Conductivity")

        elif role in CONDUCTOR_SHEET_ROLES:
            bc = ci_record(boundaries_table, info.get("boundary") or name)
            sigma = parse_numeric(bc.get("conductivity_S_per_m"))
            if sigma is None or sigma <= 0.0:
                raise RuntimeError(
                    f"conductivity sheet '{name}' has no positive "
                    f"conductivity in its boundary record")
            conductivity_out.append({
                "Attributes": [attr],
                "Conductivity": float(sigma),
                "Permeability": float(
                    parse_numeric(bc.get("permeability"), 1.0)),
            })
            consume(attr, "Conductivity")

        else:
            raise RuntimeError(
                f"group '{name}' has unhandled role {role!r} -- "
                f"refusing to silently drop physics")

    # ---- gates ------------------------------------------------------
    if not materials_out:
        raise RuntimeError("no dielectric material domains resolved")
    if not pec_attrs:
        raise RuntimeError("no PEC attributes resolved")
    unconsumed = {n: a for n, a in groups.items() if a not in consumed}
    if unconsumed:
        raise RuntimeError(
            f"mesh groups with NO assigned physics: {sorted(unconsumed)}"
            f" -- every group must be consumed; refusing to drop physics")
    if device.get("has_junction") and not ports:
        raise RuntimeError(
            "device_config.json says has_junction=true but no junction "
            "group exists in the mesh -- the junction was lost upstream")
    if not ports:
        print("\n  No junction in this design: no LumpedPort emitted.")

    if conductivity_out and not solver["refine_nonlinear"]:
        print(f"\nNOTE: conductivity boundaries present and "
              f"RefineNonlinear=False (NLEPS diverges on v0.17.0-67). "
              f"sqrt(omega) surface impedance is frozen at the "
              f"{target} GHz target: conductor loss is understated by "
              f"sqrt(f/target) for modes above it.")

    boundaries_out = {"PEC": {"Attributes": unique_ints(pec_attrs)}}
    if ports:
        boundaries_out["LumpedPort"] = ports
    if impedance_out:
        boundaries_out["Impedance"] = impedance_out
    if conductivity_out:
        boundaries_out["Conductivity"] = conductivity_out

    config = {
        "Problem": {
            "Type": "Eigenmode",
            "Verbose": 2,
            "Output": "postpro",
            "OutputFormats": {"GridFunction": False, "Paraview": True},
        },
        "Model": {
            "Mesh": args.mesh,
            "L0": 1.0e-3,
            "CrackInternalBoundaryElements": True,
            "AddInterfaceBoundaryElements": True,
            "RefineCrackElements": True,
            "Refinement": {
                "MaxIts": int(solver["amr_max_its"]),
                "Tol": float(solver["amr_tol"]),
                "UpdateFraction": float(solver["amr_update_fraction"]),
                "MaxSize": int(solver["amr_max_size"]),
                "Nonconformal": False,
                "SaveAdaptIterations": False,  # True re-writes full fields every AMR pass
                # (slow IO) and caused palace.json aborts on restart-over-stale-postpro
            },
        },
        "Domains": {"Materials": materials_out},
        "Boundaries": boundaries_out,
        "Solver": {
            "Order": int(solver["solver_order"]),
            "Device": "GPU" if args.device == "gpu" else "CPU",
            "PartialAssemblyOrder": int(
                solver["partial_assembly_order"]),
            "Eigenmode": {
                "N": int(solver["n_modes"]),
                "Target": target,
                "Tol": float(solver["eig_tol"]),
                "MaxIts": int(solver["eig_max_its"]),
                "RefineNonlinear": bool(solver["refine_nonlinear"]),
            },
            "Linear": {
                "Type": str(solver["linear_type"]),
                "KSPType": str(solver["ksp_type"]),
                **({"MGMaxLevels": int(solver["mg_max_levels"])}
                   if solver["mg_max_levels"] is not None else {}),
                "Tol": float(solver["linear_tol"]),
                "MaxIts": int(solver["linear_max_its"]),
            },
        },
    }

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)

    print("\nResolved Palace assignments:")
    for m in materials_out:
        print(f"  Material     {m['Attributes']}  eps={m['Permittivity']:g}"
              f"  tand={m['LossTan']:g}")
    print(f"  PEC          {boundaries_out['PEC']['Attributes']}")
    for p in ports:
        print(f"  LumpedPort   {p['Elements'][0]['Attributes']}  "
              f"L={p['L']:g} H  C={p['C']:g} F  R={p['R']:g} ohm  "
              f"dir={p['Elements'][0]['Direction']}")
    for e in impedance_out:
        print(f"  Impedance    {e['Attributes']}  Rs={e['Rs']:g} ohm/sq")
    for e in conductivity_out:
        print(f"  Conductivity {e['Attributes']}  "
              f"sigma={e['Conductivity']:g} S/m")
    print(f"\nSolver: order {solver['solver_order']}, target {target} "
          f"GHz, {solver['n_modes']} modes, "
          f"Linear.Type={solver['linear_type']}, "
          f"MGMaxLevels={solver['mg_max_levels']}, "
          f"PartialAssemblyOrder={solver['partial_assembly_order']}, "
          f"RefineNonlinear={solver['refine_nonlinear']}")
    print(f"\nWritten: {args.output}")


if __name__ == "__main__":
    main()
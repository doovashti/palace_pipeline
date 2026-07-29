#!/usr/bin/env python3
"""
write_palace_config_single_step_from_device_config.py

Generate a Palace eigenmode configuration using:
  1. the physical-group map written by the Gmsh mesher; and
  2. device_config.json written by the HFSS exporter.

The mesh stores geometry and integer physical attributes. This script maps
the HFSS boundary/material semantics onto those attributes.

Expected mesh groups for the current device:
  3D: cavity, chip
  2D: PEC_exterior, pads, medium_lead, thin_lead, JJ, RR, Imped_pin,\n      Conductivity::<object>

Usage:
    python write_palace_config_single_step_from_device_config.py

Optional:
    python write_palace_config_single_step_from_device_config.py \
        --mesh transmon_simple_RR.msh \
        --groups transmon_simple_RR.json \
        --device-config device_config.json \
        --output palace_config.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any


DEFAULT_MESH_FILE = "transmon_simple_RR.msh"
DEFAULT_GROUP_FILES = ("transmon_simple_RR.json", "physical_groups.json")
DEFAULT_DEVICE_CONFIG = "device_config.json"
DEFAULT_OUTPUT = "palace_config.json"

# Direction is geometric information not currently exported by HFSS.
# Keep the known working value from the original writer.
DEFAULT_JJ_DIRECTION = [0.0, 1.0, 0.0]

# Preserve the original linear-solver iteration setting.
DEFAULT_MAX_ITER = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Palace eigenmode JSON from mesh groups and device_config.json."
    )
    parser.add_argument("--mesh", default=DEFAULT_MESH_FILE)
    parser.add_argument("--groups", default=None)
    parser.add_argument("--device-config", default=DEFAULT_DEVICE_CONFIG)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--jj-direction",
        nargs=3,
        type=float,
        default=DEFAULT_JJ_DIRECTION,
        metavar=("DX", "DY", "DZ"),
    )
    return parser.parse_args()


def load_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return data


def choose_group_file(explicit: str | None) -> str:
    if explicit:
        if not os.path.isfile(explicit):
            raise FileNotFoundError(f"Physical-group file not found: {explicit}")
        return explicit

    for candidate in DEFAULT_GROUP_FILES:
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        "No physical-group map was found. Expected one of: "
        + ", ".join(DEFAULT_GROUP_FILES)
    )


def normalize_group_map(data: dict[str, Any]) -> dict[str, int]:
    raw = data.get("physical_groups", data)
    if not isinstance(raw, dict):
        raise RuntimeError("Physical-group JSON must contain a name-to-attribute object")

    groups: dict[str, int] = {}
    for name, value in raw.items():
        # Ignore metadata values that are not integer attributes.
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            groups[str(name)] = value
        elif isinstance(value, float) and value.is_integer():
            groups[str(name)] = int(value)

    if not groups:
        raise RuntimeError("No integer physical-group attributes were found")
    return groups


def unique_ints(values: list[int]) -> list[int]:
    return sorted(set(int(value) for value in values))


def parse_numeric(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)

    match = re.search(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
        str(value),
    )
    if not match:
        return default
    return float(match.group(0))


def material_record(
    attributes: list[int],
    material_info: dict[str, Any],
    *,
    default_eps: float,
    default_mu: float = 1.0,
    default_loss: float = 0.0,
) -> dict[str, Any]:
    return {
        "Attributes": unique_ints(attributes),
        "Permittivity": float(
            material_info.get("permittivity")
            if material_info.get("permittivity") is not None
            else default_eps
        ),
        "Permeability": float(
            material_info.get("permeability")
            if material_info.get("permeability") is not None
            else default_mu
        ),
        "LossTan": float(
            material_info.get("loss_tangent")
            if material_info.get("loss_tangent") is not None
            else default_loss
        ),
    }


def boundary_mesh_groups(
    boundary_name: str,
    boundary_info: dict[str, Any],
    groups: dict[str, int],
) -> list[str]:
    """
    Resolve an HFSS boundary to mesh physical-group names.

    Standalone sheet boundaries such as Perfect E and Lumped RLC are
    represented by their assigned STEP body names.

    Face-specific impedance and finite-conductivity boundaries are represented
    by dedicated 2-D physical groups. For an impedance face, use only the
    boundary-name group, such as "Imped_pin". Do not also add the owning 3-D
    body, such as "cavity".
    """
    role = str(boundary_info.get("role") or "")
    boundary_type = str(boundary_info.get("type") or "")
    assignments = [str(name) for name in boundary_info.get("assignment", [])]

    if (
        role in ("impedance_sheet", "conductivity_sheet")
        or boundary_type in (
            "Impedance",
            "Layered Impedance",
            "Finite Conductivity",
        )
    ):
        return [boundary_name] if boundary_name in groups else []

    return [name for name in assignments if name in groups]


def case_insensitive_record(
    records: dict[str, Any],
    wanted_name: str,
) -> dict[str, Any]:
    """Return a dictionary record using a case-insensitive name lookup."""
    wanted = str(wanted_name or "").strip().lower()

    for known_name, raw_info in records.items():
        if str(known_name).strip().lower() == wanted:
            return raw_info if isinstance(raw_info, dict) else {}

    return {}


def conductivity_entries_from_groups(
    groups: dict[str, int],
    device: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Convert mesher groups named "Conductivity::<object>" into Palace
    finite-conductivity boundary entries.

    The object record supplies the material name, and the material record
    supplies conductivity in S/m and relative permeability.
    """
    entries: list[dict[str, Any]] = []
    objects = device.get("objects", {})
    materials = device.get("materials", {})

    prefix = "Conductivity::"

    for group_name, attribute in sorted(
        groups.items(),
        key=lambda item: item[1],
    ):
        if not group_name.startswith(prefix):
            continue

        body_name = group_name[len(prefix):].strip()
        if not body_name:
            raise RuntimeError(
                f"Invalid conductivity physical-group name: {group_name!r}"
            )

        object_info = case_insensitive_record(objects, body_name)
        if not object_info:
            raise RuntimeError(
                f"Conductivity mesh group '{group_name}' refers to object "
                f"'{body_name}', but no matching object record exists in "
                "device_config.json"
            )

        material_name = str(object_info.get("material") or "").strip()
        if not material_name:
            raise RuntimeError(
                f"Object '{body_name}' has no material in device_config.json"
            )

        material_info = case_insensitive_record(
            materials,
            material_name,
        )
        if not material_info:
            raise RuntimeError(
                f"Object '{body_name}' uses material '{material_name}', "
                "but that material has no record in device_config.json"
            )

        conductivity = parse_numeric(
            material_info.get("conductivity_S_per_m")
        )
        if conductivity is None or conductivity <= 0.0:
            raise RuntimeError(
                f"Material '{material_name}' for object '{body_name}' has "
                f"invalid conductivity: {conductivity!r}"
            )

        permeability = parse_numeric(
            material_info.get("permeability"),
            1.0,
        )
        if permeability is None or permeability <= 0.0:
            raise RuntimeError(
                f"Material '{material_name}' for object '{body_name}' has "
                f"invalid permeability: {permeability!r}"
            )

        strategy = str(
            object_info.get("modeling_strategy") or ""
        ).strip()
        role = str(
            object_info.get("domain_role") or ""
        ).strip()

        if (
            strategy
            and strategy != "surface_conductor"
            and role != "conductor_solid"
        ):
            raise RuntimeError(
                f"Object '{body_name}' has conductivity group '{group_name}' "
                f"but modeling_strategy={strategy!r} and domain_role={role!r}"
            )

        entries.append(
            {
                "Attributes": [int(attribute)],
                "Conductivity": float(conductivity),
                "Permeability": float(permeability),
            }
        )

    return entries


def impedance_entry(
    boundary_name: str,
    info: dict[str, Any],
    attributes: list[int],
    target_freq_ghz: float,
) -> dict[str, Any]:
    props = info.get("props", {})
    rs = parse_numeric(info.get("Rs_ohm_per_sq"))
    if rs is None:
        rs = parse_numeric(props.get("Resistance"), 0.0)
    rs = float(rs or 0.0)

    reactance = float(parse_numeric(props.get("Reactance"), 0.0) or 0.0)
    ls = 0.0
    cs = 0.0

    # Palace represents reactive surface impedance with Ls or Cs. HFSS exports
    # a constant reactance value. A nonzero value can only be matched exactly
    # at one reference frequency, so use the target eigenfrequency and report it.
    if abs(reactance) > 0.0:
        omega = 2.0 * math.pi * target_freq_ghz * 1.0e9
        if reactance > 0.0:
            ls = reactance / omega
        else:
            cs = -1.0 / (omega * reactance)
        print(
            f"WARNING: '{boundary_name}' has HFSS Reactance={reactance} ohm/sq. "
            f"It is converted at the target frequency {target_freq_ghz} GHz."
        )

    return {
        "Attributes": unique_ints(attributes),
        "Rs": rs,
        "Ls": ls,
        "Cs": cs,
    }


def main() -> None:
    args = parse_args()

    group_file = choose_group_file(args.groups)
    group_data = load_json(group_file)
    groups = normalize_group_map(group_data)

    device = load_json(args.device_config)

    print(f"Mesh:                 {args.mesh}")
    print(f"Physical-group map:   {group_file}")
    print(f"Device configuration: {args.device_config}")
    print("\nPhysical groups loaded:")
    for name, attribute in sorted(groups.items(), key=lambda item: item[1]):
        print(f"  {name:<18} -> attribute {attribute}")

    if not os.path.isfile(args.mesh):
        raise FileNotFoundError(f"Mesh file not found: {args.mesh}")

    materials_info = device.get("materials", {})
    solver_info = device.get("palace_solver", {})

    target_freq = float(solver_info.get("target_freq_GHz", 1.0))
    num_modes = int(solver_info.get("n_modes", 3))
    solver_order = int(solver_info.get("solver_order", 1))
    tol = float(solver_info.get("tol", 1.0e-3))
    amr_max_its = int(solver_info.get("amr_max_its", 1))

    cavity_attrs = [groups["cavity"]] if "cavity" in groups else []
    chip_attrs = [groups["chip"]] if "chip" in groups else []

    if not cavity_attrs:
        raise RuntimeError("Mesh group 'cavity' was not found")

    materials = [
        material_record(
            cavity_attrs,
            materials_info.get("vacuum", {}),
            default_eps=1.0,
        )
    ]

    if chip_attrs:
        sapphire = dict(materials_info.get("sapphire", {}))
        if sapphire.get("permittivity") is None:
            sapphire["permittivity"] = device.get("sapphire_permittivity", 10.0)
        materials.append(
            material_record(
                chip_attrs,
                sapphire,
                default_eps=float(device.get("sapphire_permittivity", 10.0)),
            )
        )

    pec_attrs: list[int] = []
    if "PEC_exterior" in groups:
        pec_attrs.append(groups["PEC_exterior"])

    jj_ports: list[dict[str, Any]] = []
    impedance_entries: list[dict[str, Any]] = []
    conductivity_entries = conductivity_entries_from_groups(
        groups,
        device,
    )
    consumed_boundary_attributes: dict[int, str] = {}

    next_port_index = 1
    for boundary_name, raw_info in device.get("boundaries", {}).items():
        if not isinstance(raw_info, dict):
            continue

        info = raw_info
        role = str(info.get("role") or "")
        boundary_type = str(info.get("type") or "")
        mesh_names = boundary_mesh_groups(boundary_name, info, groups)
        attrs = unique_ints([groups[name] for name in mesh_names])

        if role == "pec_sheet" or boundary_type in ("Perfect E", "PerfectE"):
            if not attrs:
                raise RuntimeError(
                    f"PEC boundary '{boundary_name}' matched no mesh groups. "
                    f"HFSS assignments were {info.get('assignment', [])}"
                )
            pec_attrs.extend(attrs)

        elif role == "junction" or boundary_type in (
            "Lumped RLC",
            "LumpedRLC",
            "Inductance",
        ):
            if not attrs:
                raise RuntimeError(
                    f"Junction boundary '{boundary_name}' matched no mesh group"
                )

            l_nh = parse_numeric(info.get("L_nH"), device.get("L_JJ_nH", 0.0))
            c_ff = parse_numeric(info.get("C_fF"), device.get("C_JJ_fF", 0.0))
            r_ohm = 0.0
            props = info.get("props", {})
            use_resist = str(props.get("Use Resist", "False")).lower() == "true"
            if use_resist:
                r_ohm = float(parse_numeric(props.get("Resistance"), 0.0) or 0.0)

            jj_ports.append(
                {
                    "Index": next_port_index,
                    "Elements": [
                        {
                            "Attributes": attrs,
                            "Direction": [float(v) for v in args.jj_direction],
                        }
                    ],
                    "R": r_ohm,
                    "L": float(l_nh or 0.0) * 1.0e-9,
                    "C": float(c_ff or 0.0) * 1.0e-15,
                }
            )
            next_port_index += 1

        elif role == "impedance_sheet" or boundary_type in (
            "Impedance",
            "Layered Impedance",
        ):
            if not attrs:
                raise RuntimeError(
                    f"Impedance boundary '{boundary_name}' matched no mesh group. "
                    f"Expected a physical group named '{boundary_name}'."
                )
            impedance_entries.append(
                impedance_entry(boundary_name, info, attrs, target_freq)
            )

    pec_attrs = unique_ints(pec_attrs)

    if not pec_attrs:
        raise RuntimeError("No PEC physical-group attributes were resolved")
    if not jj_ports:
        raise RuntimeError("No lumped junction boundary was resolved")

    # Validate that the same mesh attribute is not assigned incompatible physics.
    for attribute in pec_attrs:
        consumed_boundary_attributes[attribute] = "PEC"

    for port in jj_ports:
        for element in port["Elements"]:
            for attribute in element["Attributes"]:
                previous = consumed_boundary_attributes.get(attribute)
                if previous:
                    raise RuntimeError(
                        f"Attribute {attribute} is assigned to both {previous} "
                        "and LumpedPort"
                    )
                consumed_boundary_attributes[attribute] = "LumpedPort"

    for entry in impedance_entries:
        for attribute in entry["Attributes"]:
            previous = consumed_boundary_attributes.get(attribute)
            if previous:
                raise RuntimeError(
                    f"Attribute {attribute} is assigned to both {previous} "
                    "and Impedance"
                )
            consumed_boundary_attributes[attribute] = "Impedance"

    for entry in conductivity_entries:
        for attribute in entry["Attributes"]:
            previous = consumed_boundary_attributes.get(attribute)
            if previous:
                raise RuntimeError(
                    f"Attribute {attribute} is assigned to both {previous} "
                    "and Conductivity"
                )
            consumed_boundary_attributes[attribute] = "Conductivity"

    boundaries: dict[str, Any] = {
        "PEC": {"Attributes": pec_attrs},
        "LumpedPort": jj_ports,
    }
    if impedance_entries:
        boundaries["Impedance"] = impedance_entries
    if conductivity_entries:
        boundaries["Conductivity"] = conductivity_entries

    config = {
        "Problem": {
            "Type": "Eigenmode",
            "Verbose": 2,
        },
        "Model": {
            "Mesh": args.mesh,
            "L0": 1.0e-3,
            "Refinement": {
                "MaxIts": amr_max_its,
                "Tol": tol,
                "Nonconformal": False,
            },
        },
        "Domains": {
            "Materials": materials,
        },
        "Boundaries": boundaries,
        "Solver": {
            "Order": solver_order,
            "Eigenmode": {
                "N": num_modes,
                "Target": target_freq,
                "Tol": tol,
                "MaxIts": DEFAULT_MAX_ITER,
                "PEPLinear": True,
            },
            "Linear": {
                "Type": "Default",
                "KSPType": "Default",
                "Tol": tol,
                "MaxIts": DEFAULT_MAX_ITER,
            },
        },
    }

    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    print("\nResolved Palace assignments:")
    print(f"  Vacuum:      {cavity_attrs}")
    print(f"  Sapphire:    {chip_attrs}")
    print(f"  PEC:         {pec_attrs}")
    for port in jj_ports:
        attrs = port["Elements"][0]["Attributes"]
        print(
            f"  LumpedPort:  {attrs}, "
            f"L={port['L']:.12g} H, C={port['C']:.12g} F"
        )
    for entry in impedance_entries:
        print(
            f"  Impedance:   {entry['Attributes']}, "
            f"Rs={entry['Rs']:.12g} ohm/sq, "
            f"Ls={entry['Ls']:.12g} H/sq, "
            f"Cs={entry['Cs']:.12g} F/sq"
        )
    for entry in conductivity_entries:
        print(
            f"  Conductivity:{entry['Attributes']}, "
            f"sigma={entry['Conductivity']:.12g} S/m, "
            f"mu_r={entry['Permeability']:.12g}"
        )

    print(f"\nWritten: {args.output}")


if __name__ == "__main__":
    main()

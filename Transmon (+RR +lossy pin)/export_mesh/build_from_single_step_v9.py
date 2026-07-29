#!/usr/bin/env python3
"""
build_from_single_step.py

Builds a conformal Gmsh mesh for Palace from one STEP file containing:
  - 3D volumes: cavity, chip, pin, optional Bondwire1
  - 2D sheets: pads, medium leads, thin leads, JJ

Important behavior:
  - Imports the STEP file only once.
  - Preserves the original imported sheet geometry.
  - Does not replace sheets with bounding-box rectangles.
  - Detects original sheet faces by their imported geometry and dimensions.
  - Cuts PEC solids out of dielectric volumes.
  - Fragments dielectric volumes with the original sheet entities.
  - Tracks sheet descendants through occ.fragment(...).
  - Verifies that every pad/lead/JJ surface is a true two-sided internal boundary.
  - Keeps pads, medium leads, thin leads, and JJ as separate physical groups.

Usage:
    python3 build_from_single_step.py

Outputs:
    real_device.msh
    real_device.vtk
    physical_groups.json

Units:
    millimetres
"""

from __future__ import annotations

import json
import itertools
import os
import re
import sys

import gmsh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step_bodies import read_step_bodies, match_gmsh_entities

# STEP file exported by export_for_palace.py.
# Change this one line to mesh a different device.
STEP_FILE = r"\\Qcrew_drive\home\vashti_enjoy\test_export_palace_export\device.step"

# Non-geometric data exported from HFSS. This is used only to recover
# exact boundary assignments such as Impedance faces. Existing geometry,
# boolean, physical-group, and mesh behavior remains unchanged.
DEVICE_CONFIG_FILE = os.path.join(
    os.path.dirname(STEP_FILE), "device_config.json"
)

# Matching tolerances for boundary faces recorded by
# export_for_palace_with_face_geometry.py.
IMPEDANCE_CENTER_TOL_MM = 0.05
IMPEDANCE_AREA_REL_TOL = 0.02
IMPEDANCE_AREA_ABS_TOL_MM2 = 1.0e-8

# Matching tolerances for finite-conductivity solid surfaces.
#
# The original conductor volume is removed from the solved domains, but
# its face geometry is recorded before the Boolean cut. These tolerances
# are then used to recover the corresponding final dielectric-boundary
# surfaces after all Boolean and fragmentation operations.
CONDUCTOR_CENTER_TOL_MM = 0.02
CONDUCTOR_BBOX_TOL_MM = 0.02
CONDUCTOR_AREA_REL_TOL = 0.02
CONDUCTOR_AREA_ABS_TOL_MM2 = 1.0e-8

OUT_MSH = "transmon_simple_RR.msh"
OUT_VTK = "transmon_simple_RR.vtk"
OUT_GROUPS = "transmon_simple_RR.json"

ENABLE_CHIP = True
ENABLE_PIN = True
ENABLE_BONDWIRE = False
ENABLE_PADS = True
ENABLE_MEDIUM_LEAD = True
ENABLE_THIN_LEAD = True
ENABLE_JJ = True
ENABLE_RR = True

# ---------------------------------------------------------------------
# Body identification is now by AEDT name read from the STEP file, not
# by hardcoded bounding-box dimensions. step_bodies.py reads the names;
# match_gmsh_entities() maps them onto Gmsh entity tags by bounding box.
#
# For a new device, the only thing that may need editing is BODY_ALIASES
# below -- if its bodies use the same names as this one, nothing at all.
# ---------------------------------------------------------------------

# Maps the name in the STEP file to the name this script uses.
# Ansys often calls the readout resonator "Rectangle1"; we call it "RR".
BODY_ALIASES = {
    "Rectangle1": "RR",
}

# Bodies present in the STEP but not meshed.
EXCLUDE_BODIES = {"Plot_Fields2D", "Plot_Fields3D"}

# Tolerance for matching a Gmsh entity bbox onto a STEP body bbox, in mm.
MATCH_TOLERANCE = 0.02

# Extra perfect-conductor sheet. In the STEP this may still be named
# Rectangle1; in this mesher/Palace output it will be named RR.
RR_SHEET_NAMES = {"Rectangle1", "RR"}

# Keep these as named physical groups even if they are one-sided
# conductor boundary sheets. Palace can then assign them as PEC by name.
KEEP_ONE_SIDED_SHEET_GROUPS = {"RR"}

# Target element size at each feature, in millimetres. These match the
# Ansys mesh operations reported by export_for_palace.py:
#   JJ_mesh_0.2um -> 0.0002   thin_lead_0.2um -> 0.0002
#   medium_3um    -> 0.003    pads_50um       -> 0.05
#   cav_pin1mm    -> 1.0
SIZE_CAVITY = 1.0
SIZE_PADS = 0.05
SIZE_MEDIUM = 0.003
SIZE_THIN = 0.0002
SIZE_JJ = 0.0002

# The RR sheet is rr_x = 0.3 mm wide (Ansys design variable).
#
# This was previously 1.0 mm -- wider than the sheet itself. No element
# could fit across the RR's width, so the 2-D mesher was forced to
# subdivide against the requested size instead of following it. It also
# put a 1.0 mm target immediately next to the 0.05 mm pads target, which
# Gmsh must reconcile with a steep gradient.
#
# 0.05 mm gives roughly six elements across the 0.3 mm width, matching
# how the pads are treated.
#
# Note: Ansys has no mesh operation for the RR (export_for_palace.py
# found only five, none covering it), so this value is our choice rather
# than something read from the design.
SIZE_RR = 0.05


def normalize_label(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def entity_name(dim: int, tag: int) -> str:
    # Works when Gmsh imports STEP labels. Empty string is fine; the
    # bounding-box classifiers below still handle the original sheets.
    try:
        return gmsh.model.getEntityName(dim, tag) or ""
    except Exception:
        return ""


def is_rr_sheet_by_name(tag: int) -> bool:
    imported = normalize_label(entity_name(2, tag))
    if not imported:
        return False
    return any(normalize_label(name) in imported for name in RR_SHEET_NAMES)


def bbox(dim: int, tag: int):
    return gmsh.model.getBoundingBox(dim, tag)


def lengths(box):
    return box[3] - box[0], box[4] - box[1], box[5] - box[2]


def center(box):
    return (
        0.5 * (box[0] + box[3]),
        0.5 * (box[1] + box[4]),
        0.5 * (box[2] + box[5]),
    )


def approx(a: float, b: float, tol: float):
    return abs(a - b) <= tol


# Filled in by _identify_bodies() once the STEP file has been matched
# onto the imported Gmsh entities.
_VOLUME_NAME_OF_TAG = {}
_SHEET_NAME_OF_TAG = {}


def _identify_bodies(step_file):
    """
    Read body names from the STEP file and match them onto Gmsh tags.

    Must be called after occ.importShapes() and occ.synchronize(), and
    before any boolean operation renumbers the entities.
    """
    bodies = read_step_bodies(step_file)
    print(f"\nNamed bodies in {step_file}:")
    for body_name, info in sorted(bodies.items()):
        dx, dy, dz = info["size"]
        print(f"  {body_name:<16}{info['geo_type']:<7}"
              f"{info['material']:<20}{dx:.4g} x {dy:.4g} x {dz:.4g}")

    for dim, store in ((3, _VOLUME_NAME_OF_TAG), (2, _SHEET_NAME_OF_TAG)):
        tag_to_name, unmatched = match_gmsh_entities(
            gmsh, bodies, dim, tolerance=MATCH_TOLERANCE
        )
        for tag, raw_name in tag_to_name.items():
            store[tag] = BODY_ALIASES.get(raw_name, raw_name)

        label = "volumes" if dim == 3 else "surfaces"
        print(f"\nMatched {len(store)} Gmsh {label} to STEP names:")
        for tag in sorted(store):
            print(f"  tag {tag:4d} -> {store[tag]}")

        # Surfaces of solids are expected not to match a sheet body, so
        # only report unmatched entities that look like real sheets.
        if unmatched and dim == 3:
            print(f"  {len(unmatched)} unmatched volume(s):")
            for tag, box, best, score in unmatched:
                print(f"    tag {tag:4d} bbox={box} "
                      f"(closest '{best}', off by {score:.4g} mm)")

    return bodies


def classify_volume(tag: int) -> str:
    """Body name for a Gmsh volume, from the STEP file's AEDT names."""
    name = _VOLUME_NAME_OF_TAG.get(tag)
    if name is None:
        raise RuntimeError(
            f"Gmsh volume {tag} (bbox={bbox(3, tag)}) matched no named "
            f"body in the STEP file within {MATCH_TOLERANCE} mm. Either "
            f"the body is unnamed in Ansys, or MATCH_TOLERANCE is too "
            f"tight."
        )
    return name


def classify_sheet(tag: int) -> str:
    """Body name for a Gmsh surface, from the STEP file's AEDT names."""
    name = _SHEET_NAME_OF_TAG.get(tag)
    if name is None:
        raise RuntimeError(f"Gmsh surface {tag} matched no named body")
    return name


def _superseded_classify_volume(tag: int) -> str:
    box = bbox(3, tag)
    dx, dy, dz = lengths(box)

    if dx > 7.5 and dy > 25.0 and dz > 34.0:
        return "cavity"
    if approx(dx, 3.0, 0.05) and approx(dy, 3.0, 0.05) and dz > 16.5:
        return "pin"
    if approx(dx, 4.0, 0.05) and approx(dy, 20.0, 0.05) and approx(dz, 0.43, 0.02):
        return "chip"
    if dx < 0.02 and 4.0 < dy < 4.3 and 0.15 < dz < 0.25:
        return "Bondwire1"

    raise RuntimeError(
        f"Could not classify volume tag {tag}, bbox={box}, size=({dx}, {dy}, {dz})"
    )


def _superseded_classify_sheet(tag: int) -> str:
    """
    Superseded by name lookup. Kept only for reference.

    Classify a planar sheet from its bounding-box side lengths.

    This is orientation-independent: the two largest bbox dimensions are
    treated as the in-plane dimensions, so x/y-swapped or differently
    oriented imported faces are still recognized.
    """
    box = bbox(2, tag)
    dims = sorted(lengths(box))
    thickness, short_side, long_side = dims

    # Reject clearly non-planar faces.
    if thickness > 1.0e-4:
        raise RuntimeError(
            f"Surface {tag} is not sufficiently planar, bbox={box}, dims={dims}"
        )

    # Extra perfect-conductor rectangle/readout-resonator sheet.
    # Prefer STEP/AEDT name matching so it does not have to share the
    # pad/lead/JJ dimensions.
    if is_rr_sheet_by_name(tag):
        return "RR"

    # Fallback for exports where Gmsh collapses the STEP/AEDT label to
    # "Shapes/COMPOUND". In transmon_RR_nnm.step, the RR sheet imports as
    # a planar 0.3 mm x 8.0 mm rectangle.
    if (
        0.25 <= short_side <= 0.35
        and 7.5 <= long_side <= 8.5
    ):
        return "RR"

    # JJ: about 1.0 um x 1.9 um
    if (
        0.0007 <= short_side <= 0.0013
        and 0.0015 <= long_side <= 0.0023
    ):
        return "JJ"

    # Thin lead: about 1 um x 25 um
    if (
        0.0007 <= short_side <= 0.0013
        and 0.009 <= long_side <= 0.030
    ):
        return "thin_lead"

    # Medium lead: about 10 um x 30 um
    if (
        0.007 <= short_side <= 0.013
        and 0.024 <= long_side <= 0.036
    ):
        return "medium_lead"

    # Pad: about 0.5 mm x 1.0 mm
    if (
        0.45 <= short_side <= 0.60
        and 0.90 <= long_side <= 1.10
    ):
        return "pads"

    raise RuntimeError(
        f"Could not classify sheet tag {tag}, bbox={box}, dims={dims}"
    )


def boundary_points(surface_tags):
    points = set()
    for surface in surface_tags:
        curves = gmsh.model.getBoundary([(2, surface)], oriented=False, recursive=False)
        for dim, curve in curves:
            if dim != 1:
                continue
            endpoints = gmsh.model.getBoundary([(1, curve)], oriented=False, recursive=False)
            points.update(point for dim2, point in endpoints if dim2 == 0)
    return sorted(points)


def set_surface_point_size(surface_tags, size):
    points = boundary_points(surface_tags)
    if points:
        gmsh.model.mesh.setSize([(0, point) for point in points], size)
    return len(points)


def count_2d_elements(surface_tags):
    count = 0
    for surface in surface_tags:
        _, element_tags, _ = gmsh.model.mesh.getElements(2, surface)
        count += sum(len(tags) for tags in element_tags)
    return count


def load_device_config():
    """Load the optional HFSS export metadata without changing legacy behavior."""
    if not os.path.isfile(DEVICE_CONFIG_FILE):
        print(
            f"\nNo device config found at {DEVICE_CONFIG_FILE}; "
            "continuing without config-driven impedance extraction."
        )
        return {}

    with open(DEVICE_CONFIG_FILE, "r", encoding="utf-8") as file:
        config = json.load(file)

    print(f"\nLoaded device config: {DEVICE_CONFIG_FILE}")
    return config


def surface_area(tag: int) -> float:
    """Return OCC surface area in square millimetres."""
    return float(gmsh.model.occ.getMass(2, tag))


def distance3(a, b) -> float:
    return sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)) ** 0.5

def bbox_center(box):
    """Center of a six-component axis-aligned bounding box."""
    return (
        0.5 * (float(box[0]) + float(box[3])),
        0.5 * (float(box[1]) + float(box[4])),
        0.5 * (float(box[2]) + float(box[5])),
    )


def bbox_max_error(first, second):
    """Maximum absolute coordinate difference between two bounding boxes."""
    return max(
        abs(float(first[index]) - float(second[index]))
        for index in range(6)
    )


def bbox_is_inside(inner, outer, tolerance):
    """
    Return True when one bounding box lies inside another, allowing a
    numerical tolerance.
    """
    return (
        float(inner[0]) >= float(outer[0]) - tolerance
        and float(inner[1]) >= float(outer[1]) - tolerance
        and float(inner[2]) >= float(outer[2]) - tolerance
        and float(inner[3]) <= float(outer[3]) + tolerance
        and float(inner[4]) <= float(outer[4]) + tolerance
        and float(inner[5]) <= float(outer[5]) + tolerance
    )


def record_surface_conductor_geometry(
    config,
    volume_by_name,
):
    """
    Record geometry for every solve-inside-false conductor before it is
    removed by the dielectric Boolean cut.

    Returns:

        {
            "Pin_1": {
                "material": "copper",
                "volume_bbox": (...),
                "faces": [
                    {
                        "surface": original_surface_tag,
                        "bbox": (...),
                        "center": (...),
                        "area": ...,
                    }
                ],
            }
        }

    The original tags will not survive the Boolean operations. The
    geometric signatures are used later to recover the final surfaces.
    """
    records = {}

    for body_name in surface_conductor_object_names(config):
        if body_name not in volume_by_name:
            raise RuntimeError(
                f"Surface-conductor object '{body_name}' was exported in "
                "device_config.json but no matching STEP volume was found"
            )

        volume_tag = volume_by_name[body_name]
        object_info = config_object_info(config, body_name)
        material_name = str(
            object_info.get("material") or ""
        ).strip()

        material_info = config_material_info(
            config,
            material_name,
        )
        conductivity = material_info.get(
            "conductivity_S_per_m"
        )

        if conductivity is None or float(conductivity) <= 0.0:
            raise RuntimeError(
                f"Surface conductor '{body_name}' uses material "
                f"'{material_name}', but no positive conductivity was "
                "exported"
            )

        face_records = []

        for dim, surface in gmsh.model.getBoundary(
            [(3, volume_tag)],
            oriented=False,
            recursive=False,
        ):
            if dim != 2:
                continue

            surface_box = bbox(2, surface)
            face_records.append(
                {
                    "surface": int(surface),
                    "bbox": tuple(float(value) for value in surface_box),
                    "center": bbox_center(surface_box),
                    "area": surface_area(surface),
                }
            )

        if not face_records:
            raise RuntimeError(
                f"Surface conductor '{body_name}' has no readable "
                "boundary faces"
            )

        records[body_name] = {
            "material": material_name,
            "conductivity_S_per_m": float(conductivity),
            "permeability": material_info.get("permeability"),
            "volume_bbox": tuple(
                float(value)
                for value in bbox(3, volume_tag)
            ),
            "faces": face_records,
        }

        print(
            f"\nRecorded surface conductor '{body_name}': "
            f"material='{material_name}', "
            f"sigma={float(conductivity):.12g} S/m, "
            f"{len(face_records)} original faces"
        )

        for record in face_records:
            print(
                f"  original surface={record['surface']}, "
                f"center={record['center']}, "
                f"area={record['area']:.12g} mm^2, "
                f"bbox={record['bbox']}"
            )

    return records


def config_body_role(config, body_name: str):
    """Return the exported semantic role for a named STEP body."""
    return str(config.get("body_roles", {}).get(body_name, "")).strip()

def config_object_info(config, body_name: str):
    """Return the schema-v2 object record for a named HFSS body."""
    info = config.get("objects", {}).get(body_name, {})
    return info if isinstance(info, dict) else {}


def config_material_info(config, material_name: str):
    """Return a material record using a case-insensitive name lookup."""
    wanted = str(material_name or "").strip().lower()

    for known_name, info in config.get("materials", {}).items():
        if str(known_name).strip().lower() == wanted:
            return info if isinstance(info, dict) else {}

    return {}


def surface_conductor_object_names(config):
    """
    Return the names of solids that HFSS exported as surface conductors.

    These are 3-D CAD solids with solve_inside=False. Their volumes are
    cut from the dielectric mesh, but their exposed surfaces must later
    receive a Palace finite-conductivity boundary rather than PEC.
    """
    names = []

    for body_name, raw_info in config.get("objects", {}).items():
        if not isinstance(raw_info, dict):
            continue

        dimension = raw_info.get("dimension")
        strategy = str(
            raw_info.get("modeling_strategy") or ""
        ).strip()
        role = str(
            raw_info.get("domain_role") or ""
        ).strip()

        if (
            dimension == 3
            and (
                strategy == "surface_conductor"
                or role == "conductor_solid"
            )
        ):
            names.append(str(body_name))

    # Backward-compatible fallback for an older device_config.
    if not names:
        for body_name, role in config.get("body_roles", {}).items():
            if str(role).strip() == "conductor_solid":
                names.append(str(body_name))

    return sorted(set(names))


def impedance_boundary_records(config):
    """
    Return HFSS impedance boundary records that include face geometry.

    The existing assignment and assignment_ids fields are left untouched.
    Only assignment_faces is used for exact STEP-to-Gmsh face recovery.
    """
    records = []
    for boundary_name, info in config.get("boundaries", {}).items():
        role = str(info.get("role") or "")
        boundary_type = str(info.get("type") or "")
        if (
            role != "impedance_sheet"
            and boundary_type not in ("Impedance", "Layered Impedance")
        ):
            continue

        faces = info.get("assignment_faces") or []
        if not faces:
            raise RuntimeError(
                f"Impedance boundary '{boundary_name}' has no "
                "assignment_faces geometry. Re-export with the updated "
                "HFSS exporter."
            )

        records.append((boundary_name, info, faces))
    return records


def match_impedance_surfaces(config, final_cavity, final_chip, excluded_surfaces):
    """
    Match each recorded HFSS impedance face to the final Gmsh surface(s).

    The normal case is a one-to-one center/area match. A conductor cut can,
    however, split one original HFSS face into several final OCC surfaces.
    When no single exact match exists, this function conservatively searches
    for a small coplanar set whose combined area and area-weighted centroid
    reproduce the recorded HFSS face.
    """
    matched_by_boundary = {}
    already_claimed = set()

    def owner_matches(adjacent_volumes, owner):
        adjacent_volumes = set(adjacent_volumes)
        if owner == "cavity":
            return bool(adjacent_volumes & set(final_cavity))
        if owner == "chip":
            return bool(adjacent_volumes & set(final_chip))
        return True

    def planar_axis_and_coordinate(surface):
        box = bbox(2, surface)
        dims = lengths(box)
        axis = min(range(3), key=lambda i: abs(dims[i]))
        # Imported OCC bounding boxes have roughly 2e-7 mm numerical thickness.
        scale = max(max(abs(v) for v in dims), 1.0)
        if abs(dims[axis]) > max(1.0e-5, 1.0e-6 * scale):
            return None, None
        return axis, 0.5 * (box[axis] + box[axis + 3])

    for boundary_name, info, face_records in impedance_boundary_records(config):
        boundary_matches = set()

        for face_record in face_records:
            target_center = face_record.get("center_mm")
            target_area = face_record.get("area_mm2")
            owner = str(face_record.get("body") or "").strip()

            if target_center is None or target_area is None:
                raise RuntimeError(
                    f"Impedance boundary '{boundary_name}' face "
                    f"{face_record.get('face_id')} lacks center_mm or area_mm2"
                )

            target_center = tuple(float(v) for v in target_center)
            target_area = float(target_area)
            area_tolerance = max(
                IMPEDANCE_AREA_ABS_TOL_MM2,
                IMPEDANCE_AREA_REL_TOL * abs(target_area),
            )

            # Gather all usable one-sided boundary surfaces for this owner.
            owner_surfaces = []
            for _, surface in gmsh.model.getEntities(2):
                if surface in excluded_surfaces or surface in already_claimed:
                    continue

                upward, _ = gmsh.model.getAdjacencies(2, surface)
                adjacent_volumes = [int(tag) for tag in upward]
                if len(adjacent_volumes) != 1:
                    continue
                if not owner_matches(adjacent_volumes, owner):
                    continue

                candidate_center = center(bbox(2, surface))
                candidate_area = surface_area(surface)
                owner_surfaces.append(
                    {
                        "surface": surface,
                        "center": candidate_center,
                        "area": candidate_area,
                        "adjacent": adjacent_volumes,
                    }
                )

            # First preserve the original strict one-surface behavior.
            exact_candidates = []
            for item in owner_surfaces:
                center_error = distance3(item["center"], target_center)
                area_error = abs(item["area"] - target_area)
                if (
                    center_error <= IMPEDANCE_CENTER_TOL_MM
                    and area_error <= area_tolerance
                ):
                    relative_area_error = area_error / max(
                        abs(target_area), IMPEDANCE_AREA_ABS_TOL_MM2
                    )
                    score = (
                        center_error / IMPEDANCE_CENTER_TOL_MM
                        + relative_area_error / IMPEDANCE_AREA_REL_TOL
                    )
                    exact_candidates.append((score, item))

            if exact_candidates:
                exact_candidates.sort(key=lambda item: item[0])
                best = exact_candidates[0][1]
                selected = [best]
            else:
                # Fallback for a face split by a conductor boolean.
                #
                # Infer the face plane from nearby final surfaces. The original
                # exported bbox can be incomplete for circular faces, so use
                # the reliable center and area. sqrt(A/pi) gives a conservative
                # in-plane search radius for a disk-like boundary.
                characteristic_radius = max(
                    (abs(target_area) / 3.141592653589793) ** 0.5,
                    IMPEDANCE_CENTER_TOL_MM,
                )
                plane_tol = max(IMPEDANCE_CENTER_TOL_MM, 1.0e-4)
                in_plane_tol = 1.25 * characteristic_radius

                split_candidates = []
                for item in owner_surfaces:
                    axis, plane_coordinate = planar_axis_and_coordinate(
                        item["surface"]
                    )
                    if axis is None:
                        continue
                    if abs(plane_coordinate - target_center[axis]) > plane_tol:
                        continue

                    other_axes = [i for i in range(3) if i != axis]
                    in_plane_distance = sum(
                        (
                            float(item["center"][i])
                            - float(target_center[i])
                        ) ** 2
                        for i in other_axes
                    ) ** 0.5
                    if in_plane_distance > in_plane_tol:
                        continue

                    enriched = dict(item)
                    enriched["axis"] = axis
                    enriched["plane_coordinate"] = plane_coordinate
                    enriched["in_plane_distance"] = in_plane_distance
                    split_candidates.append(enriched)

                # Keep the combinatorial search bounded and deterministic.
                split_candidates.sort(
                    key=lambda item: (
                        item["in_plane_distance"],
                        -item["area"],
                        item["surface"],
                    )
                )
                split_candidates = split_candidates[:12]

                best_group = None
                max_group_size = min(6, len(split_candidates))
                for group_size in range(2, max_group_size + 1):
                    for group in itertools.combinations(
                        split_candidates, group_size
                    ):
                        axes = {item["axis"] for item in group}
                        if len(axes) != 1:
                            continue

                        total_area = sum(item["area"] for item in group)
                        area_error = abs(total_area - target_area)
                        if area_error > area_tolerance:
                            continue

                        centroid = tuple(
                            sum(
                                item["area"] * item["center"][i]
                                for item in group
                            ) / total_area
                            for i in range(3)
                        )
                        centroid_error = distance3(centroid, target_center)
                        if centroid_error > IMPEDANCE_CENTER_TOL_MM:
                            continue

                        score = (
                            area_error / max(area_tolerance, 1.0e-30)
                            + centroid_error
                            / max(IMPEDANCE_CENTER_TOL_MM, 1.0e-30)
                            + 1.0e-6 * group_size
                        )
                        if best_group is None or score < best_group[0]:
                            best_group = (
                                score,
                                list(group),
                                total_area,
                                centroid,
                            )

                if best_group is None:
                    # A conductor cut can remove part of the original HFSS
                    # boundary rather than split it into surfaces whose areas
                    # still sum to the original area. Example: an impedance
                    # disk of radius 1 mm with a concentric conductor of radius
                    # 0.5 mm becomes an exposed annulus of area 0.75*pi.
                    #
                    # In that case the correct final boundary has:
                    #   - the same owning dielectric;
                    #   - the same plane and center;
                    #   - a smaller positive area;
                    #   - a unique match.
                    clipped_candidates = []
                    for item in split_candidates:
                        center_error = distance3(item["center"], target_center)
                        if center_error > IMPEDANCE_CENTER_TOL_MM:
                            continue
                        if not (
                            IMPEDANCE_AREA_ABS_TOL_MM2
                            < item["area"]
                            < target_area + area_tolerance
                        ):
                            continue

                        retained_fraction = item["area"] / max(
                            target_area, IMPEDANCE_AREA_ABS_TOL_MM2
                        )
                        # Reject tiny fragments; a clipped impedance face
                        # should retain a substantial part of the original.
                        if retained_fraction < 0.25:
                            continue

                        score = (
                            center_error
                            / max(IMPEDANCE_CENTER_TOL_MM, 1.0e-30)
                            + (1.0 - retained_fraction)
                        )
                        clipped_candidates.append(
                            (score, retained_fraction, item)
                        )

                    clipped_candidates.sort(
                        key=lambda entry: (entry[0], -entry[1], entry[2]["surface"])
                    )

                    if clipped_candidates:
                        best_score, retained_fraction, best_item = (
                            clipped_candidates[0]
                        )
                        if (
                            len(clipped_candidates) > 1
                            and abs(clipped_candidates[1][0] - best_score)
                            < 1.0e-6
                        ):
                            raise RuntimeError(
                                f"Ambiguous clipped-face match for impedance "
                                f"boundary '{boundary_name}', HFSS face "
                                f"{face_record.get('face_id')}: candidate "
                                f"surfaces {best_item['surface']} and "
                                f"{clipped_candidates[1][2]['surface']}."
                            )

                        selected = [best_item]
                        print(
                            f"  {boundary_name}: HFSS face "
                            f"{face_record.get('face_id')} was clipped by a "
                            f"conductor boolean; using final exposed surface "
                            f"{best_item['surface']} with "
                            f"area={best_item['area']:.12g} mm^2 "
                            f"({100.0 * retained_fraction:.6g}% of the "
                            f"original {target_area:.12g} mm^2)"
                        )
                    else:
                        diagnostic = sorted(
                            (
                                item["surface"],
                                item["center"],
                                item["area"],
                                item["adjacent"],
                            )
                            for item in owner_surfaces
                            if distance3(item["center"], target_center)
                            <= max(2.0 * characteristic_radius, 0.25)
                        )
                        raise RuntimeError(
                            f"Could not match HFSS impedance boundary "
                            f"'{boundary_name}', face "
                            f"{face_record.get('face_id')}, owner={owner!r}, "
                            f"center={target_center}, area={target_area} mm^2. "
                            f"Nearby final one-sided surfaces were: "
                            f"{diagnostic}"
                        )
                else:
                    _, selected, combined_area, combined_centroid = best_group
                    print(
                        f"  {boundary_name}: original HFSS face "
                        f"{face_record.get('face_id')} was split by OCC "
                        f"booleans; recovered {len(selected)} descendants "
                        f"with combined area={combined_area:.12g} mm^2 and "
                        f"centroid={combined_centroid}"
                    )

            for item in selected:
                surface = item["surface"]
                boundary_matches.add(surface)
                already_claimed.add(surface)
                print(
                    f"    HFSS face={face_record.get('face_id')} "
                    f"body={owner!r} -> Gmsh surface={surface}, "
                    f"center={item['center']}, "
                    f"area={item['area']:.12g} mm^2, "
                    f"adjacent_volumes={item['adjacent']}"
                )

        matched_by_boundary[boundary_name] = boundary_matches

    return matched_by_boundary

def match_surface_conductor_surfaces(
    conductor_records,
    final_cavity,
    final_chip,
    excluded_surfaces,
):
    """
    Recover the final dielectric-boundary surfaces created by subtracting
    each solve-inside-false conductor solid.

    Matching is performed after all Boolean and sheet-fragmentation
    operations, because those operations can renumber or split surfaces.

    Exact one-surface matches are preferred. If an original conductor
    face was split, a small collection of descendants is accepted when
    its combined area and area-weighted centroid reproduce the original
    face.

    Returns:

        {
            "Pin_1": {surface_tag, ...}
        }
    """
    matched_by_object = {}
    already_claimed = set()
    solved_domains = set(final_cavity) | set(final_chip)

    # Gather the final one-sided surfaces of solved material domains.
    final_boundary_surfaces = []

    for _, surface in gmsh.model.getEntities(2):
        if surface in excluded_surfaces:
            continue

        upward, _ = gmsh.model.getAdjacencies(2, surface)
        adjacent_volumes = [int(tag) for tag in upward]

        if len(adjacent_volumes) != 1:
            continue

        if not (set(adjacent_volumes) & solved_domains):
            continue

        surface_box = bbox(2, surface)

        final_boundary_surfaces.append(
            {
                "surface": int(surface),
                "bbox": tuple(
                    float(value)
                    for value in surface_box
                ),
                "center": bbox_center(surface_box),
                "area": surface_area(surface),
                "adjacent": adjacent_volumes,
            }
        )

    for body_name, conductor_record in conductor_records.items():
        body_matches = set()
        volume_box = conductor_record["volume_bbox"]

        # Restrict matching to surfaces geometrically contained within
        # the original conductor's volume bounding box. This prevents a
        # nearby cavity wall or impedance annulus from being selected.
        body_candidates = [
            item
            for item in final_boundary_surfaces
            if item["surface"] not in already_claimed
            and bbox_is_inside(
                item["bbox"],
                volume_box,
                CONDUCTOR_BBOX_TOL_MM,
            )
        ]

        if not body_candidates:
            raise RuntimeError(
                f"No final dielectric-boundary surfaces lie within the "
                f"original bounding box of surface conductor "
                f"'{body_name}': {volume_box}"
            )

        print(
            f"\nMatching final surfaces for surface conductor "
            f"'{body_name}':"
        )

        for face_record in conductor_record["faces"]:
            target_box = face_record["bbox"]
            target_center = face_record["center"]
            target_area = float(face_record["area"])

            area_tolerance = max(
                CONDUCTOR_AREA_ABS_TOL_MM2,
                CONDUCTOR_AREA_REL_TOL * abs(target_area),
            )

            exact_matches = []

            for item in body_candidates:
                if item["surface"] in already_claimed:
                    continue

                center_error = distance3(
                    item["center"],
                    target_center,
                )
                area_error = abs(
                    float(item["area"]) - target_area
                )
                box_error = bbox_max_error(
                    item["bbox"],
                    target_box,
                )

                if (
                    center_error <= CONDUCTOR_CENTER_TOL_MM
                    and area_error <= area_tolerance
                    and box_error <= CONDUCTOR_BBOX_TOL_MM
                ):
                    score = (
                        center_error
                        / max(CONDUCTOR_CENTER_TOL_MM, 1.0e-30)
                        + area_error
                        / max(area_tolerance, 1.0e-30)
                        + box_error
                        / max(CONDUCTOR_BBOX_TOL_MM, 1.0e-30)
                    )
                    exact_matches.append((score, item))

            if exact_matches:
                exact_matches.sort(
                    key=lambda entry: (
                        entry[0],
                        entry[1]["surface"],
                    )
                )

                if (
                    len(exact_matches) > 1
                    and abs(
                        exact_matches[1][0]
                        - exact_matches[0][0]
                    ) < 1.0e-8
                ):
                    raise RuntimeError(
                        f"Ambiguous conductor-face match for "
                        f"'{body_name}', original surface "
                        f"{face_record['surface']}: candidates "
                        f"{exact_matches[0][1]['surface']} and "
                        f"{exact_matches[1][1]['surface']}"
                    )

                selected = [exact_matches[0][1]]

            else:
                # The original conductor face may have been split by a
                # later Boolean or sheet fragmentation. Search for a
                # small set of descendants contained within its original
                # bounding box.
                descendant_candidates = [
                    item
                    for item in body_candidates
                    if item["surface"] not in already_claimed
                    and bbox_is_inside(
                        item["bbox"],
                        target_box,
                        CONDUCTOR_BBOX_TOL_MM,
                    )
                ]

                descendant_candidates.sort(
                    key=lambda item: (
                        distance3(
                            item["center"],
                            target_center,
                        ),
                        -float(item["area"]),
                        item["surface"],
                    )
                )

                # Keep the combinatorial search bounded.
                descendant_candidates = descendant_candidates[:12]

                best_group = None
                max_group_size = min(
                    6,
                    len(descendant_candidates),
                )

                for group_size in range(
                    2,
                    max_group_size + 1,
                ):
                    for group in itertools.combinations(
                        descendant_candidates,
                        group_size,
                    ):
                        total_area = sum(
                            float(item["area"])
                            for item in group
                        )

                        if total_area <= 0.0:
                            continue

                        area_error = abs(
                            total_area - target_area
                        )

                        if area_error > area_tolerance:
                            continue

                        centroid = tuple(
                            sum(
                                float(item["area"])
                                * float(item["center"][axis])
                                for item in group
                            )
                            / total_area
                            for axis in range(3)
                        )

                        centroid_error = distance3(
                            centroid,
                            target_center,
                        )

                        if (
                            centroid_error
                            > CONDUCTOR_CENTER_TOL_MM
                        ):
                            continue

                        score = (
                            area_error
                            / max(area_tolerance, 1.0e-30)
                            + centroid_error
                            / max(
                                CONDUCTOR_CENTER_TOL_MM,
                                1.0e-30,
                            )
                            + 1.0e-6 * group_size
                        )

                        if (
                            best_group is None
                            or score < best_group[0]
                        ):
                            best_group = (
                                score,
                                list(group),
                                total_area,
                                centroid,
                            )

                if best_group is None:
                    # Not every original conductor face must necessarily
                    # border a solved dielectric. For example, one face
                    # can be buried inside another excluded conductor.
                    #
                    # Skip only when no geometrically plausible final
                    # candidate exists. The object-level validation below
                    # still requires at least one exposed surface.
                    plausible = [
                        item
                        for item in body_candidates
                        if item["surface"] not in already_claimed
                        and bbox_is_inside(
                            item["bbox"],
                            target_box,
                            CONDUCTOR_BBOX_TOL_MM,
                        )
                    ]

                    if plausible:
                        raise RuntimeError(
                            f"Could not uniquely recover conductor "
                            f"'{body_name}' original surface "
                            f"{face_record['surface']}; plausible final "
                            f"surfaces were "
                            f"{[item['surface'] for item in plausible]}"
                        )

                    print(
                        f"  original conductor surface "
                        f"{face_record['surface']} has no exposed "
                        "descendant in a solved dielectric; skipping"
                    )
                    continue

                _, selected, combined_area, combined_centroid = (
                    best_group
                )

                print(
                    f"  original conductor surface "
                    f"{face_record['surface']} was split; recovered "
                    f"{len(selected)} descendants with "
                    f"combined area={combined_area:.12g} mm^2 and "
                    f"centroid={combined_centroid}"
                )

            for item in selected:
                surface = item["surface"]

                if surface in excluded_surfaces:
                    raise RuntimeError(
                        f"Conductor surface {surface} for "
                        f"'{body_name}' is already reserved for another "
                        "boundary"
                    )

                body_matches.add(surface)
                already_claimed.add(surface)

                print(
                    f"  original surface={face_record['surface']} "
                    f"-> final surface={surface}, "
                    f"center={item['center']}, "
                    f"area={item['area']:.12g} mm^2, "
                    f"adjacent_volumes={item['adjacent']}"
                )

        if not body_matches:
            raise RuntimeError(
                f"Surface conductor '{body_name}' produced no exposed "
                "finite-conductivity boundary surfaces"
            )

        matched_by_object[body_name] = body_matches

    return matched_by_object


def main():
    gmsh.initialize()

    try:
        gmsh.model.add("palace_device")
        occ = gmsh.model.occ

        print(f"\nImporting combined STEP file: {STEP_FILE}")
        print("Importing all dimensions, including standalone 2-D sheets")
        try:
            gmsh.option.setNumber("Geometry.OCCImportLabels", 1)
        except Exception:
            pass
        occ.importShapes(STEP_FILE, highestDimOnly=False)
        occ.synchronize()

        # Read the AEDT names from the STEP file and match them onto the
        # imported Gmsh entities. Everything downstream identifies bodies
        # by name from here on, rather than by guessed dimensions.
        _identify_bodies(STEP_FILE)
        device_config = load_device_config()

        volume_by_name = {}
        print("\nImported volumes:")
        for _, tag in gmsh.model.getEntities(3):
            name = classify_volume(tag)
            if name in volume_by_name:
                raise RuntimeError(
                    f"Duplicate volume classification for '{name}': {volume_by_name[name]} and {tag}"
                )
            volume_by_name[name] = tag
            print(f"  {name:12s} tag={tag:4d} bbox={bbox(3, tag)}")
            
        # Record finite-conductivity solid geometry before any Boolean
        # operation removes or renumbers its faces.
        surface_conductor_records = (
            record_surface_conductor_geometry(
                device_config,
                volume_by_name,
            )
        )

        required_volumes = {"cavity"}
        if ENABLE_CHIP:
            required_volumes.add("chip")
        if ENABLE_PIN:
            required_volumes.add("pin")
        if ENABLE_BONDWIRE:
            required_volumes.add("Bondwire1")

        missing_volumes = required_volumes - set(volume_by_name)
        if missing_volumes:
            raise RuntimeError(
                f"Missing required volumes: {sorted(missing_volumes)}; found: {sorted(volume_by_name)}"
            )

        # Remove imported volumes that are present in the combined STEP
        # but disabled for this diagnostic run. Otherwise they remain as
        # unassigned 3-D regions later in the pipeline.
        disabled_volume_names = []

        if not ENABLE_PIN and "pin" in volume_by_name:
            disabled_volume_names.append("pin")

        if not ENABLE_BONDWIRE and "Bondwire1" in volume_by_name:
            disabled_volume_names.append("Bondwire1")

        if not ENABLE_CHIP and "chip" in volume_by_name:
            disabled_volume_names.append("chip")

        if disabled_volume_names:
            disabled_dim_tags = [
                (3, volume_by_name[name])
                for name in disabled_volume_names
            ]
            occ.remove(disabled_dim_tags, recursive=True)
            occ.synchronize()

            print(
                "\nRemoved disabled imported volumes:",
                disabled_volume_names,
            )

            for name in disabled_volume_names:
                del volume_by_name[name]

        # In a combined STEP export, sheets that lie exactly on a solid
        # face can already report one adjacent volume after import.
        # Therefore, zero upward adjacency is not a reliable sheet test.
        # Instead, scan every imported 2-D entity and keep only surfaces
        # whose bounding boxes match the known pad/lead/JJ dimensions.
        sheet_inputs_by_name = {
            "pads": [],
            "medium_lead": [],
            "thin_lead": [],
            "JJ": [],
            "RR": [],
        }

        print("\nImported sheet candidates:")
        print("  Scanning all imported 2-D entities...")
        unclassified_planar = []

        for _, tag in gmsh.model.getEntities(2):
            box = bbox(2, tag)
            dims = sorted(lengths(box))

            try:
                name = classify_sheet(tag)
            except RuntimeError:
                if dims[0] <= 1.0e-4:
                    unclassified_planar.append((tag, dims, box))
                continue

            upward, _ = gmsh.model.getAdjacencies(2, tag)
            sheet_inputs_by_name[name].append(tag)

            print(
                f"  {name:14s} tag={tag:4d} "
                f"imported_name={entity_name(2, tag)!r} "
                f"adjacent_volumes={list(upward)} "
                f"dims={dims} bbox={box}"
            )

        if unclassified_planar:
            print("\nUnclassified planar surfaces:")
            for tag, dims, box in sorted(
                unclassified_planar,
                key=lambda item: (item[1][2], item[1][1]),
            ):
                print(
                    f"  tag={tag:4d} imported_name={entity_name(2, tag)!r} dims={dims} bbox={box}"
                )

        total_sheet_candidates = sum(
            len(tags) for tags in sheet_inputs_by_name.values()
        )

        if total_sheet_candidates == 0:
            raise RuntimeError(
                "No pad, lead, or JJ sheet surfaces were identified. "
                "Update classify_sheet() tolerances using the imported "
                "surface bounding boxes."
            )

        enabled_sheet_names = []
        if ENABLE_PADS:
            enabled_sheet_names.append("pads")
        if ENABLE_MEDIUM_LEAD:
            enabled_sheet_names.append("medium_lead")
        if ENABLE_THIN_LEAD:
            enabled_sheet_names.append("thin_lead")
        if ENABLE_JJ:
            enabled_sheet_names.append("JJ")
        if ENABLE_RR:
            enabled_sheet_names.append("RR")

        for name in enabled_sheet_names:
            if not sheet_inputs_by_name[name]:
                raise RuntimeError(f"No enabled '{name}' sheet was found")

        if ENABLE_JJ and len(sheet_inputs_by_name["JJ"]) != 1:
            raise RuntimeError(
                f"Expected exactly one JJ sheet, found {sheet_inputs_by_name['JJ']}"
            )

        active_sheet_tags = [
            tag
            for name in enabled_sheet_names
            for tag in sheet_inputs_by_name[name]
        ]

        disabled_sheet_tags = [
            tag
            for name, tags in sheet_inputs_by_name.items()
            if name not in enabled_sheet_names
            for tag in tags
        ]
        if disabled_sheet_tags:
            occ.remove([(2, tag) for tag in disabled_sheet_tags], recursive=True)
            occ.synchronize()
            print("\nRemoved disabled sheet tags:", sorted(disabled_sheet_tags))

        cavity = (3, volume_by_name["cavity"])
        chip = (3, volume_by_name["chip"]) if ENABLE_CHIP else None

        if ENABLE_CHIP:
            cavity_parts, _ = occ.cut(
                [cavity], [chip], removeObject=True, removeTool=False
            )
            occ.synchronize()
            cavity_parts = [dt for dt in cavity_parts if dt[0] == 3]
            chip_parts = [chip]
        else:
            cavity_parts = [cavity]
            chip_parts = []

        pec_solid_tools = []
        if ENABLE_PIN:
            pec_solid_tools.append((3, volume_by_name["pin"]))
        if ENABLE_BONDWIRE:
            pec_solid_tools.append((3, volume_by_name["Bondwire1"]))

        # Preserve the existing explicit PEC handling, and additionally cut
        # finite-conductivity solids exported as "conductor_solid". This keeps
        # copper objects such as Pin_1 out of the dielectric material domains.
        existing_tool_tags = {tag for _, tag in pec_solid_tools}
        for body_name, tag in sorted(volume_by_name.items()):
            if tag in existing_tool_tags:
                continue
            if config_body_role(device_config, body_name) == "conductor_solid":
                pec_solid_tools.append((3, tag))
                existing_tool_tags.add(tag)
                print(
                    f"\nTreating finite-conductivity solid '{body_name}' "
                    f"(tag {tag}) as a conductor cut tool"
                )

        dielectric_inputs = cavity_parts + chip_parts
        if pec_solid_tools:
            _, cut_map = occ.cut(
                dielectric_inputs,
                pec_solid_tools,
                removeObject=True,
                removeTool=True,
            )
            occ.synchronize()
        else:
            cut_map = [[dt] for dt in dielectric_inputs]

        cavity_input_count = len(cavity_parts)
        cavity_domains = set()
        chip_domains = set()

        for index, descendants in enumerate(cut_map[:len(dielectric_inputs)]):
            volume_descendants = {tag for dim, tag in descendants if dim == 3}
            if index < cavity_input_count:
                cavity_domains.update(volume_descendants)
            else:
                chip_domains.update(volume_descendants)

        all_current_volumes = {tag for _, tag in gmsh.model.getEntities(3)}

        if ENABLE_CHIP and (not cavity_domains or not chip_domains):
            chip_box = (-2.0001, 2.4999, 16.5699, 2.0001, 22.5001, 17.0001)
            cavity_domains.clear()
            chip_domains.clear()
            for tag in all_current_volumes:
                cx, cy, cz = center(bbox(3, tag))
                inside_chip = (
                    chip_box[0] <= cx <= chip_box[3]
                    and chip_box[1] <= cy <= chip_box[4]
                    and chip_box[2] <= cz <= chip_box[5]
                )
                if inside_chip:
                    chip_domains.add(tag)
                else:
                    cavity_domains.add(tag)
        elif not ENABLE_CHIP and not cavity_domains:
            cavity_domains = set(all_current_volumes)

        print("\nDielectric domains after solid cuts:")
        print("  cavity:", sorted(cavity_domains))
        print("  chip:  ", sorted(chip_domains))

        if not cavity_domains:
            raise RuntimeError("Failed to preserve the cavity domain")
        if ENABLE_CHIP and not chip_domains:
            raise RuntimeError("Failed to preserve the chip domain")

        # Make the touching cavity/chip material interface conformal before
        # imprinting the metal sheets. A simple cut can leave coincident but
        # topologically separate faces. Sheets placed on that interface then
        # appear one-sided. Fragmenting the two dielectric domain sets together
        # forces a shared material-interface face.
        if ENABLE_CHIP:
            cavity_before_interface = [
                (3, tag) for tag in sorted(cavity_domains)
            ]
            chip_before_interface = [
                (3, tag) for tag in sorted(chip_domains)
            ]

            _, interface_map = occ.fragment(
                cavity_before_interface,
                chip_before_interface,
                removeObject=True,
                removeTool=True,
            )
            occ.synchronize()

            new_cavity_domains = set()
            new_chip_domains = set()
            n_cavity_before_interface = len(cavity_before_interface)

            for index, descendants in enumerate(interface_map):
                descendants_3d = {
                    tag for dim, tag in descendants if dim == 3
                }

                if index < n_cavity_before_interface:
                    new_cavity_domains.update(descendants_3d)
                else:
                    new_chip_domains.update(descendants_3d)

            current_volumes = {
                tag for _, tag in gmsh.model.getEntities(3)
            }
            new_cavity_domains &= current_volumes
            new_chip_domains &= current_volumes

            interface_overlap = (
                new_cavity_domains & new_chip_domains
            )
            if interface_overlap:
                raise RuntimeError(
                    "Cavity/chip conformalization produced overlapping "
                    f"material assignments: {sorted(interface_overlap)}"
                )

            if not new_cavity_domains or not new_chip_domains:
                raise RuntimeError(
                    "Failed to conformalize the cavity/chip interface"
                )

            cavity_domains = new_cavity_domains
            chip_domains = new_chip_domains

            print("\nConformalized cavity/chip interface:")
            print("  cavity:", sorted(cavity_domains))
            print("  chip:  ", sorted(chip_domains))

        domain_inputs = (
            [(3, tag) for tag in sorted(cavity_domains)]
            + [(3, tag) for tag in sorted(chip_domains)]
        )
        sheet_inputs = [(2, tag) for tag in active_sheet_tags]

        cavity_input_count = len(cavity_domains)
        chip_input_count = len(chip_domains)
        final_cavity = set()
        final_chip = set()
        sheet_descendants = {tag: set() for tag in active_sheet_tags}

        if sheet_inputs:
            _, fragment_map = occ.fragment(
                domain_inputs,
                sheet_inputs,
                removeObject=True,
                removeTool=True,
            )
            occ.synchronize()

            for index, descendants in enumerate(fragment_map):
                if index < cavity_input_count:
                    final_cavity.update(tag for dim, tag in descendants if dim == 3)
                elif index < cavity_input_count + chip_input_count:
                    final_chip.update(tag for dim, tag in descendants if dim == 3)
                else:
                    sheet_index = index - cavity_input_count - chip_input_count
                    original_sheet = active_sheet_tags[sheet_index]
                    sheet_descendants[original_sheet].update(
                        tag for dim, tag in descendants if dim == 2
                    )
        else:
            final_cavity = set(cavity_domains)
            final_chip = set(chip_domains)

        final_volumes = {tag for _, tag in gmsh.model.getEntities(3)}
        final_cavity &= final_volumes
        final_chip &= final_volumes

        overlap = final_cavity & final_chip
        if overlap:
            raise RuntimeError(
                f"Volume fragments assigned to both cavity and chip: {sorted(overlap)}"
            )

        unassigned_volumes = final_volumes - final_cavity - final_chip
        if unassigned_volumes:
            raise RuntimeError(
                "Unassigned final volume fragments: "
                f"{sorted(unassigned_volumes)}. Check body_roles in "
                "device_config.json; finite-conductivity metal solids should "
                "be exported as 'conductor_solid'."
            )

        final_sheets_by_name = {
            "pads": set(),
            "medium_lead": set(),
            "thin_lead": set(),
            "JJ": set(),
            "RR": set(),
        }
        for name, original_tags in sheet_inputs_by_name.items():
            for original_tag in original_tags:
                final_sheets_by_name[name].update(
                    sheet_descendants.get(original_tag, set())
                )

        print("\nFinal material and sheet fragments:")
        print("  cavity:", sorted(final_cavity))
        print("  chip:  ", sorted(final_chip))
        for name, tags in final_sheets_by_name.items():
            print(f"  {name:14s}: {sorted(tags)}")
            if name in enabled_sheet_names and not tags:
                raise RuntimeError(f"No final surface descendants found for '{name}'")

        one_sided_contact_surfaces = set()

        print("\nSheet adjacency validation:")
        for name in enabled_sheet_names:
            two_sided_surfaces = set()

            for surface in sorted(final_sheets_by_name[name]):
                upward, _ = gmsh.model.getAdjacencies(2, surface)
                adjacent_volumes = list(upward)
                print(
                    f"  {name:14s} surface={surface:4d} "
                    f"adjacent_volumes={adjacent_volumes} "
                    f"bbox={bbox(2, surface)}"
                )

                if len(adjacent_volumes) == 2:
                    two_sided_surfaces.add(surface)
                elif len(adjacent_volumes) == 1 and name in KEEP_ONE_SIDED_SHEET_GROUPS:
                    two_sided_surfaces.add(surface)
                    print(
                        f"    -> keeping one-sided {name} surface "
                        f"{surface} as its own PEC boundary group"
                    )
                elif len(adjacent_volumes) == 1 and name != "JJ":
                    one_sided_contact_surfaces.add(surface)
                    print(
                        f"    -> moving one-sided {name} contact surface "
                        f"{surface} to PEC_exterior"
                    )
                else:
                    raise RuntimeError(
                        f"Sheet '{name}' surface {surface} is not a valid internal boundary: "
                        f"found {len(adjacent_volumes)} adjacent volumes"
                    )

            final_sheets_by_name[name] = two_sided_surfaces

        all_enabled_sheet_surfaces = set().union(
            *[final_sheets_by_name[name] for name in enabled_sheet_names]
        )

        print("\nConfig-driven impedance boundary matching:")
        impedance_surfaces_by_name = match_impedance_surfaces(
            device_config,
            final_cavity,
            final_chip,
            all_enabled_sheet_surfaces,
        )
        all_impedance_surfaces = set().union(
            *impedance_surfaces_by_name.values()
        ) if impedance_surfaces_by_name else set()

        print("\nConfig-driven finite-conductivity matching:")
        conductor_exclusions = (
            set(all_enabled_sheet_surfaces)
            | set(all_impedance_surfaces)
        )

        conductor_surfaces_by_name = (
            match_surface_conductor_surfaces(
                surface_conductor_records,
                final_cavity,
                final_chip,
                conductor_exclusions,
            )
        )

        all_conductor_surfaces = set().union(
            *conductor_surfaces_by_name.values()
        ) if conductor_surfaces_by_name else set()

        overlap_impedance_conductor = (
            all_impedance_surfaces
            & all_conductor_surfaces
        )
        if overlap_impedance_conductor:
            raise RuntimeError(
                "The following surfaces were assigned to both "
                "Impedance and finite Conductivity: "
                f"{sorted(overlap_impedance_conductor)}"
            )

        pec_exterior_surfaces = set(
            one_sided_contact_surfaces
        )

        # Explicit non-PEC boundaries must never fall into PEC_exterior.
        pec_exterior_surfaces -= all_impedance_surfaces
        pec_exterior_surfaces -= all_conductor_surfaces

        for _, surface in gmsh.model.getEntities(2):
            if (
                surface in all_enabled_sheet_surfaces
                or surface in all_impedance_surfaces
                or surface in all_conductor_surfaces
            ):
                continue

            upward, _ = gmsh.model.getAdjacencies(2, surface)

            if len(upward) == 1:
                pec_exterior_surfaces.add(surface)

            elif len(upward) > 2:
                raise RuntimeError(
                    f"Non-manifold surface {surface} has "
                    f"{len(upward)} adjacent volumes"
                )

        # Validate that no physical surface is assigned to incompatible
        # boundary groups.
        overlap_pec_impedance = (
            pec_exterior_surfaces
            & all_impedance_surfaces
        )

        if overlap_pec_impedance:
            raise RuntimeError(
                "Surfaces assigned to both PEC_exterior and "
                f"Impedance: {sorted(overlap_pec_impedance)}"
            )

        overlap_pec_conductor = (
            pec_exterior_surfaces
            & all_conductor_surfaces
        )

        if overlap_pec_conductor:
            raise RuntimeError(
                "Surfaces assigned to both PEC_exterior and "
                "finite Conductivity: "
                f"{sorted(overlap_pec_conductor)}"
            )

        overlap_impedance_conductor = (
            all_impedance_surfaces
            & all_conductor_surfaces
        )

        if overlap_impedance_conductor:
            raise RuntimeError(
                "Surfaces assigned to both Impedance and finite "
                "Conductivity: "
                f"{sorted(overlap_impedance_conductor)}"
            )

        physical_groups = {}

        physical_groups = {}
        physical_groups["cavity"] = gmsh.model.addPhysicalGroup(
            3, sorted(final_cavity), name="cavity"
        )
        if ENABLE_CHIP:
            physical_groups["chip"] = gmsh.model.addPhysicalGroup(
                3, sorted(final_chip), name="chip"
            )

        physical_groups["PEC_exterior"] = gmsh.model.addPhysicalGroup(
            2, sorted(pec_exterior_surfaces), name="PEC_exterior"
        )

        for boundary_name, surfaces in impedance_surfaces_by_name.items():
            if not surfaces:
                continue
            physical_groups[boundary_name] = gmsh.model.addPhysicalGroup(
                2, sorted(surfaces), name=boundary_name
            )
            
        for body_name, surfaces in conductor_surfaces_by_name.items():
            if surfaces:
                group_name = f"Conductivity::{body_name}"
                print(
                    f"  {group_name:24s}: "
                    f"{count_2d_elements(sorted(surfaces))} triangles"
                )
            
        for body_name, surfaces in conductor_surfaces_by_name.items():
            if not surfaces:
                continue

            group_name = f"Conductivity::{body_name}"

            physical_groups[group_name] = (
                gmsh.model.addPhysicalGroup(
                    2,
                    sorted(surfaces),
                    name=group_name,
                )
            )

        for name in ("pads", "medium_lead", "thin_lead", "JJ", "RR"):
            tags = sorted(final_sheets_by_name[name])
            if tags:
                physical_groups[name] = gmsh.model.addPhysicalGroup(
                    2, tags, name=name
                )

        print("\nPhysical groups:")
        for name, attribute in physical_groups.items():
            print(f"  {name:16s} -> attribute {attribute}")

        gmsh.option.setNumber("Mesh.MeshSizeMin", SIZE_JJ)
        gmsh.option.setNumber("Mesh.MeshSizeMax", SIZE_CAVITY)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1)

        size_groups = []
        if ENABLE_PADS:
            size_groups.append(("pads", final_sheets_by_name["pads"], SIZE_PADS))
        if ENABLE_MEDIUM_LEAD:
            size_groups.append(("medium_lead", final_sheets_by_name["medium_lead"], SIZE_MEDIUM))
        if ENABLE_THIN_LEAD:
            size_groups.append(("thin_lead", final_sheets_by_name["thin_lead"], SIZE_THIN))
        if ENABLE_JJ:
            size_groups.append(("JJ", final_sheets_by_name["JJ"], SIZE_JJ))
        if ENABLE_RR:
            size_groups.append(("RR", final_sheets_by_name["RR"], SIZE_RR))

        print("\nApplied point mesh sizes:")
        for name, surfaces, size in size_groups:
            point_count = set_surface_point_size(sorted(surfaces), size)
            print(f"  {name:14s}: {size:g} mm on {point_count} boundary points")

        gmsh.model.mesh.generate(3)

        total_tetrahedra = 0
        _, volume_element_tags, _ = gmsh.model.mesh.getElements(3)
        for tags in volume_element_tags:
            total_tetrahedra += len(tags)

        material_tetrahedra = 0
        material_names = ["cavity"] + (["chip"] if ENABLE_CHIP else [])
        for name in material_names:
            attribute = physical_groups[name]
            volume_tags = gmsh.model.getEntitiesForPhysicalGroup(3, attribute)
            for volume in volume_tags:
                _, element_tags, _ = gmsh.model.mesh.getElements(3, volume)
                material_tetrahedra += sum(len(tags) for tags in element_tags)

        print("\nMesh validation:")
        print(f"  total tetrahedra:          {total_tetrahedra}")
        print(f"  material tetrahedra:       {material_tetrahedra}")
        print(f"  total nodes:               {len(gmsh.model.mesh.getNodes()[0])}")

        if total_tetrahedra != material_tetrahedra:
            raise RuntimeError(
                "Not all tetrahedra belong to a material group: "
                f"{material_tetrahedra}/{total_tetrahedra}"
            )

        print("\nSheet mesh element counts:")
        for name in ("pads", "medium_lead", "thin_lead", "JJ", "RR"):
            surfaces = sorted(final_sheets_by_name[name])
            if surfaces:
                print(f"  {name:14s}: {count_2d_elements(surfaces)} triangles")
        for boundary_name, surfaces in impedance_surfaces_by_name.items():
            if surfaces:
                print(
                    f"  {boundary_name:14s}: "
                    f"{count_2d_elements(sorted(surfaces))} triangles"
                )
                
                print("\nFinite-conductivity boundary summary:")
                
        for body_name, surfaces in conductor_surfaces_by_name.items():
            record = surface_conductor_records[body_name]
            print(
                f"  Conductivity::{body_name}: "
                f"surfaces={sorted(surfaces)}, "
                f"material='{record['material']}', "
                f"sigma={record['conductivity_S_per_m']:.12g} S/m"
            )         
        
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        gmsh.write(OUT_MSH)
        gmsh.write(OUT_VTK)

        with open(OUT_GROUPS, "w") as file:
            json.dump(physical_groups, file, indent=2)

        print(f"\nWritten: {OUT_MSH}")
        print(f"Written: {OUT_VTK}")
        print(f"Written: {OUT_GROUPS}")
        print(f"\nOpen the mesh with:\n  gmsh {OUT_MSH}")

    finally:
        gmsh.finalize()


if __name__ == "__main__":
    main()
"""
palace_matchers.py

Flag-free geometric-matching library for mesh_any.py, extracted verbatim
from build_from_single_step.py (patched 2026-08-04 version). Contains NO
device names, NO ENABLE_* toggles, NO main() -- pure functions over the
open gmsh model plus device_config.json.

Provides:
    record_surface_conductor_geometry(config, volume_by_name)
    match_impedance_surfaces(config, diel_a, diel_b, excluded)
    match_surface_conductor_surfaces(records, diel_a, diel_b, excluded)
    set_surface_point_size(surface_tags, size)
    count_2d_elements(surface_tags)
plus the small config/bbox helpers they depend on.

Note: match_impedance_surfaces' owner check compares against the literal
owner names "cavity"/"chip" recorded by the HFSS exporter; callers with
generic dielectrics should pass the full solved-volume set as BOTH the
second and third arguments (mesh_any.py does).
"""

from __future__ import annotations

import itertools
import re

import gmsh


IMPEDANCE_CENTER_TOL_MM = 0.05
IMPEDANCE_AREA_REL_TOL = 0.02
IMPEDANCE_AREA_ABS_TOL_MM2 = 1.0e-8
CONDUCTOR_CENTER_TOL_MM = 0.02
CONDUCTOR_BBOX_TOL_MM = 0.02
CONDUCTOR_AREA_REL_TOL = 0.02
CONDUCTOR_AREA_ABS_TOL_MM2 = 1.0e-8


def normalize_label(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def entity_name(dim: int, tag: int) -> str:
    # Works when Gmsh imports STEP labels. Empty string is fine; the
    # bounding-box classifiers below still handle the original sheets.
    try:
        return gmsh.model.getEntityName(dim, tag) or ""
    except Exception:
        return ""


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


def config_body_role(config, body_name: str):
    """Return the exported semantic role for a named STEP body."""
    return str(config.get("body_roles", {}).get(body_name, "")).strip()


def config_object_info(config, body_name: str):
    """Return the schema-v2/v3 object record for a named HFSS body."""
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
                        # BURIED-FACE CASE. An impedance boundary can be
                        # assigned to a face of a solve-inside-false solid
                        # (conductor/PEC) that sits FLUSH with the exterior
                        # of the solved domain -- e.g. a pin whose end cap
                        # is coplanar with the cavity wall. After the
                        # Boolean cut that face has no descendant at all:
                        # the solved vacuum never touches it, so it carries
                        # no field in HFSS either. Skipping it is correct
                        # physics, exactly as the conductor matcher skips
                        # original faces with no exposed descendant.
                        #
                        # Guarded on BOTH conditions so a genuine matching
                        # failure on a dielectric-owned face still raises:
                        #   1. the owner body was removed from the domain
                        #      (pec_solid / conductor_solid / excluded);
                        #   2. no coplanar candidate could plausibly be a
                        #      descendant (nothing fits inside the original
                        #      face's in-plane footprint).
                        owner_role = str(
                            (config.get("body_roles") or {}).get(owner, "")
                        ).strip().lower()
                        owner_removed = owner_role in (
                            "pec_solid",
                            "conductor_solid",
                            "surface_pec",
                            "surface_conductor",
                            "exclude",
                        )

                        half_extent = (
                            1.05 * characteristic_radius
                            + IMPEDANCE_CENTER_TOL_MM
                        )
                        plausible_descendants = []
                        for item in split_candidates:
                            axis = item["axis"]
                            in_plane_axes = [
                                i for i in range(3) if i != axis
                            ]
                            candidate_box = bbox(2, item["surface"])
                            fits_inside_footprint = all(
                                candidate_box[i]
                                >= target_center[i] - half_extent
                                and candidate_box[i + 3]
                                <= target_center[i] + half_extent
                                for i in in_plane_axes
                            )
                            if fits_inside_footprint:
                                plausible_descendants.append(item)

                        if owner_removed and not plausible_descendants:
                            print(
                                f"  NOTE: impedance boundary "
                                f"'{boundary_name}' HFSS face "
                                f"{face_record.get('face_id')} lies on "
                                f"removed solid '{owner}' flush with the "
                                f"exterior of the solved domain (no final "
                                f"surface exists there; no solved "
                                f"dielectric touches it, so it carried no "
                                f"field in HFSS either). Skipping this "
                                f"face -- the boundary is physically "
                                f"inert here."
                            )
                            continue

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

        if not boundary_matches:
            print(
                f"  NOTE: impedance boundary '{boundary_name}' matched "
                f"NO final surfaces (every assigned face was buried in "
                f"the domain exterior). It will get no physical group "
                f"and no Impedance block in the Palace config. If this "
                f"boundary was meant to model loss, reassign it in HFSS "
                f"to a face the vacuum actually touches."
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
        deferred_faces = []
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

                    if not plausible:
                        print(
                            f"  original conductor surface "
                            f"{face_record['surface']} has no exposed "
                            "descendant in a solved dielectric; skipping"
                        )
                        continue

                    # DEFERRED DECISION. Plausible candidates exist but
                    # nothing matched cleanly. A sibling face of this
                    # same body may still claim one of them as its
                    # exact match (a cylinder's end cap lies inside the
                    # lateral wall's bounding box, for example), so
                    # postpone judgement until every face of this body
                    # has had its exact/split pass.
                    deferred_faces.append(face_record)
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

        # Second pass over faces whose first-pass match was ambiguous.
        # By now every sibling face has claimed its surfaces, so what
        # remains inside a deferred face's bounding box is either its
        # own clipped remnant (accept), several unexplained surfaces
        # (refuse), or nothing (skip -- the face vanished into a wall).
        for face_record in deferred_faces:
            target_box = face_record["bbox"]
            target_center = face_record["center"]
            target_area = float(face_record["area"])
            area_tolerance = max(
                CONDUCTOR_AREA_ABS_TOL_MM2,
                CONDUCTOR_AREA_REL_TOL * abs(target_area),
            )

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

            if not plausible:
                print(
                    f"  original conductor surface "
                    f"{face_record['surface']}: all nearby final "
                    f"surfaces were claimed by sibling faces; no "
                    f"exposed descendant remains, skipping"
                )
                continue

            if len(plausible) > 1:
                raise RuntimeError(
                    f"Could not uniquely recover conductor "
                    f"'{body_name}' original surface "
                    f"{face_record['surface']}; plausible final "
                    f"surfaces were "
                    f"{[item['surface'] for item in plausible]}"
                )

            # SINGLE-CLIPPED-FACE CASE. The combinatorial split search
            # needs >= 2 descendants, so an original face clipped into
            # exactly ONE smaller exposed piece (e.g. a conductor face
            # partly pressed against a wall or another solid) can only
            # be recovered here. Accept the lone candidate when it is
            # geometrically credible: inside the original face's
            # bounding box (already enforced), smaller than the
            # original face, and retaining a substantial fraction of
            # its area -- the same criteria the impedance matcher
            # applies to clipped faces.
            candidate = plausible[0]
            candidate_area = float(candidate["area"])
            retained_fraction = candidate_area / max(
                target_area, CONDUCTOR_AREA_ABS_TOL_MM2
            )
            if not (
                CONDUCTOR_AREA_ABS_TOL_MM2
                < candidate_area
                < target_area + area_tolerance
                and retained_fraction >= 0.25
            ):
                raise RuntimeError(
                    f"Conductor '{body_name}' original surface "
                    f"{face_record['surface']} (area "
                    f"{target_area:.12g} mm^2, center "
                    f"{target_center}) has exactly one plausible "
                    f"descendant, surface {candidate['surface']} "
                    f"(area {candidate_area:.12g} mm^2, "
                    f"{100.0 * retained_fraction:.6g}% of the "
                    f"original), but it does not look like a clipped "
                    f"remnant of that face (needs 25%-100% of the "
                    f"original area). Refusing to guess."
                )

            print(
                f"  original conductor surface "
                f"{face_record['surface']} was clipped by a boolean; "
                f"using final exposed surface {candidate['surface']} "
                f"with area={candidate_area:.12g} mm^2 "
                f"({100.0 * retained_fraction:.6g}% of the original "
                f"{target_area:.12g} mm^2)"
            )

            if candidate["surface"] in excluded_surfaces:
                raise RuntimeError(
                    f"Conductor surface {candidate['surface']} for "
                    f"'{body_name}' is already reserved for another "
                    "boundary"
                )

            body_matches.add(candidate["surface"])
            already_claimed.add(candidate["surface"])
            print(
                f"  original surface={face_record['surface']} "
                f"-> final surface={candidate['surface']}, "
                f"center={candidate['center']}, "
                f"area={candidate['area']:.12g} mm^2, "
                f"adjacent_volumes={candidate['adjacent']}"
            )

        if not body_matches:
            raise RuntimeError(
                f"Surface conductor '{body_name}' produced no exposed "
                "finite-conductivity boundary surfaces"
            )

        matched_by_object[body_name] = body_matches

    return matched_by_object



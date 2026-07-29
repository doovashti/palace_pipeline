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
import gmsh

STEP_FILE = "no_non_model.step"

OUT_MSH = "non_model_w_bond_wire_cf.msh"
OUT_VTK = "non_model_w_bond_wire_cf.vtk"
OUT_GROUPS = "physical_groups_w_bond_wire_cf.json"

ENABLE_CHIP = True
ENABLE_PIN = True
ENABLE_BONDWIRE = True
ENABLE_PADS = True
ENABLE_MEDIUM_LEAD = True
ENABLE_THIN_LEAD = True
ENABLE_JJ = True

SIZE_CAVITY = 1.0
SIZE_PADS = 0.05
SIZE_MEDIUM = 0.003
SIZE_THIN = 0.0002
SIZE_JJ = 0.0002


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


def classify_volume(tag: int) -> str:
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


def classify_sheet(tag: int) -> str:
    """
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


def main():
    gmsh.initialize()

    try:
        gmsh.model.add("palace_device")
        occ = gmsh.model.occ

        print(f"\nImporting combined STEP file: {STEP_FILE}")
        print("Importing all dimensions, including standalone 2-D sheets")
        occ.importShapes(STEP_FILE, highestDimOnly=False)
        occ.synchronize()

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
                    f"  tag={tag:4d} dims={dims} bbox={box}"
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
                f"Unassigned final volume fragments: {sorted(unassigned_volumes)}"
            )

        final_sheets_by_name = {
            "pads": set(),
            "medium_lead": set(),
            "thin_lead": set(),
            "JJ": set(),
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

        pec_exterior_surfaces = set(one_sided_contact_surfaces)
        for _, surface in gmsh.model.getEntities(2):
            if surface in all_enabled_sheet_surfaces:
                continue
            upward, _ = gmsh.model.getAdjacencies(2, surface)
            if len(upward) == 1:
                pec_exterior_surfaces.add(surface)
            elif len(upward) > 2:
                raise RuntimeError(
                    f"Non-manifold surface {surface} has {len(upward)} adjacent volumes"
                )

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

        for name in ("pads", "medium_lead", "thin_lead", "JJ"):
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
        for name in ("pads", "medium_lead", "thin_lead", "JJ"):
            surfaces = sorted(final_sheets_by_name[name])
            if surfaces:
                print(f"  {name:14s}: {count_2d_elements(surfaces)} triangles")

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

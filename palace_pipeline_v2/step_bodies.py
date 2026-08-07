"""
step_bodies.py

Reads named bodies directly from an Ansys-exported STEP file: name,
geometry type, material, and bounding box.

Why this exists
---------------
Gmsh collapses the AEDT labels to "Shapes/COMPOUND" on import, so
gmsh.model.getEntityName() cannot tell you which entity is the cavity and
which is the chip. The names are in the STEP file, though, so we read them
here and match them onto Gmsh entities by bounding box.

Bounding boxes
--------------
An earlier version of this parser collected only VERTEX_POINT entities.
That is wrong for any body with a circular face: a cylinder has no
vertices around its circles, only a seam line. It gave, for example:

    pin     (1.5, 0, 0) -> (1.5, 0, 17)     -- a line, not a cylinder
    cavity  (0, 0, 0)   -> (4, 22.5, 35)    -- half the true width

This version also reads CIRCLE geometry (radius + centre + axis),
ELLIPSE geometry, and B-spline control points. The bounding box of a
circle with centre c, radius r and unit normal n is
c_i +/- r*sqrt(1 - n_i^2) along each axis i. For an ellipse with unit
major/minor directions u, v and semi-axes r1, r2 it is
c_i +/- sqrt((r1*u_i)^2 + (r2*v_i)^2).

Verified against Gmsh's own bounding boxes for all five solids in
test_export_for_gmsh.step: cavity, pin, chip, Bondwire1, Plot_Fields3D.

KNOWN LIMITATION (documented, not fixed): a partial arc contributes its
FULL circle's bounding box, which is conservative (too large). For bodies
whose faces are full circles or straight edges -- everything in the
current device -- the boxes are exact. A future geometry with large
fillet arcs may need trim-parameter handling here, and will show up as a
match_gmsh_entities() failure rather than a silent misassignment.

Units: millimetres (whatever the STEP file uses -- Gmsh reads the same
file, so the two sides are consistent by construction).

REVISION NOTES (this version):
  * _numbers() strips quoted strings before extracting numerics.
    Previously a CARTESIAN_POINT or CIRCLE with a non-empty AEDT label
    containing digits (e.g. 'P1') leaked the label's digits into the
    coordinate/radius list and silently corrupted the bounding box.
    AEDT writes '' labels today, which is the only reason this never
    fired.
  * ELLIPSE entities contribute exact extremes instead of being
    silently ignored.
  * Duplicate body names in one STEP file are now a hard error instead
    of a silent dict overwrite (the second body used to replace the
    first, and the mesher then failed later with a confusing message
    or, worse, meshed the wrong body).
  * match_gmsh_entities() detects ambiguous matches: if the best and
    second-best candidates carry DIFFERENT body names and their scores
    are within 0.25 * tolerance of each other, it raises instead of
    guessing. (A clear win by margin -- e.g. a JJ surface whose
    neighbouring thin-lead face is also inside the tolerance but 10x
    further -- still matches normally.)
"""

from __future__ import annotations

import math
import re
from collections import defaultdict

# Topology entities we walk through to reach geometry.
# Deliberately excludes surface definitions (PLANE, CYLINDRICAL_SURFACE):
# those are unbounded, and their placement points can sit outside the
# actual face. Only the bounding edges define a face's true extent.
_TOPOLOGY_TYPES = {
    "MANIFOLD_SOLID_BREP",
    "SHELL_BASED_SURFACE_MODEL",
    "OPEN_SHELL",
    "CLOSED_SHELL",
    "ADVANCED_FACE",
    "FACE_BOUND",
    "FACE_OUTER_BOUND",
    "EDGE_LOOP",
    "ORIENTED_EDGE",
    "EDGE_CURVE",
}

_BODY_TYPES = {
    "MANIFOLD_SOLID_BREP": "solid",
    "SHELL_BASED_SURFACE_MODEL": "sheet",
}

_ENTITY_PATTERN = re.compile(
    r"#(\d+)\s*=\s*([A-Z_0-9]+)\s*\((.*?)\)\s*;", re.DOTALL
)


def _parse_entities(path):
    with open(path, errors="replace") as handle:
        text = handle.read()
    entities = {}
    for match in _ENTITY_PATTERN.finditer(text):
        entities[int(match.group(1))] = (match.group(2), match.group(3))
    return entities


def _refs(args):
    return [int(x) for x in re.findall(r"#(\d+)", args)]


def _numbers(args):
    # Strip quoted strings first: an AEDT label like 'P1' must not leak
    # its digits into the numeric list. Then strip #123 references so
    # they are not read as numbers either.
    cleaned = re.sub(r"'[^']*'", " ", args)
    cleaned = re.sub(r"#\d+", "", cleaned)
    return [
        float(x)
        for x in re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", cleaned)
    ]


def _cartesian_point(entities, entity_id):
    entry = entities.get(entity_id)
    if not entry or entry[0] != "CARTESIAN_POINT":
        return None
    values = _numbers(entry[1])
    return tuple(values[:3]) if len(values) >= 3 else None


def _direction(entities, entity_id):
    entry = entities.get(entity_id)
    if not entry or entry[0] != "DIRECTION":
        return None
    values = _numbers(entry[1])
    return tuple(values[:3]) if len(values) >= 3 else None


def _unit(vector):
    length = math.sqrt(sum(component * component for component in vector))
    if length == 0.0:
        return None
    return tuple(component / length for component in vector)


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _placement_frame(entities, placement_id):
    """
    Centre, axis (normal), and reference (major) direction of an
    AXIS2_PLACEMENT_3D. Missing directions fall back to +z / +x.
    """
    placement = entities.get(placement_id)
    if not placement or placement[0] != "AXIS2_PLACEMENT_3D":
        return None, None, None

    placement_refs = _refs(placement[1])
    if not placement_refs:
        return None, None, None

    centre = _cartesian_point(entities, placement_refs[0])
    axis = None
    ref_direction = None
    if len(placement_refs) > 1:
        axis = _direction(entities, placement_refs[1])
    if len(placement_refs) > 2:
        ref_direction = _direction(entities, placement_refs[2])

    axis = _unit(axis) if axis else None
    if axis is None:
        axis = (0.0, 0.0, 1.0)
    ref_direction = _unit(ref_direction) if ref_direction else None

    return centre, axis, ref_direction


def _circle_extremes(entities, args):
    """
    Corner points of a circle's bounding box.

    STEP: #38=CIRCLE('',#37,4.) where #37 is an AXIS2_PLACEMENT_3D giving
    the centre and the axis direction, and 4. is the radius.
    """
    references = _refs(args)
    values = _numbers(args)
    if not references or not values:
        return []

    radius = values[-1]
    centre, axis, _ = _placement_frame(entities, references[0])
    if centre is None:
        return []

    extents = [
        radius * math.sqrt(max(0.0, 1.0 - axis[i] ** 2)) for i in range(3)
    ]
    return [
        tuple(centre[i] - extents[i] for i in range(3)),
        tuple(centre[i] + extents[i] for i in range(3)),
    ]


def _ellipse_extremes(entities, args):
    """
    Corner points of an ellipse's bounding box.

    STEP: #38=ELLIPSE('',#37,r1,r2) where #37 is an AXIS2_PLACEMENT_3D
    (centre, normal axis, major-axis reference direction) and r1/r2 are
    the semi-major and semi-minor axes. Extent along coordinate axis i is
    sqrt((r1*u_i)^2 + (r2*v_i)^2) with u the unit major direction and
    v = axis x u.
    """
    references = _refs(args)
    values = _numbers(args)
    if not references or len(values) < 2:
        return []

    r1, r2 = values[-2], values[-1]
    centre, axis, major = _placement_frame(entities, references[0])
    if centre is None:
        return []

    if major is None:
        # Without a recorded major direction, fall back to a bounding
        # circle of the larger semi-axis: conservative but safe.
        radius = max(abs(r1), abs(r2))
        extents = [
            radius * math.sqrt(max(0.0, 1.0 - axis[i] ** 2))
            for i in range(3)
        ]
    else:
        minor = _unit(_cross(axis, major)) or (0.0, 0.0, 0.0)
        extents = [
            math.sqrt((r1 * major[i]) ** 2 + (r2 * minor[i]) ** 2)
            for i in range(3)
        ]

    return [
        tuple(centre[i] - extents[i] for i in range(3)),
        tuple(centre[i] + extents[i] for i in range(3)),
    ]


def _body_bbox(entities, start_id):
    points = []
    visited = set()
    stack = [start_id]

    while stack:
        current = stack.pop()
        if current in visited or current not in entities:
            continue
        visited.add(current)
        etype, args = entities[current]

        if etype == "VERTEX_POINT":
            references = _refs(args)
            if references:
                point = _cartesian_point(entities, references[0])
                if point:
                    points.append(point)

        elif etype == "CIRCLE":
            points.extend(_circle_extremes(entities, args))

        elif etype == "ELLIPSE":
            points.extend(_ellipse_extremes(entities, args))

        elif etype.startswith("B_SPLINE_CURVE"):
            # A B-spline lies inside the convex hull of its control
            # points, so this is correct but slightly conservative.
            for reference in _refs(args):
                point = _cartesian_point(entities, reference)
                if point:
                    points.append(point)

        elif etype in _TOPOLOGY_TYPES:
            stack.extend(_refs(args))

    if not points:
        return None

    xs, ys, zs = zip(*points)
    return (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))


_SHELL_TYPES = {"OPEN_SHELL", "CLOSED_SHELL"}


def _faces_of(entities, body_id):
    """
    The ADVANCED_FACE children of a body, reached via its shell(s).

    This matters because Gmsh imports each face of a sheet body as its
    own dim-2 entity. The "pads" body, for example, is one
    SHELL_BASED_SURFACE_MODEL containing two faces, and Gmsh gives two
    surfaces. Matching against the body's union bounding box would fail;
    matching per face works.
    """
    faces = []
    visited = set()
    stack = [body_id]
    while stack:
        current = stack.pop()
        if current in visited or current not in entities:
            continue
        visited.add(current)
        etype, args = entities[current]
        if etype == "ADVANCED_FACE":
            faces.append(current)
        elif etype in _SHELL_TYPES or etype in _BODY_TYPES:
            stack.extend(_refs(args))
    return faces


def _material_by_shape(entities):
    """
    Walk the AEDT PMI property chain to find each body's material.

    DESCRIPTIVE_REPRESENTATION_ITEM -> REPRESENTATION
      -> PROPERTY_DEFINITION_REPRESENTATION -> PROPERTY_DEFINITION
      -> SHAPE_ASPECT -> GEOMETRIC_ITEM_SPECIFIC_USAGE -> the shape
    """
    by_type = defaultdict(list)
    for entity_id, (etype, _) in entities.items():
        by_type[etype].append(entity_id)

    descriptive = {}
    for entity_id in by_type.get("DESCRIPTIVE_REPRESENTATION_ITEM", []):
        match = re.match(r"'([^']*)'\s*,\s*'([^']*)'", entities[entity_id][1])
        if match:
            descriptive[entity_id] = (match.group(1), match.group(2))

    representation_of = {}
    for entity_id in by_type.get("REPRESENTATION", []):
        for reference in _refs(entities[entity_id][1]):
            if reference in descriptive:
                representation_of[reference] = entity_id

    property_of_representation = {}
    for entity_id in by_type.get("PROPERTY_DEFINITION_REPRESENTATION", []):
        references = _refs(entities[entity_id][1])
        if len(references) >= 2:
            property_of_representation[references[1]] = references[0]

    aspect_of_property = {}
    for entity_id in by_type.get("PROPERTY_DEFINITION", []):
        references = _refs(entities[entity_id][1])
        if references:
            aspect_of_property[entity_id] = references[-1]

    shape_of_aspect = {}
    for entity_id in by_type.get("GEOMETRIC_ITEM_SPECIFIC_USAGE", []):
        references = _refs(entities[entity_id][1])
        if len(references) >= 2:
            shape_of_aspect[references[0]] = references[-1]

    materials = {}
    for item_id, (label, value) in descriptive.items():
        if label != "AEDT_MaterialName_V1":
            continue
        representation = representation_of.get(item_id)
        prop = property_of_representation.get(representation)
        aspect = aspect_of_property.get(prop)
        shape = shape_of_aspect.get(aspect)
        if shape is not None:
            materials[shape] = value
    return materials


def read_step_bodies(path):
    """
    Return {name: {"geo_type", "material", "bbox", "size"}} for every
    named body in the STEP file.

    geo_type is "solid" or "sheet".
    bbox is (xmin, ymin, zmin, xmax, ymax, zmax).
    size is (dx, dy, dz).

    Raises on duplicate body names: two same-named bodies cannot be
    told apart downstream, and a silent overwrite meshes the wrong one.
    """
    entities = _parse_entities(path)
    materials = _material_by_shape(entities)

    bodies = {}
    for entity_id, (etype, args) in entities.items():
        geo_type = _BODY_TYPES.get(etype)
        if geo_type is None:
            continue

        match = re.match(r"'([^']*)'", args)
        if not match or not match.group(1):
            continue
        name = match.group(1)

        bbox = _body_bbox(entities, entity_id)
        if bbox is None:
            continue

        if name in bodies:
            raise ValueError(
                f"STEP file contains two bodies named '{name}' "
                f"(entities #{bodies[name]['entity_id']} and "
                f"#{entity_id}). Rename one in Ansys; same-named bodies "
                f"cannot be matched unambiguously."
            )

        face_boxes = []
        for face_id in _faces_of(entities, entity_id):
            face_bbox = _body_bbox(entities, face_id)
            if face_bbox is not None:
                face_boxes.append(face_bbox)

        bodies[name] = {
            "entity_id": entity_id,
            "geo_type": geo_type,
            "material": materials.get(entity_id, "unknown"),
            "bbox": bbox,
            "size": (
                bbox[3] - bbox[0],
                bbox[4] - bbox[1],
                bbox[5] - bbox[2],
            ),
            # One bbox per face. Gmsh imports sheet faces individually,
            # so these are what a Gmsh dim-2 entity should be matched to.
            "face_bboxes": face_boxes,
        }
    return bodies


def match_gmsh_entities(gmsh_module, bodies, dim, tolerance=0.01):
    """
    Map Gmsh entity tags to STEP body names by bounding box.

    dim == 3: each Gmsh volume is matched to a body's overall bbox.
    dim == 2: each Gmsh surface is matched to a body's individual face
              bbox, since Gmsh splits sheet bodies into separate faces.

    Returns (tag_to_name, unmatched_tags).

    Ambiguity guard: when the best and second-best candidates carry
    DIFFERENT body names, both lie within the tolerance, and their
    scores are within 0.25 * tolerance of each other, the match is
    genuinely ambiguous and a RuntimeError is raised instead of a
    silent guess. A clear winner (best score much smaller than the
    runner-up, as for adjacent JJ / thin-lead faces) matches normally.

    Requires the caller to pass the gmsh module, so this file stays
    importable and testable without Gmsh installed.
    """
    wanted = "solid" if dim == 3 else "sheet"

    targets = []  # (name, bbox)
    for body_name, info in bodies.items():
        if info["geo_type"] != wanted:
            continue
        if dim == 2 and info["face_bboxes"]:
            for face_bbox in info["face_bboxes"]:
                targets.append((body_name, face_bbox))
        else:
            targets.append((body_name, info["bbox"]))

    tag_to_name = {}
    unmatched = []

    for _, tag in gmsh_module.model.getEntities(dim):
        gmsh_bbox = gmsh_module.model.getBoundingBox(dim, tag)

        best_name = None
        best_score = float("inf")
        second_name = None
        second_score = float("inf")
        for body_name, target_bbox in targets:
            score = max(
                abs(gmsh_bbox[i] - target_bbox[i]) for i in range(6)
            )
            if score < best_score:
                if body_name != best_name:
                    second_name, second_score = best_name, best_score
                best_name, best_score = body_name, score
            elif body_name != best_name and score < second_score:
                second_name, second_score = body_name, score

        if best_name is not None and best_score <= tolerance:
            if (
                second_name is not None
                and second_score <= tolerance
                and (second_score - best_score) < 0.25 * tolerance
            ):
                raise RuntimeError(
                    f"Ambiguous bbox match for Gmsh dim-{dim} entity "
                    f"{tag}: '{best_name}' (score {best_score:.4g} mm) "
                    f"vs '{second_name}' (score {second_score:.4g} mm) "
                    f"are both within tolerance {tolerance} mm and too "
                    f"close to call. Tighten the tolerance or rename/"
                    f"separate the bodies."
                )
            tag_to_name[tag] = best_name
        else:
            unmatched.append((tag, gmsh_bbox, best_name, best_score))

    return tag_to_name, unmatched


if __name__ == "__main__":
    import sys

    step_path = sys.argv[1] if len(sys.argv) > 1 else "device.step"
    found = read_step_bodies(step_path)

    print(f"reading: {step_path}\n")
    print(f"{'name':<16}{'type':<8}{'material':<20}{'size (mm)'}")
    print("-" * 72)
    for body_name, info in sorted(found.items()):
        dx, dy, dz = info["size"]
        print(
            f"{body_name:<16}{info['geo_type']:<8}{info['material']:<20}"
            f"{dx:.4g} x {dy:.4g} x {dz:.4g}"
        )
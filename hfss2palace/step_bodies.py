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

PARTIAL ARCS ARE NOW TRIMMED: an EDGE_CURVE whose underlying curve is a
CIRCLE or an ELLIPSE contributes only its true arc's bounding box --
the two endpoint vertices plus any axis-aligned extreme of the curve
that lies ON the arc (the arc's angular interval follows the edge's
.T./.F. sense flag). A closed curve (same start and end vertex) still
contributes the full curve. This is what makes curved couplers/fillets
(Sophi_coupler) match: the old full-curve boxes overshot the true face
by up to the fillet radius and match_gmsh_entities() refused them.

B-SPLINES ARE NOW SAMPLED: a B_SPLINE_CURVE_WITH_KNOTS is evaluated at
64 parameter values by De Boor's algorithm, so its bbox is the box of
points ON the curve (sampling error ~ r*(dtheta)^2/8, far below the
matching tolerance) instead of the convex hull of its control points,
which for a curved coupler trace overshoots by a large fraction of the
local radius. Splines whose knot data cannot be parsed (e.g. a plain
B_SPLINE_CURVE with no knot list) fall back to the control-point hull
-- conservative, too large, never too small.

Units: millimetres (whatever the STEP file uses -- Gmsh reads the same
file, so the two sides are consistent by construction).

REVISION NOTES (this version):
  * SAME-NAMED BODIES ARE MERGED, not rejected. AEDT's
    DuplicateAlongLine (and multi-lump united objects) legitimately
    export several STEP solids carrying one object name -- e.g. a row
    of ground bondwires. Those lumps share material, physics role,
    boundary membership, and mesh size by construction, so they are
    ONE logical body with several lumps:
        bodies[name]["entity_ids"]  -> all STEP entity ids
        bodies[name]["lump_bboxes"] -> one bbox per lump
        bodies[name]["bbox"]        -> union of the lumps
        bodies[name]["face_bboxes"] -> faces of ALL lumps
    match_gmsh_entities(dim=3) matches each Gmsh volume against the
    per-lump boxes, so several Gmsh volumes may map to the same name
    (the mesher must accept lists of volumes per name -- mesh_any.py
    does). A solid and a sheet sharing one name is still a hard error:
    that IS ambiguous.
  * _numbers() strips quoted strings before extracting numerics.
    Previously a CARTESIAN_POINT or CIRCLE with a non-empty AEDT label
    containing digits (e.g. 'P1') leaked the label's digits into the
    coordinate/radius list and silently corrupted the bounding box.
    AEDT writes '' labels today, which is the only reason this never
    fired.
  * ELLIPSE entities contribute exact extremes instead of being
    silently ignored.
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


def _vertex_coords(entities, vertex_id):
    entry = entities.get(vertex_id)
    if not entry or entry[0] != "VERTEX_POINT":
        return None
    references = _refs(entry[1])
    return _cartesian_point(entities, references[0]) if references else None


def _conic_frame(entities, curve):
    """
    Parametrisation p(theta) = c + r1*cos(theta)*u + r2*sin(theta)*v of
    a CIRCLE or ELLIPSE entity: returns (centre, axis, u_major, r1, r2)
    or None if the curve is neither, or its placement is unreadable.
    For a circle r1 == r2 and u_major may be None (any in-plane
    direction serves; the caller builds one from an arc endpoint).
    """
    etype, cargs = curve
    references = _refs(cargs)
    values = _numbers(cargs)
    if not references:
        return None
    if etype == "CIRCLE":
        if not values:
            return None
        r1 = r2 = values[-1]
    elif etype == "ELLIPSE":
        if len(values) < 2:
            return None
        r1, r2 = values[-2], values[-1]
    else:
        return None
    centre, axis, major = _placement_frame(entities, references[0])
    if centre is None or r1 <= 0.0 or r2 <= 0.0:
        return None
    return centre, axis, major, r1, r2


def _full_curve_extremes(entities, curve):
    if curve[0] == "CIRCLE":
        return _circle_extremes(entities, curve[1])
    if curve[0] == "ELLIPSE":
        return _ellipse_extremes(entities, curve[1])
    return []


def _arc_points(entities, edge_args):
    """
    Bounding points of an EDGE_CURVE whose curve is a CIRCLE or an
    ELLIPSE: the two endpoint vertices plus every axis-aligned extreme
    of the curve that lies ON the arc between them.

    STEP: EDGE_CURVE('',#v1,#v2,#curve,.T.) -- with sense .T. the edge
    runs from v1 to v2 in the curve's own (counterclockwise about the
    placement axis) parameter direction; .F. runs against it. Either
    way the POINT SET is one specific arc, and that is all its bbox
    may contain -- the old code added the full curve's extremes,
    overshooting curved faces by up to the radius and breaking bbox
    matching for filleted sheets.

    Returns None when the edge's curve is not a circle/ellipse (caller
    falls back to the generic traversal). A closed curve (v1 == v2)
    returns the full curve's extremes.
    """
    references = _refs(edge_args)
    if len(references) < 3:
        return None
    v1, v2, curve_id = references[0], references[1], references[2]
    curve = entities.get(curve_id)
    if not curve:
        return None
    frame = _conic_frame(entities, curve)
    if frame is None:
        return None
    centre, axis, major, r1, r2 = frame

    p1 = _vertex_coords(entities, v1)
    p2 = _vertex_coords(entities, v2)
    if p1 is None or p2 is None:
        return None
    points = [p1, p2]

    if v1 == v2:
        return points + _full_curve_extremes(entities, curve)

    # In-plane frame (u, v) with u perpendicular to the axis. An
    # ellipse NEEDS its recorded major direction (theta is measured
    # from it); a circle can take any in-plane direction, so fall back
    # to the first endpoint's radial direction.
    u = major
    if u is None:
        if curve[0] == "ELLIPSE":
            return points + _full_curve_extremes(entities, curve)
        u = _unit(tuple(p1[i] - centre[i] for i in range(3)))
    if u is not None:
        along = sum(u[i] * axis[i] for i in range(3))
        u = _unit(tuple(u[i] - along * axis[i] for i in range(3)))
    if u is None:
        return points + _full_curve_extremes(entities, curve)
    v = _cross(axis, u)

    def angle_of(p):
        d = tuple(p[i] - centre[i] for i in range(3))
        return math.atan2(sum(d[i] * v[i] for i in range(3)) / r2,
                          sum(d[i] * u[i] for i in range(3)) / r1)

    sense = edge_args.rstrip().rstrip(");").rstrip().endswith(".T.")
    t1, t2 = angle_of(p1), angle_of(p2)
    start, end = (t1, t2) if sense else (t2, t1)
    span = (end - start) % (2.0 * math.pi)

    two_pi = 2.0 * math.pi
    for i in range(3):
        # d/dtheta [r1*cos(theta)*u_i + r2*sin(theta)*v_i] = 0  at
        # tan(theta) = r2*v_i / (r1*u_i); for a circle this reduces to
        # the old atan2(v_i, u_i).
        base = math.atan2(r2 * v[i], r1 * u[i])
        for theta in (base, base + math.pi):
            if ((theta - start) % two_pi) <= span:
                points.append(tuple(
                    centre[j] + r1 * math.cos(theta) * u[j]
                    + r2 * math.sin(theta) * v[j]
                    for j in range(3)))
    return points


def _de_boor(t, degree, knots, ctrl):
    """One point on a B-spline by De Boor's algorithm."""
    n = len(ctrl)
    k = n - 1
    for i in range(degree, n):
        if t < knots[i + 1]:
            k = i
            break
    d = [list(ctrl[j + k - degree]) for j in range(degree + 1)]
    for r in range(1, degree + 1):
        for j in range(degree, r - 1, -1):
            denom = knots[j + 1 + k - r] - knots[j + k - degree]
            alpha = 0.0 if denom == 0.0 else (
                (t - knots[j + k - degree]) / denom)
            d[j] = [(1.0 - alpha) * d[j - 1][i] + alpha * d[j][i]
                    for i in range(3)]
    return tuple(d[degree])


def _bspline_points(entities, args, samples=64):
    """
    Points ON a B_SPLINE_CURVE_WITH_KNOTS, sampled at `samples`
    parameter values by De Boor evaluation.

    STEP: B_SPLINE_CURVE_WITH_KNOTS('',3,(#p1,...,#pn),.UNSPECIFIED.,
    .F.,.F.,(mult_1,...),(knot_1,...),.UNSPECIFIED.) -- degree, control
    points, knot multiplicities, distinct knots. The full knot vector
    repeats each knot by its multiplicity; the curve lives on
    [knots[degree], knots[n]].

    Returns None when the entity cannot be parsed as a valid non-
    rational spline (caller falls back to the control-point hull,
    which is conservative but never too small). Rational splines
    appear only inside STEP complex entities, which the file-level
    parser skips entirely, so they never reach here.
    """
    header = re.match(r"\s*'[^']*'\s*,\s*(\d+)", args)
    if not header:
        return None
    degree = int(header.group(1))

    ctrl = [_cartesian_point(entities, r) for r in _refs(args)]
    ctrl = [p for p in ctrl if p is not None]
    n = len(ctrl)
    if n < degree + 1:
        return None

    numeric_groups = []
    for group in re.findall(r"\(([^()]*)\)", args):
        if "#" in group:
            continue
        numbers = re.findall(
            r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", group)
        if numbers:
            numeric_groups.append([float(x) for x in numbers])
    if len(numeric_groups) < 2:
        return None
    multiplicities, distinct_knots = numeric_groups[0], numeric_groups[1]
    if len(multiplicities) != len(distinct_knots):
        return None

    knots = []
    for multiplicity, knot in zip(multiplicities, distinct_knots):
        knots.extend([knot] * int(round(multiplicity)))
    if len(knots) != n + degree + 1:
        return None

    lo, hi = knots[degree], knots[n]
    if not hi > lo:
        return None
    return [
        _de_boor(lo + (hi - lo) * i / (samples - 1), degree, knots, ctrl)
        for i in range(samples)
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

        if etype == "EDGE_CURVE":
            arc_points = _arc_points(entities, args)
            if arc_points is not None:
                points.extend(arc_points)
                continue
            stack.extend(_refs(args))

        elif etype == "VERTEX_POINT":
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
            # Sample points ON the curve (De Boor). Fallback: the
            # convex hull of the control points -- conservative,
            # too large, never too small.
            sampled = _bspline_points(entities, args)
            if sampled:
                points.extend(sampled)
            else:
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
    Return {name: {"geo_type", "material", "bbox", "size", "entity_ids",
    "lump_bboxes", "face_bboxes"}} for every named body in the STEP
    file.

    geo_type is "solid" or "sheet".
    bbox is (xmin, ymin, zmin, xmax, ymax, zmax) -- the UNION over all
    lumps carrying this name.
    size is (dx, dy, dz) of that union box.
    lump_bboxes is one bbox per same-named STEP body: one entry for a
    normal body; several for DuplicateAlongLine children / multi-lump
    united objects, which AEDT exports as separate solids sharing the
    object name.

    Same-named lumps are MERGED into one logical body: they share
    material, physics role, boundary membership, and mesh size by
    construction, so there is nothing ambiguous about them. The only
    hard error is a solid and a sheet sharing one name -- that cannot
    be classified.
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

        face_boxes = []
        for face_id in _faces_of(entities, entity_id):
            face_bbox = _body_bbox(entities, face_id)
            if face_bbox is not None:
                face_boxes.append(face_bbox)

        if name in bodies:
            existing = bodies[name]
            if existing["geo_type"] != geo_type:
                raise ValueError(
                    f"STEP file contains a solid AND a sheet both named "
                    f"'{name}' (entities #{existing['entity_id']} and "
                    f"#{entity_id}). That is genuinely ambiguous -- "
                    f"rename one in Ansys."
                )
            material = materials.get(entity_id, "unknown")
            if (material != "unknown"
                    and existing["material"] != "unknown"
                    and material != existing["material"]):
                print(
                    f"  WARNING: same-named lumps of '{name}' carry "
                    f"different materials ('{existing['material']}' vs "
                    f"'{material}'); keeping '{existing['material']}'"
                )
            existing["entity_ids"].append(entity_id)
            existing["lump_bboxes"].append(bbox)
            merged = existing["bbox"]
            existing["bbox"] = (
                min(merged[0], bbox[0]),
                min(merged[1], bbox[1]),
                min(merged[2], bbox[2]),
                max(merged[3], bbox[3]),
                max(merged[4], bbox[4]),
                max(merged[5], bbox[5]),
            )
            union = existing["bbox"]
            existing["size"] = (
                union[3] - union[0],
                union[4] - union[1],
                union[5] - union[2],
            )
            existing["face_bboxes"].extend(face_boxes)
            continue

        bodies[name] = {
            "entity_id": entity_id,          # first lump (back-compat)
            "entity_ids": [entity_id],       # all lumps
            "geo_type": geo_type,
            "material": materials.get(entity_id, "unknown"),
            "bbox": bbox,
            "size": (
                bbox[3] - bbox[0],
                bbox[4] - bbox[1],
                bbox[5] - bbox[2],
            ),
            "lump_bboxes": [bbox],
            # One bbox per face (across all lumps). Gmsh imports sheet
            # faces individually, so these are what a Gmsh dim-2 entity
            # should be matched to.
            "face_bboxes": face_boxes,
        }
    return bodies


def match_gmsh_entities(gmsh_module, bodies, dim, tolerance=0.01):
    """
    Map Gmsh entity tags to STEP body names by bounding box.

    dim == 3: each Gmsh volume is matched to a body's per-LUMP bboxes.
              A multi-lump body (a DuplicateAlongLine bondwire row)
              legitimately matches SEVERAL Gmsh volumes to one name;
              callers must accept lists of volumes per name.
    dim == 2: each Gmsh surface is matched to a body's individual face
              bbox, since Gmsh splits sheet bodies into separate faces.

    Returns (tag_to_name, unmatched_tags).

    Ambiguity guard: when the best and second-best candidates carry
    DIFFERENT body names, both lie within the tolerance, and their
    scores are within 0.25 * tolerance of each other, the match is
    genuinely ambiguous and a RuntimeError is raised instead of a
    silent guess. A clear winner (best score much smaller than the
    runner-up, as for adjacent JJ / thin-lead faces) matches normally.
    Two lumps of the SAME body never trigger the guard -- whichever
    lump wins, the name is identical.

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
        elif dim == 3:
            for lump_bbox in (info.get("lump_bboxes") or [info["bbox"]]):
                targets.append((body_name, lump_bbox))
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
    print(f"{'name':<16}{'type':<8}{'material':<20}{'lumps':<7}"
          f"{'size (mm)'}")
    print("-" * 76)
    for body_name, info in sorted(found.items()):
        dx, dy, dz = info["size"]
        print(
            f"{body_name:<16}{info['geo_type']:<8}{info['material']:<20}"
            f"{len(info['entity_ids']):<7}"
            f"{dx:.4g} x {dy:.4g} x {dz:.4g}"
        )
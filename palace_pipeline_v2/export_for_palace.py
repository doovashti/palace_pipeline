"""
export_for_palace.py

Run this script inside Ansys HFSS 2024 via Tools -> Run Script.
It exports everything needed for the Ansys -> Palace pipeline:
  1. A STEP file containing all geometry with AEDT metadata
     (body names, material names embedded as PMI properties)
  2. A device_config.json containing all non-geometric information:
     - Junction inductance and capacitance (from boundary conditions)
     - Sapphire permittivity (from material library)
     - Mesh operation sizes (from HFSS mesh operations table)
     - Boundary condition assignments (PEC, impedance, conductivity,
       lumped ports), with values resolved through design variables
     - All design variables (jj_x, jj_y, pad_x, pad_y, etc.)
     - The HFSS solution-setup properties (adaptive passes, MaxDeltaFreq,
       basis order, ...) for benchmark provenance
     - Solver settings template for Palace

After running this script, the output folder contains:
    device.step          -> feed into build_from_single_step.py
    device_config.json   -> feed into write_palace_config.py

No other files needed from Ansys.

Usage:
    In Ansys HFSS: Tools -> Run Script -> select this file

REVISION NOTES (this version):
  * AEDT boolean props ("True"/"False" STRINGS) are now parsed with
    is_true(); previously bool("False") == True silently enabled the
    Use Cap / Use Induct / Use Resist branches.
  * C_JJ starts as None, is set to 0.0 ONLY when the RLC explicitly has
    Use Cap == False, and the export HARD-FAILS if a junction cap is
    enabled but unparseable. Same for a missing L_JJ. No more silent
    C = 0 configs.
  * Export hard-fails if any model body ends up with no physics role
    (the exporter-side twin of the mesher/config validator).
  * Impedance boundaries: Reactance is exported alongside Resistance,
    values resolve through design variables, and a nonzero reactance
    triggers a loud warning (Palace needs it converted to Ls/Cs, not
    copied). Exponent notation ("1.5e3ohm") parses correctly.
  * Finite Conductivity boundaries now export their sigma / permeability /
    thickness props instead of just a role tag.
  * HFSS solution setups (passes, MaxDeltaFreq, basis order, min freq)
    are recorded under "ansys_setups" for provenance.
  * Full objects table restored (all solids and sheets, with material,
    model flag, solve_inside) so the mesher does not regress.
  * palace_solver template: target 3.5 GHz (safely below the lowest
    mode), explicit eig_tol / linear_tol / AMR fields. No downstream
    guessing.
  * Unit parsers: cm/nm handled in lengths, mH in inductance, uF in
    capacitance.
"""

import ast
import datetime
import os
import json
import re
import shutil

import pyaedt


# =====================================================================
# CONFIGURATION
# =====================================================================

# Default Palace solver settings template written into device_config.json.
# Explicit on purpose: nothing downstream should have to guess a tolerance.
PALACE_SOLVER_TEMPLATE = {
    "target_freq_GHz": 3.5,      # safely below the lowest mode (~4.07 GHz);
                                 # do NOT set this inside the spectrum
    "n_modes":         5,
    "solver_order":    2,
    "eig_tol":         1.0e-5,
    "linear_tol":      1.0e-8,
    "linear_max_its":  500,
    "eig_max_its":     500,      # v0.16+ defaults to 1e6 -> stalls
    "amr_max_its":     3,
    "amr_tol":         5.0e-4,
    "amr_update_fraction": 0.7,
    "amr_max_size":    4000000,
}

# Mesher toggles written into device_config.json as "mesher". The Gmsh
# mesher reads these; CLI flags on mesh_any.py override per run.
#   mesh_order 2 -> curvilinear 10-node tets (curved pins meshed exactly;
#                   pair with fewer curvature segments, e.g. 8)
MESHER_TEMPLATE = {
    "mesh_order": 2,
    "curvature_elements_per_2pi": 24,
    "grading": True,
}

# Per-body mesh sizes (mm) injected into "mesh_operations" at export
# time, keyed by HFSS body name. Use this to pin down sizes without
# creating a real mesh operation in the HFSS design.
#
#   - An entry here ALWAYS WINS over an HFSS mesh operation assigned to
#     the same body (loudly, in the log). Delete the entry to hand
#     control back to HFSS.
#   - A body name not present in the current design is skipped with a
#     note (non-fatal), so this table can be shared across designs --
#     but beware: an entry DOES apply to every design that contains a
#     body with that name.
#   - Leave the dict empty ({}) for pure HFSS-driven sizing.
MESH_SIZE_OVERRIDES_MM = {
    "pin":    0.5,
    "Pin_1":  0.4,
    "cavity": 2.0,
}

# Design to export. Set to None to use whichever design is currently active,
# but that is unreliable when a project holds several designs.
DESIGN_NAME = "cavity_pin_loss_chip_JJ"


# =====================================================================
# HELPERS
# =====================================================================

def is_true(props, key, default=False):
    """
    Read an AEDT boolean property.

    AEDT returns booleans as the STRINGS "True"/"False"; bool("False") is
    True in Python, so a plain truthiness test silently enables disabled
    branches. Compare text instead.
    """
    return str(props.get(key, default)).strip().lower() == "true"


def parse_value_to_mm(val_str):
    """
    Parse an Ansys value string with units to millimetres.
    Handles: nm, um, mm, cm, m, mil, in
    Examples: '0.2um' -> 0.0002, '1mm' -> 1.0, '17mm' -> 17.0
    """
    val_str = str(val_str).strip().lower().replace(" ", "")
    suffixes = [
        ("nm", 1e-6), ("um", 1e-3), ("mm", 1.0), ("cm", 10.0),
        ("mil", 0.0254), ("in", 25.4), ("m", 1e3),
    ]
    for suffix, scale in suffixes:
        if val_str.endswith(suffix):
            try:
                return float(val_str[: -len(suffix)]) * scale
            except ValueError:
                return None
    try:
        return float(val_str)
    except ValueError:
        return None


def parse_value_to_nh(val_str):
    """Parse inductance string to nH. Handles pH, nH, uH, mH, H."""
    val_str = str(val_str).strip().lower().replace(" ", "")
    suffixes = [("ph", 1e-3), ("nh", 1.0), ("uh", 1e3), ("mh", 1e6),
                ("h", 1e9)]
    for suffix, scale in suffixes:
        if val_str.endswith(suffix):
            try:
                return float(val_str[: -len(suffix)]) * scale
            except ValueError:
                return None
    try:
        return float(val_str)
    except ValueError:
        return None


def parse_value_to_ff(val_str):
    """Parse capacitance string to fF. Handles fF, pF, nF, uF, F."""
    val_str = str(val_str).strip().lower().replace(" ", "")
    suffixes = [("ff", 1.0), ("pf", 1e3), ("nf", 1e6), ("uf", 1e9),
                ("f", 1e15)]
    for suffix, scale in suffixes:
        if val_str.endswith(suffix):
            try:
                return float(val_str[: -len(suffix)]) * scale
            except ValueError:
                return None
    try:
        return float(val_str)
    except ValueError:
        return None


def parse_value_to_ohm(val_str):
    """
    Parse a resistance/reactance string to ohms.

    Strips 'ohm'/'ohms' as a unit SUFFIX (rather than deleting characters
    anywhere, which mangled exponent notation like '1.5e3ohm').
    """
    if val_str is None:
        return None
    val_str = str(val_str).strip().lower().replace(" ", "")
    for suffix in ("ohms", "ohm"):
        if val_str.endswith(suffix):
            val_str = val_str[: -len(suffix)]
            break
    try:
        return float(val_str)
    except ValueError:
        return None


def parse_value_to_s_per_m(val_str):
    """Parse a conductivity string to S/m (AEDT gives 'siemens/m' or bare)."""
    if val_str is None:
        return None
    val_str = str(val_str).strip().lower().replace(" ", "")
    for suffix in ("siemens/m", "s/m", "sie/m"):
        if val_str.endswith(suffix):
            val_str = val_str[: -len(suffix)]
            break
    try:
        return float(val_str)
    except ValueError:
        return None


def resolve_expression(expr, design_vars_raw):
    """
    Ansys stores boundary values as either a literal ('3fF') or a
    reference to a design variable ('L_JJ'). Resolve references against
    the design variable table, following one level of indirection.
    """
    if expr is None:
        return None
    expr_str = str(expr).strip()

    # Already a literal with units or a bare number.
    if re.match(r"^[-+]?[\d.]+([eE][-+]?\d+)?\s*[a-zA-Z/]*$", expr_str):
        # Distinguish '3fF' (literal) from 'L_JJ' (identifier).
        if re.match(r"^[-+]?[\d.]", expr_str):
            return expr_str

    # Strip a leading '$' used by project-level variables.
    key = expr_str.lstrip("$")

    if key in design_vars_raw:
        return str(design_vars_raw[key])

    # Case-insensitive fallback.
    for var_name, var_val in design_vars_raw.items():
        if var_name.lower() == key.lower():
            return str(var_val)

    return expr_str


def build_face_to_body_map(hfss):
    """
    Map Ansys internal face IDs to the body names they belong to.

    Needed because Ansys reports the two things inconsistently:

        mesh operations -> body NAMES   e.g. assignment=['JJ']
        boundaries      -> face IDs     e.g. assignment=['137']

    Body names survive into the STEP file as AEDT metadata; face IDs do
    not, so a boundary's assignment is unusable downstream until it is
    resolved back to a name. With this map, "which sheet is the junction"
    is answered by Ansys (whatever carries the Lumped RLC) rather than by
    hardcoding a name in the mesher.

    Returns {face_id_string: body_name}.
    """
    face_to_body = {}
    try:
        object_names = hfss.modeler.object_names
    except Exception as error:
        print(f"  WARNING: could not list modeler objects: {error}")
        return face_to_body

    for object_name in object_names:
        try:
            obj = hfss.modeler[object_name]
            if obj is None:
                continue
            for face in obj.faces:
                face_to_body[str(face.id)] = object_name
        except Exception as error:
            print(f"  WARNING: could not read faces of "
                  f"'{object_name}': {error}")

    return face_to_body


def resolve_assignment(raw_assignment, face_to_body):
    """
    Turn a boundary assignment into body names.

    Ansys may give face IDs (['137']) or object names (['pads']) depending
    on how the boundary was assigned, so handle both. Returns the unique
    body names, plus any entries that could not be resolved.
    """
    names = []
    unresolved = []

    for item in raw_assignment:
        item_str = str(item)
        if item_str in face_to_body:
            names.append(face_to_body[item_str])
        elif not item_str.lstrip("-").isdigit():
            # Already a name rather than an ID.
            names.append(item_str)
        else:
            unresolved.append(item_str)

    # Preserve order, drop duplicates -- several faces map to one body.
    seen = set()
    unique_names = []
    for name in names:
        if name not in seen:
            seen.add(name)
            unique_names.append(name)

    return unique_names, unresolved


def _model_length_to_mm(value, model_units):
    """Convert a numeric model-coordinate value to millimetres."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    unit = str(model_units or "mm").strip().lower()
    scale = {
        "nm": 1.0e-6,
        "um": 1.0e-3,
        "mm": 1.0,
        "cm": 10.0,
        "m": 1.0e3,
        "mil": 0.0254,
        "in": 25.4,
    }.get(unit)
    if scale is None:
        return None
    return number * scale


def _vector_to_mm(values, model_units):
    """Convert a three-component model-coordinate vector to millimetres."""
    if values is None:
        return None
    try:
        converted = [
            _model_length_to_mm(values[index], model_units)
            for index in range(3)
        ]
    except (TypeError, IndexError):
        return None
    return converted if all(value is not None for value in converted) else None


def _face_sample_bbox_mm(face, model_units):
    """
    Estimate a face bounding box from non-destructive PyAEDT samples.

    Samples include face vertices, edge midpoints, and the face center. For
    ordinary planar polygonal faces this is the exact axis-aligned bounding
    box. For curved faces it is a sampled box, so center/area/normal are also
    exported and should be used by the downstream matcher.
    """
    points = []

    try:
        for vertex in face.vertices:
            position = _vector_to_mm(vertex.position, model_units)
            if position is not None:
                points.append(position)
    except Exception:
        pass

    try:
        for edge in face.edges:
            midpoint = _vector_to_mm(edge.midpoint, model_units)
            if midpoint is not None:
                points.append(midpoint)
    except Exception:
        pass

    try:
        face_center = _vector_to_mm(face.center, model_units)
        if face_center is not None:
            points.append(face_center)
    except Exception:
        pass

    if not points:
        return None

    return [
        min(point[0] for point in points),
        min(point[1] for point in points),
        min(point[2] for point in points),
        max(point[0] for point in points),
        max(point[1] for point in points),
        max(point[2] for point in points),
    ]


def build_face_geometry_map(hfss, face_to_body):
    """
    Record geometric signatures for AEDT faces used by boundaries.

    Returns a mapping keyed by the string AEDT face ID. The data is purely
    descriptive and does not alter the HFSS model. It lets the Gmsh mesher
    recover a face after STEP export, where AEDT face IDs themselves are not
    preserved.
    """
    geometry = {}

    try:
        model_units = hfss.modeler.model_units
    except Exception:
        model_units = "mm"

    try:
        object_names = hfss.modeler.object_names
    except Exception as error:
        print(f"  WARNING: could not list objects for face geometry: {error}")
        return geometry

    for object_name in object_names:
        try:
            obj = hfss.modeler[object_name]
            if obj is None:
                continue

            for face in obj.faces:
                face_id = str(face.id)
                center_mm = None
                area_mm2 = None
                normal = None
                is_planar = None

                try:
                    center_mm = _vector_to_mm(face.center, model_units)
                except Exception:
                    pass

                try:
                    area_scale = _model_length_to_mm(1.0, model_units)
                    if area_scale is not None:
                        area_mm2 = float(face.area) * area_scale * area_scale
                except Exception:
                    pass

                try:
                    is_planar = bool(face.is_planar)
                except Exception:
                    pass

                if is_planar:
                    try:
                        raw_normal = face.normal
                        if raw_normal is not None:
                            normal = [float(raw_normal[i]) for i in range(3)]
                    except Exception:
                        pass

                geometry[face_id] = {
                    "face_id": face_id,
                    "body": face_to_body.get(face_id, object_name),
                    "center_mm": center_mm,
                    "bbox_sample_mm": _face_sample_bbox_mm(face, model_units),
                    "area_mm2": area_mm2,
                    "normal": normal,
                    "is_planar": is_planar,
                }
        except Exception as error:
            print(
                f"  WARNING: could not read face geometry of "
                f"'{object_name}': {error}"
            )

    return geometry


# Ansys boundary type -> the role that body plays in the Palace model.
# This is what removes the hardcoded body names from the mesher: the
# junction is whatever Ansys put a Lumped RLC on, not whatever is called
# "JJ".
BOUNDARY_TYPE_TO_ROLE = {
    "Lumped RLC":        "junction",
    "LumpedRLC":         "junction",
    "Perfect E":         "pec_sheet",
    "PerfectE":          "pec_sheet",
    "Finite Conductivity": "conductivity_sheet",
    "Impedance":         "impedance_sheet",
    "Layered Impedance": "impedance_sheet",
}

ROLE_TO_STRATEGY = {
    "junction":           "junction",
    "pec_sheet":          "pec_sheet",
    "impedance_sheet":    "impedance_sheet",
    "conductivity_sheet": "surface_conductor",
    "pec_solid":          "surface_pec",
    "conductor_solid":    "surface_conductor",
    "dielectric":         "volume_dielectric",
}


# =====================================================================
# MAIN
# =====================================================================

def main(design_name_arg=None, mesh_size_overrides=None,
         palace_solver_overrides=None, mesher_overrides=None):
    """
    Export the active (or named) HFSS design for the Palace pipeline.

    All arguments are optional; without them the module-level constants
    apply, so `python export_for_palace.py` behaves as before. From a
    notebook:

        import export_for_palace as exp
        run_dir = exp.main(
            design_name_arg="my_design",
            mesh_size_overrides={"JJ": 0.0002, "pads": 0.15},
            palace_solver_overrides={"target_freq_GHz": 3.5},
            mesher_overrides={"mesh_order": 2},
        )

    Returns the path of the created run folder.
    """
    requested_design = DESIGN_NAME if design_name_arg is None \
        else design_name_arg
    size_overrides = dict(MESH_SIZE_OVERRIDES_MM) \
        if mesh_size_overrides is None else dict(mesh_size_overrides)
    solver_template = dict(PALACE_SOLVER_TEMPLATE)
    if palace_solver_overrides:
        solver_template.update(palace_solver_overrides)
    mesher_template = dict(MESHER_TEMPLATE)
    if mesher_overrides:
        mesher_template.update(mesher_overrides)

    print("="*60)
    print("export_for_palace.py")
    print("Connecting to running Ansys HFSS instance...")
    print("="*60)

    # Collect fatal problems and refuse to write the config at the end if
    # any exist. A device_config.json with silently missing physics is
    # worse than no export at all.
    export_errors = []

    # connect to the already-open HFSS session
    if requested_design:
        hfss = pyaedt.Hfss(design=requested_design, new_desktop=False)
    else:
        hfss = pyaedt.Hfss(new_desktop=False)

    project_name = hfss.project_name
    design_name  = hfss.design_name
    project_path = hfss.project_path

    print(f"\nProject: {project_name}")
    print(f"Design:  {design_name}")
    if requested_design and design_name != requested_design:
        raise RuntimeError(
            "Active design is '%s' but the requested design is '%s'. "
            "Aborting so the export does not mix data from two designs."
            % (design_name, requested_design)
        )
    print(f"Path:    {project_path}")

    # Create a FRESH, timestamped run folder next to the .aedt file.
    # Every export is self-contained: geometry + config + the pipeline
    # scripts needed to mesh it, so a run folder can be archived or
    # copied to another machine as-is and old exports are never
    # silently overwritten.
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        project_path, f"{design_name}_export_{stamp}"
    )
    os.makedirs(output_dir, exist_ok=False)
    print(f"Output:  {output_dir}")

    # Copy the meshing pipeline into the run folder, from wherever this
    # exporter script itself lives (keep them in one folder together).
    PIPELINE_REQUIRED = [
        "mesh_any.py", "palace_matchers.py", "step_bodies.py",
    ]
    PIPELINE_OPTIONAL = [
        "mesh_stats.py", "pipeline.ipynb",
        "write_palace_config.py", "run_palace.pbs",
        "mesh_size_database.json", "palace_runlog.py",
    ]
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = None
        print("  WARNING: cannot locate this script's folder (__file__ "
              "undefined); pipeline files NOT copied -- copy mesh_any.py, "
              "palace_matchers.py and step_bodies.py in by hand")
    if script_dir:
        missing_required = []
        for fname in PIPELINE_REQUIRED + PIPELINE_OPTIONAL:
            src_path = os.path.join(script_dir, fname)
            if os.path.isfile(src_path):
                shutil.copy2(src_path, os.path.join(output_dir, fname))
                print(f"  copied:  {fname}")
            elif fname in PIPELINE_REQUIRED:
                missing_required.append(fname)
            else:
                print(f"  (optional {fname} not found next to exporter; "
                      f"skipped)")
        if missing_required:
            export_errors.append(
                "pipeline files missing next to export_for_palace.py "
                f"(the run folder cannot mesh without them): "
                f"{missing_required}")

    # ----------------------------------------------------------------
    # 1. Export STEP file
    # ----------------------------------------------------------------
    print("\n" + "-"*40)
    print("Exporting STEP file...")

    step_filename = "device.step"
    step_path = os.path.join(output_dir, step_filename)

    # pyAEDT 0.17.x does not accept assignment/subassignment here.
    hfss.export_3d_model(
        file_name=os.path.splitext(step_filename)[0],
        file_path=output_dir,
        file_format=".step",
    )
    print(f"  written: {step_path}")

    # ----------------------------------------------------------------
    # 2. Read design variables
    # ----------------------------------------------------------------
    print("\n" + "-"*40)
    print("Reading design variables...")

    design_vars_raw = {}
    design_vars_mm  = {}
    try:
        for var_name, var_obj in hfss.variable_manager.design_variables.items():
            expr = str(var_obj.expression)
            design_vars_raw[var_name] = expr
            mm_val = parse_value_to_mm(expr)
            if mm_val is not None:
                design_vars_mm[var_name] = mm_val
            print(f"  {var_name} = {expr}"
                  + (f" = {mm_val} mm" if mm_val is not None else ""))
    except Exception as e:
        print(f"  WARNING: could not read design variables: {e}")

    # ----------------------------------------------------------------
    # 3. Read solution setups (benchmark provenance)
    # ----------------------------------------------------------------
    # Every comparison against Ansys is only as good as the setup it ran
    # with (MaxDeltaFreq, passes, basis order). Stamp them into the export
    # so a result file can always be traced to its convergence contract.
    print("\n" + "-"*40)
    print("Reading solution setups...")

    ansys_setups = {}
    try:
        for setup in hfss.setups:
            try:
                props = {k: str(v) for k, v in dict(setup.props).items()}
            except Exception:
                props = {}
            ansys_setups[setup.name] = props
            interesting = {k: v for k, v in props.items() if k in (
                "MinimumFrequency", "NumModes", "MaxDeltaFreq",
                "MaximumPasses", "MinimumPasses", "MinimumConvergedPasses",
                "BasisOrder", "PercentRefinement", "DoLambdaRefine",
            )}
            print(f"  {setup.name}: {interesting}")
    except Exception as e:
        print(f"  WARNING: could not read solution setups: {e}")

    # ----------------------------------------------------------------
    # 4. Read boundary conditions
    # ----------------------------------------------------------------
    print("\n" + "-"*40)
    print("Reading boundary conditions...")

    # Resolve face IDs to body names, so a boundary's assignment can be
    # matched against the AEDT names in the STEP file.
    face_to_body = build_face_to_body_map(hfss)
    print(f"  mapped {len(face_to_body)} face IDs to "
          f"{len(set(face_to_body.values()))} bodies")

    # AEDT face IDs do not survive STEP export. Save a geometric signature
    # for every face so the downstream Gmsh mesher can recover the exact
    # face used by PEC, impedance, conductivity, or lumped boundaries.
    face_geometry = build_face_geometry_map(hfss, face_to_body)
    print(f"  recorded geometry for {len(face_geometry)} faces")

    boundaries_info = {}
    L_JJ_nH = None
    C_JJ_fF = None            # None = not determined; 0.0 = explicitly no cap
    junction_bc_name = None

    try:
        for bc in hfss.boundaries:
            bc_name = bc.name
            bc_type = str(bc.type)

            # pyAEDT 0.17+ exposes boundary data via .properties.
            # Older versions used .props. Try both.
            bc_props = {}
            for attr in ("properties", "props"):
                try:
                    candidate = getattr(bc, attr, None)
                    if candidate:
                        bc_props = dict(candidate)
                        if bc_props:
                            break
                except Exception:
                    continue

            # pyAEDT 0.17.x does not expose the boundary assignment in
            # .properties. Use the native AEDT scripting API instead.
            bc_assign_raw = []
            try:
                boundary_module = hfss.odesign.GetModule("BoundarySetup")
                raw_assign = boundary_module.GetBoundaryAssignment(bc_name)
                bc_assign_raw = (
                    [str(item) for item in raw_assign] if raw_assign else []
                )
            except Exception as assign_error:
                print(f"    WARNING: could not read assignment for "
                      f"'{bc_name}': {assign_error}")

            bc_assign, unresolved = resolve_assignment(
                bc_assign_raw, face_to_body
            )
            if unresolved:
                print(f"    WARNING: {len(unresolved)} face ID(s) on "
                      f"'{bc_name}' did not resolve to a body: "
                      f"{unresolved[:5]}")

            missing_face_geometry = [
                face_id for face_id in bc_assign_raw
                if face_id.lstrip("-").isdigit() and face_id not in face_geometry
            ]
            if missing_face_geometry:
                print(f"    WARNING: geometry was not recorded for "
                      f"{len(missing_face_geometry)} assigned face(s) on "
                      f"'{bc_name}': {missing_face_geometry[:5]}")

            assignment_faces = [
                face_geometry[face_id]
                for face_id in bc_assign_raw
                if face_id in face_geometry
            ]

            info = {
                "type":            bc_type,
                "assignment":      bc_assign,       # body names
                "assignment_ids":  bc_assign_raw,   # raw Ansys face IDs
                # Geometric signatures used to recover the exact faces after
                # STEP import. This supplements, rather than replaces, the
                # existing body-name assignment.
                "assignment_faces": assignment_faces,
                "role":            BOUNDARY_TYPE_TO_ROLE.get(bc_type),
                "props":           {k: str(v) for k, v in bc_props.items()},
            }

            # ---- Lumped RLC (the junction) -------------------------------
            # Ansys may store values as a literal ('3fF') or as a design
            # variable name ('L_JJ'); resolve through the variable table.
            # NOTE: all Use-* flags are STRING booleans -- use is_true().
            if bc_type in ("Lumped RLC", "LumpedRLC", "Inductance"):
                junction_bc_name = bc_name
                info["rlc_type"] = str(bc_props.get("RLC Type", "unknown"))

                if is_true(bc_props, "Use Induct", True) \
                        and "Inductance" in bc_props:
                    resolved = resolve_expression(
                        bc_props["Inductance"], design_vars_raw
                    )
                    L_JJ_nH = parse_value_to_nh(resolved)
                    info["L_nH"] = L_JJ_nH
                    info["L_expression"] = str(bc_props["Inductance"])
                    print(f"    Inductance: {bc_props['Inductance']!r} "
                          f"-> {resolved!r} -> {L_JJ_nH} nH")
                    if L_JJ_nH is None:
                        export_errors.append(
                            f"junction '{bc_name}': inductance "
                            f"'{bc_props['Inductance']}' resolved to "
                            f"'{resolved}' but could not be parsed to nH")

                if is_true(bc_props, "Use Cap"):
                    if "Capacitance" in bc_props:
                        resolved = resolve_expression(
                            bc_props["Capacitance"], design_vars_raw
                        )
                        C_JJ_fF = parse_value_to_ff(resolved)
                        info["C_fF"] = C_JJ_fF
                        info["C_expression"] = str(bc_props["Capacitance"])
                        print(f"    Capacitance: {bc_props['Capacitance']!r} "
                              f"-> {resolved!r} -> {C_JJ_fF} fF")
                        if C_JJ_fF is None:
                            export_errors.append(
                                f"junction '{bc_name}': Use Cap is enabled "
                                f"but capacitance "
                                f"'{bc_props['Capacitance']}' could not be "
                                f"parsed. Refusing to default to 0 -- that "
                                f"is a 45 MHz silent error on this device.")
                    else:
                        export_errors.append(
                            f"junction '{bc_name}': Use Cap is enabled but "
                            f"no Capacitance property was found")
                else:
                    # Explicitly no capacitor in the RLC -- 0 is correct.
                    C_JJ_fF = 0.0
                    print("    Capacitance: Use Cap is False -> C = 0 fF "
                          "(explicit)")

                if is_true(bc_props, "Use Resist") \
                        and "Resistance" in bc_props:
                    resolved = resolve_expression(
                        bc_props["Resistance"], design_vars_raw
                    )
                    info["R_expression"] = str(bc_props["Resistance"])
                    info["R_ohm"] = parse_value_to_ohm(resolved)
                    print(f"    WARNING: junction '{bc_name}' has an "
                          f"ENABLED resistance {resolved!r} -- Palace "
                          f"LumpedPort R must match, do not drop this.")

            # ---- Impedance sheet -----------------------------------------
            if bc_type in ("Impedance", "Layered Impedance"):
                rs_resolved = resolve_expression(
                    bc_props.get("Resistance"), design_vars_raw)
                xs_resolved = resolve_expression(
                    bc_props.get("Reactance"), design_vars_raw)
                rs = parse_value_to_ohm(rs_resolved)
                xs = parse_value_to_ohm(xs_resolved)
                info["Rs_expression"] = str(bc_props.get("Resistance"))
                info["Xs_expression"] = str(bc_props.get("Reactance"))
                info["Rs_ohm_per_sq"] = rs
                info["Xs_ohm_per_sq"] = xs if xs is not None else 0.0
                if rs is None and bc_type == "Impedance":
                    export_errors.append(
                        f"impedance boundary '{bc_name}': resistance "
                        f"'{bc_props.get('Resistance')}' could not be parsed")
                if xs not in (None, 0.0):
                    print(f"    WARNING: impedance boundary '{bc_name}' has "
                          f"nonzero reactance {xs} ohm/sq. Palace expresses "
                          f"reactance as Ls (X=wL) or Cs (X=-1/wC) per "
                          f"square -- it must be CONVERTED at the mode "
                          f"frequency, not copied. Downstream config "
                          f"generation must handle this explicitly.")
                if bc_type == "Layered Impedance":
                    print(f"    WARNING: '{bc_name}' is a Layered Impedance "
                          f"boundary; its layer stack is exported in props "
                          f"but NOT interpreted. Verify the Palace "
                          f"equivalent manually.")

            # ---- Finite Conductivity sheet -------------------------------
            if bc_type in ("Finite Conductivity", "FiniteCond"):
                sigma = parse_value_to_s_per_m(resolve_expression(
                    bc_props.get("Conductivity"), design_vars_raw))
                mu = None
                try:
                    mu_raw = resolve_expression(
                        bc_props.get("Permeability"), design_vars_raw)
                    mu = float(mu_raw) if mu_raw is not None else None
                except (TypeError, ValueError):
                    pass
                info["conductivity_S_per_m"] = sigma
                info["permeability"] = mu if mu is not None else 1.0
                info["use_thickness"] = is_true(bc_props, "Use Thickness")
                if info["use_thickness"]:
                    info["thickness_expression"] = str(
                        bc_props.get("Thickness"))
                    info["thickness_mm"] = parse_value_to_mm(
                        resolve_expression(bc_props.get("Thickness"),
                                           design_vars_raw))
                if sigma is None:
                    export_errors.append(
                        f"finite-conductivity boundary '{bc_name}': "
                        f"conductivity "
                        f"'{bc_props.get('Conductivity')}' could not be "
                        f"parsed")
                else:
                    print(f"    Conductivity: {sigma:g} S/m, mu_r = "
                          f"{info['permeability']}, thickness = "
                          f"{'enabled' if info['use_thickness'] else 'semi-infinite'}")

            boundaries_info[bc_name] = info
            role = info["role"] or "unmapped"
            if info["role"] is None:
                export_errors.append(
                    f"boundary '{bc_name}' has type '{bc_type}' with no "
                    f"role mapping -- add it to BOUNDARY_TYPE_TO_ROLE or "
                    f"remove it from the design")
            print(f"  {bc_name}: type={bc_type}  role={role}  "
                  f"bodies={bc_assign}")

    except Exception as e:
        print(f"  WARNING: could not read boundaries: {e}")

    # A junction is a FACT of the design, not a requirement of the
    # pipeline. No Lumped RLC boundary -> has_junction False, and no
    # stale design variable may smuggle in an L for a device without
    # one (schema 3 wrote L_JJ_nH: 10.0 into a junction-less export).
    has_junction = junction_bc_name is not None

    if has_junction and L_JJ_nH is None:
        export_errors.append(
            f"junction '{junction_bc_name}' exists but its inductance "
            f"could not be resolved -- refusing a junction with no L")
    if has_junction and C_JJ_fF is None:
        export_errors.append(
            f"junction '{junction_bc_name}': capacitance state could not "
            f"be determined (neither parsed nor explicitly disabled)")
    if not has_junction:
        L_JJ_nH = None
        C_JJ_fF = None
        print("\n  No Lumped RLC boundary: exporting has_junction=False")
    else:
        print(f"\n  L_JJ = {L_JJ_nH} nH, C_JJ = {C_JJ_fF} fF")

    # ----------------------------------------------------------------
    # 5. Read mesh operations
    # ----------------------------------------------------------------
    print("\n" + "-"*40)
    print("Reading mesh operations...")

    mesh_ops = {}
    try:
        for mesh_op in hfss.mesh.meshoperations:
            op_name = mesh_op.name
            op_type = str(mesh_op.type) if hasattr(mesh_op, "type") else "unknown"

            op_props = {}
            for attr in ("props", "properties"):
                try:
                    candidate = getattr(mesh_op, attr, None)
                    if candidate:
                        op_props = dict(candidate)
                        if op_props:
                            break
                except Exception:
                    continue

            # Ansys uses "Max Length" with a space, not "MaxLength".
            size_mm = None
            for key in ("Max Length", "MaxLength",
                        "Restrict Length", "RestrictLength"):
                if key in op_props:
                    size_mm = parse_value_to_mm(str(op_props[key]))
                    if size_mm is not None:
                        break

            assignment = (
                op_props.get("Objects")
                or op_props.get("Faces")
                or op_props.get("Edges")
                or []
            )

            mesh_ops[op_name] = {
                "type":       op_type,
                "size_mm":    size_mm,
                "assignment": assignment,
                "props":      {k: str(v) for k, v in op_props.items()},
            }
            print(f"  {op_name}: size={size_mm} mm, assignment={assignment}")

    except Exception as e:
        print(f"  WARNING: could not read mesh operations: {e}")

    # ----------------------------------------------------------------
    # 5b. Export mesh statistics (per-body RMS edge of the ADAPTED mesh)
    # ----------------------------------------------------------------
    # What HFSS's refinement actually settled on for this design. The
    # mesher uses it as the middle sizing tier: op > stats > fallback.
    # Only exists if the design was solved; failure here is non-fatal.
    print("\n" + "-"*40)
    print("Exporting mesh statistics...")

    def parse_mesh_stats_file(path, known_bodies, scale):
        stats = {}
        try:
            text = open(path, errors="replace").read()
        except OSError:
            return stats
        for line in text.splitlines():
            parts = line.replace(",", " ").replace("|", " ").split()
            if not parts:
                continue
            name = parts[0].strip("'\"")
            if name not in known_bodies:
                continue
            nums = []
            for p in parts[1:]:
                try:
                    nums.append(float(p))
                except ValueError:
                    pass
            if len(nums) >= 4:
                stats[name] = {
                    "num_tets":    int(nums[0]),
                    "min_edge_mm": nums[1] * scale,
                    "max_edge_mm": nums[2] * scale,
                    "rms_edge_mm": nums[3] * scale,
                }
        return stats

    ansys_mesh_stats = {}
    try:
        _stat_units = hfss.modeler.model_units
    except Exception:
        _stat_units = "mm"
    _unit_scale = _model_length_to_mm(1.0, _stat_units) or 1.0
    try:
        _known_bodies = set(hfss.modeler.object_names)
    except Exception:
        _known_bodies = set()
    try:
        for setup in hfss.setups:
            try:
                ms_path = os.path.join(
                    output_dir, "mesh_stats_%s.ms" % setup.name)
                hfss.export_mesh_stats(setup.name, variations="",
                                       mesh_path=ms_path)
                parsed = parse_mesh_stats_file(
                    ms_path, _known_bodies, _unit_scale)
                if parsed:
                    ansys_mesh_stats[setup.name] = {
                        "bodies": parsed,
                        "file": os.path.basename(ms_path),
                    }
                    print("  %s: %d tets across %d bodies" % (
                        setup.name,
                        sum(b["num_tets"] for b in parsed.values()),
                        len(parsed)))
                else:
                    print("  %s: stats file written but not parsed -- "
                          "raw file kept for inspection" % setup.name)
            except Exception as error:
                print("  NOTE: no mesh stats for '%s' (%s) -- design "
                      "not solved? Non-fatal." % (setup.name, error))
    except Exception as error:
        print("  WARNING: mesh statistics export skipped: %s" % error)

    # ----------------------------------------------------------------
    # 6. Read material properties
    # ----------------------------------------------------------------
    print("\n" + "-"*40)
    print("Reading material properties...")

    materials_info = {}
    sapphire_permittivity = None

    def read_material_value(mat, prop_name):
        """
        pyAEDT returns a property object, not a plain number. Try the
        common access patterns and return a float or None.
        """
        try:
            prop = getattr(mat, prop_name, None)
            if prop is None:
                return None
            for attr in ("value", "evaluated_value"):
                val = getattr(prop, attr, None)
                if val is not None:
                    try:
                        return float(str(val))
                    except (TypeError, ValueError):
                        continue
            return float(str(prop))
        except (TypeError, ValueError, AttributeError):
            return None

    try:
        for mat_name, mat in hfss.materials.material_keys.items():
            eps  = read_material_value(mat, "permittivity")
            mu   = read_material_value(mat, "permeability")
            tand = read_material_value(mat, "dielectric_loss_tangent")
            conductivity = read_material_value(mat, "conductivity")
            if conductivity is None and mat_name.lower() == "copper":
                conductivity = 5.8e7

            materials_info[mat_name] = {
                "permittivity": eps,
                "permeability": mu,
                "loss_tangent": tand if tand is not None else 0.0,
            }
            if conductivity is not None and conductivity > 0.0:
                materials_info[mat_name]["conductivity_S_per_m"] = conductivity
            print(f"  {mat_name}: eps={eps}, mu={mu}, tand={tand}")

            if "sapphire" in mat_name.lower() and eps is not None:
                sapphire_permittivity = eps

    except Exception as e:
        print(f"  WARNING: could not read materials: {e}")

    if sapphire_permittivity is None:
        print("  WARNING: sapphire permittivity not found, defaulting to 10.0")
        sapphire_permittivity = 10.0

    # ----------------------------------------------------------------
    # 7. Build simplified mesh size map from mesh operations
    # ----------------------------------------------------------------
    # Map mesh operation sizes to canonical component names
    # by matching operation names to known keywords
    mesh_sizes_mm = {
        "JJ":          0.0002,   # defaults if not found in mesh ops
        "thin_lead":   0.0002,
        "medium_lead": 0.003,
        "pads":        0.05,
        "cavity":      1.0,
        "RR":          1.0,
    }

    keyword_map = {
        "jj":          "JJ",
        "junction":    "JJ",
        "thin":        "thin_lead",
        "medium":      "medium_lead",
        "pad":         "pads",
        "cav":         "cavity",
        "rr":          "RR",
        "resonator":   "RR",
    }

    for op_name, op_info in mesh_ops.items():
        if op_info["size_mm"] is None:
            continue
        op_lower = op_name.lower()
        for keyword, component in keyword_map.items():
            if keyword in op_lower:
                mesh_sizes_mm[component] = op_info["size_mm"]
                print(f"  mesh size: {component} = {op_info['size_mm']} mm "
                      f"(from op '{op_name}')")
                break

    # Authoritative per-body sizes from operation ASSIGNMENTS. The
    # keyword map above matches the operation NAME, which lies:
    # 'cav_pin1mm' is assigned to the pin, not the cavity.
    mesh_sizes_by_body = {}
    for op_name, op_info in mesh_ops.items():
        size = op_info.get("size_mm")
        if size is None:
            continue
        assignment = op_info.get("assignment") or []
        if isinstance(assignment, str):
            try:
                assignment = ast.literal_eval(assignment)
            except (ValueError, SyntaxError):
                assignment = [assignment]
        if not isinstance(assignment, (list, tuple)):
            assignment = [assignment]
        for body in assignment:
            body = str(body)
            prev = mesh_sizes_by_body.get(body)
            mesh_sizes_by_body[body] = (
                size if prev is None else min(prev, size))
            print(f"  mesh size by body: {body} = "
                  f"{mesh_sizes_by_body[body]} mm (op '{op_name}')")

    # ----------------------------------------------------------------
    # 7b. Apply script-side mesh size overrides
    # ----------------------------------------------------------------
    # MESH_SIZE_OVERRIDES_MM always wins for its body. The mesher
    # min-combines every operation covering a body, so simply ADDING an
    # override operation could not make a body COARSER than an existing
    # HFSS operation. To guarantee the override wins in both directions,
    # the body is removed from any HFSS operation's assignment and given
    # its own synthetic operation instead.
    def _assignment_as_list(raw):
        if isinstance(raw, str):
            try:
                raw = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                raw = [raw]
        if not isinstance(raw, (list, tuple)):
            raw = [raw]
        return [str(item) for item in raw]

    try:
        _existing_bodies = set(
            str(name) for name in hfss.modeler.object_names)
    except Exception:
        _existing_bodies = None   # cannot verify; apply overrides anyway

    for body, size in size_overrides.items():
        body = str(body)
        if _existing_bodies is not None and body not in _existing_bodies:
            print(f"  override: body '{body}' not in this design; skipped")
            continue

        for op_name, op_info in mesh_ops.items():
            assignment = _assignment_as_list(op_info.get("assignment"))
            if body in assignment:
                print(f"  override: '{body}' removed from HFSS op "
                      f"'{op_name}' ({op_info.get('size_mm')} mm) -- "
                      f"MESH_SIZE_OVERRIDES_MM wins")
                op_info["assignment"] = [
                    item for item in assignment if item != body]

        override_op_name = f"script_override_{body}"
        mesh_ops[override_op_name] = {
            "type": "Length Based",
            "size_mm": float(size),
            "assignment": [body],
            "props": {
                "Name": override_op_name,
                "Source": ("MESH_SIZE_OVERRIDES_MM in "
                           "export_for_palace.py"),
            },
        }
        mesh_sizes_by_body[body] = float(size)
        print(f"  mesh size by body: {body} = {float(size):g} mm "
              f"(script override)")

    # ----------------------------------------------------------------
    # 8. Derive body roles + full objects table
    # ----------------------------------------------------------------
    # This is what lets the mesher stop hardcoding body names.
    #
    #   solids  -> role from the material
    #                "perfect conductor"  -> pec_solid
    #                copper/aluminum/...  -> conductor_solid
    #                anything else        -> dielectric
    #   sheets  -> role from the Ansys boundary assigned to them
    #                Lumped RLC -> junction
    #                Perfect E  -> pec_sheet
    #                Impedance  -> impedance_sheet
    #
    # So the junction is whatever Ansys put a Lumped RLC on, rather than
    # whatever is named "JJ".
    print("\n" + "-"*40)
    print("Deriving body roles...")

    body_roles = {}
    objects_info = {}

    # Only true SHEET bodies may take their role from a boundary. A
    # boundary assigned to a face of a solid (e.g. an impedance disk on
    # one cavity wall) stays at face level in assignment_faces --
    # stamping it onto the solid is how the cavity got mislabelled
    # 'impedance_sheet' and never became 'dielectric' (schema 3 bug).
    try:
        _sheet_bodies = set(hfss.modeler.sheet_names)
    except Exception:
        _sheet_bodies = set()

    for bc_name, bc_info in boundaries_info.items():
        role = bc_info.get("role")
        if not role:
            continue
        for body_name in bc_info["assignment"]:
            if body_name not in _sheet_bodies:
                print(f"  NOTE: boundary '{bc_name}' touches solid "
                      f"'{body_name}' via faces only; role kept at "
                      f"face level (assignment_faces), not on the body")
                continue
            existing = body_roles.get(body_name)
            if existing and existing != role:
                print(f"  WARNING: '{body_name}' has conflicting roles "
                      f"'{existing}' and '{role}' -- keeping '{existing}'")
                continue
            body_roles[body_name] = role
            print(f"  {body_name:<16} -> {role:<18} (from '{bc_name}')")

    # Solids get their role from the material, which the STEP carries.
    conductor_materials = {
        "copper", "aluminum", "aluminium", "silver", "gold", "brass",
        "bronze",
    }
    solid_names = []
    try:
        solid_names = list(hfss.modeler.solid_names)
        for obj_name in solid_names:
            if obj_name in body_roles:
                continue
            try:
                material = str(
                    hfss.modeler[obj_name].material_name).strip().lower()
            except Exception:
                material = ""
            if "perfect conductor" in material or material == "pec":
                body_roles[obj_name] = "pec_solid"
            elif material in conductor_materials:
                # Finite-conductivity metals are conductor geometry, not
                # dielectric FEM domains. The mesher cuts these solids out
                # of the dielectric while separately preserving any explicit
                # HFSS impedance boundary assigned to their exposed face.
                body_roles[obj_name] = "conductor_solid"
            else:
                body_roles[obj_name] = "dielectric"
            print(f"  {obj_name:<16} -> {body_roles[obj_name]:<18} "
                  f"(material '{material}')")
    except Exception as error:
        print(f"  WARNING: could not read solid materials: {error}")

    # Anything Ansys excludes from the solve is not device geometry.
    non_model = set()
    try:
        for obj_name in hfss.modeler.object_names:
            if not hfss.modeler[obj_name].model:
                non_model.add(obj_name)
                body_roles[obj_name] = "exclude"
                print(f"  {obj_name:<16} -> {'exclude':<18} "
                      f"(non-model object)")
    except Exception as error:
        print(f"  WARNING: could not read model flags: {error}")

    # Full objects table (all solids and sheets) -- keep the mesher's
    # richer view so nothing downstream has to re-query Ansys.
    try:
        sheet_names = []
        try:
            sheet_names = list(hfss.modeler.sheet_names)
        except Exception:
            pass
        for obj_name in solid_names + sheet_names:
            is_solid = obj_name in solid_names
            try:
                obj = hfss.modeler[obj_name]
                material = (str(obj.material_name).strip()
                            if is_solid else "")
            except Exception:
                obj, material = None, ""
            solve_inside = None
            if is_solid and obj is not None:
                try:
                    solve_inside = bool(obj.solve_inside)
                except Exception:
                    pass
            role = body_roles.get(obj_name)
            objects_info[obj_name] = {
                "name": obj_name,
                "dimension": 3 if is_solid else 2,
                "geometry_type": "solid" if is_solid else "sheet",
                "material": material,
                "model": obj_name not in non_model,
                "solve_inside": solve_inside,
                "domain_role": role,
                "modeling_strategy": ROLE_TO_STRATEGY.get(role),
            }
    except Exception as error:
        print(f"  WARNING: could not build objects table: {error}")

    # Cross-check: an HFSS solid the solver treats as a volume
    # (solve_inside True) but the pipeline treats as a surface conductor
    # would be a real model difference.
    for name, entry in objects_info.items():
        if entry.get("domain_role") == "conductor_solid" \
                and entry.get("solve_inside"):
            print(f"  WARNING: '{name}' is a conductor with Solve Inside "
                  f"ENABLED in HFSS. Palace models it as a surface "
                  f"impedance -- results will differ if internal fields "
                  f"matter (thin conductors!). Disable Solve Inside or "
                  f"flag this consciously.")

    # HARD GATE: every model body must have a physics role.
    unassigned = [
        name for name, entry in objects_info.items()
        if entry["model"] and entry.get("domain_role") is None
    ]
    if unassigned:
        export_errors.append(
            f"model bodies with NO physics role (would be silently dropped "
            f"downstream): {unassigned}")

    # ----------------------------------------------------------------
    # 9. Validate, then write device_config.json
    # ----------------------------------------------------------------
    print("\n" + "-"*40)

    if export_errors:
        print("EXPORT FAILED -- device_config.json NOT written.")
        print("Fix these in the HFSS design (or this script) and re-run:\n")
        for i, err in enumerate(export_errors, 1):
            print(f"  [{i}] {err}")
        hfss.release_desktop(close_projects=False, close_desktop=False)
        raise RuntimeError(
            f"{len(export_errors)} fatal export problem(s) -- see log above")

    print("Writing device_config.json...")

    config = {
        "schema_version": 4,
        "project":      project_name,
        "design":       design_name,
        "step_file":    step_filename,

        # junction parameters (has_junction distinguishes 'no junction'
        # from 'junction with unparsed values' -- the latter hard-fails)
        "has_junction": has_junction,
        "L_JJ_nH":     L_JJ_nH,
        "C_JJ_fF":     C_JJ_fF,

        # material properties
        "sapphire_permittivity": sapphire_permittivity,
        "materials":             materials_info,

        # mesh sizing (mm) -- one entry per component type
        "mesh_sizes_mm": mesh_sizes_mm,
        "mesh_sizes_by_body": mesh_sizes_by_body,
        "mesh_operations": mesh_ops,
        "ansys_mesh_stats": ansys_mesh_stats,

        # boundary conditions from Ansys, with assignments resolved to
        # body names -- write_palace_config.py maps these to Palace
        # boundary types, and the mesher uses them to identify bodies
        # without hardcoding names
        "boundaries":  boundaries_info,

        # body name -> role, derived above. This is what the mesher reads
        # instead of a hardcoded BODY_ROLES dict.
        "body_roles":  body_roles,
        "objects":     objects_info,

        # design variables (lengths in mm)
        "design_variables_mm":  design_vars_mm,
        "design_variables_raw": design_vars_raw,

        # HFSS solution setups -- provenance for any Ansys comparison
        "ansys_setups": ansys_setups,

        # Palace solver settings template (explicit; edit consciously)
        "palace_solver": dict(solver_template),

        # Gmsh mesher toggles (mesh order / curvature / grading)
        "mesher": dict(mesher_template),
    }

    config_path = os.path.join(output_dir, "device_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"  written: {config_path}")

    # ----------------------------------------------------------------
    # 10. Summary
    # ----------------------------------------------------------------
    print("\n" + "="*60)
    print("Export complete.")
    print(f"  STEP file:   {step_path}")
    print(f"  Config file: {config_path}")
    print(f"\nKey parameters extracted:")
    print(f"  L_JJ              = {L_JJ_nH} nH")
    print(f"  C_JJ              = {C_JJ_fF} fF")
    print(f"  sapphire eps_r    = {sapphire_permittivity}")
    print(f"  mesh sizes (mm)   = {mesh_sizes_mm}")
    print(f"  body roles        = {body_roles}")
    print(f"  setups recorded   = {list(ansys_setups)}")
    print(f"\nNext steps (the run folder is self-contained):")
    print(f"  cd {output_dir}")
    print(f"  1. python mesh_any.py            (no arguments needed)")
    print(f"  2. python mesh_stats.py device.msh")
    print(f"  3. python write_palace_config.py")
    print(f"  4. Upload mesh + palace_config.json to HPC and run Palace")
    print("="*60)

    hfss.release_desktop(close_projects=False, close_desktop=False)
    return output_dir


if __name__ == "__main__":
    main()
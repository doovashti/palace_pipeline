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
     - Boundary condition assignments (PEC, impedance, lumped ports)
     - All design variables (jj_x, jj_y, pad_x, pad_y, etc.)
     - Solver settings template

After running this script, the output folder contains:
    device.step          -> feed into build_from_single_step.py
    device_config.json   -> feed into write_palace_config.py

No other files needed from Ansys.

Usage:
    In Ansys HFSS: Tools -> Run Script -> select this file
"""

import os
import json
import re

import pyaedt


# =====================================================================
# CONFIGURATION
# =====================================================================

# Default solver settings -- override in device_config.json after export
DEFAULT_TARGET_FREQ_GHZ = 5.0
DEFAULT_N_MODES          = 2
DEFAULT_SOLVER_ORDER     = 1
DEFAULT_TOL              = 1e-3
DEFAULT_AMR_MAX_ITS      = 8

# Design to export. Set to None to use whichever design is currently active,
# but that is unreliable when a project holds several designs.
DESIGN_NAME = "galvanic_one_rr_pin_no_wirebond"

# =====================================================================
# HELPERS
# =====================================================================

def parse_value_to_mm(val_str):
    """
    Parse an Ansys value string with units to millimetres.
    Handles: um, mm, m, mil, in
    Examples: '0.2um' -> 0.0002, '1mm' -> 1.0, '17mm' -> 17.0
    """
    val_str = str(val_str).strip().lower().replace(" ", "")
    if val_str.endswith("um"):
        return float(val_str[:-2]) * 1e-3
    elif val_str.endswith("mm"):
        return float(val_str[:-2])
    elif val_str.endswith("mil"):
        return float(val_str[:-3]) * 0.0254
    elif val_str.endswith("in"):
        return float(val_str[:-2]) * 25.4
    elif val_str.endswith("m"):
        return float(val_str[:-1]) * 1e3
    else:
        try:
            return float(val_str)
        except ValueError:
            return None


def parse_value_to_nh(val_str):
    """Parse inductance string to nH."""
    val_str = str(val_str).strip().lower().replace(" ", "")
    if val_str.endswith("nh"):
        return float(val_str[:-2])
    elif val_str.endswith("uh"):
        return float(val_str[:-2]) * 1e3
    elif val_str.endswith("ph"):
        return float(val_str[:-2]) * 1e-3
    elif val_str.endswith("h"):
        return float(val_str[:-1]) * 1e9
    else:
        try:
            return float(val_str)
        except ValueError:
            return None


def parse_value_to_ff(val_str):
    """Parse capacitance string to fF."""
    val_str = str(val_str).strip().lower().replace(" ", "")
    if val_str.endswith("ff"):
        return float(val_str[:-2])
    elif val_str.endswith("pf"):
        return float(val_str[:-2]) * 1e3
    elif val_str.endswith("nf"):
        return float(val_str[:-2]) * 1e6
    elif val_str.endswith("f"):
        return float(val_str[:-1]) * 1e15
    else:
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
    if re.match(r"^[-+]?[\d.]+([eE][-+]?\d+)?\s*[a-zA-Z]*$", expr_str):
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


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("="*60)
    print("export_for_palace.py")
    print("Connecting to running Ansys HFSS instance...")
    print("="*60)

    # connect to the already-open HFSS session
    if DESIGN_NAME:
        hfss = pyaedt.Hfss(design=DESIGN_NAME, new_desktop=False)
    else:
        hfss = pyaedt.Hfss(new_desktop=False)

    project_name = hfss.project_name
    design_name  = hfss.design_name
    project_path = hfss.project_path

    print(f"\nProject: {project_name}")
    print(f"Design:  {design_name}")
    if DESIGN_NAME and design_name != DESIGN_NAME:
        raise RuntimeError(
            "Active design is '%s' but DESIGN_NAME requested '%s'. "
            "Aborting so the export does not mix data from two designs."
            % (design_name, DESIGN_NAME)
        )
    print(f"Path:    {project_path}")

    # create output folder next to the .aedt file
    output_dir = os.path.join(
        project_path, f"{project_name}_palace_export"
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output:  {output_dir}")

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
    # 3. Read boundary conditions
    # ----------------------------------------------------------------
    print("\n" + "-"*40)
    print("Reading boundary conditions...")

    # Resolve face IDs to body names, so a boundary's assignment can be
    # matched against the AEDT names in the STEP file.
    face_to_body = build_face_to_body_map(hfss)
    print(f"  mapped {len(face_to_body)} face IDs to "
          f"{len(set(face_to_body.values()))} bodies")

    boundaries_info = {}
    L_JJ_nH = None
    C_JJ_fF = 0.0

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

            info = {
                "type":            bc_type,
                "assignment":      bc_assign,       # body names
                "assignment_ids":  bc_assign_raw,   # raw Ansys face IDs
                "role":            BOUNDARY_TYPE_TO_ROLE.get(bc_type),
                "props":           {k: str(v) for k, v in bc_props.items()},
            }

            # Extract junction parameters. Ansys may store these as a
            # literal ('3fF') or as a design variable name ('L_JJ'), so
            # resolve through the design variable table first.
            if bc_type in ("Lumped RLC", "LumpedRLC", "Inductance"):
                info["rlc_type"] = str(bc_props.get("RLC Type", "unknown"))

                if bc_props.get("Use Induct", True) and "Inductance" in bc_props:
                    resolved = resolve_expression(
                        bc_props["Inductance"], design_vars_raw
                    )
                    L_JJ_nH = parse_value_to_nh(resolved)
                    info["L_nH"] = L_JJ_nH
                    info["L_expression"] = str(bc_props["Inductance"])
                    print(f"    Inductance: {bc_props['Inductance']!r} "
                          f"-> {resolved!r} -> {L_JJ_nH} nH")

                if bc_props.get("Use Cap", False) and "Capacitance" in bc_props:
                    resolved = resolve_expression(
                        bc_props["Capacitance"], design_vars_raw
                    )
                    parsed_c = parse_value_to_ff(resolved)
                    C_JJ_fF = parsed_c if parsed_c is not None else 0.0
                    info["C_fF"] = C_JJ_fF
                    info["C_expression"] = str(bc_props["Capacitance"])
                    print(f"    Capacitance: {bc_props['Capacitance']!r} "
                          f"-> {resolved!r} -> {C_JJ_fF} fF")

                if bc_props.get("Use Resist", False) and "Resistance" in bc_props:
                    info["R_expression"] = str(bc_props["Resistance"])

            # extract impedance surface resistance
            if bc_type in ("Impedance", "Layered Impedance"):
                if "Resistance" in bc_props:
                    rs_str = str(bc_props["Resistance"])
                    try:
                        info["Rs_ohm_per_sq"] = float(
                            re.sub(r"[^\d.]", "", rs_str)
                        )
                    except ValueError:
                        info["Rs_ohm_per_sq"] = rs_str

            boundaries_info[bc_name] = info
            role = info["role"] or "unmapped"
            print(f"  {bc_name}: type={bc_type}  role={role}  "
                  f"bodies={bc_assign}")

    except Exception as e:
        print(f"  WARNING: could not read boundaries: {e}")

    # fallback: read L_JJ from design variables if not found in boundaries
    if L_JJ_nH is None:
        for var_name, var_expr in design_vars_raw.items():
            if "l_jj" in var_name.lower() or var_name.upper() == "L_JJ":
                L_JJ_nH = parse_value_to_nh(var_expr)
                print(f"\n  L_JJ read from design variable '{var_name}': "
                      f"{L_JJ_nH} nH")
                break

    if L_JJ_nH is not None:
        print(f"\n  L_JJ = {L_JJ_nH} nH")
    else:
        print("\n  WARNING: L_JJ not found in boundaries or design variables")

    # ----------------------------------------------------------------
    # 4. Read mesh operations
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
    # 5. Read material properties
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

            materials_info[mat_name] = {
                "permittivity": eps,
                "permeability": mu,
                "loss_tangent": tand if tand is not None else 0.0,
            }
            print(f"  {mat_name}: eps={eps}, mu={mu}, tand={tand}")

            if "sapphire" in mat_name.lower() and eps is not None:
                sapphire_permittivity = eps

    except Exception as e:
        print(f"  WARNING: could not read materials: {e}")

    if sapphire_permittivity is None:
        print("  WARNING: sapphire permittivity not found, defaulting to 10.0")
        sapphire_permittivity = 10.0

    # ----------------------------------------------------------------
    # 6. Build simplified mesh size map from mesh operations
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

    # ----------------------------------------------------------------
    # 7. Write device_config.json
    # ----------------------------------------------------------------
    print("\n" + "-"*40)
    print("Writing device_config.json...")

    # ----------------------------------------------------------------
    # Derive body roles
    # ----------------------------------------------------------------
    # This is what lets the mesher stop hardcoding body names.
    #
    #   solids  -> role from the STEP material
    #                "perfect conductor" -> pec_solid
    #                anything else       -> dielectric
    #   sheets  -> role from the Ansys boundary assigned to them
    #                Lumped RLC -> junction
    #                Perfect E  -> pec_sheet
    #                Impedance  -> impedance_sheet
    #
    # So the junction is whatever Ansys put a Lumped RLC on, rather than
    # whatever happens to be named "JJ".
    print("\n" + "-"*40)
    print("Deriving body roles...")

    body_roles = {}

    for bc_name, bc_info in boundaries_info.items():
        role = bc_info.get("role")
        if not role:
            print(f"  '{bc_name}' has type '{bc_info['type']}' with no "
                  f"role mapping -- bodies left unassigned")
            continue
        for body_name in bc_info["assignment"]:
            existing = body_roles.get(body_name)
            if existing and existing != role:
                print(f"  WARNING: '{body_name}' has conflicting roles "
                      f"'{existing}' and '{role}' -- keeping '{existing}'")
                continue
            body_roles[body_name] = role
            print(f"  {body_name:<16} -> {role:<18} (from '{bc_name}')")

    # Solids get their role from the material, which the STEP carries.
    try:
        for obj_name in hfss.modeler.solid_names:
            if obj_name in body_roles:
                continue
            try:
                material = str(hfss.modeler[obj_name].material_name).lower()
            except Exception:
                material = ""
            if "perfect conductor" in material or "pec" == material:
                body_roles[obj_name] = "pec_solid"
            else:
                body_roles[obj_name] = "dielectric"
            print(f"  {obj_name:<16} -> {body_roles[obj_name]:<18} "
                  f"(material '{material}')")
    except Exception as error:
        print(f"  WARNING: could not read solid materials: {error}")

    # Anything Ansys excludes from the solve is not device geometry.
    try:
        for obj_name in hfss.modeler.object_names:
            if not hfss.modeler[obj_name].model:
                body_roles[obj_name] = "exclude"
                print(f"  {obj_name:<16} -> {'exclude':<18} "
                      f"(non-model object)")
    except Exception as error:
        print(f"  WARNING: could not read model flags: {error}")

    config = {
        "project":      project_name,
        "design":       design_name,
        "step_file":    step_filename,

        # junction parameters
        "L_JJ_nH":     L_JJ_nH,
        "C_JJ_fF":     C_JJ_fF,

        # material properties
        "sapphire_permittivity": sapphire_permittivity,
        "materials":             materials_info,

        # mesh sizing (mm) -- one entry per component type
        "mesh_sizes_mm": mesh_sizes_mm,

        # boundary conditions from Ansys, with assignments resolved to
        # body names -- write_palace_config.py maps these to Palace
        # boundary types, and build_from_step_named.py uses them to
        # identify bodies without hardcoding names
        "boundaries":  boundaries_info,

        # body name -> role, derived above. This is what the mesher reads
        # instead of a hardcoded BODY_ROLES dict.
        "body_roles":  body_roles,

        # design variables (lengths in mm)
        "design_variables_mm":  design_vars_mm,
        "design_variables_raw": design_vars_raw,

        # Palace solver settings template -- edit before running
        "palace_solver": {
            "target_freq_GHz": DEFAULT_TARGET_FREQ_GHZ,
            "n_modes":         DEFAULT_N_MODES,
            "solver_order":    DEFAULT_SOLVER_ORDER,
            "tol":             DEFAULT_TOL,
            "amr_max_its":     DEFAULT_AMR_MAX_ITS,
        }
    }

    config_path = os.path.join(output_dir, "device_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"  written: {config_path}")

    # ----------------------------------------------------------------
    # 8. Summary
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
    print(f"  design variables  = {design_vars_mm}")
    print(f"\nNext steps:")
    print(f"  1. python build_from_single_step.py "
          f"--step {step_path} --config {config_path}")
    print(f"  2. python write_palace_config.py --config {config_path}")
    print(f"  3. Upload mesh + palace_config.json to HPC and run Palace")
    print("="*60)

    hfss.release_desktop(close_projects=False, close_desktop=False)


if __name__ == "__main__":
    main()
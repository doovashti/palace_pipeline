# Simple transmon

This folder packages the geometry, mesh, Palace input, pyEPR analysis, and representative results for the simple transmon with a bond wire.

## Rebuild

Run `geometry_mesh/build_from_single_step_v6_contact_fix.py` from this folder's `geometry_mesh` directory to regenerate `non_model_w_bond_wire_cf.msh` and `physical_groups_w_bond_wire_cf.json` from `no_non_model.step`.

Before running Palace, place `non_model_w_bond_wire_cf.msh` beside `palace/palace_config.json` (or adjust the `Model.Mesh` path). The JJ is physical surface attribute 7 and is used by the lumped port in the Palace configuration.

`sample_results` contains selected eigensolver, port-EPR, and AMR indicators from three saved iterations. The Ansys project is included for inspection; it is not required for the Palace rebuild.

# Transmon + readout resonator + lossy pin

This folder contains a reproducible Ansys HFSS -> STEP/Gmsh -> Palace -> pyEPR
workflow for the `galvanic_one_rr_pin_no_wirebond` design.

## Contents

- `ansys/test_export.aedt` - source HFSS project. Open the
  `galvanic_one_rr_pin_no_wirebond` design and use `Setup2` for the Ansys
  reference calculation.
- `export_mesh/` - exported device geometry and the scripts used to export
  Ansys metadata and build the Gmsh mesh.
- `palace/palace_config.json` - Palace eigenmode input used for the supplied Palace result set.
- `pyepr/` - Palace-to-pyEPR bridge, analysis notebook, and a small sample of
  Palace postprocessing output.

## Physical model

- Josephson junction: physical mesh group `9` (`JJ`), 10 nH inductance. The Palace
  run uses no explicit parallel junction capacitance (`C: 0.0`). The HFSS reference
  model has a 3 fF parallel capacitance, so this is a known model difference.
- Substrate: sapphire, relative permittivity 10.
- The JJ, pads, leads, and readout resonator are two-sided internal mesh
  sheets. The JJ lumped-port direction is `[0, 1, 0]`.

## Rebuild workflow

1. Install Ansys Electronics Desktop/HFSS and Python dependencies required by
   `export_for_palace.py` (including PyAEDT). Open `ansys/test_export.aedt`.
2. Export the selected HFSS design with `export_mesh/export_for_palace.py`.
   This produces `device.step` and `device_config.json`.
3. Run `export_mesh/build_from_single_step_v9.py` from the export directory to
   regenerate `transmon_simple_RR_pin_nowire.msh`. It requires the Gmsh Python
   package and the matching STEP/config files.
4. Run Palace with a copy of `palace/palace_config.json`, editing the
   mesh path and output directory for the local machine. Do not overwrite the
   supplied sample results.
5. In the `qiskit_metal` / pyEPR environment, open
   `pyepr/palace_epr_nb (2).ipynb`. Set `POSTPRO` to the new Palace output
   directory and set `PALACE_CONFIG` to the exact JSON used for that run.

## Included sample data

`pyepr/sample_results/` contains `eig.csv`, `port-EPR.csv`, and
`error-indicators.csv` from a Palace run. These are enough to run the
postprocessing notebook after setting `POSTPRO` accordingly.

## Current comparison finding

The Palace JJ port is correctly present and the inferred qubit-resonator
coupling agrees closely with Ansys. The remaining discrepancy is the bare
cavity-qubit detuning. The supplied Palace configuration uses first-order
elements. Before drawing a convergence conclusion, compare the final AMR Norm in
`error-indicators.csv` with `Model.Refinement.Tol`, and check whether the run stopped
at `MaxIts` or `MaxSize`.

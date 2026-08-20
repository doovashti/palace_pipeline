# HFSS to Palace

This folder is a complete workflow for exporting an Ansys HFSS eigenmode
design, meshing it for AWS Palace, and running the resulting simulation on
Vanda HPC.

## Run this first

Open `palace_pipeline.ipynb` and run its cells from top to bottom. Edit only
the settings cell unless you are intentionally changing the workflow.

**Every execution creates a new, timestamped run folder next to the HFSS
`.aedt` project.** That run folder contains the exported geometry, mesh,
Palace configuration, PBS submission script, and a copy of the pipeline
files used to create it. It is the self-contained simulation package.

After the notebook finishes, drag **that newly created run folder** into
Vanda HPC, change into it, and submit:

```bash
qsub run_palace.pbs
```

Do not drag this source folder to Vanda as the simulation input; drag the
new run folder produced by the notebook. Keeping one folder per run prevents
meshes, configurations, and results from different designs or sweep points
from being overwritten or mixed.

When the Vanda job is complete, copy its `postpro/` results back into the
same run folder. Then open `palace_pyepr.ipynb` **from inside that run
folder** to analyze the results.

## What each notebook does

| File | What a user does with it |
| --- | --- |
| `palace_pipeline.ipynb` | Main driver notebook. Set the design, run tag, optional mesh overrides, and HPC settings; then run top-to-bottom to create one fresh HPC-ready run folder. |
| `palace_pyepr.ipynb` | Post-processing notebook. Open it inside a completed run folder after copying `postpro/` back from Vanda; it loads Palace output and performs pyEPR/EPR analysis. |
| `runlog.ipynb` | Optional local run registry viewer/editor for recording and comparing completed Palace jobs. |

## What each script does

| File | Purpose |
| --- | --- |
| `export_for_palace.py` | Connects to the open HFSS project through pyAEDT and exports `device.step` plus `device_config.json` into a fresh run folder. It captures geometry, materials, boundaries, junction data, mesh operations, and solver settings. |
| `aedt_extract.py` | Reads an AEDT project file directly to inspect designs, variables, mesh operations, and length-based mesh information. Useful when HFSS/pyAEDT extraction needs support or diagnosis. |
| `hfss_size_report.py` | Reports HFSS mesh-operation and solved-mesh sizes by body so they can be reviewed before meshing for Palace. |
| `find_crossing_bodies.py` | Diagnostic utility that identifies bodies that intersect or cross a selected region, helping diagnose geometry/mesh issues. |
| `mesh_any.py` | Gmsh mesher. Converts exported STEP geometry and device metadata into a Palace mesh (`.msh`) and physical-group metadata JSON. |
| `mesh_conformity_check.py` | Checks a generated Gmsh mesh for non-conforming shared faces or other mesh connectivity problems. |
| `mesh_stats.py` | Produces mesh quality, edge-size, element-count, and budget statistics from a `.msh` file. |
| `palace_matchers.py` | Geometry-matching library that recovers HFSS boundary faces on the final Gmsh geometry after Boolean operations and imprinting. |
| `step_bodies.py` | Reads names, materials, extents, and entity relationships from the exported STEP file and helps map them into Gmsh. |
| `write_palace_config.py` | Converts device metadata and mesh physical groups into the final `palace_config_<tag>.json` used by Palace. |
| `pipeline_helpers.py` | Functions called by the main notebook to run each stage, apply overrides, write the PBS script, collect reports, and verify a run folder before HPC transfer. |
| `pipeline_state.py` | Saves and restores notebook state so the pipeline can find or resume its most recent run folder. |
| `palace_epr.py` | Loads Palace eigenmode/EPR output, validates junction ports, identifies modes, and calculates pyEPR-style quantities including coupling and nonlinear results. |
| `palace_runlog.py` | Maintains optional local CSV/JSONL/HTML records of completed Palace runs and parsed result summaries. |
| `run_mesh.pbs` | PBS batch-script template/reference for meshing or related scheduled work; the main notebook generates the `run_palace.pbs` used for the final Vanda solve. |

## Supporting file

`mesh_size_database.json` contains lab reference mesh sizes in millimetres.
The notebook can use these values alongside HFSS mesh operations and any
per-run overrides you specify.

## Expected handoff sequence

1. In HFSS, open the project and ensure no solve is running.
2. Run `palace_pipeline.ipynb` locally from this source folder.
3. Locate the new timestamped run folder created beside the `.aedt` file.
4. Drag that entire new folder into Vanda HPC.
5. On Vanda, run `cd <new-run-folder>` followed by `qsub run_palace.pbs`.
6. Copy the resulting `postpro/` folder back into that same local run folder.
7. Open `palace_pyepr.ipynb` in the run folder to inspect frequencies, Qs,
   junction participation, and pyEPR results.

The generated run folder is both the Vanda input and the durable record of
exactly what was run.

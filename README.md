# palace_pipeline
The full pipeline to take a design from Ansys HFSS and run it using Palace on the HPC, with sample results to be run in pyEPR

## palace_pipeline_v2

`palace_pipeline_v2/` contains the second-generation, end-to-end HFSS-to-Palace pipeline. It exports device geometry and simulation metadata from HFSS, creates a Gmsh mesh and Palace configuration, and prepares the HPC run artifacts. See [its README](palace_pipeline_v2/README.md) for setup and usage details.

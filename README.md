# Palace Pipeline

Tools and examples for exporting Ansys HFSS designs, generating AWS Palace meshes/configurations, running simulations on Vanda HPC, and analyzing results with pyEPR.

## Main workflow

Use [`hfss2palace/`](hfss2palace/) for the current pipeline.

1. Open `hfss2palace/palace_pipeline.ipynb`.
2. Run it from top to bottom with your HFSS design open.
3. Each run creates a new timestamped run folder beside your `.aedt` file.
4. Drag that new run folder to Vanda HPC and submit:

   ```bash
   qsub run_palace.pbs

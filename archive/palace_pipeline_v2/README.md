# Ansys HFSS → AWS Palace pipeline

Eigenmode simulation of superconducting devices (3D cavities, pins, chips,
transmon qubits with Josephson junctions), exported from Ansys HFSS 2024 and
solved with [AWS Palace](https://awslabs.github.io/palace/) on the HPC.

The design goal of this pipeline: **no hardcoded device knowledge anywhere**.
Every body name, material, boundary condition, junction value, and mesh size
is read from the HFSS design itself. Any geometry — with or without a JJ,
pads, pins, or a chip — goes through the same scripts without editing them.

---

## 1. The short version

1. Open your project in HFSS. Open `palace_pipeline.ipynb` in Jupyter.
2. Edit **cell 1** (design name, tag, optional mesh size overrides, HPC
   parameters). Run all cells.
3. A timestamped run folder appears next to your `.aedt` file containing the
   mesh, the Palace config, and `run_palace.pbs`.
4. Copy that folder to the HPC and run:

   ```bash
   cd <folder>
   qsub run_palace.pbs
   ```

5. The mode table (frequencies + Q) appears in the job's `.o` log;
   machine-readable results in `postpro/eig.csv`.

Everything below is what happens under the hood, and what to do when
something goes wrong.

---

## 2. Files

| file | role |
|---|---|
| `palace_pipeline.ipynb` | the driver — run this |
| `pipeline_helpers.py` | notebook support: size table, subprocess steps, PBS writer, checklist |
| `export_for_palace.py` | talks to HFSS via pyAEDT; writes `device.step` + `device_config.json` into a fresh run folder |
| `mesh_any.py` | gmsh mesher: STEP + config → `device_<tag>.msh` + `device_groups_<tag>.json` |
| `palace_matchers.py` | geometric matching library (HFSS faces ↔ final gmsh surfaces after booleans) |
| `step_bodies.py` | reads body names/materials/bboxes directly from the STEP file |
| `write_palace_config.py` | groups + config → `palace_config_<tag>.json` |
| `mesh_stats.py` | standalone mesh quality report from any `.msh` |
| `mesh_size_database.json` | the lab's tested mesh sizes per component (reference, mm) |

Every export copies these scripts into the run folder, so a run folder is a
self-contained, archivable record of exactly what produced its results.

---

## 3. Stage by stage

### 3.1 Export (`export_for_palace.py`)

Connects to the **running** HFSS session (nothing in your project is
modified) and writes:

* `device.step` — all geometry, with AEDT body names embedded.
* `device_config.json` — everything non-geometric:
  * **materials** (permittivity, permeability, loss tangent, conductivity);
  * **boundaries** with their *assignments resolved to body names* and a
    geometric signature (`assignment_faces`: center, area, bbox) for every
    assigned face — HFSS face IDs do not survive STEP export, so faces are
    later recovered geometrically;
  * **junction**: `has_junction`, L and C resolved through design variables.
    A design without a Lumped RLC exports `has_junction: false` — absence is
    a fact, not an error;
  * **mesh operations** (sizes keyed by *assignment*, never by the
    operation's name — an op called `cav_pin1mm` assigned to the pin sizes
    the pin);
  * **HFSS mesh statistics** per solid (RMS edge of the adapted mesh), if
    the design was ever solved;
  * **HFSS setup properties** (passes, MaxDeltaFreq, basis order) — the
    convergence contract behind any HFSS number you compare against;
  * the Palace solver template and mesher toggles.

The export **hard-fails** rather than writing a config with silently missing
physics: unparseable junction values, model bodies with no derivable role,
boundary types with no mapping — all fatal, listed at the end.

### 3.2 Roles (who is what)

The mesher classifies every body from material + `solve_inside`, never from
a boundary that merely touches a face of it:

| condition | role | in Palace |
|---|---|---|
| solid, solve_inside true | dielectric | volume with that material |
| solid, solve_inside false, σ ≥ 1e10 or "perfect" | pec_solid | subtracted; its exposed faces → PEC |
| solid, solve_inside false, 0 < σ < 1e10 | conductor_solid | subtracted; exposed faces → finite Conductivity |
| sheet named by a Lumped RLC | junction | LumpedPort (L, C, direction all derived) |
| sheet named by Perfect E | pec_sheet | PEC |
| sheet named by Impedance | impedance_sheet | Rs/Xs surface impedance |
| non-model, or role "exclude" | removed | — |

A model sheet with **no** boundary condition is a hard error — assign one in
HFSS or make the body non-model. Everything the vacuum touches that is not
otherwise claimed becomes `PEC_exterior` (this reproduces HFSS's
"everything outside the model is PEC" convention explicitly).

**Boundary conditions only act on faces the vacuum touches.** A boundary on
a buried face (e.g. the underside of a pin standing on the cavity floor) is
inert in HFSS and is skipped, with a printed note, by the pipeline.

### 3.3 Mesh sizes (three tiers + overrides)

For each body, the size resolves in priority order:

1. **HFSS mesh operation** assigned to it (this is where the lab's tested
   values — `mesh_size_database.json` — should live, set in HFSS);
2. **HFSS mesh statistics** (RMS edge of what HFSS's own refinement settled
   on) — solids only, since the `.ms` file doesn't cover sheets;
3. **analytic fallback**: sheets → narrowest in-plane dimension / 6;
   dielectrics → min(λ/12 at target, mid-dimension/2); conductors →
   smallest dimension / 3.

On top of all tiers, per-run overrides:

```
python mesh_any.py --tag myrun --size Pin_1=0.4 --size pads=0.15
```

`--size` beats everything and hard-errors on unknown body names (a typo
must not silently sweep nothing). `--tag` stamps all output filenames so
sweep points never overwrite each other. The mesher prints every resolved
size with its source — that printout is the authoritative record of what a
mesh was built with.

Units are **mm** everywhere: `0.0002` = 0.2 µm, `0.05` = 50 µm.

### 3.4 Grading (how fine regions meet coarse ones)

Every sized feature gets a Distance+Threshold field: elements held at size
`s` out to `10·s` from the feature, ramping linearly to `250·s` by `500·s`,
and **no constraint beyond that** (`StopAtDistMax` — see §6.4 for the bug
this fixes). The mesh size at any point is the minimum over all features'
fields, the global bulk size, and the curvature rule. This grading is what
keeps aspect ratios in single digits even with a 0.2 µm junction inside a
35 mm cavity (the pre-grading junction meshes had aspect-200 slivers).

**Curvature rule**: `curvature_elements_per_2pi` (default 24) independently
caps elements on curved surfaces at `2πr/N`. A 1.5 mm-radius pin is capped
near 0.39 mm regardless of its nominal size — to genuinely coarsen round
parts, lower both the size *and* the curvature setting. With curvilinear
meshing (order 2), N = 8 is geometrically fine.

### 3.5 Curvilinear meshing

`mesher.mesh_order: 2` produces 10-node tets whose edges follow curved CAD
surfaces (Palace logs `Mesh curvature order: 2`). DOF count is unchanged —
that's set by `Solver.Order`. High-order optimization runs automatically;
watch the gmsh log for "negative Jacobian" (never observed so far — that
would be a broken mesh, do not solve on it).

### 3.6 Face matching (`palace_matchers.py`)

After the boolean cuts and sheet imprints, HFSS's boundary faces must be
found among the final gmsh surfaces. The matcher tries, in order: exact
center+area match → a group of fragments whose areas sum back to the
original → a single clipped remnant (≥25 % of the original area, inside its
footprint) → *buried face* (belongs to a removed solid, flush with the
exterior, no descendant: skipped with a note — correct physics, see §3.2).
Anything genuinely ambiguous raises rather than guesses.

### 3.7 Palace config (`write_palace_config.py`)

Reads the groups file (with its `_meta` block describing what each physical
group *is*) and the device config, and emits the full Palace JSON: one
material per dielectric, PEC, Impedance, Conductivity, and one LumpedPort
per junction with L/C from the boundary record and the port direction
derived from the junction face's geometry. Gates: every mesh group must be
consumed (silently dropped physics is how a junction once got shorted to
PEC), `has_junction: true` with no junction group is fatal, no-PEC is fatal.

The solver block it writes is the validated one for Palace v0.17 on this
cluster:

```json
"Solver": {
  "Order": 2,
  "PartialAssemblyOrder": 100,        // forces full assembly
  "Eigenmode": { "N": 5, "Target": 3.5, "Tol": 1e-05,
                 "MaxIts": 500, "RefineNonlinear": false },
  "Linear":    { "Type": "SuperLU", "KSPType": "Default",
                 "MGMaxLevels": 1, "Tol": 1e-08, "MaxIts": 500 }
}
```

Why these matter (hard-won):

* `Solver.Linear.Type` is the **coarse level** of Palace's multigrid
  hierarchy, not the fine-level preconditioner. `MGMaxLevels: 1` +
  `PartialAssemblyOrder: 100` collapse the hierarchy so SuperLU factorizes
  the fine-level operator directly. Without this, GMRES stalls for hours on
  meshes spanning µm to mm. Success signature in the log: `Operator
  assembly level: Full`, a single-level hierarchy with an NNZ count, GMRES
  converging in ~11–15 iterations.
* `RefineNonlinear: false`: the quasi-Newton eigenvalue refinement diverges
  in this Palace version (Armijo line search accepts uphill steps). Cost of
  disabling ≈ 0.4 % on conductor-dominated Q.
* `Target` must sit **below** the lowest expected mode.

### 3.8 The PBS file and HPC

`run_palace.pbs` (generated by the notebook) uses the lab's validated job
settings — gcc11/openblas/openmpi modules, one MPI rank per core,
`--bind-to core`, OMP pinned to 1 thread — plus `rm -rf postpro` before the
run (§6.2). Results: the mode table in the `.o` log, `postpro/eig.csv`, and
optionally ParaView field files (§5).

---

## 4. Reading the results

**The mode table**: `Re{f}` (GHz), `Im{f}`, and Q = Re/(2·Im).

**Q has a numerical floor.** Q is only meaningful when the eigensolver can
resolve Im{f}; with `Tol: 1e-5` the trustworthy ceiling is roughly
Q ~ 1/(2·Tol) ≈ 5e6 (tighten Tol to 1e-7 for Q studies). A mode reporting
Q = 1e8, or a tiny *negative* Im{f}, is an essentially lossless mode whose
imaginary part is solver noise — the sign and magnitude are not physics.
The tell: noise-floor Qs swing wildly between AMR passes while everything
physical converges. Both HFSS and Palace do this; their noise just differs.
To compare "all Qs" between codes, scale the losses up identically in both
(Rs ×100, σ ÷ a few decades) so every mode's Q lands in the resolvable
range, verify agreement there, and argue by linearity.

**AMR is the convergence referee.** Watch each mode's frequency and Q
across passes; a value that flattens is converged, one still moving needs
more mesh (or more passes). This is stronger evidence than agreement with
an HFSS run, whose own convergence contract (`MaxDeltaFreq`, often 10 %,
frequency-only, no Q criterion) is recorded in `device_config.json` under
`ansys_setups` — check it before treating an HFSS number as truth.

**Mode pairing**: in crowded parts of the spectrum, pair modes between
codes by *field pattern* (ParaView vs HFSS plots), not by index.

**Memory rule of thumb** (this cluster, SuperLU direct): ≈ **60 KB per ND
unknown**, and unknowns ≈ 6.5 × tets at Order 2. A 250 GB node therefore
supports ~4M unknowns ≈ 600k tets *for a single solve* — but AMR grows the
mesh each pass, so **start near 150–200k tets (~1.3M unknowns)** to leave
room for 3 passes. Set `Refinement.MaxSize` ≈ (mem_gb / 0.06) with margin;
note it is checked *before* refining, so the last pass can overshoot it.

---

## 5. Field plots (ParaView)

Add `"Save": 5` inside `Solver.Eigenmode` to save the first 5 modes'
fields. Palace writes `postpro/paraview/eigenmode/eigenmode.pvd` plus data
subfolders — copy the **whole** `paraview/` folder (tar it first; it's
thousands of small files). In ParaView: open the `.pvd`, **Apply**; modes
are the *time steps*; color by `E_real` → Magnitude, rescale per mode; add
a Slice to see inside; use a log color map; set Nonlinear Subdivision 2–3
so curvilinear elements render curved. The mesher's `.vtk` output contains
the mesh only — no fields.

---

## 6. Troubleshooting (all of these actually happened)

### 6.1 GMRES stalls for hours, reduction factor ≈ 0.97
The direct-solver block isn't active (old config?). Check for `Operator
assembly level: Full` and the single-level hierarchy. Fix: the Solver block
in §3.7.

### 6.2 `MFEM abort: Unable to open metadata file "postpro/palace.json"`
Either `SaveAdaptIterations: true` (its directory rotation orphans the
metadata file on the final pass — keep it **false**), or a leftover
`postpro/` from a previous crashed run (then it dies at pass 1). Fix:
`rm -rf postpro` and resubmit — the generated PBS does this automatically.

### 6.3 Killed with exit 137 / `cgroup/OOM`
The factorization outgrew the job's memory, usually on a late AMR pass.
Fix: smaller starting mesh (§4 memory rule), lower `Refinement.MaxSize`,
or more memory in the PBS.

### 6.4 Meshing "never finishes, millions of nodes" after adding a fine feature
Historical bug, fixed: grading fields used to impose their `250·size`
ceiling over the whole domain (min-combination), which only bites when
`250 × finest_size < bulk_size` — i.e. the first µm-scale feature. If you
ever see it again, confirm `StopAtDistMax` is set in
`add_grading_field()` and that you're not running a stale `mesh_any.py`.

### 6.5 Mesh is correct but huge
Run `mesh_stats.py` and look at the **median** edge length and the centroid
clusters — they tell you which feature owns the tets. Remember every size
costs ~1/h³ in its `10·s` slab: thin leads at 0.2 µm and pads at 50 µm are
the usual suspects. Coarsen the over-resolved feature and let AMR spend the
saved budget adaptively.

### 6.6 `Could not match HFSS impedance boundary ...` / `Could not uniquely recover conductor ...`
The matcher refusing to guess. Usually means the boundary sits on a face
the vacuum doesn't touch (move it to a face it does), or a genuinely
ambiguous fragment set — the error lists the candidate surfaces with
centers and areas; compare against the geometry.

### 6.7 A body meshed with a "wrong" size
Read the mesher's size printout — it names the source of every size
(operation / stats / fallback / CLI override). The last line for a body
wins. `mesh_sizes_mm` in the config is a legacy block the mesher ignores.

---

## 7. Mesh sweeps

One config, one variable per run, no file editing:

```bash
python mesh_any.py --tag base
python mesh_any.py --tag Pin1_0p2 --size Pin_1=0.2
python mesh_any.py --tag pads0p1 --size pads=0.1
python write_palace_config.py --groups device_groups_Pin1_0p2.json \
       --output palace_config_Pin1_0p2.json
```

Each tag gets its own mesh/groups/config; run each Palace job in its own
folder (or with its own `Output`) so `postpro/` never collides. Put the
value in the tag (`Pin1_0p2`, not `run3`). This is also how to turn the
lab's mesh-size table — whose entries currently span 5–50× between users —
into evidence: sweep one component on a representative design and let the
AMR trajectories adjudicate what resolution the physics actually needs.

---

## 8. Provenance / verified against

Palace `v0.17.0-67-g9b9d6524d`, built with SuperLU_DIST, OpenMPI 4.1.8,
36 ranks/node. pyAEDT with HFSS 2024. gmsh (pip) with OCC. Validated on:
copper-box analytic tests, cavity+pin designs (cylindrical and
rectangular pins, curvilinear), cavity+chip+pin, and a full transmon
(JJ + pads + leads + chip) with derived LumpedPort. HFSS cross-checks:
frequencies <1 % on all comparable modes; resolvable Qs within 5 %.

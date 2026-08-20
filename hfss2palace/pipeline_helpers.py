"""
pipeline_helpers.py

Support functions for palace_pipeline.ipynb -- the one-notebook
Ansys -> Palace workflow. Keeps the notebook cells thin; all logic that
deserves testing lives here.

Nothing in this module talks to Ansys. It operates on the run folder
that export_for_palace.main() created (device.step + device_config.json
+ the pipeline scripts).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys


# ---------------------------------------------------------------------
# size review
# ---------------------------------------------------------------------

def _resolve_sizes(config):
    """Replicate mesh_any's sizing tiers (operation > HFSS stats >
    'fallback at mesh time') and return {body: (size_mm|None, source)}."""
    sizes = {}

    # tier 1: mesh operations, min per body
    import ast as _ast
    for op_name, op in (config.get("mesh_operations") or {}).items():
        s = op.get("size_mm")
        if s is None:
            continue
        assignment = op.get("assignment") or []
        if isinstance(assignment, str):
            try:
                assignment = _ast.literal_eval(assignment)
            except (ValueError, SyntaxError):
                assignment = [assignment]
        if not isinstance(assignment, (list, tuple)):
            assignment = [assignment]
        for body in [str(b) for b in assignment]:
            prev = sizes.get(body)
            new = float(s) if prev is None else min(prev[0], float(s))
            sizes[body] = (new, f"operation '{op_name}'")

    # tier 2: HFSS mesh stats (most-refined setup wins)
    stats = config.get("ansys_mesh_stats") or {}
    best, best_n, best_setup = None, -1, None
    for setup, data in stats.items():
        bodies = (data or {}).get("bodies") or {}
        n = sum(int(b.get("num_tets") or 0) for b in bodies.values())
        if n > best_n:
            best, best_n, best_setup = bodies, n, setup
    if best:
        for body, rec in best.items():
            if body in sizes:
                continue
            rms = rec.get("rms_edge_mm")
            if rms and float(rms) > 0:
                sizes[body] = (float(rms),
                               f"HFSS mesh stats ({best_setup} RMS edge)")

    # remaining model bodies -> decided by the mesher's analytic fallback
    for name, obj in (config.get("objects") or {}).items():
        if obj.get("model") and name not in sizes:
            sizes[name] = (None, "analytic fallback (set at mesh time)")

    return sizes


def _database_hint(body_name, database):
    """Loose display-only match of a body name against the lab database
    categories. NEVER used to set a size automatically."""
    if not database:
        return ""
    name = body_name.lower()
    aliases = {
        "JJ": ("jj", "junction"),
        "qubit_pads": ("pad",),
        "thin_lead": ("thin",),
        "coarse_lead": ("coarse", "medium"),
        "resonator": ("rr", "resonator"),
        "transmission_lines": ("tl", "transmission", "feedline"),
        "cavity_pin": ("pin",),
        "hairpin": ("hairpin",),
    }
    for category, keys in aliases.items():
        if any(k in name for k in keys):
            rng = (database.get("ranges_observed") or {}).get(category)
            std = (database.get("standard") or {}).get(category)
            parts = []
            if std is not None:
                parts.append(f"standard {std:g}")
            if rng:
                parts.append(f"lab range {rng[0]:g}-{rng[1]:g}")
            if parts:
                return f"[{category}: " + ", ".join(parts) + "] mm"
    return ""


def size_report(run_dir, database_path=None):
    """Print a table of every body: role, resolved base size + source,
    and (display-only) lab-database hints. Returns the resolved dict."""
    config_path = os.path.join(run_dir, "device_config.json")
    with open(config_path, encoding="utf-8") as fh:
        config = json.load(fh)

    database = None
    if database_path and os.path.isfile(database_path):
        with open(database_path, encoding="utf-8") as fh:
            database = json.load(fh)

    sizes = _resolve_sizes(config)
    roles = config.get("body_roles") or {}

    print(f"{'body':<16}{'role':<20}{'base size':<12}source")
    print("-" * 78)
    for body in sorted(sizes):
        size, source = sizes[body]
        size_text = f"{size:g} mm" if size is not None else "(auto)"
        hint = _database_hint(body, database)
        print(f"{body:<16}{str(roles.get(body, '?')):<20}"
              f"{size_text:<12}{source}")
        if hint:
            print(f"{'':<16}{'':<20}{'':<12}{hint}")
    print("-" * 78)
    print("To change a size for THIS run: add it to SIZES in the next "
          "cell\n(units mm; e.g. 0.0002 = 0.2 um). Leave SIZES empty to "
          "accept the table above.")
    return {b: s for b, (s, _) in sizes.items()}


# ---------------------------------------------------------------------
# HFSS mesh-op import (aedt_extract) -> SIZES
# ---------------------------------------------------------------------

def merge_hfss_sizes(run_dir, sizes=None, aedt_path=None, design=None,
                     exclude=()):
    """Pull the ENABLED length-based mesh operations out of an .aedt
    project file (via aedt_extract, pure text -- no Ansys session) and
    merge them into the notebook's SIZES dict. Returns the merged dict;
    the input dict is not modified.

    Precedence, most binding first:
      1. your explicit SIZES entries        (never touched)
      2. junction-role bodies               (never touched -- the jj
                                             port sheet keeps its config
                                             op; a coarse HFSS op there
                                             would wreck the port)
      3. HFSS length ops from the .aedt     (this function)
      4. config mesh_operations / stats / fallbacks (mesh_any tiers)

    Where the exported device_config.json ALREADY carries the same op
    at the same value (the exporter captures live mesh ops), the entry
    is skipped as redundant -- mesh_any will apply it from the config
    anyway. Bodies named in the .aedt but absent from this export are
    dropped with a warning instead of crashing mesh_any (--size on an
    unknown body is a hard error by design).

    aedt_path=None: auto-locate -- newest *.aedt in the run folder's
    parent (the exporter creates run folders next to the project).
    design=None: sole design in the file, else first (with a printed
    note); pass the notebook's DESIGN_NAME to pin it.
    exclude: extra body names to leave alone.

    Physical caveat, read before trusting the merge: HFSS length ops
    seed an ADAPTIVE solve -- HFSS refines beyond them where needed.
    Palace AMR plays that role here, but the GPU budget is the binding
    constraint: after merging, the mesher's "GPU / AMR budget" block is
    still the gate. An HFSS op tuned for a 500-core CPU box can be too
    fine for the 7M-unknown cap."""
    import glob as _glob

    import aedt_extract

    if aedt_path is None:
        parent = os.path.dirname(os.path.abspath(run_dir))
        cands = sorted(_glob.glob(os.path.join(parent, "*.aedt")),
                       key=os.path.getmtime, reverse=True)
        if not cands:
            raise RuntimeError(
                f"no .aedt found next to {run_dir}; pass "
                f"aedt_path= explicitly (AEDT_PATH in cell 1)")
        aedt_path = cands[0]
        if len(cands) > 1:
            print(f"multiple .aedt files found; using newest: "
                  f"{os.path.basename(aedt_path)}")

    print(f"Reading HFSS mesh operations from "
          f"{os.path.basename(aedt_path)} ...")
    dname, ops = aedt_extract.extract_length_sizes(aedt_path,
                                                   design=design)
    print(f"design {dname!r}: {len(ops)} body(ies) under enabled "
          f"length-based ops")

    config_path = os.path.join(run_dir, "device_config.json")
    with open(config_path, encoding="utf-8") as fh:
        config = json.load(fh)
    known = set(config.get("objects") or {})
    roles = config.get("body_roles") or {}
    config_ops = _resolve_sizes(config)   # {body: (mm|None, source)}

    merged = dict(sizes or {})
    rows = []
    for body, rec in ops.items():
        mm, opname = rec["mm"], rec["op"]
        if body in (sizes or {}):
            rows.append((body, mm, opname,
                         f"SKIP: your SIZES[{body!r}]="
                         f"{merged[body]:g} wins"))
        elif body in exclude:
            rows.append((body, mm, opname, "SKIP: in exclude list"))
        elif body not in known:
            rows.append((body, mm, opname,
                         "SKIP: not a body in this export "
                         "(renamed/removed?)"))
        elif roles.get(body) == "junction":
            rows.append((body, mm, opname,
                         "SKIP: junction body keeps its config op"))
        else:
            have = config_ops.get(body)
            if (have and have[0] is not None
                    and abs(have[0] - mm) < 1e-9
                    and have[1].startswith("operation")):
                rows.append((body, mm, opname,
                             f"already in config ({have[1]}) -- "
                             f"mesh_any applies it"))
            else:
                merged[body] = mm
                note = "APPLIED"
                if have and have[0] is not None:
                    note += (f" (config had {have[0]:g} mm from "
                             f"{have[1]})")
                rows.append((body, mm, opname, note))

    print(f"\n{'body':<26}{'HFSS mm':<12}{'HFSS op':<20}action")
    print("-" * 78)
    for body, mm, opname, note in rows:
        print(f"{body:<26}{mm:<12g}{opname:<20}{note}")
    print("-" * 78)
    applied = sum(1 for r in rows if r[3].startswith("APPLIED"))
    print(f"{applied} size(s) merged into SIZES. The budget block in "
          f"the mesh cell is still the gate -- read it before qsub.")
    return merged


# ---------------------------------------------------------------------
# subprocess steps (mesher / writer / stats), run inside the run folder
# ---------------------------------------------------------------------

def _run(cmd, cwd):
    """Run a pipeline step, STREAMING its output live (a silent cell
    during a 10-minute mesh is indistinguishable from a hang)."""
    print("$ " + " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            bufsize=1)
    lines = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(
            f"step failed (exit {proc.returncode}): {' '.join(cmd)}")
    return "".join(lines)


def run_mesher(run_dir, tag, sizes=None, mesh_order=None,
               curvature_segments=None, grading=True, scale=None,
               threads=None, algo3d=None, vtk=True, boolean_tol=None):
    """Run mesh_any.py in the run folder. `sizes` is {body: mm} applied
    as --size overrides. `scale` multiplies EVERY resolved body size
    uniformly (junction bodies and explicit `sizes` entries exempt);
    element count in fine regions falls like 1/scale^3, so scale=1.5
    cuts a fine-dominated mesh to ~30% of its count. `boolean_tol`
    (mm) turns on OCC fuzzy booleans: geometry closer than this is
    treated as coincident during fragment, which repairs crossing
    edges that otherwise kill 3D meshing with 'PLC Error: ...
    intersect at point' (use 1e-4 = 0.1 um; stay below the smallest
    real gap). Requires the current mesh_any.py in the run folder.
    Returns (msh, groups) filenames."""
    cmd = [sys.executable, "mesh_any.py",
           "--config", "device_config.json", "--tag", tag]
    for body, mm in (sizes or {}).items():
        cmd += ["--size", f"{body}={mm}"]
    if scale is not None and float(scale) != 1.0:
        cmd += ["--scale", str(scale)]
    if mesh_order is not None:
        cmd += ["--mesh-order", str(mesh_order)]
    if curvature_segments is not None:
        cmd += ["--curvature-segments", str(curvature_segments)]
    if not grading:
        cmd += ["--no-grading"]
    if threads is not None:
        cmd += ["--threads", str(threads)]
    if algo3d is not None:
        cmd += ["--algo3d", algo3d]
    if not vtk:
        cmd += ["--no-vtk"]
    if boolean_tol is not None and float(boolean_tol) > 0:
        cmd += ["--boolean-tol", str(boolean_tol)]
    _run(cmd, run_dir)
    return f"device_{tag}.msh", f"device_groups_{tag}.json"


def apply_solver_overrides(run_dir, palace):
    """Stamp the notebook's PALACE dict into the run folder's
    device_config.json (palace_solver block) so the config writer sees
    the CURRENT notebook values even when the export (which normally
    embeds them) happened before you last edited PALACE. Call this
    right before run_writer -- it makes 'edit PALACE, rerun cell 9'
    work without a re-export."""
    path = os.path.join(run_dir, "device_config.json")
    with open(path, encoding="utf-8") as fh:
        dc = json.load(fh)
    merged = dict(dc.get("palace_solver") or {})
    merged.update({k: v for k, v in (palace or {}).items()
                   if k != "device"})   # device picks solver profile via
                                        # run_writer, not the config
    dc["palace_solver"] = merged
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(dc, fh, indent=2)
    print(f"palace_solver refreshed from notebook PALACE: "
          f"{sorted(merged)}")


def run_mesh_stats(run_dir, msh):
    if os.path.isfile(os.path.join(run_dir, "mesh_stats.py")):
        _run([sys.executable, "mesh_stats.py", msh], run_dir)
    else:
        print("(mesh_stats.py not in run folder; skipped)")


def run_conformity_check(run_dir, msh):
    """Verify every surface triangle in the mesh is a face of a tet.
    Palace/MFEM hard-requires this and dies with the cryptic
    'MFEM abort ... STable3D' otherwise -- 4 wasted seconds on the
    cluster vs 4 free seconds locally. Raises on a broken mesh so the
    notebook stops BEFORE the qsub cells."""
    script = os.path.join(run_dir, "mesh_conformity_check.py")
    if not os.path.isfile(script):
        print("(mesh_conformity_check.py not in run folder; skipped -- "
              "copy it in: this is the check that catches the MFEM "
              "STable3D abort before you burn a queue slot)")
        return
    result = subprocess.run([sys.executable, "mesh_conformity_check.py",
                             msh], cwd=run_dir)
    if result.returncode != 0:
        raise RuntimeError(
            f"mesh {msh} is NON-CONFORMAL -- Palace will abort on it. "
            f"See the orphan-triangle report above for which surfaces "
            f"and where. Do not submit; remesh first.")


def run_writer(run_dir, groups, out_config, mesh=None, postpro_dir=None,
               device="cpu"):
    """Generate the Palace config; `mesh` is the tagged .msh filename
    (the writer defaults to device.msh, which doesn't exist for tagged
    runs). If postpro_dir is given, point Problem.Output at it so every
    run's results live in their own folder (postpro_<tag>) and never
    collide or poison each other. device="gpu" selects the validated
    iterative/partial-assembly GPU solver profile (pair with
    write_pbs(gpu=True))."""
    cmd = [sys.executable, "write_palace_config.py",
           "--groups", groups,
           "--device-config", "device_config.json",
           "--output", out_config,
           "--device", device]
    if mesh:
        cmd += ["--mesh", mesh]
    _run(cmd, run_dir)
    if postpro_dir:
        path = os.path.join(run_dir, out_config)
        with open(path, encoding="utf-8") as fh:
            pc = json.load(fh)
        pc.setdefault("Problem", {})["Output"] = postpro_dir
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(pc, fh, indent=2)
        print(f"Problem.Output -> {postpro_dir}")
    return out_config


# ---------------------------------------------------------------------
# PBS file + final checklist
# ---------------------------------------------------------------------

# Mirrors the lab's validated run_palace.pbs (modules, binding, OMP
# pinning), parameterized, plus the crashed-run postpro guard. Works on
# 1 node or several: mem= is PER NODE, and the --prefix / -x flags let
# mpirun start orted on remote nodes where module loads don't apply.
PBS_TEMPLATE = """#!/bin/bash
#PBS -r n
#PBS -N {jobname}
#PBS -l select={nodes}:ncpus={ncpus}:mpiprocs={ncpus}:ompthreads=1:mem={mem_gb}gb
#PBS -l walltime={walltime}
#PBS -j oe

set -uo pipefail
cd "$PBS_O_WORKDIR" || exit 1

module purge
module load gcc11/11.3.0
module load openblas/dynamic/0.3.18
module load openmpi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

NCPUS=$(wc -l < "$PBS_NODEFILE")
CONFIG=${{1:-{config}}}
POSTPRO={postpro_dir}

echo "PBS_JOBID = $PBS_JOBID"
echo "NCPUS     = $NCPUS"
echo "CONFIG    = $CONFIG"
mpirun --version | head -1

# Where the module's OpenMPI lives -- needed so remote nodes can start
# orted (module loads do NOT propagate to other nodes over ssh).
MPI_ROOT=$(dirname "$(dirname "$(command -v mpirun)")")
echo "MPI_ROOT  = $MPI_ROOT"

# A leftover output folder from a crashed run makes the next run abort
# at its first postprocessing step (missing palace.json). Always start
# clean; results you want to keep are safe -- each run has its own
# postpro_<tag> folder.
rm -rf "$POSTPRO"

/usr/bin/time -v mpirun --prefix "$MPI_ROOT" \\
  -x PATH -x LD_LIBRARY_PATH \\
  -np "$NCPUS" -hostfile "$PBS_NODEFILE" \\
  --bind-to core --map-by core --report-bindings \\
  {palace_bin} \\
  "$CONFIG"
"""


# GPU variant: small request (memory-follows-cores gives ncpus x ~7 GB
# host RAM -- the iterative solver needs little), ngpus routes billing
# to the GPU-hours pool, ranks share the GPUs. Pair with a config
# written by `write_palace_config.py --device gpu`.
GPU_PBS_TEMPLATE = """#!/bin/bash
#PBS -r n
#PBS -N {jobname}
#PBS -q auto_free
#PBS -l select=1:ncpus={ncpus}:ngpus={ngpus}:mem={mem_gb}gb
#PBS -l walltime={walltime}
#PBS -j oe
#PBS -k oed

set -uo pipefail
cd "$PBS_O_WORKDIR" || exit 1
export TERM=xterm

module purge
module load gcc11/11.3.0
module load openblas/dynamic/0.3.18
module load openmpi
module load cuda12.2/toolkit/12.2.2

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

CONFIG=${{1:-{config}}}
POSTPRO={postpro_dir}

echo "PBS_JOBID = $PBS_JOBID"
echo "CONFIG    = $CONFIG"
nvidia-smi -L
mpirun --version | head -1

# A leftover output folder from a crashed run makes the next run abort
# at its first postprocessing step (missing palace.json). Always start
# clean; results you want to keep are safe -- each run has its own
# postpro_<tag> folder.
rm -rf "$POSTPRO"

/usr/bin/time -v mpirun -np {ranks} \\
  {palace_bin} \\
  "$CONFIG"
"""


def write_pbs(run_dir, config, jobname="palace", nodes=1, ncpus=72,
              mem_gb=500, walltime="06:00:00", postpro_dir="postpro",
              palace_bin=None, gpu=False, ngpus=2, ranks=8):
    """Write run_palace.pbs into the run folder.

    gpu=False (default): the validated CPU/direct-solver launch. Vanda
    compute nodes are 72 cores / ~501 GiB (confirmed via `pbsnodes`);
    memory follows cores at ~7 GB/core regardless of mem_gb, so size
    ncpus*nodes to the factorization (~60 KB/unknown).

    gpu=True: single-node GPU launch for the iterative solver profile
    (`write_palace_config.py --device gpu`), submitted via the
    auto_free ROUTING queue (direct submission to the `gpu` execution
    queue is denied -- it is route-only; auto_free routes GPU chunks
    there at charge rate 0): `ranks` MPI ranks sharing `ngpus` A40s. Host memory is real:
    the Vostok-2 p=1 AMR run peaked at 137 GB host-side, so mem_gb is
    clamped to the queue's 240 GB default rather than the old 56 GB
    guess. VRAM ceiling ~8M unknowns per 2-GPU node; keep the config's
    AMR MaxSize <= 7M (or AMR off for a mesh at the ceiling).

    Newlines forced to \\n so the file survives Windows -> Linux."""
    if gpu:
        if palace_bin is None:
            palace_bin = "$HOME/palace_build_cuda/bin/palace-x86_64.bin"
        text = GPU_PBS_TEMPLATE.format(
            jobname=jobname, ncpus=ncpus if ncpus <= 16 else 8,
            ngpus=ngpus, mem_gb=min(mem_gb, 240), walltime=walltime,
            config=config, palace_bin=palace_bin,
            postpro_dir=postpro_dir, ranks=ranks)
    else:
        if palace_bin is None:
            palace_bin = "$HOME/palace_build_openmpi/bin/palace-x86_64.bin"
        text = PBS_TEMPLATE.format(jobname=jobname, nodes=nodes,
                                   ncpus=ncpus, mem_gb=mem_gb,
                                   walltime=walltime, config=config,
                                   palace_bin=palace_bin,
                                   postpro_dir=postpro_dir)
    path = os.path.join(run_dir, "run_palace.pbs")
    with open(path, "w", newline="\n") as fh:
        fh.write(text)
    print(f"written: {path}")
    return path


def final_checklist(run_dir, msh, palace_config):
    """Verify the folder is HPC-complete and print the two commands the
    user runs on the cluster. Fails loudly on anything missing."""
    required = [msh, palace_config, "run_palace.pbs"]
    missing = [f for f in required
               if not os.path.isfile(os.path.join(run_dir, f))]
    if missing:
        raise RuntimeError(f"run folder incomplete, missing: {missing}")

    # the palace config must reference the mesh that exists
    with open(os.path.join(run_dir, palace_config), encoding="utf-8") as fh:
        pc = json.load(fh)
    mesh_in_config = pc.get("Model", {}).get("Mesh")
    if mesh_in_config != msh:
        raise RuntimeError(
            f"palace config points at mesh '{mesh_in_config}' but the "
            f"meshed file is '{msh}' -- regenerate the config")
    # and the pbs must reference the palace config
    with open(os.path.join(run_dir, "run_palace.pbs"),
              encoding="utf-8") as fh:
        if palace_config not in fh.read():
            raise RuntimeError(
                f"run_palace.pbs does not reference {palace_config} -- "
                f"regenerate it")

    sizes = {f: os.path.getsize(os.path.join(run_dir, f)) / 1e6
             for f in required}
    print("Run folder is complete:")
    print(f"  {run_dir}")
    for f, mb in sizes.items():
        print(f"    {f:<40}{mb:8.1f} MB")
    print("\nOn the HPC (after copying the folder over):")
    print(f"  cd <copied folder>")
    print(f"  qsub run_palace.pbs")
    print("\nResults land in <folder>/postpro/ (eig.csv + the log).")
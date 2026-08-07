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
# subprocess steps (mesher / writer / stats), run inside the run folder
# ---------------------------------------------------------------------

def _run(cmd, cwd):
    print("$ " + " ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(
            f"step failed (exit {result.returncode}): {' '.join(cmd)}")
    if result.stderr.strip():
        print(result.stderr)
    return result.stdout


def run_mesher(run_dir, tag, sizes=None, mesh_order=None,
               curvature_segments=None, grading=True):
    """Run mesh_any.py in the run folder. `sizes` is {body: mm} applied
    as --size overrides. Returns (msh, groups) filenames."""
    cmd = [sys.executable, "mesh_any.py",
           "--config", "device_config.json", "--tag", tag]
    for body, mm in (sizes or {}).items():
        cmd += ["--size", f"{body}={mm}"]
    if mesh_order is not None:
        cmd += ["--mesh-order", str(mesh_order)]
    if curvature_segments is not None:
        cmd += ["--curvature-segments", str(curvature_segments)]
    if not grading:
        cmd += ["--no-grading"]
    _run(cmd, run_dir)
    return f"device_{tag}.msh", f"device_groups_{tag}.json"


def run_mesh_stats(run_dir, msh):
    if os.path.isfile(os.path.join(run_dir, "mesh_stats.py")):
        _run([sys.executable, "mesh_stats.py", msh], run_dir)
    else:
        print("(mesh_stats.py not in run folder; skipped)")


def run_writer(run_dir, groups, out_config, postpro_dir=None):
    """Generate the Palace config; if postpro_dir is given, point
    Problem.Output at it so every run's results live in their own
    folder (postpro_<tag>) and never collide or poison each other."""
    cmd = [sys.executable, "write_palace_config.py",
           "--groups", groups,
           "--device-config", "device_config.json",
           "--output", out_config]
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
# pinning), parameterized, plus the crashed-run postpro guard.
PBS_TEMPLATE = """#!/bin/bash
#PBS -r n
#PBS -N {jobname}
#PBS -l select=1:ncpus={ncpus}:mpiprocs={ncpus}:ompthreads=1:mem={mem_gb}gb
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

echo "PBS_JOBID = $PBS_JOBID"
echo "NCPUS     = $NCPUS"
echo "CONFIG    = $CONFIG"
mpirun --version | head -1

# A leftover output folder from a crashed run makes the next run abort
# at its first postprocessing step (missing palace.json). Always start
# clean; results you want to keep are safe -- each run has its own
# postpro_<tag> folder.
rm -rf {postpro_dir}

/usr/bin/time -v mpirun -np "$NCPUS" -hostfile "$PBS_NODEFILE" \\
  --bind-to core --map-by core --report-bindings \\
  {palace_bin} \\
  "$CONFIG"
"""


def write_pbs(run_dir, config, jobname="palace", ncpus=36, mem_gb=250,
              walltime="06:00:00", postpro_dir="postpro",
              palace_bin="~/palace_build_openmpi/bin/palace-x86_64.bin"):
    """Write run_palace.pbs into the run folder. Newlines forced to \\n
    so the file survives the Windows -> Linux trip."""
    text = PBS_TEMPLATE.format(jobname=jobname, ncpus=ncpus,
                               mem_gb=mem_gb, walltime=walltime,
                               config=config, palace_bin=palace_bin,
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

#!/usr/bin/env python3
"""
pipeline_state.py

Persist the handful of notebook variables that point at real files, so a
kernel restart does not mean re-exporting from HFSS.

Everything the pipeline produces after the export already lives on disk:
device.step, device_config.json, device_<tag>.msh, device_groups_<tag>.json,
palace_config_<tag>.json. What a restart destroys is only the Python names
pointing at them -- RUN_DIR above all, because recovering it any other way
means running the exporter again, which needs HFSS open and creates a fresh
timestamped folder you did not want.

Two files are written:

  <run_dir>/notebook_state.json   the state itself, so a run folder stays
                                  self-contained and can be archived or
                                  copied to another machine as-is
  <notebook_dir>/.last_run        a pointer to the newest run folder, so
                                  resume() needs no arguments

The state file is also a provenance record. It says which sizes produced
which mesh, which is exactly the question you cannot answer later from the
mesh file alone.

Usage in the notebook:

    import pipeline_state as ps

    ps.save(RUN_DIR, tag=TAG, sizes=SIZES)          # after each step
    state = ps.resume()                             # after a restart
    RUN_DIR, TAG, SIZES = state["run_dir"], state["tag"], state["sizes"]
"""

from __future__ import annotations

import datetime
import json
import os

STATE_NAME = "notebook_state.json"
POINTER_NAME = ".last_run"
SCHEMA = 1


def _pointer_path(notebook_dir=None):
    return os.path.join(notebook_dir or os.getcwd(), POINTER_NAME)


def state_path(run_dir):
    return os.path.join(run_dir, STATE_NAME)


def save(run_dir, notebook_dir=None, **fields):
    """
    Merge `fields` into the run folder's state file and update the pointer.

    Merging rather than overwriting means each step can save only what it
    produced, and re-running one cell does not wipe the others. Pass a
    value of None to clear a field: that matters after a re-mesh, when the
    old config no longer matches the new mesh and should not be left
    behind looking valid.
    """
    path = state_path(run_dir)
    state = {}
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as fh:
                state = json.load(fh)
        except (OSError, ValueError) as error:
            print("pipeline_state: could not read existing state "
                  "({0}); starting a new one".format(error))
            state = {}

    state["schema"] = SCHEMA
    state["run_dir"] = run_dir
    state["saved_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    for key, value in fields.items():
        if value is None:
            state.pop(key, None)
        else:
            state[key] = value

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)

    try:
        with open(_pointer_path(notebook_dir), "w", encoding="utf-8") as fh:
            fh.write(run_dir)
    except OSError as error:
        print("pipeline_state: state saved but the pointer could not be "
              "written ({0}); resume() will need an explicit run_dir"
              .format(error))

    return state


def last_run_dir(notebook_dir=None):
    """The run folder most recently saved from this notebook folder."""
    path = _pointer_path(notebook_dir)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            run_dir = fh.read().strip()
    except OSError:
        return None
    return run_dir or None


def load(run_dir=None, notebook_dir=None):
    """Read a state file. Returns {} if there is nothing to read."""
    if run_dir is None:
        run_dir = last_run_dir(notebook_dir)
    if not run_dir:
        return {}
    path = state_path(run_dir)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError) as error:
        print("pipeline_state: could not read {0}: {1}".format(path, error))
        return {}


def resume(run_dir=None, notebook_dir=None, verbose=True):
    """
    Reload a previous run's state and report what is actually still there.

    Every recorded file is checked on disk rather than trusted, because a
    state file can outlive the files it names: a folder gets copied
    without its mesh, a mesh is deleted to free space, a re-mesh under a
    new tag leaves the old config pointing at a mesh that no longer
    matches. Reporting a missing file is far better than resuming into a
    step that quietly uses the wrong one.
    """
    state = load(run_dir, notebook_dir)
    if not state:
        if verbose:
            print("No saved state found. Run the export cell to start a "
                  "new run, or pass resume(run_dir=r'...') explicitly.")
        return {}

    resolved = state.get("run_dir")
    if verbose:
        print("Resuming: {0}".format(resolved))
        print("  saved at:  {0}".format(state.get("saved_at", "?")))
        print("  tag:       {0}".format(state.get("tag", "?")))
        print("  design:    {0}".format(state.get("design", "?")))

    if not os.path.isdir(resolved or ""):
        print("  WARNING: that folder does not exist any more. Nothing "
              "downstream will work until you re-export or fix the path.")
        return state

    checks = [("mesh", state.get("mesh")),
              ("groups", state.get("groups")),
              ("config", state.get("config"))]
    missing = []
    for label, name in checks:
        if not name:
            if verbose:
                print("  {0:<10} not built yet".format(label + ":"))
            continue
        full = os.path.join(resolved, name)
        if os.path.isfile(full):
            size_mb = os.path.getsize(full) / 1e6
            if verbose:
                print("  {0:<10} {1}  ({2:.1f} MB)".format(
                    label + ":", name, size_mb))
        else:
            missing.append((label, name))
            print("  {0:<10} {1}  MISSING FROM DISK".format(
                label + ":", name))

    sizes = state.get("sizes") or {}
    if verbose:
        print("  sizes:     {0} explicit override(s)".format(len(sizes)))

    if missing:
        print("\n  Re-run the step that produces each missing file. The "
              "state is a record of what was done, not proof the files "
              "survived.")

    return state

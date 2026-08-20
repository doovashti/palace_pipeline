#!/usr/bin/env python3
"""
palace_runlog.py

Run registry for the Ansys -> Palace pipeline: parse a finished (or
crashed) Palace job log, count mesh elements per component, and append
one record to a persistent registry. Regenerates an HTML report with a
sortable table of all runs, per-run mode tables, and an optional design
screenshot per run.

Usage (from the run folder, after copying the job's .o log back):

    python palace_runlog.py add --log palace_eigen.o1289998 \
        [--run-dir .] [--registry ..\\palace_runs] \
        [--screenshot design.png] [--notes "lead 0.5um, pads 50um"]

    python palace_runlog.py report [--registry ..\\palace_runs]

The registry directory holds:
    runs.jsonl   -- one JSON record per run (full detail, append-only)
    runs.csv     -- flat summary table (opens in Excel)
    runlog.html  -- the report (regenerated on every add/report)
    img/         -- copied screenshots
"""

from __future__ import annotations

import argparse
import base64
import csv
import datetime
import glob
import html
import json
import os
import re
import shutil
import sys


# ---------------------------------------------------------------------
# log parsing
# ---------------------------------------------------------------------

MODE_ROW = re.compile(
    r"^\s*(\d+),\s*([+-][\d.e+-]+),\s*([+-][\d.e+-]+),\s*([+-][\d.e+-]+),"
    r"\s*([+-][\d.e+-]+),\s*([+-][\d.e+-]+)\s*$")


def _empty_record():
    return {
        "jobid": None, "config": None, "palace_version": None,
        "ranks": None, "curvature_order": None,
        "passes": [], "amr_iterations": 0,
        "wall_clock": None, "wall_clock_s": None,
        "status": "unknown", "errors": [],
    }


def parse_log(text):
    """Extract everything comparable from a Palace PBS log."""
    out = _empty_record()

    m = re.search(r"PBS_JOBID\s*=\s*(\S+)", text)
    if m:
        out["jobid"] = m.group(1).split(".")[0]
    m = re.search(r"CONFIG\s*=\s*(\S+)", text)
    if m:
        out["config"] = m.group(1)
    m = re.search(r"Git changeset ID:\s*(\S+)", text)
    if m:
        out["palace_version"] = m.group(1)
    m = re.search(r"Running with (\d+) MPI processes", text)
    if m:
        out["ranks"] = int(m.group(1))
    m = re.search(r"Mesh curvature order:\s*(\d+)", text)
    if m:
        out["curvature_order"] = int(m.group(1))

    # element totals: every "Parallel Mesh Stats" block has an
    # " elements   min avg max total" row; totals in order of appearance
    element_totals = [int(t) for t in re.findall(
        r"^\s*elements\s+\d+\s+\d+\s+\d+\s+(\d+)\s*$", text, re.M)]
    unknowns = [int(u) for u in re.findall(
        r"ND \(p = \d+\):\s*(\d+)", text)]

    # per-pass mode tables follow "performing postprocessing"
    tables = []
    for chunk in re.split(r"performing postprocessing", text)[1:]:
        rows = []
        for line in chunk.splitlines()[:40]:
            m = MODE_ROW.match(line)
            if m:
                rows.append({
                    "m": int(m.group(1)),
                    "re_GHz": float(m.group(2)),
                    "im_GHz": float(m.group(3)),
                    "Q": float(m.group(4)),
                    "err_bkwd": float(m.group(5)),
                    "err_abs": float(m.group(6)),
                })
        if rows:
            tables.append(rows)

    # per-pass peak memory ("Estimated peak per-node memory usage")
    mems = [float(v) for v in re.findall(
        r"Estimated peak per-node memory usage is:.*?Avg\.\s*([\d.]+)G",
        text)]
    # per-pass cumulative time (last "Total  <s>" of each timing block)
    times = [float(v) for v in re.findall(
        r"^Total\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s*$", text, re.M)]

    for i, rows in enumerate(tables):
        out["passes"].append({
            "pass": i + 1,
            "tets": element_totals[i] if i < len(element_totals) else None,
            "nd_unknowns": unknowns[i] if i < len(unknowns) else None,
            "modes": rows,
            "peak_mem_gb": mems[i] if i < len(mems) else None,
            "cum_time_s": times[i] if i < len(times) else None,
        })

    out["amr_iterations"] = len(re.findall(
        r"Adaptive mesh refinement \(AMR\) iteration \d+:", text))

    m = re.search(r"Elapsed \(wall clock\) time.*?:\s*([\d:.]+)", text)
    if m:
        out["wall_clock"] = m.group(1)
        parts = [float(p) for p in m.group(1).split(":")]
        s = 0.0
        for p in parts:
            s = s * 60 + p
        out["wall_clock_s"] = round(s, 1)

    # how did it end?
    if "cgroup/OOM" in text or "signal 9 (Killed)" in text:
        out["status"] = "OOM_KILLED"
        out["errors"].append("killed by memory limit (cgroup OOM)")
    elif "MFEM abort" in text:
        out["status"] = "MFEM_ABORT"
        m = re.search(r"MFEM abort: (.*)", text)
        if m:
            out["errors"].append(m.group(1).strip())
    elif "PBS: job killed: walltime" in text:
        out["status"] = "WALLTIME"
        out["errors"].append("killed by walltime limit")
    elif re.search(r"Completed \d+ iterations? of adaptive mesh", text) \
            or (tables and "Elapsed Time Report" in text
                and "Exit status: 0" in text):
        out["status"] = "COMPLETED"
    elif tables:
        out["status"] = "PARTIAL"

    m = re.search(r"Completed (\d+) iterations? of adaptive mesh", text)
    if m and int(m.group(1)) == 0 and out["status"] == "COMPLETED":
        out["errors"].append("AMR did 0 refinements (MaxSize reached "
                             "at start?)")
    return out


# ---------------------------------------------------------------------
# per-component element counts from the .msh (no gmsh needed)
# ---------------------------------------------------------------------

TET_TYPES = {4, 11}       # 4-node and 10-node tets
TRI_TYPES = {2, 9}        # 3- and 6-node triangles


def count_elements_per_group(msh_path, groups_path):
    """Return {group_name: {'tets': n} or {'tris': n}} using the msh2.2
    physical tags and the mesher's groups JSON."""
    with open(groups_path, encoding="utf-8") as fh:
        groups = json.load(fh)
    name_by_attr = {v: k for k, v in groups.items()
                    if k != "_meta" and isinstance(v, int)}

    counts = {}
    with open(msh_path, encoding="utf-8", errors="replace") as fh:
        in_elements = False
        for line in fh:
            line = line.strip()
            if line == "$Elements":
                in_elements = True
                next(fh)  # count line
                continue
            if line == "$EndElements":
                break
            if not in_elements:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            etype = int(parts[1])
            ntags = int(parts[2])
            if ntags < 1:
                continue
            phys = int(parts[3])
            name = name_by_attr.get(phys)
            if name is None:
                continue
            slot = counts.setdefault(name, {"tets": 0, "tris": 0})
            if etype in TET_TYPES:
                slot["tets"] += 1
            elif etype in TRI_TYPES:
                slot["tris"] += 1
    return counts


# ---------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------

def parse_eig_csv(path):
    """Palace postpro/eig.csv -> final-pass mode list."""
    modes = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh)
        for row in reader:
            row = [c.strip() for c in row]
            if not row or not row[0] or not row[0][0].isdigit():
                continue
            try:
                modes.append({
                    "m": int(float(row[0])),
                    "re_GHz": float(row[1]),
                    "im_GHz": float(row[2]),
                    "Q": float(row[3]) if len(row) > 3 else None,
                    "err_bkwd": float(row[4]) if len(row) > 4 else None,
                    "err_abs": float(row[5]) if len(row) > 5 else None,
                })
            except (ValueError, IndexError):
                continue
    return modes


def add_run(args):
    run_dir = os.path.abspath(args.run_dir)
    registry = os.path.abspath(args.registry)
    os.makedirs(os.path.join(registry, "img"), exist_ok=True)

    if not args.log and not args.postpro:
        raise SystemExit("give --log (the PBS .o file), --postpro (the "
                         "postpro folder Palace wrote), or both")

    if args.log:
        with open(args.log, encoding="utf-8", errors="replace") as fh:
            record = parse_log(fh.read())
        record["log_file"] = os.path.basename(args.log)
    else:
        record = _empty_record()
        record["log_file"] = None

    record["date"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    record["run_dir"] = run_dir
    record["notes"] = args.notes or ""

    # config: from the log, or the only palace_config*.json in the folder
    if not record.get("config"):
        candidates = sorted(glob.glob(
            os.path.join(run_dir, "palace_config*.json")))
        if len(candidates) == 1:
            record["config"] = os.path.basename(candidates[0])
        elif args.postpro and len(candidates) > 1:
            # pick the config whose Problem.Output matches the postpro dir
            want = os.path.basename(os.path.normpath(args.postpro))
            for c in candidates:
                try:
                    with open(c, encoding="utf-8") as fh:
                        if json.load(fh).get("Problem", {}) \
                                        .get("Output") == want:
                            record["config"] = os.path.basename(c)
                            break
                except (OSError, json.JSONDecodeError):
                    pass

    # design / tag from the run folder's device_config + config name
    cfg_path = os.path.join(run_dir, "device_config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path, encoding="utf-8") as fh:
            record["design"] = json.load(fh).get("design")
    else:
        record["design"] = None

    # locate mesh + groups + postpro dir from the palace config
    record["mesh"] = None
    record["components"] = {}
    postpro_dir = args.postpro
    if record.get("config"):
        pc_path = os.path.join(run_dir, record["config"])
        if os.path.isfile(pc_path):
            with open(pc_path, encoding="utf-8") as fh:
                pc = json.load(fh)
            record["mesh"] = pc.get("Model", {}).get("Mesh")
            record["solver"] = {
                "order": pc.get("Solver", {}).get("Order"),
                "target": pc.get("Solver", {}).get("Eigenmode", {})
                                              .get("Target"),
                "tol": pc.get("Solver", {}).get("Eigenmode", {})
                                           .get("Tol"),
            }
            if postpro_dir is None:
                out_name = pc.get("Problem", {}).get("Output", "postpro")
                candidate = os.path.join(run_dir, out_name)
                if os.path.isdir(candidate):
                    postpro_dir = candidate

    # postpro/eig.csv: the authoritative FINAL modes. If the log gave us
    # per-pass tables too, eig.csv replaces/patches the last pass; with
    # no log at all it becomes the single recorded pass.
    if postpro_dir:
        eig = os.path.join(postpro_dir, "eig.csv")
        if os.path.isfile(eig):
            modes = parse_eig_csv(eig)
            if modes:
                if record["passes"]:
                    record["passes"][-1]["modes"] = modes
                else:
                    record["passes"].append({
                        "pass": 1, "tets": None, "nd_unknowns": None,
                        "modes": modes, "peak_mem_gb": None,
                        "cum_time_s": None})
                if record["status"] == "unknown":
                    record["status"] = "COMPLETED"
                record["postpro"] = os.path.basename(
                    os.path.normpath(postpro_dir))
    if record["mesh"]:
        msh = os.path.join(run_dir, record["mesh"])
        tag_match = re.match(r"device_?(.*)\.msh",
                             os.path.basename(record["mesh"]))
        record["tag"] = tag_match.group(1) if tag_match else ""
        groups_guess = os.path.join(
            run_dir, f"device_groups_{record['tag']}.json") \
            if record["tag"] else os.path.join(run_dir,
                                               "device_groups.json")
        if os.path.isfile(msh) and os.path.isfile(groups_guess):
            record["components"] = count_elements_per_group(msh,
                                                            groups_guess)
    else:
        record["tag"] = ""

    # mesh size provenance: replay the --size overrides recorded by the
    # mesher into the groups file? Simplest honest record: the user note.

    run_id = record["jobid"] or datetime.datetime.now().strftime(
        "%Y%m%d_%H%M%S")
    record["run_id"] = run_id

    # screenshot: explicit flag, else any png/jpg in the run folder
    shot = args.screenshot
    if not shot:
        candidates = sorted(glob.glob(os.path.join(run_dir, "*.png"))
                            + glob.glob(os.path.join(run_dir, "*.jpg")))
        shot = candidates[0] if candidates else None
    if shot and os.path.isfile(shot):
        ext = os.path.splitext(shot)[1].lower()
        dest = os.path.join(registry, "img", f"{run_id}{ext}")
        shutil.copy2(shot, dest)
        record["screenshot"] = os.path.relpath(dest, registry)
    else:
        record["screenshot"] = None

    # append to jsonl
    with open(os.path.join(registry, "runs.jsonl"), "a",
              encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")

    _write_csv(registry)
    _write_html(registry)
    final = record["passes"][-1] if record["passes"] else None
    print(f"recorded run {run_id}: status={record['status']}, "
          f"{record['amr_iterations']} AMR its, "
          f"{len(record['passes'])} pass(es)"
          + (f", final tets={final['tets']}" if final else ""))
    print(f"report: {os.path.join(registry, 'runlog.html')}")


def _load_all(registry):
    path = os.path.join(registry, "runs.jsonl")
    if not os.path.isfile(path):
        return []
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_csv(registry):
    records = _load_all(registry)
    path = os.path.join(registry, "runs.csv")
    cols = ["run_id", "date", "design", "tag", "status", "amr_iterations",
            "passes", "tets_initial", "tets_final", "wall_clock",
            "peak_mem_gb", "f1_GHz", "Q1", "f2_GHz", "Q2", "f3_GHz",
            "Q3", "f4_GHz", "Q4", "f5_GHz", "Q5", "notes"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in records:
            p = r.get("passes") or []
            final = p[-1] if p else {}
            modes = {m["m"]: m for m in final.get("modes", [])}
            row = [r.get("run_id"), r.get("date"), r.get("design"),
                   r.get("tag"), r.get("status"),
                   r.get("amr_iterations"), len(p),
                   p[0].get("tets") if p else None,
                   final.get("tets"),
                   r.get("wall_clock"),
                   max((x.get("peak_mem_gb") or 0) for x in p)
                   if p else None]
            for i in range(1, 6):
                mm = modes.get(i)
                row += [mm["re_GHz"] if mm else None,
                        mm["Q"] if mm else None]
            row.append(r.get("notes"))
            w.writerow(row)


def _fmt_q(q):
    if q is None:
        return ""
    return f"{q:.3g}" if (q < 1e4) else f"{q:.2e}"


def _write_html(registry):
    records = _load_all(registry)
    rows = []
    details = []
    for r in reversed(records):   # newest first
        p = r.get("passes") or []
        final = p[-1] if p else {}
        status = r.get("status", "?")
        color = {"COMPLETED": "#0a7a3d", "OOM_KILLED": "#b3261e",
                 "MFEM_ABORT": "#b3261e", "WALLTIME": "#b3261e",
                 "PARTIAL": "#a05a00"}.get(status, "#555")
        rows.append(
            "<tr>"
            f"<td><a href='#run-{html.escape(str(r.get('run_id')))}'>"
            f"{html.escape(str(r.get('run_id')))}</a></td>"
            f"<td>{html.escape(str(r.get('date') or ''))}</td>"
            f"<td>{html.escape(str(r.get('design') or ''))}</td>"
            f"<td>{html.escape(str(r.get('tag') or ''))}</td>"
            f"<td style='color:{color};font-weight:600'>"
            f"{html.escape(status)}</td>"
            f"<td>{r.get('amr_iterations')}</td>"
            f"<td>{p[0].get('tets') if p else ''}</td>"
            f"<td>{final.get('tets') or ''}</td>"
            f"<td>{html.escape(str(r.get('wall_clock') or ''))}</td>"
            f"<td>{max((x.get('peak_mem_gb') or 0) for x in p):.0f}"
            if p else "<td>"
            "</td>"
            f"<td>{html.escape(r.get('notes') or '')}</td>"
            "</tr>")

        # detail block per run
        d = [f"<h3 id='run-{html.escape(str(r.get('run_id')))}'>"
             f"Run {html.escape(str(r.get('run_id')))} — "
             f"{html.escape(str(r.get('design') or ''))} "
             f"[{html.escape(str(r.get('tag') or ''))}]</h3>"]
        if r.get("screenshot"):
            d.append(f"<img src='{html.escape(r['screenshot'])}' "
                     f"style='max-width:480px;border:1px solid #ccc'>")
        if r.get("errors"):
            d.append("<p style='color:#b3261e'>" + "; ".join(
                html.escape(e) for e in r["errors"]) + "</p>")
        comp = r.get("components") or {}
        if comp:
            d.append("<p><b>elements per component:</b> " + ", ".join(
                f"{html.escape(k)}: "
                + (f"{v['tets']} tets" if v.get("tets")
                   else f"{v.get('tris', 0)} tris")
                for k, v in sorted(comp.items())) + "</p>")
        for pp in p:
            d.append(f"<p><b>pass {pp['pass']}</b> — "
                     f"{pp.get('tets')} tets, "
                     f"{pp.get('nd_unknowns')} unknowns, "
                     f"peak {pp.get('peak_mem_gb')} GB, "
                     f"cum. {pp.get('cum_time_s')} s</p>")
            d.append("<table><tr><th>m</th><th>Re f (GHz)</th>"
                     "<th>Im f (GHz)</th><th>Q</th></tr>")
            for mm in pp.get("modes", []):
                d.append(f"<tr><td>{mm['m']}</td>"
                         f"<td>{mm['re_GHz']:.6f}</td>"
                         f"<td>{mm['im_GHz']:.3e}</td>"
                         f"<td>{_fmt_q(mm['Q'])}</td></tr>")
            d.append("</table>")
        details.append("\n".join(d))

    doc = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Palace run log</title><style>
body {{ font-family: system-ui, sans-serif; margin: 2em; color: #222; }}
table {{ border-collapse: collapse; margin: 0.6em 0; }}
th, td {{ border: 1px solid #ccc; padding: 4px 10px; text-align: left;
          font-size: 14px; }}
th {{ background: #f2f2f2; }}
h3 {{ margin-top: 2em; border-top: 2px solid #ddd; padding-top: 1em; }}
</style></head><body>
<h1>Palace run log</h1>
<p>{len(records)} run(s). Newest first. Click a run id for details.</p>
<table>
<tr><th>run</th><th>date</th><th>design</th><th>tag</th><th>status</th>
<th>AMR its</th><th>tets (start)</th><th>tets (final)</th>
<th>wall</th><th>peak GB</th><th>notes</th></tr>
{''.join(rows)}
</table>
{''.join(details)}
</body></html>"""
    with open(os.path.join(registry, "runlog.html"), "w",
              encoding="utf-8") as fh:
        fh.write(doc)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add", help="record one finished run")
    a.add_argument("--log", default=None, help="the PBS .o log file")
    a.add_argument("--postpro", default=None,
                   help="the postpro folder Palace wrote (eig.csv); "
                        "auto-detected from the config's Output if the "
                        "folder is present in --run-dir")
    a.add_argument("--run-dir", default=".")
    a.add_argument("--registry", default=os.path.join("..", "palace_runs"))
    a.add_argument("--screenshot", default=None,
                   help="image of the design (else: first png/jpg in "
                        "the run folder)")
    a.add_argument("--notes", default=None,
                   help="free text: mesh sizes, what this run tests")
    r = sub.add_parser("report", help="regenerate runlog.html + runs.csv")
    r.add_argument("--registry", default=os.path.join("..", "palace_runs"))
    args = ap.parse_args()
    if args.cmd == "add":
        add_run(args)
    else:
        _write_csv(os.path.abspath(args.registry))
        _write_html(os.path.abspath(args.registry))
        print(os.path.join(os.path.abspath(args.registry), "runlog.html"))


if __name__ == "__main__":
    main()

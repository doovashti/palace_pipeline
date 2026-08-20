#!/usr/bin/env python3
"""
hfss_size_report.py

Read the length-based mesh operations out of an Ansys .aedt project file
and print them as a copy-paste-ready SIZES dict for mesh_any.py.

    python hfss_size_report.py Vostok-1_Lakeside.aedt
    python hfss_size_report.py project.aedt --design Lakeside
    python hfss_size_report.py project.aedt --list
    python hfss_size_report.py project.aedt --config device_config.json

Why this exists separately from aedt_extract.py
-----------------------------------------------
aedt_extract.py resolved mesh-operation assignments through the geometry
tree by indexing each Operation's own ID. That is the wrong key. A mesh
operation writes

    Objects(457696)

and inside the GeometryPart named 'GND' the operation block carries

    ID=457695            <- the OPERATION's id
    ParentPartID=457696  <- the BODY's id, and what Objects() references

so every lookup missed and every op was silently dropped. This module
indexes ParentPartID (mapping ID too, harmlessly, since the two id spaces
share one counter and never collide). Same fix, applied in
aedt_extract.collect_part_ids, makes the notebook's step-3b merge work.

What it reports
---------------
  * every enabled LengthBased op: name, the expression exactly as typed
    in HFSS, the value in mm, and the body names it applies to
  * skin-depth ops contribute SurfTriMaxLength only, never the sub-micron
    skin depth itself, which as a volume size would be catastrophic
  * where several ops cover one body, the MINIMUM wins -- that is the
    constraint that actually bound in HFSS
  * with --config, bodies in device_config.json that NO HFSS op covers,
    which is the list you must size yourself (the cavity is the usual
    case: it is background vacuum in HFSS and never gets an op)

Read the printed sizes before pasting them. HFSS length ops are
refinement CAPS from an adaptive solve on a large CPU box; several are
far finer than this pipeline's analytic fallbacks, and copying them
wholesale onto a coarsened mesh inflates the element count instead of
reducing it. The merge is a coarsening win only where an HFSS op is
COARSER than the fallback.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import OrderedDict

BEGIN_RE = re.compile(r"^\s*\$begin\s+'(.*)'\s*$")
END_RE = re.compile(r"^\s*\$end\s+'(.*)'\s*$")
VARPROP_RE = re.compile(
    r"VariableProp(?:32)?\(\s*'([^']*)'\s*,\s*'[^']*'\s*,\s*"
    r"'[^']*'\s*,\s*'([^']*)'")
INT_RE = re.compile(r"\d+")

# unit -> SI multiplier
UNIT_SI = {
    "fm": 1e-15, "pm": 1e-12, "nm": 1e-9, "um": 1e-6, "mm": 1e-3,
    "cm": 1e-2, "dm": 1e-1, "meter": 1.0, "m": 1.0, "km": 1e3,
    "mil": 25.4e-6, "in": 25.4e-3, "inch": 25.4e-3, "ft": 0.3048,
    "Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9, "THz": 1e12,
    "s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9, "ps": 1e-12,
    "deg": math.pi / 180.0, "rad": 1.0, "cel": 1.0,
    "H": 1.0, "mH": 1e-3, "uH": 1e-6, "nH": 1e-9, "pH": 1e-12,
    "F": 1.0, "mF": 1e-3, "uF": 1e-6, "nF": 1e-9, "pF": 1e-12,
    "fF": 1e-15, "Ohm": 1.0, "ohm": 1.0, "kOhm": 1e3, "MOhm": 1e6,
    "S": 1.0, "V": 1.0, "mV": 1e-3, "A": 1.0, "mA": 1e-3, "W": 1.0,
}
NUM_UNIT_RE = re.compile(
    r"(?<![\w.])([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\s*"
    r"(" + "|".join(sorted(UNIT_SI, key=len, reverse=True)) + r")\b")

SAFE_NS = {
    "pi": math.pi, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "sqrt": math.sqrt, "abs": abs, "exp": math.exp, "log": math.log,
    "log10": math.log10, "pow": pow, "max": max, "min": min,
}

# Bodies that typically carry no HFSS mesh operation, with a starting
# suggestion. The cavity is background vacuum in HFSS: the adaptive
# solver sizes it on wavelength, so there is no length op to read.
# These are SUGGESTIONS, not measurements -- edit before use.
COMMON_UNMESHED_HINTS = {
    "cavity": "wavelength-driven in HFSS; ~1-2 mm is a usual start",
    "chip": "substrate bulk; the mesher's lambda/12 fallback is fine",
    "vacuum": "background volume",
    "Vacuum": "background volume",
}


def unquote(text):
    text = str(text).strip()
    if len(text) >= 2 and text[0] == "'" and text[-1] == "'":
        return text[1:-1]
    return text


def units_to_si(expr):
    """'terminal_rect_wid/2' stays; '5um' -> '(5*1e-06)'."""
    return NUM_UNIT_RE.sub(
        lambda m: "({0}*{1!r})".format(m.group(1), UNIT_SI[m.group(2)]),
        expr)


def evaluate_variables(var_exprs):
    """Resolve {name: expression} to SI floats, iterating to a fixed
    point so variables defined in terms of other variables resolve."""
    si_exprs = {n: units_to_si(e) for n, e in var_exprs.items()}
    values = {}
    for _ in range(len(si_exprs) + 3):
        progressed = False
        for name, expr in si_exprs.items():
            if name in values:
                continue
            ns = dict(SAFE_NS)
            ns.update(values)
            try:
                values[name] = float(
                    eval(expr, {"__builtins__": {}}, ns))  # noqa: S307
                progressed = True
            except Exception:
                pass
        if not progressed:
            break
    return values


def expression_to_mm(text, values):
    if text is None:
        return None
    ns = dict(SAFE_NS)
    ns.update(values)
    try:
        return float(
            eval(units_to_si(text), {"__builtins__": {}}, ns)) * 1e3
    except Exception:
        return None


# ---------------------------------------------------------------------
# single-pass scanner
# ---------------------------------------------------------------------

def scan(path):
    """
    One pass over the .aedt, returning per design:
        {design: {"vars": {...}, "ids": {int: body},
                  "ops": [ {...}, ... ]}}

    The .aedt is a plain-text $begin/$end tree, so this is a state
    machine over the block stack rather than a full parse. That keeps
    memory flat on 20 MB+ project files and needs no Ansys licence.

    A "design" here is the enclosing block of an HFSSModel subtree; a
    project file may hold dozens.
    """
    records = []           # one per HFSSModel block, in file order
    stack = []
    entry = None           # current design record
    part_name = None       # current GeometryPart name
    op = None              # current mesh-operation dict
    op_block = None        # block name that opened the current op

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\r\n")

            begin = BEGIN_RE.match(line)
            if begin:
                name = begin.group(1)
                stack.append(name)
                # Each design is one top-level 'HFSSModel' block; its
                # own Name= field (read below) is the design name AEDT
                # shows in the project tree. A project file commonly
                # holds several, and the file is NOT named after them.
                if name == "HFSSModel":
                    entry = {"name": None, "vars": OrderedDict(),
                             "ids": {}, "ops": []}
                    records.append(entry)
                    part_name = None
                # a mesh operation is any block directly under
                # MeshOperations (blocks are named after the OP, not
                # the string "MeshOperation")
                elif len(stack) >= 2 and stack[-2] == "MeshOperations":
                    op = {"name": name, "type": None, "enabled": True,
                          "max_length": None, "surf_tri": None,
                          "skin_depth": None, "ids": []}
                    op_block = name
                continue

            if END_RE.match(line):
                closing = stack.pop() if stack else None
                if op is not None and closing == op_block:
                    if entry is not None:
                        entry["ops"].append(op)
                    op, op_block = None, None
                elif closing == "HFSSModel":
                    entry, part_name = None, None
                continue

            text = line.strip()
            if not text:
                continue

            if entry is None:
                continue

            # ---- the design's own name -----------------------------
            if stack[-1] == "HFSSModel" and text.startswith("Name="):
                entry["name"] = unquote(text.split("=", 1)[1])
                continue

            # ---- design variables ----------------------------------
            var = VARPROP_RE.search(text)
            if var:
                entry["vars"].setdefault(var.group(1), var.group(2))
                continue

            # ---- geometry: body id map -----------------------------
            # Objects(...) in a mesh op references the BODY id, which
            # AEDT stores as ParentPartID on each of the part's
            # operations -- NOT the operation's own ID.
            if "GeometryPart" in stack:
                if stack[-1] == "Attributes" and text.startswith("Name="):
                    part_name = unquote(text.split("=", 1)[1])
                elif part_name and text.startswith("ParentPartID="):
                    try:
                        entry["ids"][int(text.split("=", 1)[1])] = part_name
                    except ValueError:
                        pass
                elif part_name and text.startswith("ID="):
                    # harmless second key; the two id spaces share one
                    # counter so they never collide
                    try:
                        entry["ids"].setdefault(
                            int(text.split("=", 1)[1]), part_name)
                    except ValueError:
                        pass

            # ---- mesh operation fields -----------------------------
            if op is not None:  # noqa: SIM102
                if text.startswith("Type="):
                    op["type"] = unquote(text.split("=", 1)[1])
                elif text.startswith("Enabled="):
                    op["enabled"] = text.split("=", 1)[1].strip() == "true"
                elif text.startswith("MaxLength="):
                    op["max_length"] = unquote(text.split("=", 1)[1])
                elif text.startswith("SurfTriMaxLength="):
                    op["surf_tri"] = unquote(text.split("=", 1)[1])
                elif text.startswith("SkinDepth="):
                    op["skin_depth"] = unquote(text.split("=", 1)[1])
                elif text.startswith("Objects("):
                    op["ids"].extend(int(x) for x in INT_RE.findall(text))

    designs = OrderedDict()
    for index, rec in enumerate(records):
        name = rec["name"] or "design_{0}".format(index)
        while name in designs:          # duplicate names cannot happen
            name += "_dup"              # in AEDT, but never lose a block
        designs[name] = rec
    return designs


# ---------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------

def resolve(design_data):
    """Return (ops, by_body, unresolved) with sizes in mm."""
    values = evaluate_variables(design_data["vars"])
    ids = design_data["ids"]

    ops, unresolved = [], []
    for op in design_data["ops"]:
        if op["type"] != "LengthBased" and op["surf_tri"] is None:
            continue
        raw = op["max_length"] or op["surf_tri"]
        mm = expression_to_mm(raw, values)
        bodies, missing = [], []
        for oid in op["ids"]:
            name = ids.get(oid)
            if name is None:
                missing.append(oid)
            else:
                bodies.append(name)
        if missing:
            unresolved.append((op["name"], missing))
        ops.append({"name": op["name"], "expr": raw, "mm": mm,
                    "bodies": bodies, "enabled": op["enabled"],
                    "skin_depth": op["skin_depth"]})

    by_body = {}
    for op in ops:
        if not op["enabled"] or op["mm"] is None or op["mm"] <= 0:
            continue
        for body in op["bodies"]:
            prev = by_body.get(body)
            if prev is None or op["mm"] < prev["mm"]:
                by_body[body] = {"mm": op["mm"], "op": op["name"],
                                 "expr": op["expr"]}
    return ops, by_body, unresolved


def print_report(design_name, ops, by_body, unresolved, config_path=None):
    print("=" * 78)
    print("HFSS length-based mesh operations -- design {0!r}".format(
        design_name))
    print("=" * 78)

    if not ops:
        print("\nNo length-based mesh operations found in this design.")
        return

    print("\n{0:<26}{1:<26}{2:>10}  bodies".format(
        "op", "expression as typed", "mm"))
    print("-" * 78)
    for op in sorted(ops, key=lambda o: (o["mm"] is None, o["mm"] or 0)):
        flag = "" if op["enabled"] else "   [DISABLED]"
        shown = "?" if op["mm"] is None else "{0:.6g}".format(op["mm"])
        bodies = ", ".join(op["bodies"]) or "(unresolved)"
        if len(bodies) > 300:
            bodies = bodies[:297] + "..."
        print("{0:<26}{1:<26}{2:>10}  {3}{4}".format(
            op["name"][:25], (op["expr"] or "")[:25], shown, bodies, flag))

    if unresolved:
        print("\nWARNING: {0} op(s) had assignment ids with no matching "
              "geometry part:".format(len(unresolved)))
        for name, missing in unresolved[:8]:
            print("  {0}: {1}".format(name, missing[:6]))
        print("  These bodies are NOT in the block below. If this list is "
              "long, the id map is still wrong -- do not paste the "
              "result.")

    # ---- bodies with no HFSS op ------------------------------------
    unmeshed = []
    if config_path:
        try:
            with open(config_path, encoding="utf-8") as fh:
                cfg = json.load(fh)
            for name, entry in (cfg.get("objects") or {}).items():
                if entry.get("model") is False:
                    continue
                if name not in by_body:
                    unmeshed.append((name, entry.get("domain_role")))
        except OSError as error:
            print("\n(could not read {0}: {1})".format(config_path, error))

    # ---- the copy-paste block --------------------------------------
    print("\n" + "=" * 78)
    print("COPY-PASTE INTO THE SIZES CELL  (mm; 0.0005 = 0.5 um)")
    print("=" * 78)
    print("SIZES = {")
    print("    # ---- read straight out of HFSS, design {0!r} ----".format(
        design_name))
    print("    # These are HFSS REFINEMENT CAPS from an adaptive solve.")
    print("    # Several are finer than this pipeline's fallbacks: "
          "pasting")
    print("    # them all will GROW the mesh. Delete the ones you do not")
    print("    # need and check the budget block before qsub.")
    width = max((len(b) for b in by_body), default=10) + 4
    for body in sorted(by_body, key=lambda b: (by_body[b]["mm"], b)):
        rec = by_body[body]
        print('    {0:<{w}}{1:<12}# op {2!r} ({3})'.format(
            '"{0}":'.format(body), "{0:g},".format(rec["mm"]),
            rec["op"], rec["expr"], w=width))

    if unmeshed:
        print()
        print("    # ---- in the design but NOT meshed by any HFSS op ----")
        print("    # HFSS sizes these from wavelength during its adaptive")
        print("    # solve, so there is no length to read. Uncomment and")
        print("    # set a value, or leave commented to accept the")
        print("    # mesher's analytic fallback.")
        for name, role in sorted(unmeshed):
            hint = COMMON_UNMESHED_HINTS.get(name)
            note = "  # {0}".format(hint) if hint else \
                   ("  # role: {0}".format(role) if role else "")
            print('    # "{0}": ,{1}'.format(name, note))
    else:
        print()
        print("    # ---- not meshed in HFSS: add your own ----")
        print("    # The cavity is background vacuum in HFSS and never")
        print("    # gets a length op, so it is not in the list above.")
        print("    # Pass --config device_config.json to have this script")
        print("    # list every such body for you.")
        print('    # "cavity": 2.0,')
    print("}")


def main():
    ap = argparse.ArgumentParser(
        description="Extract HFSS length-based mesh sizes from an .aedt")
    ap.add_argument("aedt")
    ap.add_argument("--design", default=None,
                    help="design name (default: the only one, or list)")
    ap.add_argument("--list", action="store_true",
                    help="list the designs in the file and exit")
    ap.add_argument("--config", default=None,
                    help="device_config.json, to list bodies that have "
                         "no HFSS mesh operation")
    ap.add_argument("--json", default=None,
                    help="also write the resolved sizes to this file")
    args = ap.parse_args()

    print("Reading {0} ...".format(args.aedt))
    designs = scan(args.aedt)
    if not designs:
        print("No designs found. Is this an .aedt project file?")
        return 2

    named = [d for d, v in designs.items() if v["ops"]]
    if args.list:
        print("\nDesigns ({0} total, {1} with mesh operations):".format(
            len(designs), len(named)))
        for name, data in designs.items():
            print("  {0:<40} {1} op(s), {2} var(s)".format(
                name, len(data["ops"]), len(data["vars"])))
        return 0

    if args.design:
        if args.design not in designs:
            print("\nDesign {0!r} not found. Available: {1}".format(
                args.design, ", ".join(list(designs)[:20])))
            return 2
        chosen = [args.design]
    elif len(named) == 1:
        chosen = named
    elif named:
        print("\n{0} designs have mesh operations: {1}".format(
            len(named), ", ".join(named[:20])))
        print("Pick one with --design NAME (or --list to see all).")
        return 2
    else:
        print("\nNo design in this file has any mesh operations.")
        return 2

    out = {}
    for name in chosen:
        ops, by_body, unresolved = resolve(designs[name])
        print_report(name, ops, by_body, unresolved, args.config)
        out[name] = {b: r["mm"] for b, r in by_body.items()}

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        print("\nWritten: {0}".format(args.json))
    return 0


# ---------------------------------------------------------------------
# notebook entry point
# ---------------------------------------------------------------------

def sizes_from_aedt(aedt_path, design=None, config_path=None,
                    quiet=False):
    """
    Notebook-friendly wrapper.

        import hfss_size_report as hs
        sizes = hs.sizes_from_aedt(AEDT_PATH, design=DESIGN_NAME,
                                   config_path=RUN_DIR + "/device_config.json")

    Prints the report and the copy-paste block, and returns
    {body: size_mm} so the values can be used directly if preferred.
    """
    designs = scan(aedt_path)
    if not designs:
        raise RuntimeError("no designs found in {0}".format(aedt_path))
    if design is None:
        named = [d for d, v in designs.items() if v["ops"]]
        if len(named) != 1:
            raise RuntimeError(
                "specify design=; candidates with mesh ops: {0}".format(
                    ", ".join(named[:20]) or "(none)"))
        design = named[0]
    if design not in designs:
        raise RuntimeError(
            "design {0!r} not in file; present: {1}".format(
                design, ", ".join(list(designs)[:20])))

    ops, by_body, unresolved = resolve(designs[design])
    if not quiet:
        print_report(design, ops, by_body, unresolved, config_path)
    return {body: rec["mm"] for body, rec in by_body.items()}


if __name__ == "__main__":
    sys.exit(main())

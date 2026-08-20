#!/usr/bin/env python3
"""
Extract HFSS mesh operations, design variables, and adaptive-setup
settings from an .aedt project file -- WITHOUT opening Ansys.

    python aedt_extract.py MyProject.aedt
    python aedt_extract.py MyProject.aedt --list
    python aedt_extract.py MyProject.aedt --design Vostok-1
    python aedt_extract.py MyProject.aedt --json settings.json

The .aedt file is a plain-text $begin/$end tree; this script parses it
the same way step_bodies.py parses STEP -- pure text, runs in seconds,
and works on ARCHIVED project files (old design variants) with no
license checkout. That makes it the right tool for the chronological
record: run it on each variant's .aedt and diff the JSONs.

Note that a project file is NOT named after the designs inside it, and
one file routinely holds several designs with different mesh operations
and different body names. Always pass --design, or check --list first.

What it reports, per HFSS design found in the file:

  1. Mesh operations (MeshSetup/MeshOperations): name, type
     (LengthBased / SkinDepthBased / ...), enabled, restrict-length,
     max length (converted to mm), skin depth / layers, and the list
     of objects or faces each op is assigned to.
  2. Design variables (VariableProp entries): name, the expression
     exactly as typed in HFSS, and -- where the expression can be
     resolved (numbers, units, other variables, arithmetic) -- the
     evaluated value in SI and in mm. Project ($-prefixed) variables
     are picked up too.
  3. Adaptive solve setups (eigenmode: NumModes / MinimumFrequency /
     MaximumPasses / MaxDeltaFreq; driven: Frequency / MaxDeltaS) --
     these are the numbers the recipe's HFSS<->Palace AMR comparison
     needs (Max Delta Freq is HFSS's convergence criterion).
  4. A "mesh_any --size suggestions" section: every enabled
     length-based op with a max length becomes a candidate
     --size BODY=MM override (HFSS object names match the STEP body
     names since the STEP is exported from this same model).

Everything parsed is also dumped RAW into the JSON (--json), so if
your AEDT version spells a field differently, nothing is lost.

Caveats (honest ones):
  * Field names vary a little across AEDT versions; the common 2019+
    spellings are handled, unknown fields land in the raw JSON.
  * Face-assigned ops report face IDs, not body names (the .aedt does
    not store the face->body map in text form); object-assigned ops
    report names.
  * Expression evaluation handles units + arithmetic + variable
    references; anything using HFSS functions this script does not
    know stays as an unevaluated expression string.
  * HFSS length ops are refinement CAPS from an adaptive solve. Most
    are much finer than this pipeline's analytic fallbacks, so
    adopting them all makes the mesh BIGGER. Read the budget block in
    the mesher before using these numbers.

REVISION NOTES (this version):
  * FIXED the id resolution that made every op report "id:NNN" and
    silently drop out of extract_length_sizes. A mesh operation writes
    its assignment as BODY ids, and AEDT stores a body's id as
    ParentPartID on each of the part's operations, not as the
    operation's own ID. See collect_part_ids for the worked example.
  * find_designs now keys on the HFSSModel block's own Name field and
    refuses to return the same block twice, so a project holding
    several designs cannot silently merge them.
  * --list prints the designs in a file with their op and variable
    counts, for when you do not know what a project contains.
"""

import argparse
import json
import math
import re
import sys
from collections import OrderedDict

# ---------------------------------------------------------------------
# 1. generic $begin/$end tree parser
# ---------------------------------------------------------------------

BEGIN_RE = re.compile(r"^\s*\$begin\s+'(.*)'\s*$")
END_RE = re.compile(r"^\s*\$end\s+'(.*)'\s*$")
KV_RE = re.compile(r"^\s*([\w$][\w $.\-()/]*?)=(.*)$")


class Block(object):
    """One $begin/$end block: named children (repeats -> list) plus the
    raw non-block lines, plus a key=value dict of those lines."""

    __slots__ = ("name", "children", "lines", "kv")

    def __init__(self, name):
        self.name = name
        self.children = OrderedDict()   # name -> Block or [Block, ...]
        self.lines = []                 # raw stripped lines
        self.kv = OrderedDict()         # parsed Key=Value (last wins)

    def add_child(self, blk):
        if blk.name in self.children:
            cur = self.children[blk.name]
            if isinstance(cur, list):
                cur.append(blk)
            else:
                self.children[blk.name] = [cur, blk]
        else:
            self.children[blk.name] = blk

    def child_list(self, name):
        c = self.children.get(name)
        if c is None:
            return []
        return c if isinstance(c, list) else [c]

    def first(self, name):
        lst = self.child_list(name)
        return lst[0] if lst else None

    def walk(self):
        yield self
        for c in self.children.values():
            for blk in (c if isinstance(c, list) else [c]):
                for sub in blk.walk():
                    yield sub

    def to_raw(self):
        out = OrderedDict()
        if self.kv:
            out["_kv"] = OrderedDict(self.kv)
        extra = [ln for ln in self.lines if not KV_RE.match(ln)]
        if extra:
            out["_lines"] = extra
        for name, c in self.children.items():
            if isinstance(c, list):
                out[name] = [b.to_raw() for b in c]
            else:
                out[name] = c.to_raw()
        return out


def parse_aedt(path):
    root = Block("<root>")
    stack = [root]
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\r\n")
            m = BEGIN_RE.match(line)
            if m:
                blk = Block(m.group(1))
                stack[-1].add_child(blk)
                stack.append(blk)
                continue
            if END_RE.match(line):
                if len(stack) > 1:
                    stack.pop()
                continue
            s = line.strip()
            if not s:
                continue
            cur = stack[-1]
            cur.lines.append(s)
            kv = KV_RE.match(s)
            if kv:
                cur.kv[kv.group(1).strip()] = kv.group(2).strip()
    return root


# ---------------------------------------------------------------------
# 2. value / unit / expression handling
# ---------------------------------------------------------------------

# multipliers to SI base units
UNIT_SI = {
    "fm": 1e-15, "pm": 1e-12, "nm": 1e-9, "um": 1e-6, "mm": 1e-3,
    "cm": 1e-2, "dm": 1e-1, "meter": 1.0, "m": 1.0, "km": 1e3,
    "mil": 25.4e-6, "in": 25.4e-3, "inch": 25.4e-3, "ft": 0.3048,
    "Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9, "THz": 1e12,
    "s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9, "ps": 1e-12,
    "deg": math.pi / 180.0, "rad": 1.0, "cel": 1.0,
    "H": 1.0, "mH": 1e-3, "uH": 1e-6, "nH": 1e-9, "pH": 1e-12,
    "fH": 1e-15,
    "F": 1.0, "mF": 1e-3, "uF": 1e-6, "nF": 1e-9, "pF": 1e-12,
    "fF": 1e-15,
    "Ohm": 1.0, "ohm": 1.0, "kOhm": 1e3, "MOhm": 1e6,
    "W": 1.0, "mW": 1e-3, "uW": 1e-6,
    "V": 1.0, "mV": 1e-3, "A": 1.0, "mA": 1e-3,
}
LENGTH_UNITS = {"fm", "pm", "nm", "um", "mm", "cm", "dm", "meter", "m",
                "km", "mil", "in", "inch", "ft"}

NUM_UNIT_RE = re.compile(
    r"(?<![\w.])([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\s*"
    r"(" + "|".join(sorted(UNIT_SI, key=len, reverse=True)) + r")\b")

SAFE_EVAL_NS = {
    "pi": math.pi, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "sqrt": math.sqrt, "abs": abs, "exp": math.exp, "log": math.log,
    "log10": math.log10, "pow": pow, "max": max, "min": min,
}


def units_to_si_expr(expr):
    """'2*pad_gap + 5um' -> '2*pad_gap + (5*1e-06)' (numbers w/ units
    become SI); also note whether a length unit appeared."""
    saw_length = [False]

    def sub(m):
        num, unit = m.group(1), m.group(2)
        if unit in LENGTH_UNITS:
            saw_length[0] = True
        return "({0}*{1!r})".format(num, UNIT_SI[unit])

    return NUM_UNIT_RE.sub(sub, expr), saw_length[0]


def evaluate_variables(var_exprs):
    """Iteratively resolve {name: expression} to SI floats. Returns
    {name: {"expr", "si", "mm", "is_length"}}; si is None where the
    expression could not be resolved."""
    si_exprs, is_len = {}, {}
    for name, expr in var_exprs.items():
        e, saw = units_to_si_expr(expr)
        si_exprs[name] = e
        is_len[name] = saw

    values = {}
    for _ in range(len(var_exprs) + 2):     # iterate to fixed point
        progressed = False
        for name, e in si_exprs.items():
            if name in values:
                continue
            ns = dict(SAFE_EVAL_NS)
            ns.update(values)
            try:
                v = eval(  # noqa: S307 -- restricted namespace
                    e, {"__builtins__": {}}, ns)
                values[name] = float(v)
                progressed = True
            except Exception:
                pass
        if not progressed:
            break

    out = OrderedDict()
    for name, expr in var_exprs.items():
        si = values.get(name)
        # a variable is length-flavoured if its own text used a length
        # unit, or it references a length-flavoured variable
        length = is_len[name] or any(
            is_len.get(ref, False)
            for ref in re.findall(r"[A-Za-z_$][\w$]*", expr)
            if ref in is_len)
        out[name] = {
            "expr": expr,
            "si": si,
            "mm": (si * 1e3 if (si is not None and length) else None),
            "is_length": length,
        }
    return out


def quantity_to_mm(text, var_values):
    """'0.05mm' / '30um' / 'gap/2' -> mm float, or None."""
    if text is None:
        return None
    e, _ = units_to_si_expr(text)
    ns = dict(SAFE_EVAL_NS)
    ns.update({n: d["si"] for n, d in var_values.items()
               if d["si"] is not None})
    try:
        return float(eval(e, {"__builtins__": {}}, ns)) * 1e3
    except Exception:
        return None


def unquote(v):
    v = v.strip()
    if len(v) >= 2 and v[0] == "'" and v[-1] == "'":
        return v[1:-1]
    return v


# ---------------------------------------------------------------------
# 3. targeted extraction
# ---------------------------------------------------------------------

VARPROP_RE = re.compile(
    r"VariableProp(?:32)?\(\s*'([^']*)'\s*,\s*'[^']*'\s*,\s*"
    r"'[^']*'\s*,\s*'([^']*)'")

NAME_LIST_RE = re.compile(r"'([^']+)'")
INT_LIST_RE = re.compile(r"\b(\d+)\b")


def find_designs(root):
    """Every design block in the project, in file order.

    A design is a child block of the project that contains a MeshSetup,
    ModelSetup or AnalysisSetup subtree; in practice these are the
    'HFSSModel' blocks, each carrying its own Name field. A project
    routinely holds SEVERAL, and the file name says nothing about them,
    so the caller must pick one: merging two designs would mix body
    names and mesh sizes from different devices with no warning.

    Returns (project_block, [design blocks]).
    """
    proj = None
    for c in root.children.values():
        blk = c[0] if isinstance(c, list) else c
        if blk.name.endswith("Project") or blk.name == "AnsoftProject":
            proj = blk
            break
    scope = proj if proj is not None else root

    designs, seen = [], set()
    for _name, c in scope.children.items():
        for blk in (c if isinstance(c, list) else [c]):
            if id(blk) in seen:
                continue
            names = {sub.name for sub in blk.walk()}
            if ("MeshSetup" in names or "ModelSetup" in names
                    or "AnalysisSetup" in names):
                seen.add(id(blk))
                designs.append(blk)
    return scope, designs


def design_name(blk):
    n = blk.kv.get("Name")
    return unquote(n) if n else blk.name


def collect_variables(blk):
    """All VariableProp entries in this subtree -> {name: expr}."""
    out = OrderedDict()
    for sub in blk.walk():
        for line in sub.lines:
            m = VARPROP_RE.search(line)
            if m:
                out.setdefault(m.group(1), m.group(2))
    return out


INLINE_LIST_RE = re.compile(r"^(Objects|Faces|Assignment)\s*[\[(](.*)$")


def collect_part_ids(design_blk):
    """Map numeric geometry IDs -> body names for one design.

    AEDT stores mesh-op assignments as internal ids, not names:

        $begin 'GND'
            Type='LengthBased'
            Objects(457696)
            MaxLength='2.5mm'
        $end 'GND'

    The id in Objects() is the BODY's id. Inside the GeometryPart named
    'GND', the operations that built that body look like:

        $begin 'Operation'
            OperationType='Rectangle'
            ID=457695              <- the OPERATION's own id
            ParentPartID=457696    <- the BODY's id, what Objects() uses

    Indexing 'ID' alone (as this function originally did) therefore
    missed every assignment by a small offset: the map filled with
    hundreds of useless entries and every op was dropped as
    unresolvable, so extract_length_sizes returned nothing at all while
    reporting success.

    Both keys are mapped, ParentPartID written last so it wins any
    collision, because a body built from several operations may be
    referenced through more than one of them.
    """
    id_map = {}
    for sub in design_blk.walk():
        if sub.name != "GeometryPart":
            continue
        attrs = sub.first("Attributes")
        if attrs is None or "Name" not in attrs.kv:
            continue
        pname = unquote(attrs.kv["Name"])
        for deep in sub.walk():
            if deep.name != "Operation":
                continue
            for key in ("ID", "ParentPartID"):
                if key in deep.kv:
                    try:
                        id_map[int(unquote(deep.kv[key]))] = pname
                    except ValueError:
                        pass
    return id_map


def collect_assignment(op_blk, id_map=None):
    """Objects/faces an op applies to, from every known spelling:
    child blocks ($begin 'Objects'), Key=Value lines, inline arrays
    like  Objects[2: 'a', 'b'],  and numeric-ID lists like
    Objects(127, 245) resolved through the geometry id_map."""
    id_map = id_map or {}
    objects, obj_ids, faces = [], [], []
    for key in ("Objects", "Assignment", "Entities"):
        sub = op_blk.first(key)
        if sub is not None:
            for line in sub.lines:
                if KV_RE.match(line):
                    continue
                if "'" in line:
                    objects.extend(NAME_LIST_RE.findall(line))
                else:
                    obj_ids.extend(
                        int(x) for x in INT_LIST_RE.findall(line))
    for key in ("Objects", "Assignment", "Faces"):
        val = op_blk.kv.get(key)
        if val:
            if "'" in val:
                objects.extend(NAME_LIST_RE.findall(val))
            elif key == "Faces":
                faces.extend(int(x) for x in INT_LIST_RE.findall(val))
            else:
                obj_ids.extend(
                    int(x) for x in INT_LIST_RE.findall(val))
    for line in op_blk.lines:
        m = INLINE_LIST_RE.match(line)
        if not m:
            continue
        body = m.group(2)
        # strip an array count prefix ("2: 'a', 'b'") if present
        payload = body.split(":", 1)[-1] if ":" in body else body
        if "'" in body:
            objects.extend(NAME_LIST_RE.findall(body))
        elif m.group(1) == "Faces":
            faces.extend(int(x) for x in INT_LIST_RE.findall(payload))
        else:
            obj_ids.extend(int(x) for x in INT_LIST_RE.findall(payload))
    sub = op_blk.first("Faces")
    if sub is not None:
        for line in sub.lines:
            faces.extend(int(x) for x in INT_LIST_RE.findall(line))

    # resolve numeric object IDs through the geometry map
    for oid in obj_ids:
        name = id_map.get(oid)
        objects.append(name if name is not None else "id:{0}".format(oid))

    # dedupe, keep order
    seen = set()
    objects = [o for o in objects
               if not (o in seen or seen.add(o))]
    return objects, faces


def collect_mesh_ops(blk, var_values, id_map=None):
    ops = []
    for sub in blk.walk():
        if sub.name != "MeshSetup":
            continue
        mo = sub.first("MeshOperations")
        scan = mo if mo is not None else sub
        for name, c in scan.children.items():
            if name == "MeshSettings":
                continue    # global slider/curvilinear block, not an op
            for op in (c if isinstance(c, list) else [c]):
                if not op.kv and not op.children:
                    continue
                kv = op.kv
                if "Type" not in kv and "MaxLength" not in kv \
                        and "SkinDepth" not in kv:
                    continue    # bookkeeping block, not a mesh op
                objects, faces = collect_assignment(op, id_map)
                entry = OrderedDict()
                entry["name"] = unquote(kv.get("Name", name))
                entry["type"] = unquote(kv.get("Type", op.name))
                entry["enabled"] = unquote(
                    kv.get("Enabled", "true")).lower() == "true"
                entry["objects"] = objects
                entry["faces"] = faces
                for src, dst in (("MaxLength", "max_length"),
                                 ("SkinDepth", "skin_depth"),
                                 ("SurfTriMaxLength", "surf_tri_max"),
                                 ("MaxElemLength", "max_elem_length")):
                    if src in kv:
                        raw = unquote(kv[src])
                        entry[dst] = raw
                        mm = quantity_to_mm(raw, var_values)
                        if mm is not None:
                            entry[dst + "_mm"] = mm
                for src, dst in (("RestrictLength", "restrict_length"),
                                 ("RestrictElem", "restrict_elem"),
                                 ("RefineInside", "refine_inside")):
                    if src in kv:
                        entry[dst] = unquote(kv[src]).lower() == "true"
                for src in ("NumLayers", "NumMaxElem"):
                    if src in kv:
                        entry[src] = unquote(kv[src])
                ops.append(entry)
    return ops


SETUP_KEYS = ("NumModes", "MinimumFrequency", "MaxDeltaFreq",
              "MaximumPasses", "MinimumPasses",
              "MinimumConvergedPasses", "PercentRefinement",
              "Frequency", "MaxDeltaS", "MaxDeltaE",
              "BasisOrder", "SolveType")


def collect_setups(blk):
    setups = []
    # prefer the inner 'SolveSetups' container; fall back to
    # 'AnalysisSetup' only when no SolveSetups block exists (older
    # files) -- scanning both double-reports every setup.
    containers = [s for s in blk.walk() if s.name == "SolveSetups"]
    if not containers:
        containers = [s for s in blk.walk()
                      if s.name == "AnalysisSetup"]
    for sub in containers:
        for sname, c in sub.children.items():
            for st in (c if isinstance(c, list) else [c]):
                found = OrderedDict()
                for deep in st.walk():
                    for k in SETUP_KEYS:
                        if k in deep.kv and k not in found:
                            found[k] = unquote(deep.kv[k])
                if found:
                    found_named = OrderedDict()
                    found_named["name"] = unquote(
                        st.kv.get("Name", sname))
                    found_named.update(found)
                    setups.append(found_named)
    return setups


# ---------------------------------------------------------------------
# 4. importable API (used by pipeline_helpers.merge_hfss_sizes)
# ---------------------------------------------------------------------

def list_designs(aedt_path):
    """[(name, n_mesh_ops, n_variables)] for every design in the file.

    Cheap way to find out what a project actually contains, since the
    file is not named after its designs.
    """
    root = parse_aedt(aedt_path)
    _scope, designs = find_designs(root)
    out = []
    for des in designs:
        variables = collect_variables(des)
        ops = collect_mesh_ops(des, {}, None)
        out.append((design_name(des), len(ops), len(variables)))
    return out


def extract_length_sizes(aedt_path, design=None):
    """Return (design_name, {body: {"mm", "op", "type"}}) for every
    ENABLED object-assigned mesh op with a resolvable length in the
    given .aedt file. Multiple ops on one body -> the MINIMUM length
    (the most restrictive constraint is the one that bound in HFSS).
    Skin-depth ops contribute their SurfTriMaxLength (the surface
    triangle cap -- the number comparable to a size), never the
    sub-micron skin depth itself, which would be a catastrophic
    volume-mesh size. Raises RuntimeError if the design is not found."""
    root = parse_aedt(aedt_path)
    _scope, designs = find_designs(root)
    chosen = None
    names = []
    for des in designs:
        dname = design_name(des)
        names.append(dname)
        if design is None or dname == design:
            if chosen is None:
                chosen = des
            if design is not None:
                break
    if chosen is None:
        raise RuntimeError(
            "design {0!r} not found in {1}; designs present: {2}".format(
                design, aedt_path, ", ".join(names) or "(none)"))
    if design is None and len(designs) > 1:
        print("aedt_extract: {0} designs in this file ({1}); using {2!r}. "
              "Pass design= to choose -- body names and mesh sizes "
              "differ between designs.".format(
                  len(designs), ", ".join(names), design_name(chosen)))

    values = evaluate_variables(collect_variables(chosen))
    id_map = collect_part_ids(chosen)
    out = OrderedDict()
    unresolved = []
    n_ops = 0
    for op in collect_mesh_ops(chosen, values, id_map):
        if not op["enabled"] or not op["objects"]:
            continue
        mm = op.get("max_length_mm")
        if mm is None:
            mm = op.get("surf_tri_max_mm")
        if mm is None or mm <= 0:
            continue
        n_ops += 1
        for body in op["objects"]:
            if body.startswith("id:"):
                unresolved.append((op["name"], body))
                continue
            prev = out.get(body)
            if prev is None or mm < prev["mm"]:
                out[body] = {"mm": mm, "op": op["name"],
                             "type": op["type"]}
    if unresolved:
        print("aedt_extract: {0} assignment id(s) had no matching "
              "geometry part ({1}) -- those entries were skipped".format(
                  len(unresolved),
                  ", ".join("{0}->{1}".format(o, b)
                            for o, b in unresolved[:5])))
        if not out and n_ops:
            print("aedt_extract: NOTHING resolved out of {0} sized op(s). "
                  "The id map has {1} entr(ies). If this persists, the "
                  "geometry tree in this AEDT version stores body ids "
                  "under a key collect_part_ids does not read -- do not "
                  "treat the empty result as 'no mesh operations'.".format(
                      n_ops, len(id_map)))
    return design_name(chosen), out


# ---------------------------------------------------------------------
# 5. report
# ---------------------------------------------------------------------

def fmt_mm(mm):
    return "?" if mm is None else "{0:.6g} mm".format(mm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("aedt", help=".aedt project file")
    ap.add_argument("--design", default=None,
                    help="only report this design (by Name)")
    ap.add_argument("--list", action="store_true",
                    help="list the designs in the file and exit")
    ap.add_argument("--json", default=None,
                    help="write everything (incl. raw parse of "
                         "MeshSetup) to this JSON file")
    args = ap.parse_args()

    print("Reading {0} ...".format(args.aedt))
    root = parse_aedt(args.aedt)
    scope, designs = find_designs(root)

    if not designs:
        print("No design-like blocks found. Top-level blocks were:")
        for n in scope.children:
            print("   ", n)
        return 2

    if args.list:
        print("\nDesigns ({0}):".format(len(designs)))
        for des in designs:
            ops = collect_mesh_ops(des, {}, None)
            variables = collect_variables(des)
            print("  {0:<40} {1:>3} mesh op(s), {2:>4} variable(s)".format(
                design_name(des), len(ops), len(variables)))
        return 0

    if args.design and args.design not in [design_name(d) for d in designs]:
        print("\nDesign {0!r} not in this file. Present: {1}".format(
            args.design, ", ".join(design_name(d) for d in designs)))
        return 2

    if not args.design and len(designs) > 1:
        print("\n{0} designs in this file: {1}".format(
            len(designs), ", ".join(design_name(d) for d in designs)))
        print("Reporting ALL of them. Body names and mesh sizes differ "
              "between designs -- do not mix them.")

    # project ($) variables live at project scope, outside the designs
    proj_vars_raw = OrderedDict()
    for name, c in scope.children.items():
        if name in ("ProjectVariables", "Properties"):
            for blk in (c if isinstance(c, list) else [c]):
                proj_vars_raw.update(collect_variables(blk))

    dump = OrderedDict()
    dump["project_variables"] = None  # filled below
    dump["designs"] = OrderedDict()

    for des in designs:
        dname = design_name(des)
        if args.design and dname != args.design:
            continue
        print("\n" + "=" * 70)
        print("DESIGN: {0}   (block '{1}')".format(dname, des.name))
        print("=" * 70)

        des_vars_raw = collect_variables(des)
        all_raw = OrderedDict(proj_vars_raw)
        all_raw.update(des_vars_raw)
        values = evaluate_variables(all_raw)

        # ---- variables ------------------------------------------------
        print("\nDesign variables ({0}):".format(len(des_vars_raw)))
        for name in des_vars_raw:
            d = values[name]
            if d["mm"] is not None:
                shown = fmt_mm(d["mm"])
            elif d["si"] is not None:
                shown = "{0:.6g} (SI)".format(d["si"])
            else:
                shown = "<unevaluated>"
            print("  {0:<32} = {1:<24} -> {2}".format(
                name, d["expr"], shown))
        if proj_vars_raw:
            print("\nProject variables ({0}):".format(
                len(proj_vars_raw)))
            for name in proj_vars_raw:
                d = values[name]
                shown = (fmt_mm(d["mm"]) if d["mm"] is not None else
                         ("{0:.6g} (SI)".format(d["si"])
                          if d["si"] is not None else "<unevaluated>"))
                print("  {0:<32} = {1:<24} -> {2}".format(
                    name, d["expr"], shown))

        # ---- mesh ops -------------------------------------------------
        id_map = collect_part_ids(des)
        ops = collect_mesh_ops(des, values, id_map)
        print("\nMesh operations ({0}); geometry id map: {1} entries".format(
            len(ops), len(id_map)))
        for op in ops:
            state = "" if op["enabled"] else "  [DISABLED]"
            print("  {0}  ({1}){2}".format(
                op["name"], op["type"], state))
            if "max_length" in op:
                print("      max length = {0}  ({1})".format(
                    op["max_length"],
                    fmt_mm(op.get("max_length_mm"))))
            if "skin_depth" in op:
                print("      skin depth = {0}  ({1}), layers = {2}"
                      .format(op["skin_depth"],
                              fmt_mm(op.get("skin_depth_mm")),
                              op.get("NumLayers", "?")))
            if "surf_tri_max" in op:
                print("      surf tri max = {0}  ({1})".format(
                    op["surf_tri_max"], fmt_mm(op.get("surf_tri_max_mm"))))
            if op["objects"]:
                print("      objects: {0}".format(
                    ", ".join(op["objects"])))
            if op["faces"]:
                print("      faces: {0} face id(s)".format(
                    len(op["faces"])))

        # ---- setups ---------------------------------------------------
        setups = collect_setups(des)
        if setups:
            print("\nSolve setups:")
            for st in setups:
                pairs = ", ".join("{0}={1}".format(k, v)
                                  for k, v in st.items() if k != "name")
                print("  {0}: {1}".format(st["name"], pairs))

        # ---- --size suggestions --------------------------------------
        sugg = []
        for op in ops:
            if not op["enabled"]:
                continue
            mm = op.get("max_length_mm") or op.get("surf_tri_max_mm")
            if mm is None or not op["objects"]:
                continue
            for obj in op["objects"]:
                if not obj.startswith("id:"):
                    sugg.append((obj, mm, op["name"]))
        if sugg:
            print("\nmesh_any --size candidates from length-based ops")
            print("(HFSS object names == STEP body names. These are HFSS")
            print(" refinement CAPS from an adaptive solve: most are")
            print(" FINER than this pipeline's fallbacks, so adopting")
            print(" them all GROWS the mesh. Check each against the")
            print(" budget block before using.)")
            for obj, mm, opname in sorted(sugg, key=lambda s: s[1]):
                print("  --size {0}={1:.6g}   # HFSS op '{2}'".format(
                    obj, mm, opname))

        # ---- raw dump -------------------------------------------------
        ddump = OrderedDict()
        ddump["variables"] = OrderedDict(
            (n, values[n]) for n in des_vars_raw)
        ddump["mesh_operations"] = ops
        ddump["solve_setups"] = setups
        ddump["mesh_setup_raw"] = [
            sub.to_raw() for sub in des.walk()
            if sub.name == "MeshSetup"]
        dump["designs"][dname] = ddump

    dump["project_variables"] = OrderedDict(
        (n, evaluate_variables(proj_vars_raw)[n])
        for n in proj_vars_raw) if proj_vars_raw else {}

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(dump, fh, indent=2)
        print("\nFull dump (incl. raw MeshSetup tree) -> {0}".format(
            args.json))
    return 0


if __name__ == "__main__":
    sys.exit(main())

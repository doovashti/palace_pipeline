#!/usr/bin/env python3
"""
Check that a Gmsh 2.2 .msh file is CONFORMAL: every surface triangle's
corner nodes must form a face of at least one tetrahedron. Palace/MFEM
requires this and aborts on violation with the cryptic

    MFEM abort: (r,c,f) = (...)
     ... in function: int mfem::STable3D::operator()(int, int, int)

Run this LOCALLY after meshing, before qsub:

    python mesh_conformity_check.py device_run1.msh

Exit code 0 = conformal (safe to submit), 1 = broken (do NOT submit).
On failure it reports, per physical surface group, how many triangles
are orphaned (not a tet face) and prints the coordinates of a few, so
you can see WHERE the mesh is broken (e.g. clustered at a fuzzy-boolean
repair site, or on one particular sheet).

Handles linear and curved elements (tri3/tri6, tet4/tet10) -- only
corner nodes matter for conformity.
"""

import sys
from collections import defaultdict

try:
    import numpy as np
except ImportError:
    np = None

TRI_TYPES = {2: 3, 9: 6}     # type -> nodes stored (corners first)
TET_TYPES = {4: 4, 11: 10}


def parse_msh22(path):
    """Minimal Gmsh 2.2 ASCII parser: nodes, elements, physical names."""
    phys_names = {}
    nodes = {}
    tris = []    # (phys_tag, (n1, n2, n3))
    tets = []    # (n1, n2, n3, n4)
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        line = fh.readline()
        while line:
            key = line.strip()
            if key == "$PhysicalNames":
                count = int(fh.readline())
                for _ in range(count):
                    parts = fh.readline().split(maxsplit=2)
                    dim, tag = int(parts[0]), int(parts[1])
                    name = parts[2].strip().strip('"')
                    phys_names[(dim, tag)] = name
            elif key == "$Nodes":
                count = int(fh.readline())
                for _ in range(count):
                    parts = fh.readline().split()
                    nodes[int(parts[0])] = (float(parts[1]),
                                            float(parts[2]),
                                            float(parts[3]))
            elif key == "$Elements":
                count = int(fh.readline())
                for _ in range(count):
                    parts = fh.readline().split()
                    etype = int(parts[1])
                    ntags = int(parts[2])
                    phys = int(parts[3]) if ntags >= 1 else 0
                    conn = [int(x) for x in parts[3 + ntags:]]
                    if etype in TRI_TYPES:
                        tris.append((phys, tuple(conn[:3])))
                    elif etype in TET_TYPES:
                        tets.append(tuple(conn[:4]))
            line = fh.readline()
    return phys_names, nodes, tris, tets


def encode(a, b, c, base):
    """Order-independent face key packed into one int."""
    x, y, z = sorted((a, b, c))
    return (x * base + y) * base + z


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    path = sys.argv[1]
    print(f"Reading {path} ...")
    phys_names, nodes, tris, tets = parse_msh22(path)
    print(f"  {len(nodes)} nodes, {len(tets)} tets, "
          f"{len(tris)} surface triangles")
    if not tets or not tris:
        print("Nothing to check (missing tets or triangles).")
        return 2

    base = max(nodes) + 1
    face_local = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))

    print("Building tet-face table ...")
    if np is not None:
        T = np.asarray(tets, dtype=np.int64)
        keys = []
        for i, j, k in face_local:
            f = np.sort(np.stack([T[:, i], T[:, j], T[:, k]], axis=1),
                        axis=1)
            keys.append((f[:, 0] * base + f[:, 1]) * base + f[:, 2])
        tet_faces = np.unique(np.concatenate(keys))
        tri_keys = np.asarray(
            [encode(a, b, c, base) for _, (a, b, c) in tris],
            dtype=np.int64)
        ok = np.isin(tri_keys, tet_faces)
        bad_idx = np.nonzero(~ok)[0].tolist()
    else:
        tet_faces = set()
        for t in tets:
            for i, j, k in face_local:
                tet_faces.add(encode(t[i], t[j], t[k], base))
        bad_idx = [n for n, (_, (a, b, c)) in enumerate(tris)
                   if encode(a, b, c, base) not in tet_faces]

    if not bad_idx:
        print("\nCONFORMAL: every surface triangle is a tet face. "
              "Safe for Palace.")
        return 0

    print(f"\nNON-CONFORMAL: {len(bad_idx)} of {len(tris)} surface "
          f"triangles are NOT a face of any tetrahedron.")
    print("Palace/MFEM will abort on this mesh (STable3D). "
          "Do NOT submit it.\n")

    by_group = defaultdict(list)
    for n in bad_idx:
        by_group[tris[n][0]].append(n)
    print("Orphan triangles by physical surface group:")
    for phys, idxs in sorted(by_group.items(),
                             key=lambda kv: -len(kv[1])):
        name = phys_names.get((2, phys), "?")
        print(f"  attr {phys:<4} {name:<28} {len(idxs)} orphan(s)")
        for n in idxs[:3]:
            a, b, c = tris[n][1]
            cx = sum(nodes[v][0] for v in (a, b, c)) / 3.0
            cy = sum(nodes[v][1] for v in (a, b, c)) / 3.0
            cz = sum(nodes[v][2] for v in (a, b, c)) / 3.0
            print(f"      centroid ({cx:.5g}, {cy:.5g}, {cz:.5g})")
        if len(idxs) > 3:
            print(f"      ... and {len(idxs) - 3} more")

    # Duplicate-node scan around the orphans: the signature of a
    # fuzzy-boolean/fragment failure is a coincident-but-distinct copy
    # of the surface -- two node ids at the same coordinates.
    bad_nodes = {v for n in bad_idx for v in tris[n][1]}
    coord_map = defaultdict(list)
    for nid, xyz in nodes.items():
        coord_map[(round(xyz[0], 7), round(xyz[1], 7),
                   round(xyz[2], 7))].append(nid)
    dup_hits = sum(1 for v in bad_nodes
                   if len(coord_map[(round(nodes[v][0], 7),
                                     round(nodes[v][1], 7),
                                     round(nodes[v][2], 7))]) > 1)
    if dup_hits:
        print(f"\n{dup_hits}/{len(bad_nodes)} orphan-triangle nodes "
              f"have a DUPLICATE node at the same coordinates: the "
              f"surface exists twice (coincident copies) -- a boolean/"
              f"fragment failure. Remesh with different --boolean-tol "
              f"(try 0 if the design never had a PLC error, larger if "
              f"it did), or fix the source geometry.")
    else:
        print(f"\nNo duplicate nodes among the orphan triangles: the "
              f"triangles genuinely float free of the volume mesh "
              f"(a surface meshed but not stitched into any volume).")
    return 1


if __name__ == "__main__":
    sys.exit(main())

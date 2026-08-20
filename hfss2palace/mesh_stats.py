#!/usr/bin/env python3
"""
mesh_stats.py

Independent element-size and quality report for a Gmsh 2.2 ASCII mesh.
No gmsh dependency -- parses the file directly, so it can be run
anywhere the .msh file lives.

Answers the question Palace's one-line "h min / h max" cannot: what are
the actual element edge lengths, are any elements larger than the
requested Mesh.MeshSizeMax, and where are they.

Usage:
    python3 mesh_stats.py galvanic_with_pin.msh [--max-size 1.0]
                                                [--top 20]

    --max-size  Requested global size cap in mesh units (mm here).
                Elements exceeding it are listed explicitly.
    --top       How many of the largest elements to print.

Reports, in mesh units (mm for this pipeline):
    * edge-length distribution (min / percentiles / max)
    * element volume distribution
    * aspect ratio  = longest edge / shortest edge, per element
    * the N largest elements with their centroids, so they can be
      located in Gmsh with a clipping plane
    * count and fraction of elements above --max-size
"""
from __future__ import annotations

import argparse
import math
import sys

TET4 = 4          # Gmsh element type: 4-node tetrahedron
TET10 = 11        # Gmsh element type: 10-node (second-order) tetrahedron

EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def read_msh22(path):
    """Return (nodes: {id: (x,y,z)}, tets: [(n0,n1,n2,n3), ...])."""
    nodes = {}
    tets = []
    curvature_order = 1

    with open(path, "r") as handle:
        section = None
        remaining = 0

        for line in handle:
            line = line.strip()
            if not line:
                continue

            if line.startswith("$End"):
                section = None
                continue
            if line.startswith("$"):
                section = line[1:]
                remaining = -1          # next line is the count
                continue

            if section == "Nodes":
                if remaining == -1:
                    remaining = int(line)
                    continue
                parts = line.split()
                nodes[int(parts[0])] = (
                    float(parts[1]), float(parts[2]), float(parts[3])
                )

            elif section == "Elements":
                if remaining == -1:
                    remaining = int(line)
                    continue
                parts = [int(v) for v in line.split()]
                etype, ntags = parts[1], parts[2]
                conn = parts[3 + ntags:]
                if etype == TET4:
                    tets.append(tuple(conn[:4]))
                elif etype == TET10:
                    curvature_order = 2
                    tets.append(tuple(conn[:4]))   # corner nodes only

    return nodes, tets, curvature_order


def tet_volume(p0, p1, p2, p3):
    a = [p1[i] - p0[i] for i in range(3)]
    b = [p2[i] - p0[i] for i in range(3)]
    c = [p3[i] - p0[i] for i in range(3)]
    det = (a[0] * (b[1] * c[2] - b[2] * c[1])
           - a[1] * (b[0] * c[2] - b[2] * c[0])
           + a[2] * (b[0] * c[1] - b[1] * c[0]))
    return abs(det) / 6.0


def dist(p, q):
    return math.sqrt(sum((p[i] - q[i]) ** 2 for i in range(3)))


def percentile(sorted_values, fraction):
    if not sorted_values:
        return float("nan")
    index = min(len(sorted_values) - 1,
                max(0, int(round(fraction * (len(sorted_values) - 1)))))
    return sorted_values[index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh")
    parser.add_argument("--max-size", type=float, default=None,
                        help="Requested global Mesh.MeshSizeMax, in mesh units")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    nodes, tets, curvature_order = read_msh22(args.mesh)
    if not tets:
        sys.exit("No tetrahedra found -- is this a Gmsh 2.2 ASCII file?")

    print(f"file:              {args.mesh}")
    print(f"nodes:             {len(nodes)}")
    print(f"tetrahedra:        {len(tets)}")
    print(f"mesh order:        {curvature_order} "
          f"({'straight-sided' if curvature_order == 1 else 'curved'})")

    max_edges = []
    min_edges = []
    volumes = []
    aspects = []
    records = []

    for tet in tets:
        try:
            pts = [nodes[n] for n in tet]
        except KeyError:
            continue

        lengths = [dist(pts[i], pts[j]) for i, j in EDGES]
        longest, shortest = max(lengths), min(lengths)
        volume = tet_volume(*pts)

        max_edges.append(longest)
        min_edges.append(shortest)
        volumes.append(volume)
        aspects.append(longest / shortest if shortest > 0 else float("inf"))

        centroid = tuple(
            sum(p[i] for p in pts) / 4.0 for i in range(3)
        )
        records.append((longest, volume, aspects[-1], centroid))

    max_edges.sort()
    min_edges.sort()
    volumes.sort()
    aspects.sort()

    def line(label, values, fmt="{:.6g}"):
        print(f"  {label:<22}"
              f"min={fmt.format(values[0]):>12}  "
              f"p50={fmt.format(percentile(values, 0.50)):>12}  "
              f"p99={fmt.format(percentile(values, 0.99)):>12}  "
              f"max={fmt.format(values[-1]):>12}")

    print("\nPer-element statistics (mesh units):")
    line("longest edge", max_edges)
    line("shortest edge", min_edges)
    line("volume", volumes)
    line("aspect (long/short)", aspects)

    if args.max_size is not None:
        over = [r for r in records if r[0] > args.max_size]
        fraction = 100.0 * len(over) / len(records)
        print(f"\nElements with longest edge > {args.max_size:g}: "
              f"{len(over)} ({fraction:.4f}%)")
        if over:
            print("  -> Mesh.MeshSizeMax clamps the size FIELD; Delaunay "
                  "refinement does not guarantee every edge obeys it. "
                  "Overshoot up to ~2x in open regions is ordinary. "
                  "Check the aspect ratios below instead.")

    print(f"\n{args.top} largest elements by longest edge:")
    print(f"  {'longest edge':>14} {'volume':>14} {'aspect':>10}   centroid")
    for longest, volume, aspect, centroid in sorted(
        records, key=lambda r: -r[0]
    )[:args.top]:
        print(f"  {longest:>14.6g} {volume:>14.6g} {aspect:>10.4g}   "
              f"({centroid[0]:.4g}, {centroid[1]:.4g}, {centroid[2]:.4g})")

    print(f"\n{args.top} worst elements by aspect ratio "
          f"(long edge / short edge):")
    print(f"  {'aspect':>10} {'longest edge':>14} {'volume':>14}   centroid")
    worst = sorted(records, key=lambda r: -r[2])[:args.top]
    for longest, volume, aspect, centroid in worst:
        print(f"  {aspect:>10.4g} {longest:>14.6g} {volume:>14.6g}   "
              f"({centroid[0]:.4g}, {centroid[1]:.4g}, {centroid[2]:.4g})")

    # Spatial spread of the badly-shaped population, so it can be told
    # apart from a localized cluster on one boundary.
    bad = [r for r in records if r[2] > 50.0]
    if bad:
        print(f"\nElements with aspect > 50: {len(bad)} "
              f"({100.0 * len(bad) / len(records):.4f}%)")
        for axis, label in enumerate("xyz"):
            values = sorted(r[3][axis] for r in bad)
            print(f"  {label}: min={values[0]:.4g}  "
                  f"p50={percentile(values, 0.50):.4g}  "
                  f"max={values[-1]:.4g}")


if __name__ == "__main__":
    main()

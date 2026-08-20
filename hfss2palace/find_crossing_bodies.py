#!/usr/bin/env python3
"""
Find which named HFSS bodies own geometry at a given point.

Usage:
    python find_crossing_bodies.py model.step 1.04746 11.0984 0
    python find_crossing_bodies.py model.step 1.04746 11.0984 0 --margin 0.05

When the mesher dies with
    PLC Error: Two segments intersect at point (x, y, z)
or
    PLC Error: A segment and a facet intersect at point
this script tells you WHICH objects to look at in HFSS: it lists every
named body whose bounding box (and, for finer localisation, whose
individual FACE bounding boxes) contains the offending point.

The bodies whose *edges* actually cross will both appear here -- usually
two coplanar sheets (or a sheet and a solid face) on the chip plane.
Fix them in HFSS (unite, trim, or align so the edges share vertices),
re-export, and re-mesh.

Pure STEP-text parsing via step_bodies.py -- no gmsh, runs in seconds.
"""

import argparse
import sys

from step_bodies import read_step_bodies


def box_contains(box, pt, margin):
    return (box[0] - margin <= pt[0] <= box[3] + margin
            and box[1] - margin <= pt[1] <= box[4] + margin
            and box[2] - margin <= pt[2] <= box[5] + margin)


def box_str(box):
    return (f"[{box[0]:.4g}, {box[1]:.4g}, {box[2]:.4g}] .. "
            f"[{box[3]:.4g}, {box[4]:.4g}, {box[5]:.4g}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", help="STEP file exported from HFSS")
    ap.add_argument("x", type=float)
    ap.add_argument("y", type=float)
    ap.add_argument("z", type=float)
    ap.add_argument("--margin", type=float, default=0.02,
                    help="slack around the point in model units "
                         "(default 0.02 -- 20 um if units are mm)")
    args = ap.parse_args()

    pt = (args.x, args.y, args.z)
    print(f"Reading {args.step} ...")
    bodies = read_step_bodies(args.step)
    print(f"{len(bodies)} named bodies parsed.\n")
    print(f"Bodies whose bounding box contains "
          f"({pt[0]:g}, {pt[1]:g}, {pt[2]:g}) +/- {args.margin:g}:\n")

    hits = []
    for name, info in sorted(bodies.items()):
        if not box_contains(info["bbox"], pt, args.margin):
            continue
        touching_faces = [
            fb for fb in info["face_bboxes"]
            if box_contains(fb, pt, args.margin)
        ]
        hits.append((name, info, touching_faces))

    if not hits:
        print("  (none -- try a larger --margin, or check that the "
              "point is in the STEP file's units)")
        return 1

    # Bodies with a FACE box containing the point are the prime
    # suspects; body-box-only hits are just big enclosures (vacuum box,
    # substrate) that trivially contain everything.
    hits.sort(key=lambda h: (len(h[2]) == 0, -len(h[2])))

    for name, info, faces in hits:
        marker = "  <-- SUSPECT" if faces else ""
        print(f"  {name}  ({info['geo_type']}, "
              f"material={info['material']}){marker}")
        print(f"      body box: {box_str(info['bbox'])}")
        if faces:
            print(f"      {len(faces)} face box(es) contain the point:")
            for fb in faces[:6]:
                print(f"        {box_str(fb)}")
            if len(faces) > 6:
                print(f"        ... and {len(faces) - 6} more")
        print()

    suspects = [name for name, _info, faces in hits if faces]
    small = [name for name in suspects
             if max(bodies[name]["size"]) < 50]  # skip vacuum/substrate
    print("Most likely offenders (small bodies with a face at the "
          "point):")
    for name in (small or suspects):
        print(f"  - {name}")
    print("\nOpen HFSS, zoom to the point, and check how these "
          "objects' edges meet. Overlapping-but-not-identical edges "
          "must be made to coincide (unite the objects, or trim/align "
          "so edges share vertices).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

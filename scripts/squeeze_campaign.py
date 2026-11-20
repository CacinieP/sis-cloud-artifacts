"""Full-packing fq squeeze campaign: smooth-continuum search complementing hop.

For each n: try targets L0 - delta for a ladder of deltas; each target runs
skeleton.quench (evand's fq rectangle mode) from the current packing with
loosen/kick/seed jitter.  Any result with side < L0 - 1e-9 that certifies
(official metric mirror after float64 round trip + 1e-12 dilation) is kept.

Usage: python squeeze_campaign.py SUB.csv TARGETS.txt SECONDS_PER_N PROCS [SEED]
Appends certified improvements as JSON lines to work/sq_out/results.jsonl
(each line: n, side, rows in radians).  The harvester reads them back per n.
"""
import csv
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

KIT = "/Users/caciniep/Developer/kaggle/squares-in-squares/refs/alexchilton_squares-in-squares-starter-kit"
ROOT = "/Users/caciniep/Developer/kaggle/squares-in-squares"
sys.path.insert(0, KIT)
import ensemble  # noqa: E402
import skeleton as S  # noqa: E402

OUT_JSONL = ROOT + "/work/sq_out/results.jsonl"


def append_result(n, side, rows):
    os.makedirs(os.path.dirname(OUT_JSONL), exist_ok=True)
    line = json.dumps({"n": n, "side": side,
                       "rows": [[i, repr(float(x)), repr(float(y)), repr(float(t))]
                                for i, (x, y, t) in enumerate(rows)]})
    with open(OUT_JSONL, "a") as fh:
        fh.write(line + "\n")


def run(job):
    n, sq, seconds, seed = job
    t0 = time.time()
    L0 = float(ensemble.side(sq))
    best = None
    deltas = (0.0005, 0.002, 0.008)
    k = 0
    while time.time() - t0 < seconds:
        delta = deltas[k % len(deltas)]
        target = L0 - delta
        loosen = 1.01 + 0.01 * (k % 3)
        kick = (0.0, 0.005, 0.02)[k % 3]
        out_sq, f = S.quench(sq, target, target, loosen=loosen, kick=kick,
                             seed=seed * 1000 + k, outer=60)
        k += 1
        if not np.isfinite(f) or f > 1e-7:
            continue
        L1 = float(ensemble.side(out_sq))
        if L1 >= L0 - 1e-9:
            continue
        if best is not None and L1 >= best[0] - 1e-12:
            continue
        cx = (out_sq[:, 0].min() + out_sq[:, 0].max()) / 2
        cy = (out_sq[:, 1].min() + out_sq[:, 1].max()) / 2
        new = out_sq.copy()
        new[:, 0] = cx + (new[:, 0] - cx) * (1 + 1e-12)
        new[:, 1] = cy + (new[:, 1] - cy) * (1 + 1e-12)
        r = ensemble.certify(n, new)
        if r is None:
            continue
        if best is None or r[0] < best[0]:
            best = (r[0], new)
            append_result(n, float(r[0]), new)
            print(f"n={n}: CERTIFIED {r[0]:.9f} vs {L0:.9f} gain {(L0 - r[0]) / math.sqrt(n):.2e} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    if best is None:
        return n, 0.0
    return n, (L0 - best[0]) / math.sqrt(n)


def main():
    sub, targets, seconds, procs = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    global OUT_JSONL
    if len(sys.argv) > 6:
        OUT_JSONL = sys.argv[6]
    sq_by_n = {}
    with open(sub) as f:
        for r in csv.DictReader(f):
            n = int(r["n"])
            sq_by_n.setdefault(n, []).append((float(r["x"]), float(r["y"]), float(r["angle"])))
    ns = [int(l) for l in open(targets) if l.strip().isdigit()]
    jobs = []
    for n in ns:
        sq = np.array(sq_by_n[n], dtype=float)
        jobs.append((n, sq, seconds, seed))
    print(f"squeeze campaign: {len(jobs)} puzzles, {seconds}s each, {procs} procs", flush=True)
    total = 0.0
    with Pool(procs) as pool:
        for n, gain in pool.imap_unordered(run, jobs):
            total += gain
            if gain > 0:
                print(f"KEEP n={n} gain {gain:.2e} | total {total:.6f}", flush=True)
    print(f"DONE total certified score gain {total:.6f}", flush=True)


if __name__ == "__main__":
    main()

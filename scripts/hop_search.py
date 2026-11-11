"""Basin hopping on our current best packings with evand's Rust engine (external/evand/search/packer, MIT):
`packer hop --in start --iters I --out best` = SAT-overlap L-BFGS at a fixed side, random perturbations, Metropolis,
and every time the configuration is feasible it squeezes and sets the next target just below the best side.

Per target n: start = our current packing of n (translated to the box [0, L]^2), one hop job for MINUTES (hard
timeout; the engine writes its best feasible packing whenever it improves). A result counts only if its side is
below the start by > 1e-9 AND the official metric certifies it after a float64 round trip (dilated by 1 + 1e-12).
Accepted packings go to out/hop/<n>.csv (submission rows for that n); one log line per n in the log.

    .venv/bin/python -u src/hop_search.py SUB.csv TARGETS.txt MINUTES PROCS [SEED] > logs/hop_<tag>.log
"""
import math, os, subprocess, sys, time
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ensemble import side, certify

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PACKER = os.environ.get("PACKER_BIN", os.path.join(ROOT, "external/evand/search/packer/target/release/packer"))
WORK = os.environ.get("HOP_WORK", os.path.join(ROOT, "out/hop"))


def corners(sq):
    x, y, t = sq[:, 0], sq[:, 1], sq[:, 2]
    c, s = np.cos(t) * .5, np.sin(t) * .5
    return np.r_[x + c - s, x - c - s, x - c + s, x + c + s], np.r_[y + s + c, y - s + c, y - s - c, y + s - c]


def run(job):
    n, sq, minutes, seed = job
    X, Y = corners(sq)
    L0 = side(sq)
    sq = sq.copy(); sq[:, 0] -= X.min(); sq[:, 1] -= Y.min()
    start, best = f"{WORK}/start_{n}_{seed}.txt", f"{WORK}/best_{n}_{seed}.txt"
    with open(start, "w") as f:
        f.write(f"{n} {L0 + 1e-12:.17f}\n")
        for x, y, t in sq:
            f.write(f"{x:.17f} {y:.17f} {math.degrees(t) % 90.0:.15f}\n")
    t0 = time.time()
    try:
        subprocess.run([PACKER, "hop", "--in", start, "--seed", str(seed), "--iters", "1000000000", "--out", best, "--quiet"],
                       timeout=minutes * 60, capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        pass
    if not os.path.exists(best):
        return n, L0, None, "no feasible output", time.time() - t0
    rows = [l.split() for l in open(best) if l.strip() and not l.startswith("#")][1:]
    new = np.array([[float(a), float(b), math.radians(float(c))] for a, b, c, *_ in rows])
    L1 = side(new)
    if L1 >= L0 - 1e-9:
        return n, L0, L1, "no gain", time.time() - t0
    cx, cy = (new[:, 0].min() + new[:, 0].max()) / 2, (new[:, 1].min() + new[:, 1].max()) / 2
    new[:, 0] = cx + (new[:, 0] - cx) * (1 + 1e-12); new[:, 1] = cy + (new[:, 1] - cy) * (1 + 1e-12)
    r = certify(n, new)
    if r is None:
        return n, L0, L1, "gain but NOT certified", time.time() - t0
    prev = f"{WORK}/{n}.csv"                   # never overwrite a better earlier result for this n
    if os.path.exists(prev):
        old = pd.read_csv(prev).sort_values("square_id")[["x", "y", "angle"]].to_numpy(float)
        if side(old) <= r[0]:
            return n, L0, r[0], "CERTIFIED but an earlier run is better", time.time() - t0
    with open(prev, "w") as f:
        f.write("n,square_id,x,y,angle\n")
        for i, (x, y, t) in enumerate(new):
            f.write(f"{n},{i},{float(x)!r},{float(y)!r},{float(t)!r}\n")
    return n, L0, r[0], "CERTIFIED", time.time() - t0


if __name__ == "__main__":
    sub, targets, minutes, procs = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    os.makedirs(WORK, exist_ok=True)
    df = pd.read_csv(sub)
    ns = [int(l) for l in open(targets) if l.strip().isdigit()]
    jobs = [(n, df[df.n == n].sort_values("square_id")[["x", "y", "angle"]].to_numpy(float), minutes, seed) for n in ns]
    print(f"{time.strftime('%H:%M:%S')} hop on {len(jobs)} puzzles, {minutes} min each, {procs} procs, seed {seed}", flush=True)
    gain = 0.0
    with Pool(procs) as pool:
        for n, L0, L1, why, sec in pool.imap_unordered(run, jobs):
            d = (L0 - L1) / math.sqrt(n) if why == "CERTIFIED" else 0.0
            gain += d
            print(f"{time.strftime('%H:%M:%S')} n={n} start {L0:.9f} best {L1 if L1 is None else f'{L1:.9f}'} {why} "
                  f"score gain {d:.2e} ({sec:.0f}s) | total gain {gain:.6f}", flush=True)
    print(f"{time.strftime('%H:%M:%S')} DONE total certified score gain {gain:.6f}", flush=True)

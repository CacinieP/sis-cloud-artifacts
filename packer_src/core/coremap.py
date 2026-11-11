#!/usr/bin/env python3
"""Side-pinning cores of records vs their exact-side alternates (10-18).  Core = squares carrying contact force in the
fixed-angle LP (rational/fixlp.py duals) at the packing's own point; angles frozen, so this is the position-only
first-order stress network (a vertex of the dual set: degenerate packings can have other force patterns).
Alternate aligned to the record by the best of 8 box symmetries + Hungarian on centres.  Per pair: core sizes,
pose-identical squares (centre < 1e-3, angle < 0.01 deg), core overlap.
  coremap.py [--dir runs/shove/ties1018] [--out runs/core/core1.json]"""
import sys, os, glob, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'rational'))
import numpy as np
from scipy.optimize import linear_sum_assignment
import fixlp


def D4(sq, s):
    x, y, t = sq[:, 0], sq[:, 1], sq[:, 2]
    for X, Y, sg in ((x, y, 1), (s - x, y, -1), (x, s - y, -1), (s - x, s - y, 1),
                     (y, x, -1), (s - y, x, 1), (y, s - x, 1), (s - y, s - x, -1)):
        yield np.c_[X, Y, (sg * t) % 90]


def angd(a, b):
    d = np.abs(a - b) % 90
    return np.minimum(d, 90 - d)


def core(sq, s, mode='locked'):
    """mode 'dual': force support of one LP dual vertex (not unique: degenerate); 'locked' (default): squares with no
    first-order motion at fixed side (fixlp.mobility < 1e-7), = the union of all force supports"""
    out, s2 = fixlp.polish(sq, s, iters=4)
    if mode == 'dual':
        st = fixlp.polish.stress
        return st > 1e-7 * max(st.max(), 1e-30), s2 - s
    return fixlp.mobility(sq, s) < 1e-7, s2 - s


def align(rec, alt, s):
    best = None
    for A in D4(alt, s):
        C = np.sqrt(((rec[:, None, :2] - A[None, :, :2]) ** 2).sum(-1))
        r, c = linear_sum_assignment(C)
        m = C[r, c].mean()
        if best is None or m < best[0]:
            best = (m, A, c)
    return best


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dir', default='runs/shove/ties1018')
    ap.add_argument('--out', default='runs/core/core2.json')
    a = ap.parse_args()
    from pk.store import Store
    from pk.packing import read
    st = Store()
    rows = []
    cache = {}
    for f in sorted(glob.glob(os.path.join(a.dir, '*.txt'))):
        alt = read(f); n = alt.n
        if n not in cache:
            rec = st.best(n)
            cache[n] = (rec, *core(rec.sq, rec.s))
        rec, crec, drec = cache[n]
        if abs(alt.s - rec.s) > 1e-7:
            continue
        m, A, perm = align(rec.sq, alt.sq, rec.s)
        calt, dalt = core(A, rec.s)
        same = (np.sqrt(((rec.sq[:, :2] - A[perm, :2]) ** 2).sum(-1)) < 1e-3) & (angd(rec.sq[:, 2], A[perm, 2]) < 0.01)
        cr = crec; ca = calt[perm]                                   # alt core in record indexing
        rows.append(dict(file=os.path.basename(f), n=n, s=rec.s, md=float(m), core_rec=int(cr.sum()), core_alt=int(ca.sum()),
                         same=int(same.sum()), core_same=int((cr & ca & same).sum()),
                         core_rec_moved=int((cr & ~same).sum()), jacc=float((cr & ca & same).sum() / max(1, (cr | ca).sum())),
                         lp_drop_rec=float(drec), lp_drop_alt=float(dalt)))
        r = rows[-1]
        print(f"{r['file']:16s} n={n} md={m:.3f} core rec {r['core_rec']:3d} alt {r['core_alt']:3d} | identical squares {r['same']:3d}/{n} "
              f"| shared core {r['core_same']:3d}  rec-core moved {r['core_rec_moved']:3d}  J={r['jacc']:.2f}  lpdrop {drec:+.1e}/{dalt:+.1e}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(rows, open(a.out, 'w'), indent=0)


if __name__ == '__main__':
    main()

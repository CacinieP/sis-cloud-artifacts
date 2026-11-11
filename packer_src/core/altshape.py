#!/usr/bin/env python3
"""What do exact-side alternates look like (10-18)?  Align each alternate to its record (best of 8 symmetries +
Hungarian), then split the moved squares (pose change > 1e-3) into rigid groups: same displacement (1e-4) and same
angle change (1e-3 deg).  Also: angle changes, spatial components of the moved set, how many moved squares were loose
(fixlp.mobility > 1e-7) in the record.
  altshape.py [--dir runs/shove/ties1018] [--extra 'runs/symsub/dup3/ref/n*/dup_*'] [--out runs/core/altshape.json]"""
import sys, os, glob, json, argparse, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'rational'))
import numpy as np
from coremap import align, angd
import fixlp


def groups(D, tol=1e-4):
    """greedy clustering of displacement rows (dx, dy, dth)"""
    lab = -np.ones(len(D), int); g = 0
    for i in range(len(D)):
        if lab[i] >= 0:
            continue
        m = (np.abs(D - D[i]) < [tol, tol, 1e-3]).all(1) & (lab < 0)
        lab[m] = g; g += 1
    return lab


def components(P, idx, link=1.5):
    idx = list(idx); seen = set(); comps = 0
    for s in idx:
        if s in seen:
            continue
        comps += 1; st = [s]; seen.add(s)
        while st:
            u = st.pop()
            for v in idx:
                if v not in seen and np.hypot(*(P[u, :2] - P[v, :2])) < link:
                    seen.add(v); st.append(v)
    return comps


def analyse(rec, alt, mob):
    m, A, perm = align(rec.sq, alt.sq, rec.s)
    B = A[perm]
    dth = ((B[:, 2] - rec.sq[:, 2] + 45) % 90) - 45
    D = np.c_[B[:, :2] - rec.sq[:, :2], dth]
    moved = (np.hypot(D[:, 0], D[:, 1]) > 1e-3) | (np.abs(dth) > 0.01)
    idx = np.nonzero(moved)[0]
    lab = groups(D[idx]) if len(idx) else np.array([], int)
    sizes = sorted(collections.Counter(lab).values(), reverse=True)
    big = sum(x for x in sizes if x >= 3)
    rotated = int((np.abs(dth[idx]) > 0.01).sum())
    loose = int((mob[idx] > 1e-7).sum())
    disp = [tuple(np.round(D[idx][lab == g][0], 4)) for g in range(len(sizes))] if len(idx) else []
    return dict(n=rec.n, md=float(m), moved=int(len(idx)), groups=len(sizes), sizes=sizes[:8], in_big_groups=int(big),
                singletons=sum(1 for x in sizes if x == 1), rotated=rotated, loose_moved=loose,
                comps=components(rec.sq, idx), top_disp=sorted(disp, key=lambda d: -np.hypot(d[0], d[1]))[:4])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dir', default='runs/shove/ties1018')
    ap.add_argument('--extra', default='runs/symsub/dup3/ref/n*/dup_*'); ap.add_argument('--out', default='runs/core/altshape.json')
    a = ap.parse_args()
    from pk.store import Store
    from pk.packing import read
    st = Store(); cache = {}; R = []
    files = sorted(glob.glob(os.path.join(a.dir, '*.txt')))
    for d in sorted(glob.glob(a.extra)):                             # dup3 refined bests at the record side
        best = None
        for l in open(os.path.join(d, 'archive.jsonl')):
            e = json.loads(l)
            if not e.get('unpolished') and (best is None or e['s'] < best['s']):
                best = e
        if best:
            files.append(best['path'])
    for f in files:
        alt = read(f); n = alt.n
        if n not in cache:
            rec = st.best(n); cache[n] = (rec, fixlp.mobility(rec.sq, rec.s))
        rec, mob = cache[n]
        if abs(alt.s - rec.s) > 1e-7:
            continue
        r = analyse(rec, alt, mob); r['file'] = f; R.append(r)
    json.dump(R, open(a.out, 'w'), indent=0)
    print(len(R), 'exact-side alternates')
    cls = collections.Counter()
    for r in R:
        if r['moved'] == 0:
            c = 'identical'
        elif r['rotated'] == 0 and r['singletons'] <= 2 and r['in_big_groups'] >= 0.8 * r['moved']:
            c = 'rigid blocks (translations)'
        elif r['loose_moved'] >= 0.8 * r['moved']:
            c = 'loose squares only'
        else:
            c = 'mixed / complicated'
        r['class'] = c; cls[c] += 1
    for c, k in cls.most_common():
        print(f'{k:4d}  {c}')
    print('\nper alternate: n md moved groups sizes rotated loose_moved comps class')
    for r in sorted(R, key=lambda r: (r['class'], r['n'])):
        print(r['n'], f"{r['md']:.3f}", r['moved'], r['groups'], r['sizes'][:6], r['rotated'], r['loose_moved'], r['comps'], r['class'][:12],
              os.path.basename(os.path.dirname(r['file'])) if 'dup3' in r['file'] else os.path.basename(r['file']))
    json.dump(R, open(a.out, 'w'), indent=0)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""The s(174) strip and its bends (10-18; Evan: "that wonky bend is interesting ... can we place that bend anywhere in
the strip?  move pieces around in the local area and see if it relaxes?").
Bases: the record (Göbel, 13 + 1/sqrt 2) + its 3 exact-side alternates (ties1018 n-174_*), aligned to the record.
Fact (fixlp.mobility): all 16 strip squares are loose in the record (slack >= 0.2); the uprights carry the side.
  E1 insert: one extra square (45 deg) at each of the 17 gaps along the strip (ordered along (1,1)), harden reharden
     (r0 0.1 / 0.2; box held at the record side, or free while soft), free polish -> s(175) candidates (record
     13.7641743).
  E2 melt: a window (radius 1.5 / 2.5) around each strip square: its squares deleted and the same number re-inserted at
     random poses inside it, harden with the box held, polish -> 174 outcomes: side, bends (non-45 tilted squares).
  bend174.py e1|e2 --out runs/core/bend174
"""
import sys, os, json, argparse, subprocess, tempfile, collections
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, 'core')); sys.path.insert(0, os.path.join(HERE, 'symsub'))
import numpy as np
from coremap import align
from symharden import write_cfg, HARDEN


def bases():
    from pk.store import Store
    from pk.packing import read
    rec = Store().best(174)
    out = [('rec', rec.sq)]
    for k in range(3):
        a = read(os.path.join(HERE, f'runs/shove/ties1018/n-174_{k}.txt'))
        m, A, perm = align(rec.sq, a.sq, rec.s)
        out.append((f'alt{k}', A))
    return rec.s, out


def strip_order(sq):
    t = sq[:, 2] % 90
    tilt = np.abs(((t + 45) % 90) - 45) > 1e-3
    idx = np.nonzero(tilt)[0]
    proj = (sq[idx, 0] + sq[idx, 1]) / np.sqrt(2)
    return idx[np.argsort(proj)]


def relax(s, sq, seed, r0, box):
    from pk.engine import fq_quench, POLISH
    from pk.packing import read
    with tempfile.TemporaryDirectory() as tmp:
        a, b = os.path.join(tmp, 'a'), os.path.join(tmp, 'b')
        write_cfg(a, s, sq)
        args = [HARDEN, 'reharden', '--in', a, '--out', b, '--seed', str(seed), '--r0', str(r0)]
        args += ['--hold-move', '1'] if box == 'free' else ['--hold', '20000']
        subprocess.run(args, capture_output=True, text=True, timeout=900)
        if not os.path.exists(b):
            return None
        run = fq_quench(b, POLISH, tmp=tmp, timeout=900)
        p = read(run.out) if run.out else None
    if p is None or p.max_pen() > 1e-9:
        return None
    t = p.sq[:, 2] % 90
    tilt = np.abs(((t + 45) % 90) - 45) > 1e-3
    bend = int((tilt & (np.abs(t - 45) > 0.5)).sum())
    return float(p.s), p.n, int(tilt.sum()), bend, p.sq.tolist()


def e1_job(t):
    name, s, sq, k, r0, box, seed = t
    sq = np.array(sq); order = strip_order(sq)
    P = sq[order, :2]
    if k == 0:
        c = P[0] - (P[1] - P[0])
    elif k == len(P):
        c = P[-1] + (P[-1] - P[-2])
    else:
        c = 0.5 * (P[k - 1] + P[k])
    c = np.clip(c, 0.5, s - 0.5)
    x = np.r_[sq, [[c[0], c[1], 45.0]]]
    r = relax(s, x, seed, r0, box)
    return dict(base=name, k=k, r0=r0, box=box, seed=seed, res=None if r is None else r[:4], sq=None if r is None else r[4])


def e2_job(t):
    name, s, sq, k, rad, seed = t
    sq = np.array(sq); order = strip_order(sq); rng = np.random.default_rng(seed)
    c = sq[order[k], :2]
    inw = np.hypot(sq[:, 0] - c[0], sq[:, 1] - c[1]) < rad
    m = int(inw.sum()); keep = sq[~inw]
    ang = rng.uniform(0, 2 * np.pi, m); rr = rad * np.sqrt(rng.random(m))
    new = np.c_[np.clip(c[0] + rr * np.cos(ang), 0.5, s - 0.5), np.clip(c[1] + rr * np.sin(ang), 0.5, s - 0.5), rng.uniform(0, 90, m)]
    r = relax(s, np.r_[keep, new], seed, 0.15, 'held')
    return dict(base=name, k=k, rad=rad, seed=seed, removed=m, res=None if r is None else r[:4], sq=None if r is None else r[4])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('cmd'); ap.add_argument('--out', default='runs/core/bend174')
    a = ap.parse_args()
    from pk import tasks
    os.makedirs(a.out, exist_ok=True)
    s, B = bases()
    if a.cmd == 'e1':
        T = [(nm, s, sq.tolist(), k, r0, box, seed) for nm, sq in B for k in range(len(strip_order(sq)) + 1)
             for r0 in (0.1, 0.2) for box in ('held', 'free') for seed in (1, 2)]
        with tasks.Executor() as ex:
            R = list(ex.map(e1_job, T))
        json.dump(R, open(os.path.join(a.out, 'e1.json'), 'w'))
        ok = [r for r in R if r['res']]
        print(len(T), 'runs', len(ok), 'feasible;  s(175) record 13.7641743175, s(174) 13.7071067812')
        for box in ('held', 'free'):
            v = sorted(r['res'][0] for r in ok if r['box'] == box and r['res'][1] == 175)
            print(box, 'best 175 sides', [round(x, 6) for x in v[:8]])
        best = min(ok, key=lambda r: r['res'][0])
        print('best overall', best['base'], best['k'], best['r0'], best['box'], best['res'][:4])
        byk = collections.defaultdict(list)
        for r in ok:
            byk[r['k']].append(r['res'][0])
        print('best by insertion position:', {k: round(min(v), 5) for k, v in sorted(byk.items())})
    else:
        T = [(nm, s, sq.tolist(), k, rad, seed) for nm, sq in B for k in range(len(strip_order(sq)))
             for rad in (1.5, 2.5) for seed in (1, 2, 3)]
        with tasks.Executor() as ex:
            R = list(ex.map(e2_job, T))
        json.dump(R, open(os.path.join(a.out, 'e2.json'), 'w'))
        ok = [r for r in R if r['res']]
        tie = [r for r in ok if r['res'][0] < s + 1e-9]
        print(len(T), 'runs', len(ok), 'feasible;', len(tie), 'at the record side (<= +1e-9)', 'below:', sum(r['res'][0] < s - 1e-9 for r in ok))
        print('ties with bends (non-45 tilted squares), by window position k:')
        bk = collections.defaultdict(list)
        for r in tie:
            bk[r['k']].append(r['res'][3])
        print({k: v for k, v in sorted(bk.items())})
        g = sorted(r['res'][0] - s for r in ok)
        print('outcome gaps: median', f'{np.median(g):+.2e}', 'quartiles', np.round(np.percentile(g, [25, 75]), 4))


if __name__ == '__main__':
    main()

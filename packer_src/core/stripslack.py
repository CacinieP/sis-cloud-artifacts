#!/usr/bin/env python3
"""Consolidated strip slack (10-18): at the record side, with fixed angles, how far can the top-most tilted square of
a diagonal strip slide along (1, 1) (LP min and max of its diagonal coordinate, every other square free, separators
from the current geometry over pairs within 3.0, box walls exact)?  The range = the strip's slack gathered at one end.
  stripslack.py N [N ...]"""
import sys, os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, 'rational'))
import numpy as np, scipy.sparse as sp
from scipy.optimize import linprog
import fixlp


def slide_range(sq, s, which):
    n = len(sq); i, j = fixlp.pairs(sq, 3.0); d, need, gap = fixlp.separators(sq, i, j)
    th = sq[:, 2]
    hx = fixlp.halfw(th, np.tile([1.0, 0.0], (n, 1))); hy = fixlp.halfw(th, np.tile([0.0, 1.0], (n, 1)))
    m = len(i); r = np.arange(m)
    A = sp.csr_matrix((np.concatenate([d[:, 0], d[:, 1], -d[:, 0], -d[:, 1]]),
                       (np.concatenate([r, r, r, r]), np.concatenate([2 * i, 2 * i + 1, 2 * j, 2 * j + 1]))), shape=(m, 2 * n))
    b = -np.minimum(need, need + gap)
    bounds = [(hx[q // 2], s - hx[q // 2]) if q % 2 == 0 else (hy[q // 2], s - hy[q // 2]) for q in range(2 * n)]
    c = np.zeros(2 * n); c[2 * which] = 1; c[2 * which + 1] = 1
    lo = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    hi = linprog(-c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    if lo.status or hi.status:
        return None
    return (-hi.fun - lo.fun) / np.sqrt(2)


if __name__ == '__main__':
    from pk.store import Store
    st = Store()
    for n in map(int, sys.argv[1:]):
        p = st.best(n); sq = p.sq; t = sq[:, 2] % 90
        strip = np.nonzero(np.abs(t - 45) < 1e-3)[0]
        if len(strip) == 0:
            print(n, 'no 45-degree squares'); continue
        top = strip[np.argmax(sq[strip, 0] + sq[strip, 1])]; bot = strip[np.argmin(sq[strip, 0] + sq[strip, 1])]
        print(n, f's={p.s:.9f}', '45deg', len(strip), f'top-square slide range {slide_range(sq, p.s, top):.4f}',
              f'bottom-square {slide_range(sq, p.s, bot):.4f}')

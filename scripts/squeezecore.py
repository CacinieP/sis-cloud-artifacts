from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing
import os
import sys
import time

import numpy as np
from numba import njit
from scipy.optimize import minimize



# ----------------------------------------------------------------------------- numba core

@njit
def _sgn(v):
    if v > 0.0:
        return 1.0
    if v < 0.0:
        return -1.0
    return 0.0


@njit
def find_pairs(p, n, cutoff):
    """All pairs with centre distance < cutoff (O(n^2), fine for n <= 1000)."""
    cut2 = cutoff * cutoff
    cnt = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = p[j] - p[i]
            dy = p[n + j] - p[n + i]
            if dx * dx + dy * dy < cut2:
                cnt += 1
    pi = np.empty(cnt, np.int64)
    pj = np.empty(cnt, np.int64)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = p[j] - p[i]
            dy = p[n + j] - p[n + i]
            if dx * dx + dy * dy < cut2:
                pi[k] = i
                pj[k] = j
                k += 1
    return pi, pj


@njit
def pair_depth(p, n, i, j):
    """SAT penetration depth (positive = overlap) and the minimising axis index."""
    dx = p[j] - p[i]
    dy = p[n + j] - p[n + i]
    ti = p[2 * n + i]
    tj = p[2 * n + j]
    ci = math.cos(ti); si = math.sin(ti)
    cj = math.cos(tj); sj = math.sin(tj)
    best = 1e300
    bk = -1
    for a in range(4):
        if a == 0:
            ux = ci; uy = si
        elif a == 1:
            ux = -si; uy = ci
        elif a == 2:
            ux = cj; uy = sj
        else:
            ux = -sj; uy = cj
        hi = 0.5 * (abs(ux * ci + uy * si) + abs(-ux * si + uy * ci))
        hj = 0.5 * (abs(ux * cj + uy * sj) + abs(-ux * sj + uy * cj))
        f = hi + hj - abs(ux * dx + uy * dy)
        if f < best:
            best = f
            bk = a
    return best, bk


@njit
def energy_grad(p, n, L, pi, pj):
    E = 0.0
    g = np.zeros(3 * n)
    # walls
    for i in range(n):
        x = p[i]; y = p[n + i]; t = p[2 * n + i]
        c = math.cos(t); s = math.sin(t)
        h = 0.5 * (abs(c) + abs(s))
        dh = 0.5 * (-_sgn(c) * s + _sgn(s) * c)
        v = h - x
        if v > 0.0:
            E += v * v; g[i] -= 2.0 * v; g[2 * n + i] += 2.0 * v * dh
        v = x + h - L
        if v > 0.0:
            E += v * v; g[i] += 2.0 * v; g[2 * n + i] += 2.0 * v * dh
        v = h - y
        if v > 0.0:
            E += v * v; g[n + i] -= 2.0 * v; g[2 * n + i] += 2.0 * v * dh
        v = y + h - L
        if v > 0.0:
            E += v * v; g[n + i] += 2.0 * v; g[2 * n + i] += 2.0 * v * dh
    # pairs
    for k in range(pi.shape[0]):
        i = pi[k]; j = pj[k]
        dx = p[j] - p[i]
        dy = p[n + j] - p[n + i]
        if dx * dx + dy * dy >= 2.0:
            continue
        ti = p[2 * n + i]; tj = p[2 * n + j]
        ci = math.cos(ti); si = math.sin(ti)
        cj = math.cos(tj); sj = math.sin(tj)
        best = 1e300; bk = -1
        bux = 0.0; buy = 0.0
        for a in range(4):
            if a == 0:
                ux = ci; uy = si
            elif a == 1:
                ux = -si; uy = ci
            elif a == 2:
                ux = cj; uy = sj
            else:
                ux = -sj; uy = cj
            hi = 0.5 * (abs(ux * ci + uy * si) + abs(-ux * si + uy * ci))
            hj = 0.5 * (abs(ux * cj + uy * sj) + abs(-ux * sj + uy * cj))
            f = hi + hj - abs(ux * dx + uy * dy)
            if f < best:
                best = f; bk = a; bux = ux; buy = uy
        if best <= 0.0:
            continue
        E += best * best
        w = 2.0 * best
        ux = bux; uy = buy
        proj = ux * dx + uy * dy
        s = _sgn(proj)
        # positions
        g[i] += w * s * ux; g[n + i] += w * s * uy
        g[j] -= w * s * ux; g[n + j] -= w * s * uy
        # angles, u fixed
        ai = ux * ci + uy * si; bi = -ux * si + uy * ci
        aj = ux * cj + uy * sj; bj = -ux * sj + uy * cj
        g[2 * n + i] += w * 0.5 * (_sgn(ai) * bi - _sgn(bi) * ai)
        g[2 * n + j] += w * 0.5 * (_sgn(aj) * bj - _sgn(bj) * aj)
        # axis ownership: u rotates with its owner's angle
        if bk == 0 or bk == 1:
            # owner i; du/dt_i = e_i2 if u = e_i1, else -e_i1
            if bk == 0:
                dux = -si; duy = ci
            else:
                dux = -ci; duy = -si
            dhj = 0.5 * (_sgn(aj) * (dux * cj + duy * sj) + _sgn(bj) * (-dux * sj + duy * cj))
            g[2 * n + i] += w * (dhj - s * (dx * dux + dy * duy))
        else:
            if bk == 2:
                dux = -sj; duy = cj
            else:
                dux = -cj; duy = -sj
            dhi = 0.5 * (_sgn(ai) * (dux * ci + duy * si) + _sgn(bi) * (-dux * si + duy * ci))
            g[2 * n + j] += w * (dhi - s * (dx * dux + dy * duy))
    return E, g


@njit
def max_penetration(p, n):
    """Largest SAT penetration over all pairs (float64), and the pair."""
    worst = -1e300; wi = -1; wj = -1
    for i in range(n):
        for j in range(i + 1, n):
            dx = p[j] - p[i]
            dy = p[n + j] - p[n + i]
            if dx * dx + dy * dy >= 2.0:
                continue
            d, _ = pair_depth(p, n, i, j)
            if d > worst:
                worst = d; wi = i; wj = j
    return worst, wi, wj


@njit
def repair(p, n, iters, eps):
    """Gauss-Seidel push-apart of residual penetrations (positions only). Returns max penetration."""
    for it in range(iters):
        worst = -1e300
        for i in range(n):
            for j in range(i + 1, n):
                dx = p[j] - p[i]
                dy = p[n + j] - p[n + i]
                if dx * dx + dy * dy >= 2.0:
                    continue
                d, a = pair_depth(p, n, i, j)
                if d > worst:
                    worst = d
                if d > 0.0:
                    ti = p[2 * n + i]; tj = p[2 * n + j]
                    if a == 0:
                        ux = math.cos(ti); uy = math.sin(ti)
                    elif a == 1:
                        ux = -math.sin(ti); uy = math.cos(ti)
                    elif a == 2:
                        ux = math.cos(tj); uy = math.sin(tj)
                    else:
                        ux = -math.sin(tj); uy = math.cos(tj)
                    s = _sgn(ux * dx + uy * dy)
                    if s == 0.0:
                        s = 1.0
                    m = 0.5 * (d + eps)
                    p[i] -= s * m * ux; p[n + i] -= s * m * uy
                    p[j] += s * m * ux; p[n + j] += s * m * uy
        if worst <= 0.0:
            return worst
    w, _, _ = max_penetration(p, n)
    return w


@njit
def bbox(p, n):
    xmin = 1e300; xmax = -1e300; ymin = 1e300; ymax = -1e300
    for i in range(n):
        x = p[i]; y = p[n + i]; t = p[2 * n + i]
        c = math.cos(t); s = math.sin(t)
        h = 0.5 * (abs(c) + abs(s))
        if x - h < xmin: xmin = x - h
        if x + h > xmax: xmax = x + h
        if y - h < ymin: ymin = y - h
        if y + h > ymax: ymax = y + h
    return xmin, xmax, ymin, ymax


# ----------------------------------------------------------------------------- driver

def _fun(p, n, L, pi, pj):
    return energy_grad(p, n, L, pi, pj)


def solve(p0, n, L, maxiter=600, rounds=4):
    p = p0.copy()
    E = 1e300
    for r in range(rounds):
        pi, pj = find_pairs(p, n, 2.0)
        res = minimize(_fun, p, args=(n, L, pi, pj), jac=True, method='L-BFGS-B',
                       options=dict(maxiter=maxiter, ftol=0.0, gtol=1e-14, maxcor=20))
        p = res.x
        pi, pj = find_pairs(p, n, 2.0)
        E, _ = energy_grad(p, n, L, pi, pj)
        if E < 1e-22:
            break
    return p, E


def normalise(p, n):
    """Shift so the bounding box starts at (0,0); return (p, L_box)."""
    xmin, xmax, ymin, ymax = bbox(p, n)
    q = p.copy()
    q[:n] -= xmin
    q[n:2 * n] -= ymin
    return q, max(xmax - xmin, ymax - ymin)


def rows_from_p(p, n):
    """Decimal rows (repr, full double) -> dilated decimal rows like every submitted packing."""
    rows = []
    for i in range(n):
        t = p[2 * n + i]
        rows.append((i, repr(float(p[i])), repr(float(p[n + i])), '0' if t == 0.0 else repr(float(t))))
    return dilate_rows(rows, DELTA)


def exact_check(rows, n):
    d = [{'square_id': str(i), 'x': x, 'y': y, 'angle': a} for i, x, y, a in rows]
    r1 = evaluate_puzzle(n, d, roundtrip=False)
    r2 = evaluate_puzzle(n, d, roundtrip=True)
    ok = r1[3] == 'ok' and r2[3] == 'ok'
    return ok, (float(r1[2]) if r1[2] else float('inf')), r1[3]


def p_from_rows(rows):
    n = len(rows)
    p = np.zeros(3 * n)
    for i, x, y, a in rows:
        p[i] = float(x); p[n + i] = float(y); p[2 * n + i] = float(a)
    return p


def finish(p, n, rng):
    """Repair residual penetrations, normalise, and return (p, L_box, max_pen)."""
    w = repair(p, n, 300, 1e-12)
    if w > 1e-11:
        return p, float('inf'), w
    q, Lb = normalise(p, n)
    return q, Lb, w


def perturb(p_seed, n, rng, jitter, big_move):
    p0 = p_seed.copy()
    if jitter > 0:
        p0[:2 * n] += rng.normal(0.0, jitter, 2 * n)
        tilted = np.abs(np.sin(2 * p0[2 * n:])) > 1e-9
        p0[2 * n:][tilted] += rng.normal(0.0, jitter, int(tilted.sum()))
    if big_move == 'reinsert' and n > 2:
        # remove the square with the largest current penetration/crowding and re-place it where it fits best
        pi, pj = find_pairs(p0, n, 2.0)
        crowd = np.zeros(n)
        for i, j in zip(pi, pj):
            d, _ = pair_depth(p0, n, int(i), int(j))
            if d > -0.05:
                crowd[i] += 1; crowd[j] += 1
        k = int(np.argmax(crowd + rng.uniform(0, 0.5, n)))
        keep = [i for i in range(n) if i != k]
        q = np.concatenate([p0[keep], p0[[n + i for i in keep]], p0[[2 * n + i for i in keep]]])
        xmin, xmax, ymin, ymax = bbox(q, n - 1)
        p0 = place_new_square(q, n - 1, max(xmax - xmin, ymax - ymin), rng)
    elif big_move == 'rotate' and n > 2:
        # rotate a random cluster of tilted squares by a small common angle
        tilted = np.where(np.abs(np.sin(2 * p0[2 * n:])) > 1e-9)[0]
        if len(tilted):
            c = tilted[rng.integers(len(tilted))]
            cx, cy = p0[c], p0[n + c]
            near = [i for i in tilted if (p0[i] - cx) ** 2 + (p0[n + i] - cy) ** 2 < 4.0]
            da = rng.normal(0.0, 0.05)
            for i in near:
                p0[2 * n + i] += da
    return p0


def try_target(p_seed, n, L_target, rng, jitter, restarts, deadline, big_moves=('none', 'none', 'reinsert', 'rotate')):
    best = None
    for r in range(restarts):
        if time.time() > deadline:
            break
        p0 = perturb(p_seed, n, rng, jitter, big_moves[r % len(big_moves)])
        p, E = solve(p0, n, L_target)
        if E < 1e-22:
            q, Lb, w = finish(p, n, rng)
            if Lb < 1e9 and (best is None or Lb < best[1]):
                best = (q, Lb, w)
    return best


def search_shrink(rec_p, n, L_rec, budget, rng, step0=0.01, step_min=1e-4):
    """Persistent shrink ladder: target L_rec - step; on failure halve the step and widen the jitter;
    on success keep going from the new packing. Uses the whole budget."""
    deadline = time.time() + budget
    cur_p, cur_L = rec_p.copy(), L_rec
    step = step0
    fails = 0
    improved = False
    while time.time() < deadline and step >= step_min:
        jitter = min(0.15, 0.02 * (1 + fails))
        res = try_target(cur_p, n, cur_L - step, rng, jitter=jitter, restarts=4, deadline=deadline)
        if res is not None and res[1] < cur_L - 1e-10:
            cur_p, cur_L = res[0], res[1]
            improved = True
            fails = 0
        else:
            fails += 1
            if fails % 2 == 0:
                step *= 0.5
    return (cur_p, cur_L) if improved else None


def place_new_square(p, n, L, rng, tries=400):
    """Append one square at the position/angle with the least penalty (random scan)."""
    best = None
    q = np.zeros(3 * (n + 1))
    q[:n] = p[:n]; q[n + 1:2 * n + 1] = p[n:2 * n]; q[2 * n + 2:3 * n + 2] = p[2 * n:3 * n]
    for _ in range(tries):
        x = rng.uniform(0.5, L - 0.5); y = rng.uniform(0.5, L - 0.5)
        a = rng.choice([0.0, math.pi / 4, rng.uniform(0, math.pi / 2)])
        q[n] = x; q[2 * n + 1] = y; q[3 * n + 2] = a
        pi, pj = find_pairs(q, n + 1, 2.0)
        E, _ = energy_grad(q, n + 1, L, pi, pj)
        if best is None or E < best[0]:
            best = (E, x, y, a)
    q[n] = best[1]; q[2 * n + 1] = best[2]; q[3 * n + 2] = best[3]
    return q


def search_insert(prev_p, n, L_prev, L_rec, budget, rng):
    """From the record of n-1 (side L_prev), add a square and look for a side below L_rec, then shrink."""
    deadline = time.time() + budget
    best = None
    steps = [0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.004, 0.002]
    gap = L_rec - L_prev
    for frac in steps:
        if time.time() > deadline:
            break
        L_t = L_rec - frac * gap
        if L_t <= L_prev:
            continue
        for _ in range(3):
            if time.time() > deadline:
                break
            q = place_new_square(prev_p, n - 1, L_t, rng)
            res = try_target(q, n, L_t, rng, jitter=0.03, restarts=3, deadline=deadline)
            if res is not None and res[1] < L_rec - 1e-9 and (best is None or res[1] < best[1]):
                best = (res[0], res[1])
        if best is not None:
            more = search_shrink(best[0], n, best[1], max(5.0, deadline - time.time()), rng, step0=0.005)
            if more is not None:
                best = more
            break
    return best


def worker(task):
    n, strategy, rows_n, rows_prev, L_n_dilated, budget, seed = task
    rng = np.random.default_rng(seed)
    t0 = time.time()
    rec_p = p_from_rows(rows_n)
    rec_p, L_rec = normalise(rec_p, n)
    out = dict(n=n, strategy=strategy, L_record=L_n_dilated, improved=False, seconds=0.0)
    try:
        if strategy == 'shrink':
            res = search_shrink(rec_p, n, L_rec, budget, rng)
        else:
            prev_p = p_from_rows(rows_prev)
            prev_p, L_prev = normalise(prev_p, n - 1)
            res = search_insert(prev_p, n, L_prev, L_rec, budget, rng)
        if res is not None:
            q, Lb = res
            rows = rows_from_p(q, n)
            ok, Lx, reason = exact_check(rows, n)
            out.update(L_float=Lb, exact_ok=ok, L_exact=Lx, reason=reason)
            if ok and Lx < L_n_dilated - 1e-9:
                out['improved'] = True
                out['rows'] = rows
    except Exception as e:  # noqa: BLE001
        out['error'] = repr(e)
    out['seconds'] = round(time.time() - t0, 1)
    return out

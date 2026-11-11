"""Fault-line gadget DP (Deotte's composition, rebuilt from docs/fable-fault-dp-design-2026-10-22.md).

n = K^2 + D. Lower-left continent: upright cells on the integer lattice. Upper-right continent: the same lattice
shifted by F = f + 2e in x and y. Along the anti-diagonal fault a chain of gadgets (crops of donor packings, w x h
cells holding w*h + dm squares inside [0, w+f] x [0, h+f]) adds D squares; the box side is K + f + 2e.

1. Extraction: for every donor and each of the 8 symmetries, crop the w*h + dm squares with the smallest overhang
   r = max(0, xhi - w, yhi - h); f is that order statistic. Keep per (w, h, dm, Ybr bin, Xtl bin) the minimum f,
   then drop dominated gadgets.
2. DP over (columns, rows, surplus, Ybr bin of the last gadget) minimising max f; seams: mode 0 (corners touch,
   tip rule), X (one lattice column between), Y (one lattice row between).
3. Compose, certify with the official metric after a float64 round trip, keep if below the current side.

    .venv/bin/python -u src/fault_dp.py BASE.csv OUTDIR [DONOR.csv ...] > logs/fault_<tag>.log\n\nDonors: BASE plus every DONOR file (e.g. couzo.csv, hop.csv, couzo_big.csv = Couzo's witnesses 1000 < n <= 3500).
"""
import glob, math, os, sys, time
from fractions import Fraction
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ensemble import side, certify

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WMAX, DMAX, FMAX, E = 31, 62, 0.97, 1e-9
THR = np.array([0.0, 0.3, 0.45, 0.6, 0.75, 0.9])         # bins for the free tip lengths Ybr / Xtl
NB = len(THR)
SYMS = [(sx, sy, tr) for sx in (1, -1) for sy in (1, -1) for tr in (0, 1)]
T0 = time.time()


def log(*a):
    print(f"{time.strftime('%H:%M:%S')} [{time.time() - T0:6.0f}s]", *a, flush=True)


def load_csv(p):
    out = {}
    for n, g in pd.read_csv(p).groupby("n"):
        sq = g.sort_values("square_id")[["x", "y", "angle"]].to_numpy(float)
        if len(sq) == n and side(sq) < math.ceil(math.sqrt(n)) - 1e-9:
            out[int(n)] = sq
    return out


def load_cert(p):                                                      # Couzo .cert: x y tan(theta/2) as rationals
    rows = [l.split() for l in open(p) if l.strip() and not l.startswith("#")]
    return int(rows[0][0]), np.array([(float(Fraction(x)), float(Fraction(y)), 2 * math.atan(float(Fraction(t)))) for x, y, t in rows[1:]])


def mapped(sq, sym):
    """Donor under a dihedral map, translated so its bounding box starts at (0, 0). Returns (sq, xlo, xhi, ylo, yhi)."""
    sx, sy, tr = sym
    x, y, t = sq[:, 0] * sx, sq[:, 1] * sy, sq[:, 2] * sx * sy
    if tr:
        x, y, t = y, x, -t
    he = 0.5 * (np.abs(np.cos(t)) + np.abs(np.sin(t)))
    x, y = x - (x - he).min(), y - (y - he).min()
    return np.c_[x, y, t], x - he, x + he, y - he, y + he


def extract(donors):
    """donors: list of float arrays. Returns gadget table (G, 8): w, h, dm, f, ybr, xtl, donor, sym."""
    W = np.arange(1, WMAX + 1)
    DM = np.arange(1, DMAX + 1)
    best = {}
    for di, sq in enumerate(donors):
        n = len(sq)
        L = side(sq)
        parts = []
        for si, sym in enumerate(SYMS):
            _, xlo, xhi, ylo, yhi = mapped(sq, sym)
            for h in range(1, min(WMAX, math.ceil(L)) + 1):
                R = np.maximum(np.maximum(xhi[:, None] - W[None, :], (yhi - h)[:, None]), 0.0)
                order = np.argsort(R, axis=0, kind="stable")
                Rs = np.take_along_axis(R, order, axis=0)
                ybr = np.minimum.accumulate(np.where(xhi[order] > W[None, :] + 1e-9, ylo[order], np.inf), axis=0)
                xtl = np.minimum.accumulate(np.where(yhi[order] > h + 1e-9, xlo[order], np.inf), axis=0)
                idx = W[:, None] * h + DM[None, :] - 1                       # (w, dm)
                ok = (idx < n) & (W[:, None] <= math.ceil(L))
                wi, dj = np.nonzero(ok)
                k = idx[wi, dj]
                f = Rs[k, wi]
                m = f <= FMAX
                wi, dj, k, f = wi[m], dj[m], k[m], f[m]
                parts.append(np.c_[wi + 1, np.full(len(f), h), dj + 1, f, np.minimum(ybr[k, wi], 9.0),
                                   np.minimum(xtl[k, wi], 9.0), np.full(len(f), di), np.full(len(f), si)])
        P = np.concatenate(parts) if parts else np.zeros((0, 8))
        if not len(P):
            continue
        qb = np.maximum(np.searchsorted(THR, P[:, 4], side="right") - 1, 0)
        qt = np.maximum(np.searchsorted(THR, P[:, 5], side="right") - 1, 0)
        key = (((P[:, 0] * 32 + P[:, 1]) * 64 + P[:, 2]) * NB + qb) * NB + qt
        o = np.lexsort((P[:, 3], key))
        first = np.r_[True, key[o][1:] != key[o][:-1]]
        for kk, row in zip(key[o][first].astype(np.int64), P[o][first]):
            if kk not in best or row[3] < best[kk][3]:
                best[kk] = row
        if di % 100 == 0:
            log(f"extract donor {di}/{len(donors)}: {len(best)} keys")
    G = np.array(list(best.values()))
    qb = np.maximum(np.searchsorted(THR, G[:, 4], side="right") - 1, 0)
    qt = np.maximum(np.searchsorted(THR, G[:, 5], side="right") - 1, 0)
    M = np.full((WMAX + 2, WMAX + 2, DMAX + 2, NB, NB), np.inf)
    for (w, h, d, f), b, t in zip(G[:, :4], qb, qt):
        M[int(w), int(h), int(d), b, t] = min(M[int(w), int(h), int(d), b, t], f)
    for ax in (0, 1):
        M = np.minimum.accumulate(M, axis=ax)
    for ax in (2, 3, 4):
        M = np.flip(np.minimum.accumulate(np.flip(M, axis=ax), axis=ax), axis=ax)
    keep = G[:, 3] <= M[G[:, 0].astype(int), G[:, 1].astype(int), G[:, 2].astype(int), qb, qt] + 1e-15
    log(f"gadgets: {len(G)} per-key, {keep.sum()} after dominance")
    return G[keep]


def run_dp(G):
    gw, gh, gd = G[:, 0].astype(int), G[:, 1].astype(int), G[:, 2].astype(int)
    gf, gx = G[:, 3], G[:, 5]
    gq = np.maximum(np.searchsorted(THR, G[:, 4], side="right") - 1, 0)
    shape = (WMAX + 1, WMAX + 1, DMAX + 1, NB)
    dp = np.full(shape, np.inf)
    PG = np.full(shape, -1, np.int32); PM = np.zeros(shape, np.int8); PQ = np.zeros(shape, np.int8)
    dp[0, 0, 0, NB - 1] = 0.0

    def relax(cw2, h0, d, q, src, g, mode):
        vals, qprev = src.min(axis=2), src.argmin(axis=2)
        tgt = dp[cw2, h0:h0 + vals.shape[0], d:d + vals.shape[1], q]
        better = vals < tgt
        if better.any():
            tgt[better] = vals[better]
            PG[cw2, h0:h0 + vals.shape[0], d:d + vals.shape[1], q][better] = g
            PM[cw2, h0:h0 + vals.shape[0], d:d + vals.shape[1], q][better] = mode
            PQ[cw2, h0:h0 + vals.shape[0], d:d + vals.shape[1], q][better] = qprev[better]

    for cw in range(WMAX + 1):
        cur = dp[cw]
        if not np.isfinite(cur).any():
            continue
        for g in range(len(G)):
            w, h, d, f = gw[g], gh[g], gd[g], gf[g]
            if cw + w > WMAX:
                continue
            val = np.maximum(cur, f)
            ok0 = (THR[None, None, :] >= f + E) | (gx[g] >= cur + E)
            relax(cw + w, h, d, gq[g], np.where(ok0, val, np.inf)[:WMAX + 1 - h, :DMAX + 1 - d], g, 0)
            if cw == 0:
                continue                                              # a chain starts with its first gadget
            if h + 1 <= WMAX:
                relax(cw + w, h + 1, d, gq[g], val[:WMAX - h, :DMAX + 1 - d], g, 2)       # mode Y
            if cw + w + 1 <= WMAX:
                relax(cw + w + 1, h, d, gq[g], val[:WMAX + 1 - h, :DMAX + 1 - d], g, 1)   # mode X
    return dp, PG, PM, PQ


def chain(dp, PG, PM, PQ, G, K, D):
    sub = dp[:K + 1, :K + 1, D, :]
    if not np.isfinite(sub).any():
        return None, None
    cw, ch, q = np.unravel_index(np.argmin(sub), sub.shape)
    f, d, out = sub[cw, ch, q], D, []
    while cw or ch or d:
        g, mode, qp = PG[cw, ch, d, q], PM[cw, ch, d, q], PQ[cw, ch, d, q]
        out.append((int(g), int(mode)))
        w, h, dd = int(G[g, 0]), int(G[g, 1]), int(G[g, 2])
        cw, ch, d, q = cw - w - (mode == 1), ch - h - (mode == 2), d - dd, qp
    return float(f), out[::-1]


def crop(donor, sym, w, h, dm):
    sq, xlo, xhi, ylo, yhi = mapped(donor, SYMS[sym])
    R = np.maximum(np.maximum(xhi - w, yhi - h), 0.0)
    take = np.argsort(R, kind="stable")[:w * h + dm]
    return sq[take], R[take].max()


def compose(K, D, f, links, G, donors):
    M = len(links)
    F = f + (M + 2) * E                                                # every gadget offset stays below this
    a, b, gad = np.full(K, K), np.full(K, K), []
    ox, oy = 0, K
    for m, (g, mode) in enumerate(links):
        w, h, dm, di, si = (int(v) for v in G[g, [0, 1, 2, 6, 7]])
        if m:
            ox += (mode == 1)
            oy -= (mode == 2)
        oy -= h
        if mode == 1:                                                  # buffer column between the gadgets
            a[ox - 1] = b[ox - 1] = oy + h
        a[ox:ox + w], b[ox:ox + w] = oy, oy + h
        sq, fg = crop(donors[di], si, w, h, dm)
        assert fg <= f + 1e-12, (fg, f)
        sq = sq.copy(); sq[:, :2] *= 1 + 1e-11
        sq[:, 0] += ox + E * (1 + m); sq[:, 1] += oy + E * (M - m)     # later gadgets sit right of and below earlier ones by >= E at every seam
        gad.append(sq)
        ox += w
    a[ox:], b[ox:] = 0, 0
    assert oy >= 0 and ox <= K, (ox, oy)
    assert all(a[i + 1] <= b[i] for i in range(K - 1)) and all(a <= b)
    cells = [(i + .5, j + .5, 0.0) for i in range(K) for j in range(a[i])]
    cells += [(i + .5 + F, j + .5 + F, 0.0) for i in range(K) for j in range(b[i], K)]
    sq = np.concatenate([np.array(cells).reshape(-1, 3)] + gad)
    assert len(sq) == K * K + D, (len(sq), K * K + D)
    return sq


def certify_ladder(n, sq):
    r = certify(n, sq)
    for eps in (1e-12, 1e-11, 1e-10):
        if r:
            return r
        c = (sq[:, :2].min(0) + sq[:, :2].max(0)) / 2
        s2 = sq.copy(); s2[:, :2] = c + (sq[:, :2] - c) * (1 + eps)
        r = certify(n, s2)
    return r


def main():
    base, outdir, extra = sys.argv[1], sys.argv[2], sys.argv[3:]
    nmin, nmax = 325, 1000
    os.makedirs(outdir, exist_ok=True)
    seen, donors = set(), []
    for p in [base] + extra:                                           # donors: every packing below its grid side
        for n, sq in load_csv(p).items():
            k = (n, round(side(sq), 9))
            if k not in seen:
                seen.add(k); donors.append(sq)
    log(f"{len(donors)} donors")
    G = extract(donors)
    np.save(f"{outdir}/gadgets.npy", G)
    dp, PG, PM, PQ = run_dp(G)
    log("dp done")
    cur = load_csv(base)
    total = 0.0
    for n in range(nmin, nmax + 1):
        K = math.isqrt(n); D = n - K * K
        if D == 0:
            continue
        L0 = side(cur[n]) if n in cur else math.ceil(math.sqrt(n))
        f, links = chain(dp, PG, PM, PQ, G, K, D)
        if f is None or K + f + 2 * E >= L0 - 1e-6:
            continue
        sq = compose(K, D, f, links, G, donors)
        r = certify_ladder(n, sq)
        if r is None:
            log(f"n={n}: DP {K + f:.9f} < ours {L0:.9f} but NOT certified, chain {links}")
            continue
        d = (L0 - r[0]) / math.sqrt(n)
        total += d
        with open(f"{outdir}/{n}.csv", "w") as fh:
            fh.write("n,square_id,x,y,angle\n")
            for i, (x, y, t) in enumerate(r[1]):
                fh.write(f"{n},{i},{float(x)!r},{float(y)!r},{float(t)!r}\n")
        log(f"n={n}: CERTIFIED {r[0]:.9f} vs ours {L0:.9f}, gain {d:.2e}, {len(links)} gadgets "
            f"modes {[m for _, m in links]} | total {total:.6f}")
    log(f"DONE total certified gain {total:.6f}")


if __name__ == "__main__":
    main()

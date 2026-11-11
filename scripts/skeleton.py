"""Verified geometry helpers for LLM-written gadget programs (docs/fable-llm-search-design-2026-10-22.md section 2).
Every helper returns an (N, 3) float array of (x, y, theta): unit-square centres and angles in radians.
Programs combine these and choose sizes and positions; the helpers do the geometry.
"""
import math, os, subprocess, tempfile
import numpy as np

R2 = math.sqrt(2.0)


def upright(cells):
    """Upright unit squares on integer cells: cells = iterable of (i, j) -> squares [i, i+1] x [j, j+1]."""
    c = np.asarray(list(cells), float).reshape(-1, 2)
    return np.c_[c + 0.5, np.zeros(len(c))]


def staircase(t, corner, W, H):
    """t(t+1)/2 upright squares stacked as a staircase in one corner ('BL', 'BR', 'TL', 'TR') of the box [0, W] x [0, H].
    Its outer corners lie on a 45-degree line at distance (t + 1) / sqrt(2) from the box corner."""
    sq = []
    for c in range(t):
        for r in range(t - c):
            u, v = c + 0.5, r + 0.5
            x = u if corner in ("TL", "BL") else W - u
            y = H - v if corner in ("TL", "TR") else v
            sq.append((x, y, 0.0))
    return np.asarray(sq, float).reshape(-1, 3)


def tilted_block(a, b, theta, cx, cy, gap=1e-9):
    """a x b unit squares turned by theta, edge to edge (plus `gap`), as one rigid block centred at (cx, cy).
    Its bounding box has width a cos + b sin and height a sin + b cos (theta in [0, pi/2])."""
    c, s = math.cos(theta), math.sin(theta)
    sp = 1.0 + gap
    i, j = np.meshgrid(np.arange(a) - (a - 1) / 2, np.arange(b) - (b - 1) / 2, indexing="ij")
    x = cx + (i * c - j * s).ravel() * sp
    y = cy + (i * s + j * c).ravel() * sp
    return np.c_[x, y, np.full(x.size, theta)]


def tilted_row(b, theta, cx, cy, gap=1e-9):
    """A row of b squares turned by theta (Ryu's row), centred at (cx, cy): tilted_block(b, 1, ...)."""
    return tilted_block(b, 1, theta, cx, cy, gap)


def goebel_side(TL, TR, BR, BL, a, b):
    """Side of the goebel() packing, or None if the shape is invalid. For each pair of adjacent corners (p, q):
    p + q <= side (the two staircases do not overlap on their shared edge) and p + q + 2 >= side (the block corner
    stays inside the box). Accepts all 558 enumerator shapes without a shared edge cell (scratch/enum45.txt);
    1655 accepted shapes with all parameters < 7 were checked: no overlap, all inside the box."""
    s = max((TL + BR + 2) / 2 + a / R2, (TR + BL + 2) / 2 + b / R2)
    pairs = ((TL, TR), (TR, BR), (BR, BL), (BL, TL))
    return s if all(p + q <= s + 1e-9 and p + q + 2 >= s - 1e-9 for p, q in pairs) else None


def goebel(TL, TR, BR, BL, a, b):
    """Goebel / Ellsworth 45-degree packing: four corner staircases (rows TL, TR, BR, BL) and an a x b block turned
    45 degrees on the diagonal. Returns (squares, side). Raises ValueError for a shape goebel_side() rejects."""
    s = goebel_side(TL, TR, BR, BL, a, b)
    if s is None:
        raise ValueError(f"invalid goebel shape {(TL, TR, BR, BL, a, b)}: see goebel_side()")
    parts = [staircase(TL, "TL", s, s), staircase(TR, "TR", s, s), staircase(BR, "BR", s, s), staircase(BL, "BL", s, s)]
    d1_0, d2_0 = (TL + 1) / R2, (BL + 1) / R2
    blk = []
    for i in range(a):
        for j in range(b):
            d1, d2 = d1_0 + i + 0.5, d2_0 + j + 0.5
            blk.append(((d1 * R2 - s + d2 * R2) / 2, (d2 * R2 - d1 * R2 + s) / 2, math.pi / 4))
    parts.append(np.asarray(blk, float).reshape(-1, 3))
    return np.concatenate(parts), s


def free_cells(tilted, W, H, margin=1e-9):
    """Integer cells (i, j) of [0, W) x [0, H) whose upright square does not overlap any square in `tilted`."""
    t = np.asarray(tilted, float).reshape(-1, 3)
    I, J = np.meshgrid(np.arange(int(W)), np.arange(int(H)), indexing="ij")
    ux, uy = (I + 0.5).ravel(), (J + 0.5).ravel()
    if not len(t):
        return list(zip(I.ravel(), J.ravel()))
    ok = np.ones(len(ux), bool)
    for x, y, th in t:
        c, s = abs(math.cos(th)), abs(math.sin(th))
        he = 0.5 * (c + s)
        dx, dy = ux - x, uy - y
        sep = (np.abs(dx) >= 0.5 + he + margin) | (np.abs(dy) >= 0.5 + he + margin)
        pu, pv = np.abs(dx * math.cos(th) + dy * math.sin(th)), np.abs(-dx * math.sin(th) + dy * math.cos(th))
        sep |= (pu >= 0.5 + he + margin) | (pv >= 0.5 + he + margin)
        ok &= sep
    return [(int(i), int(j)) for i, j in zip(I.ravel()[ok], J.ravel()[ok])]


def dilate(sq, eps=1e-9):
    """Scale all centres by (1 + eps) about their mean: turns exact contacts into small gaps."""
    sq = np.asarray(sq, float).copy()
    c = sq[:, :2].mean(0)
    sq[:, :2] = c + (sq[:, :2] - c) * (1 + eps)
    return sq


def _bounds(sq):
    sq = np.asarray(sq, float).reshape(-1, 3)
    he = 0.5 * (np.abs(np.cos(sq[:, 2])) + np.abs(np.sin(sq[:, 2])))
    return sq, sq[:, 0] - he, sq[:, 0] + he, sq[:, 1] - he, sq[:, 1] + he


def overhang(sq, w, h):
    """f of a layout: translate its minimum corner to (0, 0); f = max(xmax - w, ymax - h, 0). Smaller is better."""
    _, xlo, xhi, ylo, yhi = _bounds(sq)
    return float(max((xhi - xlo.min()).max() - w, (yhi - ylo.min()).max() - h, 0.0))


def crop(sq, w, h, n):
    """Keep the n squares that stick out least past [0, w] x [0, h] (after moving the minimum corner to (0, 0)).
    Removing squares never creates an overlap, so any valid layout with >= n squares can be cropped to a key."""
    sq, xlo, xhi, ylo, yhi = _bounds(sq)
    r = np.maximum(np.maximum(xhi - xlo.min() - w, yhi - ylo.min() - h), 0.0)
    return sq[np.argsort(r, kind="stable")[:n]]


FQ = os.environ.get("FQ_BIN") or os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "external", "evand", "search", "packer",
                                  "target", "release", "fq"))


def _gadget_f(sq, w, h):
    _, xlo, xhi, ylo, yhi = _bounds(sq)
    return float(max((xhi - xlo.min()).max() - w, (yhi - ylo.min()).max() - h, 0.0))


def quench(sq, w, h, loosen=1.02, kick=0.0, seed=1, round_r0=0.0, outer=60):
    """evand's smooth quench (external/evand/search/packer, MIT) in rectangle mode: minimise f with the squares inside
    [0, w + f] x [0, h + f]. Loosens the box by `loosen`, kicks centres by N(0, kick) and angles by N(0, 20 kick)
    degrees (seed), optionally starts from rounded squares (soft-to-rigid, corner radius round_r0), then squeezes.
    Returns (squares, f); the result is feasible in float64 (fq repairs by scaling). On failure returns (sq, inf).
    Typical time 0.1 to 8 s for 60 to 350 squares."""
    sq = np.asarray(sq, float).reshape(-1, 3).copy()
    _, xlo, _, ylo, _ = _bounds(sq)
    sq[:, 0] -= xlo.min(); sq[:, 1] -= ylo.min()
    m = min(w, h)
    s = m + _gadget_f(sq, w, h)
    with tempfile.TemporaryDirectory() as d:
        a, b = os.path.join(d, "a.txt"), os.path.join(d, "b.txt")
        with open(a, "w") as fh:
            fh.write(f"{len(sq)} {s + 1e-9:.17f}\n")
            for x, y, t in sq:
                fh.write(f"{x:.17f} {y:.17f} {math.degrees(t):.15f}\n")
        cmd = [FQ, "quench", "--in", a, "--out", b, "--rect-dx", repr(float(w - m)), "--rect-dy", repr(float(h - m)),
               "--loosen", repr(float(loosen)), "--kick", repr(float(kick)), "--seed", str(int(seed)), "--outer", str(int(outer))]
        if round_r0 > 0:
            cmd += ["--round-r0", repr(float(round_r0))]
        try:
            subprocess.run(cmd, capture_output=True, timeout=60)
        except subprocess.TimeoutExpired:
            return sq, math.inf
        if not os.path.exists(b):
            return sq, math.inf
        rows = open(b).read().split("\n")[1:]
    out = np.array([[float(v) for v in r.split()] for r in rows if r.strip()])
    if out.shape != sq.shape or not np.isfinite(out).all():
        return sq, math.inf
    out[:, 2] = np.radians(out[:, 2])
    return out, _gadget_f(out, w, h)


def kick(sq, sigma, seed, idx=None, angle_deg=0.0):
    """Gaussian kick of the centres (sigma) and angles (angle_deg) of squares idx (all if None); deterministic in seed."""
    sq = np.asarray(sq, float).copy()
    r = np.random.default_rng(seed)
    idx = np.arange(len(sq)) if idx is None else np.asarray(idx)
    sq[idx, :2] += r.normal(0, sigma, (len(idx), 2))
    sq[idx, 2] += np.radians(r.normal(0, angle_deg, len(idx)))
    return sq


def edge_squares(sq, w, h, side="right", k=5):
    """Indices of the k squares sticking out furthest past the right (x > w) or top (y > h) edge."""
    _, xlo, xhi, ylo, yhi = _bounds(sq)
    v = (xhi - xlo.min()) if side == "right" else (yhi - ylo.min())
    return np.argsort(-v)[:k]

from __future__ import annotations

import argparse
import contextlib
import csv
import math
import multiprocessing
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from fractions import Fraction

import mpmath

L_DPS = 60
L_DECIMALS = 50
UNDECIDED_TOL = 1e-8
_ESCALATION = [50, 100, 200, 500]
_MAX_LITERAL_CHARS = 1000
_DECIMAL_RE = re.compile(r'^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$')
MAX_N = 1000


# ----------------------------------------------------------------------------- precision

@contextlib.contextmanager
def working_dps(dps: int):
    old_iv, old_mp = mpmath.iv.dps, mpmath.mp.dps
    try:
        mpmath.iv.dps = dps
        mpmath.mp.dps = dps
        yield
    finally:
        mpmath.iv.dps = old_iv
        mpmath.mp.dps = old_mp


def iv_le(a, b):
    """True if a <= b certified, False if a > b certified, None if undecided."""
    try:
        return a <= b
    except ValueError:
        return None


def iv_min(values):
    return mpmath.iv.mpf([min(v.a for v in values), min(v.b for v in values)])


def iv_max(values):
    return mpmath.iv.mpf([max(v.a for v in values), max(v.b for v in values)])


# ----------------------------------------------------------------------------- parsing

def is_decimal(s: str) -> bool:
    return len(s) <= _MAX_LITERAL_CHARS and bool(_DECIMAL_RE.match(s.strip()))


def check_decimal(s: str) -> str:
    s = s.strip()
    if not is_decimal(s):
        raise ValueError(f'not a decimal number: {s[:40]!r}')
    return s


def parse_float(s: str) -> float:
    s = check_decimal(s)
    v = float(s)
    if not math.isfinite(v):
        raise ValueError(f'value out of range for a double: {s!r}')
    return v


def parse_mp(s: str, dps: int = L_DPS):
    s = check_decimal(s)
    old = mpmath.mp.dps
    try:
        mpmath.mp.dps = dps
        return mpmath.mpf(s)
    finally:
        mpmath.mp.dps = old


def parse_iv(s: str, dps: int = 50):
    with working_dps(dps):
        return mpmath.iv.mpf(check_decimal(s))


def roundtrip_str(s: str) -> str:
    """What a float64 read + default to_csv does to a coordinate string."""
    return repr(parse_float(s))


# ----------------------------------------------------------------------------- geometry

@dataclass
class Square:
    x: float
    y: float
    angle: float
    x_str: str = ''
    y_str: str = ''
    angle_str: str = ''
    _corners: list = field(default_factory=list, repr=False)

    def corners(self):
        if not self._corners:
            self._corners = corners_float(self.x, self.y, self.angle)
        return self._corners


def corners_float(cx, cy, angle):
    c, s = math.cos(angle), math.sin(angle)
    h = 0.5
    dx1, dy1 = h * c - h * s, h * s + h * c
    dx2, dy2 = -h * c - h * s, -h * s + h * c
    return [(cx + dx1, cy + dy1), (cx + dx2, cy + dy2), (cx - dx1, cy - dy1), (cx - dx2, cy - dy2)]


def corners_mp(cx, cy, angle):
    c, s = mpmath.cos(angle), mpmath.sin(angle)
    h = mpmath.mpf(1) / 2
    dx1, dy1 = h * c - h * s, h * s + h * c
    dx2, dy2 = -h * c - h * s, -h * s + h * c
    return [(cx + dx1, cy + dy1), (cx + dx2, cy + dy2), (cx - dx1, cy - dy1), (cx - dx2, cy - dy2)]


def corners_iv(cx, cy, angle):
    c, s = mpmath.iv.cos(angle), mpmath.iv.sin(angle)
    h = mpmath.iv.mpf('0.5')
    dx1, dy1 = h * c - h * s, h * s + h * c
    dx2, dy2 = -h * c - h * s, -h * s + h * c
    return [(cx + dx1, cy + dy1), (cx + dx2, cy + dy2), (cx - dx1, cy - dy1), (cx - dx2, cy - dy2)]


def measure_side_mp(squares, dps=L_DPS):
    old = mpmath.mp.dps
    try:
        mpmath.mp.dps = dps
        xs, ys = [], []
        for sq in squares:
            for x, y in corners_mp(parse_mp(sq.x_str, dps), parse_mp(sq.y_str, dps), parse_mp(sq.angle_str, dps)):
                xs.append(x)
                ys.append(y)
        return max(max(xs) - min(xs), max(ys) - min(ys))
    finally:
        mpmath.mp.dps = old


def aabb(corners):
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return (min(xs), min(ys), max(xs), max(ys))


def aabb_overlap(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _project(corners, ax, ay):
    p = [ax * x + ay * y for x, y in corners]
    return min(p), max(p)


def sat_float(sq1: Square, sq2: Square, tol=UNDECIDED_TOL):
    """Returns ('separated'|'overlapping'|'undecided', min_overlap).

    min_overlap is the smallest projected overlap over the four axes (negative = gap on that
    axis). It is the metric's decision variable; we also return it for the margin diagnostic.
    """
    c1, c2 = sq1.corners(), sq2.corners()
    a1, b1 = math.cos(sq1.angle), math.sin(sq1.angle)
    a2, b2 = math.cos(sq2.angle), math.sin(sq2.angle)
    axes = [(a1, b1), (-b1, a1), (a2, b2), (-b2, a2)]
    min_overlap = float('inf')
    for ax, ay in axes:
        lo1, hi1 = _project(c1, ax, ay)
        lo2, hi2 = _project(c2, ax, ay)
        overlap = min(hi1, hi2) - max(lo1, lo2)
        if overlap < -tol:
            return 'separated', overlap
        min_overlap = min(min_overlap, overlap)
    if min_overlap > tol:
        return 'overlapping', min_overlap
    return 'undecided', min_overlap


def best_axis_gap(sq1: Square, sq2: Square) -> float:
    """float64 diagnostic: max over the four axes of (-overlap). Positive = a real gap exists."""
    c1, c2 = sq1.corners(), sq2.corners()
    a1, b1 = math.cos(sq1.angle), math.sin(sq1.angle)
    a2, b2 = math.cos(sq2.angle), math.sin(sq2.angle)
    best = -float('inf')
    for ax, ay in [(a1, b1), (-b1, a1), (a2, b2), (-b2, a2)]:
        lo1, hi1 = _project(c1, ax, ay)
        lo2, hi2 = _project(c2, ax, ay)
        best = max(best, -(min(hi1, hi2) - max(lo1, lo2)))
    return best


def sat_interval(c1, c2, ang1, ang2):
    """True = certified separated, False = certified overlapping, None = undecided."""
    cs1, sn1 = mpmath.iv.cos(ang1), mpmath.iv.sin(ang1)
    cs2, sn2 = mpmath.iv.cos(ang2), mpmath.iv.sin(ang2)
    undecided = False
    for ax, ay in [(cs1, sn1), (-sn1, cs1), (cs2, sn2), (-sn2, cs2)]:
        p1 = [ax * x + ay * y for x, y in c1]
        p2 = [ax * x + ay * y for x, y in c2]
        lo1, hi1 = iv_min(p1), iv_max(p1)
        lo2, hi2 = iv_min(p2), iv_max(p2)
        overlap = iv_min([hi1, hi2]) - iv_max([lo1, lo2])
        r = iv_le(overlap, mpmath.iv.mpf('0'))
        if r is True:
            return True
        if r is None:
            undecided = True
    return None if undecided else False


def sort_and_sweep(squares):
    n = len(squares)
    if n <= 1:
        return []
    boxes = sorted(((i, aabb(sq.corners())) for i, sq in enumerate(squares)), key=lambda t: t[1][0])
    pairs = []
    for a in range(n):
        i, bi = boxes[a]
        for b in range(a + 1, n):
            j, bj = boxes[b]
            if bj[0] > bi[2]:
                break
            if aabb_overlap(bi, bj):
                pairs.append((i, j) if i < j else (j, i))
    return pairs


def verify_puzzle(squares):
    """(feasible, reason, n_undecided_pairs, min_margin_float)."""
    pairs = sort_and_sweep(squares)
    undecided = []
    min_margin = float('inf')
    for i, j in pairs:
        r, ov = sat_float(squares[i], squares[j])
        min_margin = min(min_margin, best_axis_gap(squares[i], squares[j]))
        if r == 'overlapping':
            return False, f'Squares {i} and {j} overlap (float check)', 0, min_margin
        if r == 'undecided':
            undecided.append((i, j))
    if undecided:
        for dps in _ESCALATION:
            with working_dps(dps):
                still = []
                verdict = None
                for i, j in undecided:
                    si, sj = squares[i], squares[j]
                    ai, aj = parse_iv(si.angle_str, dps), parse_iv(sj.angle_str, dps)
                    ci = corners_iv(parse_iv(si.x_str, dps), parse_iv(si.y_str, dps), ai)
                    cj = corners_iv(parse_iv(sj.x_str, dps), parse_iv(sj.y_str, dps), aj)
                    r = sat_interval(ci, cj, ai, aj)
                    if r is False:
                        verdict = False
                        break
                    if r is None:
                        still.append((i, j))
                if verdict is False:
                    return False, 'Squares overlap (certified)', len(undecided), min_margin
                if not still:
                    return True, 'feasible', len(undecided), min_margin
                undecided = still
        return False, 'Overlap undecidable at max precision', len(undecided), min_margin
    return True, 'feasible', 0, min_margin


# ----------------------------------------------------------------------------- scoring

def score_puzzle(n: int, L: float) -> float:
    return L / math.sqrt(n) - 1.0


def infeasible_score(n: int) -> float:
    return math.ceil(math.sqrt(n)) / math.sqrt(n) - 1.0


def format_L(value) -> str:
    exact = Fraction(value.man) * Fraction(2) ** value.exp
    scale = 10 ** L_DECIMALS
    units = round(exact * scale)
    whole, frac = divmod(abs(units), scale)
    return f'{"-" if units < 0 else ""}{whole}.{frac:0{L_DECIMALS}d}'


def build_squares(n, rows, roundtrip=False):
    by_id = {}
    for r in rows:
        try:
            sid = int(r['square_id'])
        except (TypeError, ValueError):
            raise ValueError(f'n={n}: invalid square_id {r["square_id"]!r}')
        by_id[sid] = r
    if len(rows) != n or set(by_id) != set(range(n)):
        raise ValueError(f'n={n}: expected ids 0..{n - 1}, got {len(rows)} rows')
    squares = []
    for sid in range(n):
        r = by_id[sid]
        xs, ys, as_ = r['x'].strip(), r['y'].strip(), r['angle'].strip()
        if roundtrip:
            xs, ys, as_ = roundtrip_str(xs), roundtrip_str(ys), roundtrip_str(as_)
        squares.append(Square(parse_float(xs), parse_float(ys), parse_float(as_), xs, ys, as_))
    return squares


def evaluate_puzzle(n, rows, roundtrip=False):
    """(n, score, L_text or None, reason, n_undecided, min_margin)."""
    try:
        squares = build_squares(n, rows, roundtrip)
        L = measure_side_mp(squares)
    except ValueError as e:
        return (n, infeasible_score(n), None, str(e), 0, float('nan'))
    ok, reason, nund, margin = verify_puzzle(squares)
    if not ok:
        return (n, infeasible_score(n), None, f'infeasible: {reason}', nund, margin)
    return (n, score_puzzle(n, float(L)), format_L(L), 'ok', nund, margin)


def _worker(args):
    n, rows, roundtrip = args
    try:
        return evaluate_puzzle(n, rows, roundtrip)
    except Exception as e:  # noqa: BLE001
        return (n, infeasible_score(n), None, f'worker error: {e}', 0, float('nan'))


def read_submission(path):
    puzzles = defaultdict(list)
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        need = {'n', 'square_id', 'x', 'y', 'angle'}
        if not need.issubset(reader.fieldnames or []):
            raise ValueError(f'missing columns: {need - set(reader.fieldnames or [])}')
        for row in reader:
            n = int(row['n'])
            if not 1 <= n <= MAX_N:
                raise ValueError(f'n={n} out of range')
            puzzles[n].append(row)
    return dict(puzzles)


def score_submission(path, ns=None, roundtrip=False, workers=4, progress=False):
    ns = set(ns) if ns is not None else set(range(1, MAX_N + 1))
    puzzles = read_submission(path)
    work = [(n, rows, roundtrip) for n, rows in sorted(puzzles.items()) if n in ns]
    if workers > 1 and len(work) >= 4:
        with multiprocessing.Pool(workers) as pool:
            results = []
            for k, r in enumerate(pool.imap_unordered(_worker, work, chunksize=4), 1):
                results.append(r)
                if progress and k % 100 == 0:
                    print(f'  {k}/{len(work)}', file=sys.stderr, flush=True)
    else:
        results = [_worker(w) for w in work]
    seen = {r[0] for r in results}
    for n in ns - seen:
        results.append((n, infeasible_score(n), None, 'missing from submission', 0, float('nan')))
    results.sort(key=lambda r: r[0])
    return sum(r[1] for r in results), results

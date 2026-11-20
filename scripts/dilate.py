"""Minimal extraction of the notebook's dilation helpers (no dataset mounting)."""
import math

import mpmath

DIGITS = 34
DELTA = 2e-8
L_DPS = 60


def corners_float(cx, cy, angle):
    c = math.cos(angle)
    s = math.sin(angle)
    h = 0.5
    dx1, dy1 = h * c - h * s, h * s + h * c
    dx2, dy2 = -h * c - h * s, -h * s + h * c
    return [
        (cx + dx1, cy + dy1),
        (cx + dx2, cy + dy2),
        (cx - dx1, cy - dy1),
        (cx - dx2, cy - dy2),
    ]


def dilate_rows(rows, delta=DELTA):
    """Dilate a packing about its bbox centre by (1+delta); exact-decimal output."""
    xs, ys = [], []
    for _, x, y, a in rows:
        for cx, cy in corners_float(float(x), float(y), float(a)):
            xs.append(cx)
            ys.append(cy)
    with mpmath.workdps(60):
        cx = (mpmath.mpf(min(xs)) + mpmath.mpf(max(xs))) / 2
        cy = (mpmath.mpf(min(ys)) + mpmath.mpf(max(ys))) / 2
        one_plus = mpmath.mpf(1) + mpmath.mpf(delta)
        out = []
        for i, x, y, a in rows:
            xd = cx + one_plus * (mpmath.mpf(x) - cx)
            yd = cy + one_plus * (mpmath.mpf(y) - cy)
            out.append((i, mpmath.nstr(xd, DIGITS, strip_zeros=True),
                        mpmath.nstr(yd, DIGITS, strip_zeros=True), a))
    return out


def side_float(rows):
    xs, ys = [], []
    for _, x, y, a in rows:
        for cx, cy in corners_float(float(x), float(y), float(a)):
            xs.append(cx)
            ys.append(cy)
    return max(max(xs) - min(xs), max(ys) - min(ys))

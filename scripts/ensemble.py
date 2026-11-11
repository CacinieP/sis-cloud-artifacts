"""Per-n best over several submission files, each candidate certified by the official metric on its float64 values.
The board parses the submitted CSV as float64 (correctly rounded) before the metric sees it, so a literal is scored as
its nearest double: measured 2026-10-31 on four submissions, the float64 prediction matched the board to 7
decimals (22.6797880, 25.1903470, 22.6308887) while certifying the text predicted 22.2611-22.2616.
A candidate that fails as float64 is dilated about its centre by 1 + 1e-12 (a numerically optimised packing can miss
contact by ~1e-12) and certified again. The winner is written as float64 repr, which is exactly what the board scores.
Falls back to the grid.

    .venv/bin/python -u src/ensemble.py OUT.csv name=path [name=path ...]
"""
import math, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metric


def side(sq):
    x, y, t = sq[:, 0], sq[:, 1], sq[:, 2]
    c, s = np.cos(t) * .5, np.sin(t) * .5
    X = np.concatenate([x + c - s, x - c - s, x - c + s, x + c + s]); Y = np.concatenate([y + s + c, y - s + c, y - s - c, y + s - c])
    return max(X.max() - X.min(), Y.max() - Y.min())


def float_text(sq):
    return [(repr(float(a)), repr(float(b)), repr(float(t))) for a, b, t in sq]


def board_values(text):
    """The doubles the board gets for these literals: correctly rounded float64 (as float(), or pandas.read_csv with
    float_precision='round_trip'). Measured 2026-10-31: this model predicts six board scores exactly (22.2615877,
    22.6797880, 25.1903470, 22.6308887, 25.1808952 and the 10-29 22.2619999); pandas' DEFAULT parser, which is not
    correctly rounded, predicts 22.4809 and 25.3220 for the files the board scored 22.2615877 and 25.1808952."""
    return np.array([[float(a), float(b), float(t)] for a, b, t in text])


def certify(n, sq, text=None):
    """(L, sq, text) if the metric accepts the packing as the board will read `text` (default: float64 repr of sq):
    pandas-parsed doubles, written back as repr, then certified. The returned text is what gets written."""
    text = text or float_text(sq)
    rows = [dict(n=str(n), square_id=str(i), x=a, y=b, angle=t) for i, (a, b, t) in enumerate(float_text(board_values(text)))]
    _, sc, L, why = metric.evaluate_single_puzzle(n, rows)
    return (float(L), sq, text) if why == "ok" else None


def main():
    out, srcs = sys.argv[1], [a.split("=", 1) for a in sys.argv[2:]]
    cands = {n: [] for n in range(1, 1001)}
    for name, path in srcs:
        df = pd.read_csv(path, dtype=str); raw = df.copy()
        for c in ("n", "square_id", "x", "y", "angle"):        # a value the metric cannot parse drops that puzzle
            df[c] = pd.to_numeric(df[c], errors="coerce")
        for n, g in df.groupby("n"):
            if n not in cands or len(g) != n or g[["x", "y", "angle"]].isna().any().any():   # n > 1000 seen in a public file 2026-10-29
                continue
            g = g.sort_values("square_id")
            sq = g[["x", "y", "angle"]].to_numpy(float)
            text = list(zip(*(raw.loc[g.index, c].str.strip() for c in ("x", "y", "angle"))))
            cands[int(n)].append((side(sq), name, sq, text))
    won, best, dil = {}, {}, 0
    for n in range(1, 1001):
        k = math.ceil(math.sqrt(n))
        for L, name, sq, text in sorted(cands[n], key=lambda c: c[0]):
            if L >= k - 1e-12:
                break
            r = certify(n, sq)
            if r is None:
                c = (sq[:, :2].min(0) + sq[:, :2].max(0)) / 2
                sq2 = sq.copy(); sq2[:, :2] = c + (sq[:, :2] - c) * (1 + 1e-12)
                r = certify(n, sq2)
                if r: dil += 1
            if r:
                best[n], won[n] = r, name
                break
        if n not in best:
            g = np.array([[i % k + .5, i // k + .5, 0.0] for i in range(n)])
            best[n] = (float(k), g, float_text(g)); won[n] = "grid"
    rid = 0
    with open(out, "w") as f:
        f.write("id,n,square_id,x,y,angle\n")
        for n in range(1, 1001):
            for i, (a, b, t) in enumerate(best[n][2]):
                f.write(f"{rid},{n},{i},{a},{b},{t}\n"); rid += 1
    tot = sum(best[n][0] / math.sqrt(n) - 1 for n in best)
    print(f"wrote {out}: certified total {tot:.6f}; needed 1e-12 dilation: {dil}")
    print("puzzles won by source:", pd.Series(won).value_counts().to_dict())
    pd.Series(won).to_csv(out.replace(".csv", "_sources.csv"))


if __name__ == "__main__":
    main()

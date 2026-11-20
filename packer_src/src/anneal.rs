//! anneal: hard rounded squares in a square box, NPT compression MC with the shape morphing from disk to square
//! (10-08, Evan: start as a hot disk packing, anneal to squares as it cools).
//!
//! Shape: inner square of half-side h = 0.5 - r, Minkowski-rounded by radius r (r = 0.5: unit-diameter disk; r = 0: unit
//! square).  Overlap: Euclidean distance between the inner squares < 2r (r = 0: SAT).  Walls: inner corners >= r from
//! each wall of [0, s]^2.
//! MC: single-particle translate / rotate, isotropic box rescale (NPT, ln-area moves: accept exp(-bP dA + (N+1) ln(A'/A))),
//! shape step toward the scheduled r (accepted if overlap-free; if not, the smallest box growth <= 5 % that makes it
//! feasible: an annealing heuristic, not detailed balance).  Adaptive steps toward ~40 % acceptance.
//! Schedule over progress t in [0, 1] (sweeps): bP = bp0 (bp1/bp0)^t; r = 0.5 (1 - clamp((t - t0)/(t1 - t0)))^p.
//!
//!   anneal run --n 110 --seed 1 --sweeps 40000 --bp0 1 --bp1 3000 --t0 0.2 --t1 0.8 --p 1 --out X.txt
//!   anneal test        (distance routine vs brute force)
//! Output: packer format ("n s", then "x y theta_deg"), centres in [0, s]^2; a JSON summary on stdout.

use std::f64::consts::{FRAC_PI_2, PI, SQRT_2};

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        let mut r = Rng(seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(0x632BE59BD9B4E019) | 1);
        for _ in 0..8 { r.next(); }
        r
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn f(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 }
    fn u(&mut self) -> f64 { 2.0 * self.f() - 1.0 }
    fn idx(&mut self, n: usize) -> usize { (self.next() % n as u64) as usize }
}

fn verts(x: f64, y: f64, t: f64, h: f64) -> [(f64, f64); 4] {
    let (c, s) = (t.cos(), t.sin());
    let mut v = [(0.0, 0.0); 4];
    for (k, &(a, b)) in [(1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)].iter().enumerate() {
        v[k] = (x + h * (a * c - b * s), y + h * (a * s + b * c));
    }
    v
}

/// SAT gap of two squares of half-side h (max over the 4 face normals of the projected separation; < 0 = overlap).
fn sat_gap(xi: f64, yi: f64, ti: f64, xj: f64, yj: f64, tj: f64, h: f64) -> f64 {
    let mut best = f64::NEG_INFINITY;
    let (dx, dy) = (xj - xi, yj - yi);
    for &t in &[ti, ti + FRAC_PI_2, tj, tj + FRAC_PI_2] {
        let (ax, ay) = (t.cos(), t.sin());
        let dc = (dx * ax + dy * ay).abs();
        let ri = h * ((ti - t).cos().abs() + (ti - t).sin().abs());
        let rj = h * ((tj - t).cos().abs() + (tj - t).sin().abs());
        best = best.max(dc - ri - rj);
    }
    best
}

fn pt_seg(px: f64, py: f64, a: (f64, f64), b: (f64, f64)) -> f64 {
    let (ex, ey) = (b.0 - a.0, b.1 - a.1);
    let l2 = ex * ex + ey * ey;
    let t = if l2 > 0.0 { (((px - a.0) * ex + (py - a.1) * ey) / l2).clamp(0.0, 1.0) } else { 0.0 };
    (px - a.0 - t * ex).hypot(py - a.1 - t * ey)
}

/// Euclidean distance between two closed squares of half-side h (0 if they intersect).
fn sq_dist(xi: f64, yi: f64, ti: f64, xj: f64, yj: f64, tj: f64, h: f64) -> f64 {
    if sat_gap(xi, yi, ti, xj, yj, tj, h) <= 0.0 { return 0.0; }
    let (a, b) = (verts(xi, yi, ti, h), verts(xj, yj, tj, h));
    let mut d = f64::INFINITY;
    for k in 0..4 {
        for e in 0..4 {
            d = d.min(pt_seg(a[k].0, a[k].1, b[e], b[(e + 1) % 4]));
            d = d.min(pt_seg(b[k].0, b[k].1, a[e], a[(e + 1) % 4]));
        }
    }
    d
}

struct Sys { n: usize, x: Vec<f64>, s: f64, r: f64 }

impl Sys {
    /// does particle i at (x, y, t) overlap particle j (shape parameter r)?
    fn pair_ov(&self, x: f64, y: f64, t: f64, j: usize, r: f64) -> bool {
        let (xj, yj, tj) = (self.x[3 * j], self.x[3 * j + 1], self.x[3 * j + 2]);
        let h = 0.5 - r;
        let d2 = (x - xj).powi(2) + (y - yj).powi(2);
        if d2 < 1.0 - 1e-12 { return true; }                     // inscribed disks (radius 0.5) overlap
        let rc = h * SQRT_2 + r;
        if d2 >= 4.0 * rc * rc { return false; }                  // circumscribed disks apart
        if r <= 0.0 { return sat_gap(x, y, t, xj, yj, tj, h) < 0.0; }
        let g = sat_gap(x, y, t, xj, yj, tj, h);
        if g >= 2.0 * r { return false; }
        if g <= 0.0 { return true; }
        sq_dist(x, y, t, xj, yj, tj, h) < 2.0 * r
    }
    fn wall_ok(&self, x: f64, y: f64, t: f64, s: f64, r: f64) -> bool {
        let w = (0.5 - r) * (t.cos().abs() + t.sin().abs()) + r;
        x >= w && x <= s - w && y >= w && y <= s - w
    }
    fn one_ok(&self, i: usize, x: f64, y: f64, t: f64) -> bool {
        if !self.wall_ok(x, y, t, self.s, self.r) { return false; }
        (0..self.n).all(|j| j == i || !self.pair_ov(x, y, t, j, self.r))
    }
    /// full check of a configuration xs, side s, shape r
    fn all_ok(&self, xs: &[f64], s: f64, r: f64) -> bool {
        let tmp = Sys { n: self.n, x: xs.to_vec(), s, r };
        for i in 0..self.n {
            let (x, y, t) = (xs[3 * i], xs[3 * i + 1], xs[3 * i + 2]);
            if !tmp.wall_ok(x, y, t, s, r) { return false; }
            for j in i + 1..self.n { if tmp.pair_ov(x, y, t, j, r) { return false; } }
        }
        true
    }
}

fn area(r: f64) -> f64 { let h = 0.5 - r; 4.0 * h * h + 8.0 * h * r + PI * r * r }

fn arg<T: std::str::FromStr>(a: &[String], k: &str, def: T) -> T {
    a.iter().rposition(|x| x == k).and_then(|i| a.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(def)
}

fn test() {
    let mut rng = Rng::new(3);
    let mut worst: f64 = 0.0;
    for trial in 0..3000 {
        let h = [0.5, 0.3, 0.1, 0.0][trial % 4];
        let (xi, yi, ti) = (0.0, 0.0, rng.f() * PI);
        let (xj, yj, tj) = (rng.u() * 2.0, rng.u() * 2.0, rng.f() * PI);
        let d = sq_dist(xi, yi, ti, xj, yj, tj, h);
        // brute force: dense boundary sampling of both squares (or the point) + inside test
        let samp = |x: f64, y: f64, t: f64| -> Vec<(f64, f64)> {
            let v = verts(x, y, t, h);
            let mut p = Vec::new();
            for e in 0..4 { for q in 0..400 { let s = q as f64 / 400.0; p.push((v[e].0 + s * (v[(e + 1) % 4].0 - v[e].0), v[e].1 + s * (v[(e + 1) % 4].1 - v[e].1))); } }
            p
        };
        let (pa, pb) = (samp(xi, yi, ti), samp(xj, yj, tj));
        let mut bd = f64::INFINITY;
        for a in &pa { for b in &pb { bd = bd.min((a.0 - b.0).hypot(a.1 - b.1)); } }
        let inside = sat_gap(xi, yi, ti, xj, yj, tj, h) <= 0.0;
        let err = if inside { d } else { (d - bd).abs() };
        // sampling resolution ~ 2h/400 * sqrt2
        if !inside && bd > 1e-9 && err > 2.0 * h / 400.0 + 1e-12 { println!("MISMATCH h {h} d {d} brute {bd}"); }
        if inside && d != 0.0 { println!("MISMATCH overlap h {h} d {d}"); }
        worst = worst.max(if inside { 0.0 } else { err });
    }
    println!("test done: worst |exact - brute| {worst:.2e} (brute resolution ~ {:.1e})", 2.0 * 0.5 / 400.0);
}

fn run(a: &[String]) {
    let n: usize = arg(a, "--n", 110);
    let seed: u64 = arg(a, "--seed", 1);
    let sweeps: usize = arg(a, "--sweeps", 40000);
    let (bp0, bp1): (f64, f64) = (arg(a, "--bp0", 1.0), arg(a, "--bp1", 3000.0));
    let (t0, t1, pw): (f64, f64, f64) = (arg(a, "--t0", 0.2), arg(a, "--t1", 0.8), arg(a, "--p", 1.0));
    let r_init: f64 = arg(a, "--r0", 0.5);
    let phi0: f64 = arg(a, "--phi0", 0.25);
    let out: String = arg(a, "--out", String::from("anneal.txt"));
    let mut rng = Rng::new(seed);
    let rsched = |t: f64| -> f64 {
        if t1 <= t0 { return if t >= t0 { 0.0 } else { r_init }; }
        r_init * (1.0 - ((t - t0) / (t1 - t0)).clamp(0.0, 1.0)).powf(pw)
    };
    let r_start = rsched(0.0);
    let s0 = (n as f64 * area(r_start) / phi0).sqrt();
    let mut sys = Sys { n, x: Vec::with_capacity(3 * n), s: s0, r: r_start };
    // random sequential placement
    let mut placed = 0;
    let mut tries = 0usize;
    while placed < n {
        tries += 1;
        assert!(tries < 10_000_000, "placement failed");
        let (x, y, t) = (rng.f() * s0, rng.f() * s0, rng.f() * FRAC_PI_2);
        sys.n = placed;
        if sys.one_ok(usize::MAX, x, y, t) { sys.x.extend_from_slice(&[x, y, t]); placed += 1; }
    }
    sys.n = n;
    let (mut dt, mut dr, mut dv) = (0.3f64, 0.3f64, 0.01f64);
    let (mut at, mut nt, mut av, mut nv) = (0usize, 0usize, 0usize, 0usize);
    let mut shape_fail = 0usize;
    let t_start = std::time::Instant::now();
    let report = sweeps / 10;
    for sw in 0..sweeps {
        let t = sw as f64 / sweeps as f64;
        let bp = bp0 * (bp1 / bp0).powf(t);
        // shape step toward the schedule
        let rt = rsched(t);
        if rt < sys.r - 1e-15 {
            let xs = sys.x.clone();
            if sys.all_ok(&xs, sys.s, rt) { sys.r = rt; }
            else {
                // smallest box growth (<= 5 %) that makes the new shape feasible (bisection on the scale factor)
                let feas = |f: f64, sys: &Sys| { let xf: Vec<f64> = xs.iter().enumerate().map(|(k, &v)| if k % 3 == 2 { v } else { v * f }).collect(); (sys.all_ok(&xf, sys.s * f, rt), xf) };
                let (ok, _) = feas(1.05, &sys);
                if ok {
                    let (mut lo, mut hi) = (1.0, 1.05);
                    for _ in 0..20 { let m = 0.5 * (lo + hi); if feas(m, &sys).0 { hi = m; } else { lo = m; } }
                    let (_, xf) = feas(hi, &sys);
                    sys.x = xf; sys.s *= hi; sys.r = rt;
                } else { shape_fail += 1; }
            }
        }
        for _ in 0..n {
            let i = rng.idx(n);
            let (x, y, th) = (sys.x[3 * i], sys.x[3 * i + 1], sys.x[3 * i + 2]);
            let (nx, ny, nth) = if rng.f() < 0.5 || sys.r >= 0.5 { (x + dt * rng.u(), y + dt * rng.u(), th) } else { (x, y, th + dr * rng.u()) };
            nt += 1;
            if sys.one_ok(i, nx, ny, nth) { sys.x[3 * i] = nx; sys.x[3 * i + 1] = ny; sys.x[3 * i + 2] = nth; at += 1; }
        }
        // box move (ln-area)
        {
            let dl = dv * rng.u();
            let f = (0.5 * dl).exp();
            let s1 = sys.s * f;
            let da = s1 * s1 - sys.s * sys.s;
            let lacc = -bp * da + (n as f64 + 1.0) * dl;
            nv += 1;
            if lacc >= 0.0 || rng.f() < lacc.exp() {
                let xf: Vec<f64> = sys.x.iter().enumerate().map(|(k, &v)| if k % 3 == 2 { v } else { v * f }).collect();
                if f >= 1.0 || sys.all_ok(&xf, s1, sys.r) { sys.x = xf; sys.s = s1; av += 1; }
            }
        }
        if sw % 50 == 49 {
            let ra = at as f64 / nt.max(1) as f64;
            let fa = if ra > 0.45 { 1.1 } else if ra < 0.35 { 0.9 } else { 1.0 };
            dt = (dt * fa).clamp(1e-7, 0.5); dr = (dr * fa).clamp(1e-7, 0.5);
            let rv = av as f64 / nv.max(1) as f64;
            dv = (dv * if rv > 0.45 { 1.1 } else if rv < 0.35 { 0.9 } else { 1.0 }).clamp(1e-9, 0.1);
            at = 0; nt = 0; av = 0; nv = 0;
        }
        if std::env::var("AN_V").is_ok() && report > 0 && sw % report == 0 {
            eprintln!("sw {sw} t {t:.2} bP {bp:.1} r {:.4} s {:.5} phi {:.4} dt {dt:.1e} dv {dv:.1e} shape_fail {shape_fail}", sys.r, sys.s, n as f64 * area(sys.r) / (sys.s * sys.s));
        }
    }
    // if the schedule did not reach r = 0, force it with box growth (report how much)
    let r_end = sys.r;
    let mut grow = 1.0;
    if sys.r > 0.0 {
        let xs = sys.x.clone();
        let mut f = 1.0;
        loop {
            let xf: Vec<f64> = xs.iter().enumerate().map(|(k, &v)| if k % 3 == 2 { v } else { v * f }).collect();
            if sys.all_ok(&xf, sys.s * f, 0.0) { sys.x = xf; sys.s *= f; sys.r = 0.0; grow = f; break; }
            f *= 1.01;
            if f > 2.0 { break; }
        }
    }
    let mut txt = format!("{} {:?}\n", n, sys.s);
    for i in 0..n { txt += &format!("{:?} {:?} {:?}\n", sys.x[3 * i], sys.x[3 * i + 1], (sys.x[3 * i + 2].to_degrees()).rem_euclid(90.0)); }
    std::fs::write(&out, txt).unwrap();
    println!("{{\"s\": {:.12}, \"r_end\": {:.4}, \"grow\": {:.4}, \"phi\": {:.5}, \"shape_fail\": {}, \"sec\": {:.2}}}",
             sys.s, r_end, grow, n as f64 / (sys.s * sys.s), shape_fail, t_start.elapsed().as_secs_f64());
}

// ---------------------------------------------------------------- regional melt (10-08)
fn verts_h(x: f64, y: f64, t: f64, h: f64) -> [(f64, f64); 4] { verts(x, y, t, h) }

/// SAT gap of two squares with half-sides hi, hj.
fn sat_gap2(xi: f64, yi: f64, ti: f64, hi: f64, xj: f64, yj: f64, tj: f64, hj: f64) -> f64 {
    let mut best = f64::NEG_INFINITY;
    let (dx, dy) = (xj - xi, yj - yi);
    for &t in &[ti, ti + FRAC_PI_2, tj, tj + FRAC_PI_2] {
        let (ax, ay) = (t.cos(), t.sin());
        let dc = (dx * ax + dy * ay).abs();
        let ri = hi * ((ti - t).cos().abs() + (ti - t).sin().abs());
        let rj = hj * ((tj - t).cos().abs() + (tj - t).sin().abs());
        best = best.max(dc - ri - rj);
    }
    best
}

fn sq_dist2(xi: f64, yi: f64, ti: f64, hi: f64, xj: f64, yj: f64, tj: f64, hj: f64) -> f64 {
    if sat_gap2(xi, yi, ti, hi, xj, yj, tj, hj) <= 0.0 { return 0.0; }
    let (a, b) = (verts_h(xi, yi, ti, hi), verts_h(xj, yj, tj, hj));
    let mut d = f64::INFINITY;
    for k in 0..4 { for e in 0..4 {
        d = d.min(pt_seg(a[k].0, a[k].1, b[e], b[(e + 1) % 4]));
        d = d.min(pt_seg(b[k].0, b[k].1, a[e], a[(e + 1) % 4]));
    } }
    d
}

/// Rounded squares with per-particle radius rr[i] (inner half-side 0.5 - rr[i]).
struct Sys2 { n: usize, x: Vec<f64>, s: f64, rr: Vec<f64> }
impl Sys2 {
    fn ov(&self, x: f64, y: f64, t: f64, ri: f64, j: usize) -> bool {
        let (xj, yj, tj, rj) = (self.x[3 * j], self.x[3 * j + 1], self.x[3 * j + 2], self.rr[j]);
        let d2 = (x - xj).powi(2) + (y - yj).powi(2);
        if d2 < 1.0 - 1e-12 { return true; }
        let (hi, hj) = (0.5 - ri, 0.5 - rj);
        let rc = hi * SQRT_2 + ri + hj * SQRT_2 + rj;
        if d2 >= rc * rc { return false; }
        let g = sat_gap2(x, y, t, hi, xj, yj, tj, hj);
        if ri + rj <= 0.0 { return g < 0.0; }
        if g >= ri + rj { return false; }
        if g <= 0.0 { return true; }
        sq_dist2(x, y, t, hi, xj, yj, tj, hj) < ri + rj
    }
    fn wall(&self, x: f64, y: f64, t: f64, ri: f64, s: f64) -> bool {
        let w = (0.5 - ri) * (t.cos().abs() + t.sin().abs()) + ri;
        x >= w && x <= s - w && y >= w && y <= s - w
    }
    fn ok(&self, i: usize, x: f64, y: f64, t: f64, ri: f64) -> bool {
        self.wall(x, y, t, ri, self.s) && (0..self.n).all(|j| j == i || !self.ov(x, y, t, ri, j))
    }
    fn all_ok(&self, xs: &[f64], s: f64) -> bool {
        let tmp = Sys2 { n: self.n, x: xs.to_vec(), s, rr: self.rr.clone() };
        (0..self.n).all(|i| { let (x, y, t) = (xs[3 * i], xs[3 * i + 1], xs[3 * i + 2]);
            tmp.wall(x, y, t, tmp.rr[i], s) && (i + 1..self.n).all(|j| !tmp.ov(x, y, t, tmp.rr[i], j)) })
    }
}

/// anneal melt --in F --out G --cx X --cy Y --rad R --rmax 0.1 --sweeps 4000 --bp 2000 --seed K
/// Squares within R of (cx, cy) get corner radius rmax (1 - d/R) (ramp up over t in [0, .25], hold to .55, down to 0 by
/// .85, then square); only squares within R + 1.5 move; NPT box moves at pressure bp.  Residual rounding at the end is
/// removed by the smallest uniform box growth.  Output packer format + JSON summary.
fn melt(a: &[String]) {
    let inp: String = arg(a, "--in", String::new());
    let out: String = arg(a, "--out", String::from("melt.txt"));
    let (cx, cy, rad): (f64, f64, f64) = (arg(a, "--cx", 0.0), arg(a, "--cy", 0.0), arg(a, "--rad", 2.5));
    let rmax: f64 = arg(a, "--rmax", 0.1);
    let sweeps: usize = arg(a, "--sweeps", 4000);
    let bp: f64 = arg(a, "--bp", 2000.0);
    let mut rng = Rng::new(arg(a, "--seed", 1u64));
    let txt = std::fs::read_to_string(&inp).expect("read --in");
    let mut it = txt.split_whitespace().map(|w| w.parse::<f64>().unwrap());
    let n = it.next().unwrap() as usize;
    let s0 = it.next().unwrap();
    let mut x = Vec::with_capacity(3 * n);
    for _ in 0..n { let (a1, b1, c1) = (it.next().unwrap(), it.next().unwrap(), it.next().unwrap()); x.extend_from_slice(&[a1, b1, c1.to_radians()]); }
    let w: Vec<f64> = (0..n).map(|i| (1.0 - (x[3 * i] - cx).hypot(x[3 * i + 1] - cy) / rad).max(0.0)).collect();
    let mobile: Vec<usize> = (0..n).filter(|&i| (x[3 * i] - cx).hypot(x[3 * i + 1] - cy) < rad + 1.5).collect();
    let mut sys = Sys2 { n, x, s: s0, rr: vec![0.0; n] };
    let prof = |t: f64| -> f64 { if t < 0.25 { t / 0.25 } else if t < 0.55 { 1.0 } else if t < 0.85 { 1.0 - (t - 0.55) / 0.3 } else { 0.0 } };
    let (mut dt, mut dr, mut dv) = (0.02f64, 0.02f64, 1e-3f64);
    let (mut at, mut nt, mut av, mut nv) = (0usize, 0usize, 0usize, 0usize);
    let t0 = std::time::Instant::now();
    for sw in 0..sweeps {
        let tt = sw as f64 / sweeps as f64;
        // shape: each particle toward its scheduled radius (growing the shape only where it fits)
        for &i in &mobile {
            let target = rmax * w[i] * prof(tt);
            if (target - sys.rr[i]).abs() < 1e-15 { continue; }
            let (xi, yi, ti) = (sys.x[3 * i], sys.x[3 * i + 1], sys.x[3 * i + 2]);
            if target > sys.rr[i] || sys.ok(i, xi, yi, ti, target) { sys.rr[i] = target; }
        }
        for _ in 0..mobile.len() {
            let i = mobile[rng.idx(mobile.len())];
            let (xi, yi, ti) = (sys.x[3 * i], sys.x[3 * i + 1], sys.x[3 * i + 2]);
            let (nx, ny, nth) = if rng.f() < 0.5 { (xi + dt * rng.u(), yi + dt * rng.u(), ti) } else { (xi, yi, ti + dr * rng.u()) };
            nt += 1;
            if sys.ok(i, nx, ny, nth, sys.rr[i]) { sys.x[3 * i] = nx; sys.x[3 * i + 1] = ny; sys.x[3 * i + 2] = nth; at += 1; }
        }
        if sw % 4 == 0 {
            let dl = dv * rng.u();
            let f = (0.5 * dl).exp();
            let s1 = sys.s * f;
            let lacc = -bp * (s1 * s1 - sys.s * sys.s) + (n as f64 + 1.0) * dl;
            nv += 1;
            if lacc >= 0.0 || rng.f() < lacc.exp() {
                let xf: Vec<f64> = sys.x.iter().enumerate().map(|(k, &v)| if k % 3 == 2 { v } else { v * f }).collect();
                if f >= 1.0 || sys.all_ok(&xf, s1) { sys.x = xf; sys.s = s1; av += 1; }
            }
        }
        if sw % 50 == 49 {
            let ra = at as f64 / nt.max(1) as f64;
            let fa = if ra > 0.45 { 1.1 } else if ra < 0.35 { 0.9 } else { 1.0 };
            dt = (dt * fa).clamp(1e-7, 0.3); dr = (dr * fa).clamp(1e-7, 0.3);
            let rv = av as f64 / nv.max(1) as f64;
            dv = (dv * if rv > 0.45 { 1.1 } else if rv < 0.35 { 0.9 } else { 1.0 }).clamp(1e-9, 0.05);
            at = 0; nt = 0; av = 0; nv = 0;
        }
    }
    let left = sys.rr.iter().filter(|&&r| r > 0.0).count();
    let mut grow = 1.0;
    if left > 0 {
        sys.rr = vec![0.0; n];
        let xs = sys.x.clone();
        let mut f = 1.0;
        loop {
            let xf: Vec<f64> = xs.iter().enumerate().map(|(k, &v)| if k % 3 == 2 { v } else { v * f }).collect();
            if sys.all_ok(&xf, sys.s * f) { sys.x = xf; sys.s *= f; grow = f; break; }
            f *= 1.001;
            if f > 1.5 { break; }
        }
    }
    let mut txt = format!("{} {:?}\n", n, sys.s);
    for i in 0..n { txt += &format!("{:?} {:?} {:?}\n", sys.x[3 * i], sys.x[3 * i + 1], (sys.x[3 * i + 2].to_degrees()).rem_euclid(90.0)); }
    std::fs::write(&out, txt).unwrap();
    println!("{{\"s\": {:.12}, \"s0\": {:.12}, \"mobile\": {}, \"rounded_left\": {}, \"grow\": {:.5}, \"sec\": {:.2}}}",
             sys.s, s0, mobile.len(), left, grow, t0.elapsed().as_secs_f64());
}

// ---------------------------------------------------------------- schedule-driven anneal (10-09)
/// Piecewise curve over progress t in [0, 1]: "t0:v0,t1:v1,..." (or a single constant).  log = interpolate ln v.
struct Curve { k: Vec<(f64, f64)>, log: bool }
impl Curve {
    fn parse(s: &str, log: bool) -> Curve {
        let mut k: Vec<(f64, f64)> = Vec::new();
        for part in s.split(',').filter(|p| !p.is_empty()) {
            let mut it = part.split(':');
            let a: f64 = it.next().unwrap().parse().expect("curve knot");
            match it.next() { Some(b) => k.push((a, b.parse().expect("curve value"))), None => k.push((0.0, a)) }
        }
        if k.len() == 1 { k.push((1.0, k[0].1)); }
        k.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        Curve { k, log }
    }
    fn at(&self, t: f64) -> f64 {
        let k = &self.k;
        if t <= k[0].0 { return k[0].1; }
        for w in k.windows(2) {
            if t <= w[1].0 {
                let u = if w[1].0 > w[0].0 { (t - w[0].0) / (w[1].0 - w[0].0) } else { 1.0 };
                return if self.log { (w[0].1.ln() * (1.0 - u) + w[1].1.ln() * u).exp() } else { w[0].1 * (1.0 - u) + w[1].1 * u };
            }
        }
        k[k.len() - 1].1
    }
}

/// anneal sched [--in F | --n N --phi0 0.25] --sweeps S --seed K --out G
///   --bp CURVE     pressure bP(t), log-interpolated (default "0:1,1:3000")
///   --r CURVE      corner-radius amplitude r(t) in [0, 0.5] (0.5 = disk, 0 = square; default "0:0.5,0.2:0.5,0.8:0,1:0")
///   --prot CURVE   probability that a particle move is a rotation (default 0.5)
///   --cx X --cy Y --rad R   radial region: weight w_i = max(0, 1 - d_i / R), particle radius = r(t) w_i; only squares within
///                  R + 1.5 move (default: no region, w = 1, all move)
///   --box-every K  box move every K sweeps (default 1)
///   --snaps M      write M evenly spaced snapshots (t, s, coords, radii) to G.traj
/// Shape steps per particle toward its target radius (growth of the shape only where it fits: a heuristic, not detailed
/// balance); the residual rounding at the end is removed by the smallest uniform box growth.  JSON summary on stdout.
fn sched(a: &[String]) {
    let out: String = arg(a, "--out", String::from("sched.txt"));
    let sweeps: usize = arg(a, "--sweeps", 4000);
    let mut rng = Rng::new(arg(a, "--seed", 1u64));
    let bpc = Curve::parse(&arg(a, "--bp", String::from("0:1,1:3000")), true);
    let rc = Curve::parse(&arg(a, "--r", String::from("0:0.5,0.2:0.5,0.8:0,1:0")), false);
    let prc = Curve::parse(&arg(a, "--prot", String::from("0.5")), false);
    let box_every: usize = arg(a, "--box-every", 1usize).max(1);
    let nsnap: usize = arg(a, "--snaps", 0usize);
    // orientational bias on Q4 = mean cos(4 theta) (1: all axis-aligned, -1: all at 45 deg), in units of kT:
    //   lin:  U = lam(t) n Q4                    (lam > 0 penalises axis alignment)
    //   harm: U = lam(t) n (Q4 - q0(t))^2 / 2    (umbrella window)
    let lamc = Curve::parse(&arg(a, "--bias-lam", String::from("0")), false);
    let q0c = Curve::parse(&arg(a, "--bias-q0", String::from("0")), false);
    let harm = arg(a, "--bias", String::from("lin")) == "harm";
    let log_every: usize = arg(a, "--log-every", 0usize);
    let lag_tol: f64 = arg(a, "--lag-tol", 1e9);       // default off (old behaviour); e.g. 0.002
    let lag_grow: f64 = arg(a, "--lag-grow", 0.001);
    let mut n_grow = 0usize;
    let log_from: f64 = arg(a, "--log-from", 0.0);
    let inp: String = arg(a, "--in", String::new());
    // start: file, or random sequential placement of rounded squares at packing fraction phi0
    let (n, s0, x) = if !inp.is_empty() {
        let txt = std::fs::read_to_string(&inp).expect("read --in");
        let mut it = txt.split_whitespace().map(|w| w.parse::<f64>().unwrap());
        let n = it.next().unwrap() as usize;
        let s0 = it.next().unwrap();
        let mut x = Vec::with_capacity(3 * n);
        for _ in 0..n { let (a1, b1, c1) = (it.next().unwrap(), it.next().unwrap(), it.next().unwrap()); x.extend_from_slice(&[a1, b1, c1.to_radians()]); }
        (n, s0, x)
    } else {
        let n: usize = arg(a, "--n", 110);
        let phi0: f64 = arg(a, "--phi0", 0.25);
        let r0 = rc.at(0.0);
        let s0 = (n as f64 * area(r0) / phi0).sqrt();
        let mut sys = Sys2 { n: 0, x: Vec::new(), s: s0, rr: Vec::new() };
        let mut tries = 0usize;
        while sys.n < n {
            tries += 1; assert!(tries < 10_000_000, "placement failed");
            let (x, y, t) = (rng.f() * s0, rng.f() * s0, rng.f() * FRAC_PI_2);
            if sys.ok(usize::MAX, x, y, t, r0) { sys.x.extend_from_slice(&[x, y, t]); sys.rr.push(r0); sys.n += 1; }
        }
        (n, s0, sys.x)
    };
    let region = a.iter().any(|q| q == "--rad");
    let (cx, cy, rad): (f64, f64, f64) = (arg(a, "--cx", 0.0), arg(a, "--cy", 0.0), arg(a, "--rad", 1e9));
    let w: Vec<f64> = (0..n).map(|i| if region { (1.0 - (x[3 * i] - cx).hypot(x[3 * i + 1] - cy) / rad).max(0.0) } else { 1.0 }).collect();
    let mobile: Vec<usize> = (0..n).filter(|&i| !region || (x[3 * i] - cx).hypot(x[3 * i + 1] - cy) < rad + 1.5).collect();
    // initial radii: from a file the shapes start square (r = 0) and grow rounding as scheduled; from scratch: r(0) w
    let r_init: Vec<f64> = if inp.is_empty() { (0..n).map(|i| rc.at(0.0) * w[i]).collect() } else { vec![0.0; n] };
    let mut sys = Sys2 { n, x, s: s0, rr: r_init };
    let (mut dt, mut dr, mut dv) = (0.05f64, 0.05f64, 1e-3f64);
    let (mut at, mut nt, mut ar, mut nr, mut av, mut nv) = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
    let t_start = std::time::Instant::now();
    let mut traj = String::new();
    let mut s_min = f64::INFINITY;
    let mut lag_max = 0.0f64;
    let mut c4: f64 = (0..n).map(|i| (4.0 * sys.x[3 * i + 2]).cos()).sum();
    let mut logtxt = String::new();
    let mut q_sum = 0.0f64; let mut q_cnt = 0usize;
    for sw in 0..sweeps {
        let tt = sw as f64 / sweeps as f64;
        let bp = bpc.at(tt);
        let lam = lamc.at(tt);
        let q0 = q0c.at(tt);
        let ubias = |q: f64| -> f64 { if harm { 0.5 * lam * n as f64 * (q - q0).powi(2) } else { lam * n as f64 * q } };
        let ramp = rc.at(tt).clamp(0.0, 0.5);
        let prot = prc.at(tt).clamp(0.0, 1.0);
        let mut lag = 0.0f64;
        for &i in &mobile {
            let target = ramp * w[i];
            if (target - sys.rr[i]).abs() < 1e-15 { continue; }
            let (xi, yi, ti) = (sys.x[3 * i], sys.x[3 * i + 1], sys.x[3 * i + 2]);
            if target > sys.rr[i] || sys.ok(i, xi, yi, ti, target) { sys.rr[i] = target; }
            lag = lag.max(sys.rr[i] - target);
        }
        lag_max = lag_max.max(if tt > 0.9 { lag } else { 0.0 });
        // keep shapes on schedule: while some particle lags its target radius by > lag_tol, expand the box a little
        // (uniform rescale, always feasible); pressure recompresses afterwards.  Heuristic, as the shape steps.
        if lag > lag_tol {
            let f = 1.0 + lag_grow;
            for k in 0..3 * n { if k % 3 != 2 { sys.x[k] *= f; } }
            sys.s *= f;
            n_grow += 1;
        }
        for _ in 0..mobile.len() {
            let i = mobile[rng.idx(mobile.len())];
            let (xi, yi, ti) = (sys.x[3 * i], sys.x[3 * i + 1], sys.x[3 * i + 2]);
            let rot = rng.f() < prot;
            let (nx, ny, nth) = if !rot { (xi + dt * rng.u(), yi + dt * rng.u(), ti) } else { (xi, yi, ti + dr * rng.u()) };
            if rot { nr += 1 } else { nt += 1 }
            if rot && lam != 0.0 {
                let c4n = c4 - (4.0 * ti).cos() + (4.0 * nth).cos();
                let du = ubias(c4n / n as f64) - ubias(c4 / n as f64);
                if du > 0.0 && rng.f() >= (-du).exp() { continue; }
            }
            if sys.ok(i, nx, ny, nth, sys.rr[i]) {
                if rot { c4 += (4.0 * nth).cos() - (4.0 * ti).cos(); }
                sys.x[3 * i] = nx; sys.x[3 * i + 1] = ny; sys.x[3 * i + 2] = nth;
                if rot { ar += 1 } else { at += 1 }
            }
        }
        if log_every > 0 && sw % log_every == 0 && tt >= log_from {
            c4 = (0..n).map(|i| (4.0 * sys.x[3 * i + 2]).cos()).sum();      // resync (rounding drift)
            logtxt += &format!("{:.5} {:.10} {:.6}\n", tt, sys.s, c4 / n as f64);
            q_sum += c4 / n as f64; q_cnt += 1;
        }
        if sw % box_every == 0 {
            let dl = dv * rng.u();
            let f = (0.5 * dl).exp();
            let s1 = sys.s * f;
            let lacc = -bp * (s1 * s1 - sys.s * sys.s) + (n as f64 + 1.0) * dl;
            nv += 1;
            if lacc >= 0.0 || rng.f() < lacc.exp() {
                let xf: Vec<f64> = sys.x.iter().enumerate().map(|(k, &v)| if k % 3 == 2 { v } else { v * f }).collect();
                if f >= 1.0 || sys.all_ok(&xf, s1) { sys.x = xf; sys.s = s1; av += 1; }
            }
        }
        if sw % 50 == 49 {
            let adj = |acc: usize, tot: usize| if tot == 0 { 1.0 } else { let r = acc as f64 / tot as f64; if r > 0.45 { 1.1 } else if r < 0.35 { 0.9 } else { 1.0 } };
            dt = (dt * adj(at, nt)).clamp(1e-8, 0.5); dr = (dr * adj(ar, nr)).clamp(1e-8, 0.5);
            dv = (dv * adj(av, nv)).clamp(1e-10, 0.05);
            at = 0; nt = 0; ar = 0; nr = 0; av = 0; nv = 0;
        }
        s_min = s_min.min(sys.s);
        if nsnap > 0 && (sw + 1) % (sweeps / nsnap).max(1) == 0 {
            traj += &format!("# t {:.4} s {:.12} bp {:.4e} r {:.4}\n{} {:?}\n", (sw + 1) as f64 / sweeps as f64, sys.s, bp, ramp, n, sys.s);
            for i in 0..n { traj += &format!("{:?} {:?} {:?} {:?}\n", sys.x[3 * i], sys.x[3 * i + 1], (sys.x[3 * i + 2].to_degrees()).rem_euclid(90.0), sys.rr[i]); }
        }
    }
    let s_round = sys.s;
    let left = sys.rr.iter().filter(|&&r| r > 0.0).count();
    let mut grow = 1.0;
    if left > 0 {
        sys.rr = vec![0.0; n];
        let xs = sys.x.clone();
        let mut f = 1.0;
        loop {
            let xf: Vec<f64> = xs.iter().enumerate().map(|(k, &v)| if k % 3 == 2 { v } else { v * f }).collect();
            if sys.all_ok(&xf, sys.s * f) { sys.x = xf; sys.s *= f; grow = f; break; }
            f *= 1.001;
            if f > 2.0 { break; }
        }
    }
    let mut txt = format!("{} {:?}\n", n, sys.s);
    for i in 0..n { txt += &format!("{:?} {:?} {:?}\n", sys.x[3 * i], sys.x[3 * i + 1], (sys.x[3 * i + 2].to_degrees()).rem_euclid(90.0)); }
    std::fs::write(&out, txt).unwrap();
    if nsnap > 0 { std::fs::write(format!("{out}.traj"), traj).unwrap(); }
    if log_every > 0 { std::fs::write(format!("{out}.log"), logtxt).unwrap(); }
    let q4: f64 = (0..n).map(|i| (4.0 * sys.x[3 * i + 2]).cos()).sum::<f64>() / n as f64;
    println!("{{\"s\": {:.12}, \"s0\": {:.12}, \"s_round\": {:.12}, \"s_min\": {:.12}, \"mobile\": {}, \"rounded_left\": {}, \"grow\": {:.5}, \"lag_late\": {:.4}, \"q4\": {:.5}, \"q4_mean\": {:.5}, \"lag_grows\": {}, \"sec\": {:.2}}}",
             sys.s, s0, s_round, s_min, mobile.len(), left, grow, lag_max, q4, if q_cnt > 0 { q_sum / q_cnt as f64 } else { f64::NAN }, n_grow, t_start.elapsed().as_secs_f64());
}

// ---------------------------------------------------------------- GLS feasibility search (10-15, TT3 arm L)
/// anneal gls --in F --out G --shrink D --ms T --seed K [--ng 32 --nl 16 --strike-iters 8 --strikes 1000000]
/// Sparrow-style (Gardeyn & Wauters, guided local search for strip packing), for unit squares in a square box.
/// Target side S = s (1 - D).  Shrink like Sparrow's strip shrink: on each axis, squares whose centre lies beyond a random
/// cut slide inward by s - S (so the overlap sits on two seams, not everywhere).  Then: repeatedly move each colliding
/// square to the best of ng global + nl local samples (+ the current spot) under the weighted overlap, refined by a
/// coordinate search; overlap of a pair = SAT penetration depth (exact minimal translation for convex polygons), wall
/// overlap = depth outside the box.  When strike-iters passes bring no 1 % improvement: GLS weight update (overlapping
/// pairs x (1.2 .. 2 by relative depth), others decay x 0.95 toward 1) = one strike.  Stops when the raw overlap is
/// < 1e-12 (feasible), or at the time / strike budget.  Writes the lowest-raw-overlap state at side S (the explorer's
/// screen resolves what is left).  JSON summary on stdout.
struct Gls { n: usize, s: f64, x: Vec<f64>, w: Vec<f32>, ww: Vec<f32>, cell: f64, nc: usize, grid: Vec<Vec<usize>>, at: Vec<usize> }

fn pen(xi: f64, yi: f64, ti: f64, xj: f64, yj: f64, tj: f64) -> f64 {
    let d2 = (xi - xj).powi(2) + (yi - yj).powi(2);
    if d2 >= 2.0 { return 0.0; }
    (-sat_gap(xi, yi, ti, xj, yj, tj, 0.5)).max(0.0)
}


/// SAT penetration depth of squares i, j (half-side 0.5) and the unit axis u along which moving j by +u separates them
fn pen_axis(xi: f64, yi: f64, ti: f64, xj: f64, yj: f64, tj: f64) -> (f64, f64, f64) {
    let (dx, dy) = (xj - xi, yj - yi);
    if dx * dx + dy * dy >= 2.0 { return (0.0, 0.0, 0.0); }
    let mut best = (f64::INFINITY, 0.0, 0.0);
    for &t in &[ti, ti + FRAC_PI_2, tj, tj + FRAC_PI_2] {
        let (ax, ay) = (t.cos(), t.sin());
        let pr = dx * ax + dy * ay;
        let ri = 0.5 * ((ti - t).cos().abs() + (ti - t).sin().abs());
        let rj = 0.5 * ((tj - t).cos().abs() + (tj - t).sin().abs());
        let ov = ri + rj - pr.abs();
        if ov < best.0 { let sg = if pr >= 0.0 { 1.0 } else { -1.0 }; best = (ov, sg * ax, sg * ay); }
    }
    if best.0 <= 0.0 { (0.0, 0.0, 0.0) } else { best }
}

impl Gls {
    fn cidx(&self, x: f64, y: f64) -> usize {
        let cx = ((x / self.cell).floor().max(0.0) as usize).min(self.nc - 1);
        let cy = ((y / self.cell).floor().max(0.0) as usize).min(self.nc - 1);
        cy * self.nc + cx
    }
    fn rebuild(&mut self) {
        for g in self.grid.iter_mut() { g.clear(); }
        for i in 0..self.n { let c = self.cidx(self.x[3 * i], self.x[3 * i + 1]); self.grid[c].push(i); self.at[i] = c; }
    }
    fn place(&mut self, i: usize, x: f64, y: f64, t: f64) {
        self.x[3 * i] = x; self.x[3 * i + 1] = y; self.x[3 * i + 2] = t;
        let c = self.cidx(x, y);
        if c != self.at[i] {
            let o = self.at[i]; let p = self.grid[o].iter().position(|&k| k == i).unwrap(); self.grid[o].swap_remove(p);
            self.grid[c].push(i); self.at[i] = c;
        }
    }
    fn wall(&self, x: f64, y: f64, t: f64) -> f64 {
        let h = 0.5 * (t.cos().abs() + t.sin().abs());
        (h - x).max(0.0) + (x - (self.s - h)).max(0.0) + (h - y).max(0.0) + (y - (self.s - h)).max(0.0)
    }
    /// weighted (or raw if !wt) overlap of square i placed at (x, y, t) with its neighbours and the walls
    fn cost(&self, i: usize, x: f64, y: f64, t: f64, wt: bool) -> f64 {
        let mut c = self.wall(x, y, t) * if wt { self.ww[i] as f64 } else { 1.0 };
        let cx = ((x / self.cell).floor().max(0.0) as isize).min(self.nc as isize - 1);
        let cy = ((y / self.cell).floor().max(0.0) as isize).min(self.nc as isize - 1);
        for dy in -1..=1 { for dx in -1..=1 {
            let (a, b) = (cx + dx, cy + dy);
            if a < 0 || b < 0 || a >= self.nc as isize || b >= self.nc as isize { continue; }
            for &j in &self.grid[b as usize * self.nc + a as usize] {
                if j == i { continue; }
                let p = pen(x, y, t, self.x[3 * j], self.x[3 * j + 1], self.x[3 * j + 2]);
                if p > 0.0 { c += p * if wt { self.w[i * self.n + j] as f64 } else { 1.0 }; }
            }
        } }
        c
    }
    fn raw_total(&self) -> (f64, Vec<usize>) {
        let mut tot = 0.0; let mut col = Vec::new();
        for i in 0..self.n {
            let c = self.cost(i, self.x[3 * i], self.x[3 * i + 1], self.x[3 * i + 2], false);
            if c > 0.0 { col.push(i); }
            tot += c;
        }
        (tot, col)
    }
    /// weighted squared overlap of square i at (x, y, t) (for the continuous relaxation)
    fn e2(&self, i: usize, x: f64, y: f64, t: f64) -> f64 {
        let h = 0.5 * (t.cos().abs() + t.sin().abs());
        let wl = [(h - x).max(0.0), (x - (self.s - h)).max(0.0), (h - y).max(0.0), (y - (self.s - h)).max(0.0)];
        let mut c = self.ww[i] as f64 * wl.iter().map(|v| v * v).sum::<f64>();
        let cx = ((x / self.cell).floor().max(0.0) as isize).min(self.nc as isize - 1);
        let cy = ((y / self.cell).floor().max(0.0) as isize).min(self.nc as isize - 1);
        for dy in -1..=1 { for dx in -1..=1 {
            let (a, b) = (cx + dx, cy + dy);
            if a < 0 || b < 0 || a >= self.nc as isize || b >= self.nc as isize { continue; }
            for &j in &self.grid[b as usize * self.nc + a as usize] {
                if j == i { continue; }
                let p = pen(x, y, t, self.x[3 * j], self.x[3 * j + 1], self.x[3 * j + 2]);
                if p > 0.0 { c += p * p * self.w[i * self.n + j] as f64; }
            }
        } }
        c
    }
    /// gradient descent on E = sum w p^2 (pairs once) + walls; translation gradient analytic, rotation by central
    /// differences; step backtracks on E.  Returns E.
    fn relax(&mut self, iters: usize, eta: &mut f64) -> f64 {
        let n = self.n;
        let etot = |g: &Gls| -> f64 { (0..n).map(|i| g.e2(i, g.x[3 * i], g.x[3 * i + 1], g.x[3 * i + 2])).sum::<f64>() * 0.5 };
        let mut e = etot(self);
        for _ in 0..iters {
            if e < 1e-26 { break; }
            let mut gr = vec![0.0; 3 * n];
            for i in 0..n {
                let (x, y, t) = (self.x[3 * i], self.x[3 * i + 1], self.x[3 * i + 2]);
                let h = 0.5 * (t.cos().abs() + t.sin().abs());
                let wv = self.ww[i] as f64;
                gr[3 * i] += 2.0 * wv * (-(h - x).max(0.0) + (x - (self.s - h)).max(0.0));
                gr[3 * i + 1] += 2.0 * wv * (-(h - y).max(0.0) + (y - (self.s - h)).max(0.0));
                let c = self.at[i]; let (cx, cy) = ((c % self.nc) as isize, (c / self.nc) as isize);
                let mut any = (h - x).max(0.0) + (x - (self.s - h)).max(0.0) + (h - y).max(0.0) + (y - (self.s - h)).max(0.0) > 0.0;
                for dy in -1..=1isize { for dx in -1..=1isize {
                    let (a, b) = (cx + dx, cy + dy);
                    if a < 0 || b < 0 || a >= self.nc as isize || b >= self.nc as isize { continue; }
                    for &j in &self.grid[b as usize * self.nc + a as usize] {
                        if j <= i { continue; }
                        let (p, ux, uy) = pen_axis(x, y, t, self.x[3 * j], self.x[3 * j + 1], self.x[3 * j + 2]);
                        if p > 0.0 {
                            let f = 2.0 * self.w[i * n + j] as f64 * p;
                            gr[3 * i] += f * ux; gr[3 * i + 1] += f * uy; gr[3 * j] -= f * ux; gr[3 * j + 1] -= f * uy;
                            any = true;
                        }
                    }
                } }
                if any {
                    let d = 1e-6;
                    gr[3 * i + 2] += (self.e2(i, x, y, t + d) - self.e2(i, x, y, t - d)) / (2.0 * d) * 0.5;
                }
            }
            // rotation terms above count pairs from i's side only (x 0.5 of the symmetric sum, same for j's own term)
            let x0 = self.x.clone();
            loop {
                for k in 0..3 * n { self.x[k] = x0[k] - *eta * gr[k]; }
                self.rebuild();
                let e1 = etot(self);
                if e1 < e { e = e1; *eta = (*eta * 1.2).min(1.0); break; }
                *eta *= 0.5;
                if *eta < 1e-9 { self.x = x0.clone(); self.rebuild(); *eta = 0.05; return e; }
            }
        }
        e
    }
    fn bump(&mut self) {
        // GLS update: overlapping pairs (and walls) up by 1.2 .. 2 by relative depth; others decay toward 1
        let n = self.n;
        let mut pairs = Vec::new(); let mut walls = Vec::new(); let mut mx: f64 = 0.0;
        for i in 0..n {
            let (x, y, t) = (self.x[3 * i], self.x[3 * i + 1], self.x[3 * i + 2]);
            let wv = self.wall(x, y, t); if wv > 0.0 { walls.push((i, wv)); mx = mx.max(wv); }
            for j in i + 1..n {
                let p = pen(x, y, t, self.x[3 * j], self.x[3 * j + 1], self.x[3 * j + 2]);
                if p > 0.0 { pairs.push((i, j, p)); mx = mx.max(p); }
            }
        }
        for v in self.w.iter_mut() { *v = (*v * 0.95).max(1.0); }
        for v in self.ww.iter_mut() { *v = (*v * 0.95).max(1.0); }
        for &(i, j, p) in &pairs {
            let f = (1.2 + 0.8 * p / mx) as f32 / 0.95;
            let v = (self.w[i * n + j] * f).max(1.0); self.w[i * n + j] = v; self.w[j * n + i] = v;
        }
        for &(i, p) in &walls { self.ww[i] = (self.ww[i] * (1.2 + 0.8 * p / mx) as f32 / 0.95).max(1.0); }
    }
    fn move_item(&mut self, i: usize, rng: &mut Rng, ng: usize, nl: usize) {
        let (x0, y0, t0) = (self.x[3 * i], self.x[3 * i + 1], self.x[3 * i + 2]);
        let mut best = (self.cost(i, x0, y0, t0, true), x0, y0, t0);
        let lo = 0.5; let hi = self.s - 0.5;
        for k in 0..ng + nl {
            let (x, y, t) = if k < ng { (lo + rng.f() * (hi - lo), lo + rng.f() * (hi - lo), rng.f() * FRAC_PI_2) }
                            else { let sc = [0.3, 0.1, 0.03][k % 3]; (x0 + sc * rng.u(), y0 + sc * rng.u(), t0 + sc * rng.u()) };
            let c = self.cost(i, x, y, t, true);
            if c < best.0 { best = (c, x, y, t); }
        }
        // coordinate refinement (adaptive step)
        let (mut c, mut x, mut y, mut t) = best;
        let mut st = 0.05;
        let mut evals = 0;
        while st > 1e-6 && c > 0.0 && evals < 80 {
            let mut imp = false;
            for &(dx, dy, dt) in &[(st, 0.0, 0.0), (-st, 0.0, 0.0), (0.0, st, 0.0), (0.0, -st, 0.0), (0.0, 0.0, st), (0.0, 0.0, -st)] {
                let cc = self.cost(i, x + dx, y + dy, t + dt, true); evals += 1;
                if cc < c { c = cc; x += dx; y += dy; t += dt; imp = true; break; }
            }
            if !imp { st *= 0.5; }
        }
        self.place(i, x, y, t);
    }
}

fn gls(a: &[String]) {
    let inp: String = arg(a, "--in", String::new());
    let out: String = arg(a, "--out", String::from("gls.txt"));
    let shrink: f64 = arg(a, "--shrink", 1e-3);
    let ms: f64 = arg(a, "--ms", 500.0);
    let ng: usize = arg(a, "--ng", 32); let nl: usize = arg(a, "--nl", 16);
    let ring = arg(a, "--ring", 0u32) == 1; let relax_it: usize = arg(a, "--relax", 20); let mut eta = 0.05f64; let siters: usize = arg(a, "--strike-iters", 8); let max_strikes: usize = arg(a, "--strikes", 1_000_000);
    let mut rng = Rng::new(arg(a, "--seed", 1u64));
    let txt = std::fs::read_to_string(&inp).expect("read --in");
    let num = |w: &str| -> f64 { match w.split_once('/') { Some((p, q)) => p.parse::<f64>().unwrap() / q.parse::<f64>().unwrap(), None => w.parse().unwrap() } };
    let lines: Vec<Vec<&str>> = txt.lines().filter(|l| !l.trim().is_empty() && !l.starts_with('#')).map(|l| l.split_whitespace().collect()).collect();
    let n: usize = lines[0][0].parse().unwrap();
    let s0 = num(lines[0][1]);
    let mut x = Vec::with_capacity(3 * n);
    for l in &lines[1..=n] { x.extend_from_slice(&[num(l[0]), num(l[1]), num(l[2]).to_radians()]); }
    let s = s0 * (1.0 - shrink); let d = s0 - s;
    for ax in 0..2 {
        let cut = rng.f() * s0;
        for i in 0..n { if x[3 * i + ax] > cut { x[3 * i + ax] -= d; } }
    }
    let cell = 1.5f64; let nc = ((s / cell).ceil() as usize).max(1);
    let mut g = Gls { n, s, x, w: vec![1.0; n * n], ww: vec![1.0; n], cell, nc, grid: vec![Vec::new(); nc * nc], at: vec![0; n] };
    g.rebuild();
    let t0 = std::time::Instant::now();
    let (mut cur, mut col) = g.raw_total();
    let (o0, c0) = (cur, col.len());
    let (mut best, mut bestx) = (cur, g.x.clone());
    let (mut strikes, mut passes, mut stale, mut ref_cost) = (0usize, 0usize, 0usize, cur);
    while cur > 1e-12 && t0.elapsed().as_secs_f64() * 1e3 < ms && strikes < max_strikes {
        // one pass: move every colliding square (random order)
        for k in (1..col.len()).rev() { let j = rng.idx(k + 1); col.swap(k, j); }
        if relax_it > 0 { g.relax(relax_it, &mut eta); }
        for &i in &col { g.move_item(i, &mut rng, ng, nl); }
        if ring {                                   // let slack travel: neighbours of colliding squares take local moves
            let mut seen = vec![false; n]; for &i in &col { seen[i] = true; }
            let mut nb = Vec::new();
            for &i in &col { let c = g.at[i]; let (cx, cy) = ((c % g.nc) as isize, (c / g.nc) as isize);
                for dy in -1..=1isize { for dx in -1..=1isize { let (a2, b2) = (cx + dx, cy + dy);
                    if a2 < 0 || b2 < 0 || a2 >= g.nc as isize || b2 >= g.nc as isize { continue; }
                    for &j in &g.grid[b2 as usize * g.nc + a2 as usize] { if !seen[j] { seen[j] = true; nb.push(j); } } } } }
            for &j in &nb { g.move_item(j, &mut rng, 0, nl); }
        }
        passes += 1;
        let r = g.raw_total(); cur = r.0; col = r.1;
        if cur < best { best = cur; bestx = g.x.clone(); }
        if cur < 0.99 * ref_cost { ref_cost = cur; stale = 0; } else { stale += 1; }
        if stale >= siters { g.bump(); strikes += 1; stale = 0; ref_cost = cur; }
    }
    let mut txt = format!("{} {:?}\n", n, s);
    for i in 0..n { txt += &format!("{:?} {:?} {:?}\n", bestx[3 * i], bestx[3 * i + 1], (bestx[3 * i + 2].to_degrees()).rem_euclid(90.0)); }
    std::fs::write(&out, txt).unwrap();
    println!("{{\"s\": {:.12}, \"s0\": {:.12}, \"feasible\": {}, \"overlap0\": {:.3e}, \"colliding0\": {}, \"overlap\": {:.3e}, \"passes\": {}, \"strikes\": {}, \"sec\": {:.3}}}",
             s, s0, best <= 1e-12, o0, c0, best, passes, strikes, t0.elapsed().as_secs_f64());
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    match a.get(1).map(|s| s.as_str()) {
        Some("test") => test(),
        Some("run") => run(&a),
        Some("melt") => melt(&a),
        Some("sched") => sched(&a),
        Some("gls") => gls(&a),
        _ => eprintln!("usage: anneal test | anneal run --n N --seed K --sweeps S --bp0 --bp1 --t0 --t1 --p --out F"),
    }
}

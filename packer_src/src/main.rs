//! packer: heuristic search for packings of n unit squares in a square of side s.
//!
//! Screening only (f64).  A result is a candidate; certification is a separate exact step.
//!
//! Formulation: fixed container side s; minimise the overlap energy
//!     E = sum_pairs pen(i,j)^2 + sum_i sum_walls viol(i,wall)^2
//! where pen is the separating-axis penetration depth of two unit squares (0 when disjoint)
//! and viol is how far a square sticks out of [0,s]^2.  E = 0 iff the configuration packs.
//! Local solver: L-BFGS over (x, y, theta) with a Verlet neighbour list.
//! Global: basin hopping with region moves; on success the target side is lowered
//! ("descending target"), so a run reports the tightest side it reached.
//!
//! File format (text): first line "n s", then n lines "x y theta_degrees".

use std::env;
use std::fs;
use std::io::Write;
use std::time::Instant;

const H: f64 = 0.5; // half side of a unit square
const PI2: f64 = std::f64::consts::FRAC_PI_2;
const SQRT2: f64 = std::f64::consts::SQRT_2;

#[inline]
fn sgn(x: f64) -> f64 {
    if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 }
}

// ---------------------------------------------------------------- rng
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        let mut r = Rng(seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(0x632BE59BD9B4E019) | 1);
        for _ in 0..8 { r.next(); }
        r
    }
    fn next(&mut self) -> u64 {
        // xorshift64*
        let mut x = self.0;
        x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn f(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 }
    fn below(&mut self, n: usize) -> usize { (self.f() * n as f64) as usize % n.max(1) }
    fn gauss(&mut self) -> f64 {
        let u1 = self.f().max(1e-300);
        let u2 = self.f();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// ---------------------------------------------------------------- geometry
/// Penetration depth of squares i, j (positive = overlap) and its gradient wrt
/// (dx, dy, ti, tj), dx = xj - xi.  Returns None when separated.
#[inline]
fn pen(dx: f64, dy: f64, ti: f64, tj: f64) -> Option<(f64, [f64; 4])> {
    let mut best = f64::INFINITY;
    let mut bg = [0.0; 4];
    // axes of i (owner i, other j) then axes of j (owner j, other i)
    for owner in 0..2 {
        let (to, tot) = if owner == 0 { (ti, tj) } else { (tj, ti) };
        for k in 0..2 {
            let phi = to + k as f64 * PI2;
            let (sp, cp) = phi.sin_cos();
            let proj = dx * cp + dy * sp;
            let al = phi - tot;
            let (sa, ca) = al.sin_cos();
            let o = H + H * (ca.abs() + sa.abs()) - proj.abs();
            if o <= 0.0 {
                return None;
            }
            if o < best {
                best = o;
                let ga = H * (-sgn(ca) * sa + sgn(sa) * ca); // d r_other / d alpha
                let sp_ = sgn(proj);
                let dphi = ga - sp_ * (-dx * sp + dy * cp); // d o / d theta_owner
                let dother = -ga; // d o / d theta_other
                let ddx = -sp_ * cp;
                let ddy = -sp_ * sp;
                if owner == 0 { bg = [ddx, ddy, dphi, dother]; } else { bg = [ddx, ddy, dother, dphi]; }
            }
        }
    }
    Some((best, bg))
}

/// Separation gap (largest separating-axis gap; negative = overlap) for reporting.
fn gap(dx: f64, dy: f64, ti: f64, tj: f64) -> f64 {
    let mut best = f64::NEG_INFINITY;
    for owner in 0..2 {
        let (to, tot) = if owner == 0 { (ti, tj) } else { (tj, ti) };
        for k in 0..2 {
            let phi = to + k as f64 * PI2;
            let (sp, cp) = phi.sin_cos();
            let proj = dx * cp + dy * sp;
            let (sa, ca) = (phi - tot).sin_cos();
            let o = H + H * (ca.abs() + sa.abs()) - proj.abs();
            best = best.max(-o);
        }
    }
    best
}

// ---------------------------------------------------------------- state
struct Sys {
    n: usize,
    s: f64,
    skin: f64,
    pairs: Vec<(u32, u32)>,
    refpos: Vec<f64>,
    frozen: Vec<bool>,
    evals: u64,
    des: f64,
}

impl Sys {
    fn new(n: usize, s: f64) -> Self {
        Sys { n, s, skin: 0.6, pairs: vec![], refpos: vec![f64::NAN; 2 * n], frozen: vec![false; n], evals: 0, des: 0.0 }
    }

    fn rebuild(&mut self, v: &[f64]) {
        let n = self.n;
        let cut = SQRT2 + self.skin;
        let cut2 = cut * cut;
        // cell grid
        let lo = -1.0;
        let span = self.s + 2.0;
        let nc = ((span / cut).floor() as usize).max(1);
        let cs = span / nc as f64;
        let cell = |x: f64| -> usize { (((x - lo) / cs).floor().max(0.0) as usize).min(nc - 1) };
        let mut heads = vec![u32::MAX; nc * nc];
        let mut next = vec![u32::MAX; n];
        for i in 0..n {
            let c = cell(v[3 * i]) * nc + cell(v[3 * i + 1]);
            next[i] = heads[c];
            heads[c] = i as u32;
        }
        self.pairs.clear();
        for i in 0..n {
            let (cx, cy) = (cell(v[3 * i]), cell(v[3 * i + 1]));
            for ax in cx.saturating_sub(1)..=(cx + 1).min(nc - 1) {
                for ay in cy.saturating_sub(1)..=(cy + 1).min(nc - 1) {
                    let mut j = heads[ax * nc + ay];
                    while j != u32::MAX {
                        let ju = j as usize;
                        if ju > i {
                            let dx = v[3 * ju] - v[3 * i];
                            let dy = v[3 * ju + 1] - v[3 * i + 1];
                            if dx * dx + dy * dy < cut2 && !(self.frozen[i] && self.frozen[ju]) {
                                self.pairs.push((i as u32, j));
                            }
                        }
                        j = next[ju];
                    }
                }
            }
        }
        for i in 0..n {
            self.refpos[2 * i] = v[3 * i];
            self.refpos[2 * i + 1] = v[3 * i + 1];
        }
    }

    fn check_list(&mut self, v: &[f64]) {
        let lim = (self.skin * 0.5) * (self.skin * 0.5);
        for i in 0..self.n {
            let dx = v[3 * i] - self.refpos[2 * i];
            let dy = v[3 * i + 1] - self.refpos[2 * i + 1];
            if !(dx * dx + dy * dy <= lim) {
                self.rebuild(v);
                return;
            }
        }
    }

    /// Energy and gradient.
    fn energy(&mut self, v: &[f64], g: &mut [f64]) -> f64 {
        self.evals += 1;
        self.check_list(v);
        for x in g.iter_mut() { *x = 0.0; }
        let s = self.s;
        let mut e = 0.0;
        let mut des = 0.0;
        for i in 0..self.n {
            if self.frozen[i] { continue; }
            let (x, y, t) = (v[3 * i], v[3 * i + 1], v[3 * i + 2]);
            let (st, ct) = t.sin_cos();
            let w = H * (ct.abs() + st.abs());
            let dw = H * (-sgn(ct) * st + sgn(st) * ct);
            let a = w - x;
            if a > 0.0 { e += a * a; g[3 * i] -= 2.0 * a; g[3 * i + 2] += 2.0 * a * dw; }
            let a = x + w - s;
            if a > 0.0 { e += a * a; g[3 * i] += 2.0 * a; g[3 * i + 2] += 2.0 * a * dw; des -= 2.0 * a; }
            let a = w - y;
            if a > 0.0 { e += a * a; g[3 * i + 1] -= 2.0 * a; g[3 * i + 2] += 2.0 * a * dw; }
            let a = y + w - s;
            if a > 0.0 { e += a * a; g[3 * i + 1] += 2.0 * a; g[3 * i + 2] += 2.0 * a * dw; des -= 2.0 * a; }
        }
        for &(i, j) in &self.pairs {
            let (i, j) = (i as usize, j as usize);
            let dx = v[3 * j] - v[3 * i];
            let dy = v[3 * j + 1] - v[3 * i + 1];
            if dx * dx + dy * dy >= 2.0 { continue; } // circumcircles disjoint
            if let Some((p, d)) = pen(dx, dy, v[3 * i + 2], v[3 * j + 2]) {
                e += p * p;
                let c = 2.0 * p;
                g[3 * j] += c * d[0];
                g[3 * i] -= c * d[0];
                g[3 * j + 1] += c * d[1];
                g[3 * i + 1] -= c * d[1];
                g[3 * i + 2] += c * d[2];
                g[3 * j + 2] += c * d[3];
            }
        }
        for i in 0..self.n {
            if self.frozen[i] { g[3 * i] = 0.0; g[3 * i + 1] = 0.0; g[3 * i + 2] = 0.0; }
        }
        self.des = des;
        e
    }

    /// Per-square energy share (pairs split evenly) for targeting moves.
    fn per_square(&mut self, v: &[f64]) -> Vec<f64> {
        let mut g = vec![0.0; v.len()];
        self.energy(v, &mut g);
        let s = self.s;
        let mut out = vec![0.0; self.n];
        for i in 0..self.n {
            let (x, y, t) = (v[3 * i], v[3 * i + 1], v[3 * i + 2]);
            let w = H * (t.cos().abs() + t.sin().abs());
            for a in [w - x, x + w - s, w - y, y + w - s] {
                if a > 0.0 { out[i] += a * a; }
            }
        }
        for &(i, j) in &self.pairs {
            let (i, j) = (i as usize, j as usize);
            let dx = v[3 * j] - v[3 * i];
            let dy = v[3 * j + 1] - v[3 * i + 1];
            if let Some((p, _)) = pen(dx, dy, v[3 * i + 2], v[3 * j + 2]) {
                out[i] += 0.5 * p * p;
                out[j] += 0.5 * p * p;
            }
        }
        out
    }
}

// ---------------------------------------------------------------- L-BFGS
static mut ETOL_V: f64 = 1e-26;
#[allow(non_snake_case)]
#[inline] fn ETOL() -> f64 { unsafe { ETOL_V } }
static mut STALL_REL: f64 = 1e-12;
static mut STALL_N: usize = 20;

fn lbfgs(f: &mut dyn FnMut(&[f64], &mut [f64]) -> f64, v: &mut [f64], maxit: usize, stop: f64, stall_rel: f64, stall_n: usize) -> f64 {
    let dim = v.len();
    let m = 8;
    let mut g = vec![0.0; dim];
    let mut e = f(v, &mut g);
    let mut ss: Vec<Vec<f64>> = Vec::with_capacity(m);
    let mut ys: Vec<Vec<f64>> = Vec::with_capacity(m);
    let mut rho: Vec<f64> = Vec::with_capacity(m);
    let mut d = vec![0.0; dim];
    let mut xn = vec![0.0; dim];
    let mut gn = vec![0.0; dim];
    let mut alpha = vec![0.0; m];
    let mut fails = 0;
    for _it in 0..maxit {
        if e < stop { break; }
        // two-loop recursion
        for k in 0..dim { d[k] = -g[k]; }
        let h = ss.len();
        for k in (0..h).rev() {
            let a = rho[k] * dot(&ss[k], &d);
            alpha[k] = a;
            for q in 0..dim { d[q] -= a * ys[k][q]; }
        }
        if h > 0 {
            let gam = dot(&ss[h - 1], &ys[h - 1]) / dot(&ys[h - 1], &ys[h - 1]);
            for q in 0..dim { d[q] *= gam; }
        } else {
            let gmax = g.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
            let sc = if gmax > 0.0 { (0.05 / gmax).min(1.0) } else { 1.0 };
            for q in 0..dim { d[q] *= sc; }
        }
        for k in 0..h {
            let b = rho[k] * dot(&ys[k], &d);
            for q in 0..dim { d[q] += ss[k][q] * (alpha[k] - b); }
        }
        let mut gd = dot(&g, &d);
        if !(gd < 0.0) {
            // not a descent direction: reset
            ss.clear(); ys.clear(); rho.clear();
            let gmax = g.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
            if gmax == 0.0 { break; }
            let sc = (0.05 / gmax).min(1.0);
            for q in 0..dim { d[q] = -g[q] * sc; }
            gd = dot(&g, &d);
        }
        // cap step length (max coordinate move 0.25)
        let dmax = d.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        let mut step = if dmax > 0.25 { 0.25 / dmax } else { 1.0 };
        let mut ok = false;
        for _ in 0..40 {
            for q in 0..dim { xn[q] = v[q] + step * d[q]; }
            let en = f(&xn, &mut gn);
            if en <= e + 1e-4 * step * gd {
                // accept
                let mut sv = vec![0.0; dim];
                let mut yv = vec![0.0; dim];
                for q in 0..dim { sv[q] = xn[q] - v[q]; yv[q] = gn[q] - g[q]; }
                let sy = dot(&sv, &yv);
                if sy > 1e-30 {
                    if ss.len() == m { ss.remove(0); ys.remove(0); rho.remove(0); }
                    rho.push(1.0 / sy);
                    ss.push(sv);
                    ys.push(yv);
                }
                v.copy_from_slice(&xn);
                g.copy_from_slice(&gn);
                let rel = (e - en) / e.abs().max(1e-300);
                e = en;
                ok = true;
                if rel < stall_rel { fails += 1; } else { fails = 0; }
                break;
            }
            step *= 0.5;
        }
        if !ok {
            if ss.is_empty() { break; }
            ss.clear(); ys.clear(); rho.clear();
            fails += 1;
        }
        if fails > stall_n { break; }
    }
    e
}

fn minimize(sys: &mut Sys, v: &mut [f64], maxit: usize) -> f64 {
    let (st, sn) = unsafe { (STALL_REL, STALL_N) };
    lbfgs(&mut |x: &[f64], g: &mut [f64]| sys.energy(x, g), v, maxit, ETOL(), st, sn)
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 0..a.len() { s += a[i] * b[i]; }
    s
}

// ---------------------------------------------------------------- io
fn read_cfg(path: &str) -> (f64, Vec<f64>, Vec<bool>) {
    let txt = fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let mut lines = txt.lines().filter(|l| !l.trim().is_empty() && !l.starts_with('#'));
    let head: Vec<f64> = lines.next().unwrap().split_whitespace().map(|t| t.parse().unwrap()).collect();
    let (n, s) = (head[0] as usize, head[1]);
    let mut v = Vec::with_capacity(3 * n);
    let mut fz = Vec::with_capacity(n);
    for _ in 0..n {
        let toks: Vec<&str> = lines.next().unwrap().split_whitespace().collect();
        v.push(toks[0].parse::<f64>().unwrap());
        v.push(toks[1].parse::<f64>().unwrap());
        v.push(toks[2].parse::<f64>().unwrap().to_radians());
        fz.push(toks.get(3).map_or(false, |t| *t == "F"));
    }
    (s, v, fz)
}

fn write_cfg(path: &str, s: f64, v: &[f64], fz: &[bool]) {
    let n = v.len() / 3;
    let mut out = format!("{} {:.17}\n", n, s);
    for i in 0..n {
        let deg = v[3 * i + 2].to_degrees().rem_euclid(90.0);
        out += &format!("{:.17} {:.17} {:.15}{}\n", v[3 * i], v[3 * i + 1], deg, if fz[i] { " F" } else { "" });
    }
    let tmp = format!("{path}.tmp");
    fs::write(&tmp, out).unwrap();
    fs::rename(&tmp, path).unwrap();
}

/// Smallest pairwise separation gap and smallest wall clearance (exact-ish report, O(n^2)).
fn report(s: f64, v: &[f64]) -> (f64, f64) {
    let n = v.len() / 3;
    let mut mg = f64::INFINITY;
    for i in 0..n {
        for j in i + 1..n {
            let dx = v[3 * j] - v[3 * i];
            let dy = v[3 * j + 1] - v[3 * i + 1];
            if dx * dx + dy * dy < 2.5 {
                mg = mg.min(gap(dx, dy, v[3 * i + 2], v[3 * j + 2]));
            }
        }
    }
    let mut mw = f64::INFINITY;
    for i in 0..n {
        let t = v[3 * i + 2];
        let w = H * (t.cos().abs() + t.sin().abs());
        mw = mw.min(v[3 * i] - w).min(s - v[3 * i] - w).min(v[3 * i + 1] - w).min(s - v[3 * i + 1] - w);
    }
    (mg, mw)
}

// ---------------------------------------------------------------- squeeze
/// From a configuration feasible (E < ETOL()) at sys.s, shrink the side as far as possible.
/// Scales the configuration about the container's centre... actually about the origin
/// corner, by s1/s0, then re-minimises.  Returns the smallest feasible side reached.
/// Penalty-continuation squeeze: minimise s + mu * E(config; s) over (config, s) for increasing mu, then repair to
/// strict feasibility (E < ETOL) by growing s minimally.  Returns the feasible side.
fn squeeze_pen(sys: &mut Sys, v: &mut Vec<f64>, mu0: f64, mu1: f64, maxit: usize) -> f64 {
    let n = sys.n;
    let mut z = v.clone();
    z.push(sys.s);
    let mut mu = mu0;
    while mu <= mu1 * 1.0001 {
        let f = &mut |x: &[f64], g: &mut [f64]| -> f64 {
            sys.s = x[3 * n];
            let e = sys.energy(&x[..3 * n], &mut g[..3 * n]);
            for q in 0..3 * n { g[q] *= mu; }
            g[3 * n] = 1.0 + mu * sys.des;
            x[3 * n] + mu * e
        };
        lbfgs(f, &mut z, maxit, f64::NEG_INFINITY, 1e-13, 10);
        mu *= 10.0;
    }
    let mut s = z[3 * n];
    v.copy_from_slice(&z[..3 * n]);
    // repair: smallest growth that makes the configuration strictly feasible
    let mut grow = 1e-12;
    loop {
        let mut w = v.clone();
        sys.s = s + grow;
        let e = minimize(sys, &mut w, maxit);
        if e < ETOL() { v.copy_from_slice(&w); s += grow; break; }
        grow *= 4.0;
        if grow > 0.1 { break; }
    }
    sys.s = s;
    s
}

fn squeeze(sys: &mut Sys, v: &mut Vec<f64>, maxit: usize, min_step: f64) -> f64 {
    squeeze_from(sys, v, maxit, min_step, 1e-3)
}

fn squeeze_from(sys: &mut Sys, v: &mut Vec<f64>, maxit: usize, min_step: f64, step0: f64) -> f64 {
    let mut step = step0;
    let mut s0 = sys.s;
    let mut trial = v.clone();
    while step > min_step {
        let s1 = s0 - step;
        let f = s1 / s0;
        for i in 0..v.len() / 3 {
            trial[3 * i] = v[3 * i] * f;
            trial[3 * i + 1] = v[3 * i + 1] * f;
            trial[3 * i + 2] = v[3 * i + 2];
        }
        sys.s = s1;
        let e = minimize(sys, &mut trial, maxit);
        if e < ETOL() {
            v.copy_from_slice(&trial);
            s0 = s1;
            step *= 1.6;
        } else {
            step *= 0.4;
        }
    }
    sys.s = s0;
    s0
}

// ---------------------------------------------------------------- moves
/// Jam lines: horizontal (true) or vertical (false) lines crossed by >= floor(s)+1 near-axis squares,
/// which forces overlap at side s.  Returns (horizontal, members) with duplicates removed.
fn jam_lines(v: &[f64], s: f64) -> Vec<(bool, Vec<usize>)> {
    jam_lines_k(v, s.floor() as usize + 1)
}

fn jam_lines_k(v: &[f64], need: usize) -> Vec<(bool, Vec<usize>)> {
    let n = v.len() / 3;
    let ax: Vec<usize> = (0..n).filter(|&i| {
        let a = v[3 * i + 2].rem_euclid(PI2);
        a.min(PI2 - a) < 0.02
    }).collect();
    let mut out: Vec<(bool, Vec<usize>)> = vec![];
    for horiz in [true, false] {
        let c = if horiz { 1 } else { 0 };
        for &i in &ax {
            for off in [-0.45, 0.0, 0.45] {
                let y0 = v[3 * i + c] + off;
                let mut m: Vec<usize> = ax.iter().cloned().filter(|&j| (v[3 * j + c] - y0).abs() < 0.5).collect();
                if m.len() >= need {
                    m.sort();
                    if !out.iter().any(|(h, mm)| *h == horiz && *mm == m) { out.push((horiz, m)); }
                }
            }
        }
    }
    out
}

/// Move square i to the least-overlapping of `tries` sampled poses (angle 0 or a random free square's angle).
fn best_hole(rng: &mut Rng, v: &mut [f64], s: f64, i: usize, free: &[usize], tries: usize) {
    let n = v.len() / 3;
    let mut best = (f64::INFINITY, v[3 * i], v[3 * i + 1], v[3 * i + 2]);
    for _ in 0..tries {
        let x = 0.5 + rng.f() * (s - 1.0);
        let y = 0.5 + rng.f() * (s - 1.0);
        let t = if rng.f() < 0.5 { 0.0 } else { v[3 * free[rng.below(free.len())] + 2] };
        let w = H * (t.cos().abs() + t.sin().abs());
        let mut e = 0.0;
        for a in [w - x, x + w - s, w - y, y + w - s] { if a > 0.0 { e += a * a; } }
        for j in 0..n {
            if j == i { continue; }
            let (dx, dy) = (v[3 * j] - x, v[3 * j + 1] - y);
            if dx * dx + dy * dy >= 2.0 { continue; }
            if let Some((p, _)) = pen(dx, dy, t, v[3 * j + 2]) { e += p * p; if e > best.0 { break; } }
        }
        if e < best.0 { best = (e, x, y, t); }
    }
    v[3 * i] = best.1; v[3 * i + 1] = best.2; v[3 * i + 2] = best.3;
}

fn perturb(rng: &mut Rng, sys: &mut Sys, v: &mut [f64], per: &[f64], amp: f64) -> u8 {
    let n = sys.n;
    let s = sys.s;
    let free: Vec<usize> = (0..n).filter(|&i| !sys.frozen[i]).collect();
    if free.is_empty() { return 0; }
    // pick a centre: half the time the worst-loaded square, else random
    let pick_hot = |rng: &mut Rng| -> usize {
        let tot: f64 = free.iter().map(|&i| per[i]).sum();
        if tot <= 0.0 { return free[rng.below(free.len())]; }
        let mut r = rng.f() * tot;
        for &i in &free {
            r -= per[i];
            if r <= 0.0 { return i; }
        }
        free[free.len() - 1]
    };
    let c = if rng.f() < 0.5 { pick_hot(rng) } else { free[rng.below(free.len())] };
    let (cx, cy) = (v[3 * c], v[3 * c + 1]);
    let kind = rng.below(9) as u8;
    match kind {
        0 => {
            // local shake
            let r = 1.0 + 2.5 * rng.f();
            let sp = amp * (0.05 + 0.3 * rng.f());
            let sa = amp * 0.15 * rng.f();
            for &i in &free {
                let d = (v[3 * i] - cx).hypot(v[3 * i + 1] - cy);
                if d < r {
                    v[3 * i] += sp * rng.gauss();
                    v[3 * i + 1] += sp * rng.gauss();
                    v[3 * i + 2] += sa * rng.gauss();
                }
            }
        }
        1 => {
            // relocate one square to a random spot
            let i = c;
            v[3 * i] = 0.5 + rng.f() * (s - 1.0);
            v[3 * i + 1] = 0.5 + rng.f() * (s - 1.0);
            v[3 * i + 2] = match rng.below(3) {
                0 => 0.0,
                1 => rng.f() * PI2,
                _ => v[3 * free[rng.below(free.len())] + 2],
            };
        }
        2 => {
            // re-angle one square
            let i = c;
            v[3 * i + 2] = match rng.below(3) {
                0 => 0.0,
                1 => v[3 * i + 2] + 0.3 * amp * rng.gauss(),
                _ => v[3 * free[rng.below(free.len())] + 2],
            };
        }
        3 => {
            // rigid rotation of a cluster
            let r = 1.0 + 2.5 * rng.f();
            let phi = amp * 0.15 * rng.gauss();
            let (sp, cp) = phi.sin_cos();
            for &i in &free {
                let (dx, dy) = (v[3 * i] - cx, v[3 * i + 1] - cy);
                if dx.hypot(dy) < r {
                    v[3 * i] = cx + cp * dx - sp * dy;
                    v[3 * i + 1] = cy + sp * dx + cp * dy;
                    v[3 * i + 2] += phi;
                }
            }
        }
        4 => {
            // rigid translation of a cluster
            let r = 1.0 + 3.0 * rng.f();
            let (tx, ty) = (amp * 0.3 * rng.gauss(), amp * 0.3 * rng.gauss());
            for &i in &free {
                if (v[3 * i] - cx).hypot(v[3 * i + 1] - cy) < r {
                    v[3 * i] += tx;
                    v[3 * i + 1] += ty;
                }
            }
        }
        6 => {
            // strip slide: squares in a unit-wide row (or column) through the centre, beyond a cut, translate along it
            let horiz = rng.f() < 0.5;
            let (pc, qc) = if horiz { (cy, cx) } else { (cx, cy) };
            let cutq = qc + (rng.f() - 0.5);
            let side = if rng.f() < 0.5 { 1.0 } else { -1.0 };
            let half = 0.5 + 0.5 * rng.f();
            let dlt = amp * 0.2 * rng.gauss();
            for &i in &free {
                let (p, q) = if horiz { (v[3 * i + 1], v[3 * i]) } else { (v[3 * i], v[3 * i + 1]) };
                if (p - pc).abs() < half && (q - cutq) * side > 0.0 {
                    if horiz { v[3 * i] += dlt; } else { v[3 * i + 1] += dlt; }
                }
            }
        }
        7 => {
            // best-hole relocate of the centre square
            best_hole(rng, v, s, c, &free, 300);
        }
        8 => {
            // row break: take a member out of a jam line (relocate to best hole, or tilt it in place)
            let lines = jam_lines(v, s);
            let i = if lines.is_empty() { c } else {
                let (_, m) = &lines[rng.below(lines.len())];
                m[rng.below(m.len())]
            };
            if sys.frozen[i] { return kind; }
            if rng.f() < 0.5 {
                best_hole(rng, v, s, i, &free, 300);
            } else {
                let sg = if rng.f() < 0.5 { 1.0 } else { -1.0 };
                v[3 * i + 2] += sg * (0.26 + 0.52 * rng.f());
            }
        }
        _ => {
            // swap poses of the centre with a random square of a different angle class
            let j = free[rng.below(free.len())];
            v.swap(3 * c, 3 * j);
            v.swap(3 * c + 1, 3 * j + 1);
        }
    }
    kind
}

// ---------------------------------------------------------------- commands
fn arg<T: std::str::FromStr>(args: &[String], key: &str, def: T) -> T {
    for k in 0..args.len() {
        if args[k] == key && k + 1 < args.len() {
            return args[k + 1].parse().ok().unwrap_or_else(|| panic!("bad value for {key}"));
        }
    }
    def
}
fn sarg(args: &[String], key: &str) -> Option<String> {
    (0..args.len()).find(|&k| args[k] == key && k + 1 < args.len()).map(|k| args[k + 1].clone())
}

// ---------------------------------------------------------------- event chains (hard-square moves)
const WALL: usize = usize::MAX;
const CEPS: f64 = 1e-12;

/// Does square i at pose (x, y, t) collide (beyond CEPS) with a wall or a neighbour in nb?  Returns the obstacle.
fn collide(v: &[f64], s: f64, i: usize, x: f64, y: f64, t: f64, nb: &[usize]) -> Option<usize> {
    let w = H * (t.cos().abs() + t.sin().abs());
    if x - w < -CEPS || x + w - s > CEPS || y - w < -CEPS || y + w - s > CEPS { return Some(WALL); }
    for &j in nb {
        if j == i { continue; }
        let (dx, dy) = (v[3 * j] - x, v[3 * j + 1] - y);
        if dx * dx + dy * dy >= 2.0 { continue; }
        if let Some((p, _)) = pen(dx, dy, t, v[3 * j + 2]) { if p > CEPS { return Some(j); } }
    }
    None
}

/// Slide square i along pose direction d = (dx, dy, dt) (unit in the position part, or pure rotation) for up to len,
/// stopping at the first contact.  Returns (distance moved, obstacle).
fn max_slide(v: &[f64], s: f64, i: usize, d: (f64, f64, f64), len: f64) -> (f64, Option<usize>) {
    let n = v.len() / 3;
    let reach = SQRT2 + len * (d.0 * d.0 + d.1 * d.1).sqrt() + 0.01;
    let nb: Vec<usize> = (0..n).filter(|&j| j != i && (v[3 * j] - v[3 * i]).hypot(v[3 * j + 1] - v[3 * i + 1]) < reach).collect();
    let pose = |t: f64| (v[3 * i] + t * d.0, v[3 * i + 1] + t * d.1, v[3 * i + 2] + t * d.2);
    let h = 0.04;
    let mut t: f64 = 0.0;
    loop {
        let tn = (t + h).min(len);
        let (x, y, a) = pose(tn);
        if collide(v, s, i, x, y, a, &nb).is_some() {
            let (mut lo, mut hi) = (t, tn);
            for _ in 0..60 {
                let mid = 0.5 * (lo + hi);
                let (x, y, a) = pose(mid);
                if collide(v, s, i, x, y, a, &nb).is_some() { hi = mid; } else { lo = mid; }
                if hi - lo < 1e-14 { break; }
            }
            let (x, y, a) = pose(hi);
            return (lo, collide(v, s, i, x, y, a, &nb));
        }
        t = tn;
        if t >= len { return (len, None); }
    }
}

/// One event chain starting at square i.  Translations pass the remaining length to the square hit;
/// rotations (d.0 = d.1 = 0) stop at the first contact.  Returns the number of events.
fn chain(v: &mut [f64], s: f64, i: usize, d: (f64, f64, f64), len: f64, frozen: &[bool]) -> usize {
    let mut cur = i;
    let mut rem = len;
    let mut ev = 0;
    while rem > 1e-12 && ev < 200 {
        if frozen[cur] { break; }
        let (t, hit) = max_slide(v, s, cur, d, rem);
        v[3 * cur] += t * d.0; v[3 * cur + 1] += t * d.1; v[3 * cur + 2] += t * d.2;
        rem -= t;
        ev += 1;
        match hit {
            Some(j) if j != WALL && (d.0 != 0.0 || d.1 != 0.0) => cur = j,
            _ => break,
        }
    }
    ev
}

fn random_dir(rng: &mut Rng) -> (f64, f64, f64) {
    let r = rng.f();
    if r < 0.45 {
        match rng.below(4) { 0 => (1.0, 0.0, 0.0), 1 => (-1.0, 0.0, 0.0), 2 => (0.0, 1.0, 0.0), _ => (0.0, -1.0, 0.0) }
    } else if r < 0.7 {
        let a = rng.f() * 2.0 * std::f64::consts::PI;
        (a.cos(), a.sin(), 0.0)
    } else {
        (0.0, 0.0, if rng.f() < 0.5 { 1.0 } else { -1.0 })
    }
}

fn random_cfg(rng: &mut Rng, n: usize, s: f64) -> Vec<f64> {
    let mut v = Vec::with_capacity(3 * n);
    for _ in 0..n {
        v.push(0.5 + rng.f() * (s - 1.0));
        v.push(0.5 + rng.f() * (s - 1.0));
        v.push(if rng.f() < 0.5 { 0.0 } else { rng.f() * PI2 });
    }
    v
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let cmd = args.get(1).cloned().unwrap_or_default();
    match cmd.as_str() {
        "relax" => {
            // packer relax --in f --s S [--out g] [--squeeze]
            let (s_file, mut v, fz) = read_cfg(&sarg(&args, "--in").expect("--in"));
            let s = arg(&args, "--s", s_file);
            let mut sys = Sys::new(v.len() / 3, s);
            sys.frozen = fz.clone();
            let t0 = Instant::now();
            let e = minimize(&mut sys, &mut v, arg(&args, "--maxit", 20000));
            let mut sfin = s;
            if args.iter().any(|a| a == "--squeeze-pen") && e < ETOL() {
                sfin = squeeze_pen(&mut sys, &mut v, arg(&args, "--mu0", 10.0f64), arg(&args, "--mu1", 1e9f64), 20000);
                let fin = arg(&args, "--finish", 0.0f64);
                if fin > 0.0 {
                    unsafe { ETOL_V = 1e-20; STALL_REL = 1e-6; STALL_N = 8; }
                    sfin = squeeze_from(&mut sys, &mut v, 3000, fin, 1e-4);
                    unsafe { ETOL_V = 1e-26; STALL_REL = 1e-12; STALL_N = 20; }
                }
            }
            if args.iter().any(|a| a == "--squeeze") && e < ETOL() {
                let etol = arg(&args, "--etol", 1e-26f64);
                let stall = arg(&args, "--stall", 1e-12f64);
                unsafe { ETOL_V = etol; STALL_REL = stall; STALL_N = if stall > 1e-12 { 8 } else { 20 }; }
                sfin = squeeze_from(&mut sys, &mut v, arg(&args, "--sq-maxit", 20000usize), arg(&args, "--min-step", 1e-10),
                                    arg(&args, "--step0", 1e-3f64));
                unsafe { ETOL_V = 1e-26; STALL_REL = 1e-12; STALL_N = 20; }
            }
            let (mg, mw) = report(sfin, &v);
            println!("E={e:.3e} s={sfin:.12} mingap={mg:.3e} minwall={mw:.3e} evals={} t={:.2}s",
                     sys.evals, t0.elapsed().as_secs_f64());
            if let Some(o) = sarg(&args, "--out") { write_cfg(&o, sfin, &v, &fz); }
        }
        "hhop" => {
            // hard-square basin hopping: loosen (scale up by gamma) -> event chains -> squeeze; Metropolis on the side.
            // packer hhop --in f --seed k --iters I --out g [--temp T] [--chains C] [--chainlen L] [--gmax G]
            let seed = arg(&args, "--seed", 1u64);
            let mut rng = Rng::new(seed);
            let (s0, mut v, fz) = read_cfg(&sarg(&args, "--in").expect("--in"));
            let n = v.len() / 3;
            let iters = arg(&args, "--iters", 100000usize);
            let temp = arg(&args, "--temp", 1e-4f64);
            let nch = arg(&args, "--chains", 8usize);
            let clen = arg(&args, "--chainlen", 1.0f64);
            let gmax = arg(&args, "--gmax", 0.02f64);
            let coarse = arg(&args, "--coarse", 1e-7f64);
            let etol_c = arg(&args, "--etol", 1e-20f64);
            let forbid = arg(&args, "--forbid-lines", 0usize);
            let soft = arg(&args, "--soft", 0.0f64);
            let out = sarg(&args, "--out").expect("--out");
            let mut sys = Sys::new(n, s0);
            sys.frozen = fz.clone();
            let e = minimize(&mut sys, &mut v, 20000);
            assert!(e < ETOL(), "start must be feasible at its side (E={e:.2e})");
            let mut scur = squeeze(&mut sys, &mut v, 20000, 1e-9);
            let mut best = scur;
            write_cfg(&out, best, &v, &fz);
            println!("start s={scur:.12}");
            let t0 = Instant::now();
            let (mut acc, mut evs) = (0u64, 0u64);
            for it in 0..iters {
                let mut w = v.clone();
                let g = gmax * (1e-3f64).powf(rng.f()); // log-uniform in [gmax/1000, gmax]
                let sw = scur * (1.0 + g);
                for i in 0..n { w[3 * i] *= 1.0 + g; w[3 * i + 1] *= 1.0 + g; }
                if rng.f() < soft {
                    // soften and re-harden: penalty squeeze from a low starting mu (transient overlaps allowed)
                    sys.s = sw;
                    let mu0 = 10f64.powf(1.0 + 3.0 * rng.f());
                    let snew = squeeze_pen(&mut sys, &mut w, mu0, 1e9, 3000);
                    if forbid > 0 && !jam_lines_k(&w, forbid).is_empty() { continue; }
                    if snew < scur || rng.f() < (-(snew - scur) / temp).exp() { v = w; scur = snew; acc += 1; }
                    if scur < best - 1e-6 {
                        best = scur; write_cfg(&out, best, &v, &fz);
                        println!("it={it} best s={best:.12} (soft) t={:.1}s", t0.elapsed().as_secs_f64());
                        std::io::stdout().flush().ok();
                    }
                    continue;
                }
                let m = 1 + rng.below(nch);
                for _ in 0..m {
                    let i = rng.below(n);
                    let d = random_dir(&mut rng);
                    let l = clen * (0.1 + rng.f());
                    evs += chain(&mut w, sw, i, d, l, &fz) as u64;
                }
                sys.s = sw;
                unsafe { STALL_REL = 1e-6; STALL_N = 8; ETOL_V = etol_c; }
                let e = minimize(&mut sys, &mut w, 3000);
                if e >= ETOL() { unsafe { STALL_REL = 1e-12; STALL_N = 20; ETOL_V = 1e-26; } continue; }
                let ev0 = sys.evals;
                let mut snew = squeeze_from(&mut sys, &mut w, 3000, coarse, (sw - scur).max(1e-6));
                if std::env::var("HHDBG").is_ok() { eprintln!("g={g:.2e} squeeze evals={} snew-scur={:.2e}", sys.evals - ev0, snew - scur); }
                unsafe { STALL_REL = 1e-12; STALL_N = 20; ETOL_V = 1e-26; }
                if forbid > 0 && !jam_lines_k(&w, forbid).is_empty() { continue; }
                if snew < scur || rng.f() < (-(snew - scur) / temp).exp() {
                    v = w; scur = snew; acc += 1;
                }
                if scur < best - 1e-6 {
                    best = scur;
                    write_cfg(&out, best, &v, &fz);
                    println!("it={it} best s={best:.12} t={:.1}s", t0.elapsed().as_secs_f64());
                    std::io::stdout().flush().ok();
                }
            }
            // final polish of the best configuration
            let (_, mut vb, _) = read_cfg(&out);
            sys.s = best;
            let eb = minimize(&mut sys, &mut vb, 20000);
            if eb < ETOL() {
                let sp = squeeze_from(&mut sys, &mut vb, 20000, 1e-10, 1e-6);
                if sp < best { best = sp; write_cfg(&out, best, &vb, &fz); }
            }
            println!("done best={best:.12} cur={scur:.12} acc={acc}/{iters} events={evs} evals={} t={:.1}s", sys.evals, t0.elapsed().as_secs_f64());
        }
        "hop" => {
            // packer hop (--in f | --n N) --s S --seed k --iters I --out g [--temp T] [--amp A]
            let seed = arg(&args, "--seed", 1u64);
            let mut rng = Rng::new(seed);
            let (s0, mut v, fz) = match sarg(&args, "--in") {
                Some(p) => read_cfg(&p),
                None => {
                    let n = arg(&args, "--n", 0usize);
                    let s = arg(&args, "--s", (n as f64).sqrt().ceil());
                    let fz = vec![false; n];
                    (s, random_cfg(&mut rng, n, s), fz)
                }
            };
            let n = v.len() / 3;
            let mut s = arg(&args, "--s", s0);
            let iters = arg(&args, "--iters", 100000usize);
            let temp = arg(&args, "--temp", 0.3f64);
            let amp = arg(&args, "--amp", 1.0f64);
            let maxit = arg(&args, "--maxit", 3000usize);
            let descend = arg(&args, "--descend", 1e-4f64);
            let out = sarg(&args, "--out").expect("--out");
            let quiet = args.iter().any(|a| a == "--quiet");
            let stall_rel = arg(&args, "--stall", 1e-5f64);
            let linepen = arg(&args, "--linepen", 0.0f64);
            let mut lpen_cur = linepen * jam_lines(&v, s).len() as f64;
            let mut sys = Sys::new(n, s);
            sys.frozen = fz.clone();
            // if the start is feasible at s, squeeze first
            let mut e = minimize(&mut sys, &mut v, 20000);
            let mut best_s = f64::INFINITY;
            let t0 = Instant::now();
            let mut acc = [0u64; 9];
            let mut tried = [0u64; 9];
            let mut best_e = e;
            for it in 0..iters {
                if e < ETOL() {
                    let sq = squeeze(&mut sys, &mut v, 20000, 1e-9);
                    if sq < best_s {
                        best_s = sq;
                        write_cfg(&out, sq, &v, &fz);
                        let (mg, mw) = report(sq, &v);
                        println!("it={it} FEASIBLE s={sq:.12} mingap={mg:.2e} minwall={mw:.2e} t={:.1}s",
                                 t0.elapsed().as_secs_f64());
                        std::io::stdout().flush().ok();
                    }
                    // descend: new target just below the best side
                    s = best_s - descend;
                    sys.s = s;
                    let f = s / sq;
                    for i in 0..n { v[3 * i] *= f; v[3 * i + 1] *= f; }
                    e = minimize(&mut sys, &mut v, maxit);
                    best_e = e;
                    lpen_cur = linepen * jam_lines(&v, s).len() as f64;
                    continue;
                }
                let per = sys.per_square(&v);
                let mut w = v.clone();
                let kind = perturb(&mut rng, &mut sys, &mut w, &per, amp) as usize;
                tried[kind] += 1;
                unsafe { STALL_REL = stall_rel; STALL_N = 8; }
                let ew = minimize(&mut sys, &mut w, maxit);
                unsafe { STALL_REL = 1e-12; STALL_N = 20; }
                let lpen_w = if linepen > 0.0 { linepen * jam_lines(&w, s).len() as f64 } else { 0.0 };
                let (fe, fw) = (e + lpen_cur, ew + lpen_w);
                let accept = fw < fe || (fw.max(1e-300) / fe.max(1e-300)).ln() < -temp * rng.f().max(1e-300).ln();
                if accept {
                    acc[kind] += 1;
                    v = w;
                    e = ew;
                    lpen_cur = lpen_w;
                }
                if e < best_e {
                    best_e = e;
                    if !quiet {
                        println!("it={it} s={s:.6} E={e:.3e} lines={} t={:.1}s", jam_lines(&v, s).len(), t0.elapsed().as_secs_f64());
                    }
                    if best_s == f64::INFINITY { write_cfg(&format!("{out}.inf"), s, &v, &fz); }
                }
            }
            println!("done: best_s={best_s:.12} final E={e:.3e} at s={s:.6} evals={} t={:.1}s acc/tried={:?}/{:?}",
                     sys.evals, t0.elapsed().as_secs_f64(), acc, tried);
        }
        _ => {
            eprintln!("usage: packer relax --in f [--s S] [--squeeze] [--out g]\n       packer hop (--in f | --n N) --s S --seed k --iters I --out g");
            std::process::exit(2);
        }
    }
}

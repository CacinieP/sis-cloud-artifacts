//! fq: fast smooth quench for packings of n unit squares in a square (screening, f64).
//!
//! Formulation (min s subject to non-overlap), smooth by construction:
//! * every nearby pair (i, j) carries its own separating line: unit normal u = (cos phi, sin phi) and offset d, measured
//!   from the midpoint M of the two centres; constraints  u.(p - M) - d <= 0  for the 4 corners p of i and
//!   d - u.(q - M) <= 0  for the 4 corners q of j;
//! * every corner of every square inside [0, s]^2 (4 constraints per corner).
//! Each constraint is smooth in (x, y, theta, phi, d, s), so the PHR augmented Lagrangian
//!     L = s + sum_k (mu/2) max(0, g_k + lam_k/mu)^2 - lam_k^2/(2 mu)
//! is C^1: no max-over-axes kink (the reason L-BFGS on SAT penetration stalls at face-to-face contacts).
//! Inner: L-BFGS over (x, y, theta)_i, s, (phi, d)_pairs.  Outer: multiplier update, mu increase, Verlet pair list.
//! End: exact SAT check and repair by uniform scaling, so the written packing is feasible in f64.
//!
//! File format as packer: "n s", then "x y theta_degrees" per square.
//!
//!   fq quench --in A --out B [--loosen 1.02] [--kick SIGMA --seed K] [--mu0 10] [--v]
//!   fq check --in A           (min pair gap, min wall clearance)
//!   --rect-dx DX --rect-dy DY: container [0, s + DX] x [0, s + DY] (s is still the reported / minimised quantity)
//!   --round-r0 R [--round-compress] [--hold-f F] [--round-steps K] [--round-p P] [--round-kick SIG]: soft-to-rigid
//!     pre-stage with rounded squares (see run_quench), then the usual quench + polish
//!   --dump-lp DIR: append every polish LP to DIR/lp-<pid>.jsonl (rows/columns with stable keys, Clarabel time; lp_bench.py)
//!   --freeze SPEC: squares by index ("0-99,105", 0-based, inclusive ranges) held fixed in absolute coordinates: no
//!     gradient, no LP / Newton column, no kick, not moved by --loosen / --hold-f / --shrink rescaling; frozen-frozen pairs
//!     carry no constraints (the frozen set is assumed feasible).  Only the final repair scaling (f64 margin, ~1e-13
//!     relative) moves them; the JSON reports the largest frozen displacement as frozen_dmax (10-22)

mod ldl;
mod lpipm;
use std::fs;
use std::time::Instant;
use std::sync::atomic::{AtomicU8, AtomicU64, AtomicUsize, Ordering};
use std::sync::OnceLock;
use std::sync::RwLock;

// --freeze: per-square mask (empty = nothing frozen); set per quench by run_quench
static FROZEN: RwLock<Vec<bool>> = RwLock::new(Vec::new());
fn frozen_mask(n: usize) -> Vec<bool> {
    let m = FROZEN.read().unwrap();
    if m.len() == n && m.iter().any(|&b| b) { m.clone() } else { Vec::new() }
}
fn set_frozen(spec: Option<String>, n: usize) {
    let mut m = vec![false; n];
    if let Some(sp) = spec {
        for part in sp.split(',').map(|q| q.trim()).filter(|q| !q.is_empty()) {
            let (a, b) = match part.split_once('-') {
                Some((a, b)) => (a.parse::<usize>().expect("--freeze"), b.parse::<usize>().expect("--freeze")),
                None => { let a = part.parse::<usize>().expect("--freeze"); (a, a) }
            };
            assert!(a <= b && b < n, "--freeze range {part} outside 0..{n}");
            for i in a..=b { m[i] = true; }
        }
    }
    *FROZEN.write().unwrap() = m;
}

// --dump-lp DIR: append every polish LP (rows, columns with stable keys, Clarabel time) to DIR/lp-<pid>.jsonl
static DUMP_LP: OnceLock<String> = OnceLock::new();
static TRACE_TR: OnceLock<bool> = OnceLock::new();
// 10-25 (relax as a fast move): --deadline SECS stops every phase at the deadline and returns the repaired state
// ("deadline": true); a quench whose pair list explodes (> PAIR_CAP x n: the side collapsed and squares piled up, the
// slow tail at n ~ 1000) stops at once ("diverged": true); --traj records (outer, s, viol, mu, sec) of the compress quench.
static DEADLINE: RwLock<Option<Instant>> = RwLock::new(None);
static DIVERGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static TRAJ_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static TRAJ: std::sync::Mutex<Vec<(usize, f64, f64, f64, f64)>> = std::sync::Mutex::new(Vec::new());
const PAIR_CAP: usize = 30;
fn wall_skip() -> bool { static W: OnceLock<bool> = OnceLock::new(); *W.get_or_init(|| std::env::var("FQ_WALLSKIP").map(|v| v != "0").unwrap_or(true)) }
fn late() -> bool { DEADLINE.read().unwrap().map_or(false, |d| Instant::now() >= d) }
fn stopped() -> bool { DIVERGED.load(Ordering::Relaxed) || late() }
/// JSON numbers: Rust prints inf / NaN, which JSON cannot hold -> null
fn json_finite(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < b.len() {
        let prev_ok = i == 0 || matches!(b[i - 1], b':' | b',' | b'[' | b' ');
        let mut hit = 0;
        for tok in ["-inf", "inf", "NaN"] {
            if prev_ok && s[i..].starts_with(tok) {
                let j = i + tok.len();
                if j == b.len() || matches!(b[j], b',' | b']' | b'}' | b' ') { hit = tok.len(); break; }
            }
        }
        if hit > 0 { out.push_str("null"); i += hit; } else { out.push(b[i] as char); i += 1; }
    }
    out
}
static LP_TAG: AtomicU8 = AtomicU8::new(0); // 0 step, 1 second-order correction, 2 flip search, 3 final force network
static LP_IT: AtomicUsize = AtomicUsize::new(0);
static IDENT_FPS: std::sync::Mutex<Vec<u64>> = std::sync::Mutex::new(Vec::new()); // --ident: load-network fingerprint per accepted step

const H: f64 = 0.5;

// Container and shape knobs (set per quench from args; f64 bits in atomics so `serve` can change them per request).
// Container = [0, s + dx] x [0, s + dy] (rectangle mode; 0, 0 = the square).  Rounded squares: inner half-side H - r,
// Minkowski-dilated by r (r = 0: unit squares; r = 0.5: unit disks).  In the lifted formulation that is the same
// separator / wall constraints on the inner corners with a margin r (still smooth), so the ALM handles it unchanged.
static RDX: AtomicU64 = AtomicU64::new(0);
static RDY: AtomicU64 = AtomicU64::new(0);
static RND: AtomicU64 = AtomicU64::new(0);
#[inline] fn rdx() -> f64 { f64::from_bits(RDX.load(Ordering::Relaxed)) }
#[inline] fn rdy() -> f64 { f64::from_bits(RDY.load(Ordering::Relaxed)) }
#[inline] fn rnd() -> f64 { f64::from_bits(RND.load(Ordering::Relaxed)) }
fn set_rnd(r: f64) { RND.store(r.to_bits(), Ordering::Relaxed); }
static SOFT: AtomicU64 = AtomicU64::new(0);   // soft squares: max |log aspect| eps (pre-stage only)
#[inline] fn soft_eps() -> f64 { f64::from_bits(SOFT.load(Ordering::Relaxed)) }
fn set_soft(e: f64) { SOFT.store(e.to_bits(), Ordering::Relaxed); }
/// Corner k of a w x h rectangle (area w h = 1 when soft), rotated by (c, s), with its derivative in the log aspect a
/// (w = e^{a/2}, h = e^{-a/2}).  With w = h = 1 this is corner_off.
#[inline]
fn corner_wh(c: f64, s: f64, k: usize, w: f64, h: f64) -> (f64, f64, f64, f64) {
    let (sx, sy) = CS[k];
    let hh = H - rnd();
    (hh * (c * sx * w - s * sy * h), hh * (s * sx * w + c * sy * h),
     hh * (c * sx * w + s * sy * h) * 0.5, hh * (s * sx * w - c * sy * h) * 0.5)
}
fn set_rect(dx: f64, dy: f64) { RDX.store(dx.to_bits(), Ordering::Relaxed); RDY.store(dy.to_bits(), Ordering::Relaxed); }
const SQRT2: f64 = std::f64::consts::SQRT_2;
const CS: [(f64, f64); 4] = [(1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)];

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        let mut r = Rng(seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(0x632BE59BD9B4E019) | 1);
        for _ in 0..8 { r.next(); }
        r
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn f(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 }
    fn gauss(&mut self) -> f64 {
        let u1 = self.f().max(1e-300);
        let u2 = self.f();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// ---------------------------------------------------------------- geometry helpers
#[inline]
fn corner_off(c: f64, s: f64, k: usize) -> (f64, f64) {
    let (sx, sy) = CS[k];
    let h = H - rnd();
    (h * (c * sx - s * sy), h * (s * sx + c * sy))
}

/// SAT gap of two unit squares (largest separating gap over the 4 face normals; negative = overlap depth).
fn sat_gap(xi: f64, yi: f64, ti: f64, xj: f64, yj: f64, tj: f64) -> f64 {
    let mut best = f64::NEG_INFINITY;
    let (dx, dy) = (xj - xi, yj - yi);
    for &t in &[ti, ti + std::f64::consts::FRAC_PI_2, tj, tj + std::f64::consts::FRAC_PI_2] {
        let (ax, ay) = (t.cos(), t.sin());
        let dc = (dx * ax + dy * ay).abs();
        let ri = H * ((ti - t).cos().abs() + (ti - t).sin().abs());
        let rj = H * ((tj - t).cos().abs() + (tj - t).sin().abs());
        best = best.max(dc - ri - rj);
    }
    best
}

/// Face separators of a pair (faces 0, 1 of i, then 0, 1 of j; normals oriented i -> j): (gap, phi, d).
fn face_seps(xi: f64, yi: f64, ti: f64, xj: f64, yj: f64, tj: f64) -> Vec<(f64, f64, f64)> {
    let (mx, my) = ((xi + xj) / 2.0, (yi + yj) / 2.0);
    let mut out = Vec::with_capacity(4);
    for &t0 in &[ti, ti + std::f64::consts::FRAC_PI_2, tj, tj + std::f64::consts::FRAC_PI_2] {
        let mut t = t0;
        if (xj - xi) * t.cos() + (yj - yi) * t.sin() < 0.0 { t += std::f64::consts::PI; }
        let (ux, uy) = (t.cos(), t.sin());
        let (mut ihi, mut jlo) = (f64::NEG_INFINITY, f64::INFINITY);
        for k in 0..4 {
            let (ox, oy) = corner_off(ti.cos(), ti.sin(), k);
            ihi = ihi.max(ux * (xi + ox - mx) + uy * (yi + oy - my));
            let (ox, oy) = corner_off(tj.cos(), tj.sin(), k);
            jlo = jlo.min(ux * (xj + ox - mx) + uy * (yj + oy - my));
        }
        out.push((jlo - ihi, t, (jlo + ihi) / 2.0));
    }
    out
}

/// Best initial separator for a pair: face normal (or centre line) with the largest gap, oriented i -> j.
fn init_sep(xi: f64, yi: f64, ti: f64, xj: f64, yj: f64, tj: f64) -> (f64, f64) {
    let (mx, my) = ((xi + xj) / 2.0, (yi + yj) / 2.0);
    let mut cands = vec![ti, ti + std::f64::consts::FRAC_PI_2, tj, tj + std::f64::consts::FRAC_PI_2];
    cands.push((yj - yi).atan2(xj - xi));
    let (mut bphi, mut bd, mut bg) = (0.0, 0.0, f64::NEG_INFINITY);
    for &t0 in &cands {
        for &t in &[t0, t0 + std::f64::consts::PI] {
            let (ux, uy) = (t.cos(), t.sin());
            if (xj - xi) * ux + (yj - yi) * uy < 0.0 { continue; }
            let (mut ihi, mut jlo) = (f64::NEG_INFINITY, f64::INFINITY);
            for k in 0..4 {
                let (ox, oy) = corner_off(ti.cos(), ti.sin(), k);
                ihi = ihi.max(ux * (xi + ox - mx) + uy * (yi + oy - my));
                let (ox, oy) = corner_off(tj.cos(), tj.sin(), k);
                jlo = jlo.min(ux * (xj + ox - mx) + uy * (yj + oy - my));
            }
            if jlo - ihi > bg { bg = jlo - ihi; bphi = t; bd = (jlo + ihi) / 2.0; }
        }
    }
    (bphi, bd)
}

// ---------------------------------------------------------------- problem
#[derive(Clone)]
struct Pair { i: usize, j: usize, phi: f64, d: f64, lam: [f64; 8], face: usize }

struct Prob {
    n: usize,
    pairs: Vec<Pair>,
    wlam: Vec<f64>, // 16 per square: corner k, wall w (x>=0, x<=s, y>=0, y<=s)
    mu: f64,
    fix_s: bool, // fixed side: pure overlap penalty, no s objective, d/ds = 0
    b: Vec<f64>, // soft squares (pre-stage): per-square shape variable, a_i = eps tanh(b_i); empty = rigid
}

#[inline]
fn term(g: f64, lam: f64, mu: f64) -> (f64, f64) {
    // lam == 0 (inactive constraints, the common case): the general formulas without the two divisions, bit-identical
    // (t = g + 0.0 compares and squares like g; x - 0.0 = x; the inactive value is (-0.0 * 0.0) / (2 mu) = -0.0)
    if lam == 0.0 { return if g > 0.0 { (0.5 * mu * g * g, mu * g) } else { (-0.0, 0.0) }; }
    let t = g + lam / mu;
    if t > 0.0 { (0.5 * mu * t * t - lam * lam / (2.0 * mu), mu * t) } else { (-lam * lam / (2.0 * mu), 0.0) }
}

impl Prob {
    // variable layout: 3n (x, y, theta), then s, then 2 per pair (phi, d)
    fn pack(&self, x: &[f64], s: f64) -> Vec<f64> {
        let mut v = x.to_vec();
        v.push(s);
        for p in &self.pairs { v.push(p.phi); v.push(p.d); }
        v.extend_from_slice(&self.b);
        v
    }
    fn unpack(&mut self, v: &[f64]) {
        let o = 3 * self.n + 1;
        for (k, p) in self.pairs.iter_mut().enumerate() { p.phi = v[o + 2 * k]; p.d = v[o + 2 * k + 1]; }
        let ob = o + 2 * self.pairs.len();
        let nb = self.b.len();
        self.b.copy_from_slice(&v[ob..ob + nb]);
    }
    /// Per-square (w, h, da/db, index of b in v or usize::MAX).
    fn shapes(&self, v: &[f64]) -> Vec<(f64, f64, f64, usize)> {
        let n = self.n;
        let e = soft_eps();
        if self.b.len() != n || e == 0.0 { return vec![(1.0, 1.0, 0.0, usize::MAX); n]; }
        let ob = 3 * n + 1 + 2 * self.pairs.len();
        (0..n).map(|i| { let t = v[ob + i].tanh(); let a = e * t; ((0.5 * a).exp(), (-0.5 * a).exp(), e * (1.0 - t * t), ob + i) }).collect()
    }

    /// Augmented Lagrangian value and gradient.
    fn eval(&self, v: &[f64], g: &mut [f64]) -> f64 {
        let n = self.n;
        let mu = self.mu;
        for q in g.iter_mut() { *q = 0.0; }
        let s = v[3 * n];
        let mut f = if self.fix_s { 0.0 } else { s };
        g[3 * n] = if self.fix_s { 0.0 } else { 1.0 };
        let mut cs = vec![(0.0, 0.0); n];
        let fsc = fast_sc();
        let np = self.pairs.len();
        let (mut pso, mut pco) = (Vec::new(), Vec::new());
        if fsc {
            // vectorised sincos of every angle in two batches (<= 2 ulp vs libm), default since 10-13 (PACKER "Q11"); FQ_FASTSC=0 = libm
            let th: Vec<f64> = (0..n).map(|i| v[3 * i + 2]).collect();
            let (mut so, mut co) = (vec![0.0; n], vec![0.0; n]);
            sincos_batch(&th, &mut so, &mut co);
            for i in 0..n { cs[i] = (co[i], so[i]); }
            let o = 3 * n + 1;
            let ph: Vec<f64> = (0..np).map(|pk| v[o + 2 * pk]).collect();
            pso = vec![0.0; np]; pco = vec![0.0; np];
            sincos_batch(&ph, &mut pso, &mut pco);
        } else {
            for i in 0..n { let t = v[3 * i + 2]; cs[i] = (t.cos(), t.sin()); }
        }
        let sh = self.shapes(v);
        // container / rounding knobs read once per evaluation (same values as per corner; perf 10-13, bit-identical)
        let (r, wx, wy) = (rnd(), s + rdx(), s + rdy());
        // walls.  A square whose circumscribed circle clears all four walls with zero multipliers contributes only -0.0
        // to f and nothing to g: skip its 16 terms, bit-identical (FQ_WALLSKIP=0 restores the full loop)
        let wskip = self.b.is_empty() && wall_skip();
        let wm = H * SQRT2 * (1.0 + 1e-12);
        for i in 0..n {
            let (c, sn) = cs[i];
            let (x, y) = (v[3 * i], v[3 * i + 1]);
            if wskip && x - wm > r && x + wm < wx - r && y - wm > r && y + wm < wy - r && self.wlam[16 * i..16 * i + 16].iter().all(|&l| l == 0.0) { continue; }
            let (wi, hi, dab, bi) = sh[i];
            for k in 0..4 {
                let (ox, oy, oxa, oya) = corner_wh(c, sn, k, wi, hi);
                let (px, py) = (x + ox, y + oy);
                let gs = [-px + r, px - wx + r, -py + r, py - wy + r];
                for w in 0..4 {
                    let (fv, dg) = term(gs[w], self.wlam[16 * i + 4 * k + w], mu);
                    f += fv;
                    if dg != 0.0 {
                        // dpx/dtheta = -oy, dpy/dtheta = ox
                        match w {
                            0 => { g[3 * i] -= dg; g[3 * i + 2] += dg * oy; }
                            1 => { g[3 * i] += dg; g[3 * i + 2] -= dg * oy; g[3 * n] -= dg; }
                            2 => { g[3 * i + 1] -= dg; g[3 * i + 2] -= dg * ox; }
                            _ => { g[3 * i + 1] += dg; g[3 * i + 2] += dg * ox; g[3 * n] -= dg; }
                        }
                        if bi != usize::MAX {
                            let gp = [-oxa, oxa, -oya, oya][w];
                            g[bi] += dg * gp * dab;
                        }
                    }
                }
            }
        }
        // pairs
        let o = 3 * n + 1;
        let rr = rnd();
        for (pk, p) in self.pairs.iter().enumerate() {
            let (i, j) = (p.i, p.j);
            let phi = v[o + 2 * pk];
            let d = v[o + 2 * pk + 1];
            let (ux, uy) = if fsc { (pco[pk], pso[pk]) } else { (phi.cos(), phi.sin()) };
            let (hx, hy) = ((v[3 * i] - v[3 * j]) / 2.0, (v[3 * i + 1] - v[3 * j + 1]) / 2.0);
            let (mut gx, mut gti, mut gtj, mut gphi, mut gd) = (0.0, 0.0, 0.0, 0.0, 0.0);
            let (ci, si) = cs[i];
            let (cj, sj) = cs[j];
            let (wi, hi, dai, bi) = sh[i];
            let (wj, hj, daj, bj) = sh[j];
            for k in 0..4 {
                // corner of i: r = h + o_i;  g = u.r - d
                let (ox, oy, oxa, oya) = corner_wh(ci, si, k, wi, hi);
                let (rx, ry) = (hx + ox, hy + oy);
                let (fv, dg) = term(ux * rx + uy * ry - d + rr, p.lam[k], mu);
                f += fv;
                if dg != 0.0 {
                    gx += dg;
                    gti += dg * (-ux * oy + uy * ox);
                    gphi += dg * (-uy * rx + ux * ry);
                    gd -= dg;
                    if bi != usize::MAX { g[bi] += dg * (ux * oxa + uy * oya) * dai; }
                }
                // corner of j: r' = -h + o_j;  g = d - u.r'
                let (ox, oy, oxa, oya) = corner_wh(cj, sj, k, wj, hj);
                let (rx, ry) = (-hx + ox, -hy + oy);
                let (fv, dg2) = term(d - (ux * rx + uy * ry) + rr, p.lam[4 + k], mu);
                f += fv;
                if dg2 != 0.0 {
                    gx += dg2;
                    gtj -= dg2 * (-ux * oy + uy * ox);
                    gphi -= dg2 * (-uy * rx + ux * ry);
                    gd += dg2;
                    if bj != usize::MAX { g[bj] -= dg2 * (ux * oxa + uy * oya) * daj; }
                }
            }
            // both corner kinds: dg/dXi = u/2, dg/dXj = -u/2 (gx holds the summed dg)
            if gx != 0.0 {
                g[3 * i] += 0.5 * gx * ux; g[3 * i + 1] += 0.5 * gx * uy;
                g[3 * j] -= 0.5 * gx * ux; g[3 * j + 1] -= 0.5 * gx * uy;
            }
            g[3 * i + 2] += gti;
            g[3 * j + 2] += gtj;
            g[o + 2 * pk] += gphi;
            g[o + 2 * pk + 1] += gd;
        }
        if self.fix_s { g[3 * n] = 0.0; }
        let fz = frozen_mask(n);
        if !fz.is_empty() {
            for i in 0..n { if fz[i] { g[3 * i] = 0.0; g[3 * i + 1] = 0.0; g[3 * i + 2] = 0.0; if sh[i].3 != usize::MAX { g[sh[i].3] = 0.0; } } }
        }
        f
    }

    /// Max constraint violation, and multiplier update lam <- max(0, lam + mu g).
    fn viol_update(&mut self, v: &[f64], update: bool) -> f64 {
        let n = self.n;
        let mu = self.mu;
        let s = v[3 * n];
        let mut mv: f64 = 0.0;
        let sh = self.shapes(v);
        for i in 0..n {
            let t = v[3 * i + 2];
            let (c, sn) = (t.cos(), t.sin());
            for k in 0..4 {
                let (ox, oy, _, _) = corner_wh(c, sn, k, sh[i].0, sh[i].1);
                let (px, py) = (v[3 * i] + ox, v[3 * i + 1] + oy);
                let (r, wx, wy) = (rnd(), s + rdx(), s + rdy());
                let gs = [-px + r, px - wx + r, -py + r, py - wy + r];
                for w in 0..4 {
                    mv = mv.max(gs[w]);
                    if update { let l = &mut self.wlam[16 * i + 4 * k + w]; *l = (*l + mu * gs[w]).max(0.0); }
                }
            }
        }
        let o = 3 * n + 1;
        let rr = rnd();
        for (pk, p) in self.pairs.iter_mut().enumerate() {
            let (i, j) = (p.i, p.j);
            let (phi, d) = (v[o + 2 * pk], v[o + 2 * pk + 1]);
            let (ux, uy) = (phi.cos(), phi.sin());
            let (hx, hy) = ((v[3 * i] - v[3 * j]) / 2.0, (v[3 * i + 1] - v[3 * j + 1]) / 2.0);
            let (ti, tj) = (v[3 * i + 2], v[3 * j + 2]);
            for k in 0..4 {
                let (ox, oy, _, _) = corner_wh(ti.cos(), ti.sin(), k, sh[i].0, sh[i].1);
                let gi = ux * (hx + ox) + uy * (hy + oy) - d + rr;
                let (ox, oy, _, _) = corner_wh(tj.cos(), tj.sin(), k, sh[j].0, sh[j].1);
                let gj = d - (ux * (-hx + ox) + uy * (-hy + oy)) + rr;
                mv = mv.max(gi).max(gj);
                if update {
                    p.lam[k] = (p.lam[k] + mu * gi).max(0.0);
                    p.lam[4 + k] = (p.lam[4 + k] + mu * gj).max(0.0);
                }
            }
        }
        mv
    }

    /// Rebuild the pair list at cutoff; keep separators and multipliers of surviving pairs.
    fn rebuild(&mut self, x: &[f64], cut: f64) {
        let n = self.n;
        let mut old = std::collections::HashMap::new();
        for p in self.pairs.drain(..) { old.insert((p.i, p.j), p); }
        let c2 = cut * cut;
        let fz = frozen_mask(n);
        for i in 0..n {
            for j in i + 1..n {
                if !fz.is_empty() && fz[i] && fz[j] { continue; }
                let (dx, dy) = (x[3 * j] - x[3 * i], x[3 * j + 1] - x[3 * i + 1]);
                if dx * dx + dy * dy >= c2 { continue; }
                if let Some(p) = old.remove(&(i, j)) { self.pairs.push(p); continue; }
                let (phi, d) = init_sep(x[3 * i], x[3 * i + 1], x[3 * i + 2], x[3 * j], x[3 * j + 1], x[3 * j + 2]);
                self.pairs.push(Pair { i, j, phi, d, lam: [0.0; 8], face: 99 });
            }
        }
    }
}


// ---------------------------------------------------------------- SLP polish (trust region, lifted formulation)
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Spec { Wall(usize, usize, usize), PI(usize, usize), PJ(usize, usize) }
struct Con { g: f64, nz: Vec<(usize, f64)>, sp: Spec }

impl Prob {
    /// Constraints with g > -delta and their sparse gradients (variable layout as pack()).
    fn near(&self, v: &[f64], delta: f64) -> Vec<Con> {
        let n = self.n;
        let si = 3 * n;
        let s = v[si];
        let mut out = Vec::new();
        for i in 0..n {
            let t = v[3 * i + 2];
            let (c, sn) = (t.cos(), t.sin());
            for k in 0..4 {
                let (ox, oy) = corner_off(c, sn, k);
                let (px, py) = (v[3 * i] + ox, v[3 * i + 1] + oy);
                if -px > -delta { out.push(Con { g: -px, nz: vec![(3 * i, -1.0), (3 * i + 2, oy)], sp: Spec::Wall(i, k, 0) }); }
                if px - s - rdx() > -delta { out.push(Con { g: px - s - rdx(), nz: vec![(3 * i, 1.0), (3 * i + 2, -oy), (si, -1.0)], sp: Spec::Wall(i, k, 1) }); }
                if -py > -delta { out.push(Con { g: -py, nz: vec![(3 * i + 1, -1.0), (3 * i + 2, -ox)], sp: Spec::Wall(i, k, 2) }); }
                if py - s - rdy() > -delta { out.push(Con { g: py - s - rdy(), nz: vec![(3 * i + 1, 1.0), (3 * i + 2, ox), (si, -1.0)], sp: Spec::Wall(i, k, 3) }); }
            }
        }
        let o = 3 * n + 1;
        for (pk, p) in self.pairs.iter().enumerate() {
            let (i, j) = (p.i, p.j);
            let (phi, d) = (v[o + 2 * pk], v[o + 2 * pk + 1]);
            let (ux, uy) = (phi.cos(), phi.sin());
            let (hx, hy) = ((v[3 * i] - v[3 * j]) / 2.0, (v[3 * i + 1] - v[3 * j + 1]) / 2.0);
            let (ti, tj) = (v[3 * i + 2], v[3 * j + 2]);
            let xs = [(3 * i, 0.5 * ux), (3 * i + 1, 0.5 * uy), (3 * j, -0.5 * ux), (3 * j + 1, -0.5 * uy)];
            // separator attached to a face: phi moves with the owner's theta
            let phicol = match p.face { 0 | 1 => 3 * i + 2, 2 | 3 => 3 * j + 2, _ => o + 2 * pk };
            for k in 0..4 {
                let (ox, oy) = corner_off(ti.cos(), ti.sin(), k);
                let (rx, ry) = (hx + ox, hy + oy);
                let g = ux * rx + uy * ry - d;
                if g > -delta {
                    let mut nz = xs.to_vec();
                    nz.push((3 * i + 2, -ux * oy + uy * ox));
                    nz.push((phicol, -uy * rx + ux * ry));
                    nz.push((o + 2 * pk + 1, -1.0));
                    out.push(Con { g, nz, sp: Spec::PI(pk, k) });
                }
                let (ox, oy) = corner_off(tj.cos(), tj.sin(), k);
                let (rx, ry) = (-hx + ox, -hy + oy);
                let g = d - (ux * rx + uy * ry);
                if g > -delta {
                    let mut nz = xs.to_vec();
                    nz.push((3 * j + 2, ux * oy - uy * ox));
                    nz.push((phicol, uy * rx - ux * ry));
                    nz.push((o + 2 * pk + 1, 1.0));
                    out.push(Con { g, nz, sp: Spec::PJ(pk, k) });
                }
            }
        }
        let fz = frozen_mask(n);
        if !fz.is_empty() {
            for c in out.iter_mut() { c.nz.retain(|&(k, _)| !(k < si && fz[k / 3])); }
            out.retain(|c| !c.nz.is_empty());
        }
        out
    }
}

impl Prob {
    /// Apply the face tie to a trial point: phi of a face-attached separator moves with its owner's theta.
    fn tie(&self, v0: &[f64], v: &mut [f64]) {
        let o = 3 * self.n + 1;
        for (pk, p) in self.pairs.iter().enumerate() {
            let own = match p.face { 0 | 1 => p.i, 2 | 3 => p.j, _ => continue };
            v[o + 2 * pk] = v0[o + 2 * pk] + (v[3 * own + 2] - v0[3 * own + 2]);
        }
    }
    fn g_at(&self, v: &[f64], sp: Spec) -> f64 {
        let n = self.n;
        match sp {
            Spec::Wall(i, k, w) => {
                let t = v[3 * i + 2];
                let (ox, oy) = corner_off(t.cos(), t.sin(), k);
                let (px, py, s) = (v[3 * i] + ox, v[3 * i + 1] + oy, v[3 * n]);
                [-px, px - s - rdx(), -py, py - s - rdy()][w]
            }
            Spec::PI(pk, k) | Spec::PJ(pk, k) => {
                let p = &self.pairs[pk];
                let (i, j) = (p.i, p.j);
                let o = 3 * n + 1;
                let (phi, d) = (v[o + 2 * pk], v[o + 2 * pk + 1]);
                let (ux, uy) = (phi.cos(), phi.sin());
                let (hx, hy) = ((v[3 * i] - v[3 * j]) / 2.0, (v[3 * i + 1] - v[3 * j + 1]) / 2.0);
                if let Spec::PI(..) = sp {
                    let t = v[3 * i + 2];
                    let (ox, oy) = corner_off(t.cos(), t.sin(), k);
                    ux * (hx + ox) + uy * (hy + oy) - d
                } else {
                    let t = v[3 * j + 2];
                    let (ox, oy) = corner_off(t.cos(), t.sin(), k);
                    d - (ux * (-hx + ox) + uy * (-hy + oy))
                }
            }
        }
    }
}

/// One LP: min w_s + kappa tau  s.t.  grad g . w - tau <= -g/R (near constraints), |w| <= 1, tau >= 0.
/// Only columns that occur in some row (and s) are LP variables; the rest of w is 0.  Returns (w, tau, row duals).
fn slp_lp(nv: usize, si: usize, cons: &[Con], r: f64, kappa: f64, hq: &[(usize, usize, f64)], kp: &Prob) -> Option<(Vec<f64>, f64, Vec<f64>)> {
    use clarabel::algebra::*;
    use clarabel::solver::*;
    let mut col = vec![usize::MAX; nv];
    let mut used = Vec::new();
    let mut take = |k: usize, col: &mut Vec<usize>| { if col[k] == usize::MAX { col[k] = used.len(); used.push(k); } };
    take(si, &mut col);
    for c in cons { for &(k, _) in &c.nz { take(k, &mut col); } }
    for &(a, b, _) in hq { take(a, &mut col); take(b, &mut col); }
    let nu = used.len();
    let nx = nu + 1; // + tau
    let (mut ii, mut jj, mut vv) = (Vec::new(), Vec::new(), Vec::new());
    let mut b = Vec::new();
    let mut row = 0;
    for c in cons {
        for &(k, a) in &c.nz { ii.push(row); jj.push(col[k]); vv.push(a); }
        ii.push(row); jj.push(nu); vv.push(-1.0);
        b.push(-c.g / r);
        row += 1;
    }
    for k in 0..nu {
        ii.push(row); jj.push(k); vv.push(1.0); b.push(1.0); row += 1;
        ii.push(row); jj.push(k); vv.push(-1.0); b.push(1.0); row += 1;
    }
    ii.push(row); jj.push(nu); vv.push(-1.0); b.push(0.0); row += 1;
    // Own normal-equation IPM with native bounds (lpipm.rs), default since 10-13 (PACKER "Q7"); Clarabel as fallback and for SQP; FQ_LP_IPM=0 restores Clarabel
    static LP_IPM: OnceLock<bool> = OnceLock::new();
    let mut ipm_chk: Option<((Vec<f64>, f64, Vec<f64>), usize, f64, Vec<f64>)> = None;
    if hq.is_empty() && *LP_IPM.get_or_init(|| std::env::var("FQ_LP_IPM").map(|v| v != "0").unwrap_or(true)) {
        let t0 = Instant::now();
        let m = cons.len();
        let mut rows: Vec<Vec<(usize, f64)>> = Vec::with_capacity(m);
        for c in cons {
            let mut e: Vec<(usize, f64)> = Vec::with_capacity(c.nz.len() + 1);
            for &(k, a) in &c.nz { let j = col[k]; if let Some(q) = e.iter_mut().find(|q| q.0 == j) { q.1 += a; } else { e.push((j, a)); } }
            e.push((nu, -1.0));
            rows.push(e);
        }
        let bb = &b[..m];
        let mut cc = vec![0.0; nx]; cc[col[si]] = 1.0; cc[nu] = kappa;
        let mut lo = vec![-1.0; nx]; let mut hi = vec![1.0; nx]; lo[nu] = 0.0; hi[nu] = f64::INFINITY;
        let mut x0 = vec![0.0; nx];
        x0[nu] = bb.iter().fold(0.0f64, |a, &v| a.max(-v)) + 1.0;
        if let Some(sol) = lpipm::solve(nx, &rows, bb, &cc, &lo, &hi, &x0, 1e-8, 1e-8, 200) {
            if std::env::var("FQ_LPT").is_ok() { eprintln!("lp rows {row} cols {nx} iters {} sec {:.4} ipm tag {}", sol.iters, t0.elapsed().as_secs_f64(), LP_TAG.load(Ordering::Relaxed)); }
            let mut w = vec![0.0; nv];
            for (c, &k) in used.iter().enumerate() { w[k] = sol.x[c]; }
            let res = (w, sol.x[nu].max(0.0), sol.y);
            if std::env::var("FQ_IPM_CHK").is_err() { return Some(res); }
            ipm_chk = Some((res, sol.iters, t0.elapsed().as_secs_f64(), sol.x));
        } else {
        IPM_FAIL.fetch_add(1, Ordering::Relaxed);
        if std::env::var("FQ_LPT").is_ok() { eprintln!("ipm fallback"); }
        }
    }
    let a = CscMatrix::new_from_triplets(row, nx, ii, jj, vv);
    let p = if hq.is_empty() { CscMatrix::<f64>::zeros((nx, nx)) } else {
        // SQP: objective + (r/2) w^T H w in the scaled variables (H convexified by the caller), upper triangle
        let (mut pi, mut pj, mut pv) = (Vec::new(), Vec::new(), Vec::new());
        for &(a, b, val) in hq {
            let (ca, cb) = (col[a].min(col[b]), col[a].max(col[b]));
            pi.push(ca); pj.push(cb); pv.push(r * val);
        }
        CscMatrix::new_from_triplets(nx, nx, pi, pj, pv)
    };
    let mut q = vec![0.0; nx];
    q[col[si]] = 1.0;
    q[nu] = kappa;
    let cones = [NonnegativeConeT(row)];
    let mut settings = DefaultSettingsBuilder::default().verbose(false).max_iter(200).build().unwrap();
    // Clarabel iterative refinement: off by default since 10-13 (PACKER "Q0": polish ~1.55x, screen ~1.15-1.2x; not
    // bit-identical, outputs equivalent to a 1e-13 input re-roll at n = 110 and 237).  FQ_LP_IR=1 restores the old LPs.
    static LP_IR: OnceLock<bool> = OnceLock::new();
    settings.iterative_refinement_enable = *LP_IR.get_or_init(|| std::env::var("FQ_LP_IR").map(|v| v == "1").unwrap_or(false));
    let t0 = Instant::now();
    let mut solver = DefaultSolver::new(&p, &q, &a, &b, &cones, settings).ok()?;
    solver.solve();
    if std::env::var("FQ_LPT").is_ok() { eprintln!("lp rows {row} cols {nx} iters {} sec {:.4} tag {}", solver.info.iterations, t0.elapsed().as_secs_f64(), LP_TAG.load(Ordering::Relaxed)); }
    if let Some(dir) = DUMP_LP.get() { dump_lp(dir, kp, &used, cons, &b, kappa, r, row, nx, &solver, t0.elapsed().as_secs_f64()); }
    if let Some((res, it2, t2, x2)) = ipm_chk {
        let x = &solver.solution.x;
        let ob = |x: &[f64]| x[col[si]] + kappa * x[nu];
        let dw = (0..nx).fold(0.0f64, |a, c| a.max((x[c] - x2[c]).abs()));
        let z = &solver.solution.z[..cons.len()];
        let zn = z.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
        let dz = z.iter().zip(&res.2).fold(0.0f64, |a, (p, q)| a.max((p - q).abs()));
        eprintln!("ipmchk tag {} rows {} cl it {} {:.4}s obj {:.9e} | ipm it {it2} {t2:.4}s obj {:.9e} | dw {dw:.1e} dz/zmax {:.1e} {:?}",
            LP_TAG.load(Ordering::Relaxed), cons.len(), solver.info.iterations, t0.elapsed().as_secs_f64(), ob(x), ob(&x2), dz / zn.max(1e-300), solver.solution.status);
        return Some(res);
    }
    match solver.solution.status {
        SolverStatus::Solved | SolverStatus::AlmostSolved => {
            let x = &solver.solution.x;
            let mut w = vec![0.0; nv];
            for (c, &k) in used.iter().enumerate() { w[k] = x[c]; }
            Some((w, x[nu].max(0.0), solver.solution.z[..cons.len()].to_vec()))
        }
        _ => None,
    }
}

static IPM_FAIL: AtomicUsize = AtomicUsize::new(0);
static FLIP_LPS: AtomicUsize = AtomicUsize::new(0);
static SOC_STATS: [AtomicUsize; 2] = [AtomicUsize::new(0), AtomicUsize::new(0)]; // [active-set used, fell back]

/// Second-order correction on the step LP's active set (PACKER "Q2").  The step LP (scaled: grad g . w - tau <= -g / R,
/// |w| <= 1, tau >= 0) was solved at (w, tau); `cons` now carry the shifted g, `g_old` the step LP's.  Rows with slack
/// < 1e-7 stay tight at the new rhs, box-active columns (|w| > 1 - 1e-7) and tau (if 0) stay fixed; the min-norm change
/// of the remaining columns comes from one quasidefinite solve [I A^T; A -delta I].  None if the result violates any row
/// or bound (the caller then solves the SOC LP).
fn soc_active(nv: usize, _si: usize, cons: &[Con], g_old: &[f64], r: f64, w: &[f64], tau: f64) -> Option<Vec<f64>> {
    use clarabel::algebra::CscMatrix;
    use clarabel::qdldl::{QDLDLFactorisation, QDLDLSettingsBuilder};
    let dbg = std::env::var("FQ_SOC_DBG").is_ok();
    let fail = |s: &str| { SOC_STATS[1].fetch_add(1, Ordering::Relaxed); if dbg { eprintln!("soc_as fallback: {s}"); } None };
    let tau_free = tau > 1e-9;
    let lhs = |c: &Con, w: &[f64], t: f64| -> f64 { c.nz.iter().map(|&(k, a)| a * w[k]).sum::<f64>() - t };
    let mut in_act = vec![false; cons.len()];
    for i in 0..cons.len() { in_act[i] = -g_old[i] / r - lhs(&cons[i], w, tau) < 1e-7; }
    let mut fixed = vec![false; nv];          // columns held at their step value (box-active) or clamped to a bound
    let mut wb = w.to_vec();                  // base point: step w with clamped columns moved to their bound
    for k in 0..nv { if w[k].abs() >= 1.0 - 1e-7 { fixed[k] = true; } }
    // FQ_SOC_ROUNDS (PACKER "Q12"): active-set repair rounds before falling back to the SOC LP (default 4)
    static ROUNDS: OnceLock<usize> = OnceLock::new();
    let rounds = *ROUNDS.get_or_init(|| std::env::var("FQ_SOC_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(4));
    for round in 0..rounds {
        let act: Vec<usize> = (0..cons.len()).filter(|&i| in_act[i]).collect();
        if act.is_empty() { return fail("no active rows"); }
        let mut col = vec![usize::MAX; nv];
        let mut free: Vec<usize> = Vec::new();
        for &i in &act { for &(k, _) in &cons[i].nz { if col[k] == usize::MAX && !fixed[k] { col[k] = free.len(); free.push(k); } } }
        let nf = free.len() + tau_free as usize;
        let m = act.len();
        let (mut ii, mut jj, mut vv) = (Vec::new(), Vec::new(), Vec::new());
        for c in 0..nf { ii.push(c); jj.push(c); vv.push(1.0); }
        let mut rhs = vec![0.0; nf + m];
        for (q, &i) in act.iter().enumerate() {
            let cc = nf + q;
            let mut ent: Vec<(usize, f64)> = Vec::new();
            for &(k, a) in &cons[i].nz {
                if col[k] != usize::MAX { if let Some(e) = ent.iter_mut().find(|e| e.0 == col[k]) { e.1 += a; } else { ent.push((col[k], a)); } }
            }
            if tau_free { ent.push((nf - 1, -1.0)); }
            for (c, a) in ent { ii.push(c); jj.push(cc); vv.push(a); }
            ii.push(cc); jj.push(cc); vv.push(-1e-10);
            rhs[cc] = -cons[i].g / r - lhs(&cons[i], &wb, tau);
        }
        let kkt = CscMatrix::new_from_triplets(nf + m, nf + m, ii, jj, vv);
        let mut signs = vec![1i8; nf]; signs.extend(std::iter::repeat(-1i8).take(m));
        let opts = QDLDLSettingsBuilder::default().Dsigns(signs).build().ok()?;
        let mut f = match QDLDLFactorisation::new(&kkt, Some(opts)) { Ok(f) => f, Err(_) => return fail("factor") };
        f.solve(&mut rhs);
        let mut w2 = wb.clone();
        for (c, &k) in free.iter().enumerate() { w2[k] += rhs[c]; }
        let tau2 = if tau_free { tau + rhs[nf - 1] } else { tau };
        if !w2.iter().all(|x| x.is_finite()) || tau2 < -1e-12 { return fail("nan / tau"); }
        let mut ok = true;
        for k in 0..nv { if w2[k].abs() > 1.0 + 1e-9 { ok = false; fixed[k] = true; wb[k] = w2[k].signum(); } }
        for (i, c) in cons.iter().enumerate() { if lhs(c, &w2, tau2) > -c.g / r + 1e-8 { ok = false; in_act[i] = true; } }
        if ok { SOC_STATS[0].fetch_add(1, Ordering::Relaxed); if dbg { eprintln!("soc_as ok round {round}"); } return Some(w2); }
    }
    fail("rounds")
}

fn row_key(p: &Prob, v: &[f64], c: &Con) -> (u32, u32, u8, u8) {
    match c.sp {
        Spec::Wall(i, k, w) => (i as u32, 1_000_000 + w as u32, 99, k as u8),
        Spec::PI(pk, k) => { let q = &p.pairs[pk]; (q.i as u32, q.j as u32, canon_face(v, q), k as u8) }
        Spec::PJ(pk, k) => { let q = &p.pairs[pk]; (q.i as u32, q.j as u32, canon_face(v, q), 4 + k as u8) }
    }
}

/// Load-bearing contacts: rows with LP dual > 1e-6 max dual (interior point = maximal-support dual = force network).
fn load_keys(p: &Prob, v: &[f64], cons: &[Con], z: &[f64]) -> Vec<(u32, u32, u8, u8)> {
    let zm = z.iter().cloned().fold(0.0, f64::max);
    if std::env::var("FQ_ZHIST").is_ok() {
        let mut h = [0usize; 16];
        for &zz in z { let e = if zz <= 0.0 { 15 } else { ((-(zz / zm).log10()).floor() as usize).min(15) }; h[e] += 1; }
        eprintln!("zhist (decades below max) {:?}", h);
    }
    // contact level: drop the corner index (which corners of a face contact carry load is not basin-invariant)
    let mut out: Vec<_> = cons.iter().zip(z).filter(|(_, &zz)| zz > 1e-6 * zm).map(|(c, _)| { let k = row_key(p, v, c); (k.0, k.1, k.2, 0u8) }).collect();
    out.sort();
    out.dedup();
    out
}

/// Active contacts at v (g > -tol): keys (i, j or 1_000_000 + wall, face (99 = wall), corner (0-3 of i, 4-7 of j)).
fn contact_keys(p: &Prob, v: &[f64], tol: f64) -> Vec<(u32, u32, u8, u8)> {
    let mut out: Vec<(u32, u32, u8, u8)> = p.near(v, tol).iter().map(|c| match c.sp {
        Spec::Wall(i, k, w) => (i as u32, 1_000_000 + w as u32, 99, k as u8),
        Spec::PI(pk, k) => { let q = &p.pairs[pk]; (q.i as u32, q.j as u32, canon_face(v, q), k as u8) }
        Spec::PJ(pk, k) => { let q = &p.pairs[pk]; (q.i as u32, q.j as u32, canon_face(v, q), 4 + k as u8) }
    }).collect();
    out.sort();
    out
}

/// Face label independent of tie-breaks: the lowest face index parallel to the chosen face; 98 when two non-parallel faces
/// are both within 1e-7 of contact (corner-corner: either branch).
fn canon_face(v: &[f64], q: &Pair) -> u8 {
    let (i, j) = (q.i, q.j);
    let f = face_seps(v[3 * i], v[3 * i + 1], v[3 * i + 2], v[3 * j], v[3 * j + 1], v[3 * j + 2]);
    let par = |a: usize, b: usize| {
        let dd = (f[a].1 - f[b].1).rem_euclid(2.0 * std::f64::consts::PI);
        dd.min(2.0 * std::f64::consts::PI - dd) < 1e-6
    };
    let c = q.face.min(3);
    let best = (0..4).map(|k| f[k].0).fold(f64::NEG_INFINITY, f64::max);
    if (0..4).any(|k| !par(k, c) && f[k].0 > best - 1e-7 && f[k].0 > -1e-7) && f[c].0 > -1e-7 { return 98; }
    (0..4).find(|&k| par(k, c)).unwrap() as u8
}

fn fingerprint(keys: &[(u32, u32, u8, u8)]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &(a, b, c, d) in keys {
        for x in [a as u64, b as u64, c as u64, d as u64] { h ^= x; h = h.wrapping_mul(0x100000001b3); }
    }
    h
}

/// Identification statistics of an accepted step from v0 to v1: (max corner displacement over squares, smallest slack of
/// near-but-inactive constraints (g in (-0.05, -tol]), number of active, symmetric difference with `prev` active set).
fn ident(p: &Prob, v0: &[f64], v1: &[f64], prev: &mut Vec<(u32, u32, u8, u8)>, tol: f64) -> (f64, f64, usize, usize) {
    let n = p.n;
    let mut dm: f64 = 0.0;
    for i in 0..n {
        let d = (v1[3 * i] - v0[3 * i]).hypot(v1[3 * i + 1] - v0[3 * i + 1]) + H * SQRT2 * (v1[3 * i + 2] - v0[3 * i + 2]).abs();
        dm = dm.max(d);
    }
    let slack = p.near(v1, 0.05).iter().filter(|c| c.g <= -tol).map(|c| -c.g).fold(f64::INFINITY, f64::min);
    let act = contact_keys(p, v1, tol);
    let a: std::collections::HashSet<_> = act.iter().collect();
    let b: std::collections::HashSet<_> = prev.iter().collect();
    let diff = a.symmetric_difference(&b).count();
    let na = act.len();
    *prev = act;
    (dm, slack, na, diff)
}

/// Hessian of sum_k z_k g_k at v (rows from `cons`, multipliers `z`), with face-tied separators (phi -> owner theta),
/// convexified by diagonal dominance (H_aa >= sum_b |H_ab|).  Returns the upper triangle (a <= b) in original indices.
fn hess_lagr(p: &Prob, v: &[f64], cons: &[Con], z: &[f64]) -> Vec<(usize, usize, f64)> { hess_lagr2(p, v, cons, z, true) }

/// Lagrangian Hessian sum_k z_k grad^2 g_k (face tie applied); `convex`: diagonally dominant convexification (SQP),
/// else the raw upper triangle (Newton finish).
fn hess_lagr2(p: &Prob, v: &[f64], cons: &[Con], z: &[f64], convex: bool) -> Vec<(usize, usize, f64)> {
    let mut h = hess_lagr3(p, v, cons, z, convex);
    let fz = frozen_mask(p.n);
    if !fz.is_empty() { let si = 3 * p.n; h.retain(|&(a, b, _)| !(a < si && fz[a / 3]) && !(b < si && fz[b / 3])); }
    h
}

fn hess_lagr3(p: &Prob, v: &[f64], cons: &[Con], z: &[f64], convex: bool) -> Vec<(usize, usize, f64)> {
    use std::collections::HashMap;
    let n = p.n;
    let o = 3 * n + 1;
    let zm = z.iter().cloned().fold(0.0, f64::max);
    let mut h: HashMap<(usize, usize), f64> = HashMap::new();
    let mut add = |a: usize, b: usize, val: f64, h: &mut HashMap<(usize, usize), f64>| {
        if a == b { *h.entry((a, a)).or_insert(0.0) += val; }
        else { *h.entry((a, b)).or_insert(0.0) += val; *h.entry((b, a)).or_insert(0.0) += val; }
    };
    for (c, &zk) in cons.iter().zip(z) {
        if !(zk > 1e-9 * zm) || (!convex && zk <= 0.0) { continue; }
        match c.sp {
            Spec::Wall(i, k, w) => {
                let th = v[3 * i + 2];
                let (ox, oy) = corner_off(th.cos(), th.sin(), k);
                let d2 = [ox, -ox, oy, -oy][w];
                add(3 * i + 2, 3 * i + 2, zk * d2, &mut h);
            }
            Spec::PI(pk, k) | Spec::PJ(pk, k) => {
                let pr = &p.pairs[pk];
                let (i, j) = (pr.i, pr.j);
                let own = match pr.face { 0 | 1 => i, _ => j };
                let phc = 3 * own + 2;
                let map = |a: usize| if a == usize::MAX { phc } else { a };
                const PHI: usize = usize::MAX;
                let phi = v[o + 2 * pk];
                let (ux, uy) = (phi.cos(), phi.sin());
                let (dx, dy) = (-uy, ux); // u'
                let (hx, hy) = ((v[3 * i] - v[3 * j]) / 2.0, (v[3 * i + 1] - v[3 * j + 1]) / 2.0);
                let mut e: Vec<(usize, usize, f64)> = Vec::new();
                let isi = matches!(c.sp, Spec::PI(..));
                let (sq_i, sgn) = if isi { (i, 1.0) } else { (j, -1.0) };
                let th = v[3 * sq_i + 2];
                let (ox, oy) = corner_off(th.cos(), th.sin(), k);
                let (rx, ry) = if isi { (hx + ox, hy + oy) } else { (-hx + ox, -hy + oy) };
                let ts = 3 * sq_i + 2;
                e.push((ts, ts, -sgn * (ux * ox + uy * oy)));            // theta theta
                e.push((PHI, PHI, -sgn * (ux * rx + uy * ry)));          // phi phi
                e.push((PHI, ts, sgn * (dx * (-oy) + dy * ox)));         // phi theta
                for (col, val) in [(3 * i, dx / 2.0), (3 * i + 1, dy / 2.0), (3 * j, -dx / 2.0), (3 * j + 1, -dy / 2.0)] {
                    e.push((PHI, col, val));
                }
                for (a, b, val) in e {
                    let (ma, mb) = (map(a), map(b));
                    if a != b && ma == mb { add(ma, ma, 2.0 * zk * val, &mut h); } else { add(ma, mb, zk * val, &mut h); }
                }
            }
        }
    }
    if !convex {
        return h.iter().filter(|(&(a, b), &val)| a <= b && val != 0.0).map(|(&(a, b), &val)| (a, b, val)).collect();
    }
    // convexify: diagonal dominance
    let mut off: HashMap<usize, f64> = HashMap::new();
    for (&(a, b), &val) in &h { if a != b { *off.entry(a).or_insert(0.0) += val.abs(); } }
    let mut out = Vec::new();
    let mut diag: HashMap<usize, f64> = HashMap::new();
    for (&(a, b), &val) in &h { if a == b { diag.insert(a, val); } }
    for (&a, &s) in &off { let d = diag.entry(a).or_insert(0.0); if *d < s { *d = s; } }
    for (&a, &d) in &diag { if d > 0.0 { out.push((a, a, d)); } }
    for (&(a, b), &val) in &h { if a < b && val != 0.0 { out.push((a, b, val)); } }
    out
}

struct POpt { sqp: bool, rmax: f64, r0: f64, rmin: f64, maxit: usize, cc_tol: f64, flip_top: usize, ident: bool, verbose: bool, stag_w: usize, stag_tol: f64, finish: bool, tr_grow: f64, tr_shrink: f64 }

/// Snap each pair's separator to a face branch: `choice[pk]` = Some(face) forces that face while it stays ambiguous,
/// None = the face with the largest gap.  Returns the ambiguous pairs (>= 2 faces within cc_tol of contact and of the
/// best) with their candidate faces; forced choices that are no longer ambiguous are cleared.
fn dump_lp(dir: &str, p: &Prob, used: &[usize], cons: &[Con], b: &[f64], kappa: f64, r: f64, nrow: usize, nx: usize,
           solver: &clarabel::solver::DefaultSolver<f64>, sec: f64) {
    use std::io::Write;
    let n = p.n;
    let o = 3 * n + 1;
    let ck = |k: usize| -> String {
        if k < 3 * n { format!("{}{}", ["x", "y", "t"][k % 3], k / 3) } else if k == 3 * n { "s".into() } else {
            let q = &p.pairs[(k - o) / 2];
            format!("{}{}_{}", if (k - o) % 2 == 0 { "f" } else { "d" }, q.i, q.j)
        }
    };
    let rk = |c: &Con| -> String {
        match c.sp {
            Spec::Wall(i, k, w) => format!("w{i}_{k}_{w}"),
            Spec::PI(pk, k) => format!("a{}_{}_{k}", p.pairs[pk].i, p.pairs[pk].j),
            Spec::PJ(pk, k) => format!("b{}_{}_{k}", p.pairs[pk].i, p.pairs[pk].j),
        }
    };
    let mut col = std::collections::HashMap::new();
    for (c, &k) in used.iter().enumerate() { col.insert(k, c); }
    let mut s = String::with_capacity(64 * cons.len());
    s.push_str(&format!("{{\"n\":{n},\"tag\":{},\"it\":{},\"r\":{:e},\"kappa\":{:e},\"nrow_total\":{nrow},\"nx\":{nx},\"cols\":[",
        LP_TAG.load(Ordering::Relaxed), LP_IT.load(Ordering::Relaxed), r, kappa));
    for (c, &k) in used.iter().enumerate() { if c > 0 { s.push(','); } s.push_str(&format!("\"{}\"", ck(k))); }
    s.push_str("],\"rows\":[");
    for (ri, c) in cons.iter().enumerate() {
        if ri > 0 { s.push(','); }
        s.push_str(&format!("[\"{}\",{:e},[", rk(c), b[ri]));
        for (m, &(k, a)) in c.nz.iter().enumerate() { if m > 0 { s.push(','); } s.push_str(&format!("[{},{:e}]", col[&k], a)); }
        s.push_str("]]");
    }
    let x = &solver.solution.x;
    let obj = x[col[&(3 * n)]] + kappa * x[nx - 1];
    s.push_str(&format!("],\"clarabel\":{{\"status\":\"{:?}\",\"iters\":{},\"sec\":{:e},\"solve_time\":{:e},\"obj\":{:e}}}}}\n",
        solver.solution.status, solver.info.iterations, sec, solver.info.solve_time, obj));
    let path = format!("{dir}/lp-{}.jsonl", std::process::id());
    let mut f = fs::OpenOptions::new().create(true).append(true).open(path).expect("dump-lp");
    f.write_all(s.as_bytes()).unwrap();
}

fn snap(p: &mut Prob, x: &[f64], choice: &mut [Option<usize>], cc_tol: f64) -> Vec<(usize, Vec<usize>)> {
    let mut amb = Vec::new();
    for (pk, pr) in p.pairs.iter_mut().enumerate() {
        let (i, j) = (pr.i, pr.j);
        let f = face_seps(x[3 * i], x[3 * i + 1], x[3 * i + 2], x[3 * j], x[3 * j + 1], x[3 * j + 2]);
        let best = (0..4).max_by(|&a, &b| f[a].0.partial_cmp(&f[b].0).unwrap()).unwrap();
        // candidate branches: faces near contact and near the best, distinct in direction (parallel faces = one branch)
        let mut cand: Vec<usize> = vec![best];
        for q in 0..4 {
            if q == best || !(f[q].0 > f[best].0 - cc_tol && f[q].0 > -cc_tol) { continue; }
            let par = cand.iter().any(|&c| {
                let dd = (f[q].1 - f[c].1).rem_euclid(2.0 * std::f64::consts::PI);
                dd.min(2.0 * std::f64::consts::PI - dd) < 1e-6
            });
            if !par { cand.push(q); }
        }
        let mut c = best;
        if let Some(fc) = choice[pk] {
            if cand.contains(&fc) { c = fc; } else { choice[pk] = None; }
        }
        pr.phi = f[c].1;
        pr.d = f[c].2;
        pr.face = c;
        if cand.len() >= 2 {
            if std::env::var("FQ_DEBUG").is_ok() && amb.len() < 6 {
                eprintln!("amb pair {i} {j}: th {:.6} {:.6} faces {:?}", x[3*i+2].to_degrees(), x[3*j+2].to_degrees(), f.iter().map(|q| (q.0, q.1.to_degrees())).collect::<Vec<_>>());
            }
            amb.push((pk, cand));
        }
    }
    amb
}


// ---------------------------------------------------------------- Newton finish (10-08)
static FIN_LOG: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// Minimum-norm least-squares step for the rows (g, nz): solves [I J^T; J -dI][w; y] = [b1; b2] with QDLDL (sparse,
/// quasi-definite).  Returns (w over all nv columns, y).
fn kkt_solve(nv: usize, rows: &[(f64, Vec<(usize, f64)>)], b1: &[f64], b2: &[f64]) -> Option<(Vec<f64>, Vec<f64>)> { kkt_solve_h(nv, rows, None, b1, b2) }

/// As kkt_solve with top-left block H + 1e-10 I (H: upper-triangle triplets over the rows' columns) instead of I.
fn kkt_solve_h(nv: usize, rows: &[(f64, Vec<(usize, f64)>)], hh: Option<&[(usize, usize, f64)]>, b1: &[f64], b2: &[f64]) -> Option<(Vec<f64>, Vec<f64>)> {
    use clarabel::algebra::CscMatrix;
    use clarabel::qdldl::*;
    let mut col = vec![usize::MAX; nv];
    let mut used = Vec::new();
    for (_, nz) in rows { for &(k, _) in nz { if col[k] == usize::MAX { col[k] = used.len(); used.push(k); } } }
    let (nc, m) = (used.len(), rows.len());
    let dim = nc + m;
    let (mut cp, mut ri, mut vx) = (vec![0usize], Vec::new(), Vec::new());
    let mut aug = vec![0.0; nc];
    match hh {
        None => for j in 0..nc { ri.push(j); vx.push(1.0); cp.push(ri.len()); },
        Some(h) => {
            // augmented Lagrangian form: H + rho J^T J (positive definite when the reduced Hessian is), rhs + rho J^T b2
            const RHO: f64 = 1.0;
            let mut cols: Vec<Vec<(usize, f64)>> = vec![Vec::new(); nc];
            for &(a, b, val) in h {
                if col[a] == usize::MAX || col[b] == usize::MAX { continue; }
                let (ca, cb) = (col[a].min(col[b]), col[a].max(col[b]));
                cols[cb].push((ca, val));
            }
            for (r, (_, nz)) in rows.iter().enumerate() {
                let mut e: Vec<(usize, f64)> = nz.iter().map(|&(k, a)| (col[k], a)).collect();
                e.sort_by_key(|q| q.0);
                let mut acc: Vec<(usize, f64)> = Vec::new();
                for (c, a) in e { if let Some(l) = acc.last_mut() { if l.0 == c { l.1 += a; continue; } } acc.push((c, a)); }
                for x in 0..acc.len() { for y in x..acc.len() { cols[acc[y].0].push((acc[x].0, RHO * acc[x].1 * acc[y].1)); } }
                for &(c, a) in &acc { aug[c] += RHO * a * b2[r]; }
            }
            for j in 0..nc {
                let mut e = std::mem::take(&mut cols[j]);
                e.push((j, 1e-8));
                e.sort_by_key(|q| q.0);
                let mut acc: Vec<(usize, f64)> = Vec::new();
                for (c, a) in e { if let Some(l) = acc.last_mut() { if l.0 == c { l.1 += a; continue; } } acc.push((c, a)); }
                for (c, a) in acc { ri.push(c); vx.push(a); }
                cp.push(ri.len());
            }
        }
    }
    for (r, (_, nz)) in rows.iter().enumerate() {
        let mut e: Vec<(usize, f64)> = nz.iter().map(|&(k, a)| (col[k], a)).collect();
        e.sort_by_key(|q| q.0);
        let mut acc: Vec<(usize, f64)> = Vec::new();
        for (c, a) in e { if let Some(l) = acc.last_mut() { if l.0 == c { l.1 += a; continue; } } acc.push((c, a)); }
        for (c, a) in acc { ri.push(c); vx.push(a); }
        ri.push(nc + r); vx.push(-1e-12);
        cp.push(ri.len());
    }
    let k = CscMatrix { m: dim, n: dim, colptr: cp, rowval: ri, nzval: vx };
    let signs: Vec<i8> = (0..dim).map(|q| if q < nc { 1 } else { -1 }).collect();
    let opts = QDLDLSettingsBuilder::default().Dsigns(signs).build().ok()?;
    let mut f = match QDLDLFactorisation::new(&k, Some(opts)) { Ok(f) => f, Err(e) => { if std::env::var("FQ_FIN").is_ok() { eprintln!("  qdldl: {e:?}"); } return None; } };
    let mut rhs: Vec<f64> = used.iter().enumerate().map(|(q, &c)| b1[c] + aug[q]).chain(b2.iter().cloned()).collect();
    f.solve(&mut rhs);
    if rhs.iter().any(|q| !q.is_finite()) { if std::env::var("FQ_FIN").is_ok() { eprintln!("  qdldl: non-finite solution (reg {})", f.regularize_count()); } return None; }
    let mut w = vec![0.0; nv];
    for (q, &c) in used.iter().enumerate() { w[c] = rhs[q]; }
    Some((w, rhs[nc..].to_vec()))
}


/// Equilibrium LP: t* = min t s.t. |J^T lam + e_s|_inf <= t, lam >= 0 (over the columns the rows touch).  t* ~ 0 iff the
/// rows can hold the side (first-order jammed on this set).
fn eq_lp(nv: usize, si: usize, rows: &[(f64, Vec<(usize, f64)>)]) -> Option<f64> {
    use clarabel::algebra::*;
    use clarabel::solver::*;
    let mut col = vec![usize::MAX; nv];
    let mut used = Vec::new();
    for (_, nz) in rows { for &(k, _) in nz { if col[k] == usize::MAX { col[k] = used.len(); used.push(k); } } }
    let (nc, m) = (used.len(), rows.len());
    let nx = m + 1;
    let (mut ii, mut jj, mut vv, mut b) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    // rows 0..nc: (J^T lam)_c - t <= -e_s[c];  rows nc..2nc: -(J^T lam)_c - t <= e_s[c];  then -lam <= 0
    for (r, (_, nz)) in rows.iter().enumerate() {
        for &(k, a) in nz { ii.push(col[k]); jj.push(r); vv.push(a); ii.push(nc + col[k]); jj.push(r); vv.push(-a); }
    }
    for c in 0..nc {
        let es = if used[c] == si { 1.0 } else { 0.0 };
        ii.push(c); jj.push(m); vv.push(-1.0); b.push(-es);
    }
    for c in 0..nc {
        let es = if used[c] == si { 1.0 } else { 0.0 };
        ii.push(nc + c); jj.push(m); vv.push(-1.0); b.push(es);
    }
    for r in 0..m { ii.push(2 * nc + r); jj.push(r); vv.push(-1.0); b.push(0.0); }
    let a = CscMatrix::new_from_triplets(2 * nc + m, nx, ii, jj, vv);
    let p = CscMatrix::<f64>::zeros((nx, nx));
    let mut q = vec![0.0; nx]; q[m] = 1.0;
    let cones = [NonnegativeConeT(2 * nc + m)];
    let settings = DefaultSettingsBuilder::default().verbose(false).max_iter(200).tol_gap_abs(1e-12).tol_gap_rel(1e-12).tol_feas(1e-12).build().unwrap();
    let mut solver = DefaultSolver::new(&p, &q, &a, &b, &cones, settings).ok()?;
    solver.solve();
    match solver.solution.status {
        SolverStatus::Solved | SolverStatus::AlmostSolved => Some(solver.solution.x[m]),
        _ => None,
    }
}

/// Newton finish: minimum-norm Gauss-Newton projection onto {g_A = 0} for the load-bearing rows A (LP dual support),
/// then checks, all at the f64 floor: converged; every nearby constraint feasible (violators join A, a few rounds);
/// equilibrium J_A^T lam = -e_s with lam >= 0 (min-norm lam); side not above the start.  If the network supports an
/// equilibrium, s is constant to first order on {g_A = 0}, so the projection is the basin's minimum for this branch.
/// Returns Ok((s, x, GN steps)) or Err(reason).
fn newton_finish(p: &Prob, v1: &[f64], a0: &[Spec]) -> Result<(f64, Vec<f64>, usize), String> {
    let n = p.n;
    let si = 3 * n;
    let mut v = v1.to_vec();
    let s0 = v[si];
    let nv = v.len();
    let mut aset: Vec<Spec> = a0.to_vec();
    let mut steps = 0;
    let tol_g = 1e-14 * s0.max(1.0);
    for round in 0..4 {
        let mut last = f64::INFINITY;
        let mut conv = false;
        for _ in 0..30 {
            let near = p.near(&v, 1e-2);
            let idx: std::collections::HashMap<Spec, usize> = near.iter().enumerate().map(|(q, c)| (c.sp, q)).collect();
            let mut rows: Vec<(f64, Vec<(usize, f64)>)> = Vec::with_capacity(aset.len());
            for sp in &aset {
                match idx.get(sp) { Some(&q) => rows.push((near[q].g, near[q].nz.clone())), None => return Err("row-far".into()) }
            }
            let r = rows.iter().map(|c| c.0.abs()).fold(0.0, f64::max);
            if std::env::var("FQ_FIN").is_ok() { eprintln!("  gn round {round} |A| {} r {r:.3e} s {:.13}", rows.len(), v[si]); }
            if r < tol_g { conv = true; break; }
            if r > 0.5 * last { conv = r < 1e-11 * s0.max(1.0); break; }
            last = r;
            let b2: Vec<f64> = rows.iter().map(|c| -c.0).collect();
            let Some((w, _)) = kkt_solve(nv, &rows, &vec![0.0; nv], &b2) else { return Err("solve".into()) };
            steps += 1;
            let v0 = v.clone();
            for q in 0..nv { v[q] += w[q]; }
            p.tie(&v0, &mut v);
        }
        if !conv { return Err(format!("gn-noconv")); }
        let viol: Vec<Spec> = p.near(&v, 1e-6).iter().filter(|c| c.g > 1e-12 * s0.max(1.0) && !aset.contains(&c.sp)).map(|c| c.sp).collect();
        if viol.is_empty() { break; }
        if round == 3 { return Err("viol".into()); }
        aset.extend(viol);
    }
    // KKT Newton on the closed set: [H J^T; J -dI][w; lam] = [-e_s; -g], H = sum lam grad^2 g (lam from the previous
    // solve, clamped >= 0; first lam = min-norm equilibrium).  Converged when |g_A| and the stationarity residual reach
    // the f64 floor; a few violators may join the set on the way.
    let mut lam: Vec<f64> = Vec::new();
    let mut best = f64::INFINITY; let mut bad = 0;
    for _ in 0..40 {
        let near = p.near(&v, 1e-2);
        let idx: std::collections::HashMap<Spec, usize> = near.iter().enumerate().map(|(q, c)| (c.sp, q)).collect();
        let mut cs: Vec<Con> = Vec::new();
        for sp in &aset { match idx.get(sp) { Some(&q) => cs.push(Con { g: near[q].g, nz: near[q].nz.clone(), sp: *sp }), None => return Err("row-far2".into()) } }
        let rows: Vec<(f64, Vec<(usize, f64)>)> = cs.iter().map(|c| (c.g, c.nz.clone())).collect();
        let mut b1 = vec![0.0; nv]; b1[si] = -1.0;
        if lam.len() != rows.len() {
            let Some((_, l)) = kkt_solve(nv, &rows, &b1, &vec![0.0; rows.len()]) else { return Err("solve-l0".into()) };
            lam = l;
        }
        // residuals at v with the current lam
        let mut st = vec![0.0; nv]; st[si] = 1.0;
        for (c, &l) in cs.iter().zip(&lam) { for &(k, a) in &c.nz { st[k] += l * a; } }
        let sres = st.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        let gres = rows.iter().fold(0.0f64, |a, c| a.max(c.0.abs()));
        if std::env::var("FQ_FIN").is_ok() { eprintln!("  kkt |A| {} gres {gres:.2e} sres {sres:.2e} s {:.13}", rows.len(), v[si]); }
        if gres < 1e-13 * s0.max(1.0) && sres < 1e-11 { break; }
        let merit = gres + sres;
        if merit < 0.5 * best { best = merit; bad = 0; } else { bad += 1; if bad >= 4 { return Err(format!("kkt-noconv g {gres:.1e} st {sres:.1e}")); } }
        let lc: Vec<f64> = lam.iter().map(|&l| l.max(0.0)).collect();
        let h = hess_lagr2(p, &v, &cs, &lc, false);
        if std::env::var("FQ_FIN").is_ok() {
            let lm = lam.iter().fold(0.0f64, |a, &b| a.max(b.abs())); let ln = lam.iter().filter(|&&l| l < 0.0).count();
            let hm = h.iter().fold(0.0f64, |a, q| a.max(q.2.abs())); let hn = h.iter().any(|q| !q.2.is_finite());
            eprintln!("    lam max {lm:.2e} neg {ln}; H nnz {} max {hm:.2e} nonfinite {hn}", h.len());
        }
        let b2: Vec<f64> = rows.iter().map(|c| -c.0).collect();
        let Some((mut w, l2)) = kkt_solve_h(nv, &rows, Some(&h), &b1, &b2) else { return Err("solve-kkt".into()) };
        let wm = w.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        if wm > 1e-2 { for q in w.iter_mut() { *q *= 1e-2 / wm; } }
        let v0 = v.clone();
        for q in 0..nv { v[q] += w[q]; }
        p.tie(&v0, &mut v);
        lam = l2;
        steps += 1;
        let viol: Vec<Spec> = p.near(&v, 1e-6).iter().filter(|c| c.g > 1e-12 * s0.max(1.0) && !aset.contains(&c.sp)).map(|c| c.sp).collect();
        if !viol.is_empty() { if aset.len() > a0.len() + 200 { return Err("viol2".into()); } aset.extend(viol); lam.clear(); best = f64::INFINITY; }
    }
    // equilibrium at the projected point: [I J^T; J -dI][r; lam] = [-e_s; 0]  =>  lam = argmin |J^T lam + e_s|
    let near = p.near(&v, 1e-2);
    let idx: std::collections::HashMap<Spec, usize> = near.iter().enumerate().map(|(q, c)| (c.sp, q)).collect();
    let rows: Vec<(f64, Vec<(usize, f64)>)> = aset.iter().map(|sp| (near[idx[sp]].g, near[idx[sp]].nz.clone())).collect();
    if !rows.iter().any(|c| c.1.iter().any(|&(k, _)| k == si)) { return Err("no-s".into()); }
    let mut b1 = vec![0.0; nv]; b1[si] = -1.0;
    let Some((res, lam)) = kkt_solve(nv, &rows, &b1, &vec![0.0; rows.len()]) else { return Err("solve-eq".into()) };
    let eres = res.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
    let lmax = lam.iter().cloned().fold(0.0, f64::max);
    let lmin = lam.iter().cloned().fold(f64::INFINITY, f64::min);
    if eres > 1e-8 || lmin < -1e-8 * lmax.max(1e-300) {
        // min-norm lam is not unique with redundant rows: decide with the LP
        match eq_lp(nv, si, &rows) {
            Some(tt) if tt < 1e-9 => {}
            Some(tt) => return Err(format!("eq-lp {tt:.1e} (mn res {eres:.1e} lmin {:.1e})", lmin / lmax)),
            None => return Err("eq-lp-fail".into()),
        }
    }
    let s1 = v[si];
    if s1 > s0 + 1e-9 * s0.max(1.0) { return Err(format!("higher {:.1e}", s1 - s0)); }
    Ok((s1, v[..3 * n].to_vec(), steps))
}

/// Trust-region SLP on the lifted problem with face-branch separators (snapped each iteration), and a greedy branch
/// search over ambiguous (corner-corner-like) pairs when the step stalls.
/// Returns (s, x, iterations, final R, flips accepted).
fn polish(p: &mut Prob, s0: f64, x0: &[f64], o: &POpt, trace: &mut Vec<(usize, f64, f64, usize, f64, f64, usize, usize)>, p_load: &mut Vec<(u32, u32, u8, u8)>) -> (f64, Vec<f64>, usize, f64, usize) {
    let tp = Instant::now();
    let mut prev_act: Vec<(u32, u32, u8, u8)> = Vec::new();
    let n = p.n;
    let mut x = x0.to_vec();
    let mut s = s0;
    let mut r = o.r0;
    let kappa = 3.0 * s0;
    let mut it = 0;
    let mut flips = 0;
    p.rebuild(&x, SQRT2 + 0.1);
    let mut choice: Vec<Option<usize>> = vec![None; p.pairs.len()];
    let mut keys: Vec<(usize, usize)> = p.pairs.iter().map(|q| (q.i, q.j)).collect();
    let mut xref = x.clone();
    let merit = |p: &mut Prob, v: &[f64]| -> f64 { v[3 * p.n] + kappa * p.viol_update(v, false).max(0.0) };
    // LP at the current point with the current branch choice: (pred, w, tau, ncons)
    let try_lp = |p: &mut Prob, x: &[f64], s: f64, choice: &mut [Option<usize>], r: f64, hq: &[(usize, usize, f64)]| -> Option<(f64, Vec<f64>, Vec<Con>, f64, Vec<f64>, f64)> {
        snap(p, x, choice, o.cc_tol);
        let v0 = p.pack(x, s);
        let m0 = merit(p, &v0);
        let cons = p.near(&v0, 5.0 * r + 1e-9);
        let (w, tau, z) = slp_lp(v0.len(), 3 * p.n, &cons, r, kappa, hq, p)?;
        Some((m0 - (s + r * w[3 * p.n] + kappa * r * tau), w, cons, m0, z, tau))
    };
    let mut stalls = 0;
    let (mut fin_prev, mut fin_tried): (u64, std::collections::HashSet<u64>) = (0, std::collections::HashSet::new());
    let mut hq: Vec<(usize, usize, f64)> = Vec::new();
    let mut hist: Vec<f64> = Vec::new();
    while it < o.maxit {
        it += 1;
        hist.push(s);
        if o.stag_w > 0 && hist.len() > o.stag_w && hist[hist.len() - 1 - o.stag_w] - s < o.stag_tol { break; }
        let mut dmax: f64 = 0.0;
        for i in 0..n { dmax = dmax.max((x[3 * i] - xref[3 * i]).hypot(x[3 * i + 1] - xref[3 * i + 1])); }
        if dmax > 0.04 {
            // rebuild, carrying branch choices by key
            let old: std::collections::HashMap<(usize, usize), Option<usize>> = keys.iter().cloned().zip(choice.iter().cloned()).collect();
            p.rebuild(&x, SQRT2 + 0.1);
            keys = p.pairs.iter().map(|q| (q.i, q.j)).collect();
            choice = keys.iter().map(|k| *old.get(k).unwrap_or(&None)).collect();
            hq.clear();
            xref = x.clone();
        }
        LP_IT.store(it, Ordering::Relaxed); LP_TAG.store(0, Ordering::Relaxed);
        let Some((pred, w, mut cons, m0, zd, tau0)) = try_lp(p, &x, s, &mut choice, r, &hq) else { r /= 4.0; if r < o.rmin { break; } continue; };
        let small = pred < 1e-14 * s.max(1.0) || r < o.rmin;
        if small {
            // branch search: flip one ambiguous pair at a time, keep the best predicted decrease
            let amb = snap(p, &x, &mut choice, o.cc_tol);
            let rt = r.max(1e-6);
            // only pairs whose rows carry multiplier can gain from a flip: rank by total dual, try the top few
            let mut wt = vec![0.0; p.pairs.len()];
            for (c, &zz) in cons.iter().zip(zd.iter()) {
                if let Spec::PI(pk, _) | Spec::PJ(pk, _) = c.sp { wt[pk] += zz; }
            }
            let zmax = wt.iter().cloned().fold(0.0, f64::max);
            let mut amb: Vec<(usize, Vec<usize>)> = amb.into_iter().filter(|(pk, _)| wt[*pk] > 1e-6 * zmax).collect();
            amb.sort_by(|a, b| wt[b.0].partial_cmp(&wt[a.0]).unwrap());
            amb.truncate(o.flip_top);
            let mut best: Option<(f64, usize, usize)> = None;
            for (pk, cand) in &amb {
                if best.is_some() { break; }
                let (pk, cur) = (*pk, choice[*pk]);
                for &c in cand {
                    if Some(c) == cur { continue; }
                    let mut ch = choice.clone();
                    ch[pk] = Some(c);
                    LP_TAG.store(2, Ordering::Relaxed);
                    FLIP_LPS.fetch_add(1, Ordering::Relaxed);
                    if let Some((pr2, ..)) = try_lp(p, &x, s, &mut ch, rt, &hq) {
                        if pr2 > 1e-12 * s && best.map_or(true, |b| pr2 > b.0) { best = Some((pr2, pk, c)); }
                    }
                    if best.is_some() { break; }
                }
            }
            if o.verbose { eprintln!("  stall at it {it}: R {r:.1e} pred {pred:.1e}, {} ambiguous pairs, best flip {:?}", amb.len(), best.map(|b| b.0)); }
            if std::env::var("FQ_FLIPDBG").is_ok() {
                let rank = best.map_or(-1, |b| amb.iter().position(|q| q.0 == b.1).map_or(-2, |i| i as i64));
                eprintln!("flipstat it {it} namb {} rank {rank} lps {}", amb.len(), FLIP_LPS.swap(0, Ordering::Relaxed));
            }
            match best {
                Some((_, pk, c)) if stalls < 50 => { choice[pk] = Some(c); flips += 1; stalls += 1; r = rt; continue; }
                _ => break,
            }
        }
        let v0 = p.pack(&x, s);
        let nv = v0.len();
        let mut v1: Vec<f64> = (0..nv).map(|k| v0[k] + r * w[k]).collect();
        p.tie(&v0, &mut v1);
        let mut m1 = merit(p, &v1);
        let mut rho = (m0 - m1) / pred;
        let rho_pre = rho;
        let wmax = w.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        if rho < 0.75 {
            // second-order correction: shift each row by its curvature error at the trial point
            let g_old: Vec<f64> = cons.iter().map(|c| c.g).collect();
            for c in cons.iter_mut() {
                let lin: f64 = c.nz.iter().map(|&(k, a)| a * r * w[k]).sum();
                c.g = p.g_at(&v1, c.sp) - lin;
            }
            LP_TAG.store(1, Ordering::Relaxed);
            // Active-set SOC, default since 10-13 (PACKER "Q2" + race): min-norm correction on the step LP's active set, LP as fallback; FQ_SOC_AS=0 restores the SOC LP
            static SOC_AS: OnceLock<bool> = OnceLock::new();
            let as_w = if *SOC_AS.get_or_init(|| std::env::var("FQ_SOC_AS").map(|v| v != "0").unwrap_or(true)) && hq.is_empty() {
                soc_active(nv, 3 * n, &cons, &g_old, r, &w, tau0)
            } else { None };
            if let Some((w2, _, _)) = as_w.map(|w2| (w2, 0.0, Vec::new())).or_else(|| slp_lp(nv, 3 * n, &cons, r, kappa, &hq, p)) {
                let mut v2: Vec<f64> = (0..nv).map(|k| v0[k] + r * w2[k]).collect();
                p.tie(&v0, &mut v2);
                let m2 = merit(p, &v2);
                if m2 < m1 { m1 = m2; v1 = v2; rho = (m0 - m1) / pred; }
            }
        }
        if TRACE_TR.get_or_init(|| std::env::var("FQ_TR").is_ok()).clone() { eprintln!("tr {it} R {r:.2e} pred {pred:.2e} rho0 {rho_pre:.2e} rho {rho:.3} wmax {wmax:.2} s {:.13} acc {}", s, rho >= 0.1); }
        if o.verbose && (it < 20 || it % 10 == 0) { eprintln!("  slp {it}: R {r:.1e} s {:.13} pred {pred:.2e} rho {rho:.2}", s); }
        if rho >= 0.1 {
            x.copy_from_slice(&v1[..3 * n]);
            s = v1[3 * n];
            if o.sqp { hq = hess_lagr(p, &v1, &cons, &zd); }
            let (dm, sl, na, nd) = if o.ident {
                let (dm, sl, _, _) = ident(p, &v0, &v1, &mut Vec::new(), 1e-8);
                let lk = load_keys(p, &v0, &cons, &zd);
                let a: std::collections::HashSet<_> = lk.iter().collect();
                let b: std::collections::HashSet<_> = prev_act.iter().collect();
                let nd = a.symmetric_difference(&b).count();
                let na = lk.len();
                IDENT_FPS.lock().unwrap().push(fingerprint(&lk));
                prev_act = lk;
                (dm, sl, na, nd)
            } else { (0.0, 0.0, 0, 0) };
            trace.push((it, tp.elapsed().as_secs_f64(), m1, flips, dm, sl, na, nd));
            if o.finish {
                // row-level load set of this LP (dual support); stable over two consecutive LPs and untried -> finish
                let zm = zd.iter().cloned().fold(0.0, f64::max);
                let mut a: Vec<(Spec, (u32, u32, u8, u8))> = cons.iter().zip(&zd).filter(|(_, &z)| z > 1e-6 * zm).map(|(c, _)| (c.sp, row_key(p, &v0, c))).collect();
                let sup: std::collections::HashSet<Spec> = a.iter().map(|q| q.0).collect();
                a.sort_by_key(|q| q.1);
                let fpk = fingerprint(&a.iter().map(|q| q.1).collect::<Vec<_>>());
                if std::env::var("FQ_FIN").is_ok() {
                    let mut h = [[0usize; 8]; 2];
                    for (c, &z) in cons.iter().zip(&zd) {
                        let gb = if c.g >= -1e-9 { 0 } else { ((-c.g).log10() + 9.0).ceil().clamp(1.0, 7.0) as usize };
                        h[(z > 1e-6 * zm) as usize][gb] += 1;
                    }
                    eprintln!("it {it}: rows by gap decade (<=1e-9, 1e-8.., ..>1e-3): off-support {:?} support {:?}", h[0], h[1]);
                }
                if fpk == fin_prev && !fin_tried.contains(&fpk) {
                    fin_tried.insert(fpk);
                    // plus every row the next step could close (gap below the trust radius at the new point)
                    let mut specs: Vec<Spec> = a.iter().map(|q| q.0).collect();
                    for c in p.near(&v1, r) { if !sup.contains(&c.sp) { specs.push(c.sp); } }
                    match newton_finish(p, &v1, &specs) {
                        Ok((s2, x2, gn)) => {
                            FIN_LOG.lock().unwrap().push(format!("ok it {it} gn {gn} ds {:.2e}", s2 - s));
                            x = x2; s = s2;
                            trace.push((it, tp.elapsed().as_secs_f64(), s, flips, 0.0, 0.0, specs.len(), 999));
                            break;
                        }
                        Err(e) => FIN_LOG.lock().unwrap().push(format!("fail it {it}: {e}")),
                    }
                }
                fin_prev = fpk;
            }
            if rho > o.tr_grow && wmax > 0.5 { r = (2.0 * r).min(o.rmax); }
        } else {
            r /= o.tr_shrink;
        }
    }
    // leave separators consistent with x for the caller
    snap(p, &x, &mut choice, o.cc_tol);
    // final force network: one LP at small R at the final point
    LP_TAG.store(3, Ordering::Relaxed);
    if let Some((_, _, cons, _, z, _)) = try_lp(p, &x, s, &mut choice, r.max(1e-9).min(1e-6), &[]) {
        let v = p.pack(&x, s);
        *p_load = load_keys(p, &v, &cons, &z);
    }
    (s, x, it, r, flips)
}

// ---------------------------------------------------------------- L-BFGS
fn lbfgs(f: &mut dyn FnMut(&[f64], &mut [f64]) -> f64, v: &mut [f64], maxit: usize, gtol: f64, maxstep: f64) -> (f64, usize, f64) {
    let dim = v.len();
    let m = 12;
    let mut g = vec![0.0; dim];
    let mut e = f(v, &mut g);
    let mut nev = 1;
    let mut ss: Vec<Vec<f64>> = Vec::new();
    let mut ys: Vec<Vec<f64>> = Vec::new();
    let mut rho: Vec<f64> = Vec::new();
    let mut d = vec![0.0; dim];
    let mut xn = vec![0.0; dim];
    let mut gn = vec![0.0; dim];
    let mut alpha = vec![0.0; m];
    let mut spare: Option<(Vec<f64>, Vec<f64>)> = None;
    let mut stall = 0;
    let ginf = |g: &[f64]| g.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
    for _ in 0..maxit {
        if ginf(&g) < gtol || stopped() { break; }
        for k in 0..dim { d[k] = -g[k]; }
        let h = ss.len();
        let mut gd;
        if fast_dot() {
            // fused two-loop (perf 10-13): each update of d is followed in the same pass by the next dot product,
            // accumulated in fast dot's 8-lane pattern -- the same operations in the same order (bit-identical)
            if h > 0 {
                let mut t = dot(&ss[h - 1], &d);
                for k in (0..h).rev() {
                    let a = rho[k] * t;
                    alpha[k] = a;
                    if k > 0 { t = upd_dot::<true>(&mut d, &ys[k], a, &ss[k - 1]); } else { for q in 0..dim { d[q] -= a * ys[k][q]; } }
                }
                let gam = dot(&ss[h - 1], &ys[h - 1]) / dot(&ys[h - 1], &ys[h - 1]);
                for q in 0..dim { d[q] *= gam; }
                let mut t = dot(&ys[0], &d);
                gd = 0.0;
                for k in 0..h {
                    let c = alpha[k] - rho[k] * t;
                    if k + 1 < h { t = upd_dot::<false>(&mut d, &ss[k], c, &ys[k + 1]); } else { gd = upd_dot::<false>(&mut d, &ss[k], c, &g); }
                }
            } else {
                let gm = ginf(&g);
                let sc = if gm > 0.0 { (1e-3 / gm).min(1.0) } else { 1.0 };
                for q in 0..dim { d[q] *= sc; }
                gd = dot(&g, &d);
            }
        } else {
            for k in (0..h).rev() {
                let a = rho[k] * dot(&ss[k], &d);
                alpha[k] = a;
                for q in 0..dim { d[q] -= a * ys[k][q]; }
            }
            if h > 0 {
                let gam = dot(&ss[h - 1], &ys[h - 1]) / dot(&ys[h - 1], &ys[h - 1]);
                for q in 0..dim { d[q] *= gam; }
            } else {
                let gm = ginf(&g);
                let sc = if gm > 0.0 { (1e-3 / gm).min(1.0) } else { 1.0 };
                for q in 0..dim { d[q] *= sc; }
            }
            for k in 0..h {
                let b = rho[k] * dot(&ys[k], &d);
                for q in 0..dim { d[q] += ss[k][q] * (alpha[k] - b); }
            }
            gd = dot(&g, &d);
        }
        if !(gd < 0.0) {
            ss.clear(); ys.clear(); rho.clear();
            let gm = ginf(&g);
            for q in 0..dim { d[q] = -g[q] * (1e-3 / gm).min(1.0); }
            gd = dot(&g, &d);
        }
        let dmax = ginf(&d);
        let mut step = if dmax > maxstep { maxstep / dmax } else { 1.0 };
        let mut ok = false;
        for _ in 0..50 {
            for q in 0..dim { xn[q] = v[q] + step * d[q]; }
            let en = f(&xn, &mut gn);
            nev += 1;
            if en <= e + 1e-4 * step * gd {
                // history pair in reused buffers (the evicted oldest pair becomes the next spare; bit-identical)
                let (mut sv, mut yv) = spare.take().unwrap_or_else(|| (vec![0.0; dim], vec![0.0; dim]));
                for q in 0..dim { sv[q] = xn[q] - v[q]; yv[q] = gn[q] - g[q]; }
                let sy = dot(&sv, &yv);
                if sy > 1e-300 {
                    if ss.len() == m { spare = Some((ss.remove(0), ys.remove(0))); rho.remove(0); }
                    rho.push(1.0 / sy); ss.push(sv); ys.push(yv);
                } else { spare = Some((sv, yv)); }
                if e - en <= 1e-15 * e.abs().max(1.0) { stall += 1; } else { stall = 0; }
                v.copy_from_slice(&xn);
                g.copy_from_slice(&gn);
                e = en;
                ok = true;
                break;
            }
            step *= 0.5;
        }
        if !ok {
            if ss.is_empty() { break; }
            ss.clear(); ys.clear(); rho.clear();
            stall += 1;
        }
        if stall > 10 { break; }
    }
    (e, nev, ginf(&g))
}

fn fast_sc() -> bool {
    static F: OnceLock<bool> = OnceLock::new();
    *F.get_or_init(|| std::env::var("FQ_FASTSC").map(|v| v != "0").unwrap_or(true))
}

/// Vectorisable sincos of a batch (perf 10-13, PACKER "Q11"): Cody-Waite reduction by pi/2 (two-part constant, FMA),
/// Cephes minimax polynomials on [-pi/4, pi/4], branch-free quadrant select; <= 2 ulp from libm (|x| < 1e5; larger
/// arguments are redone with libm).
#[inline(never)]
fn sincos_batch(xs: &[f64], so: &mut [f64], co: &mut [f64]) {
    let n = xs.len();
    let (so, co) = (&mut so[..n], &mut co[..n]);
    for i in 0..n {
        let x = xs[i];
        let q = (x * 0.636619772367581343076).round();
        let r = (-q).mul_add(6.123233995736766035868820e-17, (-q).mul_add(1.5707963267948965579989817, x));
        let z = r * r;
        let sn = r + r * z * (-1.66666666666666307295e-1 + z * (8.33333333332211858878e-3 + z * (-1.98412698295895385996e-4
                 + z * (2.75573136213857245213e-6 + z * (-2.50507477628578072866e-8 + z * 1.58962301576546568060e-10)))));
        let cn = 1.0 - 0.5 * z + z * z * (4.16666666666665929218e-2 + z * (-1.38888888888730564116e-3 + z * (2.48015872888517045348e-5
                 + z * (-2.75573141792967388112e-7 + z * (2.08757008419747316778e-9 + z * -1.13585365213876817300e-11)))));
        let qi = q as i64;
        let (a, b) = if (qi & 1) != 0 { (cn, sn) } else { (sn, cn) };
        so[i] = if (qi & 2) != 0 { -a } else { a };
        co[i] = if ((qi + 1) & 2) != 0 { -b } else { b };
    }
    for i in 0..n { if !(xs[i].abs() < 1e5) { let (s, c) = xs[i].sin_cos(); so[i] = s; co[i] = c; } }
}

#[inline]
fn fast_dot() -> bool {
    static FAST: OnceLock<bool> = OnceLock::new();
    *FAST.get_or_init(|| std::env::var("FQ_FASTDOT").map(|v| v != "0").unwrap_or(true))
}

/// d <- d - c u (sub) or d <- d + u c, then the fast dot of w with the updated d, in one pass (bit-identical to the
/// update loop followed by dot(w, d) in fast mode)
#[inline]
fn upd_dot<const SUB: bool>(d: &mut [f64], u: &[f64], c: f64, w: &[f64]) -> f64 {
    let n = d.len();
    let (u, w) = (&u[..n], &w[..n]);
    let full = n / 8 * 8;
    let mut acc = [0.0f64; 8];
    let mut base = 0;
    while base < full {
        for l in 0..8 {
            let q = base + l;
            d[q] = if SUB { d[q] - c * u[q] } else { d[q] + u[q] * c };
            acc[l] += w[q] * d[q];
        }
        base += 8;
    }
    for q in full..n { d[q] = if SUB { d[q] - c * u[q] } else { d[q] + u[q] * c }; }
    let mut t = ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]));
    for q in full..n { t += w[q] * d[q]; }
    t
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    // 8 independent accumulators (vectorises; different rounding), default since 10-13 (PACKER "Q3", "Q9b"); FQ_FASTDOT=0 restores the sequential sum
    static FAST: OnceLock<bool> = OnceLock::new();
    if *FAST.get_or_init(|| std::env::var("FQ_FASTDOT").map(|v| v != "0").unwrap_or(true)) {
        let n = a.len().min(b.len());
        let (a, b) = (&a[..n], &b[..n]);
        let mut acc = [0.0f64; 8];
        let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
        let (ra, rb) = (ca.remainder(), cb.remainder());
        for (x, y) in ca.zip(cb) { for l in 0..8 { acc[l] += x[l] * y[l]; } }
        let mut t = ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]));
        for (x, y) in ra.iter().zip(rb) { t += x * y; }
        return t;
    }
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

// ---------------------------------------------------------------- quench
struct QOpt { mu0: f64, mu_max: f64, tol: f64, skin: f64, max_outer: usize, verbose: bool }

/// Returns (s, x, total evaluations, outer iterations, final violation before repair).
/// Minimise the pure overlap penalty at fixed side s (separators free, multipliers 0).  Returns (x, max violation, evals).
fn relax_fixed(p: &mut Prob, s: f64, x0: &[f64], skin: f64) -> (Vec<f64>, f64, usize) {
    relax_fixed_tol(p, s, x0, skin, 4000, 1e-13)
}

/// relax_fixed with the inner L-BFGS budget exposed (10-24, `--round-maxit` / `--round-gtol`: intermediate hardening
/// steps need not be solved to the f64 floor).
fn relax_fixed_tol(p: &mut Prob, s: f64, x0: &[f64], skin: f64, maxit: usize, gtol: f64) -> (Vec<f64>, f64, usize) {
    let n = p.n;
    let cut = SQRT2 + skin;
    let mut x = x0.to_vec();
    p.fix_s = true;
    p.mu = 1.0;
    for l in p.wlam.iter_mut() { *l = 0.0; }
    p.rebuild(&x, cut);
    for pr in p.pairs.iter_mut() { pr.lam = [0.0; 8]; }
    let mut nev = 0;
    let mut viol = f64::INFINITY;
    for _round in 0..20 {
        let xref = x.clone();
        let mut v = p.pack(&x, s);
        let (_, ne, _) = { let pr = &*p; lbfgs(&mut |vv: &[f64], gg: &mut [f64]| pr.eval(vv, gg), &mut v, maxit, gtol, 0.05) };
        nev += ne;
        p.unpack(&v);
        x.copy_from_slice(&v[..3 * n]);
        viol = p.viol_update(&v, false);
        let mut dmax: f64 = 0.0;
        for i in 0..n { dmax = dmax.max((x[3 * i] - xref[3 * i]).hypot(x[3 * i + 1] - xref[3 * i + 1])); }
        if dmax > 0.5 * skin { p.rebuild(&x, cut); continue; }
        break;
    }
    p.fix_s = false;
    (x, viol, nev)
}

/// Shrink-hopping descent: grow until the fixed-side overlap problem is solved, then shrink by d (positions scaled about
/// the centre) while it stays solvable, halving d on failure (back to the last feasible state) down to dmin.
/// Returns (s, x) of the last feasible state, total evals, shrink steps.
/// Hopping (10-15, SQUISH "Iterative Shrink-Hopping" reading): when a shrink step fails, up to `hops` kicks of `hk`
/// random squares (all if hk = 0; gaussian sigma on x, y and angle) from the failed relaxed state, each re-relaxed at
/// the same target side, before d is halved; a kick that resolves the step is accepted like a plain success.
#[allow(clippy::too_many_arguments)]
fn shrink(s0: f64, x0: &[f64], d0: f64, dmin: f64, vtol: f64, verbose: bool, hops: usize, sig: f64, hk: usize, seed: u64) -> (f64, Vec<f64>, usize, usize) {
    let mut rng = Rng::new(seed);
    let n = x0.len() / 3;
    let mut p = Prob { n, pairs: Vec::new(), wlam: vec![0.0; 16 * n], mu: 1.0, fix_s: true, b: Vec::new() };
    let fz = frozen_mask(n);
    let fr = |i: usize| !fz.is_empty() && fz[i];
    let scale = |x: &[f64], s: f64, f: f64| -> Vec<f64> {
        let mut y = x.to_vec();
        for i in (0..n).filter(|&i| !fr(i)) { y[3 * i] = s * f / 2.0 + (x[3 * i] - s / 2.0) * f; y[3 * i + 1] = s * f / 2.0 + (x[3 * i + 1] - s / 2.0) * f; }
        y
    };
    let mut s = s0;
    let mut x = x0.to_vec();
    let mut nev = 0;
    // grow to feasibility
    let mut feas: Option<(f64, Vec<f64>)> = None;
    for _ in 0..40 {
        let (xr, vi, ne) = relax_fixed(&mut p, s, &x, 0.3);
        nev += ne;
        if vi < vtol { feas = Some((s, xr)); break; }
        x = scale(&xr, s, 1.0 + d0);
        s *= 1.0 + d0;
    }
    let Some((mut sf, mut xf)) = feas else { return (f64::INFINITY, x, nev, 0); };
    let mut d = d0;
    let mut steps = 0;
    while d >= dmin && steps < 400 {
        steps += 1;
        let st = sf * (1.0 - d);
        let xt = scale(&xf, sf, 1.0 - d);
        let (xr, vi, ne) = relax_fixed(&mut p, st, &xt, 0.3);
        nev += ne;
        if vi < vtol { sf = st; xf = xr; if verbose { eprintln!("  shrink ok s {sf:.10} d {d:.1e}"); } continue; }
        let (mut xb, mut vb, mut ok) = (xr, vi, false);
        for h in 0..hops {
            let mut xk = xb.clone();
            let m = if hk == 0 || hk >= n { n } else { hk };
            for _ in 0..m {
                let i = if m == n { 0 } else { (rng.next() % n as u64) as usize };
                let idx: Vec<usize> = if m == n { (0..n).collect() } else { vec![i] };
                for &j in idx.iter().filter(|&&j| !fr(j)) {
                    xk[3 * j] = (xk[3 * j] + sig * rng.gauss()).clamp(H, st - H);
                    xk[3 * j + 1] = (xk[3 * j + 1] + sig * rng.gauss()).clamp(H, st - H);
                    xk[3 * j + 2] += sig * rng.gauss();
                }
                if m == n { break; }
            }
            let (xr2, vi2, ne2) = relax_fixed(&mut p, st, &xk, 0.3);
            nev += ne2;
            if vi2 < vtol { sf = st; xf = xr2; ok = true; if verbose { eprintln!("  hop {h} ok s {sf:.10} d {d:.1e}"); } break; }
            if vi2 < vb { vb = vi2; xb = xr2; }
        }
        if !ok { d /= 2.0; }
    }
    (sf, xf, nev, steps)
}

fn quench(s0: f64, x0: &[f64], o: &QOpt) -> (f64, Vec<f64>, usize, usize, f64, Prob) {
    let n = x0.len() / 3;
    let mut p = Prob { n, pairs: Vec::new(), wlam: vec![0.0; 16 * n], mu: o.mu0, fix_s: false,
                       b: if soft_eps() > 0.0 { vec![0.0; n] } else { Vec::new() } };
    let cut = SQRT2 + o.skin;
    let mut x = x0.to_vec();
    let mut s = s0;
    p.rebuild(&x, cut);
    let mut xref = x.clone();
    let mut nev = 0;
    let mut prev_v = f64::INFINITY;
    let mut outer = 0;
    let mut viol = f64::INFINITY;
    let traj = TRAJ_ON.load(Ordering::Relaxed);
    let tq = Instant::now();
    while outer < o.max_outer && !stopped() {
        outer += 1;
        let mut v = p.pack(&x, s);
        let gtol = (1e-2 / p.mu.sqrt()).max(1e-11);
        let (_, ne, gi) = {
            let pr = &p;
            lbfgs(&mut |vv: &[f64], gg: &mut [f64]| pr.eval(vv, gg), &mut v, 20000, gtol, 0.05)
        };
        nev += ne;
        p.unpack(&v);
        x.copy_from_slice(&v[..3 * n]);
        s = v[3 * n];
        // pair list still valid?
        let mut dmax: f64 = 0.0;
        for i in 0..n {
            let (dx, dy) = (x[3 * i] - xref[3 * i], x[3 * i + 1] - xref[3 * i + 1]);
            dmax = dmax.max((dx * dx + dy * dy).sqrt());
        }
        if dmax > 0.5 * o.skin {
            p.rebuild(&x, cut);
            xref = x.clone();
            if o.verbose { eprintln!("  rebuild: {} pairs", p.pairs.len()); }
            if p.pairs.len() > PAIR_CAP * n { DIVERGED.store(true, Ordering::Relaxed); }
            if traj { TRAJ.lock().unwrap().push((outer, s, f64::NAN, p.mu, tq.elapsed().as_secs_f64())); }
            continue; // re-solve with the new list before judging violation
        }
        viol = p.viol_update(&v, true);
        if traj { TRAJ.lock().unwrap().push((outer, s, viol, p.mu, tq.elapsed().as_secs_f64())); }
        if o.verbose { eprintln!("outer {outer}: mu {:.1e} s {:.12} viol {:.2e} |g| {:.1e} evals {ne}", p.mu, s, viol, gi); }
        if viol < o.tol && p.mu >= 1e3 { break; }
        if viol > 0.25 * prev_v && p.mu < o.mu_max { p.mu *= 10.0; }
        prev_v = viol;
    }
    (s, x, nev, outer, viol, p)
}

/// Exact-ish (f64 SAT) feasibility: min pair gap and min wall clearance.
fn check(s: f64, x: &[f64]) -> (f64, f64) {
    let n = x.len() / 3;
    let mut mg = f64::INFINITY;
    for i in 0..n {
        for j in i + 1..n {
            let (dx, dy) = (x[3 * j] - x[3 * i], x[3 * j + 1] - x[3 * i + 1]);
            if dx * dx + dy * dy < 2.1 {
                mg = mg.min(sat_gap(x[3 * i], x[3 * i + 1], x[3 * i + 2], x[3 * j], x[3 * j + 1], x[3 * j + 2]));
            }
        }
    }
    let mut mw = f64::INFINITY;
    for i in 0..n {
        let t = x[3 * i + 2];
        let w = H * (t.cos().abs() + t.sin().abs());
        mw = mw.min(x[3 * i] - w).min(s + rdx() - x[3 * i] - w).min(x[3 * i + 1] - w).min(s + rdy() - x[3 * i + 1] - w);
    }
    (mg, mw)
}

/// Scale centres and side by f >= 1 until the f64 SAT check passes with margin.
fn repair(s: f64, x: &mut [f64], margin: f64) -> f64 {
    let n = x.len() / 3;
    // translate so the bounding box of the squares starts at the origin (walls x >= 0, y >= 0 tight or slack)
    let mut s = s;
    for _ in 0..60 {
        let (mg, mw) = check(s, x);
        let bad = (margin - mg).max(margin - mw);
        if bad <= 0.0 { return s; }
        let f = 1.0 + 2.0 * bad + 1e-15;
        for i in 0..n {
            x[3 * i] = (x[3 * i] - s / 2.0) * f + s * f / 2.0;
            x[3 * i + 1] = (x[3 * i + 1] - s / 2.0) * f + s * f / 2.0;
        }
        // centres scale about the origin; rectangle mode: grow the side so both walls scale by >= f
        let (dx, dy) = (rdx(), rdy());
        s = if dx == 0.0 && dy == 0.0 { s * f } else { ((s + dx) * f - dx).max((s + dy) * f - dy) };
    }
    s
}

// ---------------------------------------------------------------- io
fn read_cfg(path: &str) -> (f64, Vec<f64>) {
    let txt = fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let mut lines = txt.lines().filter(|l| !l.trim().is_empty() && !l.starts_with('#'));
    let head: Vec<f64> = lines.next().unwrap().split_whitespace().map(|t| t.parse().unwrap()).collect();
    let (n, s) = (head[0] as usize, head[1]);
    let mut v = Vec::with_capacity(3 * n);
    for _ in 0..n {
        let t: Vec<f64> = lines.next().unwrap().split_whitespace().take(3).map(|t| t.parse().unwrap()).collect();
        v.push(t[0]); v.push(t[1]); v.push(t[2].to_radians());
    }
    (s, v)
}

fn write_cfg(path: &str, s: f64, v: &[f64]) {
    let n = v.len() / 3;
    let mut out = format!("{} {:.17}\n", n, s);
    for i in 0..n {
        out += &format!("{:.17} {:.17} {:.15}\n", v[3 * i], v[3 * i + 1], v[3 * i + 2].to_degrees().rem_euclid(90.0));
    }
    let tmp = format!("{path}.tmp");
    fs::write(&tmp, out).unwrap();
    fs::rename(&tmp, path).unwrap();
}

fn arg<T: std::str::FromStr>(a: &[String], k: &str, def: T) -> T {
    a.iter().rposition(|x| x == k).and_then(|i| a.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(def)
}
fn sarg(a: &[String], k: &str) -> Option<String> { a.iter().rposition(|x| x == k).and_then(|i| a.get(i + 1)).cloned() }

/// One quench (ALM screen + SLP polish, options as CLI args) on (s, x); x angles in radians.  Returns (json, s, x).
fn run_quench(a: &[String], s: f64, mut x: Vec<f64>) -> (String, f64, Vec<f64>) {
    FIN_LOG.lock().unwrap().clear();
    IDENT_FPS.lock().unwrap().clear();
        let t0 = Instant::now();
        let n = x.len() / 3;
        let dl: f64 = arg(&a, "--deadline", 0.0);
        *DEADLINE.write().unwrap() = if dl > 0.0 { Some(t0 + std::time::Duration::from_secs_f64(dl)) } else { None };
        DIVERGED.store(false, Ordering::Relaxed);
        TRAJ.lock().unwrap().clear();
        let want_traj = a.iter().any(|t| t == "--traj");
        set_rect(arg(&a, "--rect-dx", 0.0), arg(&a, "--rect-dy", 0.0));
        set_rnd(0.0);
        set_soft(0.0);
        set_frozen(sarg(&a, "--freeze"), n);
        let fz = frozen_mask(n);
        let fr = |i: usize| !fz.is_empty() && fz[i];
        let x_in = x.clone();
        let sig: f64 = arg(&a, "--kick", 0.0);
        if sig > 0.0 {
            let mut r = Rng::new(arg(&a, "--seed", 1u64));
            for i in 0..n {
                if fr(i) { r.gauss(); r.gauss(); r.gauss(); continue; }
                x[3 * i] += sig * r.gauss();
                x[3 * i + 1] += sig * r.gauss();
                x[3 * i + 2] += (20.0 * sig * r.gauss()).to_radians();
            }
        }
        // Soft-to-rigid (rounded squares, after Nakajima / Ostashev): (A) optionally compress at corner radius r0 (side
        // free); (B) hold the box at hold-f x that side and lower r to 0 in round-steps relaxations of the fixed-side
        // overlap penalty (r_k = r0 (1 - k/K)^p, optional sigma kick per step); then the usual quench + polish from there.
        let r0: f64 = arg(&a, "--round-r0", 0.0);
        let mut round_info = String::new();
        let mut s = s;
        if r0 > 0.0 {
            let ksteps: usize = arg(&a, "--round-steps", 20);
            let pw: f64 = arg(&a, "--round-p", 1.0);
            let rk: f64 = arg(&a, "--round-kick", 0.0);
            let mut rng = Rng::new(arg(&a, "--seed", 1u64) ^ 0x5eed);
            set_rnd(r0);
            let oq = QOpt { mu0: arg(&a, "--mu0", 10.0), mu_max: arg(&a, "--mu-max", 1e2), tol: arg(&a, "--tol", 1e-7),
                            skin: arg(&a, "--skin", 0.3), max_outer: arg(&a, "--outer", 60), verbose: a.iter().any(|t| t == "--v") };
            let sa = if a.iter().any(|t| t == "--round-compress") {
                TRAJ_ON.store(want_traj, Ordering::Relaxed);
                let (sq, xq, _, _, _, _) = quench(s * arg(&a, "--round-loosen", 1.0), &x, &oq);
                TRAJ_ON.store(false, Ordering::Relaxed);
                x = xq; sq
            } else { s };
            let mut sh = sa * arg(&a, "--hold-f", 1.0);
            for i in (0..n).filter(|&i| !fr(i)) { x[3 * i] *= sh / sa; x[3 * i + 1] *= sh / sa; }
            let mut p = Prob { n, pairs: Vec::new(), wlam: vec![0.0; 16 * n], mu: 1.0, fix_s: true, b: Vec::new() };
            let mut vlast = 0.0;
            // 10-24 (PACKER "RS1"): --round-maxit / --round-gtol = inner L-BFGS budget of each hardening step (default the
            // f64-floor 4000 / 1e-13); --round-press = side free at every step (an ALM quench at the current r, after
            // Nakajima's pressure-driven hardening) instead of holding the box; --round-kick-peak R [--round-kick-w W]
            // = kick sigma rk * exp(-((r - R) / W)^2) (noise peaking near the angle lock-in) instead of a constant rk.
            let rmaxit: usize = arg(&a, "--round-maxit", 4000);
            let rgtol: f64 = arg(&a, "--round-gtol", 1e-13);
            let press = a.iter().any(|t| t == "--round-press");
            let kpeak: f64 = arg(&a, "--round-kick-peak", -1.0);
            let kw: f64 = arg(&a, "--round-kick-w", 0.03);
            // --round-press-outer K: outer ALM iterations per press step (default the stage's --outer, 60).  The
            // stage's mu_max (1e2) is below the quench's early-exit mu (1e3), so without a cap every press step
            // runs all 60 outer iterations (10-24, RS2).
            let poq = QOpt { max_outer: arg(&a, "--round-press-outer", oq.max_outer), ..oq };
            let mut nev_round = 0usize;
            for k in 1..=ksteps {
                if stopped() { break; }
                let r = r0 * (1.0 - k as f64 / ksteps as f64).max(0.0).powf(pw);
                set_rnd(r);
                let sig = if kpeak >= 0.0 { rk * (-((r - kpeak) / kw).powi(2)).exp() } else { rk };
                if sig > 0.0 { for i in (0..n).filter(|&i| !fr(i)) { x[3 * i] += sig * rng.gauss(); x[3 * i + 1] += sig * rng.gauss(); x[3 * i + 2] += 2.0 * sig * rng.gauss(); } }
                if press {
                    let (sq, xq, ne, _, vi, _) = quench(sh, &x, &poq);
                    x = xq; sh = sq; vlast = vi; nev_round += ne;
                } else {
                    let (xr, vi, ne) = relax_fixed_tol(&mut p, sh, &x, oq.skin, rmaxit, rgtol);
                    x = xr; vlast = vi; nev_round += ne;
                }
                if oq.verbose { eprintln!("  round r {r:.4} side {sh:.8} viol {vlast:.2e} sig {sig:.2e}"); }
            }
            set_rnd(0.0);
            s = sh;
            round_info = format!(", \"round\": {{\"r0\": {r0}, \"s_soft\": {sa:.12}, \"s_hold\": {sh:.12}, \"viol_hard\": {vlast:.3e}, \"evals\": {nev_round}, \"press\": {press}}}");
        }
        // Soft squares (literature/soft-squares-2026-10-12.md section 2): per-square log aspect a_i = eps tanh(b_i), area 1.
        // (A) optional compress with eps0 (side free); (B) hold the box at hold-f x that side; (C) eps -> 0 in K steps of
        // fixed-side overlap relaxation over (x, y, theta, b, separators), optional kick per step; then the rigid finish.
        let e0: f64 = arg(&a, "--soft-a0", 0.0);
        let mut soft_info = String::new();
        if e0 > 0.0 {
            let ksteps: usize = arg(&a, "--soft-steps", 20);
            let pw: f64 = arg(&a, "--soft-p", 1.0);
            let sk: f64 = arg(&a, "--soft-kick", 0.0);
            let mut rng = Rng::new(arg(&a, "--seed", 1u64) ^ 0x50f7);
            set_soft(e0);
            let oq = QOpt { mu0: arg(&a, "--mu0", 10.0), mu_max: arg(&a, "--mu-max", 1e2), tol: arg(&a, "--tol", 1e-7),
                            skin: arg(&a, "--skin", 0.3), max_outer: arg(&a, "--outer", 60), verbose: a.iter().any(|t| t == "--v") };
            let mut b = vec![0.0; n];
            let sa = if a.iter().any(|t| t == "--soft-compress") {
                let (sq, xq, _, _, _, pq) = quench(s, &x, &oq);
                x = xq; b = pq.b.clone(); sq
            } else { s };
            let amax0 = b.iter().map(|q| (e0 * q.tanh()).abs()).fold(0.0, f64::max);
            let sh = sa * arg(&a, "--hold-f", 1.0);
            for i in (0..n).filter(|&i| !fr(i)) { x[3 * i] *= sh / sa; x[3 * i + 1] *= sh / sa; }
            let mut p = Prob { n, pairs: Vec::new(), wlam: vec![0.0; 16 * n], mu: 1.0, fix_s: true, b };
            let mut vlast = 0.0;
            for k in 1..=ksteps {
                let e = e0 * (1.0 - k as f64 / ksteps as f64).max(0.0).powf(pw);
                set_soft(e);
                if sk > 0.0 { for i in (0..n).filter(|&i| !fr(i)) { x[3 * i] += sk * rng.gauss(); x[3 * i + 1] += sk * rng.gauss(); x[3 * i + 2] += 2.0 * sk * rng.gauss(); } }
                if e == 0.0 { p.b.clear(); }
                let (xr, vi, _) = relax_fixed(&mut p, sh, &x, oq.skin);
                x = xr; vlast = vi;
                if oq.verbose { eprintln!("  soft eps {e:.4} hold {sh:.8} viol {vi:.2e}"); }
            }
            set_soft(0.0);
            s = sh;
            soft_info = format!(", \"soft\": {{\"a0\": {e0}, \"s_soft\": {sa:.12}, \"s_hold\": {sh:.12}, \"amax\": {amax0:.4}, \"viol_hard\": {vlast:.3e}}}");
        }
        let lo: f64 = arg(&a, "--loosen", 1.02);
        let s1 = s * lo;
        for i in (0..n).filter(|&i| !fr(i)) {
            x[3 * i] = (x[3 * i] * lo).clamp(H, s1 + rdx() - H);
            x[3 * i + 1] = (x[3 * i + 1] * lo).clamp(H, s1 + rdy() - H);
        }
        let o = QOpt {
            mu0: arg(&a, "--mu0", 10.0), mu_max: arg(&a, "--mu-max", 1e2), tol: arg(&a, "--tol", 1e-7),
            skin: arg(&a, "--skin", 0.3), max_outer: arg(&a, "--outer", 60), verbose: a.iter().any(|t| t == "--v"),
        };
        let (mut sq, mut xq, nev, outer, viol, mut prob) = if a.iter().any(|t| t == "--no-alm") || stopped() {
            // already (nearly) feasible: polish directly
            (s1, x.clone(), 0, 0, 0.0, Prob { n, pairs: Vec::new(), wlam: vec![0.0; 16 * n], mu: 1.0, fix_s: false, b: Vec::new() })
        } else if a.iter().any(|t| t == "--shrink") {
            let (ss, xs, ne, st) = shrink(s1, &x, arg(&a, "--d0", 2e-3), arg(&a, "--dmin", 1e-6), arg(&a, "--vtol", 1e-10), o.verbose,
                                          arg(&a, "--hops", 0usize), arg(&a, "--hop-sigma", 0.02), arg(&a, "--hop-k", 3usize), arg(&a, "--seed", 1u64));
            if o.verbose { eprintln!("shrink: s {ss:.10} evals {ne} steps {st}"); }
            (ss, xs, ne, st, 0.0, Prob { n, pairs: Vec::new(), wlam: vec![0.0; 16 * n], mu: 1.0, fix_s: false, b: Vec::new() })
        } else { quench(s1, &x, &o) };
        let t_alm = t0.elapsed().as_secs_f64();
        let mut xc = xq.clone();
        let s_coarse = repair(sq, &mut xc, 1e-13);
        let s_alm0 = sq;
        let mut pit = 0;
        let mut pr = 0.0;
        let mut flips = 0;
        let mut trace: Vec<(usize, f64, f64, usize, f64, f64, usize, usize)> = Vec::new();
        let mut load: Vec<(u32, u32, u8, u8)> = Vec::new();
        if !a.iter().any(|t| t == "--no-polish") && !stopped() {
            let po = POpt { sqp: a.iter().any(|t| t == "--sqp"), rmax: arg(&a, "--rmax", 4e-3), r0: arg(&a, "--r0", 1e-3), rmin: arg(&a, "--rmin", 1e-10), maxit: arg(&a, "--pit", 300), cc_tol: arg(&a, "--cc-tol", 1e-7), flip_top: arg(&a, "--flip-top", 8), ident: a.iter().any(|t| t == "--ident"), stag_w: arg(&a, "--stag-w", 15), stag_tol: arg(&a, "--stag-tol", 2e-9), finish: a.iter().any(|t| t == "--finish"), tr_grow: arg(&a, "--tr-grow", 0.5), tr_shrink: arg(&a, "--tr-shrink", 4.0), verbose: o.verbose };
            let (s2, x2, it, rr, fl) = polish(&mut prob, sq, &xq, &po, &mut trace, &mut load);
            sq = s2; xq = x2; pit = it; pr = rr; flips = fl;
        }
        let (fp, ncont) = (fingerprint(&load), load.len());
        let sr = repair(sq, &mut xq, 1e-13);
        let (mg, mw) = check(sr, &xq);
        let frozen_info = if fz.is_empty() { String::new() } else {
            let d = (0..n).filter(|&i| fz[i]).map(|i| (xq[3 * i] - x_in[3 * i]).abs().max((xq[3 * i + 1] - x_in[3 * i + 1]).abs()).max(H * SQRT2 * (xq[3 * i + 2] - x_in[3 * i + 2]).abs())).fold(0.0f64, f64::max);
            format!(", \"frozen\": {}, \"frozen_dmax\": {:.3e}", fz.iter().filter(|&&b| b).count(), d)
        };
        let json = format!("{{\"s\": {:.15}, \"s_alm\": {:.15}, \"viol\": {:.2e}, \"evals\": {}, \"outer\": {}, \"slp_it\": {}, \"slp_r\": {:.1e}, \"flips\": {}, \"min_gap\": {:.2e}, \"min_wall\": {:.2e}, \"sec\": {:.3}, \"s_coarse\": {:.15}, \"t_alm\": {:.3}, \"s_alm0\": {:.15}, \"trace\": [{}], \"fp\": \"{:016x}\", \"ncontacts\": {}{}{}}}",
                 sr, sq, viol, nev, outer, pit, pr, flips, mg, mw, t0.elapsed().as_secs_f64(), s_coarse, t_alm, s_alm0, trace.iter().map(|q| format!("[{},{:.4},{:.15},{},{:.3e},{:.3e},{},{}]", q.0, q.1, q.2, q.3, q.4, q.5, q.6, q.7)).collect::<Vec<_>>().join(","), fp, ncont, format!(", \"fin\": [{}]", FIN_LOG.lock().unwrap().iter().map(|q| format!("\"{q}\"")).collect::<Vec<_>>().join(",")) + &if a.iter().any(|t| t == "--ident") { format!(", \"fps\": [{}]", IDENT_FPS.lock().unwrap().iter().map(|h| format!("\"{:016x}\"", h)).collect::<Vec<_>>().join(",")) } else { String::new() }, round_info + &soft_info + &frozen_info);
        let tr = TRAJ.lock().unwrap();
        let extra = format!(", \"deadline\": {}, \"diverged\": {}{}", late(), DIVERGED.load(Ordering::Relaxed),
                            if want_traj { format!(", \"traj\": [{}]", tr.iter().map(|q| format!("[{},{:.12},{:.3e},{:.1e},{:.4}]", q.0, q.1, q.2, q.3, q.4)).collect::<Vec<_>>().join(",")) } else { String::new() });
        let json = json_finite(&(json[..json.len() - 1].to_string() + &extra + "}"));
        (json, sr, xq)
}

/// Persistent worker: requests on stdin, one per packing, until EOF.
///   request:  `quench ARGS...` line, `n s` line, n lines `x y deg`
///   response: the quench JSON line, then `n s` and n lines `x y deg` (same as --out), then `END`
/// Errors (bad input) -> `{"error": ...}` + `END`.  A panic inside the solver aborts the process (caller restarts).
fn serve() {
    use std::io::{BufRead, Write};
    let stdin = std::io::stdin();
    let mut it = stdin.lock().lines();
    let out = std::io::stdout();
    while let Some(Ok(head)) = it.next() {
        let head = head.trim().to_string();
        if head.is_empty() { continue; }
        let mut args: Vec<String> = vec!["fq".into()];
        args.extend(head.split_whitespace().map(|t| t.to_string()));
        let parse = |it: &mut std::io::Lines<std::io::StdinLock>| -> Result<(f64, Vec<f64>), String> {
            let l = it.next().ok_or("eof")?.map_err(|e| e.to_string())?;
            let h: Vec<&str> = l.split_whitespace().collect();
            if h.len() < 2 { return Err(format!("bad header {l}")); }
            let n: usize = h[0].parse().map_err(|_| "n")?; let s: f64 = h[1].parse().map_err(|_| "s")?;
            let mut v = Vec::with_capacity(3 * n);
            for _ in 0..n {
                let l = it.next().ok_or("eof")?.map_err(|e| e.to_string())?;
                let t: Vec<f64> = l.split_whitespace().take(3).map(|q| q.parse().unwrap_or(f64::NAN)).collect();
                if t.len() < 3 || t.iter().any(|q| !q.is_finite()) { return Err(format!("bad row {l}")); }
                v.push(t[0]); v.push(t[1]); v.push(t[2].to_radians());
            }
            Ok((s, v))
        };
        let res = parse(&mut it);
        let mut o = out.lock();
        match (args.get(1).map(|s| s.as_str()), res) {
            (Some("quench"), Ok((s, x))) => {
                let (json, sr, xq) = run_quench(&args, s, x);
                let n = xq.len() / 3;
                let mut buf = format!("{json}\n{} {:.17}\n", n, sr);
                for i in 0..n {
                    buf += &format!("{:.17} {:.17} {:.15}\n", xq[3 * i], xq[3 * i + 1], xq[3 * i + 2].to_degrees().rem_euclid(90.0));
                }
                buf += "END\n";
                o.write_all(buf.as_bytes()).unwrap();
            }
            (cmd, r) => {
                let e = match r { Err(e) => e, Ok(_) => format!("unknown command {cmd:?}") };
                writeln!(o, "{{\"error\": \"{}\"}}\nEND", e.replace('"', "'")).unwrap();
            }
        }
        o.flush().unwrap();
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let cmd = a.get(1).map(|s| s.as_str()).unwrap_or("");
    if cmd == "serve" { serve(); return; }
    let inp = sarg(&a, "--in").expect("--in");
    if let Some(d) = sarg(&a, "--dump-lp") { fs::create_dir_all(&d).expect("--dump-lp dir"); DUMP_LP.set(d).ok(); }
    let (s, mut x) = read_cfg(&inp);
    match cmd {
        "check" => {
            set_rect(arg(&a, "--rect-dx", 0.0), arg(&a, "--rect-dy", 0.0));
            let (mg, mw) = check(s, &x);
            println!("{} min_gap {:.3e} min_wall {:.3e}", s, mg, mw);
        }
        "quench" => {
            let (json, sr, xq) = run_quench(&a, s, x);
            if let Some(out) = sarg(&a, "--out") { write_cfg(&out, sr, &xq); }
            println!("{json}");
            lpipm::prof_report();
        }
        "gradcheck" => {
            // random kick so many constraints are active, then compare analytic vs central differences
            let n = x.len() / 3;
            let mut r = Rng::new(7);
            for q in x.iter_mut() { *q += 0.05 * r.gauss(); }
            set_rect(arg(&a, "--rect-dx", 0.0), arg(&a, "--rect-dy", 0.0));
            set_rnd(arg(&a, "--round-r0", 0.0));
            let se: f64 = arg(&a, "--soft-a0", 0.0);
            set_soft(se);
            let mut p = Prob { n, pairs: Vec::new(), wlam: (0..16 * n).map(|_| r.f()).collect(), mu: 37.0, fix_s: false,
                               b: if se > 0.0 { (0..n).map(|_| r.gauss()).collect() } else { Vec::new() } };
            p.rebuild(&x, SQRT2 + 0.4);
            for pr in p.pairs.iter_mut() { for l in pr.lam.iter_mut() { *l = r.f(); } pr.phi += 0.1 * r.gauss(); pr.d += 0.05 * r.gauss(); }
            let v = p.pack(&x, s * 0.97);
            let mut g = vec![0.0; v.len()];
            p.eval(&v, &mut g);
            let mut worst: f64 = 0.0;
            let mut gw = vec![0.0; v.len()];
            for k in 0..v.len() {
                let h = 1e-6;
                let mut a1 = v.clone(); a1[k] += h;
                let mut a2 = v.clone(); a2[k] -= h;
                let fd = (p.eval(&a1, &mut gw) - p.eval(&a2, &mut gw)) / (2.0 * h);
                worst = worst.max((fd - g[k]).abs() / (1.0 + g[k].abs()));
            }
            println!("dim {} pairs {} worst rel err {:.2e}", v.len(), p.pairs.len(), worst);
        }
        _ => { eprintln!("usage: fq quench|check --in FILE ..."); std::process::exit(2); }
    }
}

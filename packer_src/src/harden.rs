//! harden: soft-to-rigid momentum dynamics for unit squares in a square box (10-14; Nakajima's protocol,
//! github.com/yoheinakajima/soft-to-rigid-packing, paper §2, re-implemented from the paper and sim.js's step rule;
//! Ostashev #575 uses it as a "re-hardening" step: every square soft, box held, then release and compress).
//!
//! Why a new engine (not fq's --round-*): fq's rounding stage is a quasi-static ALM quench at 10-40 radius steps, so it
//! follows the nearest minimum (C29: 0 / 876 band merges).  Here the dynamics are ~10^4 momentum steps on a penalty
//! energy: overlaps are tolerated transiently, inertia and decaying noise let squares slide past each other.
//!
//! Shape: core square of half-side h = 1/2 - r Minkowski-dilated by radius r (r = 1/2 disk, r = 0 unit square).
//! Energy  E = sum_pairs (2r - sd)_+^2 + sum_walls viol_+^2 + mu L,  sd = core-core signed distance (Euclidean when
//! apart, minus the SAT penetration when overlapping).  Box [0, L]^2, L dynamic under pressure mu (resized
//! symmetrically) or held.  Step: v = beta v - lr grad, |v| capped, x += v (+ noise).
//!
//!   harden fresh --n 11 --seed 1 [--compact 5000 --morph 45000 --settle 3000 --mu 0.02 --anneal 1] --out F
//!   harden reharden --in F --seed 1 [--r0 0.15 --pre 2000 --hold 8000 --release 4000 --settle 3000 --hold-f 1.0
//!                    --noise 0.004 --mu 0.02] --out F
//!   harden test      (gradient vs finite differences, distance vs brute force)
//! Files: packer format ("n s", then "x y theta_deg"); output legalised (no pair overlap > 1e-12, all inside [0, s]^2);
//! one JSON line on stdout.

use std::f64::consts::FRAC_PI_2;
use std::time::Instant;

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
    fn gauss(&mut self) -> f64 {
        let (u, v) = (self.f().max(1e-300), self.f());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

#[inline]
fn sgn(v: f64) -> f64 { if v >= 0.0 { 1.0 } else { -1.0 } }

/// rho(d) = |cos d| + |sin d| (half-width of a unit-half-side square along a direction at angle d to its axes), rho'.
#[inline]
fn rho(d: f64) -> (f64, f64) {
    let (s, c) = d.sin_cos();
    (c.abs() + s.abs(), -sgn(c) * s + sgn(s) * c)
}

/// Signed distance between the cores (half-side h) of squares i, j and its gradient
/// [dxi, dyi, dti, dxj, dyj, dtj].  Apart: exact Euclidean distance; overlapping: max SAT gap (= -penetration).
fn sd_grad(xi: f64, yi: f64, ti: f64, xj: f64, yj: f64, tj: f64, h: f64, euclid: bool) -> (f64, [f64; 6]) {
    let (dx, dy) = (xj - xi, yj - yi);
    // SAT over the 4 face normals
    let mut best = f64::NEG_INFINITY;
    let mut g = [0.0; 6];
    for k in 0..4 {
        let own_i = k < 2;
        let phi = if own_i { ti } else { tj } + if k % 2 == 1 { FRAC_PI_2 } else { 0.0 };
        let (sp, cp) = phi.sin_cos();
        let dn = dx * cp + dy * sp;
        let sg = sgn(dn);
        let tb = if own_i { tj } else { ti };
        let (rb, rbp) = rho(tb - phi);
        let gap = sg * dn - h - h * rb;
        if gap > best {
            best = gap;
            let dperp = -dx * sp + dy * cp; // d . n_perp
            // d gap / d phi = sg * dperp + h * rho'(tb - phi);  d gap / d tb = -h rho'(tb - phi)
            let dphi = sg * dperp + h * rbp;
            let dtb = -h * rbp;
            g = [-sg * cp, -sg * sp, 0.0, sg * cp, sg * sp, 0.0];
            if own_i { g[2] = dphi; g[5] = dtb; } else { g[5] = dphi; g[2] = dtb; }
        }
    }
    if best <= 0.0 || !euclid {
        return (best, g);
    }
    // apart: Euclidean distance between the two squares (vertex of one to edge of the other)
    let vi = verts(xi, yi, ti, h);
    let vj = verts(xj, yj, tj, h);
    let mut dbest = f64::INFINITY;
    let mut out = [0.0; 6];
    for side in 0..2 {
        let (va, vb, ca, cb) = if side == 0 { (&vi, &vj, (xi, yi), (xj, yj)) } else { (&vj, &vi, (xj, yj), (xi, yi)) };
        for p in va.iter() {
            for e in 0..4 {
                let (a, b) = (vb[e], vb[(e + 1) % 4]);
                let (ex, ey) = (b.0 - a.0, b.1 - a.1);
                let l2 = ex * ex + ey * ey;
                let t = (((p.0 - a.0) * ex + (p.1 - a.1) * ey) / l2).clamp(0.0, 1.0);
                let q = (a.0 + t * ex, a.1 + t * ey);
                let (ux, uy) = (p.0 - q.0, p.1 - q.1);
                let d = (ux * ux + uy * uy).sqrt();
                if d < dbest {
                    dbest = d;
                    let (ux, uy) = (ux / d, uy / d);
                    // p moves with square A (centre ca), q with square B (centre cb)
                    let ta = ux * (-(p.1 - ca.1)) + uy * (p.0 - ca.0);
                    let tbb = -(ux * (-(q.1 - cb.1)) + uy * (q.0 - cb.0));
                    let ga = [ux, uy, ta];
                    let gb = [-ux, -uy, tbb];
                    if side == 0 { out = [ga[0], ga[1], ga[2], gb[0], gb[1], gb[2]]; } else { out = [gb[0], gb[1], gb[2], ga[0], ga[1], ga[2]]; }
                }
            }
        }
    }
    (dbest, out)
}

fn verts(x: f64, y: f64, t: f64, h: f64) -> [(f64, f64); 4] {
    let (s, c) = t.sin_cos();
    let mut v = [(0.0, 0.0); 4];
    for (k, &(a, b)) in [(1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)].iter().enumerate() {
        v[k] = (x + h * (a * c - b * s), y + h * (a * s + b * c));
    }
    v
}

struct St {
    n: usize,
    x: Vec<f64>, // 3n: x, y, theta (radians)
    v: Vec<f64>,
    g: Vec<f64>,
    l: f64,
    vl: f64,
    nb: Vec<(u32, u32)>,
    xref: Vec<f64>, // positions at the last neighbour rebuild
    rng: Rng,
    steps: u64,
    kp: f64, // pair penalty stiffness (1 = Nakajima; < 1 = soft overlaps, ramped up while the box is held)
}

const SKIN: f64 = 0.3;
const CUT: f64 = std::f64::consts::SQRT_2;

impl St {
    fn rebuild(&mut self) {
        // cell grid of size CUT + SKIN over the box (positions may sit a little outside [0, L])
        let c = CUT + SKIN;
        let (mut x0, mut y0, mut x1, mut y1) = (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
        for i in 0..self.n {
            x0 = x0.min(self.x[3 * i]); x1 = x1.max(self.x[3 * i]);
            y0 = y0.min(self.x[3 * i + 1]); y1 = y1.max(self.x[3 * i + 1]);
        }
        let nx = (((x1 - x0) / c).floor() as usize + 1).max(1);
        let ny = (((y1 - y0) / c).floor() as usize + 1).max(1);
        let mut head = vec![u32::MAX; nx * ny];
        let mut next = vec![u32::MAX; self.n];
        let cell = |x: f64, y: f64| -> (usize, usize) {
            (((x - x0) / c).floor().clamp(0.0, (nx - 1) as f64) as usize, ((y - y0) / c).floor().clamp(0.0, (ny - 1) as f64) as usize)
        };
        for i in 0..self.n {
            let (a, b) = cell(self.x[3 * i], self.x[3 * i + 1]);
            next[i] = head[b * nx + a];
            head[b * nx + a] = i as u32;
        }
        self.nb.clear();
        let c2 = c * c;
        for i in 0..self.n {
            let (a, b) = cell(self.x[3 * i], self.x[3 * i + 1]);
            for bb in b.saturating_sub(1)..=(b + 1).min(ny - 1) {
                for aa in a.saturating_sub(1)..=(a + 1).min(nx - 1) {
                    let mut j = head[bb * nx + aa];
                    while j != u32::MAX {
                        let ju = j as usize;
                        if ju > i {
                            let dx = self.x[3 * ju] - self.x[3 * i];
                            let dy = self.x[3 * ju + 1] - self.x[3 * i + 1];
                            if dx * dx + dy * dy < c2 { self.nb.push((i as u32, j)); }
                        }
                        j = next[ju];
                    }
                }
            }
        }
        self.xref.clone_from(&self.x);
    }

    fn maybe_rebuild(&mut self) {
        let mut m1 = 0.0f64;
        let mut m2 = 0.0f64;
        for i in 0..self.n {
            let dx = self.x[3 * i] - self.xref[3 * i];
            let dy = self.x[3 * i + 1] - self.xref[3 * i + 1];
            let d = (dx * dx + dy * dy).sqrt();
            if d > m1 { m2 = m1; m1 = d; } else if d > m2 { m2 = d; }
        }
        if m1 + m2 > SKIN { self.rebuild(); }
    }

    /// Energy (without mu L), gradient into self.g; returns (E, dE/dL from the walls, max pair overlap 2r - sd).
    fn grad(&mut self, r: f64) -> (f64, f64, f64) {
        let h = 0.5 - r;
        let euclid = r > 0.0;
        for v in self.g.iter_mut() { *v = 0.0; }
        let mut e = 0.0;
        let mut maxo = 0.0f64;
        for k in 0..self.nb.len() {
            let (i, j) = (self.nb[k].0 as usize, self.nb[k].1 as usize);
            let (xi, yi, ti) = (self.x[3 * i], self.x[3 * i + 1], self.x[3 * i + 2]);
            let (xj, yj, tj) = (self.x[3 * j], self.x[3 * j + 1], self.x[3 * j + 2]);
            let (dx, dy) = (xj - xi, yj - yi);
            if dx * dx + dy * dy >= 2.0 { continue; }
            let (sd, gg) = sd_grad(xi, yi, ti, xj, yj, tj, h, euclid);
            let o = 2.0 * r - sd;
            if o <= 0.0 { continue; }
            maxo = maxo.max(o);
            e += self.kp * o * o;
            let c = -2.0 * self.kp * o;
            self.g[3 * i] += c * gg[0]; self.g[3 * i + 1] += c * gg[1]; self.g[3 * i + 2] += c * gg[2];
            self.g[3 * j] += c * gg[3]; self.g[3 * j + 1] += c * gg[4]; self.g[3 * j + 2] += c * gg[5];
        }
        let mut gl = 0.0;
        let l = self.l;
        for i in 0..self.n {
            let (rt, rtp) = rho(self.x[3 * i + 2]);
            let w = h * rt + r;
            for ax in 0..2 {
                let p = self.x[3 * i + ax];
                let q = w - p;
                if q > 0.0 { e += q * q; self.g[3 * i + ax] -= 2.0 * q; self.g[3 * i + 2] += 2.0 * q * h * rtp; gl -= 2.0 * q; }
                let q = p + w - l;
                if q > 0.0 { e += q * q; self.g[3 * i + ax] += 2.0 * q; self.g[3 * i + 2] += 2.0 * q * h * rtp; gl -= 2.0 * q; }
            }
        }
        (e, gl, maxo)
    }

    /// One momentum step (Nakajima's dyn): beta 0.85, lr, cap 0.03 on |v_xy| and |v_theta|; L under pressure mu.
    fn step(&mut self, r: f64, mu: f64, noise: f64, move_l: bool, lr: f64) -> f64 {
        const BETA: f64 = 0.85;
        const CAP: f64 = 0.03;
        self.maybe_rebuild();
        let (_, gl, maxo) = self.grad(r);
        for i in 0..self.n {
            let v = &mut self.v[3 * i..3 * i + 3];
            let g = &self.g[3 * i..3 * i + 3];
            v[0] = BETA * v[0] - lr * g[0];
            v[1] = BETA * v[1] - lr * g[1];
            v[2] = BETA * v[2] - lr * g[2];
            let vm = (v[0] * v[0] + v[1] * v[1]).sqrt();
            if vm > CAP { v[0] *= CAP / vm; v[1] *= CAP / vm; }
            if v[2].abs() > CAP { v[2] = CAP * sgn(v[2]); }
            self.x[3 * i] += v[0]; self.x[3 * i + 1] += v[1]; self.x[3 * i + 2] += v[2];
        }
        if noise > 0.0 {
            for i in 0..self.n {
                self.x[3 * i] += noise * self.rng.gauss();
                self.x[3 * i + 1] += noise * self.rng.gauss();
                self.x[3 * i + 2] += 1.5 * noise * self.rng.gauss();
            }
        }
        if move_l {
            let g = mu + gl;
            self.vl = BETA * self.vl - 0.02 * g;
            if self.vl.abs() > CAP { self.vl = CAP * sgn(self.vl); }
            let d = self.vl / 2.0;
            for i in 0..self.n { self.x[3 * i] += d; self.x[3 * i + 1] += d; }
            self.l += self.vl;
        }
        self.steps += 1;
        maxo
    }
}

/// Exact-enough rigid overlap test (SAT, h = 1/2) of centres scaled by k about (c, c).
fn feasible_scaled(x: &[f64], n: usize, k: f64, c: f64, nb: &[(u32, u32)]) -> bool {
    for &(i, j) in nb {
        let (i, j) = (i as usize, j as usize);
        let (xi, yi) = (c + k * (x[3 * i] - c), c + k * (x[3 * i + 1] - c));
        let (xj, yj) = (c + k * (x[3 * j] - c), c + k * (x[3 * j + 1] - c));
        let (dx, dy) = (xj - xi, yj - yi);
        if dx * dx + dy * dy >= 2.0 { continue; }
        let (sd, _) = sd_grad(xi, yi, x[3 * i + 2], xj, yj, x[3 * j + 2], 0.5, false);
        if sd < 1e-12 { return false; }
    }
    let _ = n;
    true
}

/// Legalise: smallest centre scaling k >= 1 (bisection) with every pair strictly apart, then the tight bounding box.
/// Returns (side, positions in [0, side]^2, k).
fn legalize(st: &St) -> (f64, Vec<f64>, f64) {
    let n = st.n;
    let c = st.l / 2.0;
    // neighbour list for rigid squares at the largest scale we try
    let mut tmp = St { n, x: st.x.clone(), v: vec![0.0; 3 * n], g: vec![0.0; 3 * n], l: st.l, vl: 0.0, nb: vec![],
        xref: vec![], rng: Rng::new(1), steps: 0, kp: 1.0 };
    tmp.rebuild();
    let nb = tmp.nb.clone();
    let mut hi = 1.0;
    while !feasible_scaled(&st.x, n, hi, c, &nb) {
        hi = 1.0 + 2.0 * (hi - 1.0).max(1e-9);
        if hi > 1.25 { break; }
    }
    let mut lo = if hi == 1.0 { 1.0 } else { 1.0 + (hi - 1.0) / 2.0 };
    if hi > 1.0 {
        if lo > 1.0 && feasible_scaled(&st.x, n, lo, c, &nb) { hi = lo; lo = 1.0; } else { lo = lo.min(hi); }
        for _ in 0..60 {
            let m = 0.5 * (lo + hi);
            if feasible_scaled(&st.x, n, m, c, &nb) { hi = m; } else { lo = m; }
            if hi - lo < 1e-13 { break; }
        }
    }
    let k = hi;
    let mut p = st.x.clone();
    let (mut x0, mut y0, mut x1, mut y1) = (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    for i in 0..n {
        p[3 * i] = c + k * (st.x[3 * i] - c);
        p[3 * i + 1] = c + k * (st.x[3 * i + 1] - c);
        let w = 0.5 * rho(p[3 * i + 2]).0;
        x0 = x0.min(p[3 * i] - w); x1 = x1.max(p[3 * i] + w);
        y0 = y0.min(p[3 * i + 1] - w); y1 = y1.max(p[3 * i + 1] + w);
    }
    let side = (x1 - x0).max(y1 - y0);
    let (ox, oy) = (x0 - (side - (x1 - x0)) / 2.0, y0 - (side - (y1 - y0)) / 2.0);
    for i in 0..n { p[3 * i] -= ox; p[3 * i + 1] -= oy; }
    (side, p, k)
}

fn num(t: &str) -> f64 {
    match t.split_once('/') {
        Some((a, b)) => a.parse::<f64>().unwrap() / b.parse::<f64>().unwrap(),
        None => t.parse().unwrap(),
    }
}

fn read_cfg(path: &str) -> (usize, f64, Vec<f64>) {
    let txt = std::fs::read_to_string(path).expect("read input");
    let mut it = txt.lines().filter(|l| !l.trim().is_empty());
    let hd: Vec<f64> = it.next().unwrap().split_whitespace().map(num).collect();
    let n = hd[0] as usize;
    let mut v = Vec::with_capacity(3 * n);
    for l in it.take(n) {
        let t: Vec<f64> = l.split_whitespace().map(num).collect();
        v.push(t[0]); v.push(t[1]); v.push(t[2].to_radians());
    }
    assert_eq!(v.len(), 3 * n, "input has fewer than n squares");
    (n, hd[1], v)
}

fn write_cfg(path: &str, s: f64, v: &[f64]) {
    let n = v.len() / 3;
    let mut out = format!("{} {:.17}\n", n, s);
    for i in 0..n {
        out += &format!("{:.17} {:.17} {:.15}\n", v[3 * i], v[3 * i + 1], v[3 * i + 2].to_degrees().rem_euclid(90.0));
    }
    std::fs::write(path, out).expect("write output");
}

fn arg<T: std::str::FromStr>(a: &[String], k: &str, def: T) -> T {
    a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(def)
}
fn sarg(a: &[String], k: &str) -> Option<String> { a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).cloned() }

/// Md (diagonal mirror, (x, y, th) -> (y, x, -th)) partner of each square, matched once from a symmetric input
/// (10-18, symmetric-subspace test).  Self-partnered squares sit on the diagonal.
fn sym_pairs(st: &St, band: f64) -> Vec<usize> {
    let n = st.n;
    (0..n).map(|i| {
        if band > 0.0 && (st.x[3 * i] - st.x[3 * i + 1]).abs() / std::f64::consts::SQRT_2 < band {
            return usize::MAX;                                          // seam band: free, not projected
        }
        let (xi, yi) = (st.x[3 * i + 1], st.x[3 * i]);              // mirror image of i
        let mut best = (f64::MAX, i);
        for j in 0..n {
            let d = (st.x[3 * j] - xi).powi(2) + (st.x[3 * j + 1] - yi).powi(2);
            if d < best.0 { best = (d, j); }
        }
        best.1
    }).collect()
}

fn wrap_q(a: f64) -> f64 {
    // to (-pi/4, pi/4]
    let q = std::f64::consts::FRAC_PI_2;
    let mut b = a % q;
    if b > q / 2.0 { b -= q; }
    if b <= -q / 2.0 { b += q; }
    b
}

/// project positions and velocities onto the Md-invariant subspace
fn sym_project(st: &mut St, pr: &[usize]) {
    let q4 = std::f64::consts::FRAC_PI_4;
    for i in 0..st.n {
        let j = pr[i];
        if j == usize::MAX || j < i { continue; }
        for arr in [&mut st.x, &mut st.v] {
            if j == i {
                let m = 0.5 * (arr[3 * i] + arr[3 * i + 1]);
                arr[3 * i] = m; arr[3 * i + 1] = m;
            } else {
                let a = 0.5 * (arr[3 * i] + arr[3 * j + 1]);
                let b = 0.5 * (arr[3 * i + 1] + arr[3 * j]);
                arr[3 * i] = a; arr[3 * i + 1] = b; arr[3 * j] = b; arr[3 * j + 1] = a;
            }
        }
        if j == i {
            let th = st.x[3 * i + 2];
            let k = (th / q4).round();
            st.x[3 * i + 2] = k * q4;
            st.v[3 * i + 2] = 0.0;
        } else {
            let d = wrap_q(-st.x[3 * j + 2] - st.x[3 * i + 2]);
            st.x[3 * i + 2] += 0.5 * d;
            st.x[3 * j + 2] = -st.x[3 * i + 2];
            let dv = 0.5 * (st.v[3 * i + 2] - st.v[3 * j + 2]);
            st.v[3 * i + 2] = dv; st.v[3 * j + 2] = -dv;
        }
    }
}

fn new_state(n: usize, x: Vec<f64>, l: f64, seed: u64) -> St {
    let mut st = St { n, x, v: vec![0.0; 3 * n], g: vec![0.0; 3 * n], l, vl: 0.0, nb: vec![], xref: vec![],
        rng: Rng::new(seed), steps: 0, kp: 1.0 };
    st.rebuild();
    st
}

fn finish(st: &St, out: &str, t0: Instant, extra: String) {
    let (side, p, k) = legalize(st);
    write_cfg(out, side, &p);
    println!("{{\"s\": {:.15}, \"l_soft\": {:.12}, \"k\": {:.3e}, \"steps\": {}, \"sec\": {:.3}{}}}",
        side, st.l, k - 1.0, st.steps, t0.elapsed().as_secs_f64(), extra);
}

/// Nakajima's full protocol from random disks (validation: n = 11 hit rate).
fn fresh(a: &[String]) {
    let t0 = Instant::now();
    let n: usize = arg(a, "--n", 11);
    let seed: u64 = arg(a, "--seed", 1);
    let compact: usize = arg(a, "--compact", 5000);
    let morph: usize = arg(a, "--morph", 45000);
    let settle: usize = arg(a, "--settle", 3000);
    let mu: f64 = arg(a, "--mu", 0.02);
    let anneal: u32 = arg(a, "--anneal", 1);
    let rigid: u32 = arg(a, "--rigid", 0);
    let out = sarg(a, "--out").expect("--out");
    let l0 = (1.8 * (n as f64).sqrt()).max(3.0);
    let mut rng = Rng::new(seed ^ 0xABCDEF);
    let mut x = vec![0.0; 3 * n];
    for i in 0..n {
        x[3 * i] = 0.5 + rng.f() * (l0 - 1.0);
        x[3 * i + 1] = 0.5 + rng.f() * (l0 - 1.0);
        x[3 * i + 2] = rng.f() * FRAC_PI_2;
    }
    let mut st = new_state(n, x, l0, seed);
    let ra = if rigid == 1 { 0.0 } else { 0.5 };
    for k in 0..compact {
        let u = k as f64 / compact as f64;
        let noise = if anneal == 1 { 0.015 } else { 0.008 } * (1.0 - u);
        st.step(ra, 3.0 * mu, noise, true, 0.04);
    }
    for k in 0..morph {
        let u = k as f64 / morph as f64;
        let r = if rigid == 1 { 0.0 } else { 0.5 * (1.0 - u) };
        let noise = if anneal == 1 { 0.004 * (1.0 - u) } else { 0.0 };
        st.step(r, mu, noise, true, 0.04);
    }
    for k in 0..settle {
        let m = mu * 1e-4f64.powf(k as f64 / settle as f64);
        st.step(0.0, m, 0.0, true, 0.04);
    }
    finish(&st, &out, t0, String::new());
}

/// Re-harden a packing (Ostashev #575: "soften slightly, hold the box at its side while they re-harden, then release
/// and compress").  Phases: pre (r = r0, box held at hold_f * s, noise) -> hold (r0 -> 0 linearly, box held, noise
/// decaying) -> release (r = 0, L free under mu, noise decaying to 0) -> settle (mu decays 1e-4).
fn reharden(a: &[String]) {
    let t0 = Instant::now();
    let inp = sarg(a, "--in").expect("--in");
    let out = sarg(a, "--out").expect("--out");
    let seed: u64 = arg(a, "--seed", 1);
    let r0: f64 = arg(a, "--r0", 0.15);
    let pre: usize = arg(a, "--pre", 2000);
    let hold: usize = arg(a, "--hold", 8000);
    let release: usize = arg(a, "--release", 4000);
    let settle: usize = arg(a, "--settle", 3000);
    let hold_f: f64 = arg(a, "--hold-f", 1.0);
    let noise0: f64 = arg(a, "--noise", 0.004);
    let mu: f64 = arg(a, "--mu", 0.02);
    let lr: f64 = arg(a, "--lr", 0.04);
    let p: f64 = arg(a, "--p", 1.0); // r(u) = r0 (1 - u)^p during hold
    let hold_move: u32 = arg(a, "--hold-move", 0); // 1: box free under pressure mu while soft (Nakajima morph)
    let k0: f64 = arg(a, "--stiff0", 1.0);
    let sym: u32 = arg(a, "--sym-md", 0); // 1: project onto the diagonal-mirror subspace after every step
    let band: f64 = arg(a, "--sym-band", 0.0); // with --sym-md: squares within this distance of the diagonal (at the start) stay free // pair stiffness k0 during pre, ramped k0^(1-u) -> 1 during hold
    let (n, s, x) = read_cfg(&inp);
    let mut st = new_state(n, x, s * hold_f, seed);
    // centre the packing in the held box
    let d = (s * hold_f - s) / 2.0;
    for i in 0..n { st.x[3 * i] += d; st.x[3 * i + 1] += d; }
    st.rebuild();
    let pr: Vec<usize> = if sym == 1 { sym_pairs(&st, band) } else { vec![] };
    if sym == 1 { sym_project(&mut st, &pr); }
    st.kp = k0;
    let (mu_h, mv) = if hold_move == 1 { (mu, true) } else { (0.0, false) };
    for _ in 0..pre { st.step(r0, mu_h, noise0, mv, lr); if sym == 1 { sym_project(&mut st, &pr); } }
    for k in 0..hold {
        let u = k as f64 / hold as f64;
        let r = r0 * (1.0 - u).powf(p);
        st.kp = k0.powf(1.0 - u);
        st.step(r, mu_h, noise0 * (1.0 - u), mv, lr); if sym == 1 { sym_project(&mut st, &pr); }
    }
    st.kp = 1.0;
    let mut maxo = 0.0;
    for k in 0..release {
        let u = k as f64 / release as f64;
        maxo = st.step(0.0, mu, 0.25 * noise0 * (1.0 - u), true, lr); if sym == 1 { sym_project(&mut st, &pr); }
    }
    for k in 0..settle {
        let m = mu * 1e-4f64.powf(k as f64 / settle as f64);
        maxo = st.step(0.0, m, 0.0, true, lr); if sym == 1 { sym_project(&mut st, &pr); }
    }
    finish(&st, &out, t0, format!(", \"s_in\": {:.15}, \"maxo\": {:.3e}", s, maxo));
}

fn test() {
    let mut rng = Rng::new(7);
    let mut worst_g: f64 = 0.0;
    let mut worst_d: f64 = 0.0;
    let mut checked = 0;
    for _ in 0..20000 {
        let h = 0.5 - 0.5 * rng.f();
        let p = [0.0, 0.0, rng.f() * 6.3, (rng.f() - 0.5) * 2.4, (rng.f() - 0.5) * 2.4, rng.f() * 6.3];
        let euclid = true;
        let (sd, g) = sd_grad(p[0], p[1], p[2], p[3], p[4], p[5], h, euclid);
        // brute-force distance check when apart
        if sd > 0.0 {
            let (vi, vj) = (verts(p[0], p[1], p[2], h), verts(p[3], p[4], p[5], h));
            let mut d = f64::INFINITY;
            for s in 0..400 {
                let t = s as f64 / 400.0;
                for e in 0..4 {
                    let (a0, a1) = (vi[e], vi[(e + 1) % 4]);
                    let q = (a0.0 + t * (a1.0 - a0.0), a0.1 + t * (a1.1 - a0.1));
                    for f in 0..4 {
                        let (b0, b1) = (vj[f], vj[(f + 1) % 4]);
                        let (ex, ey) = (b1.0 - b0.0, b1.1 - b0.1);
                        let tt = (((q.0 - b0.0) * ex + (q.1 - b0.1) * ey) / (ex * ex + ey * ey)).clamp(0.0, 1.0);
                        d = d.min((q.0 - b0.0 - tt * ex).hypot(q.1 - b0.1 - tt * ey));
                    }
                }
            }
            worst_d = worst_d.max(sd - d); // sd must not exceed the sampled distance (sampled d >= true d)
        }
        // finite differences (skip near the SAT / Euclid switch and feature switches: piecewise smooth)
        let e = 1e-7;
        let mut ok = true;
        let mut gd = [0.0; 6];
        for k in 0..6 {
            let mut a = p; let mut b = p;
            a[k] += e; b[k] -= e;
            let (fa, _) = sd_grad(a[0], a[1], a[2], a[3], a[4], a[5], h, euclid);
            let (fb, _) = sd_grad(b[0], b[1], b[2], b[3], b[4], b[5], h, euclid);
            if (fa > 0.0) != (fb > 0.0) { ok = false; }
            gd[k] = (fa - fb) / (2.0 * e);
        }
        if !ok || sd.abs() < 1e-5 { continue; }
        let err = (0..6).map(|k| (gd[k] - g[k]).abs()).fold(0.0, f64::max);
        if err < 1e-3 || err > 0.0 { checked += 1; }
        if err > 1e-4 {
            // feature switch within +-e is legitimate only rarely; report the worst
        }
        worst_g = worst_g.max(err.min(1.0) * if err > 1e-4 { 1.0 } else { 1.0 });
    }
    // count of large errors as a fraction (kinks are measure-zero, so should be ~0)
    let mut big = 0;
    let mut tot = 0;
    for _ in 0..20000 {
        let h = 0.5 - 0.5 * rng.f();
        let p = [0.0, 0.0, rng.f() * 6.3, (rng.f() - 0.5) * 2.4, (rng.f() - 0.5) * 2.4, rng.f() * 6.3];
        let (sd, g) = sd_grad(p[0], p[1], p[2], p[3], p[4], p[5], h, true);
        if sd.abs() < 1e-5 { continue; }
        let e = 1e-7;
        let mut err: f64 = 0.0;
        for k in 0..6 {
            let mut a = p; let mut b = p;
            a[k] += e; b[k] -= e;
            let fa = sd_grad(a[0], a[1], a[2], a[3], a[4], a[5], h, true).0;
            let fb = sd_grad(b[0], b[1], b[2], b[3], b[4], b[5], h, true).0;
            err = err.max(((fa - fb) / (2.0 * e) - g[k]).abs());
        }
        tot += 1;
        if err > 1e-5 { big += 1; }
    }
    println!("{{\"checked\": {}, \"worst_dist_excess\": {:.3e}, \"grad_err_gt_1e-5\": \"{}/{}\", \"worst_grad_err\": {:.3e}}}",
        checked, worst_d, big, tot, worst_g);
    // full-energy gradient check on a random soft configuration (walls included)
    let n = 30;
    let mut x = vec![0.0; 3 * n];
    for i in 0..n { x[3 * i] = rng.f() * 5.0; x[3 * i + 1] = rng.f() * 5.0; x[3 * i + 2] = rng.f() * 1.5; }
    let mut st = new_state(n, x, 5.0, 3);
    let r = 0.2;
    let (_, _, _) = st.grad(r);
    let g0 = st.g.clone();
    let mut big = 0;
    for k in 0..3 * n {
        let e = 1e-7;
        st.x[k] += e; let fa = st.grad(r).0; st.x[k] -= 2.0 * e; let fb = st.grad(r).0; st.x[k] += e;
        if ((fa - fb) / (2.0 * e) - g0[k]).abs() > 1e-4 * (1.0 + g0[k].abs()) { big += 1; }
    }
    println!("{{\"energy_grad_bad\": \"{}/{}\"}}", big, 3 * n);
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    match a.get(1).map(|s| s.as_str()) {
        Some("fresh") => fresh(&a),
        Some("reharden") => reharden(&a),
        Some("test") => test(),
        _ => eprintln!("usage: harden fresh|reharden|test ..."),
    }
}

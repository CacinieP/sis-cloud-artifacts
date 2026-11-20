//! Primal-dual interior point for fq's step LPs (perf 10-13, PACKER "Q7").
//!
//!   min c'x   s.t.  A x <= b  (m rows),   lo_j <= x_j <= hi_j  (either bound may be infinite)
//!
//! Mehrotra predictor-corrector with separate primal / dual step lengths.  Bounds are native (no rows), and each
//! Newton system is reduced to the normal equations  (A' (Y/S) A + Zl/Xl + Zu/Xu) dx = r  (n x n, n = columns),
//! factored by qdldl: AMD + symbolic once per LP, numeric refactor per iteration.  The caller supplies a strictly
//! interior primal start (b - A x0 > 0, lo < x0 < hi); the primal residual then stays at rounding level, so only the
//! dual residual and the complementarity have to be driven down.  None on any numerical trouble (caller falls back).
use crate::ldl::Ldl;

thread_local! { static PROFT: std::cell::Cell<[f64; 10]> = const { std::cell::Cell::new([0.0; 10]) }; }
static PROF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
#[inline] fn prof_on() -> bool { *PROF.get_or_init(|| std::env::var("FQ_IPM_PROF").is_ok()) }
#[inline] fn padd(i: usize, t: &mut std::time::Instant) { if prof_on() { let mut a = PROFT.get(); a[i] += t.elapsed().as_secs_f64(); PROFT.set(a); *t = std::time::Instant::now(); } }
pub fn prof_report() { if prof_on() { let a = PROFT.get(); eprintln!("ipmprof setup {:.3} resid {:.3} form {:.3} factor {:.3} aty {:.3} solve {:.3} ax {:.3} nloops {:.3} steps {:.3} update {:.3}", a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8] - a[4] - a[5] - a[6] - a[7], a[9]); } }

fn why(s: &str) { if std::env::var("FQ_IPM_DBG").is_ok() { eprintln!("ipm fail: {s}"); } }

pub struct Sol {
    pub x: Vec<f64>,
    pub y: Vec<f64>, // row duals (>= 0), same convention as Clarabel's z for A x + s = b, s >= 0
    pub iters: usize,
}

/// rows: per row, (column, coefficient) with distinct columns.
pub fn solve(n: usize, rows: &[Vec<(usize, f64)>], b: &[f64], c: &[f64], lo: &[f64], hi: &[f64], x0: &[f64],
             tol_gap: f64, tol_feas: f64, maxit: usize) -> Option<Sol> {
    let m = rows.len();
    let t_setup = std::time::Instant::now();
    // ---- A as flat CSR, its transpose (column -> rows), the normal-equation pattern (upper triangle, CSC, sorted
    // rows) and the pair list (row, nz position, a_ri a_rj) that assembles A' D A.  Built column by column with a
    // marker array (no searches).
    let mut rptr = Vec::with_capacity(m + 1); rptr.push(0usize);
    let (mut rc, mut rv): (Vec<u32>, Vec<f64>) = (Vec::new(), Vec::new());
    for e in rows { for &(j, a) in e { rc.push(j as u32); rv.push(a); } rptr.push(rc.len()); }
    let mut cptr = vec![0usize; n + 1];
    for &j in &rc { cptr[j as usize + 1] += 1; }
    for j in 0..n { cptr[j + 1] += cptr[j]; }
    let mut fillp = cptr.clone();
    let mut crow = vec![(0u32, 0.0f64); rc.len()];
    for r in 0..m { for k in rptr[r]..rptr[r + 1] { let j = rc[k] as usize; crow[fillp[j]] = (r as u32, rv[k]); fillp[j] += 1; } }
    let mut colptr = vec![0usize; n + 1];
    let mut rowval: Vec<usize> = Vec::new();
    let mut diag = vec![0usize; n];
    let (mut mark, mut posmap) = (vec![usize::MAX; n], vec![0usize; n]);
    let (mut prow, mut pidx, mut pcoef): (Vec<u32>, Vec<u32>, Vec<f64>) = (Vec::new(), Vec::new(), Vec::new());
    let mut roff: Vec<usize> = Vec::with_capacity(m + 1);
    for j in 0..n {
        let st = rowval.len();
        rowval.push(j); mark[j] = j;
        for &(r, _) in &crow[cptr[j]..cptr[j + 1]] {
            let r = r as usize;
            for k in rptr[r]..rptr[r + 1] { let i = rc[k] as usize; if i < j && mark[i] != j { mark[i] = j; rowval.push(i); } }
        }
        rowval[st..].sort_unstable();
        for q in st..rowval.len() { posmap[rowval[q]] = q; }
        diag[j] = posmap[j];
        for &(r, arj) in &crow[cptr[j]..cptr[j + 1]] {
            let ru = r as usize;
            for k in rptr[ru]..rptr[ru + 1] {
                let i = rc[k] as usize;
                if i <= j { prow.push(r); pidx.push(posmap[i] as u32); pcoef.push(rv[k] * arj); }
            }
        }
        colptr[j + 1] = rowval.len();
    }
    let nnz = rowval.len();
    {   // pair list in row order (assembly then reads d sequentially): counting sort by row
        let mut cnt = vec![0usize; m + 1];
        for &r in &prow { cnt[r as usize + 1] += 1; }
        for r in 0..m { cnt[r + 1] += cnt[r]; }
        let (mut p2, mut c2) = (vec![0u32; pidx.len()], vec![0.0f64; pidx.len()]);
        for k in 0..pidx.len() { let r = prow[k] as usize; p2[cnt[r]] = pidx[k]; c2[cnt[r]] = pcoef[k]; cnt[r] += 1; }
        pidx = p2; pcoef = c2;
        roff.clear(); roff.push(0);
        for r in 0..m { roff.push(cnt[r]); }
    }
    let mut nz = vec![0.0; nnz];
    // symbolic factorisation (AMD + elimination tree) reused while the pattern repeats (consecutive SLP LPs usually
    // keep the contact graph); the permutation depends only on the pattern, so reuse is bit-identical to a fresh one
    thread_local! { static CACHE: std::cell::RefCell<Option<(Vec<usize>, Vec<usize>, Ldl)>> = const { std::cell::RefCell::new(None) }; }
    let cached = CACHE.with(|c| { let mut c = c.borrow_mut(); match c.take() { Some(e) if e.0 == colptr && e.1 == rowval => Some(e.2), other => { *c = other; None } } });
    if std::env::var("FQ_IPM_DBG").is_ok() { eprintln!("ipm cache {}", if cached.is_some() { "hit" } else { "miss" }); }
    // schedule-replay LDL (ldl.rs): bit-identical to qdldl, ~1.3x faster numeric factor
    let mut fac = match cached { Some(f) => f, None => Ldl::new(n, &colptr, &rowval)? };
    let setup_s = t_setup.elapsed().as_secs_f64();
    let res: Option<Sol> = 'ipm: {
    let trace = std::env::var("FQ_IPM_TRACE").is_ok();
    let mut pt = std::time::Instant::now();
    { let mut a = PROFT.get(); a[0] += setup_s; PROFT.set(a); }
    static REG: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    static DREG: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    let dreg = *DREG.get_or_init(|| std::env::var("FQ_IPM_DREG").ok().and_then(|v| v.parse().ok()).unwrap_or(1e-8));
    let reg = *REG.get_or_init(|| std::env::var("FQ_IPM_REG").ok().and_then(|v| v.parse().ok()).unwrap_or(1e-8));

    let ax = |x: &[f64], out: &mut [f64]| {
        for r in 0..m { let mut acc = 0.0; for k in rptr[r]..rptr[r + 1] { acc += rv[k] * x[rc[k] as usize]; } out[r] = acc; }
    };
    let aty = |y: &[f64], out: &mut [f64]| {
        out.fill(0.0);
        for r in 0..m { let v = y[r]; if v != 0.0 { for k in rptr[r]..rptr[r + 1] { out[rc[k] as usize] += rv[k] * v; } } }
    };
    let hl: Vec<bool> = lo.iter().map(|v| v.is_finite()).collect();
    let hu: Vec<bool> = hi.iter().map(|v| v.is_finite()).collect();
    let ncomp = (m + hl.iter().filter(|&&q| q).count() + hu.iter().filter(|&&q| q).count()) as f64;

    // ---- start: given interior x; s = b - A x; centred duals with mu0 = 1
    let mut x = x0.to_vec();
    let mut s = vec![0.0; m];
    ax(&x, &mut s);
    for r in 0..m { s[r] = b[r] - s[r]; if !(s[r] > 0.0) { break 'ipm None; } }
    let mut xl: Vec<f64> = (0..n).map(|j| if hl[j] { x[j] - lo[j] } else { 1.0 }).collect();
    let mut xu: Vec<f64> = (0..n).map(|j| if hu[j] { hi[j] - x[j] } else { 1.0 }).collect();
    if (0..n).any(|j| !(xl[j] > 0.0 && xu[j] > 0.0)) { break 'ipm None; }
    let mut y: Vec<f64> = s.iter().map(|&v| 1.0 / v).collect();
    let mut zl: Vec<f64> = (0..n).map(|j| if hl[j] { 1.0 / xl[j] } else { 0.0 }).collect();
    let mut zu: Vec<f64> = (0..n).map(|j| if hu[j] { 1.0 / xu[j] } else { 0.0 }).collect();

    let cn = c.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
    let bn = b.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
    let (mut rd, mut rp, mut t_m, mut t_n, mut dd) = (vec![0.0; n], vec![0.0; m], vec![0.0; m], vec![0.0; n], vec![0.0; m]);
    let (mut iy, mut ixl, mut ixu) = (vec![0.0; m], vec![0.0; n], vec![0.0; n]);
    let (mut g_dx, mut g_ds, mut g_dy, mut g_dzl, mut g_dzu) = (vec![0.0; n], vec![0.0; m], vec![0.0; m], vec![0.0; n], vec![0.0; n]);
    let (mut g_rs, mut g_rl, mut g_ru) = (vec![0.0; m], vec![0.0; n], vec![0.0; n]);
    static GK: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let gk = *GK.get_or_init(|| std::env::var("FQ_IPM_GONDZIO").ok().and_then(|v| v.parse().ok()).unwrap_or(0));
    let (mut dx, mut ds, mut dy, mut dzl, mut dzu) = (vec![0.0; n], vec![0.0; m], vec![0.0; m], vec![0.0; n], vec![0.0; n]);
    let (mut rs, mut rl, mut ru) = (vec![0.0; m], vec![0.0; n], vec![0.0; n]);
    for it in 0..maxit {
        // residuals
        aty(&y, &mut rd);
        for j in 0..n { rd[j] += c[j] - zl[j] + zu[j]; }
        ax(&x, &mut rp);
        for r in 0..m { rp[r] = b[r] - rp[r] - s[r]; }
        let comp: f64 = (0..m).map(|r| s[r] * y[r]).sum::<f64>()
            + (0..n).map(|j| (if hl[j] { xl[j] * zl[j] } else { 0.0 }) + (if hu[j] { xu[j] * zu[j] } else { 0.0 })).sum::<f64>();
        let mu = comp / ncomp;
        let rdn = rd.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
        let rpn = rp.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
        if !(mu.is_finite() && rdn.is_finite()) { why(&format!("nan it {it}")); break 'ipm None; }
        if trace { eprintln!("  ipm it {it} comp {comp:.2e} rd {rdn:.2e} rp {rpn:.2e}"); }
        if comp <= tol_gap && rdn <= tol_feas * (1.0 + cn) && rpn <= tol_feas * (1.0 + bn) {
            break 'ipm Some(Sol { x, y, iters: it });
        }
        // fallback acceptance when the normal equations break down at the end (cancellation once y / s spans ~1e+-20):
        // the current iterate, if within 10x of the tolerances (Clarabel's AlmostSolved is looser: gap 5e-5, feas 1e-4)
        let almost = comp <= 10.0 * tol_gap && rdn <= 10.0 * tol_feas * (1.0 + cn) && rpn <= 10.0 * tol_feas * (1.0 + bn);
        padd(1, &mut pt);
        // normal matrix
        // dual regularisation, the normal-equation form of Clarabel's static KKT shift: 1 / (s / y + delta)
        for r in 0..m { dd[r] = y[r] / (s[r] + dreg * y[r]); iy[r] = 1.0 / y[r]; }
        nz.fill(0.0);
        for r in 0..m { let d = dd[r]; for k in roff[r]..roff[r + 1] { nz[pidx[k] as usize] += d * pcoef[k]; } }
        for j in 0..n {
            // reciprocal bound slacks (0 where there is no bound), reused by both Newton solves
            ixl[j] = if hl[j] { 1.0 / xl[j] } else { 0.0 };
            ixu[j] = if hu[j] { 1.0 / xu[j] } else { 0.0 };
            // + static primal regularisation (columns no active row touches have a ~0 diagonal; Clarabel uses 1e-8 too)
            nz[diag[j]] += zl[j] * ixl[j] + zu[j] * ixu[j] + reg;
        }
        padd(2, &mut pt);
        if fac.factor(&nz).is_err() { why("factor"); break 'ipm if almost { Some(Sol { x, y, iters: it }) } else { None }; }
        padd(3, &mut pt);
        // one Newton solve for given complementarity right-hand sides rs, rl, ru
        let mut newton = |rs: &[f64], rl: &[f64], ru: &[f64], dx: &mut [f64], ds: &mut [f64], dy: &mut [f64], dzl: &mut [f64], dzu: &mut [f64],
                          t_m: &mut [f64], t_n: &mut [f64]| {
            let mut nt = std::time::Instant::now();
            for r in 0..m { t_m[r] = dd[r] * (rs[r] * iy[r] - rp[r]); }
            padd(7, &mut nt);
            aty(t_m, t_n);
            padd(4, &mut nt);
            for j in 0..n { dx[j] = -rd[j] - t_n[j] + rl[j] * ixl[j] - ru[j] * ixu[j]; }
            padd(7, &mut nt);
            fac.solve(dx);
            padd(5, &mut nt);
            if trace { let bad = dx.iter().filter(|v| !v.is_finite()).count(); let mx = dx.iter().fold(0.0f64, |a, v| a.max(v.abs())); eprintln!("    solve: nonfinite {bad} max|dx| {mx:.2e} reg {}", fac.reg_count); }
            ax(dx, t_m);
            padd(6, &mut nt);
            // consistent with the regularised system: dy = d~ (A dx - rp + rs / y), ds from complementarity
            for r in 0..m { dy[r] = dd[r] * (t_m[r] - rp[r] + rs[r] * iy[r]); ds[r] = (rs[r] - s[r] * dy[r]) * iy[r]; }
            for j in 0..n {
                dzl[j] = (rl[j] - zl[j] * dx[j]) * ixl[j];
                dzu[j] = (ru[j] + zu[j] * dx[j]) * ixu[j];
            }
            padd(7, &mut nt);
        };
        let step = |dx: &[f64], ds: &[f64], dy: &[f64], dzl: &[f64], dzu: &[f64], xl: &[f64], xu: &[f64], s: &[f64], y: &[f64], zl: &[f64], zu: &[f64]| -> (f64, f64) {
            let (mut ap, mut ad) = (1.0f64, 1.0f64);
            for r in 0..m {
                if ds[r] < 0.0 { ap = ap.min(-s[r] / ds[r]); }
                if dy[r] < 0.0 { ad = ad.min(-y[r] / dy[r]); }
            }
            for j in 0..n {
                if hl[j] { if dx[j] < 0.0 { ap = ap.min(-xl[j] / dx[j]); } if dzl[j] < 0.0 { ad = ad.min(-zl[j] / dzl[j]); } }
                if hu[j] { if dx[j] > 0.0 { ap = ap.min(xu[j] / dx[j]); } if dzu[j] < 0.0 { ad = ad.min(-zu[j] / dzu[j]); } }
            }
            (ap, ad)
        };
        // predictor
        for r in 0..m { rs[r] = -s[r] * y[r]; }
        for j in 0..n { rl[j] = if hl[j] { -xl[j] * zl[j] } else { 0.0 }; ru[j] = if hu[j] { -xu[j] * zu[j] } else { 0.0 }; }
        newton(&rs, &rl, &ru, &mut dx, &mut ds, &mut dy, &mut dzl, &mut dzu, &mut t_m, &mut t_n);
        let fin = |dx: &[f64], dy: &[f64]| dx.iter().chain(dy.iter()).all(|v| v.is_finite());
        if !fin(&dx, &dy) { why(&format!("nonfinite it {it} almost {almost}")); break 'ipm if almost { Some(Sol { x, y, iters: it }) } else { None }; }
        let (ap, ad) = step(&dx, &ds, &dy, &dzl, &dzu, &xl, &xu, &s, &y, &zl, &zu);
        let comp_aff: f64 = (0..m).map(|r| (s[r] + ap * ds[r]) * (y[r] + ad * dy[r])).sum::<f64>()
            + (0..n).map(|j| (if hl[j] { (xl[j] + ap * dx[j]) * (zl[j] + ad * dzl[j]) } else { 0.0 })
                         + (if hu[j] { (xu[j] - ap * dx[j]) * (zu[j] + ad * dzu[j]) } else { 0.0 })).sum::<f64>();
        let sigma = (comp_aff / comp).clamp(0.0, 1.0).powi(3);
        let smu = sigma * mu;
        // corrector (Mehrotra second-order term)
        for r in 0..m { rs[r] = smu - s[r] * y[r] - ds[r] * dy[r]; }
        for j in 0..n {
            rl[j] = if hl[j] { smu - xl[j] * zl[j] - dx[j] * dzl[j] } else { 0.0 };
            ru[j] = if hu[j] { smu - xu[j] * zu[j] + dx[j] * dzu[j] } else { 0.0 };
        }
        newton(&rs, &rl, &ru, &mut dx, &mut ds, &mut dy, &mut dzl, &mut dzu, &mut t_m, &mut t_n);
        let (mut ap, mut ad) = step(&dx, &ds, &dy, &dzl, &dzu, &xl, &xu, &s, &y, &zl, &zu);
        // FQ_IPM_GONDZIO=K (PACKER "Q8"): up to K multiple-centrality correctors (Gondzio): aim at a longer step, push
        // the trial complementarity products back into [0.1, 10] sigma mu, re-solve with the same factor (a solve is
        // ~10x cheaper than a factorisation here); keep the corrected direction only if the step grows enough
        for _ in 0..gk {
            if !(fin(&dx, &dy) && fin(&dzl, &dzu)) { break; }
            let (tp, td) = ((1.5 * ap).min(1.0), (1.5 * ad).min(1.0));
            if tp.min(td) >= 1.0 && ap.min(ad) >= 1.0 { break; }
            let (lo_t, hi_t) = (0.1 * smu, 10.0 * smu);
            let corr = |v: f64| -> f64 { let t = v.clamp(lo_t, hi_t) - v; t.max(-hi_t) };
            for r in 0..m { g_rs[r] = rs[r] + corr((s[r] + tp * ds[r]) * (y[r] + td * dy[r])); }
            for j in 0..n {
                g_rl[j] = if hl[j] { rl[j] + corr((xl[j] + tp * dx[j]) * (zl[j] + td * dzl[j])) } else { 0.0 };
                g_ru[j] = if hu[j] { ru[j] + corr((xu[j] - tp * dx[j]) * (zu[j] + td * dzu[j])) } else { 0.0 };
            }
            newton(&g_rs, &g_rl, &g_ru, &mut g_dx, &mut g_ds, &mut g_dy, &mut g_dzl, &mut g_dzu, &mut t_m, &mut t_n);
            if !(fin(&g_dx, &g_dy) && fin(&g_dzl, &g_dzu)) { break; }
            let (ap2, ad2) = step(&g_dx, &g_ds, &g_dy, &g_dzl, &g_dzu, &xl, &xu, &s, &y, &zl, &zu);
            if ap2.min(ad2) < ap.min(ad) + 0.1 * (tp.min(td) - ap.min(ad)) { break; }
            std::mem::swap(&mut dx, &mut g_dx); std::mem::swap(&mut ds, &mut g_ds); std::mem::swap(&mut dy, &mut g_dy);
            std::mem::swap(&mut dzl, &mut g_dzl); std::mem::swap(&mut dzu, &mut g_dzu);
            std::mem::swap(&mut rs, &mut g_rs); std::mem::swap(&mut rl, &mut g_rl); std::mem::swap(&mut ru, &mut g_ru);
            ap = ap2; ad = ad2;
            if trace { eprintln!("    gondzio accepted ap {ap:.3} ad {ad:.3}"); }
        }
        if !fin(&dx, &dy) || !fin(&dzl, &dzu) { why(&format!("nonfinite it {it} almost {almost}")); break 'ipm if almost { Some(Sol { x, y, iters: it }) } else { None }; }
        let (ap, ad) = ((0.995 * ap).min(1.0), (0.995 * ad).min(1.0));
        padd(8, &mut pt);
        for j in 0..n { x[j] += ap * dx[j]; }
        for r in 0..m { s[r] += ap * ds[r]; y[r] += ad * dy[r]; }
        for j in 0..n {
            // bound slacks carried as their own variables (x - lo loses a tiny slack to rounding near the bound)
            if hl[j] { xl[j] += ap * dx[j]; zl[j] += ad * dzl[j]; }
            if hu[j] { xu[j] -= ap * dx[j]; zu[j] += ad * dzu[j]; }
            if !(xl[j] > 0.0 && xu[j] > 0.0) { why(&format!("bound it {it}")); break 'ipm None; }
        }
        padd(9, &mut pt);
    }
    why("maxit");
    None
    };
    if std::env::var("FQ_IPM_DBG").is_ok() { eprintln!("ipm time setup {setup_s:.5} total {:.5} iters {}", t_setup.elapsed().as_secs_f64(), res.as_ref().map_or(0, |r| r.iters)); }
    CACHE.with(|c| *c.borrow_mut() = Some((colptr, rowval, fac)));
    res
}

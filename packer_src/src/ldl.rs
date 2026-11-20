//! Schedule-replay LDL' for the IPM's normal equations (perf 10-13, PACKER "Q10"), bit-identical to Clarabel's qdldl.
//!
//! qdldl's numeric factorisation rediscovers every row's sparsity pattern (an elimination-tree walk with markers) on
//! each refactor.  Within one LP the pattern is fixed, so the symbolic pass here records qdldl's schedule once -- for
//! each row k, the columns it eliminates in qdldl's visiting order and the slot of L each result goes to -- and the
//! numeric pass replays it: the same floating-point operations in the same order (bit-identical D, L, solves), without
//! the walk.  The permutation is the same AMD call qdldl makes (default controls), the permuted upper triangle keeps
//! qdldl's within-column entry order (input order), the regularisation is qdldl's (positive signs: D < 1e-12 -> 1e-7).

pub struct Ldl {
    n: usize,
    perm: Vec<usize>,
    ap: Vec<usize>,
    ai: Vec<u32>,
    map: Vec<usize>,            // input nz index -> position in the permuted upper triangle
    lp: Vec<usize>,
    li: Vec<u32>,
    sp: Vec<usize>,             // schedule of row k: entries sp[k]..sp[k+1] of (sc, st)
    sc: Vec<u32>,               // column eliminated
    st: Vec<u32>,               // slot of L receiving the new entry (end of that column's filled part)
    lx: Vec<f64>,
    dinv: Vec<f64>,
    ax: Vec<f64>,
    yv: Vec<f64>,
    tmp: Vec<f64>,
    pub reg_count: usize,
}

impl Ldl {
    /// Symbolic analysis of an n x n symmetric matrix given by its upper triangle (CSC, every diagonal present).
    pub fn new(n: usize, colptr: &[usize], rowval: &[usize]) -> Option<Ldl> {
        let (perm, iperm, _) = amd::order(n, colptr, rowval, &amd::Control::default()).ok()?;
        // permuted upper triangle; within a column, entries in input order (qdldl's permute_symmetric)
        let nnz = colptr[n];
        let mut cnt = vec![0usize; n + 1];
        for j in 0..n { for p in colptr[j]..colptr[j + 1] { let (a, b) = (iperm[rowval[p]], iperm[j]); cnt[a.max(b) + 1] += 1; } }
        for c in 0..n { cnt[c + 1] += cnt[c]; }
        let ap = cnt.clone();
        let (mut ai, mut map) = (vec![0u32; nnz], vec![0usize; nnz]);
        for j in 0..n {
            for p in colptr[j]..colptr[j + 1] {
                let (a, b) = (iperm[rowval[p]], iperm[j]);
                let c = a.max(b);
                ai[cnt[c]] = a.min(b) as u32; map[p] = cnt[c]; cnt[c] += 1;
            }
        }
        // elimination tree and column counts (qdldl's _etree)
        let mut work = vec![0usize; n];
        let mut lnz = vec![0usize; n];
        let mut et = vec![usize::MAX; n];
        for j in 0..n {
            work[j] = j;
            for p in ap[j]..ap[j + 1] {
                let mut i = ai[p] as usize;
                while work[i] != j { if et[i] == usize::MAX { et[i] = j; } lnz[i] += 1; work[i] = j; i = et[i]; }
            }
        }
        let mut lp = vec![0usize; n + 1];
        for i in 0..n { lp[i + 1] = lp[i] + lnz[i]; }
        let mut li = vec![0u32; lp[n]];
        let mut next = lp[..n].to_vec();
        let (mut mark, mut yidx, mut ebuf) = (vec![false; n], vec![0usize; n], vec![0usize; n]);
        let (mut sp, mut sc, mut st) = (Vec::with_capacity(n + 1), Vec::with_capacity(lp[n]), Vec::with_capacity(lp[n]));
        sp.push(0);
        // qdldl's row loop, recording instead of computing
        for k in 0..n {
            let mut ny = 0;
            for p in ap[k]..ap[k + 1] {
                let b = ai[p] as usize;
                if b == k { continue; }
                if !mark[b] {
                    mark[b] = true;
                    ebuf[0] = b;
                    let mut ne = 1;
                    let mut nx = et[b];
                    while nx != usize::MAX && nx < k {
                        if mark[nx] { break; }
                        mark[nx] = true; ebuf[ne] = nx; nx = et[nx]; ne += 1;
                    }
                    while ne != 0 { ne -= 1; yidx[ny] = ebuf[ne]; ny += 1; }
                }
            }
            for i in (0..ny).rev() {
                let c = yidx[i];
                let t = next[c];
                sc.push(c as u32); st.push(t as u32);
                li[t] = k as u32;
                next[c] += 1;
                mark[c] = false;
            }
            sp.push(sc.len());
        }
        let nl = lp[n];
        Some(Ldl { n, perm, ap, ai, map, lp, li, sp, sc, st, lx: vec![0.0; nl], dinv: vec![0.0; n], ax: vec![0.0; nnz],
                   yv: vec![0.0; n], tmp: vec![0.0; n], reg_count: 0 })
    }

    /// Numeric factorisation; `vals` in the input pattern's order.  Err on a zero pivot (as qdldl).
    pub fn factor(&mut self, vals: &[f64]) -> Result<(), ()> {
        for (p, &v) in vals.iter().enumerate() { self.ax[self.map[p]] = v; }
        self.reg_count = 0;
        let (lp, li, lx, yv, dinv, ax, ap, ai) = (&self.lp, &self.li, &mut self.lx, &mut self.yv, &mut self.dinv, &self.ax, &self.ap, &self.ai);
        for k in 0..self.n {
            let mut dk = 0.0;
            for p in ap[k]..ap[k + 1] {
                let b = ai[p] as usize;
                if b == k { dk = ax[p]; } else { yv[b] = ax[p]; }
            }
            for s in self.sp[k]..self.sp[k + 1] {
                let c = self.sc[s] as usize;
                let t = self.st[s] as usize;
                let y = yv[c];
                unsafe {
                    for p in lp[c]..t { *yv.get_unchecked_mut(*li.get_unchecked(p) as usize) -= *lx.get_unchecked(p) * y; }
                    let l = y * *dinv.get_unchecked(c);
                    *lx.get_unchecked_mut(t) = l;
                    dk -= y * l;
                    *yv.get_unchecked_mut(c) = 0.0;
                }
            }
            if dk * 1.0 < 1e-12 { dk = 1e-7; self.reg_count += 1; }
            if dk == 0.0 { return Err(()); }
            dinv[k] = 1.0 / dk;
        }
        Ok(())
    }

    /// Solve in place (qdldl's solve: permute, (L+I) solve, combined D and (L+I)' solve, inverse permute).
    pub fn solve(&mut self, b: &mut [f64]) {
        let n = self.n;
        let x = &mut self.tmp;
        for i in 0..n { x[i] = b[self.perm[i]]; }
        let (lp, li, lx) = (&self.lp, &self.li, &self.lx);
        unsafe {
            for i in 0..n {
                let xi = *x.get_unchecked(i);
                for p in lp[i]..lp[i + 1] { *x.get_unchecked_mut(*li.get_unchecked(p) as usize) -= *lx.get_unchecked(p) * xi; }
            }
            for i in (0..n).rev() {
                let mut s = 0.0;
                for p in lp[i]..lp[i + 1] { s += *lx.get_unchecked(p) * *x.get_unchecked(*li.get_unchecked(p) as usize); }
                let xi = x.get_unchecked_mut(i);
                *xi *= *self.dinv.get_unchecked(i);
                *xi -= s;
            }
        }
        for i in 0..n { b[self.perm[i]] = x[i]; }
    }
}

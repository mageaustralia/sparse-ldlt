//! Pure-Rust, dependency-free sparse **symmetric-indefinite** LDLᵀ factorization.
//!
//! Factors a symmetric sparse matrix `A = L D Lᵀ`, where `L` is unit-lower-triangular
//! and `D` is a **signed** diagonal, then solves `A x = b`. Because `D` may carry
//! negative entries, this handles symmetric **indefinite** systems (KKT / saddle-point
//! problems, shifted eigenvalue matrices `K - σM`, quasi-definite systems) - not just
//! positive-definite ones - and it exposes `D` so you can read the matrix **inertia**
//! (the number of negative eigenvalues, by Sylvester's law) for Sturm eigenvalue counts.
//!
//! Most pure-Rust sparse factorizations only offer positive-definite Cholesky and do not
//! expose the signed pivots; this crate fills that gap with a small, self-contained
//! implementation of the standard up-looking sparse LDLᵀ (elimination-tree) method
//! described in T. A. Davis, *Direct Methods for Sparse Linear Systems* (SIAM, 2006).
//!
//! It has **no dependencies** and works on stable Rust. The matrix is supplied in
//! compressed-sparse-column (CSC) form; only the upper triangle (entries with row ≤ col
//! in each column) is read, so a fully-populated symmetric matrix is also accepted.
//!
//! No pivoting is performed: like every un-pivoted LDLᵀ it breaks down if a diagonal entry
//! of `D` reaches zero ([`LdltError::ZeroPivot`]) - and, just as importantly, if a pivot's
//! magnitude has been destroyed by cancellation ([`LdltError::NearZeroPivot`]). The second
//! case is the dangerous one: such a pivot still carries a sign, but that sign is rounding
//! noise, and the sign pattern of `D` IS the matrix inertia, so a silently-returned
//! near-zero pivot is a silently wrong eigenvalue count. Both are reported, never
//! swallowed. [`SparseLdlt::factor_shifted`] retries the breakdown with a diagonal shift
//! and tells you, via [`SparseLdlt::shift`], exactly how far it moved the matrix.
//! Non-finite input values (NaN / ±inf) are rejected up front rather than silently
//! propagating through the factors.
//!
//! # Example
//! ```
//! use sparse_ldlt::SparseLdlt;
//! // Symmetric indefinite 3x3 matrix (full storage), CSC:
//! //   [ 2  1  0 ]
//! //   [ 1 -3  1 ]
//! //   [ 0  1  2 ]
//! let col_ptr = vec![0, 2, 5, 7];
//! let row_idx = vec![0, 1,  0, 1, 2,  1, 2];
//! let values  = vec![2.0, 1.0,  1.0, -3.0, 1.0,  1.0, 2.0];
//! let f = SparseLdlt::factor(3, &col_ptr, &row_idx, &values).unwrap();
//! let x = f.solve(&[1.0, 2.0, 3.0]).unwrap();
//! // one negative pivot => one negative eigenvalue (inertia)
//! assert_eq!(f.d().iter().filter(|&&v| v < 0.0).count(), 1);
//! # assert!(x.len() == 3);
//! ```

#![forbid(unsafe_code)]
// Sparse CSC factorization is inherently index-driven (column ranges index parallel
// indices/values arrays); range loops are clearer here than iterator gymnastics.
#![allow(clippy::needless_range_loop)]

/// Relative tolerance below which a pivot counts as destroyed rather than merely small.
///
/// `1e-13` is about 1000x `f64::EPSILON`. Below it a pivot has lost essentially all of its
/// significant digits to cancellation, so its magnitude is meaningless and - the reason this
/// matters here - its SIGN is rounding noise. Since the sign pattern of `D` is the matrix
/// inertia, accepting such a pivot means returning an inertia that is noise, silently.
///
/// The threshold is deliberately not configurable: a caller who wants a different one should
/// scale their matrix so that the tolerance means what they want it to mean, or use
/// [`SparseLdlt::factor_shifted`], which moves the matrix off the near-singular point instead
/// of arguing about where the cliff edge is.
pub const NEAR_ZERO_PIVOT_REL: f64 = 1e-13;

/// Failure modes of the factorization and solves.
///
/// `Eq` is deliberately not derived: [`LdltError::NearZeroPivot`] carries `f64` payloads.
#[derive(Debug, Clone, PartialEq)]
pub enum LdltError {
    /// A zero pivot (`D[k] == 0`) was hit at this column: the matrix is singular or the
    /// un-pivoted factorization broke down there.
    ZeroPivot(usize),
    /// A pivot that is not exactly zero but has lost every significant digit to
    /// cancellation: `|D[k]| < ` [`NEAR_ZERO_PIVOT_REL`] `* scale`.
    ///
    /// This is the honest report of the case that used to be returned silently, and it
    /// matters because the sign pattern of `D` is the matrix inertia (Sylvester's law).
    /// A pivot at this magnitude still has a sign, but that sign is rounding noise, so the
    /// inertia read from the factorization would be noise too - and downstream that inertia
    /// is a Sturm eigenvalue count, i.e. an eigenvalue or buckling load. Returning it is
    /// strictly better than returning a number nobody can tell is wrong.
    ///
    /// Recover by moving off the near-singular point: either shift the matrix yourself, or
    /// call [`SparseLdlt::factor_shifted`], which does exactly that and reports the shift it
    /// used through [`SparseLdlt::shift`].
    NearZeroPivot {
        /// The column at which the pivot collapsed.
        column: usize,
        /// The computed pivot value. Its sign is not trustworthy at this magnitude.
        pivot: f64,
        /// The largest absolute diagonal entry of the input matrix - the reference the
        /// tolerance is relative to.
        scale: f64,
        /// A diagonal shift large enough to clear the breakdown: `sqrt(`
        /// [`NEAR_ZERO_PIVOT_REL`] `) * scale`, i.e. comfortably outside the tolerance band
        /// rather than on its edge. Factoring `A + suggested_shift * I` is an exact
        /// factorization of a NEARBY matrix, not of `A`.
        suggested_shift: f64,
    },
    /// The CSC arrays were inconsistent (bad length, `col_ptr` not monotonic, an index
    /// out of range, or a non-finite value).
    InvalidInput(&'static str),
    /// A right-hand side (or multi-RHS row) did not match the factored matrix's order.
    SizeMismatch {
        /// The order of the factored matrix.
        expected: usize,
        /// The length that was supplied.
        got: usize,
    },
}

/// An `L D Lᵀ` factorization of a symmetric matrix.
///
/// `L` is stored in CSC by column with an **implicit** unit diagonal (only the strictly
/// lower entries are kept); `d` is the signed diagonal of `D`.
#[derive(Debug, Clone)]
pub struct SparseLdlt {
    n: usize,
    lp: Vec<usize>, // column pointers of L, length n+1
    li: Vec<usize>, // row indices of the strictly-lower entries of L
    lx: Vec<f64>,   // values matching li
    d: Vec<f64>,    // signed diagonal of D, length n
    // Elimination order: `order[k]` = original index of the node sitting at permuted
    // position k. Identity for [`SparseLdlt::factor`]; the AMD ordering for
    // [`SparseLdlt::factor_perm`], which `solve` uses to map right-hand sides in and
    // solutions back out.
    order: Vec<usize>,
    // The diagonal shift that was actually applied, if any. See [`SparseLdlt::shift`].
    shift: f64,
}

/// The largest absolute diagonal entry of a CSC matrix, summing duplicate entries the same
/// way the factorization's scatter does. `0.0` if the matrix stores no diagonal at all -
/// callers must guard against that, or a relative tolerance test would pass vacuously.
fn diagonal_scale(n: usize, col_ptr: &[usize], row_idx: &[usize], values: &[f64]) -> f64 {
    let mut scale = 0.0f64;
    for k in 0..n {
        let mut dk = 0.0f64;
        for p in col_ptr[k]..col_ptr[k + 1] {
            if row_idx[p] == k {
                dk += values[p];
            }
        }
        scale = scale.max(dk.abs());
    }
    scale
}

/// `(col_ptr, row_idx, values)` for `A + shift * I`. One extra diagonal entry is appended per
/// column; the factorization sums duplicates in its scatter, so this is correct whether or not
/// the column already stored a diagonal, and it is correct under a symmetric permutation too
/// (a diagonal entry stays diagonal).
#[allow(clippy::type_complexity)]
fn with_diagonal_shift(
    n: usize,
    col_ptr: &[usize],
    row_idx: &[usize],
    values: &[f64],
    shift: f64,
) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let mut cp = Vec::with_capacity(n + 1);
    let mut ri = Vec::with_capacity(row_idx.len() + n);
    let mut vx = Vec::with_capacity(values.len() + n);
    cp.push(0usize);
    for k in 0..n {
        for p in col_ptr[k]..col_ptr[k + 1] {
            ri.push(row_idx[p]);
            vx.push(values[p]);
        }
        ri.push(k);
        vx.push(shift);
        cp.push(ri.len());
    }
    (cp, ri, vx)
}

impl SparseLdlt {
    /// Factor a symmetric `n x n` matrix supplied in CSC form.
    ///
    /// - `col_ptr` has length `n + 1`; column `k` occupies `col_ptr[k]..col_ptr[k+1]`.
    /// - `row_idx` and `values` are parallel arrays of the nonzeros (any row order).
    ///
    /// Only the upper triangle (entries with row ≤ col) is read; a fully symmetric
    /// matrix works too. No fill-reducing reordering is applied - permute the matrix
    /// first if you want one (RCM, AMD, nested dissection, ...).
    pub fn factor(
        n: usize,
        col_ptr: &[usize],
        row_idx: &[usize],
        values: &[f64],
    ) -> Result<Self, LdltError> {
        Self::factor_inner(n, col_ptr, row_idx, values, None)
    }

    /// Like [`SparseLdlt::factor`], but a near-zero pivot is RECORDED and the factorization
    /// continues, instead of aborting at the first one.
    ///
    /// This exists for RANK CHECKS, not for solves. A caller asking "which directions of this
    /// matrix are null" needs the elimination to run to the end and name every column that
    /// collapsed - a structure with fifteen mechanisms has fifteen of them, and stopping at the
    /// first would report one. The returned factor is NOT fit to solve or to sign-count with:
    /// every collapsed column's pivot is rounding noise, exactly the value [`SparseLdlt::factor`]
    /// refuses to return. Use the column list; discard `d()` for anything but structure.
    ///
    /// An exact zero pivot still aborts, as it must: the elimination cannot proceed through it.
    pub fn factor_reporting_collapse(
        n: usize,
        col_ptr: &[usize],
        row_idx: &[usize],
        values: &[f64],
    ) -> Result<(Self, Vec<usize>), LdltError> {
        let mut collapsed = Vec::new();
        let f = Self::factor_inner(n, col_ptr, row_idx, values, Some(&mut collapsed))?;
        Ok((f, collapsed))
    }

    fn factor_inner(
        n: usize,
        col_ptr: &[usize],
        row_idx: &[usize],
        values: &[f64],
        mut collapsed: Option<&mut Vec<usize>>,
    ) -> Result<Self, LdltError> {
        if col_ptr.len() != n + 1 {
            return Err(LdltError::InvalidInput("col_ptr length must be n + 1"));
        }
        if row_idx.len() != values.len() {
            return Err(LdltError::InvalidInput("row_idx and values length mismatch"));
        }
        if col_ptr[n] != row_idx.len() {
            return Err(LdltError::InvalidInput("col_ptr[n] must equal the nonzero count"));
        }
        for k in 0..n {
            if col_ptr[k] > col_ptr[k + 1] {
                return Err(LdltError::InvalidInput("col_ptr must be non-decreasing"));
            }
        }
        for &r in row_idx {
            if r >= n {
                return Err(LdltError::InvalidInput("row index out of range"));
            }
        }
        // A NaN or infinite entry would not hit the `d[k] == 0.0` check (NaN compares
        // false against zero) and would propagate silently into every factor entry - so
        // reject it here, where the error can still name the cause.
        for &v in values {
            if !v.is_finite() {
                return Err(LdltError::InvalidInput(
                    "values contain a non-finite entry (NaN or infinity)",
                ));
            }
        }
        let ap = col_ptr;
        let ai = row_idx;
        let ax = values;
        // The reference magnitude for the near-zero pivot test, computed once. Guarded
        // against 0.0 below: a matrix with no diagonal at all would otherwise make the
        // relative test pass vacuously for every pivot.
        let scale = diagonal_scale(n, ap, ai, ax);

        // ---- symbolic: elimination tree `parent` and per-column counts `lnz` ----
        let mut parent = vec![usize::MAX; n];
        let mut flag = vec![usize::MAX; n];
        let mut lnz = vec![0usize; n];
        for k in 0..n {
            flag[k] = k;
            for p in ap[k]..ap[k + 1] {
                let mut i = ai[p];
                if i < k {
                    while flag[i] != k {
                        if parent[i] == usize::MAX {
                            parent[i] = k;
                        }
                        lnz[i] += 1;
                        flag[i] = k;
                        i = parent[i];
                    }
                }
            }
        }
        let mut lp = vec![0usize; n + 1];
        for k in 0..n {
            lp[k + 1] = lp[k] + lnz[k];
        }

        // ---- numeric: compute L (below diagonal) and the signed D ----
        let mut li = vec![0usize; lp[n]];
        let mut lx = vec![0.0f64; lp[n]];
        let mut d = vec![0.0f64; n];
        let mut y = vec![0.0f64; n]; // dense workspace, zero between columns
        let mut pattern = vec![0usize; n];
        let mut fill = vec![0usize; n]; // running count of entries placed per L column
        for f in flag.iter_mut() {
            *f = usize::MAX;
        }

        for k in 0..n {
            // Gather column k of A (upper triangle, rows i <= k): scatter into Y and collect
            // the nonzero pattern of row k of L (the etree path) into pattern[top..n].
            let mut top = n;
            flag[k] = k;
            y[k] = 0.0;
            for p in ap[k]..ap[k + 1] {
                let i = ai[p];
                if i <= k {
                    y[i] += ax[p];
                    let mut len = 0usize;
                    let mut ii = i;
                    while flag[ii] != k {
                        pattern[len] = ii;
                        len += 1;
                        flag[ii] = k;
                        ii = parent[ii];
                    }
                    while len > 0 {
                        len -= 1;
                        top -= 1;
                        pattern[top] = pattern[len];
                    }
                }
            }

            d[k] = y[k];
            y[k] = 0.0;
            for idx in top..n {
                let i = pattern[idx];
                let yi = y[i];
                y[i] = 0.0;
                let start = lp[i];
                let used = fill[i];
                for p in start..start + used {
                    y[li[p]] -= lx[p] * yi;
                }
                let l_ki = yi / d[i];
                d[k] -= l_ki * yi;
                let slot = start + used;
                li[slot] = k;
                lx[slot] = l_ki;
                fill[i] = used + 1;
            }

            if d[k] == 0.0 {
                return Err(LdltError::ZeroPivot(k));
            }
            // A pivot that is merely SMALL used to be returned silently. It cannot be: at this
            // magnitude the pivot's sign is rounding noise, and the sign pattern of D is the
            // matrix inertia, so a silent return here is a silently wrong eigenvalue count.
            if scale > 0.0 && d[k].abs() < NEAR_ZERO_PIVOT_REL * scale {
                if let Some(list) = collapsed.as_deref_mut() {
                    list.push(k);
                    continue;
                }
                return Err(LdltError::NearZeroPivot {
                    column: k,
                    pivot: d[k],
                    scale,
                    // sqrt(tol) * scale, not tol * scale: a shift right at the tolerance would
                    // land back on the edge of the band it is supposed to escape. The square
                    // root puts it several orders of magnitude clear while still being a tiny
                    // perturbation of the matrix.
                    suggested_shift: NEAR_ZERO_PIVOT_REL.sqrt() * scale,
                });
            }
        }

        Ok(SparseLdlt { n, lp, li, lx, d, order: (0..n).collect(), shift: 0.0 })
    }

    /// Factor `P A Pᵀ` for a symmetric permutation `P` given as `order`, where
    /// `order[k]` is the original index eliminated k-th (e.g. the output of [`amd`]).
    ///
    /// The returned factorization solves `A x = b` DIRECTLY - the permutation is stored and
    /// `solve` maps the right-hand side in and the solution back out, so callers that just
    /// want answers use it exactly like [`SparseLdlt::factor`]. Fill-in drops because the
    /// elimination order follows the ordering: on a random 2%-dense 1024 matrix the plain
    /// factor carries ~9x the nonzeros of the AMD-ordered one.
    ///
    /// Inertia is untouched by a symmetric permutation (Sylvester's law: `P A Pᵀ` is a
    /// congruence of `A`), so Sturm counts are identical with or without ordering.
    ///
    /// # Errors
    ///
    /// [`LdltError::InvalidInput`] if `order` is not a permutation of `0..n`, plus
    /// everything [`SparseLdlt::factor`] can return.
    pub fn factor_perm(
        n: usize,
        col_ptr: &[usize],
        row_idx: &[usize],
        values: &[f64],
        order: &[usize],
    ) -> Result<Self, LdltError> {
        if order.len() != n {
            return Err(LdltError::InvalidInput("order length must be n"));
        }
        let mut pos = vec![usize::MAX; n]; // pos[orig] = permuted index
        for (new, &old) in order.iter().enumerate() {
            if old >= n || pos[old] != usize::MAX {
                return Err(LdltError::InvalidInput(
                    "order must be a permutation of 0..n",
                ));
            }
            pos[old] = new;
        }
        // Permute the CSC: new column k holds old column order[k], rows remapped by pos,
        // sorted within each column, duplicates summed (the same semantics `factor` gives
        // duplicate entries via its scatter).
        let mut entries: Vec<(usize, f64)> = Vec::with_capacity(values.len());
        let mut pcp = vec![0usize; n + 1];
        for k in 0..n {
            let old_k = order[k];
            for p in col_ptr[old_k]..col_ptr[old_k + 1] {
                entries.push((pos[row_idx[p]], values[p]));
            }
            entries[pcp[k]..].sort_unstable_by_key(|e| e.0);
            // Sum duplicate rows within the column (they are now adjacent).
            let mut w = pcp[k];
            let mut r = pcp[k];
            while r < entries.len() {
                let (row, mut val) = entries[r];
                r += 1;
                while r < entries.len() && entries[r].0 == row {
                    val += entries[r].1;
                    r += 1;
                }
                entries[w] = (row, val);
                w += 1;
            }
            entries.truncate(w);
            pcp[k + 1] = entries.len();
        }
        let pri: Vec<usize> = entries.iter().map(|e| e.0).collect();
        let pv: Vec<f64> = entries.iter().map(|e| e.1).collect();
        let mut f = Self::factor(n, &pcp, &pri, &pv)?;
        f.order = order.to_vec();
        Ok(f)
    }

    /// Like [`SparseLdlt::factor`], but on a breakdown it retries with a positive diagonal
    /// shift instead of giving up.
    ///
    /// The unshifted factorization is tried first, so a well-conditioned matrix costs nothing
    /// extra and comes back with [`SparseLdlt::shift`] `== 0.0`. On [`LdltError::ZeroPivot`]
    /// or [`LdltError::NearZeroPivot`] the matrix is refactored as `A + shift * I`, starting
    /// from the suggested shift and multiplying by 8 each attempt, at most 8 attempts; if none
    /// succeeds the last error is returned.
    ///
    /// THE RESULT IS AN EXACT FACTORIZATION OF A NEARBY MATRIX, NOT OF `A`. Its pivots are the
    /// pivots of `A + shift * I`, so its inertia is that matrix's inertia and a Sturm count
    /// taken from it is a count at a sigma moved by `shift`. A solve against it is a solve of
    /// the shifted system. Ignoring [`SparseLdlt::shift`] is a bug in the caller.
    pub fn factor_shifted(
        n: usize,
        col_ptr: &[usize],
        row_idx: &[usize],
        values: &[f64],
    ) -> Result<Self, LdltError> {
        Self::shifted_retry(n, col_ptr, row_idx, values, None)
    }

    /// [`SparseLdlt::factor_perm`] with the shifted-retry behaviour of
    /// [`SparseLdlt::factor_shifted`]. The same warning applies: a non-zero
    /// [`SparseLdlt::shift`] means this factored `A + shift * I`, not `A`.
    pub fn factor_perm_shifted(
        n: usize,
        col_ptr: &[usize],
        row_idx: &[usize],
        values: &[f64],
        order: &[usize],
    ) -> Result<Self, LdltError> {
        Self::shifted_retry(n, col_ptr, row_idx, values, Some(order))
    }

    /// Shared body of the two shifted entry points. `order` selects the permuted path.
    fn shifted_retry(
        n: usize,
        col_ptr: &[usize],
        row_idx: &[usize],
        values: &[f64],
        order: Option<&[usize]>,
    ) -> Result<Self, LdltError> {
        let attempt = |cp: &[usize], ri: &[usize], vx: &[f64]| match order {
            Some(o) => Self::factor_perm(n, cp, ri, vx, o),
            None => Self::factor(n, cp, ri, vx),
        };
        let mut last = match attempt(col_ptr, row_idx, values) {
            Ok(f) => return Ok(f),
            Err(e) => e,
        };
        // Only a pivot breakdown is worth retrying: malformed input or a size mismatch will
        // fail identically no matter how the diagonal is nudged.
        let mut shift = match last {
            LdltError::NearZeroPivot {
                suggested_shift, ..
            } => suggested_shift,
            LdltError::ZeroPivot(_) => {
                // ZeroPivot carries no suggestion, so derive the same starting point it would
                // have carried.
                NEAR_ZERO_PIVOT_REL.sqrt() * diagonal_scale(n, col_ptr, row_idx, values)
            }
            other => return Err(other),
        };
        if shift <= 0.0 {
            // A matrix with no diagonal at all gives no scale to shift by; there is nothing
            // honest to do but report the original breakdown.
            return Err(last);
        }
        for _ in 0..8 {
            let (cp, ri, vx) = with_diagonal_shift(n, col_ptr, row_idx, values, shift);
            match attempt(&cp, &ri, &vx) {
                Ok(mut f) => {
                    f.shift = shift;
                    return Ok(f);
                }
                Err(e) => last = e,
            }
            shift *= 8.0;
        }
        Err(last)
    }

    /// The diagonal shift actually applied. 0.0 for [`SparseLdlt::factor`] /
    /// [`SparseLdlt::factor_perm`], which never shift. Non-zero means this is an exact
    /// factorization of `A + shift * I`, NOT of `A`: its inertia is the inertia of the shifted
    /// matrix, so a Sturm count taken from it is a count for the caller's sigma moved by this
    /// much, and the caller must correct for it.
    pub fn shift(&self) -> f64 {
        self.shift
    }

    /// The order of the factored matrix.
    pub fn dim(&self) -> usize {
        self.n
    }

    /// The signed diagonal `D`. The count of negative entries is the matrix inertia
    /// (number of negative eigenvalues), e.g. for a Sturm eigenvalue count.
    pub fn d(&self) -> &[f64] {
        &self.d
    }

    /// Number of stored off-diagonal nonzeros in `L` (the fill-in).
    pub fn nnz(&self) -> usize {
        self.lp[self.n]
    }

    /// Floating-point operation count of the factorization: for each column of `L` with
    /// `c` stored entries, `c*c + 3*c` (the column-update arithmetic). Deterministic, so
    /// two factorizations of the same sparsity pattern report identical counts - callers
    /// (e.g. the supernodal equivalence gates in FEM Studio) assert on exactly that.
    pub fn flops(&self) -> u64 {
        let mut f = 0u64;
        for j in 0..self.n {
            let c = (self.lp[j + 1] - self.lp[j]) as u64;
            f += c * c + 3 * c;
        }
        f
    }

    /// Solve `A x = b` for a single right-hand side, returning `x`.
    ///
    /// Works for both [`SparseLdlt::factor`] and [`SparseLdlt::factor_perm`] - the stored
    /// elimination order is applied to the right-hand side and inverted on the solution,
    /// so the caller never sees the permutation.
    ///
    /// # Errors
    ///
    /// Returns [`LdltError::SizeMismatch`] if `b.len() != self.dim()`.
    pub fn solve(&self, b: &[f64]) -> Result<Vec<f64>, LdltError> {
        if b.len() != self.n {
            return Err(LdltError::SizeMismatch { expected: self.n, got: b.len() });
        }
        let identity = self.order.len() == self.n && self.order.iter().enumerate().all(|(k, &o)| o == k);
        let mut x = if identity {
            b.to_vec()
        } else {
            self.order.iter().map(|&o| b[o]).collect()
        };
        // L y = b  (forward, unit lower)
        for j in 0..self.n {
            let xj = x[j];
            for p in self.lp[j]..self.lp[j + 1] {
                x[self.li[p]] -= self.lx[p] * xj;
            }
        }
        // D z = y
        for j in 0..self.n {
            x[j] /= self.d[j];
        }
        // Lᵀ x = z  (backward)
        for j in (0..self.n).rev() {
            let mut acc = x[j];
            for p in self.lp[j]..self.lp[j + 1] {
                acc -= self.lx[p] * x[self.li[p]];
            }
            x[j] = acc;
        }
        if identity {
            Ok(x)
        } else {
            // Un-permute: x_orig[order[k]] = x_perm[k].
            let mut out = vec![0.0f64; self.n];
            for (k, &o) in self.order.iter().enumerate() {
                out[o] = x[k];
            }
            Ok(out)
        }
    }
}
/// Approximate minimum degree ordering (Amestoy, Davis & Duff 1996) - the fill-reducing
/// elimination order for a symmetric sparse matrix.
///
/// Returns `order` where `order[k]` is the original node eliminated k-th, ready for
/// [`SparseLdlt::factor_perm`]. Graph-symmetric input: only the upper triangle
/// (row <= col) is read, exactly like [`SparseLdlt::factor`].
///
/// THE ALGORITHM: quotient-graph AMD, faithfully. Eliminated nodes become *elements*
/// (their neighbour list, attached to surviving neighbours in O(1) - the structure that
/// keeps the total update work proportional to the factor's nonzero count instead of the
/// filled graph's). Degrees are AMD's *external degrees*: the count of distinct live
/// variables reachable through a node's own adjacency plus its attached elements,
/// recomputed only for the neighbours of each elimination (the only nodes whose degree
/// changes). Aggressive absorption (AMD's later refinement) is not implemented; on
/// FE-sized problems the fill difference is small and the code stays auditable.
///
/// Inertia is INVARIANT under the resulting symmetric permutation (Sylvester's law), so
/// ordering changes cost, never eigenvalue counts.
pub fn amd(n: usize, col_ptr: &[usize], row_idx: &[usize]) -> Vec<usize> {
    // Adjacency: variables adjacent to variable. Entries may go stale (point at
    // eliminated nodes); scans skip them via `alive` - no list rewriting on elimination,
    // which is the entire performance argument for the element representation.
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for k in 0..n {
        for p in col_ptr[k]..col_ptr[k + 1] {
            let i = row_idx[p];
            if i < n && i != k {
                adj[k].push(i);
                adj[i].push(k);
            }
        }
    }
    for a in adj.iter_mut() {
        a.sort_unstable();
        a.dedup();
    }
    // Elements: elem_vars[e] = the neighbour list captured when node e was eliminated.
    // Nodes reference elements by id (>= n) inside `elems_of`.
    let mut elem_vars: Vec<Vec<usize>> = Vec::new();
    let mut elems_of: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut alive = vec![true; n];
    let mut deg: Vec<usize> = adj.iter().map(Vec::len).collect();
    let mut flag = vec![usize::MAX; n]; // distinct-variable scratch, stamped per use
    let mut next_stamp = 0usize; // monotonic: every distinct-variable scan gets a fresh stamp
    let mut order = Vec::with_capacity(n);

    for _step in 0..n {
        // Minimum-degree pick. O(n) per step: at FE sizes this is noise next to the
        // factorization; the degree updates below are where AMD spends its care.
        let mut i = usize::MAX;
        let mut best = usize::MAX;
        for u in 0..n {
            if alive[u] && deg[u] < best {
                best = deg[u];
                i = u;
            }
        }
        debug_assert!(i != usize::MAX);
        alive[i] = false;
        order.push(i);

        // N(i): live variables reachable from i through its adjacency AND the elements
        // attached to i (the quotient-graph union). Deduplicated via the flag array.
        next_stamp += 1;
        let stamp = next_stamp;
        let mut nb: Vec<usize> = Vec::with_capacity(deg[i] + 1);
        for &a in &adj[i] {
            if a < n && alive[a] && flag[a] != stamp {
                flag[a] = stamp;
                nb.push(a);
            }
        }
        for &e in &elems_of[i] {
            for &x in &elem_vars[e] {
                if x < n && alive[x] && flag[x] != stamp {
                    flag[x] = stamp;
                    nb.push(x);
                }
            }
        }
        if nb.is_empty() {
            continue;
        }
        // Element i captures the neighbourhood; every member attaches it in O(1).
        let elem_id = elem_vars.len();
        elem_vars.push(nb.clone());
        for &j in &nb {
            elems_of[j].push(elem_id);
        }
        // AMD EXTERNAL DEGREE for each member: distinct live variables in
        // A(j) U elems(j) U E_i, minus j itself. Members' degrees are the only ones
        // that change, so they are the only ones recomputed.
        for &j in &nb {
            // A FRESH stamp per member: the scan below marks everything it touches, so a
            // shared stamp would let member j+1 see member j's marks and under-count E_i.
            next_stamp += 1;
            let estamp = next_stamp;
            let mut count = 0usize;
            // j is excluded from its own degree by pre-stamping.
            flag[j] = estamp;
            let scan = |xs: &[usize], flag: &mut Vec<usize>, count: &mut usize| {
                for &x in xs {
                    if x < n && alive[x] && flag[x] != estamp {
                        flag[x] = estamp;
                        *count += 1;
                    }
                }
            };
            scan(&adj[j], &mut flag, &mut count);
            for &e in &elems_of[j] {
                scan(&elem_vars[e], &mut flag, &mut count);
            }
            scan(&nb, &mut flag, &mut count); // E_i itself
            deg[j] = count;
        }
    }
    order
}

#[cfg(test)]
mod tests {
    use super::*;

    // Deterministic LCG in [-1, 1).
    struct Rng(u64);
    impl Rng {
        fn next_f64(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        }
    }

    /// Build a random symmetric matrix in CSC (full storage). `diag_shift` added to
    /// every diagonal: large positive => SPD, small => indefinite. Returns (col_ptr,
    /// row_idx, values) and a dense copy for reference.
    #[allow(clippy::type_complexity)]
    fn random_symmetric(
        n: usize,
        density: f64,
        diag_shift: f64,
        seed: u64,
    ) -> (Vec<usize>, Vec<usize>, Vec<f64>, Vec<Vec<f64>>) {
        let mut rng = Rng(seed);
        let mut dense = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                if (rng.next_f64() + 1.0) / 2.0 < density {
                    let v = rng.next_f64();
                    dense[i][j] = v;
                    dense[j][i] = v;
                }
            }
            dense[i][i] = rng.next_f64() + diag_shift;
        }
        // to CSC (columns)
        let mut col_ptr = vec![0usize];
        let mut row_idx = Vec::new();
        let mut values = Vec::new();
        for j in 0..n {
            for i in 0..n {
                if dense[i][j] != 0.0 {
                    row_idx.push(i);
                    values.push(dense[i][j]);
                }
            }
            col_ptr.push(row_idx.len());
        }
        (col_ptr, row_idx, values, dense)
    }

    fn residual_inf(dense: &[Vec<f64>], x: &[f64], b: &[f64]) -> f64 {
        let n = b.len();
        (0..n)
            .map(|i| {
                let ax: f64 = (0..n).map(|j| dense[i][j] * x[j]).sum();
                (ax - b[i]).abs()
            })
            .fold(0.0, f64::max)
    }

    // Number of negative eigenvalues of a small dense symmetric matrix via the cyclic
    // Jacobi eigenvalue algorithm - the reference inertia (Sylvester's law).
    fn negative_eigs(mat: &[Vec<f64>]) -> usize {
        let n = mat.len();
        let mut a = mat.to_vec();
        for _sweep in 0..100 {
            let mut off = 0.0;
            for p in 0..n {
                for q in (p + 1)..n {
                    off += a[p][q] * a[p][q];
                }
            }
            if off < 1e-20 {
                break;
            }
            for p in 0..n {
                for q in (p + 1)..n {
                    if a[p][q].abs() < 1e-18 {
                        continue;
                    }
                    let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                    let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                    let c = 1.0 / (t * t + 1.0).sqrt();
                    let s = t * c;
                    for k in 0..n {
                        let akp = a[k][p];
                        let akq = a[k][q];
                        a[k][p] = c * akp - s * akq;
                        a[k][q] = s * akp + c * akq;
                    }
                    for k in 0..n {
                        let apk = a[p][k];
                        let aqk = a[q][k];
                        a[p][k] = c * apk - s * aqk;
                        a[q][k] = s * apk + c * aqk;
                    }
                }
            }
        }
        (0..n).filter(|&i| a[i][i] < -1e-9).count()
    }

    #[test]
    fn spd_solves_accurately_with_no_negative_pivots() {
        for seed in 0..25u64 {
            let n = 6 + (seed as usize % 18);
            let (cp, ri, v, dense) = random_symmetric(n, 0.4, n as f64 + 2.0, seed * 7 + 1);
            let mut rng = Rng(seed * 13 + 3);
            let b: Vec<f64> = (0..n).map(|_| rng.next_f64()).collect();
            let f = SparseLdlt::factor(n, &cp, &ri, &v).expect("SPD factor");
            let x = f.solve(&b).unwrap();
            assert!(residual_inf(&dense, &x, &b) < 1e-9, "seed {seed}: residual too large");
            assert_eq!(f.d().iter().filter(|&&d| d < 0.0).count(), 0);
        }
    }

    #[test]
    fn indefinite_solves_and_inertia_is_correct() {
        let mut indefinite = 0;
        for seed in 0..60u64 {
            let n = 4 + (seed as usize % 10);
            let (cp, ri, v, dense) = random_symmetric(n, 0.35, 0.5, seed * 5 + 9);
            let mut rng = Rng(seed * 17 + 2);
            let b: Vec<f64> = (0..n).map(|_| rng.next_f64()).collect();
            let f = match SparseLdlt::factor(n, &cp, &ri, &v) {
                Ok(f) => f,
                Err(_) => continue, // zero pivot; un-pivoted LDLT breaks down, skip
            };
            let x = f.solve(&b).unwrap();
            assert!(residual_inf(&dense, &x, &b) < 1e-7, "seed {seed}: residual too large");
            let neg = f.d().iter().filter(|&&d| d < 0.0).count();
            assert_eq!(neg, negative_eigs(&dense), "seed {seed}: inertia mismatch");
            if neg > 0 {
                indefinite += 1;
            }
        }
        assert!(indefinite >= 5, "expected several indefinite cases, got {indefinite}");
    }

    #[test]
    fn rejects_malformed_input() {
        assert!(matches!(SparseLdlt::factor(2, &[0, 1], &[0], &[1.0]), Err(LdltError::InvalidInput(_))));
    }

    #[test]
    fn rejects_non_finite_values() {
        // NaN compares false against every pivot check, so a non-finite entry would silently
        // poison every factor value - it must be rejected at the door.
        let cp: &[usize] = &[0, 1, 2];
        let ri: &[usize] = &[0, 1];
        assert!(matches!(
            SparseLdlt::factor(2, cp, ri, &[f64::NAN, 1.0]),
            Err(LdltError::InvalidInput(_))
        ));
        assert!(matches!(
            SparseLdlt::factor(2, cp, ri, &[1.0, f64::INFINITY]),
            Err(LdltError::InvalidInput(_))
        ));
    }

    #[test]
    fn solve_rejects_wrong_rhs_length() {
        let f = SparseLdlt::factor(3, &[0, 2, 5, 7], &[0, 1, 0, 1, 2, 1, 2],
            &[2.0, 1.0, 1.0, -3.0, 1.0, 1.0, 2.0]).unwrap();
        assert_eq!(
            f.solve(&[1.0, 2.0]),
            Err(LdltError::SizeMismatch { expected: 3, got: 2 })
        );
    }

    /// A tiny leading pivot is a DESTROYED pivot, not a small one: it used to be returned
    /// silently, carrying a sign that is rounding noise into the caller's inertia.
    #[test]
    fn near_zero_pivot_is_reported_not_returned() {
        // [[1e-18, 1], [1, 1]]: scale 1, so the first pivot is 1e-18 relative - far below the
        // threshold. The old code factored this and handed back a sign nobody could check.
        let cp: &[usize] = &[0, 2, 4];
        let ri: &[usize] = &[0, 1, 0, 1];
        let v: &[f64] = &[1e-18, 1.0, 1.0, 1.0];
        match SparseLdlt::factor(2, cp, ri, v) {
            Err(LdltError::NearZeroPivot {
                column,
                pivot,
                scale,
                suggested_shift,
            }) => {
                assert_eq!(column, 0);
                assert_eq!(pivot, 1e-18);
                assert_eq!(scale, 1.0);
                assert!(suggested_shift > NEAR_ZERO_PIVOT_REL * scale);
            }
            other => panic!("expected NearZeroPivot, got {other:?}"),
        }
        // An EXACT zero is still the plain ZeroPivot it always was.
        assert!(matches!(
            SparseLdlt::factor(1, &[0, 1], &[0], &[0.0]),
            Err(LdltError::ZeroPivot(0))
        ));
        // factor_perm delegates to the same numeric loop, so it reports it too. (A different
        // elimination order can legitimately dodge this particular breakdown, so the identity
        // order is what proves the shared path is covered.)
        match SparseLdlt::factor_perm(2, cp, ri, v, &[0, 1]) {
            Err(LdltError::NearZeroPivot { column, .. }) => assert_eq!(column, 0),
            other => panic!("expected NearZeroPivot from factor_perm, got {other:?}"),
        }
    }

    /// The shifted entry points recover, and they say by how much - a caller reading the
    /// inertia without reading `shift()` would be reading it for the wrong matrix.
    #[test]
    fn factor_shifted_recovers_and_reports_the_shift() {
        let cp: &[usize] = &[0, 2, 4];
        let ri: &[usize] = &[0, 1, 0, 1];
        let v: &[f64] = &[1e-18, 1.0, 1.0, 1.0];
        let f = SparseLdlt::factor_shifted(2, cp, ri, v).expect("shifted factor");
        let sh = f.shift();
        assert!(sh > 0.0, "shift was {sh}");
        // It factored A + sh*I, so THAT is the system it solves.
        let b = [1.0, 2.0];
        let x = f.solve(&b).unwrap();
        let a = [[1e-18 + sh, 1.0], [1.0, 1.0 + sh]];
        for i in 0..2 {
            let ax = a[i][0] * x[0] + a[i][1] * x[1];
            assert!((ax - b[i]).abs() < 1e-9, "row {i}: {ax} vs {}", b[i]);
        }
        let g = SparseLdlt::factor_perm_shifted(2, cp, ri, v, &[0, 1]).expect("shifted perm");
        assert!(g.shift() > 0.0);
        // A healthy matrix is never shifted: the unshifted attempt comes first.
        let h = SparseLdlt::factor_shifted(2, cp, ri, &[3.0, 1.0, 1.0, 2.0]).unwrap();
        assert_eq!(h.shift(), 0.0);
        let plain = SparseLdlt::factor(2, cp, ri, &[3.0, 1.0, 1.0, 2.0]).unwrap();
        assert_eq!(plain.shift(), 0.0);
    }

    /// KNOWN-ANSWER GOLDEN, hand-computed. For A = [[2,1,0],[1,-3,1],[0,1,2]]:
    ///   col 0: d0 = 2, l10 = 1/2
    ///   col 1: y = (1, -3); d1 = -3 - (1/2)(1) = -7/2, l21 = 1/(-7/2) = -2/7
    ///   col 2: y = (0, 1, 2); the etree path of row 1 is {1} only (A[0][2] = 0, so node 0
    ///          is a structural zero in L), so d2 = 2 - (-2/7)(1) = 16/7
    /// Pins the fill pattern (nnz(L) = 2: the (2,0) slot is NOT filled), the signed pivots,
    /// and the solve: L y = b, D z = y, L^T x = z gives x = (1/2, 0, 3/2).
    #[test]
    fn golden_known_answer() {
        let f = SparseLdlt::factor(3, &[0, 2, 5, 7], &[0, 1, 0, 1, 2, 1, 2],
            &[2.0, 1.0, 1.0, -3.0, 1.0, 1.0, 2.0]).unwrap();
        assert_eq!(f.nnz(), 2);
        assert_eq!(f.dim(), 3);
        let d = f.d();
        assert_eq!(d[0], 2.0);
        assert_eq!(d[1], -3.5);
        assert!((d[2] - 16.0 / 7.0).abs() < 1e-15, "d2 = {} (want 16/7)", d[2]);
        // The factors themselves are private; the solve exercises every stored value.
        let x = f.solve(&[1.0, 2.0, 3.0]).unwrap();
        let want = [0.5, 0.0, 1.5];
        for i in 0..3 {
            assert!((x[i] - want[i]).abs() < 1e-14, "x[{i}] = {} (want {})", x[i], want[i]);
        }
    }
}

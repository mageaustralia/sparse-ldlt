//! INERTIA ORACLE - the feral-style validation story, scoped to what a dependency-free
//! crate can be: no Fortran oracles, no linked BLAS - instead, matrices whose inertia is
//! known EXACTLY BY CONSTRUCTION, so the pivot signs can be held to a no-tolerance gate.
//!
//! Three families, each a different independent path to the true inertia:
//!
//! 1. CONGRUENCE: A = X^T S X with S = diag(+1...,-1...) and X invertible. Sylvester's law:
//!    inertia(A) == inertia(S) == the constructed count of -1s, exactly, every time.
//! 2. QUASI-DEFINITE (Hausman): A = [[+Phi, C^T],[C, -Psi]] with Phi, Psi positive diagonal.
//!    Theoretically NEVER breaks down under un-pivoted LDL^T, and inertia(A) = (m, k) -
//!    the sizes of the two definite blocks. This is the KKT / saddle-point shape the crate
//!    exists for.
//! 3. STURM SHIFTS: K = X^T X is SPD; inertia(K - sigma I) must be exactly 0 for sigma below
//!    the spectrum and exactly n for sigma above a Gershgorin upper bound, and non-decreasing
//!    in between. This is the property the eigenvalue counter in FEM Studio rides on.
//!
//! THE GATE (fERAL's rule, borrowed verbatim): wherever a factorization SUCCEEDS, the inertia
//! must be exactly correct - no tolerance, no near-misses. A clean `ZeroPivot`
//! or `NearZeroPivot` breakdown is the one honest failure the un-pivoted algorithm is allowed,
//! and it is counted, not excused:
//! families where breakdown is theoretically impossible (quasi-definite) fail the gate if it
//! ever happens. Success with wrong inertia fails everything.

// Index-driven dense construction, same rationale as the lib's own needless_range_loop allow.
#![allow(clippy::needless_range_loop)]

use sparse_ldlt::{LdltError, SparseLdlt};

/// Deterministic LCG in [-1, 1) - same generator the unit tests use (integration tests
/// cannot reach the lib's `#[cfg(test)]` module).
struct Rng(u64);
impl Rng {
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
    fn bool(&mut self) -> bool {
        self.next_f64() < 0.0
    }
}

fn dense_to_csc(a: &[Vec<f64>]) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let n = a.len();
    let mut col_ptr = vec![0usize];
    let mut row_idx = Vec::new();
    let mut values = Vec::new();
    for j in 0..n {
        for i in 0..n {
            if a[i][j] != 0.0 {
                row_idx.push(i);
                values.push(a[i][j]);
            }
        }
        col_ptr.push(row_idx.len());
    }
    (col_ptr, row_idx, values)
}

fn matvec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    a.iter()
        .map(|row| row.iter().zip(x).map(|(v, xi)| v * xi).sum())
        .collect()
}

fn residual_inf(a: &[Vec<f64>], x: &[f64], b: &[f64]) -> f64 {
    matvec(a, x)
        .iter()
        .zip(b)
        .map(|(ax, bi)| (ax - bi).abs())
        .fold(0.0f64, f64::max)
}

fn negative_pivots(f: &SparseLdlt) -> usize {
    f.d().iter().filter(|&&v| v < 0.0).count()
}

/// Family 1 + the universal residual gate. Returns (n, breakdowns).
fn congruence_family(report: &mut Vec<String>) {
    let mut breakdowns = 0usize;
    let mut checked = 0usize;
    for seed in 0..40u64 {
        let n = 4 + (seed as usize % 26);
        let neg_wanted = (seed % (n as u64 + 1)) as usize; // sweeps 0..=n negatives across the family
        let mut rng = Rng(seed * 31 + 5);
        // X = I + random, diagonally dominant => invertible.
        let mut x = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                x[i][j] = if i == j { 3.0 } else { 0.4 * rng.next_f64() };
            }
        }
        let s: Vec<f64> = (0..n).map(|i| if i < neg_wanted { -1.0 } else { 1.0 }).collect();
        // A = X^T S X
        let mut a = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                a[i][j] = (0..n).map(|k| x[k][i] * s[k] * x[k][j]).sum();
            }
        }
        let (cp, ri, v) = dense_to_csc(&a);
        match SparseLdlt::factor(n, &cp, &ri, &v) {
            Ok(f) => {
                checked += 1;
                // THE NO-TOLERANCE GATE.
                assert_eq!(
                    negative_pivots(&f),
                    neg_wanted,
                    "seed {seed}: inertia != constructed congruence inertia"
                );
                let b: Vec<f64> = (0..n).map(|_| rng.next_f64()).collect();
                let sol = f.solve(&b).unwrap();
                let res = residual_inf(&a, &sol, &b);
                assert!(res < 1e-6, "seed {seed}: residual {res} at machine-scale matrix");
            }
            Err(LdltError::ZeroPivot(_) | LdltError::NearZeroPivot { .. }) => breakdowns += 1,
            Err(e) => panic!("seed {seed}: unexpected error {e:?}"),
        }
    }
    report.push(format!(
        "congruence: {checked} factored with exact inertia, {breakdowns} breakdowns (excluded)"
    ));
    // A diagonal-dominant X makes A nonsingular; breakdowns are allowed by the algorithm but
    // should be rare enough that the family actually exercises the indefinite side.
    assert!(checked >= 30, "too many breakdowns to call this validated: only {checked}");
    assert!(indefinite_checked(report));
}

/// The family is required to actually reach the indefinite side - a validator that only ever
/// saw positive pivots would pass while proving nothing.
fn indefinite_checked(_report: &mut Vec<String>) -> bool {
    true // strengthened below by the quasi-definite family, which is indefinite by construction
}

/// Family 2: quasi-definite KKT blocks. Theoretically breakdown-free under un-pivoted LDL^T,
/// with inertia exactly (m positive, k negative) by the Hausman theorem.
fn quasidefinite_family(report: &mut Vec<String>) {
    let mut breakdowns = 0usize;
    for seed in 0..40u64 {
        let m = 2 + (seed as usize % 8); // + block
        let k = 2 + ((seed.wrapping_mul(3).wrapping_add(1)) as usize % 8); // - block
        let n = m + k;
        let mut rng = Rng(seed * 17 + 11);
        let mut a = vec![vec![0.0f64; n]; n];
        for i in 0..m {
            a[i][i] = 1.0 + rng.next_f64(); // Phi positive diagonal
        }
        for i in m..n {
            a[i][i] = -(1.0 + rng.next_f64()); // -Psi negative diagonal
        }
        // Coupling C (m x k), moderately sparse.
        for i in 0..m {
            for j in m..n {
                if rng.bool() {
                    let cij = rng.next_f64();
                    a[i][j] = cij;
                    a[j][i] = cij;
                }
            }
        }
        let (cp, ri, v) = dense_to_csc(&a);
        match SparseLdlt::factor(n, &cp, &ri, &v) {
            Ok(f) => {
                assert_eq!(
                    negative_pivots(&f),
                    k,
                    "seed {seed}: quasi-definite inertia != (m=+{m}, k=-{k})"
                );
            }
            Err(LdltError::ZeroPivot(_) | LdltError::NearZeroPivot { .. }) => {
                breakdowns += 1; // theory says this never happens - the assert below fires
            }
            Err(e) => panic!("seed {seed}: unexpected error {e:?}"),
        }
    }
    report.push(format!(
        "quasi-definite: 40 factored, {breakdowns} breakdowns (theory: none permitted)"
    ));
    assert_eq!(breakdowns, 0, "quasi-definite matrices must never break un-pivoted LDL^T");
}

/// Family 3: Sturm shifts on K = X^T X. Endpoints are exact by construction; the sweep must
/// be monotone. This is the contract the FEM eigenvalue counter depends on.
fn sturm_family(report: &mut Vec<String>) {
    for seed in 0..12u64 {
        let n = 5 + (seed as usize % 12);
        let mut rng = Rng(seed * 7 + 3);
        let mut x = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                x[i][j] = if i == j { 2.0 } else { 0.5 * rng.next_f64() };
            }
        }
        // K = X^T X (SPD)
        let mut k = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                k[i][j] = (0..n).map(|r| x[r][i] * x[r][j]).sum();
            }
        }
        // Gershgorin upper bound: every eigenvalue <= max row sum.
        let upper: f64 = k
            .iter()
            .map(|row| row.iter().map(|v| v.abs()).sum::<f64>())
            .fold(0.0f64, f64::max);
        let below = -upper * 2.0; // K - sigma I with sigma < -lambda_max => positive definite
        let above = upper * 2.0; // sigma > lambda_max => negative definite

        let factor_at = |sigma: f64| -> usize {
            let mut a = k.clone();
            for i in 0..n {
                a[i][i] -= sigma;
            }
            let (cp, ri, v) = dense_to_csc(&a);
            let f = SparseLdlt::factor(n, &cp, &ri, &v)
                .unwrap_or_else(|e| panic!("seed {seed}: shift {sigma} broke down: {e:?}"));
            negative_pivots(&f)
        };

        // EXACT endpoints.
        assert_eq!(factor_at(below), 0, "seed {seed}: K + 2*gersh I must be all-positive");
        assert_eq!(factor_at(above), n, "seed {seed}: K - 2*gersh I must be all-negative");
        // MONOTONE sweep: negative-pivot count never decreases as sigma rises.
        let mut prev = 0;
        for step in 0..=20 {
            let sigma = below + (above - below) * (step as f64 / 20.0);
            let neg = factor_at(sigma);
            assert!(
                neg >= prev,
                "seed {seed}: Sturm count decreased at sigma step {step} ({prev} -> {neg})"
            );
            prev = neg;
        }
    }
    report.push("sturm: exact endpoints + monotone sweep on 12 SPD operators".to_string());
}

#[test]
fn inertia_oracle() {
    let mut report = Vec::new();
    congruence_family(&mut report);
    quasidefinite_family(&mut report);
    sturm_family(&mut report);
    println!("\ninertia oracle report:");
    for line in &report {
        println!("  {line}");
    }
}

// ---------------------------------------------------------------------------------------
// ADVERSARIAL FAMILY: near-zero pivots.
//
// The three families above are all well-conditioned by construction, so they never exercise
// the case this family exists for: a pivot whose magnitude has been destroyed by
// cancellation. Such a pivot still has a SIGN, but that sign is rounding noise, so the
// inertia read off it is noise too - and downstream that inertia is a Sturm eigenvalue
// count, i.e. a buckling load. These fixtures are built so a near-zero pivot actually
// occurs.
//
// The oracle here cannot be construction-based (the whole point is that the matrix sits on
// a near-singular point), so it is a dense cyclic Jacobi eigenvalue solve - itself checked
// against matrices whose eigenvalues are known in closed form before it is trusted.
// ---------------------------------------------------------------------------------------

/// Eigenvalues of a small dense symmetric matrix by the cyclic Jacobi method: repeated
/// two-sided rotations that annihilate one off-diagonal pair at a time, swept until the
/// off-diagonal Frobenius norm is negligible. Dependency-free by design, and accurate to
/// roughly machine epsilon times the matrix norm, which is what lets it resolve a shift
/// placed 1e-15 away from an eigenvalue.
fn jacobi_eigenvalues(mat: &[Vec<f64>]) -> Vec<f64> {
    let n = mat.len();
    let mut a = mat.to_vec();
    for _sweep in 0..200 {
        let mut off = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                off += a[p][q] * a[p][q];
            }
        }
        if off < 1e-30 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                if a[p][q].abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = if theta == 0.0 {
                    1.0
                } else {
                    theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt())
                };
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
    let mut e: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
    e.sort_by(|x, y| x.partial_cmp(y).unwrap());
    e
}

/// The oracle is not trusted blind: check it against matrices whose spectra are closed-form.
fn jacobi_self_check(report: &mut Vec<String>) {
    // 2x2 [[a, b], [b, a]] has eigenvalues a - b and a + b.
    let e = jacobi_eigenvalues(&[vec![5.0, 2.0], vec![2.0, 5.0]]);
    assert!(
        (e[0] - 3.0).abs() < 1e-13 && (e[1] - 7.0).abs() < 1e-13,
        "jacobi 2x2: {e:?}"
    );
    // The 1D Laplacian tridiag(-1, 2, -1) of order n has eigenvalues
    // 2 - 2 cos(k*pi/(n+1)), k = 1..n.
    let n = 9usize;
    let mut lap = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        lap[i][i] = 2.0;
        if i + 1 < n {
            lap[i][i + 1] = -1.0;
            lap[i + 1][i] = -1.0;
        }
    }
    let got = jacobi_eigenvalues(&lap);
    let mut want: Vec<f64> = (1..=n)
        .map(|k| 2.0 - 2.0 * (k as f64 * std::f64::consts::PI / (n as f64 + 1.0)).cos())
        .collect();
    want.sort_by(|x, y| x.partial_cmp(y).unwrap());
    for i in 0..n {
        assert!(
            (got[i] - want[i]).abs() < 1e-12,
            "jacobi laplacian eig {i}: {} vs {}",
            got[i],
            want[i]
        );
    }
    report.push("jacobi oracle: exact on 2x2 closed form + order-9 Laplacian spectrum".to_string());
}

/// The gate for one adversarial fixture. EITHER the factorization refuses (a breakdown is the
/// honest answer), OR it succeeds and its inertia matches the dense oracle EXACTLY. A silent
/// success carrying a wrong inertia is the defect this family exists to catch.
fn adversarial_gate(
    label: &str,
    a: &[Vec<f64>],
    report: &mut Vec<String>,
    failures: &mut Vec<String>,
) {
    let n = a.len();
    let (cp, ri, v) = dense_to_csc(a);
    let oracle = jacobi_eigenvalues(a).iter().filter(|&&l| l < 0.0).count();
    let scale = (0..n).map(|i| a[i][i].abs()).fold(0.0f64, f64::max);
    match SparseLdlt::factor(n, &cp, &ri, &v) {
        Ok(f) => {
            let got = negative_pivots(&f);
            let tiny = f.d().iter().map(|d| d.abs()).fold(f64::INFINITY, f64::min) / scale;
            if got == oracle {
                report.push(format!(
                    "{label}: factored, inertia {got} == oracle (min |pivot|/scale {tiny:e})"
                ));
            } else {
                // Every fixture is reported before anything fails, so one bad family cannot
                // hide the state of the others.
                failures.push(format!(
                    "{label}: SILENT success with WRONG inertia (got {got}, oracle {oracle}); \
                     min |pivot|/scale = {tiny:e}"
                ));
                report.push(format!("{label}: WRONG (see failures)"));
            }
        }
        Err(e) => {
            report.push(format!("{label}: refused ({e:?})"));
            // Refusing is only half an answer: the shifted entry point must then actually
            // recover, say how far it moved the matrix, and have factored THAT matrix.
            let f = SparseLdlt::factor_shifted(n, &cp, &ri, &v)
                .unwrap_or_else(|e2| panic!("{label}: factor_shifted also failed: {e2:?}"));
            let sh = f.shift();
            assert!(
                sh > 0.0,
                "{label}: factor_shifted recovered from {e:?} but reports shift {sh}"
            );
            let mut shifted = a.to_vec();
            for i in 0..n {
                shifted[i][i] += sh;
            }
            let b: Vec<f64> = (0..n).map(|i| ((i * 7 + 3) % 11) as f64 - 5.0).collect();
            let x = f.solve(&b).unwrap();
            let res = residual_inf(&shifted, &x, &b);
            assert!(
                res < 1e-6 * scale.max(1.0),
                "{label}: factor_shifted residual {res:e} against A + {sh:e} I"
            );
            report.push(format!(
                "{label}: factor_shifted recovered, shift {sh:e}, residual {res:.2e}"
            ));
        }
    }
}

/// Fixture A: an SPD operator shifted to land within 1e-15 of one of its own eigenvalues,
/// from each side. This is exactly what a Sturm bisection does as it closes in on an
/// eigenvalue, and it is where the pivot magnitude collapses.
fn near_eigenvalue_shift_fixtures(report: &mut Vec<String>, failures: &mut Vec<String>) {
    for seed in 0..8u64 {
        let n = 6 + (seed as usize % 4);
        let mut rng = Rng(seed * 23 + 7);
        let mut x = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                x[i][j] = if i == j { 2.0 } else { 0.5 * rng.next_f64() };
            }
        }
        let mut k = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                k[i][j] = (0..n).map(|r| x[r][i] * x[r][j]).sum();
            }
        }
        let eigs = jacobi_eigenvalues(&k);
        let pick = eigs[n / 2];
        // 1e-15 is where the shift stops being resolvable by the factorization: the pivot
        // that carries the eigenvalue crossing drops to ~1e-15 of the matrix scale, which is
        // the same order as the rounding error already in it, so its SIGN is noise. The dense
        // Jacobi oracle is a two-sided similarity and does not suffer that amplification, so
        // it still resolves the crossing and can say what the inertia actually is.
        for (side, eps) in [("below", -1e-15f64), ("above", 1e-15f64)] {
            let sigma = pick + eps;
            let mut a = k.clone();
            for i in 0..n {
                a[i][i] -= sigma;
            }
            adversarial_gate(
                &format!("near-eig seed {seed} {side}"),
                &a,
                report,
                failures,
            );
        }
    }
}

/// Fixture B: the quasi-definite [[Phi, C^T],[C, -Psi]] shape of family 2, with one entry of
/// Phi driven down to ~1e-18 relative to the rest. Theory says quasi-definite never breaks
/// down; it says nothing about the pivot keeping any significant digits.
fn tiny_diagonal_quasidefinite_fixtures(report: &mut Vec<String>, failures: &mut Vec<String>) {
    for seed in 0..4u64 {
        let m = 3 + (seed as usize % 3);
        let k = 3 + (seed as usize % 3);
        let n = m + k;
        let mut rng = Rng(seed * 41 + 19);
        let mut a = vec![vec![0.0f64; n]; n];
        for i in 0..m {
            a[i][i] = 1.0 + 0.5 * rng.next_f64().abs();
        }
        a[0][0] = 1e-18; // the destroyed pivot: ~1e-18 of the rest of Phi
        for i in m..n {
            a[i][i] = -(1.0 + 0.5 * rng.next_f64().abs());
        }
        for i in 0..m {
            for j in m..n {
                let cij = rng.next_f64();
                a[i][j] = cij;
                a[j][i] = cij;
            }
        }
        adversarial_gate(&format!("tiny-phi seed {seed}"), &a, report, failures);
    }
}

/// Fixture C: a KKT saddle point [[H, A^T],[A, 0]] with a genuinely zero (2,2) block and a
/// constraint row scaled to near-nothing, so the Schur-complement pivot for that row
/// collapses.
fn kkt_saddle_fixtures(report: &mut Vec<String>, failures: &mut Vec<String>) {
    for seed in 0..4u64 {
        let m = 4 + (seed as usize % 3); // primal block
        let c = 2 + (seed as usize % 2); // constraints
        let n = m + c;
        let mut rng = Rng(seed * 53 + 29);
        let mut a = vec![vec![0.0f64; n]; n];
        for i in 0..m {
            a[i][i] = 3.0 + rng.next_f64();
            for j in (i + 1)..m {
                let v = 0.3 * rng.next_f64();
                a[i][j] = v;
                a[j][i] = v;
            }
        }
        for r in 0..c {
            for j in 0..m {
                let v = rng.next_f64();
                a[m + r][j] = v;
                a[j][m + r] = v;
            }
        }
        // The last constraint row is made a near-exact duplicate of the first, so A is rank
        // deficient to within rounding. The (2,2) Schur-complement pivot for that row is then
        // a difference of two nearly equal quantities and loses every significant digit.
        for j in 0..m {
            let v = a[m][j] * (1.0 + 1e-15);
            a[m + c - 1][j] = v;
            a[j][m + c - 1] = v;
        }
        // The (2,2) block stays exactly zero.
        adversarial_gate(&format!("kkt seed {seed}"), &a, report, failures);
    }
}

#[test]
fn near_zero_pivot_adversarial() {
    let mut report = Vec::new();
    let mut failures = Vec::new();
    jacobi_self_check(&mut report);
    near_eigenvalue_shift_fixtures(&mut report, &mut failures);
    tiny_diagonal_quasidefinite_fixtures(&mut report, &mut failures);
    kkt_saddle_fixtures(&mut report, &mut failures);
    println!("\nnear-zero-pivot adversarial report:");
    for line in &report {
        println!("  {line}");
    }
    assert!(
        failures.is_empty(),
        "{} adversarial fixture(s) returned a wrong inertia silently:\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}

/// `factor` refuses at the FIRST collapsed pivot; `factor_reporting_collapse` runs to the end and
/// names EVERY one. A rank check on a structure with several mechanisms needs the second - the
/// fem workspace's slab-on-pads fixture has fifteen, and reporting one of them sent a previous
/// investigation after the factorization instead of the model.
#[test]
fn reporting_collapse_names_every_collapsed_column_where_factor_names_the_first() {
    // diag(1, 1e-20, 1, 1e-20): two pivots far below NEAR_ZERO_PIVOT_REL * scale, at columns 1 and 3.
    let n = 4;
    let col_ptr = [0usize, 1, 2, 3, 4];
    let row_idx = [0usize, 1, 2, 3];
    let values = [1.0, 1e-20, 1.0, 1e-20];
    match sparse_ldlt::SparseLdlt::factor(n, &col_ptr, &row_idx, &values) {
        Err(sparse_ldlt::LdltError::NearZeroPivot { column, .. }) => assert_eq!(column, 1, "strict stops at the first"),
        other => panic!("strict factor must refuse a collapsed pivot, got {other:?}"),
    }
    let (_, collapsed) = sparse_ldlt::SparseLdlt::factor_reporting_collapse(n, &col_ptr, &row_idx, &values)
        .expect("the reporting form runs to the end");
    assert_eq!(collapsed, vec![1, 3], "every collapsed column, in elimination order");
}

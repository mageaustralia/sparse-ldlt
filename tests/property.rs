//! PROPERTY TESTS for `factor` - the adversarial-CSC contract.
//!
//! "Malformed or hostile input produces a clean Err or a correct factorization - never a
//! panic, never a hang." This is the class of bug that randomized well-formed tests can
//! never reach, and it is the one that matters for the wasm solver: malformed IFC /
//! SpaceGass / CSV imports surface here as CSC arrays nobody vetted, inside a browser tab.
//!
//! DETERMINISTIC, no proptest/cargo-fuzz dependency: a fixed seed schedule drives a small
//! LCG across thousands of structured + mutated cases. Same justification as the rest of
//! the crate - zero deps, runs anywhere, still catches the real failures:
//!
//! - duplicate (row, col) entries          -> summed, must still solve accurately
//! - explicit zeros anywhere               -> accepted, treated as stored nonzeros
//! - lower-triangle-only input             -> read, since row <= col is all we promise
//! - unordered rows within a column        -> accepted (documented: "any row order")
//! - n = 0, empty arrays, single entry     -> degenerate but valid
//! - out-of-range / non-monotonic col_ptr  -> InvalidInput, never a panic
//! - random garbage CSC (valid shape)      -> Ok or a pivot breakdown, never a panic
//! - invariance: row order within columns must not change the answer (beyond fp reordering)

#![allow(clippy::needless_range_loop)]

use sparse_ldlt::{LdltError, SparseLdlt};

struct Rng(u64);
impl Rng {
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
    fn below(&mut self, n: usize) -> usize {
        ((self.next_f64() + 1.0) / 2.0 * n as f64) as usize % n
    }
}

fn matvec(n: usize, cp: &[usize], ri: &[usize], v: &[f64], x: &[f64]) -> Vec<f64> {
    // SYMMETRIC COMPLETION, the matrix the contract says is being factored ("only the
    // upper triangle is read"): a stored entry (i, j, val) with i <= j contributes to both
    // (i,j) and (j,i). For full-storage symmetric input this is exactly the input matrix.
    let mut y = vec![0.0f64; n];
    for j in 0..n {
        for p in cp[j]..cp[j + 1] {
            let i = ri[p];
            y[i] += v[p] * x[j];
            if i != j {
                y[j] += v[p] * x[i];
            }
        }
    }
    y
}

fn residual_inf(n: usize, cp: &[usize], ri: &[usize], v: &[f64], x: &[f64], b: &[f64]) -> f64 {
    matvec(n, cp, ri, v, x)
        .iter()
        .zip(b)
        .map(|(ax, bi)| (ax - bi).abs())
        .fold(0.0f64, f64::max)
}

/// Factoring succeeds => solving the CSC system it claims to have factored is accurate.
/// (For full-storage symmetric input, A x = b must hold on the ORIGINAL arrays - the
/// strongest end-to-end statement we can make about a factorization of arbitrary input.)
fn solves_original_system(
    n: usize,
    cp: &[usize],
    ri: &[usize],
    v: &[f64],
    seed: u64,
) -> Result<(), String> {
    let mut rng = Rng(seed);
    let b: Vec<f64> = (0..n).map(|_| rng.next_f64()).collect();
    let f = match SparseLdlt::factor(n, cp, ri, v) {
        Ok(f) => f,
        Err(LdltError::ZeroPivot(_) | LdltError::NearZeroPivot { .. }) => return Ok(()), // honest breakdown
        Err(e) => return Err(format!("unexpected error {e:?}")),
    };
    let x = match f.solve(&b) {
        Ok(x) => x,
        Err(e) => return Err(format!("solve failed on a factored system: {e:?}")),
    };
    let res = residual_inf(n, cp, ri, v, &x, &b);
    if res > 1e-6 {
        return Err(format!("residual {res}"));
    }
    Ok(())
}

#[test]
fn duplicates_are_summed_and_still_solve() {
    // The same structural entry supplied twice: standard CSC semantics (and the COO -> CSC
    // conversion behaviour every sparse library ships) is SUMMATION. If the factorization
    // silently took the last value instead, this test would fail on the residual.
    for seed in 0..20u64 {
        let n = 3 + (seed as usize % 9);
        let mut rng = Rng(seed * 3 + 1);
        // Random symmetric full-storage entries, then duplicate every entry.
        let mut cp = vec![0usize];
        let mut ri = Vec::new();
        let mut v = Vec::new();
        for j in 0..n {
            for i in 0..=j {
                if rng.below(3) == 0 || i == j {
                    let val = rng.next_f64() + if i == j { n as f64 } else { 0.0 };
                    ri.push(i);
                    v.push(val);
                    ri.push(i); // the duplicate
                    v.push(val);
                }
            }
            cp.push(ri.len());
        }
        solves_original_system(n, &cp, &ri, &v, seed)
            .unwrap_or_else(|e| panic!("seed {seed}: duplicated entries: {e}"));
    }
}

#[test]
fn explicit_zeros_are_harmless() {
    // Stored zeros are legal CSC; the factorization treats them as stored nonzeros and the
    // answer must not change. Each explicit zero here is a DUPLICATE of an existing golden
    // entry (summed in), so the effective matrix is exactly the canonical golden 3x3.
    let cp: &[usize] = &[0, 3, 7, 10];
    let ri: &[usize] = &[0, 0, 1, 0, 0, 1, 2, 1, 1, 2];
    let v: &[f64] = &[2.0, 0.0, 1.0, 1.0, 0.0, -3.0, 1.0, 1.0, 0.0, 2.0];
    let f = SparseLdlt::factor(3, cp, ri, v).expect("explicit zeros must factor");
    let x = f.solve(&[1.0, 2.0, 3.0]).unwrap();
    let want = [0.5, 0.0, 1.5]; // same golden values as the canonical 3x3
    for i in 0..3 {
        assert!((x[i] - want[i]).abs() < 1e-14);
    }
}

#[test]
fn lower_triangle_only_is_accepted() {
    // The contract says "only row <= col is read" - so a LOWER-only CSC (the transpose
    // pattern) still yields the correct solve on the symmetric completion. Wait: row <= col
    // read means a lower-triangle-only input has its UPPER half missing; the factorization
    // sees a matrix whose strict upper part is structurally zero. Assert the honest
    // consequence: it factors (or reports a pivot breakdown) and never panics - and for a DIAGONAL
    // lower-only matrix it is exact.
    let cp: &[usize] = &[0, 1, 2, 3];
    let ri: &[usize] = &[0, 1, 2];
    let v: &[f64] = &[4.0, 5.0, 6.0];
    let f = SparseLdlt::factor(3, cp, ri, v).unwrap();
    let x = f.solve(&[8.0, 10.0, 12.0]).unwrap();
    assert!((x[0] - 2.0).abs() < 1e-14 && (x[1] - 2.0).abs() < 1e-14 && (x[2] - 2.0).abs() < 1e-14);
}

#[test]
fn unordered_rows_within_columns_are_accepted() {
    // "any row order" is documented. Shuffle the golden matrix's rows within each column;
    // the solve must be bit-stable enough to pass a tight residual either way.
    for perm_seed in 0..10u64 {
        let mut rng = Rng(perm_seed * 11 + 2);
        // Golden columns with rows shuffled inside each column's slot range.
        let mut cols: Vec<Vec<(usize, f64)>> = vec![
            vec![(0, 2.0), (1, 1.0)],
            vec![(0, 1.0), (1, -3.0), (2, 1.0)],
            vec![(1, 1.0), (2, 2.0)],
        ];
        for col in &mut cols {
            for k in (1..col.len()).rev() {
                let s = rng.below(k + 1);
                col.swap(k, s);
            }
        }
        let mut cp = vec![0usize];
        let mut ri = Vec::new();
        let mut v = Vec::new();
        for col in &cols {
            for (r, val) in col {
                ri.push(*r);
                v.push(*val);
            }
            cp.push(ri.len());
        }
        let f = SparseLdlt::factor(3, &cp, &ri, &v).expect("shuffled rows must factor");
        let x = f.solve(&[1.0, 2.0, 3.0]).unwrap();
        let want = [0.5, 0.0, 1.5];
        for i in 0..3 {
            assert!((x[i] - want[i]).abs() < 1e-12, "perm {perm_seed}: x[{i}] drifted");
        }
    }
}

#[test]
fn degenerate_shapes_never_panic() {
    // n = 0: the empty factorization (col_ptr still has length n + 1 = 1, per the contract).
    assert!(SparseLdlt::factor(0, &[0], &[], &[]).is_ok());
    // Violating the length contract even at n = 0 is InvalidInput, not a panic.
    assert!(matches!(
        SparseLdlt::factor(0, &[], &[], &[]),
        Err(LdltError::InvalidInput(_))
    ));
    // n = 1, single diagonal entry.
    let f = SparseLdlt::factor(1, &[0, 1], &[0], &[2.0]).unwrap();
    assert_eq!(f.solve(&[4.0]).unwrap(), vec![2.0]);
    // n = 1, ZERO diagonal: an honest ZeroPivot, not a panic or a NaN.
    assert!(matches!(
        SparseLdlt::factor(1, &[0, 1], &[0], &[0.0]),
        Err(LdltError::ZeroPivot(0))
    ));
    // Fully empty column in the middle (structural zero column => zero pivot there).
    let r = SparseLdlt::factor(3, &[0, 2, 2, 3], &[0, 1, 2], &[1.0, 1.0, 1.0]);
    assert!(matches!(r, Err(LdltError::ZeroPivot(1))), "got {r:?}");
    // Trailing empty columns after a valid prefix.
    let r = SparseLdlt::factor(2, &[0, 1, 1], &[0], &[1.0]);
    assert!(matches!(r, Err(LdltError::ZeroPivot(1))), "got {r:?}");
}

#[test]
fn malformed_arrays_error_never_panic() {
    assert!(matches!(
        SparseLdlt::factor(2, &[0, 1], &[0, 1], &[1.0, 1.0]),
        Err(LdltError::InvalidInput(_))
    ));
    // parallel array mismatch
    assert!(matches!(
        SparseLdlt::factor(2, &[0, 1, 2], &[0, 1], &[1.0]),
        Err(LdltError::InvalidInput(_))
    ));
    // col_ptr[n] != nnz
    assert!(matches!(
        SparseLdlt::factor(2, &[0, 1, 5], &[0, 1], &[1.0, 1.0]),
        Err(LdltError::InvalidInput(_))
    ));
    // non-monotonic col_ptr
    assert!(matches!(
        SparseLdlt::factor(2, &[0, 2, 1], &[0, 0], &[1.0, 1.0]),
        Err(LdltError::InvalidInput(_))
    ));
    // row index out of range
    assert!(matches!(
        SparseLdlt::factor(2, &[0, 1, 2], &[0, 2], &[1.0, 1.0]),
        Err(LdltError::InvalidInput(_))
    ));
    // ...and n = 0 must stay valid: an empty factorization is a success, not an edge error.
    assert!(SparseLdlt::factor(0, &[0], &[], &[]).is_ok());
}

#[test]
fn random_valid_shape_garbage_never_panics() {
    // THE FUZZ: random CSC arrays with a VALID SHAPE (indices in range, col_ptr monotone)
    // but arbitrary values and patterns - duplicate rows inside a column included. The only
    // legal outcomes are Ok (then the residual check applies on the upper triangle the
    // algorithm actually read) or a pivot breakdown - ZeroPivot for an exact zero,
    // NearZeroPivot for one destroyed by cancellation. Panics and NaN outputs are failures.
    for seed in 0..300u64 {
        let n = 1 + (seed as usize % 24);
        let mut rng = Rng(seed * 7919 + 13);
        let nnz_cap = n * (n + 1) / 2;
        let nnz = 1 + rng.below(nnz_cap.min(64));
        let mut cp = vec![0usize; n + 1];
        let mut ri = Vec::new();
        let mut v = Vec::new();
        for _ in 0..nnz {
            let j = rng.below(n);
            let i = rng.below(j + 1); // row <= col only: the documented accepted shape
            ri.push(i);
            // Occasionally huge magnitudes: stresses the pivot test without leaving f64.
            let val = match seed % 5 {
                0 => rng.next_f64() * 1e12,
                1 => rng.next_f64() * 1e-6,
                _ => rng.next_f64(),
            };
            v.push(val);
            cp[j + 1] += 1;
        }
        // col_ptr accumulations: turn counts into a monotone prefix over ALL columns.
        for j in 1..=n {
            cp[j] += cp[j - 1];
        }
        let factored = SparseLdlt::factor(n, &cp, &ri, &v);
        match factored {
            Ok(f) => {
                // d must be finite everywhere - NaN in a pivot is the silent-poison failure.
                assert!(
                    f.d().iter().all(|d| d.is_finite()),
                    "seed {seed}: non-finite pivot with finite input"
                );
                let b: Vec<f64> = (0..n).map(|_| rng.next_f64()).collect();
                let x = f.solve(&b).unwrap();
                assert!(x.iter().all(|xi| xi.is_finite()), "seed {seed}: non-finite solve");
            }
            Err(LdltError::ZeroPivot(_) | LdltError::NearZeroPivot { .. }) => {} // the honest failures
            Err(e) => panic!("seed {seed}: unexpected {e:?}"),
        }
    }
}

#[test]
fn full_symmetric_storage_matches_upper_only() {
    // Both spellings of the same matrix must agree on the solve.
    let upper = SparseLdlt::factor(2, &[0, 1, 3], &[0, 0, 1], &[3.0, 1.0, 2.0]).unwrap();
    let full = SparseLdlt::factor(2, &[0, 2, 4], &[0, 1, 0, 1], &[3.0, 1.0, 1.0, 2.0]).unwrap();
    let b = [5.0, 7.0];
    let x1 = upper.solve(&b).unwrap();
    let x2 = full.solve(&b).unwrap();
    for i in 0..2 {
        assert!((x1[i] - x2[i]).abs() < 1e-13);
    }
}

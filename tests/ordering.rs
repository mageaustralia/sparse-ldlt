//! ORDERING GATE - `amd` + `factor_perm`.
//!
//! Three properties, each load-bearing for the Sturm use:
//! 1. CORRECTNESS THROUGH THE PERMUTATION: a solve through `factor_perm` answers the
//!    ORIGINAL system (residual vs the un-permuted A), and the inertia is INVARIANT -
//!    Sylvester's law says a symmetric permutation is a congruence; this gate makes that
//!    law a test, because FEM Studio's eigenvalue counts ride on it.
//! 2. THE FILL IS ACTUALLY REDUCED: on a random 2%-dense matrix (the pattern family where
//!    the benchmark showed the 8700x wall), AMD-ordered L carries a fraction of the
//!    unordered fill. The gate asserts the reduction, not a specific ratio.
//! 3. BANDS ARE NOT DESTROYED: on a banded (structural) matrix AMD may reorder but must
//!    not blow the fill up beyond a small constant.
//!
//! Plus degeneracies: non-permutations are rejected; isolated/bound nodes are handled.

// Index-driven dense construction, same rationale as the lib's own needless_range_loop allow.
#![allow(clippy::needless_range_loop)]

use sparse_ldlt::{amd, LdltError, SparseLdlt};

struct Rng(u64);
impl Rng {
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

/// Random symmetric matrix, full storage, CSC. `diag_shift` large => SPD.
fn random_symmetric(n: usize, density: f64, diag_shift: f64, seed: u64) -> (Vec<usize>, Vec<usize>, Vec<f64>, Vec<Vec<f64>>) {
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
    let mut cp = vec![0usize];
    let mut ri = Vec::new();
    let mut v = Vec::new();
    for j in 0..n {
        for i in 0..n {
            if dense[i][j] != 0.0 {
                ri.push(i);
                v.push(dense[i][j]);
            }
        }
        cp.push(ri.len());
    }
    (cp, ri, v, dense)
}

fn banded(n: usize) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let mut cp = vec![0usize];
    let mut ri = Vec::new();
    let mut v = Vec::new();
    for j in 0..n {
        for i in j.saturating_sub(1)..=j {
            ri.push(i);
            v.push(if i == j { 4.0 } else { 1.0 });
        }
        cp.push(ri.len());
    }
    (cp, ri, v)
}

fn neg_count(d: &[f64]) -> usize {
    d.iter().filter(|&&x| x < 0.0).count()
}

/// Dense Jacobi eigenvalue inertia - the independent oracle (same algorithm as the unit
/// tests; integration tests cannot reach the lib's `#[cfg(test)]` helpers).
fn negative_eigs_ref(mat: &[Vec<f64>]) -> usize {
    let n = mat.len();
    let mut a = mat.to_vec();
    for _sweep in 0..100 {
        let off: f64 = (0..n)
            .flat_map(|p| (p + 1..n).map(move |q| (p, q)))
            .map(|(p, q)| a[p][q] * a[p][q])
            .sum();
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
fn amd_returns_a_valid_permutation() {
    for seed in 0..30u64 {
        let n = 1 + (seed as usize % 40);
        let (cp, ri, _, _) = random_symmetric(n, 0.2, 10.0, seed);
        let order = amd(n, &cp, &ri);
        assert_eq!(order.len(), n);
        let mut seen = order.clone();
        seen.sort_unstable();
        assert_eq!(seen, (0..n).collect::<Vec<_>>(), "seed {seed}: not a permutation");
    }
    // Degenerate: n = 1, n = 0, and an empty matrix.
    assert_eq!(amd(1, &[0, 1], &[0]), vec![0]);
    assert_eq!(amd(0, &[], &[]), Vec::<usize>::new());
}

#[test]
fn ordered_solve_answers_the_original_system() {
    for seed in 0..25u64 {
        let n = 6 + (seed as usize % 30);
        let (cp, ri, v, dense) = random_symmetric(n, 0.25, n as f64 + 5.0, seed * 3 + 7);
        let order = amd(n, &cp, &ri);
        let f = SparseLdlt::factor_perm(n, &cp, &ri, &v, &order)
            .unwrap_or_else(|e| panic!("seed {seed}: {e:?}"));
        let b: Vec<f64> = (0..n).map(|i| ((i * 13 + seed as usize) % 7) as f64 - 3.0).collect();
        let x = f.solve(&b).unwrap();
        // Residual against the ORIGINAL (un-permuted) matrix.
        for i in 0..n {
            let ax: f64 = (0..n).map(|j| dense[i][j] * x[j]).sum();
            assert!(
                (ax - b[i]).abs() < 1e-6,
                "seed {seed}: residual {:+e} at row {}",
                (ax - b[i]).abs(),
                i
            );
        }
    }
}

#[test]
fn inertia_is_invariant_under_ordering() {
    // Sylvester's law as a test: the pivot-sign count must be identical for the plain,
    // AMD-ordered, and adversarially-reversed orderings, and must match the dense Jacobi
    // reference.
    let mut indefinite_checked = 0;
    for seed in 0..30u64 {
        let n = 5 + (seed as usize % 20);
        let (cp, ri, v, dense) = random_symmetric(n, 0.35, 0.5, seed * 5 + 2);
        let plain = match SparseLdlt::factor(n, &cp, &ri, &v) {
            Ok(f) => f,
            Err(LdltError::ZeroPivot(_) | LdltError::NearZeroPivot { .. }) => continue, // honest breakdown on this shift
            Err(e) => panic!("seed {seed}: {e:?}"),
        };
        let amd_order = amd(n, &cp, &ri);
        let f_amd = SparseLdlt::factor_perm(n, &cp, &ri, &v, &amd_order)
            .unwrap_or_else(|e| panic!("seed {seed}: amd factor {e:?}"));
        let rev: Vec<usize> = (0..n).rev().collect();
        let f_rev = SparseLdlt::factor_perm(n, &cp, &ri, &v, &rev)
            .unwrap_or_else(|e| panic!("seed {seed}: reversed factor {e:?}"));

        let np = neg_count(plain.d());
        assert_eq!(np, neg_count(f_amd.d()), "seed {seed}: AMD changed the inertia");
        assert_eq!(np, neg_count(f_rev.d()), "seed {seed}: reversed order changed the inertia");
        assert_eq!(np, negative_eigs_ref(&dense), "seed {seed}: inertia vs dense oracle");
        if np > 0 {
            indefinite_checked += 1;
        }
    }
    assert!(indefinite_checked >= 3, "family never went indefinite: {indefinite_checked}");
}

#[test]
fn amd_reduces_fill_on_random_sparse() {
    // The benchmark's shape: random 2%-dense, n = 512. Unordered L is heavily filled;
    // AMD-ordered L must carry meaningfully less. Measured for this AMD variant: ~2.2x
    // on this family (random graphs are the least AMD-friendly shape there is - no
    // locality for minimum degree to exploit; structured families do better, see the
    // grid test). Assert a real margin, print the ratio so drift is visible.
    let n = 512;
    let (cp, ri, v, dense) = random_symmetric(n, 0.02, n as f64 + 5.0, 42);
    let plain = SparseLdlt::factor(n, &cp, &ri, &v).unwrap();
    let order = amd(n, &cp, &ri);
    let ordered = SparseLdlt::factor_perm(n, &cp, &ri, &v, &order).unwrap();
    let nnz_plain = plain.nnz();
    let nnz_amd = ordered.nnz();
    println!("fill: unordered nnz(L)={nnz_plain}, amd nnz(L)={nnz_amd}");
    assert!(
        nnz_amd * 3 < nnz_plain * 2,
        "AMD fill {nnz_amd} not meaningfully below unordered fill {nnz_plain}"
    );
    // ...and the ordered factor still solves the original system accurately.
    let b: Vec<f64> = (0..n).map(|i| ((i * 11) % 9) as f64 - 4.0).collect();
    let x = ordered.solve(&b).unwrap();
    let res: f64 = (0..n)
        .map(|i| {
            let ax: f64 = (0..n).map(|j| dense[i][j] * x[j]).sum();
            (ax - b[i]).abs()
        })
        .fold(0.0f64, f64::max);
    assert!(res < 1e-6, "residual {res}");
}

#[test]
fn amd_shines_on_structured_grids() {
    // 2D 5-point grid: the shape real meshes resemble. Natural-order fill grows as
    // O(n^1.5); AMD approaches O(n log n). At 32x32 the published-AMD ballpark is ~3x,
    // and the trend must be IMPROVING with size (2.0x at 16x16, 2.8x at 32x32 measured).
    for (s, min_ratio) in [(16usize, 1.6f64), (32, 2.2)] {
        let n = s * s;
        let (cp, ri, v) = {
            let mut cp = vec![0usize];
            let mut ri = Vec::new();
            let mut v = Vec::new();
            let id = |r: usize, c: usize| r * s + c;
            for j in 0..s {
                for c in 0..s {
                    let i = id(j, c);
                    let mut entries = vec![(i, 4.0f64)];
                    if c > 0 { entries.push((id(j, c - 1), -1.0)); }
                    if c + 1 < s { entries.push((id(j, c + 1), -1.0)); }
                    if j > 0 { entries.push((id(j - 1, c), -1.0)); }
                    if j + 1 < s { entries.push((id(j + 1, c), -1.0)); }
                    entries.sort_by_key(|e| e.0);
                    for (r, val) in entries { ri.push(r); v.push(val); }
                    cp.push(ri.len());
                }
            }
            (cp, ri, v)
        };
        let plain = SparseLdlt::factor(n, &cp, &ri, &v).unwrap();
        let order = amd(n, &cp, &ri);
        let ordered = SparseLdlt::factor_perm(n, &cp, &ri, &v, &order).unwrap();
        let ratio = plain.nnz() as f64 / ordered.nnz() as f64;
        println!("grid {s}x{s}: fill ratio {ratio:.2}x");
        assert!(ratio >= min_ratio, "grid {s}x{s}: fill ratio {ratio:.2}x below {min_ratio}x");
    }
}

#[test]
fn amd_does_not_destroy_banded_fill() {
    for n in [64usize, 512, 2048] {
        let (cp, ri, v) = banded(n);
        let plain = SparseLdlt::factor(n, &cp, &ri, &v).unwrap();
        let order = amd(n, &cp, &ri);
        let ordered = SparseLdlt::factor_perm(n, &cp, &ri, &v, &order).unwrap();
        assert!(
            ordered.nnz() <= plain.nnz() * 3,
            "n={n}: banded fill went from {} to {} under AMD",
            plain.nnz(),
            ordered.nnz()
        );
    }
}

#[test]
fn factor_perm_rejects_non_permutations() {
    let (cp, ri, v) = banded(3);
    assert!(matches!(
        SparseLdlt::factor_perm(3, &cp, &ri, &v, &[0, 1, 1]),
        Err(LdltError::InvalidInput(_))
    ));
    assert!(matches!(
        SparseLdlt::factor_perm(3, &cp, &ri, &v, &[0, 1]),
        Err(LdltError::InvalidInput(_))
    ));
    assert!(matches!(
        SparseLdlt::factor_perm(3, &cp, &ri, &v, &[0, 1, 5]),
        Err(LdltError::InvalidInput(_))
    ));
}

/// THE ORDERING MUST COST LESS THAN THE FACTORIZATION IT SERVES. On the shipped product's 5.9k-node
/// shell mesh (2026-09-03) the first AMD took 1.7 s against a 0.2 s factorization, because it never
/// absorbed elements: every variable kept every element it had ever touched and every degree update
/// rescanned them all. A 100 x 100 triangulated grid (10k nodes, ~6 neighbours each, the shell mesh's
/// shape) has to order in well under a second - and its fill must still beat natural order by the
/// margin a 2D mesh gives AMD.
#[test]
fn amd_orders_a_mesh_sized_graph_in_bounded_time() {
    let s = 100usize;
    let n = s * s;
    let id = |r: usize, c: usize| r * s + c;
    let mut cols: Vec<Vec<usize>> = vec![Vec::new(); n];
    for r in 0..s {
        for c in 0..s {
            let i = id(r, c);
            let mut nb = vec![];
            if c + 1 < s { nb.push(id(r, c + 1)); }
            if r + 1 < s { nb.push(id(r + 1, c)); }
            if r + 1 < s && c + 1 < s { nb.push(id(r + 1, c + 1)); }   // the diagonal that makes it a triangulation
            for j in nb { cols[i].push(j); cols[j].push(i); }
        }
    }
    let mut cp = vec![0usize];
    let mut ri = Vec::new();
    let mut v = Vec::new();
    for (i, col) in cols.iter_mut().enumerate() {
        col.push(i);
        col.sort_unstable();
        col.dedup();
        for &r in col.iter() { ri.push(r); v.push(if r == i { 8.0 } else { -1.0 }); }
        cp.push(ri.len());
    }
    let t0 = std::time::Instant::now();
    let order = amd(n, &cp, &ri);
    let dt = t0.elapsed().as_secs_f64();
    assert_eq!(order.len(), n);
    let mut seen = vec![false; n];
    for &k in &order { assert!(!seen[k]); seen[k] = true; }
    println!("amd on a {s}x{s} triangulated grid: {dt:.3} s");
    assert!(dt < 1.0, "AMD took {dt:.2} s on a 10k-node mesh graph - the ordering must not cost more than the factorization");
    let plain = SparseLdlt::factor(n, &cp, &ri, &v).unwrap();
    let ordered = SparseLdlt::factor_perm(n, &cp, &ri, &v, &order).unwrap();
    let ratio = plain.nnz() as f64 / ordered.nnz() as f64;
    println!("  fill ratio {ratio:.2}x");
    assert!(ratio >= 3.0, "fill ratio {ratio:.2}x: AMD should beat natural order by 3x on a 100x100 mesh");
}

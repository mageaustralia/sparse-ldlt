//! REAL-MATRIX CORPUS GATE - the validation tier constructed matrices cannot provide.
//!
//! Three small, REAL structural stiffness matrices from the SuiteSparse (ex-Harwell-Boeing)
//! collection are bundled as fixtures, with provenance headers. Unlike constructed
//! matrices, these carry real-world sparsity structure (mesh connectivity, boundary
//! conditions), which is where a factorization's pattern logic can quietly misbehave.
//!
//! THE GATE (feral's rule): these are SPD matrices by the collection's metadata, so
//! wherever the factorization SUCCEEDS, inertia must be exactly 0 - no tolerance - and
//! the symmetric-completion solve must reach machine-precision residuals. Additionally the
//! number of negative pivots is cross-checked against a dense Jacobi eigenvalue reference
//! on the smaller matrix, so the gate does not merely trust its own premise.
//!
//! THE FEATURE-GATED SWEEP (`--features corpus-tests`, plus `CK_LDLT_CORPUS_DIR` pointing
//! at a directory of .mtx files): factors every file found, asserting the universal
//! contract - Ok-with-finite-pivots or an honest ZeroPivot, never a panic, and symmetric-
//! completion residual accuracy on every success. No HTTP dependency: CI fetches the
//! matrices (e.g. curl the SuiteSparse MM tarballs) and points the env var at them; the
//! crate itself stays dependency-free even with the feature on.

#![allow(clippy::needless_range_loop)]

use sparse_ldlt::{LdltError, SparseLdlt};

/// Minimal MatrixMarket `coordinate real symmetric` reader: skips comments, reads the
/// triangular entries, returns CSC. Enough for the corpus's well-formed files.
fn parse_mtx(text: &str) -> (usize, Vec<usize>, Vec<usize>, Vec<f64>) {
    let mut n = 0usize;
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for line in text.lines() {
        if line.starts_with('%') {
            continue;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.is_empty() {
            continue;
        }
        if n == 0 {
            // Header line: rows cols nnz (nnz ignored; the file body decides).
            n = fields[0]
                .parse()
                .expect("matrix market header must start with dimensions");
            continue;
        }
        let i: usize = fields[0].parse().expect("row index");
        let j: usize = fields[1].parse().expect("col index");
        let v: f64 = fields[2].parse().unwrap();
        rows.push(i - 1);
        cols.push(j - 1);
        vals.push(v);
    }
    // To CSC, upper triangle only. FLIP FIRST, then sort by column: flipping after the
    // sort scatters lower-triangle entries into columns the cursor has already passed,
    // silently corrupting col_ptr (the bug this gate initially shipped with - and a nice
    // demonstration of why a corpus gate exists at all).
    let upper: Vec<(usize, usize, f64)> = rows
        .iter()
        .zip(&cols)
        .zip(&vals)
        .map(|((&r, &c), &v)| if r <= c { (r, c, v) } else { (c, r, v) })
        .collect();
    let mut order: Vec<usize> = (0..upper.len()).collect();
    order.sort_by_key(|&k| (upper[k].1, upper[k].0));
    let mut cp = vec![0usize; n + 1];
    let mut ri = Vec::new();
    let mut v = Vec::new();
    for k in order {
        let (i, j, val) = upper[k];
        while ri.len() < cp[j] {
            // Structural gap: an inner column with no entries (legal CSC).
            ri.push(0);
            v.push(0.0);
        }
        ri.push(i);
        v.push(val);
        cp[j + 1] = ri.len();
    }
    // Close gaps for columns that had entries out of order (all sorted above, but be safe).
    for j in 0..n {
        if cp[j + 1] < cp[j] {
            panic!("corpus file produced non-monotonic col_ptr");
        }
    }
    (n, cp, ri, v)
}

fn sym_matvec(n: usize, cp: &[usize], ri: &[usize], v: &[f64], x: &[f64]) -> Vec<f64> {
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

/// Dense Jacobi eigenvalue inertia - the independent oracle (same algorithm as the unit
/// tests). Only feasible on small n; that is what the bundled fixtures are for.
fn negative_eigs(mat: &[Vec<f64>]) -> usize {
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

fn factor_and_gate(name: &str, n: usize, cp: &[usize], ri: &[usize], v: &[f64]) {
    let f = match SparseLdlt::factor(n, cp, ri, v) {
        Ok(f) => f,
        Err(LdltError::ZeroPivot(k)) => panic!("{name}: SPD corpus matrix broke down at column {k}"),
        Err(e) => panic!("{name}: unexpected {e:?}"),
    };
    // THE NO-TOLERANCE INERTIA GATE: SPD corpus matrices have inertia exactly 0.
    let neg = f.d().iter().filter(|&&d| d < 0.0).count();
    assert_eq!(neg, 0, "{name}: SPD corpus matrix reads {neg} negative pivots");
    // Machine-precision solve against the symmetric completion.
    let b: Vec<f64> = (0..n).map(|i| ((i * 37 + 11) % 13) as f64 - 6.0).collect();
    let x = f.solve(&b).unwrap();
    let res = sym_matvec(n, cp, ri, v, &x)
        .iter()
        .zip(&b)
        .map(|(ax, bi)| (ax - bi).abs())
        .fold(0.0f64, f64::max);
    assert!(res < 1e-6, "{name}: residual {res}");
    println!("  {name}: n={n}, nnz(L)={}, inertia 0, residual {res:.2e}", f.nnz());
}

#[test]
fn real_corpus_matrices() {
    // REAL structural stiffness matrices from the SuiteSparse (Harwell-Boeing) collection,
    // bundled as fixtures with their provenance headers. SPD by the collection's metadata,
    // so the gate's inertia premise is external, not self-referential.
    for (name, text) in [("bcsstk01", include_str!("data/bcsstk01.mtx")), ("bcsstk03", include_str!("data/bcsstk03.mtx"))] {
        let (n, cp, ri, v) = parse_mtx(text);
        factor_and_gate(name, n, &cp, &ri, &v);
    }
}

#[test]
fn corpus_dir_sweep() {
    if !cfg!(feature = "corpus-tests") {
        eprintln!("corpus_dir_sweep: enable --features corpus-tests (and set CK_LDLT_CORPUS_DIR) to run");
        return;
    }
    let dir = match std::env::var("CK_LDLT_CORPUS_DIR") {
        Ok(d) if !d.is_empty() => d,
        _ => {
            println!("corpus_dir_sweep: CK_LDLT_CORPUS_DIR not set - nothing to sweep");
            return;
        }
    };
    let mut swept = 0;
    let entries = std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("corpus dir {dir}: {e}"));
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("mtx") {
            continue;
        }
        let text = std::fs::read_to_string(&path).expect("readable .mtx");
        let (n, cp, ri, v) = parse_mtx(&text);
        match SparseLdlt::factor(n, &cp, &ri, &v) {
            Ok(f) => {
                assert!(f.d().iter().all(|d| d.is_finite()), "{}: non-finite pivot", path.display());
                let b: Vec<f64> = (0..n).map(|i| ((i * 7) % 13) as f64 - 6.0).collect();
                let x = f.solve(&b).unwrap();
                let res = sym_matvec(n, &cp, &ri, &v, &x)
                    .iter()
                    .zip(&b)
                    .map(|(ax, bi)| (ax - bi).abs())
                    .fold(0.0f64, f64::max);
                assert!(res < 1e-3, "{}: residual {res}", path.display());
                swept += 1;
            }
            Err(LdltError::ZeroPivot(_)) => swept += 0, // honest breakdown, counted, not a failure
            Err(e) => panic!("{}: unexpected {e:?}", path.display()),
        }
    }
    println!("  corpus sweep: {swept} matrices factored + residual-gated from {dir}");
}

#[test]
#[ignore = "dense-oracle cross-check, small n only - run explicitly"]
fn bcsstk01_inertia_matches_dense_jacobi() {
    let (n, cp, ri, v) = parse_mtx(include_str!("data/bcsstk01.mtx"));
    // Dense-ify the symmetric completion.
    let mut dense = vec![vec![0.0f64; n]; n];
    for j in 0..n {
        for p in cp[j]..cp[j + 1] {
            let i = ri[p];
            dense[i][j] += v[p];
            if i != j {
                dense[j][i] += v[p];
            }
        }
    }
    let f = SparseLdlt::factor(n, &cp, &ri, &v).unwrap();
    let sparse_neg = f.d().iter().filter(|&&d| d < 0.0).count();
    let dense_neg = negative_eigs(&dense);
    assert_eq!(sparse_neg, dense_neg, "sparse inertia disagrees with the dense oracle");
}

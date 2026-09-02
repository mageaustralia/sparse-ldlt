# sparse-ldlt

Pure-Rust, dependency-free sparse **symmetric-indefinite LDLᵀ factorization** and solver.

Factors a symmetric sparse matrix `A = L·D·Lᵀ` (with `L` unit-lower-triangular and `D` a
**signed** diagonal), then solves `A x = b`. Because `D` may hold negative entries, it
handles symmetric **indefinite** systems - not just positive-definite ones - and it exposes
`D`, so you can read the matrix **inertia** (the number of negative eigenvalues, by
Sylvester's law of inertia) directly from the pivots.

## Why this exists

Most sparse direct solvers available in pure Rust only provide **positive-definite Cholesky**
and never expose the signed pivots. That leaves a gap for the problems where the sign of the
pivots is the whole point:

- **saddle-point / KKT systems** from constrained optimisation and mixed finite elements,
- **shifted eigenvalue matrices** `K - σM`, which are indefinite for shifts inside the
  spectrum - factoring them and counting negative pivots gives a **Sturm eigenvalue count**,
- **quasi-definite** systems (interior-point methods).

`sparse-ldlt` is a small, self-contained implementation of the standard up-looking sparse
LDLᵀ (elimination-tree) method - see T. A. Davis, *Direct Methods for Sparse Linear Systems*
(SIAM, 2006) - with **no dependencies**, no `unsafe`, and stable-Rust only.

## No pivoting - read this before relying on the inertia count

**The factorization is un-pivoted.** If a pivot `D[k]` reaches exactly zero, `factor`
fails loudly with `LdltError::ZeroPivot(k)` - but a pivot that is merely *small* factors
through with whatever sign the rounding produces. On a matrix near a singular point - a
shift `σ` landing on an eigenvalue, a mechanism in a structure - that can make an inertia
count WRONG without any error being raised. If your use case needs certified pivots under
those conditions, you need a pivoting solver (Bunch-Kaufman / multifrontal); this crate
trades that machinery for ~600 dependency-free lines. The Sturm-count mitigation is to
treat a near-zero pivot as a sign the shift is too close and re-factor at a nudged shift.

## Fill-reducing ordering (AMD)

```rust
use sparse_ldlt::{amd, SparseLdlt};

let order = amd(n, &col_ptr, &row_idx);                 // Amestoy-Davis-Duff ordering
let f = SparseLdlt::factor_perm(n, &col_ptr, &row_idx, &values, &order).unwrap();
let x = f.solve(&b).unwrap();                           // permutation handled for you
```

Without an ordering, fill-in on an irregular sparsity can cost orders of magnitude
(measured, `cargo bench`: random 2%-dense n=1024 - unordered factor ~160 ms vs 19 µs on a
banded matrix of the same order). `amd` is a quotient-graph approximate minimum degree
(Amestoy-Davis-Duff 1996) implemented in this crate with the same zero-dependency rules:
eliminated nodes become elements, degrees are the AMD external degrees recomputed over the
neighbourhood only. Ordering NEVER changes inertia (a symmetric permutation is a
congruence - Sylvester's law), so Sturm counts are identical with or without it; the
`tests/ordering.rs` gate asserts exactly that, alongside measured fill reduction.

## Usage

Supply the matrix in compressed-sparse-column (CSC) form. Only the upper triangle
(entries with row ≤ col in each column) is read, so a fully-populated symmetric matrix is
also fine.

```rust
use sparse_ldlt::SparseLdlt;

// Symmetric indefinite 3x3 matrix (full storage), in CSC:
//   [ 2  1  0 ]
//   [ 1 -3  1 ]
//   [ 0  1  2 ]
let col_ptr = vec![0, 2, 5, 7];
let row_idx = vec![0, 1,  0, 1, 2,  1, 2];
let values  = vec![2.0, 1.0,  1.0, -3.0, 1.0,  1.0, 2.0];

let f = SparseLdlt::factor(3, &col_ptr, &row_idx, &values).unwrap();

// Solve A x = b
let x = f.solve(&[1.0, 2.0, 3.0]).unwrap();

// Inertia: number of negative eigenvalues == number of negative pivots
let negative_eigenvalues = f.d().iter().filter(|&&v| v < 0.0).count();
assert_eq!(negative_eigenvalues, 1);
```

## Notes

- **Ordering:** `amd` + `factor_perm` are built in (see above). The plain `factor` still
  applies none - deterministic and unchanged since v0.1.0.
- **No pivoting** (see the section above): breakdown is loud (`LdltError::ZeroPivot`), but
  small pivots factor through with rounding's sign - near a singular point, nudge the shift.
  Non-finite input values (NaN / ±inf) are rejected.
- `solve` returns `Result<Vec<f64>, LdltError>` - a right-hand side that does not match the
  factored order is `LdltError::SizeMismatch`, never a panic.
- Correctness is gated by an **inertia oracle** (`tests/inertia_oracle.rs`) in the spirit of
  feral's consensus validation, scoped to what a dependency-free crate can run anywhere:
  wherever a factorization *succeeds*, the pivot-sign inertia must be exactly correct - no
  tolerance. The oracle families are matrices whose inertia is known by construction
  (congruence `A = XᵀSX`, quasi-definite KKT blocks, Sturm shifts with exact endpoints and a
  monotonicity sweep), plus dense-residual checks at machine precision.
- **Property tests** (`tests/property.rs`) pin the adversarial-CSC contract: duplicate
  entries are summed, explicit zeros are harmless, any row order within a column is
  accepted, degenerate shapes (n = 0, empty columns) never panic, malformed arrays return
  `InvalidInput`, and ~2 400 random valid-shape CSCs (wild magnitudes included) produce
  either a correct factorization or an honest `ZeroPivot` - never a panic or a NaN pivot.
- **Real-matrix corpus** (`tests/corpus.rs`): real structural stiffness matrices from the
  SuiteSparse (Harwell-Boeing) collection are bundled as fixtures and gated on external
  metadata - SPD by the collection, so inertia must be exactly 0 - plus a dense Jacobi
  cross-check and a `corpus-tests` feature that sweeps any directory of `.mtx` files
  (`CK_LDLT_CORPUS_DIR`) for CI-scale validation. No network, no dependencies.
- **Benchmarks** (`cargo bench`, criterion): factor/solve vs n on banded (structural) and
  random-sparse patterns. The measured fill wall - n = 1024, banded 19 µs vs random-2%
  ~160 ms - is the quantified case for a fill-reducing ordering (not yet implemented;
  permute the matrix yourself meanwhile).

## Provenance

Written from the algorithm's published description - T. A. Davis, *Direct Methods for Sparse
Linear Systems* (SIAM, 2006) - as an independent implementation. No source code from Tim
Davis's LDL or from `sprs-ldl` was copied, translated, or consulted while writing it; any
resemblance is the algorithm itself, which is published mathematics.

Created by [MAGE Engineering](https://mageengineering.com.au/) for its **FEM Analysis Studio**,
where it replaces an LGPL sparse LDLᵀ dependency in the structural analysis engine. Released
under the MIT licence so the wider Rust community can use it too.

## Credits

Created by [MAGE Engineering](https://mageengineering.com.au/) for its **FEM Analysis Studio**,
where it replaces an LGPL sparse LDLᵀ dependency in the structural analysis engine. Released
under the MIT licence so the wider Rust community can use it too.

## Licence

MIT - see [LICENSE](LICENSE).

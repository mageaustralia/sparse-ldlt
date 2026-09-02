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
(SIAM, 2006) - with **zero runtime dependencies**, no `unsafe`, and stable-Rust only.
(Its own tests and benchmarks use dev-only crates - criterion - which never ship to
consumers.)

## No pivoting - read this before relying on the inertia count

**The factorization is un-pivoted.** A pivot `D[k]` that reaches exactly zero fails loudly
with `LdltError::ZeroPivot(k)`. A pivot whose magnitude has been destroyed by cancellation
is just as dangerous and is now reported too, as
`LdltError::NearZeroPivot { column, pivot, scale, suggested_shift }`, whenever
`|D[k]| < NEAR_ZERO_PIVOT_REL * scale` (`1e-13` relative to the largest absolute diagonal
entry of the input, about 1000x `f64::EPSILON`). That case is the one worth naming: such a
pivot still carries a *sign*, but the sign is rounding noise, and the sign pattern of `D`
IS the matrix inertia - so on a matrix near a singular point (a shift `σ` landing on an
eigenvalue, a mechanism in a structure) the old behaviour was a WRONG inertia count with no
error raised. It is no longer returned silently.

Recovering from it is no longer the caller's problem either. `factor_shifted` (and
`factor_perm_shifted`) try the unshifted factorization first, and on a breakdown retry with
a positive diagonal shift - starting at `suggested_shift`, multiplying by 8, at most 8
attempts. `shift()` reports how far the matrix was moved:

```rust
let f = SparseLdlt::factor_shifted(n, &col_ptr, &row_idx, &values)?;
if f.shift() != 0.0 {
    // This is an exact factorization of A + shift*I, NOT of A. Its inertia is the shifted
    // matrix's inertia, so a Sturm count from it is a count at (sigma - shift): correct for
    // it. Ignoring shift() is a bug.
}
```

If your use case needs certified pivots without any perturbation, you need a pivoting solver
(Bunch-Kaufman / multifrontal); this crate trades that machinery for ~800 dependency-free
lines, and reports honestly where the trade bites.

## Fill-reducing ordering (AMD)

```rust
use sparse_ldlt::{amd, SparseLdlt};

let order = amd(n, &col_ptr, &row_idx);                 // Amestoy-Davis-Duff ordering
let f = SparseLdlt::factor_perm(n, &col_ptr, &row_idx, &values, &order).unwrap();
let x = f.solve(&b).unwrap();                           // permutation handled for you
```

Without an ordering, fill-in on an irregular sparsity can cost orders of magnitude
(measured, `cargo bench`: random 2%-dense n=1024 - unordered factor ~0.27 s vs ~30 µs on a
banded matrix of the same order; with `amd` + `factor_perm` the same random matrix factors
in ~70 ms). `amd` is a quotient-graph approximate minimum degree
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
- **No pivoting** (see the section above): breakdown is loud - `LdltError::ZeroPivot` for an
  exact zero, `LdltError::NearZeroPivot` for a pivot whose sign has become rounding noise.
  `factor_shifted` does the nudging and `shift()` says how much it nudged.
  Non-finite input values (NaN / ±inf) are rejected.
- `solve` returns `Result<Vec<f64>, LdltError>` - a right-hand side that does not match the
  factored order is `LdltError::SizeMismatch`, never a panic.
- Correctness is gated by an **inertia oracle** (`tests/inertia_oracle.rs`) in the spirit of
  feral's consensus validation, scoped to what a dependency-free crate can run anywhere:
  wherever a factorization *succeeds*, the pivot-sign inertia must be exactly correct - no
  tolerance. The oracle families are matrices whose inertia is known by construction
  (congruence `A = XᵀSX`, quasi-definite KKT blocks, Sturm shifts with exact endpoints and a
  monotonicity sweep), plus dense-residual checks at machine precision. A fourth,
  *adversarial* family targets the near-zero pivot directly - shifts landing 1e-15 from an
  eigenvalue, quasi-definite blocks with a diagonal entry driven to 1e-18, and KKT saddle
  points with a zero (2,2) block and a rank-deficient constraint - and is oracled against a
  dependency-free dense cyclic Jacobi eigensolver that is itself checked against closed-form
  spectra. Each fixture must either be refused or produce the exact inertia, and every
  refusal must then be recovered by `factor_shifted` with a small residual against
  `A + shift*I`.
- **Property tests** (`tests/property.rs`) pin the adversarial-CSC contract: duplicate
  entries are summed, explicit zeros are harmless, any row order within a column is
  accepted, degenerate shapes (n = 0, empty columns) never panic, malformed arrays return
  `InvalidInput`, and ~330 random valid-shape CSCs (wild magnitudes included) produce
  either a correct factorization or an honest pivot breakdown - never a panic or a NaN pivot.
- **Real-matrix corpus** (`tests/corpus.rs`): real structural stiffness matrices from the
  SuiteSparse (Harwell-Boeing) collection are bundled as fixtures and gated on external
  metadata - SPD by the collection, so inertia must be exactly 0 - plus a dense Jacobi
  cross-check and a `corpus-tests` feature that sweeps any directory of `.mtx` files
  (`CK_LDLT_CORPUS_DIR`) for CI-scale validation. No network, no dependencies.
- **Benchmarks** (`cargo bench`, criterion): factor/solve vs n on banded (structural) and
  random-sparse patterns, with and without AMD. The measured fill wall - n = 1024, banded
  ~30 µs vs random-2% ~0.27 s - is what `amd` addresses: the same random matrix factors in
  ~70 ms ordered (~0.27 s unordered). On a banded matrix AMD neither helps nor much hurts
  (n = 4096: ~121 µs unordered, ~187 µs ordered). Numbers from one machine; run
  `cargo bench` for yours.

## Provenance

Written from the algorithm's published description - T. A. Davis, *Direct Methods for Sparse
Linear Systems* (SIAM, 2006) - as an independent implementation. No source code from Tim
Davis's LDL or from `sprs-ldl` was copied, translated, or consulted while writing it; any
resemblance is the algorithm itself, which is published mathematics.

Created by [MAGE Engineering](https://mageengineering.com.au/) for its **FEM Analysis Studio**,
where it replaces an LGPL sparse LDLᵀ dependency in the structural analysis engine. Released
under the MIT licence so the wider Rust community can use it too.

## Licence

MIT - see [LICENSE](LICENSE).

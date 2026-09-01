# sparse-ldlt

Pure-Rust, dependency-free sparse **symmetric-indefinite** LDLᵀ factorization and solver.

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

- **No fill-reducing reordering** is applied. Permute the matrix yourself (RCM, AMD, nested
  dissection, ...) before factoring if fill matters for your problem size.
- **No pivoting.** Like every un-pivoted LDLᵀ it breaks down (returns `LdltError::ZeroPivot`)
  if a diagonal entry of `D` reaches zero - for example when a shift lands exactly on an
  eigenvalue. Nudge the shift and retry. Non-finite input values (NaN / ±inf) are rejected.
- `solve` returns `Result<Vec<f64>, LdltError>` - a right-hand side that does not match the
  factored order is `LdltError::SizeMismatch`, never a panic.
- Correctness is gated by an **inertia oracle** (`tests/inertia_oracle.rs`) in the spirit of
  feral's consensus validation, scoped to what a dependency-free crate can run anywhere:
  wherever a factorization *succeeds*, the pivot-sign inertia must be exactly correct - no
  tolerance. The oracle families are matrices whose inertia is known by construction
  (congruence `A = XᵀSX`, quasi-definite KKT blocks, Sturm shifts with exact endpoints and a
  monotonicity sweep), plus dense-residual checks at machine precision.

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

# sparse-ldlt

A sparse LDLᵀ factorization for symmetric matrices, including indefinite ones, in pure Rust
with no dependencies.

It factors `A = L D Lᵀ`, where `L` is unit lower triangular and `D` is diagonal, and solves
`A x = b`. `D` can hold negative entries, and it is exposed. By Sylvester's law of inertia, the
number of negative entries in `D` is the number of negative eigenvalues of `A`.

## When to use it

Most pure-Rust sparse solvers offer only Cholesky, which needs a positive-definite matrix and
does not report pivot signs. This crate is for problems where the matrix is indefinite or the
signs matter:

- saddle-point (KKT) systems from constrained optimisation and mixed finite elements;
- shifted eigenvalue problems `K - σM`, where counting negative pivots gives the number of
  eigenvalues below `σ` (a Sturm count);
- quasi-definite systems from interior-point methods.

It implements the up-looking elimination-tree method described in T. A. Davis, *Direct Methods
for Sparse Linear Systems* (SIAM, 2006). It has no runtime dependencies, uses no `unsafe` code
and builds on stable Rust.

## Usage

Pass the matrix in compressed sparse column (CSC) form. Only the upper triangle (row ≤ column)
is read, so you can pass either the upper triangle or the full matrix.

```rust
use sparse_ldlt::SparseLdlt;

//   [ 2  1  0 ]
//   [ 1 -3  1 ]
//   [ 0  1  2 ]
let col_ptr = vec![0, 2, 5, 7];
let row_idx = vec![0, 1, 0, 1, 2, 1, 2];
let values = vec![2.0, 1.0, 1.0, -3.0, 1.0, 1.0, 2.0];

let f = SparseLdlt::factor(3, &col_ptr, &row_idx, &values).unwrap();
let x = f.solve(&[1.0, 2.0, 3.0]).unwrap();

// One negative pivot, so one negative eigenvalue.
let negative = f.d().iter().filter(|&&v| v < 0.0).count();
assert_eq!(negative, 1);
```

## Ordering

For anything other than a banded matrix, reorder it first. `amd` computes an approximate
minimum degree ordering (Amestoy, Davis and Duff, 1996) and `factor_perm` applies it. `solve`
handles the permutation for you.

```rust
use sparse_ldlt::{amd, SparseLdlt};

let order = amd(n, &col_ptr, &row_idx);
let f = SparseLdlt::factor_perm(n, &col_ptr, &row_idx, &values, &order).unwrap();
let x = f.solve(&b).unwrap();
```

On one machine, a random 2%-dense matrix with n = 1024 took about 0.27 s to factor without
ordering and about 70 ms with it. On banded matrices, ordering makes little difference. Run
`cargo bench` to measure on yours.

A symmetric permutation does not change the inertia, so the negative-pivot count is the same
with or without ordering.

## Pivot breakdown

The factorization does no pivoting. When a pivot is zero or close to it, `factor` returns an
error instead of a result:

- `LdltError::ZeroPivot(k)` when pivot `k` is exactly zero.
- `LdltError::NearZeroPivot { column, pivot, scale, suggested_shift }` when
  `|D[k]| < NEAR_ZERO_PIVOT_REL * scale`, where `scale` is the largest absolute diagonal entry of
  `A` and `NEAR_ZERO_PIVOT_REL` is `1e-13`. The sign of such a pivot is rounding error, so an
  inertia count taken from it could be wrong.

`factor_shifted` and `factor_perm_shifted` recover from these errors. They try the unshifted
factorization first. If it breaks down, they factor `A + shift·I`, starting from
`suggested_shift` and multiplying the shift by 8 on each retry, for up to 8 attempts.

```rust
let f = SparseLdlt::factor_shifted(n, &col_ptr, &row_idx, &values)?;
if f.shift() != 0.0 {
    // f factors A + shift·I, not A. Its inertia, and any solve, are for the shifted matrix.
}
```

Always check `shift()` after a shifted factorization. If you need exact pivots with no
perturbation, use a solver with Bunch-Kaufman or multifrontal pivoting instead.

## Errors

Every failure is an `LdltError`, never a panic:

- `ZeroPivot` and `NearZeroPivot`, described above.
- `InvalidInput` for malformed CSC arrays or non-finite values (NaN, ±inf).
- `SizeMismatch` when a right-hand side does not match the matrix order.

## Testing

- **Inertia** (`tests/inertia_oracle.rs`): matrices whose inertia is known by construction
  (congruences `XᵀSX`, quasi-definite KKT blocks, Sturm shifts at known eigenvalues). Whenever a
  factorization succeeds, its inertia must be exactly right. A second set places pivots near
  zero on purpose. Each of those must either be refused or give the right inertia, and every
  refusal must then be recovered by `factor_shifted`.
- **Input handling** (`tests/property.rs`): duplicate entries are summed, explicit zeros and
  any row order within a column are accepted, and empty or malformed input never panics. About
  330 random matrices each give either a correct factorization or a pivot error.
- **Real matrices** (`tests/corpus.rs`): structural stiffness matrices from the SuiteSparse
  collection, which are positive definite, so every pivot must be positive. With the
  `corpus-tests` feature, the test also factors every `.mtx` file in the directory named by
  `CK_LDLT_CORPUS_DIR`.
- **Ordering** (`tests/ordering.rs`): AMD reduces fill and never changes the inertia.
- **Benchmarks** (`cargo bench`, using criterion): factor and solve times against n, for
  banded and random patterns, with and without AMD.

criterion is a dev-dependency only. It is not part of what you download.

## Provenance

This is an independent implementation, written from the published description of the method
in Davis (2006). It does not copy or translate code from Tim Davis's LDL or from `sprs-ldl`.

It was written by [MAGE Engineering](https://mageengineering.com.au/) for its FEM Analysis
Studio, where it replaced an LGPL-licensed dependency.

## Licence

MIT. See [LICENSE](LICENSE).

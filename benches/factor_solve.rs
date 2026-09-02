//! factor/solve throughput, the two numbers that decide whether this crate earns its keep
//! in the FEM Sturm loop (which factors THOUSANDS of times per eigenvalue solve).
//!
//! Two pattern families, both drawn the way structural problems actually look:
//! - BANDED: a 3D frame/brick stiffness matrix has near-banded sparsity; bandwidth ~ grid
//!   pitch. This is the shape RCM exists for - watch factor cost grow with the band.
//! - RANDOM: Erdos-Renyi sparsity at fixed density, the worst case for fill - the shape
//!   where a fill-reducing ordering would pay, and the baseline to compare against.
//!
//! n is swept so the O(n * bandwidth^2)-ish growth is visible per row of the table.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;

use sparse_ldlt::SparseLdlt;

/// Tridiagonal-plus-one (banded, bandwidth 2): the friendly structural shape.
fn banded(n: usize) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let mut cp = vec![0usize];
    let mut ri = Vec::new();
    let mut v = Vec::new();
    for j in 0..n {
        for i in j.saturating_sub(1)..=j {
            ri.push(i);
            // Dominant diagonal, mild coupling: SPD by diagonal dominance.
            v.push(if i == j { 4.0 } else { 1.0 });
        }
        cp.push(ri.len());
    }
    (cp, ri, v)
}

/// Random sparse symmetric (upper stored) with `density` off-diagonal probability.
struct Rng(u64);
impl Rng {
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

fn random_spd(n: usize, density: f64, seed: u64) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let mut rng = Rng(seed);
    let mut dense = vec![0.0f64; n * n];
    for j in 0..n {
        for i in 0..j {
            if rng.next_f64() < density * 2.0 - 1.0 {
                let v = rng.next_f64();
                dense[i * n + j] = v;
                dense[j * n + i] = v;
            }
        }
        dense[j * n + j] = n as f64 + rng.next_f64();
    }
    let mut cp = vec![0usize];
    let mut ri = Vec::new();
    let mut v = Vec::new();
    for j in 0..n {
        for i in 0..=j {
            if dense[i * n + j] != 0.0 {
                ri.push(i);
                v.push(dense[i * n + j]);
            }
        }
        cp.push(ri.len());
    }
    (cp, ri, v)
}

fn bench_factor(c: &mut Criterion) {
    let mut group = c.benchmark_group("factor");
    // n = 4096 runs on the BANDED family only: the random family at that size has no
    // fill-reducing ordering available (see README), the factor is effectively cubic, and
    // the wall it hits IS the ordering motivation - measured at 1024, extrapolated beyond.
    // The `amd` rows are the same shapes with the AMD ordering from this crate - the
    // measured answer to that wall.
    for n in [64usize, 256, 1024, 4096] {
        let (cp, ri, v) = banded(n);
        group.bench_function(&format!("banded/{n}"), |b| {
            b.iter(|| {
                SparseLdlt::factor(black_box(n), black_box(&cp), black_box(&ri), black_box(&v))
                    .unwrap()
            })
        });
        if n <= 1024 {
            let (cp, ri, v) = random_spd(n, 0.02, 42 + n as u64);
            group.bench_function(&format!("random2pct/{n}"), |b| {
                b.iter(|| {
                    SparseLdlt::factor(black_box(n), black_box(&cp), black_box(&ri), black_box(&v))
                        .unwrap()
                })
            });
            let order = sparse_ldlt::amd(n, &cp, &ri);
            group.bench_function(&format!("random2pct-amd/{n}"), |b| {
                b.iter(|| {
                    SparseLdlt::factor_perm(
                        black_box(n),
                        black_box(&cp),
                        black_box(&ri),
                        black_box(&v),
                        black_box(&order),
                    )
                    .unwrap()
                })
            });
        } else {
            // Banded + AMD at 4096: the ordering must not hurt the structural shape.
            let (cp, ri, v) = banded(n);
            let order = sparse_ldlt::amd(n, &cp, &ri);
            group.bench_function(&format!("banded-amd/{n}"), |b| {
                b.iter(|| {
                    SparseLdlt::factor_perm(
                        black_box(n),
                        black_box(&cp),
                        black_box(&ri),
                        black_box(&v),
                        black_box(&order),
                    )
                    .unwrap()
                })
            });
        }
    }
    group.finish();
}

fn bench_solve(c: &mut Criterion) {
    let mut group = c.benchmark_group("solve");
    for n in [64usize, 256, 1024, 4096] {
        let (cp, ri, v) = banded(n);
        let f = SparseLdlt::factor(n, &cp, &ri, &v).unwrap();
        let b: Vec<f64> = (0..n).map(|i| (i % 7) as f64 - 3.0).collect();
        group.bench_with_input(BenchmarkId::new("banded", n), &b, |b, b_rhs| {
            b.iter(|| f.solve(black_box(b_rhs)).unwrap())
        });
    }
    group.finish();
}

criterion_group!(benches, bench_factor, bench_solve);
criterion_main!(benches);

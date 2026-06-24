//! Performance benchmarks for TROP core computations.
//!
//! Covers: distance matrix, weight matrix, and full model estimation.
//! Uses deterministic random data (Xoshiro256PlusPlus with seed 42) for reproducibility.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array2;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rand::Rng;

use trop_core::distance::{
    compute_unit_distance_matrix_internal, UnitDistanceCache,
};
use trop_core::weights::compute_weight_matrix;
use trop_core::estimation::estimate_model;

// ---------------------------------------------------------------------------
// Data generation helpers
// ---------------------------------------------------------------------------

/// Generate a simulated panel: Y matrix (T x N, column-major) and D treatment
/// indicator. Treatment is assigned to the last `n_treated` units from period
/// `treat_start` onward.
fn generate_panel_data(
    n_units: usize,
    n_periods: usize,
    n_treated: usize,
    treat_start: usize,
) -> (Array2<f64>, Array2<f64>) {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);

    // Y: standard normal + unit fixed effect + time trend
    let y = Array2::from_shape_fn((n_periods, n_units), |(t, i)| {
        let unit_fe = i as f64 * 0.5;
        let time_trend = t as f64 * 0.1;
        let noise: f64 = rng.gen::<f64>() * 2.0 - 1.0; // Uniform(-1, 1)
        unit_fe + time_trend + noise
    });

    // D: last n_treated units treated from treat_start onward
    let d = Array2::from_shape_fn((n_periods, n_units), |(t, i)| {
        if i >= (n_units - n_treated) && t >= treat_start {
            1.0
        } else {
            0.0
        }
    });

    (y, d)
}

/// Generate time distance matrix |t - s| as i64.
fn generate_time_dist(n_periods: usize) -> Array2<i64> {
    Array2::from_shape_fn((n_periods, n_periods), |(t, s)| {
        (t as i64 - s as i64).abs()
    })
}

/// Generate control mask (u8): 1 where D=0, 0 where D=1.
fn generate_control_mask(d: &Array2<f64>) -> Array2<u8> {
    Array2::from_shape_fn(d.raw_dim(), |(t, i)| {
        if d[[t, i]] == 0.0 { 1u8 } else { 0u8 }
    })
}

// ---------------------------------------------------------------------------
// Benchmark 1: Distance matrix — small panel (N=10, T=20)
// ---------------------------------------------------------------------------

fn bench_distance_small(c: &mut Criterion) {
    let (y, d) = generate_panel_data(10, 20, 2, 15);

    c.bench_function("distance_matrix_N10_T20", |b| {
        b.iter(|| {
            black_box(compute_unit_distance_matrix_internal(
                &y.view(),
                &d.view(),
            ))
        })
    });
}

// ---------------------------------------------------------------------------
// Benchmark 2: Distance matrix — medium panel (N=50, T=30)
// ---------------------------------------------------------------------------

fn bench_distance_medium(c: &mut Criterion) {
    let (y, d) = generate_panel_data(50, 30, 5, 20);

    c.bench_function("distance_matrix_N50_T30", |b| {
        b.iter(|| {
            black_box(compute_unit_distance_matrix_internal(
                &y.view(),
                &d.view(),
            ))
        })
    });
}

// ---------------------------------------------------------------------------
// Benchmark 3: Distance matrix — large panel (N=200, T=50)
// ---------------------------------------------------------------------------

fn bench_distance_large(c: &mut Criterion) {
    let (y, d) = generate_panel_data(200, 50, 20, 35);

    c.bench_function("distance_matrix_N200_T50", |b| {
        b.iter(|| {
            black_box(compute_unit_distance_matrix_internal(
                &y.view(),
                &d.view(),
            ))
        })
    });
}

// ---------------------------------------------------------------------------
// Benchmark 4: UnitDistanceCache build + query
// ---------------------------------------------------------------------------

fn bench_distance_cache(c: &mut Criterion) {
    let (y, d) = generate_panel_data(50, 30, 5, 20);

    let mut group = c.benchmark_group("distance_cache");

    // Sub-bench: cache construction
    group.bench_function("build_N50_T30", |b| {
        b.iter(|| {
            black_box(UnitDistanceCache::build(&y.view(), &d.view()))
        })
    });

    // Sub-bench: single distance query (amortized over many pairs)
    let cache = UnitDistanceCache::build(&y.view(), &d.view());
    group.bench_function("query_100_pairs", |b| {
        b.iter(|| {
            let mut sum = 0.0_f64;
            for i in 0..10 {
                for j in 0..10 {
                    sum += cache.compute_distance(i, j, Some(5));
                }
            }
            black_box(sum)
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark 5: Weight matrix computation (Twostep)
// ---------------------------------------------------------------------------

fn bench_weights_computation(c: &mut Criterion) {
    let (y, d) = generate_panel_data(50, 30, 5, 20);
    let time_dist = generate_time_dist(30);

    let n_periods = 30;
    let n_units = 50;
    let target_unit = 48; // A treated unit
    let target_period = 25; // A treated period
    let lambda_time = 0.5;
    let lambda_unit = 1.0;

    c.bench_function("weight_matrix_N50_T30", |b| {
        b.iter(|| {
            black_box(compute_weight_matrix(
                &y.view(),
                &d.view(),
                n_periods,
                n_units,
                target_unit,
                target_period,
                lambda_time,
                lambda_unit,
                &time_dist.view(),
            ))
        })
    });
}

// ---------------------------------------------------------------------------
// Benchmark 6: Full estimation flow — small panel (Twostep, single obs)
// ---------------------------------------------------------------------------

fn bench_estimation_full(c: &mut Criterion) {
    // Small panel to keep each iteration manageable
    let n_units = 15;
    let n_periods = 20;
    let n_treated = 3;
    let treat_start = 15;

    let (y, d) = generate_panel_data(n_units, n_periods, n_treated, treat_start);
    let time_dist = generate_time_dist(n_periods);
    let control_mask = generate_control_mask(&d);

    // Build weight matrix for a specific treated observation
    let target_unit = n_units - 1; // Last unit (treated)
    let target_period = n_periods - 1; // Last period
    let lambda_time = 0.5;
    let lambda_unit = 1.0;
    let lambda_nn = 0.1;

    let w = compute_weight_matrix(
        &y.view(),
        &d.view(),
        n_periods,
        n_units,
        target_unit,
        target_period,
        lambda_time,
        lambda_unit,
        &time_dist.view(),
    );

    c.bench_function("estimate_model_N15_T20", |b| {
        b.iter(|| {
            black_box(estimate_model(
                &y.view(),
                &control_mask.view(),
                &w.view(),
                lambda_nn,
                n_periods,
                n_units,
                100,   // max_iter
                1e-6,  // tol
                None,  // exclude_obs
                None,  // warm_start
                None,  // x (no covariates)
                None,  // gamma_init
            ))
        })
    });
}

// ---------------------------------------------------------------------------
// Benchmark 7: Scaling comparison across panel sizes
// ---------------------------------------------------------------------------

fn bench_distance_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_scaling");

    for &(n, t) in &[(10, 20), (30, 30), (50, 30), (100, 40), (200, 50)] {
        let (y, d) = generate_panel_data(n, t, n / 5, t * 3 / 4);
        group.bench_with_input(
            BenchmarkId::new("compute_matrix", format!("N{}_T{}", n, t)),
            &(y, d),
            |b, (y, d)| {
                b.iter(|| {
                    black_box(compute_unit_distance_matrix_internal(
                        &y.view(),
                        &d.view(),
                    ))
                })
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Group and main
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_distance_small,
    bench_distance_medium,
    bench_distance_large,
    bench_distance_cache,
    bench_weights_computation,
    bench_estimation_full,
    bench_distance_scaling,
);
criterion_main!(benches);

//! Benchmark LOOCV on the simulated 20×30 panel exported by the Python
//! baseline runner.  Mirrors the Stata probe and is meant to answer
//! "is the Rust LOOCV itself slow, or is Stata wrapping overhead?".
//!
//! Run:
//!   cargo run --release --example bench_loocv_simulated

use ndarray::Array2;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;
use trop_core::loocv::loocv_grid_search;

fn main() {
    let csv = "../../tests/diagnostic_v311/artifacts_python/data_simulated_twostep.csv";
    let f = File::open(csv).expect("open csv");
    let reader = BufReader::new(f);

    // header + 600 rows of (unit, time, outcome, treatment)
    let mut rows: Vec<(usize, usize, f64, f64)> = Vec::with_capacity(600);
    for (i, line) in reader.lines().enumerate() {
        let line = line.unwrap();
        if i == 0 {
            continue;
        } // skip header
        let mut it = line.split(',');
        let unit: usize = it.next().unwrap().parse().unwrap();
        let time: usize = it.next().unwrap().parse().unwrap();
        let outcome: f64 = it.next().unwrap().parse().unwrap();
        let treatment: f64 = it.next().unwrap().parse().unwrap();
        rows.push((unit, time, outcome, treatment));
    }

    let n_units = 1 + rows.iter().map(|r| r.0).max().unwrap();
    let n_periods = 1 + rows.iter().map(|r| r.1).max().unwrap();
    println!("panel: {} units × {} periods = {} cells", n_units, n_periods, n_units * n_periods);

    let mut y = Array2::<f64>::zeros((n_periods, n_units));
    let mut d = Array2::<f64>::zeros((n_periods, n_units));
    let mut time_dist = Array2::<i64>::zeros((n_periods, n_periods));
    for t1 in 0..n_periods {
        for t2 in 0..n_periods {
            time_dist[[t1, t2]] = (t1 as i64 - t2 as i64).abs();
        }
    }
    for (u, t, o, tr) in rows {
        y[[t, u]] = o;
        d[[t, u]] = tr;
    }

    // control_mask[t, i] == 1 iff unit i is ALWAYS untreated at t
    // (for simulated data: treatment is staggered after pre-period)
    let mut control_mask = Array2::<u8>::zeros((n_periods, n_units));
    for t in 0..n_periods {
        for i in 0..n_units {
            if d[[t, i]] == 0.0 {
                control_mask[[t, i]] = 1;
            }
        }
    }

    // Full paper grid (same as the Python baseline / Stata probe)
    let lt_grid = vec![0.0, 0.1, 0.5, 1.0, 2.0, 5.0];
    let lu_grid = vec![0.0, 0.1, 0.5, 1.0, 2.0, 5.0];
    let ln_grid = vec![0.0, 0.01, 0.1, 1.0, 10.0];

    println!("calling loocv_grid_search with cycling search, maxiter=100, tol=1e-6 …");
    let t0 = Instant::now();
    let (best_lt, best_lu, best_ln, best_score, n_valid, n_attempted, _) = loocv_grid_search(
        &y.view(),
        &d.view(),
        &control_mask.view(),
        &time_dist.view(),
        &lt_grid,
        &lu_grid,
        &ln_grid,
        100,
        1e-6,
        None,
    ).unwrap();
    let dt = t0.elapsed();
    println!(
        "  elapsed       : {:.2?}\n  best λ        : ({}, {}, {})\n  loocv_score   : {}\n  n_valid       : {}/{}",
        dt, best_lt, best_lu, best_ln, best_score, n_valid, n_attempted,
    );

    println!("\nrepeating with tiny 1x1x1 grid …");
    let t0 = Instant::now();
    let r = loocv_grid_search(
        &y.view(),
        &d.view(),
        &control_mask.view(),
        &time_dist.view(),
        &[0.0],
        &[0.5],
        &[0.01],
        100,
        1e-6,
        None,
    ).unwrap();
    println!("  elapsed       : {:.2?}  λ=({},{},{})  score={}", t0.elapsed(), r.0, r.1, r.2, r.3);
}

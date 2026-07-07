// Benchmark: LOOCV hot path at ddcg-like scale (T=51, N=175).
// Measures (1) per-cell LOO fit cost by lambda_nn regime, (2) sequential vs
// chunk-parallel candidate evaluation, (3) score equality of the parallel
// prototype.
use ndarray::{Array1, Array2};
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use std::time::Instant;
use trop_core::distance::UnitDistanceCache;
use trop_core::estimation::estimate_model;
use trop_core::loocv::{get_control_observations, loocv_score_for_params};
use trop_core::weights::compute_weight_matrix_cached_into;

fn main() {
    let t_n = 51usize;
    let n_u = 175usize;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(20260705);

    // Rank-3 factor model + noise.
    let f: Array2<f64> = Array2::from_shape_fn((t_n, 3), |_| rng.gen::<f64>() - 0.5);
    let g: Array2<f64> = Array2::from_shape_fn((3, n_u), |_| rng.gen::<f64>() - 0.5);
    let mut y = f.dot(&g);
    for v in y.iter_mut() {
        *v += 0.1 * (rng.gen::<f64>() - 0.5);
    }

    // Switching treatment: ~25% of units treated in random middle windows.
    let mut d = Array2::<f64>::zeros((t_n, n_u));
    for i in 0..n_u {
        if rng.gen::<f64>() < 0.25 {
            let start = rng.gen_range(5..t_n - 5);
            let len = rng.gen_range(3..(t_n - start));
            for t in start..start + len {
                d[[t, i]] = 1.0;
            }
        }
    }
    let control_mask = Array2::<u8>::from_shape_fn((t_n, n_u), |(t, i)| {
        if d[[t, i]] == 0.0 { 1 } else { 0 }
    });
    let time_dist =
        Array2::<i64>::from_shape_fn((t_n, t_n), |(s, t)| (s as i64 - t as i64).abs());

    let control_obs = get_control_observations(&y.view(), &control_mask.view());
    println!("control cells: {}", control_obs.len());

    let dist_cache = UnitDistanceCache::build_with_full_matrix(&y.view(), &d.view());
    let max_iter = 500usize;
    let tol = 1e-6f64;

    // (1) Per-cell fit cost by lambda_nn regime (first 30 cells, cold start).
    for &lnn in &[0.0f64, 0.05, 1.0] {
        let t0 = Instant::now();
        let mut buf = Array2::<f64>::zeros((t_n, n_u));
        for &(t, i) in control_obs.iter().take(30) {
            compute_weight_matrix_cached_into(
                &mut buf, &y.view(), &d.view(), &dist_cache, t_n, n_u, i, t, 0.1, 1.0,
                &time_dist.view(),
            );
            let _ = estimate_model(
                &y.view(), &control_mask.view(), &buf.view(), lnn, t_n, n_u, max_iter,
                tol, Some((t, i)), None, None, None,
            );
        }
        println!(
            "lambda_nn={:>4}: {:.1} ms/cell (cold)",
            lnn,
            t0.elapsed().as_secs_f64() * 1000.0 / 30.0
        );
    }

    // (2) Full sequential candidate evaluation (warm-start path, as shipped).
    let cand = (0.1f64, 1.0f64, 1.0f64);
    let t0 = Instant::now();
    let (score_seq, nv, _) = loocv_score_for_params(
        &y.view(), &d.view(), &control_mask.view(), &time_dist.view(), &dist_cache,
        &control_obs, cand.0, cand.1, cand.2, max_iter, tol, f64::INFINITY, None,
    );
    let seq_s = t0.elapsed().as_secs_f64();
    println!("sequential: {:.1} s, score={:.6e}, n_valid={}", seq_s, score_seq, nv);

    // (3) Chunk-parallel prototype: same math, chunks keep local warm starts.
    let nthreads = rayon::current_num_threads();
    let chunk = control_obs.len().div_ceil(nthreads * 4);
    let t0 = Instant::now();
    let partials: Vec<(f64, usize)> = control_obs
        .par_chunks(chunk)
        .map(|cells| {
            let mut buf = Array2::<f64>::zeros((t_n, n_u));
            let mut sum = 0.0f64;
            let mut n = 0usize;
            let mut wa: Option<Array1<f64>> = None;
            let mut wb: Option<Array1<f64>> = None;
            let mut wl: Option<Array2<f64>> = None;
            for &(t, i) in cells {
                compute_weight_matrix_cached_into(
                    &mut buf, &y.view(), &d.view(), &dist_cache, t_n, n_u, i, t, cand.0,
                    cand.1, &time_dist.view(),
                );
                let ws = match (&wa, &wb, &wl) {
                    (Some(a), Some(b), Some(l)) => Some((a, b, l)),
                    _ => None,
                };
                if let Some((alpha, beta, l, _, _, _)) = estimate_model(
                    &y.view(), &control_mask.view(), &buf.view(), cand.2, t_n, n_u,
                    max_iter, tol, Some((t, i)), ws, None, None,
                ) {
                    let tau = y[[t, i]] - alpha[i] - beta[t] - l[[t, i]];
                    sum += tau * tau;
                    n += 1;
                    wa = Some(alpha);
                    wb = Some(beta);
                    wl = Some(l);
                }
            }
            (sum, n)
        })
        .collect();
    let par_s = t0.elapsed().as_secs_f64();
    let score_par: f64 = partials.iter().map(|p| p.0).sum();
    let nv_par: usize = partials.iter().map(|p| p.1).sum();
    println!(
        "parallel ({} threads): {:.1} s, score={:.6e}, n_valid={}, speedup={:.2}x, rel_diff={:.2e}",
        nthreads, par_s, score_par, nv_par, seq_s / par_s,
        ((score_par - score_seq) / score_seq).abs()
    );
}

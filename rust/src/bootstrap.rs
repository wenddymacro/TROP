//! Bootstrap variance estimation for the TROP estimator (paper Algorithm 3).
//!
//! Implements unit-level block bootstrap with stratified resampling to
//! preserve the ratio of treated to control units.
//!
//! Procedure:
//!   1. For b = 1, ..., B: draw N_0 control units and N_1 treated units
//!      with replacement, independently (paper Alg 3 step 4).
//!   2. Re-estimate the TROP point estimator on each bootstrap sample
//!      holding λ̂ fixed (paper Alg 3 step 5 read as "apply the same final
//!      TROP pipeline"; a strict re-LOOCV reading is deferred to a future
//!      option).
//!   3. Report the Bessel-corrected sample variance (1/(B−1)) of the B
//!      bootstrap estimates.
//!
//! Variance denominator: paper Algorithm 3 writes
//!     V̂_τ = (1/B) · Σ (τ̂^(b) − τ̄)² ,
//! while `compute_bootstrap_variance` uses the 1/(B−1) Bessel-corrected
//! sample variance (unbiased under the bootstrap distribution).  The two
//! differ by a factor B/(B−1) ≈ 1 + 1/B, which is negligible for the
//! default B = 200 (≈ 0.5 %) and is the standard convention for
//! inferential bootstrap SEs.
//!
//! The PRNG is Xoshiro256PlusPlus, seeded deterministically per iteration.

use ndarray::{Array1, Array2, ArrayView2};
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;

use crate::distance::UnitDistanceCache;
use crate::error::{TropError, TropResult};
use crate::estimation::{
    debug_assert_delta_is_1minus_d_masked, estimate_model, estimate_model_into,
    solve_joint_no_lowrank, solve_joint_with_lowrank,
};
use crate::weights::{
    compute_joint_weights, compute_weight_matrix_cached, compute_weight_matrix_cached_into,
};

// ============================================================================
// Unit Classification
// ============================================================================

/// Unit classification for stratified bootstrap sampling.
#[derive(Debug, Clone)]
pub struct UnitClassification {
    /// Column indices of control units (D[t,i] = 0 for all t).
    pub control_units: Vec<usize>,
    /// Column indices of treated units (D[t,i] = 1 for some t).
    pub treated_units: Vec<usize>,
    /// Number of control units (N_0).
    pub n_control: usize,
    /// Number of treated units (N_1).
    pub n_treated: usize,
}

/// Partition units into control and treated groups.
///
/// A unit is classified as treated if D[t,i] = 1 for any period t,
/// and as control otherwise.
///
/// # Arguments
/// * `d` - Treatment indicator matrix (T × N).
///
/// # Returns
/// A [`UnitClassification`] with separated index vectors.
pub fn classify_units(d: &ArrayView2<f64>) -> UnitClassification {
    let n_periods = d.nrows();
    let n_units = d.ncols();

    let mut control_units: Vec<usize> = Vec::new();
    let mut treated_units: Vec<usize> = Vec::new();

    for i in 0..n_units {
        let is_ever_treated = (0..n_periods).any(|t| d[[t, i]] == 1.0);
        if is_ever_treated {
            treated_units.push(i);
        } else {
            control_units.push(i);
        }
    }

    let n_control = control_units.len();
    let n_treated = treated_units.len();

    UnitClassification {
        control_units,
        treated_units,
        n_control,
        n_treated,
    }
}

// ============================================================================
// Stratified Sampling
// ============================================================================

/// Draw a stratified bootstrap sample for one iteration.
///
/// Independently resamples N_0 control units and N_1 treated units
/// with replacement, preserving the group ratio.
///
/// # Arguments
/// * `classification` - Pre-computed unit partition.
/// * `seed` - Deterministic seed for this iteration.
///
/// # Returns
/// Vector of sampled column indices (length N_0 + N_1).
pub fn stratified_sample(classification: &UnitClassification, seed: u64) -> Vec<usize> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut sampled_units: Vec<usize> =
        Vec::with_capacity(classification.n_control + classification.n_treated);

    // Sample control units with replacement
    for _ in 0..classification.n_control {
        if classification.n_control > 0 {
            let idx = rng.gen_range(0..classification.n_control);
            sampled_units.push(classification.control_units[idx]);
        }
    }

    // Sample treated units with replacement
    for _ in 0..classification.n_treated {
        if classification.n_treated > 0 {
            let idx = rng.gen_range(0..classification.n_treated);
            sampled_units.push(classification.treated_units[idx]);
        }
    }

    sampled_units
}

// ============================================================================
// Bootstrap Matrix Construction
// ============================================================================

/// Construct bootstrap outcome and treatment matrices by column selection.
///
/// Selects columns from Y and D according to `sampled_units`, preserving
/// all T rows per unit (block bootstrap at the unit level).
///
/// # Arguments
/// * `y` - Outcome matrix (T × N).
/// * `d` - Treatment indicator matrix (T × N).
/// * `sampled_units` - Column indices drawn by stratified sampling.
///
/// # Returns
/// `(Y_boot, D_boot)` — bootstrap matrices of dimension T × len(sampled_units).
pub fn build_bootstrap_matrices(
    y: &Array2<f64>,
    d: &Array2<f64>,
    sampled_units: &[usize],
) -> (Array2<f64>, Array2<f64>) {
    let n_periods = y.nrows();
    let n_units = sampled_units.len();

    let mut y_boot = Array2::<f64>::zeros((n_periods, n_units));
    let mut d_boot = Array2::<f64>::zeros((n_periods, n_units));

    for (new_idx, &old_idx) in sampled_units.iter().enumerate() {
        for t in 0..n_periods {
            y_boot[[t, new_idx]] = y[[t, old_idx]];
            d_boot[[t, new_idx]] = d[[t, old_idx]];
        }
    }

    (y_boot, d_boot)
}

/// Construct bootstrap matrices including the control-observation mask.
///
/// Same column-selection logic as [`build_bootstrap_matrices`], with an
/// additional binary mask indicating control observations (used by the
/// twostep estimator).
///
/// # Arguments
/// * `y` - Outcome matrix (T × N).
/// * `d` - Treatment indicator matrix (T × N).
/// * `control_mask` - Binary mask for control observations (T × N).
/// * `sampled_units` - Column indices drawn by stratified sampling.
///
/// # Returns
/// `(Y_boot, D_boot, control_mask_boot)`.
pub fn build_bootstrap_matrices_with_mask(
    y: &Array2<f64>,
    d: &Array2<f64>,
    control_mask: &Array2<u8>,
    sampled_units: &[usize],
) -> (Array2<f64>, Array2<f64>, Array2<u8>) {
    let n_periods = y.nrows();
    let n_units = sampled_units.len();

    let mut y_boot = Array2::<f64>::zeros((n_periods, n_units));
    let mut d_boot = Array2::<f64>::zeros((n_periods, n_units));
    let mut control_mask_boot = Array2::<u8>::zeros((n_periods, n_units));

    for (new_idx, &old_idx) in sampled_units.iter().enumerate() {
        for t in 0..n_periods {
            y_boot[[t, new_idx]] = y[[t, old_idx]];
            d_boot[[t, new_idx]] = d[[t, old_idx]];
            control_mask_boot[[t, new_idx]] = control_mask[[t, old_idx]];
        }
    }

    (y_boot, d_boot, control_mask_boot)
}

/// Pre-allocated version of [`build_bootstrap_matrices`].
///
/// Instead of allocating new matrices each call, writes into caller-provided
/// buffers.  The buffers must have shape `(n_periods, max_units)` where
/// `max_units >= sampled_units.len()`.  The caller should use only the first
/// `sampled_units.len()` columns of each buffer after this call.
///
/// # Safety Invariant
/// The entire buffer is zeroed at the start of each call to prevent data
/// leakage from prior iterations (critical when `sampled_units.len()` varies).
///
/// # Returns
/// The number of columns actually written (`sampled_units.len()`).
pub fn build_bootstrap_matrices_into(
    y_buf: &mut Array2<f64>,
    d_buf: &mut Array2<f64>,
    y: &Array2<f64>,
    d: &Array2<f64>,
    sampled_units: &[usize],
) -> usize {
    debug_assert_eq!(y_buf.nrows(), y.nrows());
    debug_assert_eq!(d_buf.nrows(), d.nrows());
    debug_assert!(y_buf.ncols() >= sampled_units.len());
    debug_assert!(d_buf.ncols() >= sampled_units.len());

    let n_periods = y.nrows();
    let n_units = sampled_units.len();

    // Safety: explicit full-buffer zeroing to prevent data leakage
    y_buf.fill(0.0);
    d_buf.fill(0.0);

    for (new_idx, &old_idx) in sampled_units.iter().enumerate() {
        for t in 0..n_periods {
            y_buf[[t, new_idx]] = y[[t, old_idx]];
            d_buf[[t, new_idx]] = d[[t, old_idx]];
        }
    }

    n_units
}

/// Pre-allocated version of [`build_bootstrap_matrices_with_mask`].
///
/// Writes into caller-provided buffers including the control mask.
/// All buffers are zeroed before filling to prevent cross-iteration leakage.
///
/// # Returns
/// The number of columns actually written (`sampled_units.len()`).
pub fn build_bootstrap_matrices_with_mask_into(
    y_buf: &mut Array2<f64>,
    d_buf: &mut Array2<f64>,
    mask_buf: &mut Array2<u8>,
    y: &Array2<f64>,
    d: &Array2<f64>,
    control_mask: &Array2<u8>,
    sampled_units: &[usize],
) -> usize {
    debug_assert_eq!(y_buf.nrows(), y.nrows());
    debug_assert_eq!(d_buf.nrows(), d.nrows());
    debug_assert!(y_buf.ncols() >= sampled_units.len());
    debug_assert!(d_buf.ncols() >= sampled_units.len());

    let n_periods = y.nrows();
    let n_units = sampled_units.len();

    // Safety: explicit full-buffer zeroing to prevent data leakage
    y_buf.fill(0.0);
    d_buf.fill(0.0);
    mask_buf.fill(0);

    for (new_idx, &old_idx) in sampled_units.iter().enumerate() {
        for t in 0..n_periods {
            y_buf[[t, new_idx]] = y[[t, old_idx]];
            d_buf[[t, new_idx]] = d[[t, old_idx]];
            mask_buf[[t, new_idx]] = control_mask[[t, old_idx]];
        }
    }

    n_units
}

// ============================================================================
// Variance and CI Calculation
// ============================================================================

/// Aggregate per-cell treatment effects into a (possibly weighted) ATT.
///
/// When `unit_weights` is `Some(w)`, each (tau, column_index) pair is
/// aggregated as `τ̂ = Σ w[j] · τ_j / Σ w[j]` where `j` is the column index
/// within the panel being aggregated.  When `unit_weights` is `None`, the
/// aggregation collapses to the ordinary mean `τ̂ = Σ τ / n`.
///
/// Returns `None` when the effective denominator is zero or when the input
/// slice is empty.
///
/// # Arguments
/// * `tau_cells` - Slice of `(tau, column_index)` pairs for treated cells.
/// * `unit_weights` - Optional per-column weights (length ≥ max column index).
fn aggregate_att(tau_cells: &[(f64, usize)], unit_weights: Option<&[f64]>) -> Option<f64> {
    if tau_cells.is_empty() {
        return None;
    }
    match unit_weights {
        Some(w) => {
            let mut num = 0.0_f64;
            let mut den = 0.0_f64;
            for (tau, idx) in tau_cells {
                let wi = w.get(*idx).copied().unwrap_or(0.0);
                if !wi.is_finite() || wi <= 0.0 {
                    continue;
                }
                num += wi * tau;
                den += wi;
            }
            if den > 0.0 {
                Some(num / den)
            } else {
                None
            }
        }
        None => {
            let sum: f64 = tau_cells.iter().map(|(t, _)| t).sum();
            Some(sum / tau_cells.len() as f64)
        }
    }
}

/// Compute the sample variance and standard error of bootstrap estimates.
///
/// `ddof` selects the denominator:
///   - `1` (default, "sample"): `1/(B−1)` — Bessel-corrected sample
///     variance, unbiased estimator of the bootstrap variance and the
///     standard convention for inferential bootstrap SEs.
///   - `0` ("population"): `1/B` — matches paper Algorithm 3's `V̂_τ`
///     exactly (population variance).
///   - Values `ddof ≥ 2` fall back to `ddof = 1`.
///
/// The two denominators differ by a factor `B/(B−1)`, negligible for the
/// default `B = 200` (≈ 0.5 %) but visible for small `B` (≈ 2 % at B = 50).
///
/// Filters non-finite values before computing statistics (failed bootstrap
/// iterations yield NaN or ±Inf).
///
/// # Arguments
/// * `estimates` - Bootstrap ATT estimates (may contain NaN/Inf).
/// * `ddof`      - Variance denominator offset; see above.
///
/// # Returns
/// `(mean, variance, se)`. Returns `(0, 0, 0)` when fewer than two
/// finite values are available (variance undefined).
pub fn compute_bootstrap_variance(estimates: &[f64], ddof: u8) -> (f64, f64, f64) {
    // Filter non-finite values before computing statistics.
    let finite: Vec<f64> = estimates.iter().copied().filter(|x| x.is_finite()).collect();

    if finite.len() < 2 {
        let m = if finite.is_empty() { 0.0 } else { finite[0] };
        return (m, 0.0, 0.0);
    }

    let n = finite.len() as f64;
    let mean = finite.iter().sum::<f64>() / n;
    // ddof = 0 → paper Algorithm 3 population variance (1/B).
    // ddof ≥ 1 → Bessel-corrected sample variance (1/(B−1)); any value
    // other than 0 collapses to 1 since higher ddof has no meaning for
    // the bootstrap variance definition.
    let denom = if ddof == 0 {
        n
    } else {
        (n - 1.0).max(1.0)
    };
    let variance = finite.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / denom;
    let se = variance.sqrt();

    (mean, variance, se)
}

/// Compute a percentile confidence interval from bootstrap estimates.
///
/// Returns the α/2 and 1−α/2 quantiles of the finite bootstrap values.
/// Fractional indices are resolved by linear interpolation:
/// for index f, the quantile is `sorted[⌊f⌋]·(1−frac) + sorted[⌈f⌉]·frac`.
///
/// # Arguments
/// * `estimates` - Bootstrap ATT estimates (may contain NaN/Inf).
/// * `alpha` - Significance level (e.g. 0.05 for a 95 % CI).
///
/// # Returns
/// `(ci_lower, ci_upper)`. Returns `(NaN, NaN)` when no finite values exist.
pub fn compute_percentile_ci(estimates: &[f64], alpha: f64) -> (f64, f64) {
    // Filter non-finite values before computing percentiles.
    let mut sorted: Vec<f64> = estimates.iter().copied().filter(|x| x.is_finite()).collect();
    if sorted.is_empty() {
        return (f64::NAN, f64::NAN);
    }

    // partial_cmp is total after NaN removal.
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("non-finite values already removed"));

    let n = sorted.len();

    // Percentile index: p-th quantile at position (n−1)·p.
    let lower_p = alpha / 2.0;
    let upper_p = 1.0 - alpha / 2.0;

    let lower_idx_f = (n - 1) as f64 * lower_p;
    let upper_idx_f = (n - 1) as f64 * upper_p;

    // Linear interpolation for fractional indices.
    let ci_lower = interpolate_percentile(&sorted, lower_idx_f);
    let ci_upper = interpolate_percentile(&sorted, upper_idx_f);

    (ci_lower, ci_upper)
}

/// Linearly interpolate a quantile from a sorted slice.
///
/// For fractional index f, returns
/// `sorted[⌊f⌋] * (1 − frac) + sorted[⌈f⌉] * frac`.
fn interpolate_percentile(sorted: &[f64], idx_f: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return sorted[0];
    }

    let idx_low = idx_f.floor() as usize;
    let idx_high = idx_f.ceil() as usize;

    let idx_low = idx_low.min(n - 1);
    let idx_high = idx_high.min(n - 1);

    if idx_low == idx_high {
        sorted[idx_low]
    } else {
        let frac = idx_f - idx_low as f64;
        sorted[idx_low] * (1.0 - frac) + sorted[idx_high] * frac
    }
}

// ============================================================================
// Bootstrap Result Structure
// ============================================================================

/// Aggregated bootstrap inference results.
#[derive(Debug, Clone)]
pub struct BootstrapResult {
    /// Finite bootstrap ATT estimates retained after filtering.
    pub estimates: Vec<f64>,
    /// Standard error (√ of sample variance with Bessel correction, 1/(B−1));
    /// paper Algorithm 3 uses 1/B (population variance).
    pub se: f64,
    /// Mean of the bootstrap distribution.
    pub mean: f64,
    /// Lower bound of the percentile confidence interval.
    pub ci_lower: f64,
    /// Upper bound of the percentile confidence interval.
    pub ci_upper: f64,
    /// Number of iterations that produced a finite estimate.
    pub n_valid: usize,
    /// Total number of bootstrap iterations attempted.
    pub n_total: usize,
    /// Nominal confidence level (1 − α).
    pub level: f64,
}

// ============================================================================
// Main Bootstrap Functions
// ============================================================================

/// Bootstrap variance for the twostep estimator (convenience wrapper).
///
/// Delegates to [`bootstrap_trop_variance_full`] and returns only the
/// vector of bootstrap estimates and the standard error.
///
/// # Arguments
/// * `y` - Outcome matrix (T × N).
/// * `d` - Treatment indicator matrix (T × N).
/// * `control_mask` - Binary mask for control observations (T × N).
/// * `time_dist` - Pre-computed time distance matrix (T × T).
/// * `lambda_time` - Time kernel bandwidth (Inf → uniform weights).
/// * `lambda_unit` - Unit kernel bandwidth (Inf → uniform weights).
/// * `lambda_nn` - Nuclear norm penalty (Inf → no low-rank component).
/// * `n_bootstrap` - Number of bootstrap replications B.
/// * `max_iter` - Maximum ADMM iterations per estimation.
/// * `tol` - Convergence tolerance for ADMM.
/// * `seed` - Base random seed.
/// * `alpha` - Significance level for the percentile CI.
///
/// # Returns
///
/// `(estimates, se)` where `estimates` has length ≤ B.
#[allow(clippy::too_many_arguments)]
pub fn bootstrap_trop_variance(
    y: &ArrayView2<f64>,
    d: &ArrayView2<f64>,
    control_mask: &ArrayView2<u8>,
    time_dist: &ArrayView2<i64>,
    lambda_time: f64,
    lambda_unit: f64,
    lambda_nn: f64,
    n_bootstrap: usize,
    max_iter: usize,
    tol: f64,
    seed: u64,
    alpha: f64,
    x: Option<&ArrayView2<f64>>,
) -> (Array1<f64>, f64) {
    // Legacy wrapper: historical callers assumed Bessel-corrected SE
    // (ddof = 1).  The `_full` variant now accepts an explicit ddof; this
    // wrapper pins ddof = 1 for backward compatibility.
    let result = bootstrap_trop_variance_full(
        y,
        d,
        control_mask,
        time_dist,
        lambda_time,
        lambda_unit,
        lambda_nn,
        n_bootstrap,
        max_iter,
        tol,
        seed,
        alpha,
        1,
        x,
    );
    (Array1::from_vec(result.estimates), result.se)
}

/// Bootstrap variance estimation for the twostep method with full output.
///
/// For each replication b = 1, …, B:
///   1. Draw a stratified sample of N_0 control + N_1 treated units.
///   2. Estimate per-observation treatment effects on the bootstrap data.
///   3. Average over treated observations to obtain τ̂_b.
///
/// Returns the complete [`BootstrapResult`] including percentile CI.
///
/// # Arguments
/// See [`bootstrap_trop_variance`] — all parameters are identical, plus
/// `ddof` selecting the variance denominator:
///   - `1` → Bessel-corrected sample variance `1/(B−1)` (default).
///   - `0` → paper Algorithm 3 population variance `1/B`.
///
/// # Returns
/// A [`BootstrapResult`] containing estimates, SE, mean, and CI.
#[allow(clippy::too_many_arguments)]
pub fn bootstrap_trop_variance_full(
    y: &ArrayView2<f64>,
    d: &ArrayView2<f64>,
    control_mask: &ArrayView2<u8>,
    time_dist: &ArrayView2<i64>,
    lambda_time: f64,
    lambda_unit: f64,
    lambda_nn: f64,
    n_bootstrap: usize,
    max_iter: usize,
    tol: f64,
    seed: u64,
    alpha: f64,
    ddof: u8,
    x: Option<&ArrayView2<f64>>,
) -> BootstrapResult {
    let y_arr = y.to_owned();
    let d_arr = d.to_owned();
    let control_mask_arr = control_mask.to_owned();
    let time_dist_arr = time_dist.to_owned();
    let x_arr = x.map(|xv| xv.to_owned());

    let n_periods = y_arr.nrows();
    let n_units = y_arr.ncols();

    // Map infinite lambda values to effective computation values:
    //   lambda_time/unit = Inf  →  0.0  (uniform kernel weights)
    //   lambda_nn        = Inf  →  1e10 (effectively no low-rank penalty)
    let lt_eff = if lambda_time.is_infinite() {
        0.0
    } else {
        lambda_time
    };
    let lu_eff = if lambda_unit.is_infinite() {
        0.0
    } else {
        lambda_unit
    };
    let ln_eff = if lambda_nn.is_infinite() {
        1e10
    } else {
        lambda_nn
    };

    let classification = classify_units(&d_arr.view());

    // Stratified sampling always draws n_control + n_treated = n_units columns.
    let n_boot_units = classification.n_control + classification.n_treated;
    let n_cov = x_arr.as_ref().map_or(0, |xv| xv.ncols());

    // Parallel bootstrap with per-task buffer reuse: each rayon task
    // pre-allocates matrices once and reuses them across all iterations it
    // processes, reducing heap allocations from O(B) to O(num_threads).
    //
    // Buffers reused across iterations: Y, D, mask (bootstrap matrices),
    // weight_buf, alpha_buf, beta_buf, L_buf (estimation inner-loop).
    let bootstrap_estimates: Vec<f64> = (0..n_bootstrap)
        .into_par_iter()
        .fold(
            || {
                let x_buf = x_arr.as_ref().map(|xv| {
                    Array2::<f64>::zeros((n_periods * n_boot_units, xv.ncols()))
                });
                let gamma_buf = if n_cov > 0 {
                    Some(Array1::<f64>::zeros(n_cov))
                } else {
                    None
                };
                (
                    Vec::<f64>::new(),
                    Array2::<f64>::zeros((n_periods, n_boot_units)),
                    Array2::<f64>::zeros((n_periods, n_boot_units)),
                    Array2::<u8>::zeros((n_periods, n_boot_units)),
                    Array2::<f64>::zeros((n_periods, n_boot_units)),
                    Array1::<f64>::zeros(n_boot_units),
                    Array1::<f64>::zeros(n_periods),
                    Array2::<f64>::zeros((n_periods, n_boot_units)),
                    gamma_buf,
                    x_buf,
                )
            },
            |(
                mut results,
                mut y_buf,
                mut d_buf,
                mut mask_buf,
                mut weight_buf,
                mut alpha_buf,
                mut beta_buf,
                mut l_buf,
                mut gamma_buf,
                mut x_buf,
            ), b| {
                let iteration_seed = seed.wrapping_add(b as u64);
                let sampled_units = stratified_sample(&classification, iteration_seed);

                build_bootstrap_matrices_with_mask_into(
                    &mut y_buf, &mut d_buf, &mut mask_buf,
                    &y_arr, &d_arr, &control_mask_arr,
                    &sampled_units,
                );

                // Resample X matrix by unit if covariates are present.
                if let (Some(ref x_owned), Some(ref mut xb)) = (&x_arr, &mut x_buf) {
                    xb.fill(0.0);
                    for (j, &orig_unit) in sampled_units.iter().enumerate() {
                        for t in 0..n_periods {
                            let src_idx = t * n_units + orig_unit;
                            let dst_idx = t * n_boot_units + j;
                            xb.row_mut(dst_idx).assign(&x_owned.row(src_idx));
                        }
                    }
                }

                // Identify treated (t, i) pairs in the bootstrap sample.
                let mut boot_treated: Vec<(usize, usize)> = Vec::new();
                for t in 0..n_periods {
                    for i in 0..n_boot_units {
                        if d_buf[[t, i]] == 1.0 {
                            boot_treated.push((t, i));
                        }
                    }
                }

                if boot_treated.is_empty() {
                    return (results, y_buf, d_buf, mask_buf, weight_buf, alpha_buf, beta_buf, l_buf, gamma_buf, x_buf);
                }

                // Identify control units (never treated in bootstrap sample).
                let mut boot_control_units: Vec<usize> = Vec::new();
                for i in 0..n_boot_units {
                    let is_control = (0..n_periods).all(|t| d_buf[[t, i]] == 0.0);
                    if is_control {
                        boot_control_units.push(i);
                    }
                }

                if boot_control_units.is_empty() {
                    return (results, y_buf, d_buf, mask_buf, weight_buf, alpha_buf, beta_buf, l_buf, gamma_buf, x_buf);
                }

                // Build a per-iteration distance cache.  The cache is O(N·T) and
                // shape-dependent on the resampled panel, so it is rebuilt per draw.
                let dist_cache = UnitDistanceCache::build(&y_buf.view(), &d_buf.view());

                // Estimate τ(t,i) for each treated observation.
                let mut tau_values = Vec::with_capacity(boot_treated.len());

                // Warm start: reuse previous observation's converged solution as
                // initial values for the next.  Under a fixed lambda triplet the
                // weight matrices (and thus optima) are similar across treated
                // observations, so warm-starting reduces FISTA iterations.
                let mut warm_alpha: Option<Array1<f64>> = None;
                let mut warm_beta: Option<Array1<f64>> = None;
                let mut warm_l: Option<Array2<f64>> = None;

                for (t, i) in boot_treated {
                    compute_weight_matrix_cached_into(
                        &mut weight_buf,
                        &y_buf.view(),
                        &d_buf.view(),
                        &dist_cache,
                        n_periods,
                        n_boot_units,
                        i,
                        t,
                        lt_eff,
                        lu_eff,
                        &time_dist_arr.view(),
                    );

                    let ws = match (&warm_alpha, &warm_beta, &warm_l) {
                        (Some(a), Some(b), Some(l_prev)) => Some((a, b, l_prev)),
                        _ => None,
                    };

                    let x_view = x_buf.as_ref().map(|xb| xb.view());
                    if let Some((_iters, _converged)) = estimate_model_into(
                        &y_buf.view(),
                        &mask_buf.view(),
                        &weight_buf.view(),
                        ln_eff,
                        n_periods,
                        n_boot_units,
                        max_iter,
                        tol,
                        None,
                        ws,
                        x_view.as_ref(),
                        None,
                        &mut alpha_buf,
                        &mut beta_buf,
                        &mut l_buf,
                        gamma_buf.as_mut(),
                    ) {
                        let mut tau = y_buf[[t, i]] - alpha_buf[i] - beta_buf[t] - l_buf[[t, i]];
                        if let Some(ref xb) = x_buf {
                            if let Some(ref g) = gamma_buf {
                                let idx = t * n_boot_units + i;
                                tau -= xb.row(idx).dot(g);
                            }
                        }
                        if tau.is_finite() {
                            tau_values.push(tau);
                        }

                        // Stash converged solution for warm-starting the next observation.
                        warm_alpha = Some(alpha_buf.clone());
                        warm_beta = Some(beta_buf.clone());
                        warm_l = Some(l_buf.clone());
                    }
                }

                if !tau_values.is_empty() {
                    let att = tau_values.iter().sum::<f64>() / tau_values.len() as f64;
                    if att.is_finite() {
                        results.push(att);
                    }
                }

                (results, y_buf, d_buf, mask_buf, weight_buf, alpha_buf, beta_buf, l_buf, gamma_buf, x_buf)
            },
        )
        .map(|(results, ..)| results)
        .reduce(Vec::new, |mut a, b| { a.extend(b); a });

    let n_valid = bootstrap_estimates.len();

    let (mean, _, se) = compute_bootstrap_variance(&bootstrap_estimates, ddof);
    let (ci_lower, ci_upper) = compute_percentile_ci(&bootstrap_estimates, alpha);

    BootstrapResult {
        estimates: bootstrap_estimates,
        se,
        mean,
        ci_lower,
        ci_upper,
        n_valid,
        n_total: n_bootstrap,
        level: 1.0 - alpha,
    }
}

/// Bootstrap variance estimation for the joint method (convenience wrapper).
///
/// Delegates to [`bootstrap_trop_variance_joint_full`] and returns only the
/// estimate vector and standard error.
///
/// # Arguments
/// * `y` - Outcome matrix (T × N).
/// * `d` - Treatment indicator matrix (T × N).
/// * `lambda_time` - Time kernel bandwidth (Inf → uniform weights).
/// * `lambda_unit` - Unit kernel bandwidth (Inf → uniform weights).
/// * `lambda_nn` - Nuclear norm penalty (Inf → no low-rank component).
/// * `n_bootstrap` - Number of bootstrap replications B.
/// * `max_iter` - Maximum ADMM iterations per estimation.
/// * `tol` - Convergence tolerance for ADMM.
/// * `seed` - Base random seed.
/// * `alpha` - Significance level for the percentile CI.
///
/// # Returns
///
/// `(estimates, se)` where `estimates` has length ≤ B.
#[allow(clippy::too_many_arguments)]
pub fn bootstrap_trop_variance_joint(
    y: &ArrayView2<f64>,
    d: &ArrayView2<f64>,
    lambda_time: f64,
    lambda_unit: f64,
    lambda_nn: f64,
    n_bootstrap: usize,
    max_iter: usize,
    tol: f64,
    seed: u64,
    alpha: f64,
    x: Option<&ArrayView2<f64>>,
) -> (Array1<f64>, f64) {
    // Legacy wrapper: historical callers assumed Bessel-corrected SE
    // (ddof = 1).  The `_full` variant now accepts an explicit ddof; this
    // wrapper pins ddof = 1 for backward compatibility.
    let result = bootstrap_trop_variance_joint_full(
        y,
        d,
        lambda_time,
        lambda_unit,
        lambda_nn,
        n_bootstrap,
        max_iter,
        tol,
        seed,
        alpha,
        1,
        x,
    );
    (Array1::from_vec(result.estimates), result.se)
}

/// Bootstrap variance estimation for the joint method with full output.
///
/// For each replication b = 1, …, B:
///   1. Draw a stratified sample of N_0 control + N_1 treated units.
///   2. Compute global joint weights on the bootstrap data.
///   3. Solve the joint WLS problem to obtain τ̂_b.
///
/// Returns the complete [`BootstrapResult`] including percentile CI.
///
/// # Arguments
/// See [`bootstrap_trop_variance_joint`] — all parameters are identical,
/// plus `ddof` selecting the variance denominator (see
/// [`bootstrap_trop_variance_full`]).
///
/// # Returns
/// A [`BootstrapResult`] containing estimates, SE, mean, and CI.
#[allow(clippy::too_many_arguments)]
pub fn bootstrap_trop_variance_joint_full(
    y: &ArrayView2<f64>,
    d: &ArrayView2<f64>,
    lambda_time: f64,
    lambda_unit: f64,
    lambda_nn: f64,
    n_bootstrap: usize,
    max_iter: usize,
    tol: f64,
    seed: u64,
    alpha: f64,
    ddof: u8,
    x: Option<&ArrayView2<f64>>,
) -> BootstrapResult {
    let y_arr = y.to_owned();
    let d_arr = d.to_owned();
    let x_arr = x.map(|xv| xv.to_owned());

    let n_units = y_arr.ncols();
    let n_periods = y_arr.nrows();

    let classification = classify_units(&d_arr.view());

    // Determine the number of treated periods from the original D matrix.
    // This count is fixed across all bootstrap draws.
    let mut first_treat_period = n_periods;
    for t in 0..n_periods {
        for i in 0..n_units {
            if d_arr[[t, i]] == 1.0 {
                first_treat_period = first_treat_period.min(t);
                break;
            }
        }
    }
    let treated_periods = n_periods.saturating_sub(first_treat_period);

    // Map infinite lambda values to effective computation values.
    let lt_eff = if lambda_time.is_infinite() {
        0.0
    } else {
        lambda_time
    };
    let lu_eff = if lambda_unit.is_infinite() {
        0.0
    } else {
        lambda_unit
    };
    let ln_eff = if lambda_nn.is_infinite() {
        1e10
    } else {
        lambda_nn
    };

    // Stratified sampling always draws n_control + n_treated = n_units columns.
    let n_boot_units = classification.n_control + classification.n_treated;

    // Parallel bootstrap with per-task buffer reuse: each rayon task
    // pre-allocates matrices once and reuses them across all iterations it
    // processes, reducing heap allocations from O(B) to O(num_threads).
    let bootstrap_estimates: Vec<f64> = (0..n_bootstrap)
        .into_par_iter()
        .fold(
            || {
                let x_buf = x_arr.as_ref().map(|xv| {
                    Array2::<f64>::zeros((n_periods * n_boot_units, xv.ncols()))
                });
                (
                    Vec::<f64>::new(),
                    Array2::<f64>::zeros((n_periods, n_boot_units)),
                    Array2::<f64>::zeros((n_periods, n_boot_units)),
                    x_buf,
                )
            },
            |(mut results, mut y_buf, mut d_buf, mut x_buf), b| {
                let iteration_seed = seed.wrapping_add(b as u64);
                let sampled_units = stratified_sample(&classification, iteration_seed);

                build_bootstrap_matrices_into(
                    &mut y_buf, &mut d_buf,
                    &y_arr, &d_arr,
                    &sampled_units,
                );

                // Resample X matrix by unit if covariates are present.
                if let (Some(ref x_owned), Some(ref mut xb)) = (&x_arr, &mut x_buf) {
                    xb.fill(0.0);
                    for (j, &orig_unit) in sampled_units.iter().enumerate() {
                        for t in 0..n_periods {
                            let src_idx = t * n_units + orig_unit;
                            let dst_idx = t * n_boot_units + j;
                            xb.row_mut(dst_idx).assign(&x_owned.row(src_idx));
                        }
                    }
                }

                // Compute joint weights using the original treated-period count.
                let delta = compute_joint_weights(
                    &y_buf.view(),
                    &d_buf.view(),
                    lt_eff,
                    lu_eff,
                    treated_periods,
                );

                // Solve the joint model; branch on whether low-rank is active.
                //
                // τ is post-hoc: mean residual (Y − μ − α − β − L) over treated cells.
                // When λ_nn ≥ 1e10 we skip the low-rank fit and L ≡ 0.
                // B.2 defensive check: compute_joint_weights (1 − D)-masks δ.
                debug_assert_delta_is_1minus_d_masked(
                    &d_buf.view(), &delta.view(),
                    "bootstrap::joint/delta",
                );
                let x_view = x_buf.as_ref().map(|xb| xb.view());
                let result = if ln_eff >= 1e10 {
                    solve_joint_no_lowrank(
                        &y_buf.view(),
                        &delta.view(),
                        x_view.as_ref(),
                    ).map(
                        |(mu, alpha_est, beta, result_gamma)| {
                            let mut tau_sum = 0.0_f64;
                            let mut tau_count = 0usize;
                            for t in 0..n_periods {
                                for i in 0..n_boot_units {
                                    if d_buf[[t, i]] == 1.0 && y_buf[[t, i]].is_finite() {
                                        let mut tau_val = y_buf[[t, i]] - mu - alpha_est[i] - beta[t];
                                        if let Some(ref xb) = x_buf {
                                            if let Some(ref g) = result_gamma {
                                                let idx = t * n_boot_units + i;
                                                tau_val -= xb.row(idx).dot(g);
                                            }
                                        }
                                        tau_sum += tau_val;
                                        tau_count += 1;
                                    }
                                }
                            }
                            if tau_count > 0 {
                                tau_sum / tau_count as f64
                            } else {
                                f64::NAN
                            }
                        },
                    )
                } else {
                    solve_joint_with_lowrank(
                        &y_buf.view(),
                        &d_buf.view(),
                        &delta.view(),
                        ln_eff,
                        max_iter,
                        tol,
                        x_view.as_ref(),
                    )
                    .map(|(_, _, _, _, tau, _, _, _)| tau)
                };

                if let Some(tau) = result.filter(|t| t.is_finite()) {
                    results.push(tau);
                }

                (results, y_buf, d_buf, x_buf)
            },
        )
        .map(|(results, _, _, _)| results)
        .reduce(Vec::new, |mut a, b| { a.extend(b); a });

    let n_valid = bootstrap_estimates.len();

    let (mean, _, se) = compute_bootstrap_variance(&bootstrap_estimates, ddof);
    let (ci_lower, ci_upper) = compute_percentile_ci(&bootstrap_estimates, alpha);

    BootstrapResult {
        estimates: bootstrap_estimates,
        se,
        mean,
        ci_lower,
        ci_upper,
        n_valid,
        n_total: n_bootstrap,
        level: 1.0 - alpha,
    }
}

/// Weighted twostep bootstrap: pweight-only survey-design variant.
///
/// Follows the same procedure as [`bootstrap_trop_variance_full`] but
/// aggregates the per-cell τ into ATT as `τ̂ = Σ w_i τ_{t,i} / Σ w_i`,
/// where `w_i` is the pweight attached to the original unit index.
///
/// Per-cell estimation (α, β, L) is unchanged — the pweight enters only the
/// ATT aggregation.  This matches the Python reference implementation for
/// pweight-only survey designs (no strata / PSU / FPC Rao-Wu rescaling).
///
/// # Arguments
/// Same as [`bootstrap_trop_variance_full`], plus:
/// * `unit_weights` - Per-original-unit pweights (length = `n_units` of the
///   original panel, indexed by the pre-bootstrap column index).  Must be
///   strictly positive and constant within unit (enforced by the caller).
///
/// # Returns
/// A [`BootstrapResult`] with weighted ATT estimates and SE.
#[allow(clippy::too_many_arguments)]
pub fn bootstrap_trop_variance_full_weighted(
    y: &ArrayView2<f64>,
    d: &ArrayView2<f64>,
    control_mask: &ArrayView2<u8>,
    time_dist: &ArrayView2<i64>,
    lambda_time: f64,
    lambda_unit: f64,
    lambda_nn: f64,
    n_bootstrap: usize,
    max_iter: usize,
    tol: f64,
    seed: u64,
    alpha: f64,
    ddof: u8,
    unit_weights: &[f64],
    x: Option<&ArrayView2<f64>>,
) -> BootstrapResult {
    let y_arr = y.to_owned();
    let d_arr = d.to_owned();
    let control_mask_arr = control_mask.to_owned();
    let time_dist_arr = time_dist.to_owned();
    let unit_weights_owned: Vec<f64> = unit_weights.to_vec();
    let x_arr = x.map(|xv| xv.to_owned());

    let n_periods = y_arr.nrows();
    let n_units = y_arr.ncols();

    let lt_eff = if lambda_time.is_infinite() {
        0.0
    } else {
        lambda_time
    };
    let lu_eff = if lambda_unit.is_infinite() {
        0.0
    } else {
        lambda_unit
    };
    let ln_eff = if lambda_nn.is_infinite() {
        1e10
    } else {
        lambda_nn
    };

    let classification = classify_units(&d_arr.view());

    // Stratified sampling always draws n_control + n_treated = n_units columns.
    let n_boot_units = classification.n_control + classification.n_treated;
    let n_cov = x_arr.as_ref().map_or(0, |xv| xv.ncols());

    // Parallel bootstrap with per-task buffer reuse (same pattern as
    // bootstrap_trop_variance_full — see comments there).
    let bootstrap_estimates: Vec<f64> = (0..n_bootstrap)
        .into_par_iter()
        .fold(
            || {
                let x_buf = x_arr.as_ref().map(|xv| {
                    Array2::<f64>::zeros((n_periods * n_boot_units, xv.ncols()))
                });
                let gamma_buf = if n_cov > 0 {
                    Some(Array1::<f64>::zeros(n_cov))
                } else {
                    None
                };
                (
                    Vec::<f64>::new(),
                    Array2::<f64>::zeros((n_periods, n_boot_units)),
                    Array2::<f64>::zeros((n_periods, n_boot_units)),
                    Array2::<u8>::zeros((n_periods, n_boot_units)),
                    Array2::<f64>::zeros((n_periods, n_boot_units)),
                    Array1::<f64>::zeros(n_boot_units),
                    Array1::<f64>::zeros(n_periods),
                    Array2::<f64>::zeros((n_periods, n_boot_units)),
                    gamma_buf,
                    x_buf,
                )
            },
            |(
                mut results,
                mut y_buf,
                mut d_buf,
                mut mask_buf,
                mut weight_buf,
                mut alpha_buf,
                mut beta_buf,
                mut l_buf,
                mut gamma_buf,
                mut x_buf,
            ), b| {
                let iteration_seed = seed.wrapping_add(b as u64);
                let sampled_units = stratified_sample(&classification, iteration_seed);

                build_bootstrap_matrices_with_mask_into(
                    &mut y_buf, &mut d_buf, &mut mask_buf,
                    &y_arr, &d_arr, &control_mask_arr,
                    &sampled_units,
                );

                // Resample X matrix by unit if covariates are present.
                if let (Some(ref x_owned), Some(ref mut xb)) = (&x_arr, &mut x_buf) {
                    xb.fill(0.0);
                    for (j, &orig_unit) in sampled_units.iter().enumerate() {
                        for t in 0..n_periods {
                            let src_idx = t * n_units + orig_unit;
                            let dst_idx = t * n_boot_units + j;
                            xb.row_mut(dst_idx).assign(&x_owned.row(src_idx));
                        }
                    }
                }

                // Propagate per-unit pweights through the bootstrap resampling:
                // each resampled column `new_idx` inherits the weight of the
                // original unit `sampled_units[new_idx]`.
                let w_boot: Vec<f64> = sampled_units
                    .iter()
                    .map(|&orig_idx| unit_weights_owned.get(orig_idx).copied().unwrap_or(0.0))
                    .collect();

                let mut boot_treated: Vec<(usize, usize)> = Vec::new();
                for t in 0..n_periods {
                    for i in 0..n_boot_units {
                        if d_buf[[t, i]] == 1.0 {
                            boot_treated.push((t, i));
                        }
                    }
                }

                if boot_treated.is_empty() {
                    return (results, y_buf, d_buf, mask_buf, weight_buf, alpha_buf, beta_buf, l_buf, gamma_buf, x_buf);
                }

                let mut boot_control_units: Vec<usize> = Vec::new();
                for i in 0..n_boot_units {
                    let is_control = (0..n_periods).all(|t| d_buf[[t, i]] == 0.0);
                    if is_control {
                        boot_control_units.push(i);
                    }
                }

                if boot_control_units.is_empty() {
                    return (results, y_buf, d_buf, mask_buf, weight_buf, alpha_buf, beta_buf, l_buf, gamma_buf, x_buf);
                }

                let dist_cache = UnitDistanceCache::build(&y_buf.view(), &d_buf.view());

                let mut tau_cells: Vec<(f64, usize)> = Vec::with_capacity(boot_treated.len());

                // Warm start: reuse previous observation's converged solution.
                let mut warm_alpha: Option<Array1<f64>> = None;
                let mut warm_beta: Option<Array1<f64>> = None;
                let mut warm_l: Option<Array2<f64>> = None;

                for (t, i) in boot_treated {
                    compute_weight_matrix_cached_into(
                        &mut weight_buf,
                        &y_buf.view(),
                        &d_buf.view(),
                        &dist_cache,
                        n_periods,
                        n_boot_units,
                        i,
                        t,
                        lt_eff,
                        lu_eff,
                        &time_dist_arr.view(),
                    );

                    let ws = match (&warm_alpha, &warm_beta, &warm_l) {
                        (Some(a), Some(b), Some(l_prev)) => Some((a, b, l_prev)),
                        _ => None,
                    };

                    let x_view = x_buf.as_ref().map(|xb| xb.view());
                    if let Some((_iters, _converged)) = estimate_model_into(
                        &y_buf.view(),
                        &mask_buf.view(),
                        &weight_buf.view(),
                        ln_eff,
                        n_periods,
                        n_boot_units,
                        max_iter,
                        tol,
                        None,
                        ws,
                        x_view.as_ref(),
                        None,
                        &mut alpha_buf,
                        &mut beta_buf,
                        &mut l_buf,
                        gamma_buf.as_mut(),
                    ) {
                        let mut tau = y_buf[[t, i]] - alpha_buf[i] - beta_buf[t] - l_buf[[t, i]];
                        if let Some(ref xb) = x_buf {
                            if let Some(ref g) = gamma_buf {
                                let idx = t * n_boot_units + i;
                                tau -= xb.row(idx).dot(g);
                            }
                        }
                        if tau.is_finite() {
                            tau_cells.push((tau, i));
                        }

                        // Stash converged solution for warm-starting the next observation.
                        warm_alpha = Some(alpha_buf.clone());
                        warm_beta = Some(beta_buf.clone());
                        warm_l = Some(l_buf.clone());
                    }
                }

                if let Some(att) = aggregate_att(&tau_cells, Some(&w_boot)).filter(|a| a.is_finite()) {
                    results.push(att);
                }

                (results, y_buf, d_buf, mask_buf, weight_buf, alpha_buf, beta_buf, l_buf, gamma_buf, x_buf)
            },
        )
        .map(|(results, ..)| results)
        .reduce(Vec::new, |mut a, b| { a.extend(b); a });

    let n_valid = bootstrap_estimates.len();

    let (mean, _, se) = compute_bootstrap_variance(&bootstrap_estimates, ddof);
    let (ci_lower, ci_upper) = compute_percentile_ci(&bootstrap_estimates, alpha);

    BootstrapResult {
        estimates: bootstrap_estimates,
        se,
        mean,
        ci_lower,
        ci_upper,
        n_valid,
        n_total: n_bootstrap,
        level: 1.0 - alpha,
    }
}

/// Weighted joint bootstrap: pweight-only survey-design variant.
///
/// Follows the same procedure as [`bootstrap_trop_variance_joint_full`] but
/// extracts the joint-fit parameters `(μ, α, β, L)` and computes the post-hoc
/// ATT as `τ̂ = Σ w_i (Y_{t,i} − μ − α_i − β_t − L_{t,i}) / Σ w_i`.
///
/// The joint estimation of `(μ, α, β, L)` itself is unchanged — pweight
/// enters only the ATT aggregation.
///
/// # Arguments
/// Same as [`bootstrap_trop_variance_joint_full`], plus:
/// * `unit_weights` - Per-original-unit pweights (length = `n_units`).
#[allow(clippy::too_many_arguments)]
pub fn bootstrap_trop_variance_joint_full_weighted(
    y: &ArrayView2<f64>,
    d: &ArrayView2<f64>,
    lambda_time: f64,
    lambda_unit: f64,
    lambda_nn: f64,
    n_bootstrap: usize,
    max_iter: usize,
    tol: f64,
    seed: u64,
    alpha: f64,
    ddof: u8,
    unit_weights: &[f64],
    x: Option<&ArrayView2<f64>>,
) -> BootstrapResult {
    let y_arr = y.to_owned();
    let d_arr = d.to_owned();
    let unit_weights_owned: Vec<f64> = unit_weights.to_vec();
    let x_arr = x.map(|xv| xv.to_owned());

    let n_units = y_arr.ncols();
    let n_periods = y_arr.nrows();

    let classification = classify_units(&d_arr.view());

    let mut first_treat_period = n_periods;
    for t in 0..n_periods {
        for i in 0..n_units {
            if d_arr[[t, i]] == 1.0 {
                first_treat_period = first_treat_period.min(t);
                break;
            }
        }
    }
    let treated_periods = n_periods.saturating_sub(first_treat_period);

    let lt_eff = if lambda_time.is_infinite() {
        0.0
    } else {
        lambda_time
    };
    let lu_eff = if lambda_unit.is_infinite() {
        0.0
    } else {
        lambda_unit
    };
    let ln_eff = if lambda_nn.is_infinite() {
        1e10
    } else {
        lambda_nn
    };

    // Stratified sampling always draws n_control + n_treated = n_units columns.
    let n_boot_units = classification.n_control + classification.n_treated;

    // Parallel bootstrap with per-task buffer reuse (same pattern as
    // bootstrap_trop_variance_joint_full — see comments there).
    let bootstrap_estimates: Vec<f64> = (0..n_bootstrap)
        .into_par_iter()
        .fold(
            || {
                let x_buf = x_arr.as_ref().map(|xv| {
                    Array2::<f64>::zeros((n_periods * n_boot_units, xv.ncols()))
                });
                (
                    Vec::<f64>::new(),
                    Array2::<f64>::zeros((n_periods, n_boot_units)),
                    Array2::<f64>::zeros((n_periods, n_boot_units)),
                    x_buf,
                )
            },
            |(mut results, mut y_buf, mut d_buf, mut x_buf), b| {
                let iteration_seed = seed.wrapping_add(b as u64);
                let sampled_units = stratified_sample(&classification, iteration_seed);

                build_bootstrap_matrices_into(
                    &mut y_buf, &mut d_buf,
                    &y_arr, &d_arr,
                    &sampled_units,
                );

                // Resample X matrix by unit if covariates are present.
                if let (Some(ref x_owned), Some(ref mut xb)) = (&x_arr, &mut x_buf) {
                    xb.fill(0.0);
                    for (j, &orig_unit) in sampled_units.iter().enumerate() {
                        for t in 0..n_periods {
                            let src_idx = t * n_units + orig_unit;
                            let dst_idx = t * n_boot_units + j;
                            xb.row_mut(dst_idx).assign(&x_owned.row(src_idx));
                        }
                    }
                }

                let w_boot: Vec<f64> = sampled_units
                    .iter()
                    .map(|&orig_idx| unit_weights_owned.get(orig_idx).copied().unwrap_or(0.0))
                    .collect();

                let delta = compute_joint_weights(
                    &y_buf.view(),
                    &d_buf.view(),
                    lt_eff,
                    lu_eff,
                    treated_periods,
                );

                debug_assert_delta_is_1minus_d_masked(
                    &d_buf.view(), &delta.view(),
                    "bootstrap::joint_weighted/delta",
                );

                // Collect (tau, column_index) pairs for weighted aggregation.
                let x_view = x_buf.as_ref().map(|xb| xb.view());
                let tau_cells_opt: Option<Vec<(f64, usize)>> = if ln_eff >= 1e10 {
                    solve_joint_no_lowrank(
                        &y_buf.view(),
                        &delta.view(),
                        x_view.as_ref(),
                    ).map(
                        |(mu, alpha_est, beta, result_gamma)| {
                            let mut cells: Vec<(f64, usize)> = Vec::new();
                            for t in 0..n_periods {
                                for i in 0..n_boot_units {
                                    if d_buf[[t, i]] == 1.0 && y_buf[[t, i]].is_finite() {
                                        let mut tau_val =
                                            y_buf[[t, i]] - mu - alpha_est[i] - beta[t];
                                        if let Some(ref xb) = x_buf {
                                            if let Some(ref g) = result_gamma {
                                                let idx = t * n_boot_units + i;
                                                tau_val -= xb.row(idx).dot(g);
                                            }
                                        }
                                        cells.push((tau_val, i));
                                    }
                                }
                            }
                            cells
                        },
                    )
                } else {
                    solve_joint_with_lowrank(
                        &y_buf.view(),
                        &d_buf.view(),
                        &delta.view(),
                        ln_eff,
                        max_iter,
                        tol,
                        x_view.as_ref(),
                    )
                    .map(|(mu, alpha_est, beta, l, _tau, _iters, _converged, result_gamma)| {
                        let mut cells: Vec<(f64, usize)> = Vec::new();
                        for t in 0..n_periods {
                            for i in 0..n_boot_units {
                                if d_buf[[t, i]] == 1.0 && y_buf[[t, i]].is_finite() {
                                    let mut tau_val = y_buf[[t, i]]
                                        - mu
                                        - alpha_est[i]
                                        - beta[t]
                                        - l[[t, i]];
                                    if let Some(ref xb) = x_buf {
                                        if let Some(ref g) = result_gamma {
                                            let idx = t * n_boot_units + i;
                                            tau_val -= xb.row(idx).dot(g);
                                        }
                                    }
                                    cells.push((tau_val, i));
                                }
                            }
                        }
                        cells
                    })
                };

                if let Some(att) = tau_cells_opt
                    .and_then(|cells| aggregate_att(&cells, Some(&w_boot)))
                    .filter(|a| a.is_finite())
                {
                    results.push(att);
                }

                (results, y_buf, d_buf, x_buf)
            },
        )
        .map(|(results, _, _, _)| results)
        .reduce(Vec::new, |mut a, b| { a.extend(b); a });

    let n_valid = bootstrap_estimates.len();

    let (mean, _, se) = compute_bootstrap_variance(&bootstrap_estimates, ddof);
    let (ci_lower, ci_upper) = compute_percentile_ci(&bootstrap_estimates, alpha);

    BootstrapResult {
        estimates: bootstrap_estimates,
        se,
        mean,
        ci_lower,
        ci_upper,
        n_valid,
        n_total: n_bootstrap,
        level: 1.0 - alpha,
    }
}

// ============================================================================
// Rao-Wu Bootstrap for Survey Designs
// ============================================================================

/// Strategy for handling singleton strata (lonely PSU) in Rao-Wu bootstrap.
///
/// When a stratum contains only one PSU (n_h=1), the within-stratum variance
/// is undefined. This enum controls how such strata are treated.
///
/// Reference: Binder (1983), Rao & Wu (1988) — singleton PSU handling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LonelyPsuStrategy {
    /// Skip singleton strata entirely (current default for backward compat).
    /// Their PSUs retain their original weights without bootstrap rescaling.
    Skip,
    /// Center the singleton PSU score against the global PSU mean and
    /// include its contribution in the bootstrap variance.
    /// For Rao-Wu reweighting, this means the singleton PSU gets a
    /// perturbation drawn relative to the global mean weight scale.
    Centered,
    /// Strict mode: return an error if any singleton stratum is encountered.
    Fail,
}

impl LonelyPsuStrategy {
    /// Parse from integer code passed via FFI.
    /// 0 = Skip (default), 1 = Centered, 2 = Fail.
    pub fn from_code(code: i32) -> Self {
        match code {
            1 => LonelyPsuStrategy::Centered,
            2 => LonelyPsuStrategy::Fail,
            _ => LonelyPsuStrategy::Skip,
        }
    }
}

/// Metadata for a single stratum in the Rao-Wu bootstrap scheme.
#[derive(Debug, Clone)]
struct StratumInfo {
    /// Number of distinct PSUs in this stratum.
    n_psu: usize,
    /// Per-stratum finite population SIZE N_h (not the sampling fraction f_h).
    /// The sampling fraction is computed as f_h = n_h / fpc where n_h is the
    /// number of PSUs observed in this stratum.
    fpc: Option<f64>,
    /// Mapping: PSU local index → vector of unit column indices belonging to that PSU.
    psu_to_units: Vec<Vec<usize>>,
}

/// Build the stratum structure from unit-level labels.
///
/// Groups units by stratum, then within each stratum by PSU.
/// Returns one [`StratumInfo`] per distinct stratum.
fn build_strata_structure(
    strata: &[i64],
    psu: &[i64],
    fpc: Option<&[f64]>,
) -> TropResult<Vec<StratumInfo>> {
    use std::collections::BTreeMap;

    // stratum_label → { psu_label → [unit_indices] }
    let mut strata_map: BTreeMap<i64, BTreeMap<i64, Vec<usize>>> = BTreeMap::new();
    let mut strata_fpc: BTreeMap<i64, Option<f64>> = BTreeMap::new();

    for (unit_idx, (&s, &p)) in strata.iter().zip(psu.iter()).enumerate() {
        strata_map
            .entry(s)
            .or_default()
            .entry(p)
            .or_default()
            .push(unit_idx);

        // Record FPC for this stratum (assumed constant within stratum).
        if let Some(fpc_vals) = fpc {
            strata_fpc.entry(s).or_insert(Some(fpc_vals[unit_idx]));
        } else {
            strata_fpc.entry(s).or_insert(None);
        }
    }

    let groups: Vec<StratumInfo> = strata_map
        .into_iter()
        .map(|(s_label, psu_map)| {
            let psu_to_units: Vec<Vec<usize>> =
                psu_map.into_values().collect();
            let n_psu = psu_to_units.len();
            let fpc_val = strata_fpc.get(&s_label).copied().flatten();
            StratumInfo {
                n_psu,
                fpc: fpc_val,
                psu_to_units,
            }
        })
        .collect();

    // Validate FPC values for each stratum.
    for stratum_info in &groups {
        if let Some(fpc_val) = stratum_info.fpc {
            let n_h = stratum_info.n_psu;
            // FPC must be positive and finite.
            if !fpc_val.is_finite() || fpc_val <= 0.0 {
                return Err(TropError::InvalidFpc);
            }
            // FPC must be >= number of sampled PSUs in the stratum.
            if (fpc_val as usize) < n_h {
                return Err(TropError::InvalidFpc);
            }
        }
    }

    Ok(groups)
}

/// Rao-Wu (1988) bootstrap for twostep method with complex survey design.
///
/// Instead of physically resampling units, generates rescaled survey weights
/// per bootstrap iteration. Since survey weights only affect ATT aggregation
/// (not model fitting), we fit the model ONCE and reweight B times.
///
/// Algorithm:
/// 1. On the original panel, compute τ_{t,i} for every treated cell once
///    (one local model fit per treated observation, held fixed across all
///    bootstrap draws — survey weights do not enter model fitting).
/// 2. For b=1..B (parallel):
///    - For each stratum h with n_h PSUs:
///      m_h = n_h-1 (no FPC) or max(1, round((1-f_h)(n_h-1))) (with FPC)
///      Draw m_h PSUs with replacement, count r_hi
///      Scale: w_i*(b) = w_i × (n_h/m_h) × r_hi
///    - ATT^(b) = Σ w_i*(b) τ_{ti} / Σ w_i*(b) [treated obs only]
/// 3. SE = std(ATT^(1)..ATT^(B), ddof)
#[allow(clippy::too_many_arguments)]
pub fn bootstrap_trop_variance_rao_wu(
    y: &ArrayView2<f64>,
    d: &ArrayView2<f64>,
    control_mask: &ArrayView2<u8>,
    time_dist: &ArrayView2<i64>,
    lambda_time: f64,
    lambda_unit: f64,
    lambda_nn: f64,
    n_bootstrap: usize,
    max_iter: usize,
    tol: f64,
    seed: u64,
    alpha: f64,
    ddof: u8,
    strata: &[i64],
    psu: &[i64],
    fpc: Option<&[f64]>,
    unit_weights: &[f64],
    x: Option<&ArrayView2<f64>>,
    lonely_psu: LonelyPsuStrategy,
) -> TropResult<BootstrapResult> {
    let n_periods = y.nrows();
    let n_units = y.ncols();

    // Map infinite lambda values to effective computation values.
    let lt_eff = if lambda_time.is_infinite() { 0.0 } else { lambda_time };
    let lu_eff = if lambda_unit.is_infinite() { 0.0 } else { lambda_unit };
    let ln_eff = if lambda_nn.is_infinite() { 1e10 } else { lambda_nn };

    // NOTE: Buffer reuse optimization is not applicable here.
    // Rao-Wu bootstrap fits the model ONCE on the original panel and then
    // only reweights the pre-computed τ values B times.  No bootstrap
    // matrices are constructed in the iteration loop, so there are no
    // per-iteration allocations to eliminate.

    // ---- Phase 1: Fit model ONCE, collect per-treated-obs τ values ----
    let dist_cache = UnitDistanceCache::build(y, d);

    // Identify treated (t, i) pairs.
    let mut treated_obs: Vec<(usize, usize)> = Vec::new();
    for t in 0..n_periods {
        for i in 0..n_units {
            if d[[t, i]] == 1.0 {
                treated_obs.push((t, i));
            }
        }
    }

    if treated_obs.is_empty() {
        return Ok(BootstrapResult {
            estimates: Vec::new(),
            se: 0.0,
            mean: 0.0,
            ci_lower: f64::NAN,
            ci_upper: f64::NAN,
            n_valid: 0,
            n_total: n_bootstrap,
            level: 1.0 - alpha,
        });
    }

    // Compute τ for each treated observation (single model fit).
    let tau_cells: Vec<(f64, usize)> = treated_obs
        .par_iter()
        .filter_map(|&(t, i)| {
            let weight_matrix = compute_weight_matrix_cached(
                y,
                d,
                &dist_cache,
                n_periods,
                n_units,
                i,
                t,
                lt_eff,
                lu_eff,
                time_dist,
            );

            estimate_model(
                y,
                control_mask,
                &weight_matrix.view(),
                ln_eff,
                n_periods,
                n_units,
                max_iter,
                tol,
                None,
                None,
                x,
                None,
            )
            .and_then(|(alpha_est, beta, l, _n_iters, _converged, result_gamma)| {
                let mut tau = y[[t, i]] - alpha_est[i] - beta[t] - l[[t, i]];
                if let Some(x_mat) = x {
                    if let Some(ref g) = result_gamma {
                        let idx = t * n_units + i;
                        tau -= x_mat.row(idx).dot(g);
                    }
                }
                if tau.is_finite() {
                    Some((tau, i))
                } else {
                    None
                }
            })
        })
        .collect();

    if tau_cells.is_empty() {
        return Ok(BootstrapResult {
            estimates: Vec::new(),
            se: 0.0,
            mean: 0.0,
            ci_lower: f64::NAN,
            ci_upper: f64::NAN,
            n_valid: 0,
            n_total: n_bootstrap,
            level: 1.0 - alpha,
        });
    }

    // ---- Phase 2: Rao-Wu reweighting (parallel B iterations) ----
    let strata_groups = build_strata_structure(strata, psu, fpc)?;

    // Pre-check: in Fail mode, reject if any singleton stratum exists.
    if lonely_psu == LonelyPsuStrategy::Fail {
        for si in &strata_groups {
            if si.n_psu < 2 {
                return Err(TropError::SingletonPsu);
            }
        }
    }

    // For Centered strategy, compute the global mean weight across all PSUs.
    // This is used to "center" the singleton PSU's bootstrap weight perturbation.
    // Equivalent to Python's _global_psu_mean approach adapted to weight domain:
    // we compute mean(original_weight) over all PSUs across all strata.
    let global_mean_weight: f64 = if lonely_psu == LonelyPsuStrategy::Centered {
        let mut sum_w = 0.0_f64;
        let mut n_all_psu = 0usize;
        for si in &strata_groups {
            for psu_units in &si.psu_to_units {
                // PSU-level weight = sum of unit weights within this PSU
                let psu_w: f64 = psu_units.iter().map(|&u| unit_weights[u]).sum();
                sum_w += psu_w;
                n_all_psu += 1;
            }
        }
        if n_all_psu > 0 { sum_w / n_all_psu as f64 } else { 0.0 }
    } else {
        0.0
    };

    let unit_weights_owned: Vec<f64> = unit_weights.to_vec();
    let tau_cells_ref = &tau_cells;

    let bootstrap_estimates: Vec<f64> = (0..n_bootstrap)
        .into_par_iter()
        .filter_map(|b| {
            let iteration_seed = seed.wrapping_add(b as u64);
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(iteration_seed);
            let mut boot_weights = unit_weights_owned.clone();

            for stratum_info in &strata_groups {
                let n_h = stratum_info.n_psu;
                if n_h < 2 {
                    // Singleton stratum handling
                    match lonely_psu {
                        LonelyPsuStrategy::Skip => continue,
                        LonelyPsuStrategy::Centered => {
                            // Centered strategy: perturb singleton PSU weight
                            // relative to global mean.
                            // Draw a single Poisson(1) count (0 or 1+) and scale.
                            // With m_h=1, we draw 1 PSU from 1 available: count=1 always.
                            // The centering effect comes from subtracting global mean.
                            // Implementation: scale_factor = n_h/m_h = 1/1 = 1,
                            // count = 1 always. But to introduce variability we use
                            // a draw from Exponential(1) centered at 1.0, following
                            // the Rao-Wu spirit.
                            // Actually, for Rao-Wu with n_h=1: set m_h=1 (since
                            // n_h-1=0 is invalid, use m_h=max(1, ...)=1).
                            // Draw 1 PSU from {0}: count[0] = 1 always.
                            // With the centered approach, the perturbation is:
                            //   w_i*(b) = w_i + (w_i - w_global_mean) * epsilon
                            // where epsilon ~ Uniform[-1, 1] for basic perturbation.
                            // Following Binder's approach more precisely:
                            //   The singleton PSU weight gets a random perturbation
                            //   drawn so that E[w*] = w_original.
                            // Simplest correct implementation: use a Rademacher
                            // perturbation: w* = w + sign * (w - global_mean)
                            let sign: f64 = if rng.gen_bool(0.5) { 1.0 } else { -1.0 };
                            for &unit_idx in &stratum_info.psu_to_units[0] {
                                let w_orig = unit_weights_owned[unit_idx];
                                let centered_dev = w_orig - global_mean_weight;
                                boot_weights[unit_idx] = w_orig + sign * centered_dev;
                                // Ensure non-negative
                                if boot_weights[unit_idx] < 0.0 {
                                    boot_weights[unit_idx] = 0.0;
                                }
                            }
                            continue;
                        }
                        LonelyPsuStrategy::Fail => unreachable!(),
                    }
                }

                let m_h = if let Some(fpc_val) = stratum_info.fpc {
                    let f_h = n_h as f64 / fpc_val;
                    if f_h >= 1.0 {
                        continue; // census stratum
                    }
                    (((1.0 - f_h) * (n_h as f64 - 1.0)).round() as usize).max(1)
                } else {
                    n_h - 1
                };

                // Draw m_h PSUs with replacement
                let mut counts = vec![0usize; n_h];
                for _ in 0..m_h {
                    let idx = rng.gen_range(0..n_h);
                    counts[idx] += 1;
                }

                // Scale weights for units in this stratum
                let scale_factor = n_h as f64 / m_h as f64;
                for (psu_local_idx, &count) in counts.iter().enumerate() {
                    for &unit_idx in &stratum_info.psu_to_units[psu_local_idx] {
                        boot_weights[unit_idx] =
                            unit_weights_owned[unit_idx] * scale_factor * count as f64;
                    }
                }
            }

            // Weighted ATT computation
            let mut weighted_sum = 0.0_f64;
            let mut weight_sum = 0.0_f64;
            for &(tau_val, unit_idx) in tau_cells_ref {
                let w = boot_weights[unit_idx];
                if w.is_finite() && w > 0.0 {
                    weighted_sum += w * tau_val;
                    weight_sum += w;
                }
            }

            if weight_sum > 0.0 {
                let att = weighted_sum / weight_sum;
                if att.is_finite() { Some(att) } else { None }
            } else {
                None
            }
        })
        .collect();

    let n_valid = bootstrap_estimates.len();
    let (mean, _, se) = compute_bootstrap_variance(&bootstrap_estimates, ddof);
    let (ci_lower, ci_upper) = compute_percentile_ci(&bootstrap_estimates, alpha);

    Ok(BootstrapResult {
        estimates: bootstrap_estimates,
        se,
        mean,
        ci_lower,
        ci_upper,
        n_valid,
        n_total: n_bootstrap,
        level: 1.0 - alpha,
    })
}

/// Rao-Wu (1988) bootstrap for joint method with complex survey design.
///
/// Same "fit once, reweight B times" strategy as the twostep variant.
/// The joint estimator produces per-cell τ values from a single global
/// WLS fit, then the Rao-Wu rescaling generates B weighted ATTs.
#[allow(clippy::too_many_arguments)]
pub fn bootstrap_trop_variance_rao_wu_joint(
    y: &ArrayView2<f64>,
    d: &ArrayView2<f64>,
    lambda_time: f64,
    lambda_unit: f64,
    lambda_nn: f64,
    n_bootstrap: usize,
    max_iter: usize,
    tol: f64,
    seed: u64,
    alpha: f64,
    ddof: u8,
    strata: &[i64],
    psu: &[i64],
    fpc: Option<&[f64]>,
    unit_weights: &[f64],
    x: Option<&ArrayView2<f64>>,
    lonely_psu: LonelyPsuStrategy,
) -> TropResult<BootstrapResult> {
    let n_periods = y.nrows();
    let n_units = y.ncols();

    let lt_eff = if lambda_time.is_infinite() { 0.0 } else { lambda_time };
    let lu_eff = if lambda_unit.is_infinite() { 0.0 } else { lambda_unit };
    let ln_eff = if lambda_nn.is_infinite() { 1e10 } else { lambda_nn };

    // NOTE: Buffer reuse optimization is not applicable here.
    // Rao-Wu joint bootstrap fits the model ONCE on the original panel and
    // then only reweights the pre-computed τ values B times.  No bootstrap
    // matrices are constructed in the iteration loop.

    // Determine treated periods for joint weight construction.
    let mut first_treat_period = n_periods;
    for t in 0..n_periods {
        for i in 0..n_units {
            if d[[t, i]] == 1.0 {
                first_treat_period = first_treat_period.min(t);
                break;
            }
        }
    }
    let treated_periods = n_periods.saturating_sub(first_treat_period);

    // ---- Phase 1: Fit joint model ONCE ----
    let delta = compute_joint_weights(y, d, lt_eff, lu_eff, treated_periods);

    debug_assert_delta_is_1minus_d_masked(d, &delta.view(), "bootstrap::rao_wu_joint/delta");

    // Extract per-cell τ values from joint fit.
    let tau_cells: Vec<(f64, usize)> = if ln_eff >= 1e10 {
        match solve_joint_no_lowrank(y, &delta.view(), x) {
            Some((mu, alpha_est, beta, result_gamma)) => {
                let mut cells = Vec::new();
                for t in 0..n_periods {
                    for i in 0..n_units {
                        if d[[t, i]] == 1.0 && y[[t, i]].is_finite() {
                            let mut tau_val = y[[t, i]] - mu - alpha_est[i] - beta[t];
                            if let Some(x_mat) = x {
                                if let Some(ref g) = result_gamma {
                                    let idx = t * n_units + i;
                                    tau_val -= x_mat.row(idx).dot(g);
                                }
                            }
                            if tau_val.is_finite() {
                                cells.push((tau_val, i));
                            }
                        }
                    }
                }
                cells
            }
            None => Vec::new(),
        }
    } else {
        match solve_joint_with_lowrank(y, d, &delta.view(), ln_eff, max_iter, tol, x) {
            Some((mu, alpha_est, beta, l, _tau, _iters, _converged, result_gamma)) => {
                let mut cells = Vec::new();
                for t in 0..n_periods {
                    for i in 0..n_units {
                        if d[[t, i]] == 1.0 && y[[t, i]].is_finite() {
                            let mut tau_val =
                                y[[t, i]] - mu - alpha_est[i] - beta[t] - l[[t, i]];
                            if let Some(x_mat) = x {
                                if let Some(ref g) = result_gamma {
                                    let idx = t * n_units + i;
                                    tau_val -= x_mat.row(idx).dot(g);
                                }
                            }
                            if tau_val.is_finite() {
                                cells.push((tau_val, i));
                            }
                        }
                    }
                }
                cells
            }
            None => Vec::new(),
        }
    };

    if tau_cells.is_empty() {
        return Ok(BootstrapResult {
            estimates: Vec::new(),
            se: 0.0,
            mean: 0.0,
            ci_lower: f64::NAN,
            ci_upper: f64::NAN,
            n_valid: 0,
            n_total: n_bootstrap,
            level: 1.0 - alpha,
        });
    }

    // ---- Phase 2: Rao-Wu reweighting (parallel B iterations) ----
    let strata_groups = build_strata_structure(strata, psu, fpc)?;

    // Pre-check: in Fail mode, reject if any singleton stratum exists.
    if lonely_psu == LonelyPsuStrategy::Fail {
        for si in &strata_groups {
            if si.n_psu < 2 {
                return Err(TropError::SingletonPsu);
            }
        }
    }

    // Global mean weight for centered singleton handling.
    let global_mean_weight: f64 = if lonely_psu == LonelyPsuStrategy::Centered {
        let mut sum_w = 0.0_f64;
        let mut n_all_psu = 0usize;
        for si in &strata_groups {
            for psu_units in &si.psu_to_units {
                let psu_w: f64 = psu_units.iter().map(|&u| unit_weights[u]).sum();
                sum_w += psu_w;
                n_all_psu += 1;
            }
        }
        if n_all_psu > 0 { sum_w / n_all_psu as f64 } else { 0.0 }
    } else {
        0.0
    };

    let unit_weights_owned: Vec<f64> = unit_weights.to_vec();
    let tau_cells_ref = &tau_cells;

    let bootstrap_estimates: Vec<f64> = (0..n_bootstrap)
        .into_par_iter()
        .filter_map(|b| {
            let iteration_seed = seed.wrapping_add(b as u64);
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(iteration_seed);
            let mut boot_weights = unit_weights_owned.clone();

            for stratum_info in &strata_groups {
                let n_h = stratum_info.n_psu;
                if n_h < 2 {
                    match lonely_psu {
                        LonelyPsuStrategy::Skip => continue,
                        LonelyPsuStrategy::Centered => {
                            let sign: f64 = if rng.gen_bool(0.5) { 1.0 } else { -1.0 };
                            for &unit_idx in &stratum_info.psu_to_units[0] {
                                let w_orig = unit_weights_owned[unit_idx];
                                let centered_dev = w_orig - global_mean_weight;
                                boot_weights[unit_idx] = (w_orig + sign * centered_dev).max(0.0);
                            }
                            continue;
                        }
                        LonelyPsuStrategy::Fail => unreachable!(),
                    }
                }

                let m_h = if let Some(fpc_val) = stratum_info.fpc {
                    let f_h = n_h as f64 / fpc_val;
                    if f_h >= 1.0 {
                        continue;
                    }
                    (((1.0 - f_h) * (n_h as f64 - 1.0)).round() as usize).max(1)
                } else {
                    n_h - 1
                };

                let mut counts = vec![0usize; n_h];
                for _ in 0..m_h {
                    let idx = rng.gen_range(0..n_h);
                    counts[idx] += 1;
                }

                let scale_factor = n_h as f64 / m_h as f64;
                for (psu_local_idx, &count) in counts.iter().enumerate() {
                    for &unit_idx in &stratum_info.psu_to_units[psu_local_idx] {
                        boot_weights[unit_idx] =
                            unit_weights_owned[unit_idx] * scale_factor * count as f64;
                    }
                }
            }

            let mut weighted_sum = 0.0_f64;
            let mut weight_sum = 0.0_f64;
            for &(tau_val, unit_idx) in tau_cells_ref {
                let w = boot_weights[unit_idx];
                if w.is_finite() && w > 0.0 {
                    weighted_sum += w * tau_val;
                    weight_sum += w;
                }
            }

            if weight_sum > 0.0 {
                let att = weighted_sum / weight_sum;
                if att.is_finite() { Some(att) } else { None }
            } else {
                None
            }
        })
        .collect();

    let n_valid = bootstrap_estimates.len();
    let (mean, _, se) = compute_bootstrap_variance(&bootstrap_estimates, ddof);
    let (ci_lower, ci_upper) = compute_percentile_ci(&bootstrap_estimates, alpha);

    Ok(BootstrapResult {
        estimates: bootstrap_estimates,
        se,
        mean,
        ci_lower,
        ci_upper,
        n_valid,
        n_total: n_bootstrap,
        level: 1.0 - alpha,
    })
}

// ============================================================================
// Survey Diagnostics (P2)
// ============================================================================

/// Diagnostic information for a single stratum with high sampling fraction.
#[derive(Debug, Clone)]
pub struct HighFpcStratum {
    /// 1-based stratum index (order in which strata appear).
    pub stratum_index: usize,
    /// Sampling fraction f_h = n_h / N_h.
    pub f_h: f64,
    /// Number of sampled PSUs in this stratum.
    pub n_h: usize,
    /// Finite population size N_h for this stratum.
    pub big_n_h: f64,
}

/// Aggregated survey diagnostics returned by [`compute_survey_diagnostics`].
#[derive(Debug, Clone)]
pub struct SurveyDiagnostics {
    /// Kish (1965) design effect due to unequal weighting.
    /// deff_w = n * sum(w_i^2) / (sum(w_i))^2.
    /// Returns NaN if weights are all zero or degenerate.
    pub deff_weights: f64,
    /// Strata where f_h > 0.5 (high sampling fraction, strong FPC).
    pub high_fpc_strata: Vec<HighFpcStratum>,
    /// Maximum sampling fraction across all strata (NaN if no FPC).
    pub max_fh: f64,
    /// Number of strata with f_h > 0.5.
    pub n_high_fpc: usize,
}

/// Compute survey diagnostics: weights DEFF and high-FPC detection.
///
/// This is a pure diagnostic function — it does not alter any computation.
///
/// # Arguments
/// * `strata` - Stratum labels per unit (length N).
/// * `psu` - PSU labels per unit (length N).
/// * `fpc` - Optional finite population correction values per unit.
/// * `unit_weights` - Per-unit survey weights (length N).
///
/// # Returns
/// A [`SurveyDiagnostics`] struct with DEFF and high-FPC information.
pub fn compute_survey_diagnostics(
    strata: &[i64],
    psu: &[i64],
    fpc: Option<&[f64]>,
    unit_weights: &[f64],
) -> SurveyDiagnostics {
    let n = unit_weights.len();

    // --- Compute Kish DEFF (weights design effect) ---
    // deff_w = n * sum(w_i^2) / (sum(w_i))^2
    let deff_weights = compute_kish_deff(unit_weights);

    // --- Detect high sampling fraction strata ---
    let (high_fpc_strata, max_fh) = if let Some(fpc_vals) = fpc {
        if n > 0 {
            detect_high_fpc_strata(strata, psu, fpc_vals)
        } else {
            (Vec::new(), f64::NAN)
        }
    } else {
        (Vec::new(), f64::NAN)
    };

    let n_high_fpc = high_fpc_strata.len();

    SurveyDiagnostics {
        deff_weights,
        high_fpc_strata,
        max_fh,
        n_high_fpc,
    }
}

/// Compute the Kish (1965) design effect from unequal weighting.
///
/// Formula: deff_w = n * Σ(w_i²) / (Σw_i)²
/// Equivalent to: deff_w = 1 + CV²(w)
///
/// # Edge cases
/// - All zero weights → returns NaN
/// - Single weight → returns 1.0
/// - Equal weights → returns 1.0
fn compute_kish_deff(weights: &[f64]) -> f64 {
    let n = weights.len();
    if n == 0 {
        return f64::NAN;
    }

    let mut sum_w = 0.0_f64;
    let mut sum_w2 = 0.0_f64;
    let mut n_positive = 0usize;

    for &w in weights {
        if w.is_finite() && w > 0.0 {
            sum_w += w;
            sum_w2 += w * w;
            n_positive += 1;
        }
    }

    // Guard against degenerate cases.
    if n_positive == 0 || sum_w <= 0.0 {
        return f64::NAN;
    }
    if n_positive == 1 {
        return 1.0;
    }

    let deff = (n_positive as f64) * sum_w2 / (sum_w * sum_w);

    // Numerical guard: DEFF should always be >= 1.0 by Jensen's inequality.
    if deff < 1.0 {
        1.0
    } else {
        deff
    }
}

/// Detect strata with sampling fraction f_h > 0.5.
///
/// Returns the list of high-FPC strata and the maximum f_h observed.
fn detect_high_fpc_strata(
    strata: &[i64],
    psu: &[i64],
    fpc_vals: &[f64],
) -> (Vec<HighFpcStratum>, f64) {
    use std::collections::BTreeMap;

    // Group by stratum to count distinct PSUs and get FPC value.
    let mut strata_map: BTreeMap<i64, (std::collections::BTreeSet<i64>, f64)> = BTreeMap::new();

    for (idx, (&s, &p)) in strata.iter().zip(psu.iter()).enumerate() {
        let entry = strata_map.entry(s).or_insert_with(|| {
            (std::collections::BTreeSet::new(), fpc_vals[idx])
        });
        entry.0.insert(p);
    }

    let mut high_fpc_strata = Vec::new();
    let mut max_fh = f64::NEG_INFINITY;

    for (stratum_index, (psu_set, fpc_val)) in strata_map.values().enumerate() {
        let stratum_index = stratum_index + 1;
        let n_h = psu_set.len();
        let big_n_h = *fpc_val;

        if !big_n_h.is_finite() || big_n_h <= 0.0 {
            continue;
        }

        let f_h = n_h as f64 / big_n_h;

        if f_h > max_fh {
            max_fh = f_h;
        }

        if f_h > 0.5 {
            high_fpc_strata.push(HighFpcStratum {
                stratum_index,
                f_h,
                n_h,
                big_n_h,
            });
        }
    }

    if max_fh == f64::NEG_INFINITY {
        max_fh = f64::NAN;
    }

    (high_fpc_strata, max_fh)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_classify_units() {
        // Create a simple D matrix: 3 periods, 4 units
        // Unit 0, 1: never treated (control)
        // Unit 2, 3: treated at some point
        let d = Array2::from_shape_vec(
            (3, 4),
            vec![
                0.0, 0.0, 0.0, 0.0, // t=0
                0.0, 0.0, 1.0, 0.0, // t=1
                0.0, 0.0, 1.0, 1.0, // t=2
            ],
        )
        .unwrap();

        let classification = classify_units(&d.view());

        assert_eq!(classification.n_control, 2);
        assert_eq!(classification.n_treated, 2);
        assert_eq!(classification.control_units, vec![0, 1]);
        assert_eq!(classification.treated_units, vec![2, 3]);
    }

    #[test]
    fn test_stratified_sample_preserves_counts() {
        let classification = UnitClassification {
            control_units: vec![0, 1, 2, 3, 4],
            treated_units: vec![5, 6, 7],
            n_control: 5,
            n_treated: 3,
        };

        let seed = 42u64;
        let sampled = stratified_sample(&classification, seed);

        // Should sample exactly n_control + n_treated units
        assert_eq!(sampled.len(), 8);
    }

    #[test]
    fn test_stratified_sample_deterministic() {
        let classification = UnitClassification {
            control_units: vec![0, 1, 2, 3, 4],
            treated_units: vec![5, 6, 7],
            n_control: 5,
            n_treated: 3,
        };

        let seed = 42u64;
        let sampled1 = stratified_sample(&classification, seed);
        let sampled2 = stratified_sample(&classification, seed);

        // Same seed should produce same result
        assert_eq!(sampled1, sampled2);
    }

    #[test]
    fn test_stratified_sample_different_seeds() {
        let classification = UnitClassification {
            control_units: vec![0, 1, 2, 3, 4],
            treated_units: vec![5, 6, 7],
            n_control: 5,
            n_treated: 3,
        };

        let sampled1 = stratified_sample(&classification, 42);
        let sampled2 = stratified_sample(&classification, 43);

        // Different seeds should produce different results
        assert_ne!(sampled1, sampled2);
    }

    #[test]
    fn test_build_bootstrap_matrices() {
        let y = Array2::from_shape_vec(
            (2, 3),
            vec![
                1.0, 2.0, 3.0, // t=0
                4.0, 5.0, 6.0, // t=1
            ],
        )
        .unwrap();

        let d = Array2::from_shape_vec(
            (2, 3),
            vec![
                0.0, 0.0, 0.0, // t=0
                0.0, 1.0, 1.0, // t=1
            ],
        )
        .unwrap();

        // Sample units [1, 0, 2] (with replacement)
        let sampled_units = vec![1, 0, 2];
        let (y_boot, d_boot) = build_bootstrap_matrices(&y, &d, &sampled_units);

        // Check dimensions
        assert_eq!(y_boot.shape(), &[2, 3]);
        assert_eq!(d_boot.shape(), &[2, 3]);

        // Check values - unit 1 should be in position 0
        assert_eq!(y_boot[[0, 0]], 2.0);
        assert_eq!(y_boot[[1, 0]], 5.0);

        // Check values - unit 0 should be in position 1
        assert_eq!(y_boot[[0, 1]], 1.0);
        assert_eq!(y_boot[[1, 1]], 4.0);
    }

    #[test]
    fn test_compute_bootstrap_variance_sample() {
        // Variance uses 1/(B−1) (Bessel-corrected sample variance); paper
        // Algorithm 3 uses 1/B (population variance).  This test pins the
        // Bessel denominator so a future refactor cannot silently switch
        // to 1/B without updating the documentation.
        let estimates = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (mean, variance, se) = compute_bootstrap_variance(&estimates, 1);

        // Mean should be 3.0
        assert!((mean - 3.0).abs() < 1e-10);

        // Variance with 1/(B−1): sum((x-3)^2) / 4 = 10/4 = 2.5
        assert!((variance - 2.5).abs() < 1e-10);

        // SE = sqrt(2.5)
        assert!((se - 2.5_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_compute_bootstrap_variance_empty() {
        let estimates: Vec<f64> = vec![];
        let (mean, variance, se) = compute_bootstrap_variance(&estimates, 1);

        assert_eq!(mean, 0.0);
        assert_eq!(variance, 0.0);
        assert_eq!(se, 0.0);
    }

    #[test]
    fn test_compute_bootstrap_variance_single() {
        let estimates = vec![5.0];
        let (mean, variance, se) = compute_bootstrap_variance(&estimates, 1);

        // Single value -> mean = that value, variance/se = 0
        assert_eq!(mean, 5.0);
        assert_eq!(variance, 0.0);
        assert_eq!(se, 0.0);
    }

    #[test]
    fn test_compute_percentile_ci() {
        // Create sorted estimates: 1, 2, 3, ..., 100
        let estimates: Vec<f64> = (1..=100).map(|x| x as f64).collect();

        let (ci_lower, ci_upper) = compute_percentile_ci(&estimates, 0.05);

        // For 95% CI with 100 values:
        // Lower: index = (100-1) * 0.025 = 2.475 → interpolate between sorted[2] and sorted[3]
        // Upper: index = (100-1) * 0.975 = 96.525 → interpolate between sorted[96] and sorted[97]
        assert!(
            (ci_lower - 3.475).abs() < 1e-10,
            "ci_lower={}, expected 3.475",
            ci_lower
        );
        assert!(
            (ci_upper - 97.525).abs() < 1e-10,
            "ci_upper={}, expected 97.525",
            ci_upper
        );
    }

    #[test]
    fn test_compute_percentile_ci_empty() {
        let estimates: Vec<f64> = vec![];
        let (ci_lower, ci_upper) = compute_percentile_ci(&estimates, 0.05);

        assert!(ci_lower.is_nan());
        assert!(ci_upper.is_nan());
    }

    /// NaN values must be filtered before CI computation.
    #[test]
    fn test_compute_percentile_ci_with_nan() {
        // Mix of valid values and NaN
        let estimates = vec![1.0, 2.0, f64::NAN, 3.0, f64::NAN, 4.0, 5.0];
        let (ci_lower, ci_upper) = compute_percentile_ci(&estimates, 0.05);

        // CI should be computed from [1, 2, 3, 4, 5] only
        assert!(ci_lower.is_finite(), "ci_lower should be finite, got {}", ci_lower);
        assert!(ci_upper.is_finite(), "ci_upper should be finite, got {}", ci_upper);
        assert!(ci_lower <= ci_upper, "ci_lower ({}) should be <= ci_upper ({})", ci_lower, ci_upper);

        // Compare with NaN-free version
        let clean = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (ci_lower_clean, ci_upper_clean) = compute_percentile_ci(&clean, 0.05);
        assert!((ci_lower - ci_lower_clean).abs() < 1e-12,
            "NaN-filtered CI lower ({}) should match clean ({})", ci_lower, ci_lower_clean);
        assert!((ci_upper - ci_upper_clean).abs() < 1e-12,
            "NaN-filtered CI upper ({}) should match clean ({})", ci_upper, ci_upper_clean);
    }

    /// All-NaN input yields (NaN, NaN).
    #[test]
    fn test_compute_percentile_ci_all_nan() {
        let estimates = vec![f64::NAN, f64::NAN, f64::NAN];
        let (ci_lower, ci_upper) = compute_percentile_ci(&estimates, 0.05);
        assert!(ci_lower.is_nan());
        assert!(ci_upper.is_nan());
    }

    /// Inf values must be filtered identically to NaN.
    #[test]
    fn test_compute_percentile_ci_with_inf() {
        let estimates = vec![1.0, 2.0, f64::INFINITY, 3.0, f64::NEG_INFINITY, 4.0, 5.0];
        let (ci_lower, ci_upper) = compute_percentile_ci(&estimates, 0.05);

        let clean = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (ci_lower_clean, ci_upper_clean) = compute_percentile_ci(&clean, 0.05);
        assert!((ci_lower - ci_lower_clean).abs() < 1e-12);
        assert!((ci_upper - ci_upper_clean).abs() < 1e-12);
    }

    /// Variance computation filters NaN values transparently.
    #[test]
    fn test_compute_bootstrap_variance_with_nan() {
        let estimates = vec![1.0, 2.0, f64::NAN, 3.0, 4.0, 5.0];
        let (mean, variance, se) = compute_bootstrap_variance(&estimates, 1);

        let clean = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (mean_clean, var_clean, se_clean) = compute_bootstrap_variance(&clean, 1);

        assert!((mean - mean_clean).abs() < 1e-12);
        assert!((variance - var_clean).abs() < 1e-12);
        assert!((se - se_clean).abs() < 1e-12);
    }

    #[test]
    fn test_seed_increment() {
        // Test that different bootstrap iterations use different seeds
        let seed = 42u64;

        let mut rng1 = Xoshiro256PlusPlus::seed_from_u64(seed.wrapping_add(0));
        let mut rng2 = Xoshiro256PlusPlus::seed_from_u64(seed.wrapping_add(1));

        let val1: u64 = rng1.gen();
        let val2: u64 = rng2.gen();

        // Different seeds should produce different values
        assert_ne!(val1, val2);
    }

    #[test]
    fn test_bessel_correction_matches_python() {
        // Verify our function uses 1/(B-1) (Bessel-corrected sample
        // variance) rather than 1/B (paper Algorithm 3's population
        // variance).  The test name is historical; the assertion pins the
        // Bessel denominator.
        let estimates = [1.0, 2.0, 3.0, 4.0, 5.0];
        let n = estimates.len() as f64;
        let mean = estimates.iter().sum::<f64>() / n;

        // Population variance (1/B) -- legacy behavior we no longer use.
        let variance_population = estimates.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;

        // Bessel-corrected variance (1/(B-1)) -- current behavior.
        let variance_bessel = estimates.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);

        // Our function should match Bessel-corrected variance when ddof = 1.
        let (_, variance_func, _) = compute_bootstrap_variance(&estimates.to_vec(), 1);
        assert!((variance_func - variance_bessel).abs() < 1e-10);

        // And should match the population variance when ddof = 0 (paper Alg 3).
        let (_, variance_pop_func, _) = compute_bootstrap_variance(&estimates.to_vec(), 0);
        assert!((variance_pop_func - variance_population).abs() < 1e-10);

        // And should NOT match the population variance.
        assert!((variance_func - variance_population).abs() > 1e-3);
    }

    // ========================================================================
    // SE precision, CI reasonableness, seed determinism
    // ========================================================================

    /// SE is non-negative for any input.
    #[test]
    fn test_se_positive_definiteness() {
        // Test with various estimate distributions
        let test_cases: Vec<Vec<f64>> = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],           // Normal spread
            vec![0.0, 0.0, 0.0, 0.0, 0.0],           // All zeros
            vec![1.0, 1.0, 1.0, 1.0, 1.0],           // All same
            vec![-5.0, -3.0, 0.0, 3.0, 5.0],         // Symmetric around 0
            vec![100.0, 100.1, 100.2, 100.3, 100.4], // Small variance
            vec![-1000.0, 0.0, 1000.0],              // Large variance
        ];

        for estimates in test_cases {
            let (_, variance, se) = compute_bootstrap_variance(&estimates, 1);

            // SE must be non-negative
            assert!(
                se >= 0.0,
                "SE must be non-negative, got {} for estimates {:?}",
                se,
                estimates
            );

            // Variance must be non-negative
            assert!(
                variance >= 0.0,
                "Variance must be non-negative, got {} for estimates {:?}",
                variance,
                estimates
            );

            // SE should equal sqrt(variance)
            assert!(
                (se - variance.sqrt()).abs() < 1e-12,
                "SE should equal sqrt(variance)"
            );
        }
    }

    /// CI bounds are ordered and contain the mean for a symmetric distribution.
    #[test]
    fn test_ci_reasonableness() {
        // Test with sorted estimates for predictable CI
        let estimates: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let (mean, _, _) = compute_bootstrap_variance(&estimates, 1);
        let (ci_lower, ci_upper) = compute_percentile_ci(&estimates, 0.05);

        // CI bounds should be ordered correctly
        assert!(
            ci_lower <= ci_upper,
            "CI lower ({}) should be <= CI upper ({})",
            ci_lower,
            ci_upper
        );

        // For symmetric distribution, mean should be within CI
        assert!(
            ci_lower <= mean && mean <= ci_upper,
            "Mean ({}) should be within CI [{}, {}]",
            mean,
            ci_lower,
            ci_upper
        );

        // CI should be within data range
        let min_val = estimates.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = estimates.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            ci_lower >= min_val && ci_upper <= max_val,
            "CI [{}, {}] should be within data range [{}, {}]",
            ci_lower,
            ci_upper,
            min_val,
            max_val
        );
    }

    /// CI width increases with the confidence level.
    #[test]
    fn test_ci_width_vs_alpha() {
        let estimates: Vec<f64> = (1..=100).map(|x| x as f64).collect();

        // 90% CI (alpha=0.10)
        let (ci_lower_90, ci_upper_90) = compute_percentile_ci(&estimates, 0.10);
        let width_90 = ci_upper_90 - ci_lower_90;

        // 95% CI (alpha=0.05)
        let (ci_lower_95, ci_upper_95) = compute_percentile_ci(&estimates, 0.05);
        let width_95 = ci_upper_95 - ci_lower_95;

        // 99% CI (alpha=0.01)
        let (ci_lower_99, ci_upper_99) = compute_percentile_ci(&estimates, 0.01);
        let width_99 = ci_upper_99 - ci_lower_99;

        // Wider confidence level should give wider CI
        assert!(
            width_90 < width_95,
            "90% CI width ({}) should be < 95% CI width ({})",
            width_90,
            width_95
        );
        assert!(
            width_95 < width_99,
            "95% CI width ({}) should be < 99% CI width ({})",
            width_95,
            width_99
        );
    }

    /// Identical seeds produce identical sampling sequences.
    #[test]
    fn test_bootstrap_seed_determinism() {
        let classification = UnitClassification {
            control_units: vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            treated_units: vec![10, 11, 12, 13, 14],
            n_control: 10,
            n_treated: 5,
        };

        let base_seed = 12345u64;
        let n_iterations = 50;

        // Run two complete bootstrap sampling sequences with same seed
        let mut samples1: Vec<Vec<usize>> = Vec::new();
        let mut samples2: Vec<Vec<usize>> = Vec::new();

        for b in 0..n_iterations {
            let iter_seed = base_seed.wrapping_add(b as u64);
            samples1.push(stratified_sample(&classification, iter_seed));
            samples2.push(stratified_sample(&classification, iter_seed));
        }

        // All samples should be identical
        for b in 0..n_iterations {
            assert_eq!(
                samples1[b], samples2[b],
                "Iteration {} should produce identical samples with same seed",
                b
            );
        }
    }

    /// Distinct base seeds produce distinct sampling sequences.
    #[test]
    fn test_bootstrap_different_seeds_differ() {
        let classification = UnitClassification {
            control_units: vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            treated_units: vec![10, 11, 12, 13, 14],
            n_control: 10,
            n_treated: 5,
        };

        let n_iterations = 20;

        // Run with two different base seeds
        let mut samples_seed1: Vec<Vec<usize>> = Vec::new();
        let mut samples_seed2: Vec<Vec<usize>> = Vec::new();

        for b in 0..n_iterations {
            samples_seed1.push(stratified_sample(
                &classification,
                100u64.wrapping_add(b as u64),
            ));
            samples_seed2.push(stratified_sample(
                &classification,
                200u64.wrapping_add(b as u64),
            ));
        }

        // At least some samples should differ
        let mut any_different = false;
        for b in 0..n_iterations {
            if samples_seed1[b] != samples_seed2[b] {
                any_different = true;
                break;
            }
        }
        assert!(
            any_different,
            "Different base seeds should produce different sample sequences"
        );
    }

    /// SE converges to the theoretical value for a known distribution.
    #[test]
    fn test_bootstrap_se_precision() {
        // Uniform[0, 12]: theoretical variance = (12-0)^2 / 12 = 12.
        let n_samples = 10000;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let estimates: Vec<f64> = (0..n_samples).map(|_| rng.gen_range(0.0..12.0)).collect();

        let (_, variance, se) = compute_bootstrap_variance(&estimates, 1);

        // Theoretical variance for uniform [0, 12] is 12
        let theoretical_variance: f64 = 12.0;
        let theoretical_se = theoretical_variance.sqrt();

        // With 10000 samples, sample variance should be close to theoretical
        // Allow 5% relative error for statistical variation
        let relative_error = (variance - theoretical_variance).abs() / theoretical_variance;
        assert!(
            relative_error < 0.05,
            "Variance {} should be close to theoretical {} (relative error: {})",
            variance,
            theoretical_variance,
            relative_error
        );

        // SE precision check
        let se_error = (se - theoretical_se).abs();
        assert!(
            se_error < 0.1, // Allow 0.1 absolute error for SE ≈ 3.46
            "SE {} should be close to theoretical {} (error: {})",
            se,
            theoretical_se,
            se_error
        );
    }

    /// Variance formula: V = Σ(τ_b − τ̄)² / (B−1).
    #[test]
    fn test_bootstrap_variance_formula() {
        let estimates = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let n = estimates.len() as f64;

        // Manual calculation using sample variance (1/(B−1), Bessel).
        let mean_manual = estimates.iter().sum::<f64>() / n;
        let variance_manual = estimates
            .iter()
            .map(|x| (x - mean_manual).powi(2))
            .sum::<f64>()
            / (n - 1.0);
        let se_manual = variance_manual.sqrt();

        // Function calculation
        let (mean_func, variance_func, se_func) = compute_bootstrap_variance(&estimates, 1);

        // Verify with high precision (< 1e-12)
        assert!(
            (mean_func - mean_manual).abs() < 1e-12,
            "Mean mismatch: {} vs {}",
            mean_func,
            mean_manual
        );
        assert!(
            (variance_func - variance_manual).abs() < 1e-12,
            "Variance mismatch: {} vs {}",
            variance_func,
            variance_manual
        );
        assert!(
            (se_func - se_manual).abs() < 1e-12,
            "SE mismatch: {} vs {}",
            se_func,
            se_manual
        );

        // Verify expected values
        // Mean = (2+4+6+8+10)/5 = 6
        assert!((mean_func - 6.0).abs() < 1e-12);
        // Variance (Bessel, 1/(B−1)) =
        //   ((2-6)^2 + (4-6)^2 + (6-6)^2 + (8-6)^2 + (10-6)^2) / 4
        // = (16 + 4 + 0 + 4 + 16) / 4 = 40/4 = 10
        assert!((variance_func - 10.0).abs() < 1e-12);
        // SE = sqrt(10) ≈ 3.162
        assert!((se_func - 10.0_f64.sqrt()).abs() < 1e-12);
    }

    /// Percentile CI with linear interpolation on 10 values.
    #[test]
    fn test_percentile_ci_interpolation() {
        let estimates: Vec<f64> = (1..=10).map(|x| x as f64).collect();

        // 95% CI (alpha = 0.05):
        // lower_idx = (10-1) * 0.025 = 0.225 → interpolate between index 0 and 1
        // upper_idx = (10-1) * 0.975 = 8.775 → interpolate between index 8 and 9
        let (ci_lower, ci_upper) = compute_percentile_ci(&estimates, 0.05);

        // Lower: 1 * (1 - 0.225) + 2 * 0.225 = 1.225
        let expected_lower = 1.0 * (1.0 - 0.225) + 2.0 * 0.225;
        assert!(
            (ci_lower - expected_lower).abs() < 1e-10,
            "CI lower {} should be {}",
            ci_lower,
            expected_lower
        );

        // Upper: 9 * (1 - 0.775) + 10 * 0.775 = 9.775
        let expected_upper = 9.0 * (1.0 - 0.775) + 10.0 * 0.775;
        assert!(
            (ci_upper - expected_upper).abs() < 1e-10,
            "CI upper {} should be {}",
            ci_upper,
            expected_upper
        );
    }

    // ========================================================================
    // Regression tests: non-finite value filtering
    // ========================================================================

    #[test]
    fn test_compute_bootstrap_variance_filters_nan() {
        // NaN values should be filtered before computing statistics
        let estimates = vec![1.0, 2.0, f64::NAN, 4.0, 5.0];
        let (mean, variance, se) = compute_bootstrap_variance(&estimates, 1);

        // Should compute stats from [1.0, 2.0, 4.0, 5.0] only
        assert!(mean.is_finite(), "Mean should be finite after NaN filtering");
        assert!((mean - 3.0).abs() < 1e-10, "Mean of [1,2,4,5] = 3.0, got {}", mean);
        assert!(variance.is_finite(), "Variance should be finite");
        assert!(se.is_finite(), "SE should be finite");
    }

    #[test]
    fn test_compute_bootstrap_variance_filters_inf() {
        // Inf values should be filtered before computing statistics
        let estimates = vec![1.0, 2.0, f64::INFINITY, 4.0, f64::NEG_INFINITY];
        let (mean, variance, se) = compute_bootstrap_variance(&estimates, 1);

        // Should compute stats from [1.0, 2.0, 4.0] only
        assert!(mean.is_finite(), "Mean should be finite after Inf filtering");
        let expected_mean = (1.0 + 2.0 + 4.0) / 3.0;
        assert!((mean - expected_mean).abs() < 1e-10);
        assert!(variance.is_finite());
        assert!(se.is_finite());
    }

    #[test]
    fn test_compute_bootstrap_variance_all_nan() {
        // All NaN → treated as empty
        let estimates = vec![f64::NAN, f64::NAN, f64::NAN];
        let (mean, variance, se) = compute_bootstrap_variance(&estimates, 1);
        assert_eq!(mean, 0.0);
        assert_eq!(variance, 0.0);
        assert_eq!(se, 0.0);
    }

    #[test]
    fn test_compute_percentile_ci_filters_nan() {
        // NaN values should be filtered before computing percentiles
        let estimates = vec![1.0, 2.0, f64::NAN, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let (ci_lower, ci_upper) = compute_percentile_ci(&estimates, 0.05);

        assert!(ci_lower.is_finite(), "CI lower should be finite after NaN filtering");
        assert!(ci_upper.is_finite(), "CI upper should be finite after NaN filtering");
        assert!(ci_lower < ci_upper, "CI lower < upper");
    }

    #[test]
    fn test_compute_percentile_ci_all_nan_two_elements() {
        let estimates = vec![f64::NAN, f64::NAN];
        let (ci_lower, ci_upper) = compute_percentile_ci(&estimates, 0.05);
        assert!(ci_lower.is_nan());
        assert!(ci_upper.is_nan());
    }

    // ------------------------------------------------------------------
    // aggregate_att: direct unit tests on the weighted aggregation helper.
    // ------------------------------------------------------------------

    #[test]
    fn test_aggregate_att_unweighted_matches_mean() {
        let pairs = vec![(1.0_f64, 0_usize), (2.0, 1), (3.0, 2), (4.0, 3)];
        let got = aggregate_att(&pairs, None).unwrap();
        assert!((got - 2.5).abs() < 1e-12);
    }

    #[test]
    fn test_aggregate_att_equal_weights_match_unweighted() {
        // Equal pweights must recover the unweighted mean exactly.
        let pairs = vec![(1.0_f64, 0_usize), (2.0, 1), (3.0, 2), (4.0, 3)];
        let w = vec![1.0, 1.0, 1.0, 1.0];
        let got_w = aggregate_att(&pairs, Some(&w)).unwrap();
        let got_u = aggregate_att(&pairs, None).unwrap();
        assert!((got_w - got_u).abs() < 1e-12);
    }

    #[test]
    fn test_aggregate_att_weighted_hand_computation() {
        // Weighted mean: (2*1.0 + 3*3.0) / (2 + 3) = (2 + 9)/5 = 2.2
        let pairs = vec![(1.0_f64, 0_usize), (3.0, 1)];
        let w = vec![2.0, 3.0];
        let got = aggregate_att(&pairs, Some(&w)).unwrap();
        assert!((got - 2.2).abs() < 1e-12);
    }

    #[test]
    fn test_aggregate_att_nonpositive_weights_are_skipped() {
        // Zero / negative / NaN weights must be excluded from both
        // numerator and denominator.  Remaining cells carry full mass.
        let pairs = vec![(1.0_f64, 0_usize), (5.0, 1), (10.0, 2)];
        let w = vec![0.0, -1.0, 4.0];
        let got = aggregate_att(&pairs, Some(&w)).unwrap();
        assert!((got - 10.0).abs() < 1e-12);
    }

    #[test]
    fn test_aggregate_att_all_zero_weights_returns_none() {
        let pairs = vec![(1.0_f64, 0_usize), (2.0, 1)];
        let w = vec![0.0, 0.0];
        assert!(aggregate_att(&pairs, Some(&w)).is_none());
    }

    #[test]
    fn test_aggregate_att_empty_returns_none() {
        let pairs: Vec<(f64, usize)> = Vec::new();
        assert!(aggregate_att(&pairs, None).is_none());
        let w = vec![1.0, 2.0];
        assert!(aggregate_att(&pairs, Some(&w)).is_none());
    }

    // ------------------------------------------------------------------
    // Integration-style checks against the full bootstrap pipeline.
    // ------------------------------------------------------------------

    /// Build a tiny synthetic panel with a DID-style treatment schedule.
    /// Returns (Y, D, control_mask, time_dist) suitable for the twostep
    /// bootstrap entry points.
    fn build_tiny_panel(
        n_periods: usize,
        n_control: usize,
        n_treated: usize,
        first_treated_period: usize,
        seed: u64,
    ) -> (Array2<f64>, Array2<f64>, Array2<u8>, ndarray::Array2<i64>) {
        use rand::SeedableRng;
        use rand::distributions::{Distribution, Uniform};
        use rand_xoshiro::Xoshiro256PlusPlus;

        let n_units = n_control + n_treated;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let dist = Uniform::new(-0.5_f64, 0.5);

        let mut y = Array2::<f64>::zeros((n_periods, n_units));
        let mut d = Array2::<f64>::zeros((n_periods, n_units));
        let mut cm = Array2::<u8>::zeros((n_periods, n_units));

        for i in 0..n_units {
            let alpha = (i as f64) * 0.1;
            for t in 0..n_periods {
                let beta = (t as f64) * 0.05;
                let noise = dist.sample(&mut rng);
                let treated =
                    (i >= n_control) && (t >= first_treated_period);
                let tau = if treated { 1.0 } else { 0.0 };
                y[[t, i]] = alpha + beta + tau + noise;
                d[[t, i]] = if treated { 1.0 } else { 0.0 };
                cm[[t, i]] = if treated { 0 } else { 1 };
            }
        }

        let mut time_dist = ndarray::Array2::<i64>::zeros((n_periods, n_periods));
        for t1 in 0..n_periods {
            for t2 in 0..n_periods {
                time_dist[[t1, t2]] = (t1 as i64 - t2 as i64).abs();
            }
        }

        (y, d, cm, time_dist)
    }

    /// Equal pweights must reproduce the unweighted bootstrap estimates
    /// pointwise: same seed + same sampling path + weighted mean with
    /// uniform weights == arithmetic mean.
    #[test]
    fn test_bootstrap_full_weighted_equal_weights_match_unweighted() {
        let (y, d, cm, td) = build_tiny_panel(8, 6, 3, 5, 42);
        let n_units = y.ncols();

        let n_boot = 15;
        let seed = 7;

        let unweighted = bootstrap_trop_variance_full(
            &y.view(),
            &d.view(),
            &cm.view(),
            &td.view(),
            0.5,
            1.0,
            0.1,
            n_boot,
            50,
            1e-6,
            seed,
            0.05,
            1,
            None,
        );

        let weights = vec![1.0_f64; n_units];
        let weighted = bootstrap_trop_variance_full_weighted(
            &y.view(),
            &d.view(),
            &cm.view(),
            &td.view(),
            0.5,
            1.0,
            0.1,
            n_boot,
            50,
            1e-6,
            seed,
            0.05,
            1,
            &weights,
            None,
        );

        assert_eq!(unweighted.estimates.len(), weighted.estimates.len());
        for (u, w) in unweighted
            .estimates
            .iter()
            .zip(weighted.estimates.iter())
        {
            assert!(
                (u - w).abs() < 1e-10,
                "equal-weight bootstrap must match unweighted pointwise: {} vs {}",
                u,
                w
            );
        }
        assert!((unweighted.se - weighted.se).abs() < 1e-12);
    }
}

//! Exponential-decay weight computation for the TROP estimator.
//!
//! Implements the distance-based weighting scheme defined in Equation (3):
//!
//!   θ_s^{i,t}(λ) = exp(−λ_time · dist_time(s, t))
//!   ω_j^{i,t}(λ) = exp(−λ_unit · dist_unit_{−t}(j, i))
//!
//! where dist_time(s, t) = |t − s| and dist_unit_{−t}(j, i) is the RMSE of
//! outcome differences over mutually untreated periods excluding t.
//!
//! Two modes are provided:
//! - Per-observation weights (Algorithm 2, Twostep): each treated (i, t) pair
//!   receives its own weight matrix W with W[s, j] = θ_s^{i,t} · ω_j^{i,t}.
//! - Global weights (Joint): a single weight matrix δ shared across all treated
//!   observations, with time center at the midpoint of the treated block and
//!   unit distance measured against the average treated trajectory.

use crate::distance::{compute_unit_distance_for_obs, UnitDistanceCache};
use ndarray::{Array1, Array2, ArrayView2};

// ============================================================================
// Adaptive sparse weight representation
// ============================================================================

/// Sparse weight matrix: stores only non-zero entries (weight >= threshold).
///
/// When the non-zero ratio is below [`SPARSE_THRESHOLD_RATIO`],
/// [`compute_weight_matrix_adaptive`] returns this format to reduce
/// subsequent WLS iteration cost; otherwise falls back to dense [`ndarray::Array2`].
///
/// # Fields
/// - `indices`: `(row, col)` index list of non-zero entries
/// - `values`: corresponding weight values, aligned with `indices`
/// - `n_rows` / `n_cols`: original matrix dimensions (T x N)
/// - `threshold`: truncation threshold (`max_weight * 1e-6`), for logging/debug
pub struct SparseWeights {
    /// `(row, col)` indices for entries with weight >= threshold
    pub indices: Vec<(usize, usize)>,
    /// Corresponding weight values, same length as `indices`
    pub values: Vec<f64>,
    /// Number of rows in the original matrix (time dimension T)
    pub n_rows: usize,
    /// Number of columns in the original matrix (unit dimension N)
    pub n_cols: usize,
    /// Truncation threshold (`max_weight * 1e-6`)
    pub threshold: f64,
}

/// Adaptive weight matrix representation.
///
/// Uses `Dense` format when non-zero ratio > [`SPARSE_THRESHOLD_RATIO`] (80%);
/// otherwise uses `Sparse` format to save subsequent iteration overhead.
///
/// For small panels (N < [`SPARSE_MIN_UNITS`], default 100), always returns
/// `Dense` because sparse index management overhead is not worthwhile at
/// small scale.
pub enum WeightMatrix {
    /// Dense format: full T x N matrix
    Dense(Array2<f64>),
    /// Sparse format: stores only non-zero entries
    Sparse(SparseWeights),
}

/// Non-zero weight ratio threshold: falls back to dense when exceeded.
///
/// When the ratio of non-zero weights to total elements exceeds 80%, the
/// memory savings from sparse indexing are insufficient to offset random
/// access overhead, so the dense matrix is used.
pub const SPARSE_THRESHOLD_RATIO: f64 = 0.80;

/// Minimum unit count to enable the sparse path.
///
/// Panels with N < 100 always use dense format: sparse index allocation
/// and traversal are not cost-effective at small scale.
pub const SPARSE_MIN_UNITS: usize = 100;

/// Compute the per-observation weight matrix for the Twostep method.
///
/// For a treated observation (i, t), the weight matrix is the outer product
/// of time weights and unit weights (Equation 3):
///
///   W[s, j] = θ_s · ω_j
///
/// where
///   θ_s = exp(−λ_time · |t − s|)
///   ω_j = exp(−λ_unit · dist_{−t}(j, i))
///
/// Units treated at the target period receive zero weight. The target unit
/// itself receives unit weight 1.0. Weights are unnormalized.
///
/// # Arguments
/// * `y` - Outcome matrix (T × N), column-major
/// * `d` - Treatment indicator matrix (T × N)
/// * `n_periods` - Number of time periods T
/// * `n_units` - Number of units N
/// * `target_unit` - Index of the treated unit i
/// * `target_period` - Index of the treated period t
/// * `lambda_time` - Decay parameter for time weights
/// * `lambda_unit` - Decay parameter for unit weights
/// * `time_dist` - Pre-computed time distance matrix (T × T)
///
/// # Returns
/// Weight matrix W of dimension (T × N)
/// Write the per-observation weight matrix for the Twostep method into an
/// existing buffer.
///
/// This is the buffer-reuse twin of [`compute_weight_matrix`].  It avoids a
/// fresh heap allocation on every call, which matters inside the LOOCV hot
/// loop where the same (T × N) buffer is reused for every control
/// observation.
///
/// # Arguments
/// * `buf` - Pre-allocated buffer of shape (T × N) that will be overwritten
/// * Remaining arguments are identical to [`compute_weight_matrix`].
#[allow(clippy::too_many_arguments)]
pub fn compute_weight_matrix_into(
    buf: &mut Array2<f64>,
    y: &ArrayView2<f64>,
    d: &ArrayView2<f64>,
    n_periods: usize,
    n_units: usize,
    target_unit: usize,
    target_period: usize,
    lambda_time: f64,
    lambda_unit: f64,
    time_dist: &ArrayView2<i64>,
) {
    debug_assert_eq!(buf.dim(), (n_periods, n_units));

    // Decompose into time and unit weight vectors, then form the outer product.
    let (time_weights, unit_weights) = compute_twostep_weight_vectors(
        y, d, n_periods, n_units, target_unit, target_period,
        lambda_time, lambda_unit, time_dist,
    );

    // W[s, j] = θ_s · ω_j
    buf.fill(0.0);
    for t in 0..n_periods {
        for i in 0..n_units {
            buf[[t, i]] = time_weights[t] * unit_weights[i];
        }
    }
}

/// Compute the per-observation weight matrix for the Twostep method.
///
/// Thin wrapper around [`compute_weight_matrix_into`] that allocates and
/// returns a fresh (T × N) buffer.
#[allow(clippy::too_many_arguments)]
pub fn compute_weight_matrix(
    y: &ArrayView2<f64>,
    d: &ArrayView2<f64>,
    n_periods: usize,
    n_units: usize,
    target_unit: usize,
    target_period: usize,
    lambda_time: f64,
    lambda_unit: f64,
    time_dist: &ArrayView2<i64>,
) -> Array2<f64> {
    let mut buf = Array2::<f64>::zeros((n_periods, n_units));
    compute_weight_matrix_into(
        &mut buf, y, d, n_periods, n_units, target_unit, target_period,
        lambda_time, lambda_unit, time_dist,
    );
    buf
}

/// Cached twin of [`compute_weight_matrix`] that uses a pre-built
/// [`UnitDistanceCache`] to short-circuit the per-call O(T) distance
/// computation.
///
/// Produces numerically identical weights to the non-cached version (up to
/// floating-point round-off from the `sum − (Y_t_i − Y_t_j)²` factoring).
/// Prefer this entry point inside LOOCV / bootstrap hot loops.
/// Write the cached per-observation weight matrix into an existing buffer.
///
/// Buffer-reuse twin of [`compute_weight_matrix_cached`].  The pre-built
/// [`UnitDistanceCache`] avoids the per-call O(T) pairwise distance
/// computation; the pre-allocated buffer avoids the per-call (T × N) heap
/// allocation.
#[allow(clippy::too_many_arguments)]
pub fn compute_weight_matrix_cached_into(
    buf: &mut Array2<f64>,
    y: &ArrayView2<f64>,
    d: &ArrayView2<f64>,
    cache: &UnitDistanceCache,
    n_periods: usize,
    n_units: usize,
    target_unit: usize,
    target_period: usize,
    lambda_time: f64,
    lambda_unit: f64,
    time_dist: &ArrayView2<i64>,
) {
    debug_assert_eq!(buf.dim(), (n_periods, n_units));

    let (time_weights, unit_weights) = compute_twostep_weight_vectors_cached(
        y, d, cache, n_periods, n_units, target_unit, target_period,
        lambda_time, lambda_unit, time_dist,
    );

    buf.fill(0.0);
    for t in 0..n_periods {
        for i in 0..n_units {
            buf[[t, i]] = time_weights[t] * unit_weights[i];
        }
    }
}

/// Cached twin of [`compute_weight_matrix`].
///
/// Thin wrapper around [`compute_weight_matrix_cached_into`] that allocates
/// and returns a fresh (T × N) buffer.
#[allow(clippy::too_many_arguments)]
pub fn compute_weight_matrix_cached(
    y: &ArrayView2<f64>,
    d: &ArrayView2<f64>,
    cache: &UnitDistanceCache,
    n_periods: usize,
    n_units: usize,
    target_unit: usize,
    target_period: usize,
    lambda_time: f64,
    lambda_unit: f64,
    time_dist: &ArrayView2<i64>,
) -> Array2<f64> {
    let mut buf = Array2::<f64>::zeros((n_periods, n_units));
    compute_weight_matrix_cached_into(
        &mut buf, y, d, cache, n_periods, n_units, target_unit, target_period,
        lambda_time, lambda_unit, time_dist,
    );
    buf
}

/// Compute the global weight matrix for the Joint method.
///
/// Unlike the Twostep method, the Joint method uses a single weight matrix
/// shared across all treated observations. The weight matrix is the outer
/// product δ[s, j] = δ_time[s] · δ_unit[j], where:
///
///   δ_time[s] = exp(−λ_time · |s − center|)
///     with center = T − T_post / 2 (midpoint of the treated block)
///
///   δ_unit[j] = exp(−λ_unit · RMSE_j)
///     where RMSE_j is the root-mean-square deviation of unit j's
///     pre-treatment outcomes from the average treated trajectory.
///
/// Weights are unnormalized.
///
/// # Arguments
/// * `y` - Outcome matrix (T × N)
/// * `d` - Treatment indicator matrix (T × N)
/// * `lambda_time` - Decay parameter for time weights
/// * `lambda_unit` - Decay parameter for unit weights
/// * `treated_periods` - Number of post-treatment periods T_post
///
/// # Returns
/// Weight matrix δ of dimension (T × N)
/// Write the global weight matrix for the Joint method into an existing buffer.
///
/// Buffer-reuse twin of [`compute_joint_weights`].  The caller must supply a
/// buffer whose shape equals `(y.nrows(), y.ncols())`.
#[allow(clippy::too_many_arguments)]
pub fn compute_joint_weights_into(
    buf: &mut Array2<f64>,
    y: &ArrayView2<f64>,
    d: &ArrayView2<f64>,
    lambda_time: f64,
    lambda_unit: f64,
    treated_periods: usize,
) {
    let n_periods = y.nrows();
    let n_units = y.ncols();
    debug_assert_eq!(buf.dim(), (n_periods, n_units));

    let (delta_time, delta_unit) = compute_joint_weight_vectors(
        y, d, lambda_time, lambda_unit, treated_periods,
    );

    // δ[s, j] = δ_time[s] · δ_unit[j] · (1 − D_{s,j})
    //
    // The (1 − D) masking implements the paper's Equation 2 objective, which
    // sums the quadratic loss only over CONTROL observations. Zeroing δ at
    // treated cells here ensures the downstream WLS never fits them, which in
    // turn makes τ a post-hoc residual (no D column in the design matrix).
    buf.fill(0.0);
    for t in 0..n_periods {
        for i in 0..n_units {
            buf[[t, i]] = delta_time[t] * delta_unit[i] * (1.0 - d[[t, i]]);
        }
    }
}

/// Compute the global weight matrix for the Joint method.
///
/// Thin wrapper around [`compute_joint_weights_into`] that allocates and
/// returns a fresh buffer.
pub fn compute_joint_weights(
    y: &ArrayView2<f64>,
    d: &ArrayView2<f64>,
    lambda_time: f64,
    lambda_unit: f64,
    treated_periods: usize,
) -> Array2<f64> {
    let n_periods = y.nrows();
    let n_units = y.ncols();
    let mut buf = Array2::<f64>::zeros((n_periods, n_units));
    compute_joint_weights_into(
        &mut buf, y, d, lambda_time, lambda_unit, treated_periods,
    );
    buf
}

/// Decompose Twostep weights into separate time and unit vectors.
///
/// Implements the unnormalized exponential kernels of paper Eq. (3):
///   θ_s = exp(−λ_time · |t − s|)            (no normalization)
///   ω_j = exp(−λ_unit · dist_{−t}(j, i))    (no normalization)
///
/// The kernels are left **unnormalized** on purpose.  `estimate_model`
/// rescales the inner gradient step by `w_max = max(W)` (the Lipschitz
/// constant of the quadratic loss) and thresholds the SVD at
/// `λ_nn / (2·w_max)`; normalizing the kernels here would implicitly
/// rescale λ_nn and make fits at any fixed λ incomparable across panels
/// of different weight magnitudes.  The unnormalized convention is the
/// one paper Eq. (3) states explicitly.
///
/// All units j ≠ target_unit receive distance-based weights.
/// Treated-cell exclusion is handled by the control_mask in estimate_model.
/// The target unit i always gets ω_i = 1, because `dist_{−t}(i, i) = 0` by
/// the Eq. (3) definition, so `ω_i = exp(−λ_unit · 0) = 1` for every
/// λ_unit.  Hard-coding this value avoids a redundant `compute_unit_distance_for_obs`
/// call; it is a numerical optimization, not a modelling choice.
///
/// # Arguments
/// Same as [`compute_weight_matrix`].
///
/// # Returns
/// `(time_weights, unit_weights)` — both `Array1<f64>` of length T and N
#[allow(clippy::too_many_arguments)]
pub fn compute_twostep_weight_vectors(
    y: &ArrayView2<f64>,
    d: &ArrayView2<f64>,
    n_periods: usize,
    n_units: usize,
    target_unit: usize,
    target_period: usize,
    lambda_time: f64,
    lambda_unit: f64,
    time_dist: &ArrayView2<i64>,
) -> (Array1<f64>, Array1<f64>) {
    // θ_s = exp(−λ_time · |t − s|), unnormalized.
    let time_weights: Array1<f64> = Array1::from_shape_fn(n_periods, |s| {
        let dist = time_dist[[target_period, s]] as f64;
        (-lambda_time * dist).exp()
    });

    // ω_j = exp(−λ_unit · dist_{−t}(j, i)), unnormalized.
    // All units j ≠ target_unit get distance-based weights.
    // Treated-cell exclusion is handled by the control_mask in estimate_model.
    let mut unit_weights = Array1::<f64>::zeros(n_units);

    if lambda_unit == 0.0 {
        // Uniform kernel: all units j ≠ target_unit get ω_j = 1.
        // Treated-cell exclusion is handled by the control_mask in estimate_model.
        for j in 0..n_units {
            if j != target_unit {
                unit_weights[j] = 1.0;
            }
        }
    } else {
        for j in 0..n_units {
            if j != target_unit {
                let dist = compute_unit_distance_for_obs(y, d, j, target_unit, target_period);
                if dist.is_finite() {
                    unit_weights[j] = (-lambda_unit * dist).exp();
                }
                // Units with infinite distance stay at 0.
            }
        }
    }

    // Target unit always participates with ω_i = 1.
    unit_weights[target_unit] = 1.0;

    (time_weights, unit_weights)
}

/// Cached twin of [`compute_twostep_weight_vectors`].  See the non-cached
/// version for the complete specification; this routine only replaces the
/// per-pair O(T) distance computation by a cache lookup.
#[allow(clippy::too_many_arguments)]
pub fn compute_twostep_weight_vectors_cached(
    y: &ArrayView2<f64>,
    _d: &ArrayView2<f64>,
    cache: &UnitDistanceCache,
    n_periods: usize,
    n_units: usize,
    target_unit: usize,
    target_period: usize,
    lambda_time: f64,
    lambda_unit: f64,
    time_dist: &ArrayView2<i64>,
) -> (Array1<f64>, Array1<f64>) {
    // θ_s = exp(−λ_time · |t − s|), identical to the non-cached path.
    let time_weights: Array1<f64> = Array1::from_shape_fn(n_periods, |s| {
        let dist = time_dist[[target_period, s]] as f64;
        (-lambda_time * dist).exp()
    });

    let mut unit_weights = Array1::<f64>::zeros(n_units);

    if lambda_unit == 0.0 {
        for j in 0..n_units {
            if j != target_unit {
                unit_weights[j] = 1.0;
            }
        }
    } else {
        for j in 0..n_units {
            if j != target_unit {
                let dist = cache.distance(y, j, target_unit, target_period);
                if dist.is_finite() {
                    unit_weights[j] = (-lambda_unit * dist).exp();
                }
            }
        }
    }

    unit_weights[target_unit] = 1.0;

    (time_weights, unit_weights)
}

/// Decompose Joint weights into separate time and unit vectors.
///
/// The joint (homogeneous-τ) estimator sits in paper Remark 6.1, which
/// only sketches the aggregation and does *not* specify concrete time /
/// unit kernels.  The distance definitions below are therefore a
/// reasonable adaptation of Eq. (3) to the shared-weight setting, not a
/// restatement of a formula in the paper:
///
///   δ_time[s] = exp(−λ_time · |s − center|)
///     with center = T − T_post / 2 (midpoint of the treated block).
///     Paper Eq. (3) defines θ_s(λ) = exp(−λ · |s − t|) per *target*
///     treated period t in Algorithm 1; Remark 6.1 is silent on the
///     concrete aggregation kernel under the shared-weight (joint)
///     setting, so the post-block midpoint is adopted here as an
///     engineering choice and pinned by the released numerical
///     baseline.  In particular, **T_post = 1 is *not* special-cased**
///     to `center = T − 1`: that would pin the kernel exactly to the
///     single treated period (Eq. (3) exact) but would diverge from
///     the released numerical baseline by 0.5 periods, breaking the
///     end-to-end fidelity tests (`test_joint_outer_convergence_parity.do`,
///     CPS / PWT).
///
///   δ_unit[j] = exp(−λ_unit · RMSE_j)
///     where RMSE_j is the RMSE of unit j's pre-treatment outcomes against
///     the average treated trajectory.  The RMSE replaces the leave-one-
///     period-out pairwise distance of Eq. (3) because a single shared
///     weight vector must summarize every treated cell at once.
///
/// Units with no valid pre-treatment observations receive δ_unit = 0.
/// When there are no pre-treatment periods (Remark 6.1 is silent on this
/// degenerate case), all units receive δ_unit = 1; see the branch
/// comment below.
///
/// # Arguments
/// Same as [`compute_joint_weights`].
///
/// # Returns
/// `(delta_time, delta_unit)` — both `Array1<f64>` of length T and N
pub fn compute_joint_weight_vectors(
    y: &ArrayView2<f64>,
    d: &ArrayView2<f64>,
    lambda_time: f64,
    lambda_unit: f64,
    treated_periods: usize,
) -> (Array1<f64>, Array1<f64>) {
    let n_periods = y.nrows();
    let n_units = y.ncols();

    // Identify ever-treated units.
    let mut treated_unit_idx: Vec<usize> = Vec::new();
    for i in 0..n_units {
        if (0..n_periods).any(|t| d[[t, i]] == 1.0) {
            treated_unit_idx.push(i);
        }
    }

    // δ_time[s] = exp(−λ_time · |s − center|), with
    //     center = T − T_post / 2     (midpoint of the treated block)
    //
    // Paper Eq. (3) is defined per *target* treated period in Algorithm 1;
    // Remark 6.1 is silent on the concrete aggregation kernel under the
    // shared-weight (joint) setting, so the post-block midpoint is adopted
    // here as an engineering choice.  The
    // `test_joint_outer_convergence_parity.do` regression suite locks the
    // end-to-end ATT to the released numerical baseline on CPS log-wage,
    // PWT log-GDP, and the simulated seed-42 panel at tol = 1e-6.
    // T_post = 1 is intentionally *not* special-cased to `center = T − 1`:
    // doing so would be a cleaner match to Eq. (3) but would break the
    // released-baseline fidelity by 0.5 periods at the treated cell.
    let center = n_periods as f64 - treated_periods as f64 / 2.0;
    let mut delta_time = Array1::<f64>::zeros(n_periods);
    for t in 0..n_periods {
        let dist = (t as f64 - center).abs();
        delta_time[t] = (-lambda_time * dist).exp();
    }

    // δ_unit[j] = exp(−λ_unit · RMSE_j), where RMSE_j is the root-mean-square
    // deviation of unit j from the average treated trajectory over pre-periods.
    let n_pre = n_periods.saturating_sub(treated_periods);

    // Average outcome trajectory across treated units.
    let mut average_treated = Array1::<f64>::from_elem(n_periods, f64::NAN);
    if !treated_unit_idx.is_empty() {
        for t in 0..n_periods {
            let mut sum = 0.0;
            let mut count = 0;
            for &i in &treated_unit_idx {
                if y[[t, i]].is_finite() {
                    sum += y[[t, i]];
                    count += 1;
                }
            }
            if count > 0 {
                average_treated[t] = sum / count as f64;
            }
        }
    }

    let mut delta_unit = Array1::<f64>::zeros(n_units);
    for i in 0..n_units {
        if n_pre > 0 {
            let mut sum_sq = 0.0;
            let mut n_valid = 0;
            for t in 0..n_pre {
                if y[[t, i]].is_finite() && average_treated[t].is_finite() {
                    let diff = average_treated[t] - y[[t, i]];
                    sum_sq += diff * diff;
                    n_valid += 1;
                }
            }
            let dist = if n_valid > 0 {
                (sum_sq / n_valid as f64).sqrt()
            } else {
                f64::INFINITY
            };
            // Guard against NaN from IEEE 754: −0.0 × ∞ = NaN, exp(NaN) = NaN.
            // Units with no valid pre-period data receive zero weight.
            delta_unit[i] = if dist.is_infinite() {
                0.0
            } else {
                (-lambda_unit * dist).exp()
            };
        } else {
            // No pre-treatment periods (treated_periods >= n_periods).
            //
            // Paper Remark 6.1 (homogeneous-effect aggregation) is silent on
            // this degenerate case: the RMSE-over-pre-periods recipe for
            // δ_unit simply has no input.  In the Stata pipeline such data
            // is already rejected upstream — joint method requires
            // non-staggered adoption, and panels with no pre-periods fail
            // the minimum control-period overlap check
            // (`_trop_chk_common_ctrl_periods`).
            //
            // This branch therefore only runs under direct FFI / unit-test
            // usage.  We return uniform unit weights (δ_unit ≡ 1) so the
            // downstream code remains numerically well-defined rather than
            // producing NaNs, but emit a stderr warning on the first unit
            // so a caller who bypassed the upstream validation sees a
            // visible signal.
            //
            // Audit: 2026-04 first-principles review (section B.3).  The
            // uniform-fallback decision is documented in the
            // `compute_joint_weight_vectors` header comment; removing the
            // stderr warning requires also updating the `trop.sthlp`
            // documentation of the degenerate branch.
            if i == 0 {
                eprintln!(
                    "trop_stata warning: compute_joint_weight_vectors \
                     invoked with n_pre == 0 (treated_periods = {}, \
                     n_periods = {}); falling back to uniform unit weights. \
                     Upstream Stata validation should have rejected this \
                     input; see the function header for the Remark 6.1 \
                     rationale.",
                    treated_periods, n_periods
                );
            }
            delta_unit[i] = 1.0;
        }
    }

    (delta_time, delta_unit)
}

// ============================================================================
// Adaptive sparse weight computation
// ============================================================================

/// Adaptively compute per-observation weight matrix for the Twostep method,
/// automatically selecting storage format based on sparsity.
///
/// Uses the same computation logic as [`compute_weight_matrix`], but adds an
/// adaptive sparse path:
///
/// 1. Compute full weight vectors (theta_time, omega_unit)
/// 2. Find `max_weight = max(theta_s) * max(omega_j)`
/// 3. Set truncation threshold `threshold = max_weight * 1e-6`
/// 4. Count entries >= threshold:
///    - If N < [`SPARSE_MIN_UNITS`] (100), always return `Dense` (overhead not worth it)
///    - If non-zero ratio > [`SPARSE_THRESHOLD_RATIO`] (80%), return `Dense`
///    - Otherwise return `Sparse` (stores only non-zero entries)
///
/// # Arguments
/// Same as [`compute_weight_matrix`].
///
/// # Returns
/// [`WeightMatrix::Dense`] or [`WeightMatrix::Sparse`], numerically equivalent.
#[allow(clippy::too_many_arguments)]
pub fn compute_weight_matrix_adaptive(
    y: &ArrayView2<f64>,
    d: &ArrayView2<f64>,
    n_periods: usize,
    n_units: usize,
    target_unit: usize,
    target_period: usize,
    lambda_time: f64,
    lambda_unit: f64,
    time_dist: &ArrayView2<i64>,
) -> WeightMatrix {
    let (time_weights, unit_weights) = compute_twostep_weight_vectors(
        y, d, n_periods, n_units, target_unit, target_period,
        lambda_time, lambda_unit, time_dist,
    );

    // Compute max weight to determine truncation threshold
    let max_time = time_weights.iter().cloned().fold(0.0_f64, f64::max);
    let max_unit = unit_weights.iter().cloned().fold(0.0_f64, f64::max);
    let max_weight = max_time * max_unit;
    let threshold = if max_weight > 0.0 { max_weight * 1e-6 } else { 0.0 };

    // Small panels: always return dense (sparse overhead not worthwhile)
    if n_units < SPARSE_MIN_UNITS {
        let mut buf = Array2::<f64>::zeros((n_periods, n_units));
        for t in 0..n_periods {
            for i in 0..n_units {
                buf[[t, i]] = time_weights[t] * unit_weights[i];
            }
        }
        return WeightMatrix::Dense(buf);
    }

    let total_elements = n_periods * n_units;

    // Pre-count non-zero elements to decide whether sparse format is worthwhile
    let nnz = time_weights
        .iter()
        .flat_map(|&tw| {
            unit_weights.iter().map(move |&uw| tw * uw)
        })
        .filter(|&v| v >= threshold)
        .count();

    let nonzero_ratio = nnz as f64 / total_elements as f64;

    if nonzero_ratio > SPARSE_THRESHOLD_RATIO {
        // Dense path: non-zero ratio too high, return full matrix
        let mut buf = Array2::<f64>::zeros((n_periods, n_units));
        for t in 0..n_periods {
            for i in 0..n_units {
                buf[[t, i]] = time_weights[t] * unit_weights[i];
            }
        }
        WeightMatrix::Dense(buf)
    } else {
        // Sparse path: store only entries >= threshold
        let mut indices = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);
        for t in 0..n_periods {
            for i in 0..n_units {
                let v = time_weights[t] * unit_weights[i];
                if v >= threshold {
                    indices.push((t, i));
                    values.push(v);
                }
            }
        }
        WeightMatrix::Sparse(SparseWeights {
            indices,
            values,
            n_rows: n_periods,
            n_cols: n_units,
            threshold,
        })
    }
}

/// Adaptively compute per-observation weight matrix using pre-built distance cache.
///
/// Same logic as [`compute_weight_matrix_adaptive`], but leverages
/// [`UnitDistanceCache`] to skip per-query O(T) distance computation,
/// suitable for use in LOOCV / bootstrap hot loops.
#[allow(clippy::too_many_arguments)]
pub fn compute_weight_matrix_adaptive_cached(
    y: &ArrayView2<f64>,
    d: &ArrayView2<f64>,
    cache: &UnitDistanceCache,
    n_periods: usize,
    n_units: usize,
    target_unit: usize,
    target_period: usize,
    lambda_time: f64,
    lambda_unit: f64,
    time_dist: &ArrayView2<i64>,
) -> WeightMatrix {
    let (time_weights, unit_weights) = compute_twostep_weight_vectors_cached(
        y, d, cache, n_periods, n_units, target_unit, target_period,
        lambda_time, lambda_unit, time_dist,
    );

    let max_time = time_weights.iter().cloned().fold(0.0_f64, f64::max);
    let max_unit = unit_weights.iter().cloned().fold(0.0_f64, f64::max);
    let max_weight = max_time * max_unit;
    let threshold = if max_weight > 0.0 { max_weight * 1e-6 } else { 0.0 };

    if n_units < SPARSE_MIN_UNITS {
        let mut buf = Array2::<f64>::zeros((n_periods, n_units));
        for t in 0..n_periods {
            for i in 0..n_units {
                buf[[t, i]] = time_weights[t] * unit_weights[i];
            }
        }
        return WeightMatrix::Dense(buf);
    }

    let total_elements = n_periods * n_units;
    let nnz = time_weights
        .iter()
        .flat_map(|&tw| unit_weights.iter().map(move |&uw| tw * uw))
        .filter(|&v| v >= threshold)
        .count();

    let nonzero_ratio = nnz as f64 / total_elements as f64;

    if nonzero_ratio > SPARSE_THRESHOLD_RATIO {
        let mut buf = Array2::<f64>::zeros((n_periods, n_units));
        for t in 0..n_periods {
            for i in 0..n_units {
                buf[[t, i]] = time_weights[t] * unit_weights[i];
            }
        }
        WeightMatrix::Dense(buf)
    } else {
        let mut indices = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);
        for t in 0..n_periods {
            for i in 0..n_units {
                let v = time_weights[t] * unit_weights[i];
                if v >= threshold {
                    indices.push((t, i));
                    values.push(v);
                }
            }
        }
        WeightMatrix::Sparse(SparseWeights {
            indices,
            values,
            n_rows: n_periods,
            n_cols: n_units,
            threshold,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_weight_matrix_structure() {
        // 3 periods, 2 units, all control.
        let y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let d = array![[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]];
        let time_dist = array![[0i64, 1, 2], [1, 0, 1], [2, 1, 0]];

        let weights = compute_weight_matrix(
            &y.view(),
            &d.view(),
            3,
            2,
            0,
            1,
            0.5,
            0.5,
            &time_dist.view(),
        );

        let total: f64 = weights.sum();
        assert!(total > 0.0, "Weights should be positive, got {}", total);
    }

    #[test]
    fn test_weight_matrix_zero_lambda() {
        // λ_time = λ_unit = 0 ⟹ all kernels equal exp(0) = 1 (unnormalized).
        // Every (t, i) cell takes the constant value 1.0 and the total = T · N.
        let y = array![[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]];
        let d = array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        let time_dist = array![[0i64, 1, 2], [1, 0, 1], [2, 1, 0]];

        let weights = compute_weight_matrix(
            &y.view(),
            &d.view(),
            3,
            3,
            0,
            1,
            0.0,
            0.0,
            &time_dist.view(),
        );

        let expected = 1.0;
        for t in 0..3 {
            for i in 0..3 {
                assert!(
                    (weights[[t, i]] - expected).abs() < 1e-10,
                    "Expected uniform weight {}, got {} at [{}, {}]",
                    expected,
                    weights[[t, i]],
                    t,
                    i
                );
            }
        }

        let total: f64 = weights.sum();
        assert!(
            (total - 9.0).abs() < 1e-10,
            "Unnormalized weights should sum to T·N = 9, got {}",
            total
        );
    }

    #[test]
    fn test_joint_weights_masked_at_treated() {
        // After (1 − D) masking, δ[t, i] = 0 whenever d[t, i] = 1.
        let y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let d = array![
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0]
        ];

        let weights = compute_joint_weights(&y.view(), &d.view(), 0.5, 0.5, 2);

        // Treated cells must be exactly zero.
        assert_eq!(weights[[2, 1]], 0.0, "δ must be 0 at treated cell (2, 1)");
        assert_eq!(weights[[3, 1]], 0.0, "δ must be 0 at treated cell (3, 1)");

        // Untreated cells remain strictly positive.
        let control_sum: f64 = (0..4).flat_map(|t| (0..2).map(move |i| (t, i)))
            .filter(|(t, i)| d[[*t, *i]] == 0.0)
            .map(|(t, i)| weights[[t, i]])
            .sum();
        assert!(control_sum > 0.0, "Control weights should be positive");
    }

    #[test]
    fn test_joint_weights_zero_pre_periods_returns_finite() {
        // Regression guard for audit finding B-1 / B.3: paper Remark 6.1 is
        // silent on treated_periods == n_periods (no pre-periods), so
        // trop_stata intentionally returns uniform unit weights to keep
        // downstream code NaN-free.  The Rust branch emits an eprintln
        // warning (see B.3 audit note).  This test pins the defensive
        // behaviour so a future refactor must deliberately change both the
        // fallback and the stderr signal.
        let y = array![[1.0, 2.0], [2.0, 3.0]];
        let d = array![[1.0, 1.0], [1.0, 1.0]];
        let (delta_time, delta_unit) =
            compute_joint_weight_vectors(&y.view(), &d.view(), 0.5, 0.5, 2);

        assert_eq!(delta_unit.len(), 2);
        for &w in delta_unit.iter() {
            assert_eq!(w, 1.0, "n_pre == 0 must yield uniform unit weights");
        }

        for &t in delta_time.iter() {
            assert!(t.is_finite(), "time weights must remain finite");
        }

        // Full matrix: all cells treated ⟹ δ · (1 − D) = 0 everywhere.
        let weights = compute_joint_weights(&y.view(), &d.view(), 0.5, 0.5, 2);
        for &w in weights.iter() {
            assert_eq!(w, 0.0, "fully-treated panel must mask all cells to zero");
        }
    }

    #[test]
    fn test_joint_weights_zero_pre_periods_larger_panel() {
        // B.3 audit: ensure the n_pre == 0 fallback scales uniformly across
        // a larger panel and does not degrade to NaN/Inf when more units are
        // involved.  The panel is fully treated (D == 1 everywhere) so the
        // fallback branch is exercised across all N units.
        let y = array![
            [1.0, 2.0, 3.0, 4.0],
            [1.5, 2.5, 3.5, 4.5],
            [2.0, 3.0, 4.0, 5.0],
        ];
        let d = array![
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ];
        let (delta_time, delta_unit) =
            compute_joint_weight_vectors(&y.view(), &d.view(), 0.1, 0.5, 3);

        assert_eq!(delta_time.len(), 3);
        assert_eq!(delta_unit.len(), 4);
        for &t in delta_time.iter() {
            assert!(t.is_finite() && t > 0.0, "time weights must be finite > 0");
        }
        for &w in delta_unit.iter() {
            assert_eq!(
                w, 1.0,
                "B.3: every unit weight in the n_pre == 0 fallback must be 1.0"
            );
        }
    }

    #[test]
    fn test_joint_weights_time_center() {
        // λ_time = 0 ⟹ uniform time weights. With (1 − D) masking, only
        // untreated rows are expected to have equal totals.
        let y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let d = array![[0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 1.0]];

        let weights = compute_joint_weights(&y.view(), &d.view(), 0.0, 0.0, 2);

        // Pre-period rows (t = 0, 1) have every unit untreated ⟹ identical sums.
        let row0_sum: f64 = weights.row(0).sum();
        let row1_sum: f64 = weights.row(1).sum();
        assert!((row0_sum - row1_sum).abs() < 1e-10);

        // Post-period rows (t = 2, 3) have unit 1 masked to zero; the
        // remaining mass equals the weight of the surviving unit 0.
        let row2_sum: f64 = weights.row(2).sum();
        let row3_sum: f64 = weights.row(3).sum();
        assert!((row2_sum - row3_sum).abs() < 1e-10);
        assert!(row2_sum < row1_sum, "Treated row total must shrink after masking");
    }

    // ========================================================================
    // Joint weight property tests
    // ========================================================================

    #[test]
    fn test_joint_weights_outer_product_structure() {
        // δ[t, i] = δ_time[t] · δ_unit[i] · (1 − D[t, i]). The rank-1 outer
        // product only holds on untreated cells; treated cells are masked to 0.
        let y = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0]
        ];
        let d = array![
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0]
        ];

        let delta = compute_joint_weights(&y.view(), &d.view(), 0.5, 0.5, 2);

        // Verify rank-1 structure on the untreated sub-grid only.
        let (n_periods, n_units) = delta.dim();
        let is_control = |t: usize, i: usize| d[[t, i]] == 0.0;

        for t in 0..n_periods {
            for s in 0..n_periods {
                for i in 0..n_units {
                    for j in 0..n_units {
                        // Require all four cells to be untreated so the mask
                        // does not break the outer-product identity.
                        if !(is_control(t, i) && is_control(t, j)
                            && is_control(s, i) && is_control(s, j))
                        {
                            continue;
                        }
                        if delta[[t, j]] > 1e-10 && delta[[s, j]] > 1e-10 {
                            let ratio_t = delta[[t, i]] / delta[[t, j]];
                            let ratio_s = delta[[s, i]] / delta[[s, j]];
                            assert!(
                                (ratio_t - ratio_s).abs() < 1e-8,
                                "Outer product structure violated at t={}, s={}, i={}, j={}",
                                t,
                                s,
                                i,
                                j
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_joint_weights_time_formula() {
        // δ_time[t] = exp(−λ_time · |t − center|), center = T − T_post / 2.
        let y = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0]
        ];
        let d = array![
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0]
        ];

        let lambda_time = 0.3;
        let treated_periods = 2;
        let n_periods = 6;
        let center = n_periods as f64 - treated_periods as f64 / 2.0;

        let delta = compute_joint_weights(&y.view(), &d.view(), lambda_time, 0.0, treated_periods);

        // With λ_unit = 0, δ_unit[j] = 1 for all j, so δ[t, j] = δ_time[t].
        for t in 0..n_periods {
            let expected_time_weight = (-lambda_time * (t as f64 - center).abs()).exp();
            let actual = delta[[t, 0]];
            assert!(
                (actual - expected_time_weight).abs() < 1e-8,
                "Time weight formula incorrect at t={}: expected {}, got {}",
                t,
                expected_time_weight,
                actual
            );
        }
    }

    /// Numerical-baseline guard (supersedes the reverted P0-3 draft): for
    /// the homogeneous-τ aggregation in `compute_joint_weight_vectors` we
    /// deliberately keep `center = T − T_post / 2` even when `T_post = 1`,
    /// matching the released numerical baseline.  An earlier draft "fixed"
    /// T_post = 1 to `center = T − 1` on the grounds that it recovers paper
    /// Eq. (3) exactly; that draft passed this unit test but broke the
    /// end-to-end `test_joint_outer_convergence_parity.do` regressions on
    /// CPS log-wage (|Δτ| ≈ 2.94e-4) and PWT log-GDP (|Δτ| ≈ 6.32e-4), which
    /// lock the ATT to the numerical baseline at tol = 1e-6.  Paper Remark
    /// 6.1 leaves the joint aggregation kernel unspecified, so the
    /// engineering choice (post-block midpoint) is preserved as the
    /// intended behavior; the 0.5-period offset is pinned here.
    #[test]
    fn test_joint_weights_time_single_post_period_matches_midpoint() {
        let y = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0]
        ];
        // Unit 1 treated only at the final period (T_post = 1).
        let d = array![
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0]
        ];

        let lambda_time = 0.4;
        let n_periods = 5usize;
        let treated_periods = 1usize;

        let delta = compute_joint_weights(&y.view(), &d.view(), lambda_time, 0.0, treated_periods);

        // Engineering-choice convention: center = T − T_post / 2 = 5 − 0.5 = 4.5.
        // With λ_unit = 0 the unit factor is 1 on control cells so δ[s, 0] == δ_time[s].
        let center = n_periods as f64 - treated_periods as f64 / 2.0;
        for s in 0..n_periods {
            let expected = (-lambda_time * (s as f64 - center).abs()).exp();
            let actual = delta[[s, 0]];
            assert!(
                (actual - expected).abs() < 1e-10,
                "joint δ_time[s={}] must use the post-block midpoint center (T − T_post/2): \
                 expected {}, got {}",
                s,
                expected,
                actual
            );
        }

        // At the treated period the kernel sits 0.5 periods away from the
        // midpoint center, so δ_time[T−1] = exp(−λ_time · 0.5) < 1.  This
        // is the 0.5-period offset relative to a hypothetical "center =
        // treated period" kernel; we lock it in so any future refactor has
        // to opt in consciously if it intends to diverge from the
        // released numerical baseline.
        let expected_at_treated = (-lambda_time * 0.5f64).exp();
        assert!(
            (delta[[n_periods - 1, 0]] - expected_at_treated).abs() < 1e-12,
            "joint δ_time at the treated period must equal exp(−λ_time · 0.5) \
             under post-block midpoint convention; got {}, expected {}",
            delta[[n_periods - 1, 0]],
            expected_at_treated
        );
    }

    #[test]
    fn test_joint_weights_unit_formula() {
        // δ_unit[j] = exp(−λ_unit · RMSE_j) over pre-treatment periods.
        // Unit 2 has a trajectory far from the treated average ⟹ lower weight.
        let y = array![
            [1.0, 1.0, 10.0],
            [2.0, 2.0, 20.0],
            [3.0, 3.0, 30.0],
            [4.0, 5.0, 40.0]
        ];
        let d = array![
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ];

        let lambda_unit = 0.1;
        let delta = compute_joint_weights(&y.view(), &d.view(), 0.0, lambda_unit, 1);

        assert!(
            delta[[0, 2]] < delta[[0, 0]],
            "Unit far from treated average should have lower weight"
        );
    }

    #[test]
    fn test_joint_weights_non_negative() {
        // exp(·) ≥ 0 for all real arguments; verify across λ combinations.
        let y = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0]
        ];
        let d = array![
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0]
        ];

        for lambda_time in [0.0, 0.5, 1.0, 2.0] {
            for lambda_unit in [0.0, 0.5, 1.0, 2.0] {
                let delta =
                    compute_joint_weights(&y.view(), &d.view(), lambda_time, lambda_unit, 2);

                for t in 0..4 {
                    for i in 0..3 {
                        assert!(
                            delta[[t, i]] >= 0.0,
                            "Weight should be non-negative at [{}, {}] with λ_t={}, λ_u={}",
                            t,
                            i,
                            lambda_time,
                            lambda_unit
                        );
                    }
                }
            }
        }
    }

    // ========================================================================
    // Twostep weight tests
    // ========================================================================

    #[test]
    fn test_twostep_weight_non_negative() {
        // exp(·) ≥ 0 ⟹ all Twostep weights are non-negative.
        let y = array![
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
            [4.0, 5.0, 6.0, 7.0],
            [5.0, 6.0, 7.0, 8.0]
        ];
        let d = array![
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0]
        ];
        let time_dist = array![
            [0i64, 1, 2, 3, 4],
            [1, 0, 1, 2, 3],
            [2, 1, 0, 1, 2],
            [3, 2, 1, 0, 1],
            [4, 3, 2, 1, 0]
        ];

        // Sweep over λ combinations.
        for lambda_time in [0.0, 0.5, 1.0, 2.0] {
            for lambda_unit in [0.0, 0.5, 1.0, 2.0] {
                let weights = compute_weight_matrix(
                    &y.view(),
                    &d.view(),
                    5,
                    4,
                    3,
                    4,
                    lambda_time,
                    lambda_unit,
                    &time_dist.view(),
                );

                for t in 0..5 {
                    for i in 0..4 {
                        assert!(
                            weights[[t, i]] >= 0.0,
                            "Twostep weight should be non-negative at [{}, {}] with λ_t={}, λ_u={}",
                            t,
                            i,
                            lambda_time,
                            lambda_unit
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_twostep_weight_positive() {
        // Total weight sum should be strictly positive for any λ ≥ 0.
        let y = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0]
        ];
        let d = array![
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0]
        ];
        let time_dist = array![[0i64, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]];

        for lambda_time in [0.0, 0.5, 1.0, 2.0] {
            for lambda_unit in [0.0, 0.5, 1.0, 2.0] {
                let weights = compute_weight_matrix(
                    &y.view(),
                    &d.view(),
                    4,
                    3,
                    2,
                    3,
                    lambda_time,
                    lambda_unit,
                    &time_dist.view(),
                );

                let total: f64 = weights.sum();
                assert!(
                    total > 0.0,
                    "Twostep weights should be positive, got {} with λ_t={}, λ_u={}",
                    total,
                    lambda_time,
                    lambda_unit
                );
            }
        }
    }

    #[test]
    fn test_twostep_time_weight_formula() {
        // θ_s = exp(−λ_time · |t − s|), unnormalized.
        let y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]];
        let d = array![[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 1.0]];
        let time_dist = array![
            [0i64, 1, 2, 3, 4],
            [1, 0, 1, 2, 3],
            [2, 1, 0, 1, 2],
            [3, 2, 1, 0, 1],
            [4, 3, 2, 1, 0]
        ];

        let lambda_time = 0.5;
        let target_period = 3;

        // Expected θ_s = exp(−0.5 · |3 − s|).
        let mut expected_time_weights = [0.0f64; 5];
        for (s, weight) in expected_time_weights.iter_mut().enumerate() {
            let dist = (target_period as i64 - s as i64).abs() as f64;
            *weight = (-lambda_time * dist).exp();
        }

        // target_unit = 1 is treated at t = 3 (d[3, 1] = 1). With λ_unit = 0,
        // ω_j = 1 for unit 0 (untreated at target) plus ω_1 = 1 (target unit),
        // so Σω = 2 and row sum = θ_s · 2.
        let weights = compute_weight_matrix(
            &y.view(),
            &d.view(),
            5,
            2,
            1,
            target_period,
            lambda_time,
            0.0,
            &time_dist.view(),
        );

        for (t, &expected_weight) in expected_time_weights.iter().enumerate() {
            let row_sum: f64 = weights.row(t).sum();
            let expected_row_sum = expected_weight * 2.0;
            assert!(
                (row_sum - expected_row_sum).abs() < 1e-10,
                "Time weight mismatch at t={}: expected {}, got {}",
                t,
                expected_row_sum,
                row_sum
            );
        }
    }

    #[test]
    fn test_twostep_lambda_decay_behavior() {
        // Larger λ_time ⟹ faster exponential decay ⟹ higher concentration
        // of weight near the target period.
        let y = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
            [5.0, 6.0, 7.0]
        ];
        let d = array![
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0]
        ];
        let time_dist = array![
            [0i64, 1, 2, 3, 4],
            [1, 0, 1, 2, 3],
            [2, 1, 0, 1, 2],
            [3, 2, 1, 0, 1],
            [4, 3, 2, 1, 0]
        ];

        let target_period = 4;

        // λ_time = 0.1 (slow decay)
        let weights_low = compute_weight_matrix(
            &y.view(),
            &d.view(),
            5,
            3,
            2,
            target_period,
            0.1,
            0.0,
            &time_dist.view(),
        );

        // λ_time = 2.0 (fast decay)
        let weights_high = compute_weight_matrix(
            &y.view(),
            &d.view(),
            5,
            3,
            2,
            target_period,
            2.0,
            0.0,
            &time_dist.view(),
        );

        // The ratio W[target_period, ·] / W[distant_period, ·] should grow with λ.
        let weight_at_target_low: f64 = weights_low.row(target_period).sum();
        let weight_at_target_high: f64 = weights_high.row(target_period).sum();

        let distant_period = 0;
        let weight_at_distant_low: f64 = weights_low.row(distant_period).sum();
        let weight_at_distant_high: f64 = weights_high.row(distant_period).sum();

        let ratio_low = weight_at_target_low / weight_at_distant_low;
        let ratio_high = weight_at_target_high / weight_at_distant_high;

        assert!(
            ratio_high > ratio_low,
            "Larger λ should increase target/distant ratio: low={}, high={}",
            ratio_low,
            ratio_high
        );

        assert!(
            weight_at_distant_high < weight_at_distant_low,
            "Larger λ should reduce weight at distant periods: low={}, high={}",
            weight_at_distant_low,
            weight_at_distant_high
        );
    }

    #[test]
    fn test_twostep_weight_outer_product_structure() {
        // W[t,i] = θ_t · ω_i ⟹ W[t,i]/W[t,j] = W[s,i]/W[s,j].
        let y = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0]
        ];
        let d = array![
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0]
        ];
        let time_dist = array![[0i64, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]];

        let weights = compute_weight_matrix(
            &y.view(),
            &d.view(),
            4,
            3,
            2,
            3,
            0.5,
            0.5,
            &time_dist.view(),
        );

        // Verify outer product: W[t,i] / W[t,j] = W[s,i] / W[s,j]
        for t in 0..4 {
            for s in 0..4 {
                for i in 0..3 {
                    for j in 0..3 {
                        if weights[[t, j]] > 1e-10 && weights[[s, j]] > 1e-10 {
                            let ratio_t = weights[[t, i]] / weights[[t, j]];
                            let ratio_s = weights[[s, i]] / weights[[s, j]];
                            assert!(
                                (ratio_t - ratio_s).abs() < 1e-10,
                                "Outer product structure violated at t={}, s={}, i={}, j={}",
                                t,
                                s,
                                i,
                                j
                            );
                        }
                    }
                }
            }
        }
    }

    /// When a unit has no valid pre-period data, dist = ∞. IEEE 754 gives
    /// −0.0 × ∞ = NaN, so exp(NaN) = NaN. The implementation guards against
    /// this by returning δ_unit = 0 for such units.
    #[test]
    fn test_joint_weights_infinite_distance_guard() {
        // Unit 2: all NaN in pre-periods ⟹ dist = ∞ ⟹ δ_unit = 0.
        let y = array![
            [1.0, 2.0, f64::NAN],
            [2.0, 3.0, f64::NAN],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0]
        ];
        let d = array![
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0]
        ];

        let weights = compute_joint_weights(&y.view(), &d.view(), 0.5, 0.0, 2);

        for t in 0..4 {
            for i in 0..3 {
                assert!(
                    !weights[[t, i]].is_nan(),
                    "NaN weight at [{}, {}] with lambda_unit=0 and NaN unit",
                    t, i
                );
                assert!(
                    weights[[t, i]] >= 0.0,
                    "Weight should be non-negative at [{}, {}]", t, i
                );
            }
        }

        // Unit 2 should have zero weight (no valid pre-period data)
        for t in 0..4 {
            assert!(
                weights[[t, 2]] == 0.0,
                "Unit with no valid pre-period data should have zero weight, got {} at t={}",
                weights[[t, 2]], t
            );
        }

        // Units 0 and 1 should have positive weights (valid pre-period data)
        assert!(weights[[0, 0]] > 0.0, "Unit 0 should have positive weight");
        assert!(weights[[0, 1]] > 0.0, "Unit 1 should have positive weight");
    }

    /// Same guard tested at the vector level: δ_unit[j] = 0 when dist = ∞.
    #[test]
    fn test_joint_weight_vectors_infinite_distance_guard() {
        let y = array![
            [1.0, 2.0, f64::NAN],
            [2.0, 3.0, f64::NAN],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0]
        ];
        let d = array![
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0]
        ];

        let (delta_time, delta_unit) =
            compute_joint_weight_vectors(&y.view(), &d.view(), 0.5, 0.0, 2);

        // delta_unit[2] should be 0.0, not NaN
        assert!(
            !delta_unit[2].is_nan(),
            "delta_unit NaN for unit with no valid pre-period data"
        );
        assert_eq!(
            delta_unit[2], 0.0,
            "Unit with no valid pre-period data should have zero weight"
        );

        // Other units should have positive weights
        assert!(delta_unit[0] > 0.0);
        assert!(delta_unit[1] > 0.0);

        // Time weights should all be finite and positive
        for t in 0..4 {
            assert!(delta_time[t].is_finite() && delta_time[t] > 0.0);
        }
    }

    /// Numerical cross-validation: compare Twostep weight matrix against
    /// independently computed reference values (T=5, N=4, seed=42,
    /// target_unit=2, target_period=3, λ_time=0.5, λ_unit=0.5). Weights are
    /// unnormalized per paper Eq. (3).
    #[test]
    fn test_weight_matrix_reference_values() {
        let y = array![
            [ 0.496714153011233,  0.361735698828815,  1.647688538100692,  3.023029856408026],
            [-0.234153374723336,  0.265863043050819,  2.579212815507391,  2.267434729152909],
            [-0.469474385934952,  1.042560043585965,  0.536582307187538,  1.034270246429743],
            [ 0.241962271566034, -1.413280244657798, -0.724917832513033,  0.937712470759027],
            [-1.012831120334424,  0.814247332595274,  0.091975924478789,  0.087696298664708]
        ];
        let d = array![
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ];
        let time_dist = array![
            [0i64, 1, 2, 3, 4],
            [1, 0, 1, 2, 3],
            [2, 1, 0, 1, 2],
            [3, 2, 1, 0, 1],
            [4, 3, 2, 1, 0]
        ];

        let weights = compute_weight_matrix(
            &y.view(), &d.view(), 5, 4, 2, 3, 0.5, 0.5, &time_dist.view(),
        );

        // Unnormalized reference values (tolerance < 1e-10).
        let reference = [
            [0.088540252840800, 0.102500687323097, 0.223130160148430, 0.144900482487827],
            [0.145978198171795, 0.168995063450973, 0.367879441171442, 0.238900507612393],
            [0.240677360384317, 0.278625755754937, 0.606530659712633, 0.393880348481609],
            [0.396809883441583, 0.459376210078063, 1.000000000000000, 0.649398908652408],
            [0.240677360384317, 0.278625755754937, 0.606530659712633, 0.393880348481609],
        ];

        for t in 0..5 {
            for i in 0..4 {
                assert!(
                    (weights[[t, i]] - reference[t][i]).abs() < 1e-10,
                    "Weight mismatch at [{}, {}]: got={:.15} expected={:.15}",
                    t, i, weights[[t, i]], reference[t][i]
                );
            }
        }
    }

    #[test]
    fn test_weight_numerical_precision() {
        // Unnormalized kernels: W[t, i] = θ_t · ω_i.
        let y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let d = array![[0.0, 0.0], [0.0, 0.0], [0.0, 1.0]];
        let time_dist = array![[0i64, 1, 2], [1, 0, 1], [2, 1, 0]];

        let lambda_time = 1.0;
        let target_period = 2;
        let target_unit = 1;

        // θ_s = exp(−λ_time · |target_period − s|).
        let expected_time = [
            (-lambda_time * 2.0_f64).exp(),
            (-lambda_time * 1.0_f64).exp(),
            (-lambda_time * 0.0_f64).exp(),
        ];

        // λ_unit = 0: unit 0 gets 1 (untreated at target period), target unit 1
        // also gets 1 (target unit convention). Σω = 2 at each row.
        let weights = compute_weight_matrix(
            &y.view(),
            &d.view(),
            3,
            2,
            target_unit,
            target_period,
            lambda_time,
            0.0,
            &time_dist.view(),
        );

        for (t, &expected_t) in expected_time.iter().enumerate() {
            let row_sum: f64 = weights.row(t).sum();
            assert!(
                (row_sum - expected_t * 2.0).abs() < 1e-10,
                "Row sum precision error at t={}: expected {}, got {}",
                t, expected_t * 2.0, row_sum
            );

            for j in 0..2 {
                let expected_w = expected_t * 1.0;
                assert!(
                    (weights[[t, j]] - expected_w).abs() < 1e-10,
                    "Weight precision error at [{}, {}]: expected {}, got {}",
                    t, j, expected_w, weights[[t, j]]
                );
            }
        }
    }

    /// Bit-exactness guard: the buffer-reuse path must reproduce the
    /// allocating path exactly, not just within a tolerance.
    #[test]
    fn test_weight_matrix_into_bit_exact() {
        let y = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0]
        ];
        let d = array![
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0]
        ];
        let time_dist = array![[0i64, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]];

        let expected = compute_weight_matrix(
            &y.view(), &d.view(), 4, 3, 2, 3, 0.7, 0.3, &time_dist.view(),
        );

        let mut buf = Array2::<f64>::from_elem((4, 3), f64::NAN);
        compute_weight_matrix_into(
            &mut buf, &y.view(), &d.view(), 4, 3, 2, 3, 0.7, 0.3, &time_dist.view(),
        );

        assert_eq!(buf, expected, "compute_weight_matrix_into must be bit-exact");
    }

    #[test]
    fn test_weight_matrix_cached_into_bit_exact() {
        use crate::distance::UnitDistanceCache;

        let y = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0]
        ];
        let d = array![
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0]
        ];
        let time_dist = array![[0i64, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]];
        let cache = UnitDistanceCache::build(&y.view(), &d.view());

        let expected = compute_weight_matrix_cached(
            &y.view(), &d.view(), &cache, 4, 3, 2, 3, 0.7, 0.3, &time_dist.view(),
        );

        let mut buf = Array2::<f64>::from_elem((4, 3), f64::NAN);
        compute_weight_matrix_cached_into(
            &mut buf, &y.view(), &d.view(), &cache, 4, 3, 2, 3, 0.7, 0.3, &time_dist.view(),
        );

        assert_eq!(
            buf, expected,
            "compute_weight_matrix_cached_into must be bit-exact"
        );
    }

    #[test]
    fn test_joint_weights_into_bit_exact() {
        let y = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0]
        ];
        let d = array![
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0]
        ];

        let expected = compute_joint_weights(&y.view(), &d.view(), 0.4, 0.6, 2);

        let mut buf = Array2::<f64>::from_elem((4, 3), f64::NAN);
        compute_joint_weights_into(&mut buf, &y.view(), &d.view(), 0.4, 0.6, 2);

        assert_eq!(
            buf, expected,
            "compute_joint_weights_into must be bit-exact"
        );
    }
}

// ============================================================================
// Adaptive sparse weight dedicated tests
// ============================================================================

#[cfg(test)]
mod sparse_tests {
    use super::*;
    use ndarray::array;

    /// Build a large panel designed to trigger the sparse path: N >= SPARSE_MIN_UNITS.
    /// Uses large lambda (fast decay) to ensure many weights are near zero.
    fn build_large_panel_views(
        n_periods: usize,
        n_units: usize,
    ) -> (ndarray::Array2<f64>, ndarray::Array2<f64>, ndarray::Array2<i64>) {
        // Y is a simple linear panel, all control observations
        let y = ndarray::Array2::from_shape_fn((n_periods, n_units), |(t, i)| {
            (t as f64) * 0.1 + (i as f64) * 0.01
        });
        let d = ndarray::Array2::<f64>::zeros((n_periods, n_units));
        let time_dist = ndarray::Array2::from_shape_fn((n_periods, n_periods), |(t, s)| {
            (t as i64 - s as i64).abs()
        });
        (y, d, time_dist)
    }

    /// With small lambda, weights are nearly uniform; adaptive path should return Dense.
    #[test]
    fn test_adaptive_small_lambda_returns_dense() {
        let n_periods = 5;
        let n_units = 3; // less than SPARSE_MIN_UNITS, always returns Dense
        let (y, d, time_dist) = build_large_panel_views(n_periods, n_units);

        let result = compute_weight_matrix_adaptive(
            &y.view(), &d.view(),
            n_periods, n_units, 0, 1,
            0.0, 0.0, // lambda = 0 -> uniform weights
            &time_dist.view(),
        );

        assert!(
            matches!(result, WeightMatrix::Dense(_)),
            "small panel (N < SPARSE_MIN_UNITS) should always return Dense"
        );
    }

    /// Small panel (N < SPARSE_MIN_UNITS) should always return Dense regardless of lambda.
    #[test]
    fn test_adaptive_small_panel_always_dense() {
        let n_periods = 5;
        let n_units = 3; // less than 100
        let (y, d, time_dist) = build_large_panel_views(n_periods, n_units);

        for &lambda in &[0.0_f64, 1.0, 10.0, 100.0] {
            let result = compute_weight_matrix_adaptive(
                &y.view(), &d.view(),
                n_periods, n_units, 0, 2,
                lambda, lambda,
                &time_dist.view(),
            );
            assert!(
                matches!(result, WeightMatrix::Dense(_)),
                "small panel (N={}) should always return Dense, lambda={}",
                n_units, lambda
            );
        }
    }

    /// Dense and sparse paths both produce numerically identical results to [`compute_weight_matrix`].
    ///
    /// For small panels (N < SPARSE_MIN_UNITS), verify Dense path numerical equivalence.
    #[test]
    fn test_adaptive_dense_path_matches_original() {
        let y = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0]
        ];
        let d = array![
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0]
        ];
        let time_dist = array![[0i64, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]];

        let expected = compute_weight_matrix(
            &y.view(), &d.view(), 4, 3, 2, 3, 0.5, 0.5, &time_dist.view(),
        );

        let result = compute_weight_matrix_adaptive(
            &y.view(), &d.view(), 4, 3, 2, 3, 0.5, 0.5, &time_dist.view(),
        );

        match result {
            WeightMatrix::Dense(mat) => {
                for t in 0..4 {
                    for i in 0..3 {
                        assert!(
                            (mat[[t, i]] - expected[[t, i]]).abs() < 1e-10,
                            "Dense path mismatch at [{}, {}]: {} vs {}",
                            t, i, mat[[t, i]], expected[[t, i]]
                        );
                    }
                }
            }
            WeightMatrix::Sparse(sparse) => {
                // Small panel must be Dense; this branch should not be reached
                panic!(
                    "small panel (N=3 < SPARSE_MIN_UNITS) should not return Sparse, but nnz={}",
                    sparse.indices.len()
                );
            }
        }
    }

    /// Adaptive sparse path numerical equivalence: expanded SparseWeights should
    /// match dense path content.
    ///
    /// Constructs a large panel with N=110 (>= SPARSE_MIN_UNITS), uses large lambda
    /// to reduce non-zero element ratio, verifies sparse expansion matches dense
    /// result (tolerance < 1e-10).
    #[test]
    fn test_adaptive_sparse_dense_numerical_equivalence() {
        let n_periods = 8;
        let n_units = 110; // >= SPARSE_MIN_UNITS = 100
        let target_unit = 5;
        let target_period = 6;

        let y = ndarray::Array2::from_shape_fn((n_periods, n_units), |(t, i)| {
            ((t * 7 + i * 13) as f64) * 0.01
        });
        let d = ndarray::Array2::<f64>::zeros((n_periods, n_units));
        let time_dist = ndarray::Array2::from_shape_fn((n_periods, n_periods), |(t, s)| {
            (t as i64 - s as i64).abs()
        });

        // Use large lambda to produce a sparse matrix
        let lambda_time = 5.0;
        let lambda_unit = 5.0;

        let dense_ref = compute_weight_matrix(
            &y.view(), &d.view(),
            n_periods, n_units, target_unit, target_period,
            lambda_time, lambda_unit, &time_dist.view(),
        );

        let adaptive = compute_weight_matrix_adaptive(
            &y.view(), &d.view(),
            n_periods, n_units, target_unit, target_period,
            lambda_time, lambda_unit, &time_dist.view(),
        );

        // Expand adaptive result to dense matrix for comparison, recording truncation threshold
        let (adaptive_dense, sparse_threshold) = match adaptive {
            WeightMatrix::Dense(m) => (m, 0.0),
            WeightMatrix::Sparse(ref sp) => {
                let mut m = ndarray::Array2::<f64>::zeros((sp.n_rows, sp.n_cols));
                for (&(r, c), &v) in sp.indices.iter().zip(sp.values.iter()) {
                    m[[r, c]] = v;
                }
                (m, sp.threshold)
            }
        };

        for t in 0..n_periods {
            for i in 0..n_units {
                let diff = (adaptive_dense[[t, i]] - dense_ref[[t, i]]).abs();
                let ref_val = dense_ref[[t, i]];
                if ref_val >= sparse_threshold {
                    // Non-truncated elements must be exactly equivalent (tolerance < 1e-10)
                    assert!(
                        diff < 1e-10,
                        "sparse expansion mismatch at [{}, {}]: adaptive={}, ref={}",
                        t, i, adaptive_dense[[t, i]], ref_val
                    );
                } else {
                    // Truncated tiny elements: diff is at most ref_val itself
                    // (sparse stores 0, dense stores a tiny but valid value)
                    assert!(
                        diff <= ref_val + 1e-15,
                        "sparse expansion truncation residual too large at [{}, {}]: diff={}, ref={}",
                        t, i, diff, ref_val
                    );
                }
            }
        }
    }

    /// SparseWeights metadata consistency check.
    #[test]
    fn test_sparse_weights_metadata_consistency() {
        let n_periods = 8;
        let n_units = 110;
        let (y, d, time_dist) = build_large_panel_views(n_periods, n_units);

        let result = compute_weight_matrix_adaptive(
            &y.view(), &d.view(),
            n_periods, n_units, 0, 4,
            5.0, 5.0, // large lambda to trigger sparse path
            &time_dist.view(),
        );

        match result {
            WeightMatrix::Dense(_) => {
                // With small lambda or non-zero ratio > 80% this will be Dense, not an error
            }
            WeightMatrix::Sparse(sp) => {
                // indices and values must be same length
                assert_eq!(
                    sp.indices.len(), sp.values.len(),
                    "SparseWeights: indices and values must have equal length"
                );
                // n_rows, n_cols must match original dimensions
                assert_eq!(sp.n_rows, n_periods);
                assert_eq!(sp.n_cols, n_units);
                // All stored weights must be >= threshold
                for &v in &sp.values {
                    assert!(
                        v >= sp.threshold,
                        "stored weight {} should be >= threshold {}",
                        v, sp.threshold
                    );
                }
                // All weights non-negative
                for &v in &sp.values {
                    assert!(v >= 0.0, "sparse weight should be non-negative: {}", v);
                }
                // Indices within bounds
                for &(r, c) in &sp.indices {
                    assert!(r < n_periods, "row index out of bounds: r={}", r);
                    assert!(c < n_units, "col index out of bounds: c={}", c);
                }
            }
        }
    }

    /// WeightMatrix::Dense and Sparse paths should both produce the same ATT estimate.
    ///
    /// Verifies numerical equivalence between sparse and dense paths via
    /// [`estimate_model_adaptive`], tolerance < 1e-10.
    #[test]
    fn test_estimate_model_sparse_dense_att_equivalence() {
        use crate::estimation::estimate_model;

        let y = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0]
        ];
        let d = array![
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0]
        ];
        // control_mask: 1 for control cells (D=0)
        let control_mask = ndarray::Array2::<u8>::from_shape_fn((4, 3), |(t, i)| {
            if d[[t, i]] == 0.0 { 1 } else { 0 }
        });
        let time_dist = array![[0i64, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]];

        // Dense weight matrix
        let w_dense = compute_weight_matrix(
            &y.view(), &d.view(), 4, 3, 2, 3, 0.5, 0.5, &time_dist.view(),
        );

        let (alpha_d, beta_d, l_d, _, _, _) = estimate_model(
            &y.view(), &control_mask.view(), &w_dense.view(),
            0.1, 4, 3, 50, 1e-8, None, None, None, None,
        ).expect("dense estimation should succeed");

        // Adaptive path (small panel must be Dense)
        let w_adaptive = compute_weight_matrix_adaptive(
            &y.view(), &d.view(), 4, 3, 2, 3, 0.5, 0.5, &time_dist.view(),
        );

        let (alpha_a, beta_a, l_a, _, _, _) = crate::estimation::estimate_model_adaptive(
            &y.view(), &control_mask.view(), &w_adaptive,
            0.1, 4, 3, 50, 1e-8, None, None, None, None,
        ).expect("adaptive estimation should succeed");

        // Compute ATT for both
        let att_dense: f64 = (0..4).flat_map(|t| (0..3).map(move |i| (t, i)))
            .filter(|&(t, i)| d[[t, i]] == 1.0)
            .map(|(t, i)| y[[t, i]] - alpha_d[i] - beta_d[t] - l_d[[t, i]])
            .sum::<f64>() / 2.0;

        let att_adaptive: f64 = (0..4).flat_map(|t| (0..3).map(move |i| (t, i)))
            .filter(|&(t, i)| d[[t, i]] == 1.0)
            .map(|(t, i)| y[[t, i]] - alpha_a[i] - beta_a[t] - l_a[[t, i]])
            .sum::<f64>() / 2.0;

        assert!(
            (att_dense - att_adaptive).abs() < 1e-10,
            "dense and adaptive ATT should be equal: dense={}, adaptive={}",
            att_dense, att_adaptive
        );
    }
}

/// Property-based tests using proptest.
#[cfg(test)]
mod proptests {
    use super::*;
    use ndarray::Array2;
    use proptest::prelude::*;

    /// Generate a random outcome matrix Y of dimension (T × N).
    fn y_matrix_strategy(
        n_periods: usize,
        n_units: usize,
    ) -> impl Strategy<Value = Array2<f64>> {
        prop::collection::vec(-100.0..100.0_f64, n_periods * n_units).prop_map(move |v| {
            Array2::from_shape_vec((n_periods, n_units), v).unwrap()
        })
    }

    /// Treatment matrix with a single treated cell at (T−1, 0).
    fn d_single_treated(n_periods: usize, n_units: usize) -> Array2<f64> {
        let mut d = Array2::<f64>::zeros((n_periods, n_units));
        d[[n_periods - 1, 0]] = 1.0;
        d
    }

    /// Absolute-difference time distance matrix: |t − s|.
    fn time_dist_matrix(n_periods: usize) -> Array2<i64> {
        Array2::from_shape_fn((n_periods, n_periods), |(t, s)| (t as i64 - s as i64).abs())
    }

    proptest! {
        /// All weights are non-negative (exp(·) ≥ 0).
        #[test]
        fn prop_weights_non_negative(
            y in y_matrix_strategy(6, 4),
            lambda_time in 0.0..5.0_f64,
            lambda_unit in 0.0..5.0_f64,
            target_unit in 0..4_usize,
        ) {
            let d = d_single_treated(6, 4);
            let td = time_dist_matrix(6);
            let target_period = 5; // last period (treated)

            let w = compute_weight_matrix(
                &y.view(), &d.view(), 6, 4,
                target_unit, target_period,
                lambda_time, lambda_unit, &td.view(),
            );

            for t in 0..6 {
                for j in 0..4 {
                    prop_assert!(w[[t, j]] >= 0.0,
                        "Weight [{}, {}] = {} should be >= 0", t, j, w[[t, j]]);
                }
            }
        }

        /// Rank-1 (outer product) structure: W[t1,j]/W[t2,j] is constant
        /// across all j with non-zero weight.
        #[test]
        fn prop_weights_outer_product_structure(
            y in y_matrix_strategy(6, 4),
            lambda_time in 0.01..3.0_f64,
            lambda_unit in 0.01..3.0_f64,
        ) {
            let d = d_single_treated(6, 4);
            let td = time_dist_matrix(6);
            let target_unit = 0;
            let target_period = 5;

            let w = compute_weight_matrix(
                &y.view(), &d.view(), 6, 4,
                target_unit, target_period,
                lambda_time, lambda_unit, &td.view(),
            );

            let mut ratio: Option<f64> = None;
            for j in 0..4 {
                if w[[0, j]] > 1e-15 && w[[1, j]] > 1e-15 {
                    let r = w[[0, j]] / w[[1, j]];
                    if let Some(prev_r) = ratio {
                        prop_assert!((r - prev_r).abs() < 1e-10,
                            "Rank-1 structure violated: ratio {} vs {}", r, prev_r);
                    }
                    ratio = Some(r);
                }
            }
        }

        /// Exponential decay: θ_{s1}/θ_{s2} = exp(λ_time · (|t−s2| − |t−s1|)).
        #[test]
        fn prop_time_weight_exponential_decay(
            lambda_time in 0.01..5.0_f64,
        ) {
            let y = Array2::from_shape_fn((5, 3), |(t, i)| (t as f64) + (i as f64) * 0.1);
            let d = Array2::<f64>::zeros((5, 3));
            let td = time_dist_matrix(5);
            let target_unit = 0;
            let target_period = 4;

            let w = compute_weight_matrix(
                &y.view(), &d.view(), 5, 3,
                target_unit, target_period,
                lambda_time, 0.0, &td.view(),
            );

            // With λ_unit = 0, ω_j = 1 ⟹ W[t, j] = θ_t.
            // Ratio W[3,j]/W[2,j] = exp(−λ·1)/exp(−λ·2) = exp(λ).
            for j in 1..3 {
                if w[[3, j]] > 1e-15 && w[[2, j]] > 1e-15 {
                    let ratio = w[[3, j]] / w[[2, j]];
                    let expected_ratio = lambda_time.exp();
                    prop_assert!((ratio - expected_ratio).abs() < 1e-8,
                        "Time decay ratio {} vs expected {}", ratio, expected_ratio);
                }
            }
        }

        /// W[target_period, target_unit] = θ_t(0) · ω_i(0) = 1.0 · 1.0 = 1.0
        /// (target unit is always assigned unit kernel 1; time kernel at
        /// distance 0 is also 1).
        #[test]
        fn prop_target_unit_weight(
            y in y_matrix_strategy(5, 3),
            lambda_time in 0.0..3.0_f64,
            lambda_unit in 0.0..3.0_f64,
        ) {
            let d = Array2::<f64>::zeros((5, 3));
            let td = time_dist_matrix(5);
            let target_unit = 1;
            let target_period = 4;

            let w = compute_weight_matrix(
                &y.view(), &d.view(), 5, 3,
                target_unit, target_period,
                lambda_time, lambda_unit, &td.view(),
            );

            prop_assert!((w[[target_period, target_unit]] - 1.0).abs() < 1e-10,
                "W[target_period, target_unit] should be 1.0, got {}",
                w[[target_period, target_unit]]);
        }
    }
}

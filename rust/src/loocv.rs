//! Leave-one-out cross-validation (LOOCV) for tuning parameter selection.
//!
//! Selects the regularization triplet (λ_time, λ_unit, λ_nn) by minimizing
//! the LOOCV criterion:
//!
//!   Q(λ) = Σ_{i,t} (1 − W_{it}) (τ̂_{it}^{loocv}(λ))²
//!
//! where τ̂_{it}^{loocv}(λ) is the pseudo-treatment effect obtained by treating
//! each control observation (i,t) as if it were treated and estimating its
//! counterfactual from the remaining control observations.
//!
//! Two search strategies are available for each of the twostep and joint
//! estimation methods:
//!   Cycling (coordinate descent) — Stage 1 univariate initialization then
//!     Stage 2 cyclic updates until convergence; complexity
//!     O(|grid| · max_cycles).  Default for twostep.
//!   Exhaustive (Cartesian)       — evaluates every (λ_time, λ_unit, λ_nn)
//!     triple in parallel; complexity O(|grid|³).  Default for joint.
//!     Use when the grid is small enough to afford the cubic cost and the
//!     Q(λ) surface may be non-convex (e.g. the paper's Basque / West
//!     Germany examples, where coordinate descent can stall at local
//!     minima).
//!
//! All four search paths ultimately compare candidate triples via
//! `better_candidate`, which combines the primary score comparison with a
//! deterministic tie-breaker (see the function documentation for details).

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use ndarray::{Array1, Array2, ArrayView2};
use rayon::prelude::*;

use crate::distance::UnitDistanceCache;

/// Key type for the cycling score cache: f64 triple encoded as bit patterns.
/// Using `to_bits()` ensures exact-match semantics (no floating-point comparison issues).
type CyclingCacheKey = (u64, u64, u64);

/// Thread-safe cache for LOOCV scores evaluated during coordinate descent cycling.
/// Maps (λ_time_bits, λ_unit_bits, λ_nn_bits) → (score, n_valid, first_failed).
type CyclingScoreCache = Mutex<HashMap<CyclingCacheKey, (f64, usize, Option<(usize, usize)>)>>;
use crate::error::{TropError, TropResult};
use crate::estimation::{
    debug_assert_delta_is_1minus_d_masked, estimate_model, solve_joint_no_lowrank,
    solve_joint_with_lowrank,
};
use crate::weights::{
    compute_joint_weights_into, compute_weight_matrix_cached_into,
};

/// Numerical tolerance inside `better_candidate` for declaring two LOOCV
/// scores equivalent.  Scores within `TIE_TOL` of each other are treated as a
/// tie and resolved by the structural tie-breaker (prefer larger λ_nn, then
/// smaller λ_time, then smaller λ_unit).
///
/// Rationale: BLAS/LAPACK implementations (Accelerate, OpenBLAS, MKL) can
/// produce Q(λ) values that differ by a few ULPs on the same data.  Using a
/// strict `<` comparison would let that jitter swing the selected λ across
/// platforms.  `TIE_TOL = 1e-10` is tighter than the lowest non-trivial score
/// magnitudes encountered in the CPS/PWT/Basque/Germany benchmarks (~1e-3)
/// yet comfortably above the ~1e-14 round-off floor, so the tie-breaker only
/// fires on genuine numerical ties.
pub const TIE_TOL: f64 = 1e-10;

/// Return `true` iff the candidate triple `new` should replace the incumbent
/// `best` under the LOOCV selection rule.
///
/// Comparison order:
///   1. Score strictly lower by more than `TIE_TOL`: prefer `new`.
///   2. Score strictly higher by more than `TIE_TOL`: keep `best`.
///   3. Otherwise (tie within `TIE_TOL`), break the tie by the Occam's razor
///      policy:
///      a. Prefer larger λ_nn (stronger nuclear-norm penalty → simpler L).
///      b. If equal, prefer smaller λ_time (more uniform time weights).
///      c. If equal, prefer smaller λ_unit (more uniform unit weights).
///
/// Each tuple component is ordered using `f64::total_cmp` so that `+Inf` and
/// finite values order correctly and `NaN` is handled deterministically
/// (NaNs sort after finite values).
///
/// This policy mirrors the paper's intuition (Section 4.3, Table 5): when
/// two tunings explain the data equally well, the one with more
/// regularization and more uniform weights is the more defensible choice
/// and is easier to reproduce across BLAS backends.
#[inline]
pub fn better_candidate(
    new: (f64, f64, f64, f64),
    best: (f64, f64, f64, f64),
) -> bool {
    use std::cmp::Ordering;

    let (_, _, _, s_new) = new;
    let (_, _, _, s_best) = best;

    // If incumbent is not finite but candidate is, the candidate always wins.
    // This mirrors the prior `< f64::INFINITY` semantics without relying on
    // the quirks of IEEE arithmetic with `f64::INFINITY` on both sides.
    match (s_new.is_finite(), s_best.is_finite()) {
        (true, false) => return true,
        (false, true) => return false,
        (false, false) => return false, // both non-finite: keep incumbent
        (true, true) => {}               // fall through to comparison
    }

    let diff = s_new - s_best;
    if diff < -TIE_TOL {
        return true;
    }
    if diff > TIE_TOL {
        return false;
    }

    // Tie on score → structural tie-breaker.
    let (lt_new, lu_new, ln_new, _) = new;
    let (lt_best, lu_best, ln_best, _) = best;

    // (a) Prefer larger λ_nn.
    match ln_new.total_cmp(&ln_best) {
        Ordering::Greater => return true,
        Ordering::Less => return false,
        Ordering::Equal => {}
    }
    // (b) Prefer smaller λ_time.
    match lt_new.total_cmp(&lt_best) {
        Ordering::Less => return true,
        Ordering::Greater => return false,
        Ordering::Equal => {}
    }
    // (c) Prefer smaller λ_unit.
    matches!(lu_new.total_cmp(&lu_best), Ordering::Less)
}

/// Type alias for the LOOCV grid search result tuple.
///
/// Fields: (best_lambda_time, best_lambda_unit, best_lambda_nn, best_score,
///          n_valid, n_attempted, first_failed_obs).
///
/// `n_attempted` equals the total number of finite control observations
/// because the LOOCV criterion (paper Eq. 5) sums over every D=0 cell.
#[allow(clippy::type_complexity)]
pub type LoocvGridSearchResult = (
    f64,
    f64,
    f64,
    f64,
    usize,
    usize,
    Option<(usize, usize)>,
);

/// Type alias for internal LOOCV result tuple.
#[allow(clippy::type_complexity)]
type LoocvResultTuple = (f64, f64, f64, f64, usize, Option<(usize, usize)>);

/// Extended LOOCV grid search result that also surfaces the Stage-1
/// univariate initialisation triple (paper Footnote 2).
///
/// Fields: `(result, stage1_lambda_time, stage1_lambda_unit,
///          stage1_lambda_nn)` where `result` is a standard
/// [`LoocvGridSearchResult`].
///
/// The Stage-1 triple is the argmin of three univariate sweeps (each
/// fixing the other two tuning parameters at Footnote-2 extrema).  When
/// the search strategy is exhaustive (no coordinate-descent polish), the
/// concept does not apply and the caller should use
/// [`LoocvGridSearchResult`] directly.
#[allow(clippy::type_complexity)]
pub type LoocvGridSearchResultWithStage1 = (LoocvGridSearchResult, f64, f64, f64);

/// Collect control observations eligible for LOOCV evaluation.
///
/// Returns every (period, unit) pair where `control_mask` is nonzero and the
/// outcome is finite.  Per the LOOCV criterion (paper Eq. 5), Q(λ) is the
/// sum of squared pseudo-treatment effects across **all** D=0 cells, so this
/// function never subsamples.
///
/// # Arguments
/// * `y`            — Outcome matrix, n_periods × n_units.
/// * `control_mask` — Nonzero entries mark control observations.
pub fn get_control_observations(
    y: &ArrayView2<f64>,
    control_mask: &ArrayView2<u8>,
) -> Vec<(usize, usize)> {
    let n_periods = y.nrows();
    let n_units = y.ncols();

    let mut obs: Vec<(usize, usize)> = Vec::new();
    for t in 0..n_periods {
        for i in 0..n_units {
            if control_mask[[t, i]] != 0 && y[[t, i]].is_finite() {
                obs.push((t, i));
            }
        }
    }

    obs
}

/// Verify that the treatment matrix `D` satisfies the simultaneous-adoption
/// invariant assumed by the joint method (paper Remark 6.1) and, on success,
/// return the count of post-treatment periods shared by every treated unit.
///
/// The joint estimator collapses the per-observation weights to a single
/// global weight matrix δ that depends only on the shared post-period count;
/// consequently it is *only* well-defined when every treated unit enters
/// treatment at the same period `T_1` and remains treated through the end of
/// the panel.  The Stata front-end (`ado/trop.ado`) already refuses staggered
/// adoption for `method(joint)`, so this check is a defence-in-depth: if
/// the Rust entry is ever reached with a non-conforming `D` (e.g. from a
/// direct Mata plugin call or a regression harness), we fail fast with
/// `TropError::InvalidDimension` instead of silently mis-reporting
/// `treated_periods` from the earliest adoption period.
///
/// # Returns
/// * `Ok(treated_periods)` — number of post-treatment periods shared by all
///   treated units.  Equal to zero when no unit is ever treated.
/// * `Err(TropError::InvalidDimension)` — `D` has at least one treated unit
///   whose adoption period differs from the group-level `T_1`, or whose
///   treatment status is not absorbing.
pub fn check_simultaneous_adoption(d: &ArrayView2<f64>) -> TropResult<usize> {
    let n_periods = d.nrows();
    let n_units = d.ncols();

    // Group-level first-treat period: smallest t with any D[t,i]==1.
    let mut t1: Option<usize> = None;
    for t in 0..n_periods {
        let mut row_treated = false;
        for i in 0..n_units {
            if d[[t, i]] == 1.0 {
                row_treated = true;
                break;
            }
        }
        if row_treated {
            t1 = Some(t);
            break;
        }
    }

    let t1 = match t1 {
        None => return Ok(0), // no treated observations
        Some(t) => t,
    };

    // For every unit that is ever treated, D[t,i] must equal 1 for t >= t1
    // and 0 for t < t1.  Anything else (late adoption, treatment switching
    // off, pre-t1 treatment) violates the invariant.
    for i in 0..n_units {
        let mut ever_treated = false;
        for t in 0..n_periods {
            if d[[t, i]] == 1.0 {
                ever_treated = true;
                break;
            }
        }
        if !ever_treated {
            continue;
        }
        for t in 0..n_periods {
            let expected = if t >= t1 { 1.0 } else { 0.0 };
            if d[[t, i]] != expected {
                return Err(TropError::InvalidDimension);
            }
        }
    }

    Ok(n_periods.saturating_sub(t1))
}

/// Validates that the treatment matrix contains at least one treated observation.
///
/// Scans the `D` matrix for any cell equal to 1.0.  Returns `Ok(())` when at
/// least one treated cell exists, or `Err(TropError::NoTreated)` if the panel
/// has no treatment at all.
///
/// This guard prevents silent no-op estimation when the user inadvertently
/// passes an all-zero treatment indicator (e.g., wrong variable name or
/// subsetting that drops all treated units).
pub fn validate_has_treated_units(d: &ArrayView2<f64>) -> Result<(), TropError> {
    for t in 0..d.nrows() {
        for i in 0..d.ncols() {
            if d[[t, i]] == 1.0 {
                return Ok(());
            }
        }
    }
    Err(TropError::NoTreated)
}

/// Evaluate the LOOCV criterion for a single (λ_time, λ_unit, λ_nn) triple
/// under the twostep method (paper Algorithm 2 step 2).
///
/// For each control observation (t, i), computes the pseudo-treatment
/// effect τ̂_{ti}^{loocv}(λ) by fitting Eq. (4) with observation (t, i)
/// excluded (equivalent to the "include target with τ column" form of
/// Eq. (4): both reparametrisations yield the same τ̂), and returns
///
///   Q(λ) = Σ_{(t, i) ∈ control_obs} τ̂²_{ti}
///
/// which is the paper's Eq. (5) criterion restricted to control cells with
/// finite Y (cells with missing Y are filtered at
/// `get_control_observations` intake).
///
/// Returns `(score, n_valid, first_failed_obs)`.  If any observation fails
/// to produce a valid estimate, the score is set to +∞ and the failing
/// (period, unit) pair is reported.
#[allow(clippy::too_many_arguments)]
pub fn loocv_score_for_params(
    y: &ArrayView2<f64>,
    d: &ArrayView2<f64>,
    control_mask: &ArrayView2<u8>,
    time_dist: &ArrayView2<i64>,
    dist_cache: &UnitDistanceCache,
    control_obs: &[(usize, usize)],
    lambda_time: f64,
    lambda_unit: f64,
    lambda_nn: f64,
    max_iter: usize,
    tol: f64,
    best_score_so_far: f64,
    x: Option<&ArrayView2<f64>>,
) -> (f64, usize, Option<(usize, usize)>) {
    let n_periods = y.nrows();
    let n_units = y.ncols();

    let mut tau_sq_sum = 0.0;
    let mut n_valid = 0usize;

    // Warm start: reuse previous observation's solution as initial values
    // for the next. Under the same lambda combination, alpha/beta (fixed
    // effects) are similar across control observations, so this reduces
    // the number of outer iterations needed for convergence.
    let mut warm_alpha: Option<Array1<f64>> = None;
    let mut warm_beta: Option<Array1<f64>> = None;
    let mut warm_l: Option<Array2<f64>> = None;

    // Reuse a single weight buffer across all control observations to avoid
    // repeated (T × N) heap allocations inside the LOOCV hot loop.
    let mut weight_buf = Array2::<f64>::zeros((n_periods, n_units));

    for &(t, i) in control_obs {
        // Compute observation-specific weight matrix (using the cache so
        // we avoid repeating the O(T) pairwise distance computation for
        // every control observation).
        compute_weight_matrix_cached_into(
            &mut weight_buf,
            y,
            d,
            dist_cache,
            n_periods,
            n_units,
            i,
            t,
            lambda_time,
            lambda_unit,
            time_dist,
        );

        // Build warm start reference from previous successful fit.
        let ws = match (&warm_alpha, &warm_beta, &warm_l) {
            (Some(a), Some(b), Some(l_prev)) => Some((a, b, l_prev)),
            _ => None,
        };

        // Estimate model excluding this observation.
        match estimate_model(
            y,
            control_mask,
            &weight_buf.view(),
            lambda_nn,
            n_periods,
            n_units,
            max_iter,
            tol,
            Some((t, i)),
            ws,
            x,
            None,
        ) {
            Some((alpha, beta, l, _n_iters, _converged, gamma)) => {
                // Pseudo-treatment effect: τ̂ = Y_{ti} − α_i − β_t − L_{ti} − X'γ.
                let mut tau = y[[t, i]] - alpha[i] - beta[t] - l[[t, i]];
                if let Some(x_mat) = x {
                    if let Some(ref g) = gamma {
                        let idx = t * n_units + i;
                        tau -= x_mat.row(idx).dot(g);
                    }
                }
                tau_sq_sum += tau * tau;
                n_valid += 1;

                // Stash solution for warm-starting the next observation.
                warm_alpha = Some(alpha);
                warm_beta = Some(beta);
                warm_l = Some(l);

                // Early termination: Q(λ) = Σ τ̂² is a sum of non-negative
                // terms, so the partial sum is monotonically non-decreasing.
                // Once it exceeds the current best score, no subsequent terms
                // can bring it below best — safe to abandon this λ.
                if tau_sq_sum > best_score_so_far {
                    return (tau_sq_sum, n_valid, None);
                }
            }
            None => {
                // Estimation failure invalidates this λ combination.
                return (f64::INFINITY, n_valid, Some((t, i)));
            }
        }
    }

    if n_valid == 0 {
        (f64::INFINITY, 0, None)
    } else {
        (tau_sq_sum, n_valid, None)
    }
}

/// Lock-free, shared best-score tracker for dynamic early-termination pruning
/// during candidate-parallel LOOCV search (Task 27, item 2).
///
/// Stores an `f64` via its IEEE-754 bit pattern (`to_bits`/`from_bits`) inside
/// an `AtomicU64`, and is updated with a compare-and-swap loop implementing
/// `fetch_min` semantics (the stored value only ever decreases).
///
/// ## Correctness — numerical identity guarantee
/// The tracked value is used **only** as an early-termination pruning bound,
/// exposed through [`AtomicBestScore::pruning_bound`] which adds a `TIE_TOL`
/// margin.  Concretely, a candidate is abandoned only when its *partial*
/// `Q = Σ τ̂²` already exceeds `best + TIE_TOL`.  Because `Q` is a sum of
/// non-negative terms, that candidate's *full* `Q` also exceeds
/// `best + TIE_TOL`; hence `better_candidate` rejects it in BOTH the pruned and
/// the unpruned computation — the score is strictly worse than the incumbent by
/// more than `TIE_TOL`, so it can never tie and can never be selected.
///
/// Consequently:
///   * the eventual argmin is **never** pruned (its partial sums stay ≤ its
///     full `Q` ≤ every observed complete score, so the `> bound` test never
///     fires), and its `Q` is therefore always computed in full;
///   * the selected `(λ*, score)` — and every downstream quantity derived from
///     it (`att`, `loocv_score`, …) — is identical to the snapshot / no-prune
///     version.
/// Only the *timing* of pruning is non-deterministic across runs, which affects
/// performance but never the result.  `Relaxed` ordering therefore suffices:
/// the value is a heuristic bound, not a synchronisation signal, and no other
/// memory is published through it.
struct AtomicBestScore(AtomicU64);

impl AtomicBestScore {
    #[inline]
    fn new(initial: f64) -> Self {
        AtomicBestScore(AtomicU64::new(initial.to_bits()))
    }

    /// Current best complete score observed so far.
    #[inline]
    fn get(&self) -> f64 {
        f64::from_bits(self.0.load(Ordering::Relaxed))
    }

    /// Early-termination bound: best score plus a `TIE_TOL` margin.  Pruning a
    /// candidate whose partial `Q` exceeds this bound is provably outcome-
    /// preserving (see the type-level docs).
    #[inline]
    fn pruning_bound(&self) -> f64 {
        self.get() + TIE_TOL
    }

    /// `fetch_min`: atomically lower the stored value to `candidate` if (and
    /// only if) `candidate` is strictly smaller than the current value.  A
    /// `NaN` candidate is ignored (the `<` test is always false for `NaN`).
    #[inline]
    fn observe(&self, candidate: f64) {
        let mut cur = self.0.load(Ordering::Relaxed);
        loop {
            if !(candidate < f64::from_bits(cur)) {
                return;
            }
            match self.0.compare_exchange_weak(
                cur,
                candidate.to_bits(),
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return,
                Err(observed) => cur = observed,
            }
        }
    }
}

/// Parallel-over-control-observations variant of [`loocv_score_for_params`]
/// (Task 27, item 1).
///
/// Computes the full LOOCV criterion `Q(λ) = Σ_{(t,i)} τ̂²_{ti}` by splitting
/// `control_obs` into contiguous chunks and evaluating each chunk on a
/// separate rayon worker.  This is used **only** on the small-grid branch of
/// the univariate searches, where the number of candidates is smaller than the
/// thread count and parallelising across candidates would leave workers idle.
/// Exactly one parallel layer is ever active (candidates are iterated serially
/// in that branch), so the thread pool is never oversubscribed and the
/// `CyclingScoreCache` mutex access pattern does not degrade.
///
/// ## Semantics preserved w.r.t. the serial version
///   * **Warm start** — each chunk keeps its own warm-start chain; the first
///     cell of every chunk starts cold.  Warm starts only influence the
///     convergence *path*, not the converged fixed point, so once each fit
///     reaches `tol` the per-cell τ̂ agree with the serial chain to within the
///     convergence tolerance (drift ≪ 1e-10).
///   * **Failure** — if any fit in any chunk fails, the whole candidate's
///     `Q = +∞`, and the reported failing `(t, i)` is the earliest one in
///     control-observation traversal order (chunks are contiguous and returned
///     in order, so the first chunk carrying a failure yields the earliest
///     cell).
///   * **Summation order** — partial sums are collected into a `Vec` in fixed
///     chunk order and summed sequentially, independent of thread completion
///     order, minimising floating-point summation-order variance.
///
/// No cross-candidate early termination is applied here: the grid is tiny, so
/// computing every `Q` in full is cheap and maximises determinism.
#[allow(clippy::too_many_arguments)]
fn loocv_score_for_params_parallel(
    y: &ArrayView2<f64>,
    d: &ArrayView2<f64>,
    control_mask: &ArrayView2<u8>,
    time_dist: &ArrayView2<i64>,
    dist_cache: &UnitDistanceCache,
    control_obs: &[(usize, usize)],
    lambda_time: f64,
    lambda_unit: f64,
    lambda_nn: f64,
    max_iter: usize,
    tol: f64,
    x: Option<&ArrayView2<f64>>,
) -> (f64, usize, Option<(usize, usize)>) {
    let n_periods = y.nrows();
    let n_units = y.ncols();
    let n_obs = control_obs.len();
    if n_obs == 0 {
        return (f64::INFINITY, 0, None);
    }

    let n_threads = rayon::current_num_threads().max(1);
    let chunk_size = n_obs.div_ceil(n_threads).max(1);

    // Each chunk mirrors the serial inner loop of `loocv_score_for_params`
    // over its slice of `control_obs`, with a chunk-local warm-start chain and
    // a chunk-local (T × N) weight buffer.
    let partials: Vec<(f64, usize, Option<(usize, usize)>)> = control_obs
        .par_chunks(chunk_size)
        .map(|chunk| {
            let mut tau_sq_sum = 0.0;
            let mut n_valid = 0usize;
            let mut warm_alpha: Option<Array1<f64>> = None;
            let mut warm_beta: Option<Array1<f64>> = None;
            let mut warm_l: Option<Array2<f64>> = None;
            let mut weight_buf = Array2::<f64>::zeros((n_periods, n_units));

            for &(t, i) in chunk {
                compute_weight_matrix_cached_into(
                    &mut weight_buf,
                    y,
                    d,
                    dist_cache,
                    n_periods,
                    n_units,
                    i,
                    t,
                    lambda_time,
                    lambda_unit,
                    time_dist,
                );

                let ws = match (&warm_alpha, &warm_beta, &warm_l) {
                    (Some(a), Some(b), Some(l_prev)) => Some((a, b, l_prev)),
                    _ => None,
                };

                match estimate_model(
                    y,
                    control_mask,
                    &weight_buf.view(),
                    lambda_nn,
                    n_periods,
                    n_units,
                    max_iter,
                    tol,
                    Some((t, i)),
                    ws,
                    x,
                    None,
                ) {
                    Some((alpha, beta, l, _n_iters, _converged, gamma)) => {
                        let mut tau = y[[t, i]] - alpha[i] - beta[t] - l[[t, i]];
                        if let Some(x_mat) = x {
                            if let Some(ref g) = gamma {
                                let idx = t * n_units + i;
                                tau -= x_mat.row(idx).dot(g);
                            }
                        }
                        tau_sq_sum += tau * tau;
                        n_valid += 1;
                        warm_alpha = Some(alpha);
                        warm_beta = Some(beta);
                        warm_l = Some(l);
                    }
                    None => {
                        // First failing cell in this chunk; the caller promotes
                        // any chunk failure to Q = +∞ for the whole candidate.
                        return (tau_sq_sum, n_valid, Some((t, i)));
                    }
                }
            }
            (tau_sq_sum, n_valid, None)
        })
        .collect();

    // Failure semantics preserved: any chunk failure ⇒ candidate Q = +∞.
    // Report the earliest failing cell in traversal order.
    let mut total_valid = 0usize;
    for &(_, nv, ff) in &partials {
        total_valid += nv;
        if let Some(cell) = ff {
            return (f64::INFINITY, total_valid, Some(cell));
        }
    }

    // Fixed-order accumulation (chunk order == grid order), independent of
    // thread completion order.
    let tau_sq_sum: f64 = partials.iter().map(|&(s, _, _)| s).sum();
    if total_valid == 0 {
        (f64::INFINITY, 0, None)
    } else {
        (tau_sq_sum, total_valid, None)
    }
}

/// Evaluate Q(λ) without short-circuiting on the first failure.
///
/// Mirrors [`loocv_score_for_params`] but keeps iterating through all
/// control observations, recording every (t, i) for which `estimate_model`
/// returns `None`.  Used by the public entry points for a single "final
/// evaluation" pass at the LOOCV-selected λ so users can inspect the
/// complete failure pattern rather than just the first failure.
///
/// The score follows the paper's convention: if any observation fails,
/// Q = +∞ (selection should have already excluded such λ).  When all
/// observations succeed, Q = Σ τ̂² (same as the short-circuiting path).
///
/// Returns `(score, n_valid, failed_obs)` where `failed_obs` is the full
/// list of failing (period, unit) pairs in control-observation traversal
/// order (sorted by (t, i)).
#[allow(clippy::too_many_arguments)]
pub fn loocv_score_for_params_full_diagnostic(
    y: &ArrayView2<f64>,
    d: &ArrayView2<f64>,
    control_mask: &ArrayView2<u8>,
    time_dist: &ArrayView2<i64>,
    dist_cache: &UnitDistanceCache,
    control_obs: &[(usize, usize)],
    lambda_time: f64,
    lambda_unit: f64,
    lambda_nn: f64,
    max_iter: usize,
    tol: f64,
    x: Option<&ArrayView2<f64>>,
) -> (f64, usize, Vec<(usize, usize)>) {
    let n_periods = y.nrows();
    let n_units = y.ncols();

    let mut tau_sq_sum = 0.0;
    let mut n_valid = 0usize;
    let mut failed_obs: Vec<(usize, usize)> = Vec::new();

    // Warm start: same logic as loocv_score_for_params.
    let mut warm_alpha: Option<Array1<f64>> = None;
    let mut warm_beta: Option<Array1<f64>> = None;
    let mut warm_l: Option<Array2<f64>> = None;

    // Reuse a single weight buffer across all control observations.
    let mut weight_buf = Array2::<f64>::zeros((n_periods, n_units));

    for &(t, i) in control_obs {
        compute_weight_matrix_cached_into(
            &mut weight_buf,
            y,
            d,
            dist_cache,
            n_periods,
            n_units,
            i,
            t,
            lambda_time,
            lambda_unit,
            time_dist,
        );

        let ws = match (&warm_alpha, &warm_beta, &warm_l) {
            (Some(a), Some(b), Some(l_prev)) => Some((a, b, l_prev)),
            _ => None,
        };

        match estimate_model(
            y,
            control_mask,
            &weight_buf.view(),
            lambda_nn,
            n_periods,
            n_units,
            max_iter,
            tol,
            Some((t, i)),
            ws,
            x,
            None,
        ) {
            Some((alpha, beta, l, _n_iters, _converged, gamma)) => {
                let mut tau = y[[t, i]] - alpha[i] - beta[t] - l[[t, i]];
                if let Some(x_mat) = x {
                    if let Some(ref g) = gamma {
                        let idx = t * n_units + i;
                        tau -= x_mat.row(idx).dot(g);
                    }
                }
                tau_sq_sum += tau * tau;
                n_valid += 1;
                warm_alpha = Some(alpha);
                warm_beta = Some(beta);
                warm_l = Some(l);
            }
            None => {
                failed_obs.push((t, i));
            }
        }
    }

    let score = if !failed_obs.is_empty() || n_valid == 0 {
        f64::INFINITY
    } else {
        tau_sq_sum
    };

    (score, n_valid, failed_obs)
}

/// Univariate LOOCV search over a single tuning parameter.
///
/// Evaluates Q(λ) for each value in `grid` while holding the other two
/// parameters at the supplied fixed values.  Used in Stage 1 of the
/// two-stage search to obtain initial estimates, and in Stage 2 as the
/// inner step of coordinate descent.
///
/// # Arguments
/// * `param_type` — Which parameter to search: 0 = λ_time, 1 = λ_unit,
///   2 = λ_nn.
///
/// # Returns
/// `(best_value, best_score)`
#[allow(clippy::too_many_arguments)]
pub fn univariate_loocv_search(
    y: &ArrayView2<f64>,
    d: &ArrayView2<f64>,
    control_mask: &ArrayView2<u8>,
    time_dist: &ArrayView2<i64>,
    dist_cache: &UnitDistanceCache,
    control_obs: &[(usize, usize)],
    grid: &[f64],
    fixed_time: f64,
    fixed_unit: f64,
    fixed_nn: f64,
    param_type: usize,
    max_iter: usize,
    tol: f64,
    x: Option<&ArrayView2<f64>>,
) -> (f64, f64) {
    let mut best_score = f64::INFINITY;
    let mut best_value = grid.first().copied().unwrap_or(0.0);

    // Effective fixed values for the held-constant parameters during the
    // Stage-1 univariate search (paper footnote 2).
    //
    //   λ_time, λ_unit: MUST be finite.  The ADO/Mata layers reject `.`
    //     (Stata missing / +∞) at the grid-construction stage because paper
    //     Eq 3 defines the exponential kernels only on [0, ∞); λ = 0
    //     already encodes the uniform-weight case.  A +∞ that reaches this
    //     point indicates an internal bug.
    //
    //   λ_nn: MAY be +∞, which the paper (Eq 2 remark) recognizes as the
    //     DID/TWFE special case (L ≡ 0).  We map it to the large-finite
    //     sentinel 1e10 for the downstream estimator.
    //
    // We pass `f64::INFINITY` for λ_time/λ_unit internally in Stage 1 of
    // `loocv_grid_search` to express "this axis is not yet under search";
    // that sentinel is a calling convention, not a user-supplied value.
    // It is mapped to 0.0 here to select the uniform kernel on that axis.
    let fixed_time_eff = if fixed_time.is_infinite() { 0.0 } else { fixed_time };
    let fixed_unit_eff = if fixed_unit.is_infinite() { 0.0 } else { fixed_unit };
    let fixed_nn_eff = if fixed_nn.is_infinite() { 1e10 } else { fixed_nn };

    // Task 27: candidate-inner parallelism heuristic.  We never nest two
    // parallel layers (which would oversubscribe the pool and worsen the
    // `CyclingScoreCache` mutex contention).  Instead we activate exactly one
    // layer, chosen by grid size relative to the thread count:
    //   * grid.len() >= n_threads (large grid): parallelise ACROSS candidates
    //     (existing behaviour) and keep each candidate's control-obs loop
    //     serial, now pruned by a dynamically shared best score.
    //   * grid.len() <  n_threads (small / coarse grid, e.g. 2 values per
    //     axis): iterate candidates SERIALLY and parallelise the control-obs
    //     leave-one-out loop inside each candidate via `par_chunks`.
    //
    // Record the full (λ_time, λ_unit, λ_nn, score) tuple so that
    // `better_candidate` can apply the structural tie-breaker on the dimension
    // currently being searched.
    let n_threads = rayon::current_num_threads().max(1);
    let results: Vec<(f64, f64, f64, f64, f64)> = if grid.len() >= n_threads {
        // --- Large-grid branch: candidate-parallel + dynamic best score. ---
        // `atomic_best` tracks the lowest COMPLETE Q(λ) seen so far (fetch_min);
        // each candidate reads it (plus a TIE_TOL margin) as its early-
        // termination bound.  See `AtomicBestScore` docs for the numerical
        // identity proof: pruning only drops candidates that are strictly
        // worse than the incumbent beyond TIE_TOL, so the argmin is never
        // pruned and the selected (λ*, score) is unchanged.
        let atomic_best = AtomicBestScore::new(best_score);
        grid.par_iter()
            .map(|&value| {
                let (lambda_time, lambda_unit, lambda_nn) = match param_type {
                    0 => (value, fixed_unit_eff, fixed_nn_eff),
                    1 => (fixed_time_eff, value, fixed_nn_eff),
                    _ => (fixed_time_eff, fixed_unit_eff, value),
                };

                let (score, n_valid, _) = loocv_score_for_params(
                    y,
                    d,
                    control_mask,
                    time_dist,
                    dist_cache,
                    control_obs,
                    lambda_time,
                    lambda_unit,
                    lambda_nn,
                    max_iter,
                    tol,
                    atomic_best.pruning_bound(),
                    x,
                );
                // Only complete evaluations are valid Q values; early-
                // terminated partials must not lower the shared bound.
                if n_valid == control_obs.len() {
                    atomic_best.observe(score);
                }
                (value, lambda_time, lambda_unit, lambda_nn, score)
            })
            .collect()
    } else {
        // --- Small-grid branch: serial candidates, parallel control-obs. ---
        // Only one parallel layer is active (inside each candidate).  No
        // cross-candidate early termination (grid is tiny; full Q for every
        // candidate is cheap and maximally deterministic).
        grid.iter()
            .map(|&value| {
                let (lambda_time, lambda_unit, lambda_nn) = match param_type {
                    0 => (value, fixed_unit_eff, fixed_nn_eff),
                    1 => (fixed_time_eff, value, fixed_nn_eff),
                    _ => (fixed_time_eff, fixed_unit_eff, value),
                };

                let (score, _, _) = loocv_score_for_params_parallel(
                    y,
                    d,
                    control_mask,
                    time_dist,
                    dist_cache,
                    control_obs,
                    lambda_time,
                    lambda_unit,
                    lambda_nn,
                    max_iter,
                    tol,
                    x,
                );
                (value, lambda_time, lambda_unit, lambda_nn, score)
            })
            .collect()
    };

    // Track the incumbent triple (for tie-breaking) alongside the univariate
    // grid coordinate.
    let mut best_lt = fixed_time_eff;
    let mut best_lu = fixed_unit_eff;
    let mut best_ln = fixed_nn_eff;
    for (value, lt, lu, ln, score) in results {
        if better_candidate((lt, lu, ln, score), (best_lt, best_lu, best_ln, best_score)) {
            best_score = score;
            best_value = value;
            best_lt = lt;
            best_lu = lu;
            best_ln = ln;
        }
    }

    (best_value, best_score)
}

/// Cache-aware univariate LOOCV search for the twostep cycling path.
///
/// Identical to [`univariate_loocv_search`] but checks `score_cache` before
/// evaluating `loocv_score_for_params`.  Complete evaluations (n_valid ==
/// control_obs.len()) are stored in the cache; early-terminated results are
/// NOT cached to preserve correctness (they depend on the `best_score`
/// bound which varies across cycles).
#[allow(clippy::too_many_arguments)]
fn univariate_loocv_search_cached(
    y: &ArrayView2<f64>,
    d: &ArrayView2<f64>,
    control_mask: &ArrayView2<u8>,
    time_dist: &ArrayView2<i64>,
    dist_cache: &UnitDistanceCache,
    control_obs: &[(usize, usize)],
    grid: &[f64],
    fixed_time: f64,
    fixed_unit: f64,
    fixed_nn: f64,
    param_type: usize,
    max_iter: usize,
    tol: f64,
    x: Option<&ArrayView2<f64>>,
    score_cache: &CyclingScoreCache,
) -> (f64, f64) {
    let mut best_score = f64::INFINITY;
    let mut best_value = grid.first().copied().unwrap_or(0.0);

    let fixed_time_eff = if fixed_time.is_infinite() { 0.0 } else { fixed_time };
    let fixed_unit_eff = if fixed_unit.is_infinite() { 0.0 } else { fixed_unit };
    let fixed_nn_eff = if fixed_nn.is_infinite() { 1e10 } else { fixed_nn };

    let n_control = control_obs.len();

    // Task 27: same candidate-inner parallelism heuristic as
    // `univariate_loocv_search` (see that function for the full rationale),
    // combined with the P2.1 score cache.
    let n_threads = rayon::current_num_threads().max(1);
    let results: Vec<(f64, f64, f64, f64, f64)> = if grid.len() >= n_threads {
        // --- Large-grid branch: candidate-parallel + dynamic best score. ---
        // Cached scores are always COMPLETE evaluations (only complete or
        // failed results are inserted below), so they may safely lower the
        // shared pruning bound.
        let atomic_best = AtomicBestScore::new(best_score);
        grid.par_iter()
            .map(|&value| {
                let (lambda_time, lambda_unit, lambda_nn) = match param_type {
                    0 => (value, fixed_unit_eff, fixed_nn_eff),
                    1 => (fixed_time_eff, value, fixed_nn_eff),
                    _ => (fixed_time_eff, fixed_unit_eff, value),
                };

                let cache_key = (
                    lambda_time.to_bits(),
                    lambda_unit.to_bits(),
                    lambda_nn.to_bits(),
                );

                // Check cache first.
                if let Ok(cache) = score_cache.lock() {
                    if let Some(&(cached_score, _, _)) = cache.get(&cache_key) {
                        atomic_best.observe(cached_score);
                        return (value, lambda_time, lambda_unit, lambda_nn, cached_score);
                    }
                }

                let (score, n_valid, first_failed) = loocv_score_for_params(
                    y,
                    d,
                    control_mask,
                    time_dist,
                    dist_cache,
                    control_obs,
                    lambda_time,
                    lambda_unit,
                    lambda_nn,
                    max_iter,
                    tol,
                    atomic_best.pruning_bound(),
                    x,
                );

                // Cache only complete evaluations (not early-terminated).
                if n_valid == n_control || score == f64::INFINITY {
                    if let Ok(mut cache) = score_cache.lock() {
                        cache.insert(cache_key, (score, n_valid, first_failed));
                    }
                }
                if n_valid == n_control {
                    atomic_best.observe(score);
                }

                (value, lambda_time, lambda_unit, lambda_nn, score)
            })
            .collect()
    } else {
        // --- Small-grid branch: serial candidates, parallel control-obs. ---
        grid.iter()
            .map(|&value| {
                let (lambda_time, lambda_unit, lambda_nn) = match param_type {
                    0 => (value, fixed_unit_eff, fixed_nn_eff),
                    1 => (fixed_time_eff, value, fixed_nn_eff),
                    _ => (fixed_time_eff, fixed_unit_eff, value),
                };

                let cache_key = (
                    lambda_time.to_bits(),
                    lambda_unit.to_bits(),
                    lambda_nn.to_bits(),
                );

                // Check cache first.
                if let Ok(cache) = score_cache.lock() {
                    if let Some(&(cached_score, _, _)) = cache.get(&cache_key) {
                        return (value, lambda_time, lambda_unit, lambda_nn, cached_score);
                    }
                }

                let (score, n_valid, first_failed) = loocv_score_for_params_parallel(
                    y,
                    d,
                    control_mask,
                    time_dist,
                    dist_cache,
                    control_obs,
                    lambda_time,
                    lambda_unit,
                    lambda_nn,
                    max_iter,
                    tol,
                    x,
                );

                // Cache only complete evaluations (the parallel path always
                // computes the full Q, so a finite score is always complete).
                if n_valid == n_control || score == f64::INFINITY {
                    if let Ok(mut cache) = score_cache.lock() {
                        cache.insert(cache_key, (score, n_valid, first_failed));
                    }
                }

                (value, lambda_time, lambda_unit, lambda_nn, score)
            })
            .collect()
    };

    let mut best_lt = fixed_time_eff;
    let mut best_lu = fixed_unit_eff;
    let mut best_ln = fixed_nn_eff;
    for (value, lt, lu, ln, score) in results {
        if better_candidate((lt, lu, ln, score), (best_lt, best_lu, best_ln, best_score)) {
            best_score = score;
            best_value = value;
            best_lt = lt;
            best_lu = lu;
            best_ln = ln;
        }
    }

    (best_value, best_score)
}

/// Coordinate descent cycling over (λ_unit, λ_time, λ_nn) for the twostep
/// method.
///
/// Starting from the Stage 1 initial values, successively optimizes each
/// parameter while holding the other two at their most recent optimal values.
/// Cycling order: λ_unit → λ_time → λ_nn.
/// Terminates when |Q_new − Q_old| < 1e-6 or `max_cycles` is reached.
///
/// Uses a score cache to avoid re-evaluating λ triples that were already
/// computed in previous cycles (P2.1 optimisation).
#[allow(clippy::too_many_arguments)]
pub fn cycling_parameter_search(
    y: &ArrayView2<f64>,
    d: &ArrayView2<f64>,
    control_mask: &ArrayView2<u8>,
    time_dist: &ArrayView2<i64>,
    dist_cache: &UnitDistanceCache,
    control_obs: &[(usize, usize)],
    lambda_time_grid: &[f64],
    lambda_unit_grid: &[f64],
    lambda_nn_grid: &[f64],
    initial_time: f64,
    initial_unit: f64,
    initial_nn: f64,
    max_iter: usize,
    tol: f64,
    max_cycles: usize,
    x: Option<&ArrayView2<f64>>,
) -> (f64, f64, f64) {
    let mut lambda_time = initial_time;
    let mut lambda_unit = initial_unit;
    let mut lambda_nn = initial_nn;
    let mut prev_score = f64::INFINITY;

    // P2.1: Score cache shared across all cycles to avoid redundant evaluations.
    let score_cache: CyclingScoreCache = Mutex::new(HashMap::new());

    for _cycle in 0..max_cycles {
        // Optimize λ_unit (fix λ_time, λ_nn)
        let (new_unit, _) = univariate_loocv_search_cached(
            y,
            d,
            control_mask,
            time_dist,
            dist_cache,
            control_obs,
            lambda_unit_grid,
            lambda_time,
            0.0,
            lambda_nn,
            1,
            max_iter,
            tol,
            x,
            &score_cache,
        );
        lambda_unit = new_unit;

        // Optimize λ_time (fix λ_unit, λ_nn)
        let (new_time, _) = univariate_loocv_search_cached(
            y,
            d,
            control_mask,
            time_dist,
            dist_cache,
            control_obs,
            lambda_time_grid,
            0.0,
            lambda_unit,
            lambda_nn,
            0,
            max_iter,
            tol,
            x,
            &score_cache,
        );
        lambda_time = new_time;

        // Optimize λ_nn (fix λ_unit, λ_time)
        let (new_nn, score) = univariate_loocv_search_cached(
            y,
            d,
            control_mask,
            time_dist,
            dist_cache,
            control_obs,
            lambda_nn_grid,
            lambda_time,
            lambda_unit,
            0.0,
            2,
            max_iter,
            tol,
            x,
            &score_cache,
        );
        lambda_nn = new_nn;

        // Check cycling convergence.  The 1e-6 threshold governs the outer
        // coordinate descent loop and is distinct from `tol`, which controls
        // convergence of the inner estimation solver.
        if (score - prev_score).abs() < 1e-6 {
            break;
        }
        prev_score = score;
    }

    (lambda_time, lambda_unit, lambda_nn)
}

/// Two-stage LOOCV search for the twostep method.
///
/// Selects λ̂ = argmin_{λ ∈ Λ} Q(λ) using:
///   Stage 1 — Univariate initialization with extreme fixed values.
///   Stage 2 — Coordinate descent cycling until convergence.
///
/// # Returns
/// `(best_λ_time, best_λ_unit, best_λ_nn, best_score, n_valid,
///   n_attempted, first_failed_obs)`
///
/// Callers that additionally want the Stage-1 univariate initialisation
/// triple (paper Footnote 2) should use
/// [`loocv_grid_search_with_stage1`] instead.
#[allow(clippy::too_many_arguments)]
pub fn loocv_grid_search(
    y: &ArrayView2<f64>,
    d: &ArrayView2<f64>,
    control_mask: &ArrayView2<u8>,
    time_dist: &ArrayView2<i64>,
    lambda_time_grid: &[f64],
    lambda_unit_grid: &[f64],
    lambda_nn_grid: &[f64],
    max_iter: usize,
    tol: f64,
    x: Option<&ArrayView2<f64>>,
) -> TropResult<LoocvGridSearchResult> {
    let (result, _, _, _) = loocv_grid_search_with_stage1(
        y,
        d,
        control_mask,
        time_dist,
        lambda_time_grid,
        lambda_unit_grid,
        lambda_nn_grid,
        max_iter,
        tol,
        x,
    )?;
    Ok(result)
}

/// Two-stage LOOCV search for the twostep method, returning the Stage-1
/// univariate initialisation alongside the Stage-2 polished result.
///
/// The Stage-1 triple `(λ_time_init, λ_unit_init, λ_nn_init)` is the
/// argmin of three independent univariate sweeps, each fixing the other
/// two parameters at the Footnote-2 extrema:
///   - λ_time sweep: fix λ_unit = 0, λ_nn = ∞ (factor model off).
///   - λ_unit sweep: fix λ_time = 0, λ_nn = ∞ (factor model off).
///   - λ_nn   sweep: fix λ_time = 0, λ_unit = 0 (uniform weights).
///
/// Stage 2 then performs coordinate-descent cycling from this seed.
/// Exposing the Stage-1 triple lets downstream diagnostics compare the
/// initial and refined optima; a large gap indicates that Stage-2 did
/// non-trivial work on a non-convex Q(λ) surface.
#[allow(clippy::too_many_arguments)]
pub fn loocv_grid_search_with_stage1(
    y: &ArrayView2<f64>,
    d: &ArrayView2<f64>,
    control_mask: &ArrayView2<u8>,
    time_dist: &ArrayView2<i64>,
    lambda_time_grid: &[f64],
    lambda_unit_grid: &[f64],
    lambda_nn_grid: &[f64],
    max_iter: usize,
    tol: f64,
    x: Option<&ArrayView2<f64>>,
) -> TropResult<LoocvGridSearchResultWithStage1> {
    // Guard: ensure at least one treated cell exists to avoid silent no-op.
    validate_has_treated_units(d)?;

    // Per paper Eq. 5: Q(λ) sums over every D=0 cell.
    let control_obs = get_control_observations(y, control_mask);
    let n_attempted = control_obs.len();

    // Build the pairwise distance cache once; it is reused across every
    // grid point and every control observation within each LOOCV call.
    let dist_cache = UnitDistanceCache::build(y, d);

    // Stage 1: Univariate searches for initial values.
    // λ_time search: fix λ_unit = 0, λ_nn = ∞.
    let (lambda_time_init, _) = univariate_loocv_search(
        y,
        d,
        control_mask,
        time_dist,
        &dist_cache,
        &control_obs,
        lambda_time_grid,
        0.0,
        0.0,
        f64::INFINITY,
        0,
        max_iter,
        tol,
        x,
    );

    // λ_nn search: fix λ_time = 0, λ_unit = 0 (uniform weights).
    //
    // Paper Footnote 2 describes the two-stage scheme as "letting λ_nn = ∞
    // and λ_unit = 0, we minimize over a grid of values of λ_time, and
    // similarly for the other two penalty parameters" — i.e. each stage
    // fixes the *other* two at extrema that do not interfere with the
    // parameter under search.  For λ_nn that means uniform weights
    // (λ_time = 0, λ_unit = 0); seeding with λ_time = ∞ would collapse the
    // time kernel onto the target period alone and mask exactly the factor
    // structure λ_nn is meant to control.  Because Stage-2 cycling only
    // polishes from this seed on a non-convex Q(λ) surface, a biased
    // initial point can yield a different local optimum.
    let (lambda_nn_init, _) = univariate_loocv_search(
        y,
        d,
        control_mask,
        time_dist,
        &dist_cache,
        &control_obs,
        lambda_nn_grid,
        0.0,
        0.0,
        0.0,
        2,
        max_iter,
        tol,
        x,
    );

    // λ_unit search: fix λ_nn = ∞, λ_time = 0.
    let (lambda_unit_init, _) = univariate_loocv_search(
        y,
        d,
        control_mask,
        time_dist,
        &dist_cache,
        &control_obs,
        lambda_unit_grid,
        0.0,
        0.0,
        f64::INFINITY,
        1,
        max_iter,
        tol,
        x,
    );

    // Stage 2: Coordinate descent refinement.
    let (best_time, best_unit, best_nn) = cycling_parameter_search(
        y,
        d,
        control_mask,
        time_dist,
        &dist_cache,
        &control_obs,
        lambda_time_grid,
        lambda_unit_grid,
        lambda_nn_grid,
        lambda_time_init,
        lambda_unit_init,
        lambda_nn_init,
        max_iter,
        tol,
        10,
        x,
    );

    // Final evaluation at the selected parameters.
    //
    // Uses the full-diagnostic variant so that the exported `n_valid`
    // counts every successful LOO fit (not "all successes up to the first
    // failure").  When the selected λ produces no failures — the
    // overwhelmingly common case — this is bit-identical to the
    // short-circuit path; when it does produce failures, the diagnostic
    // variant lets us report the first failure without under-counting
    // the other successful observations.
    let (best_score, n_valid, failed_obs) = loocv_score_for_params_full_diagnostic(
        y,
        d,
        control_mask,
        time_dist,
        &dist_cache,
        &control_obs,
        best_time,
        best_unit,
        best_nn,
        max_iter,
        tol,
        x,
    );

    // Preserve the historical "first failed observation" contract.
    let first_failed = failed_obs.first().copied();

    let result = (
        best_time,
        best_unit,
        best_nn,
        best_score,
        n_valid,
        n_attempted,
        first_failed,
    );
    Ok((result, lambda_time_init, lambda_unit_init, lambda_nn_init))
}

/// Exhaustive (Cartesian) LOOCV grid search for the twostep method.
///
/// Evaluates every (λ_time, λ_unit, λ_nn) combination in parallel and
/// returns the triple that minimises the LOOCV criterion Q(λ) under the
/// per-observation weighting scheme of Algorithm 2.  Complexity is
/// O(|Λ_time| · |Λ_unit| · |Λ_nn|); prefer the coordinate-descent variant
/// [`loocv_grid_search`] for large grids.
///
/// Compared with the cycling path this search is guaranteed to return the
/// global grid minimum (subject to `TIE_TOL`), so the selected λ is
/// independent of the Stage-1 initialisation.  On small panels with a
/// non-convex Q(λ) surface (e.g. Basque, Germany) this eliminates the
/// platform-dependent λ drift observed with cycling + BLAS jitter.
///
/// # Returns
/// `(best_λ_time, best_λ_unit, best_λ_nn, best_score, n_valid,
///   n_attempted, first_failed_obs)`
#[allow(clippy::too_many_arguments)]
pub fn loocv_grid_search_exhaustive(
    y: &ArrayView2<f64>,
    d: &ArrayView2<f64>,
    control_mask: &ArrayView2<u8>,
    time_dist: &ArrayView2<i64>,
    lambda_time_grid: &[f64],
    lambda_unit_grid: &[f64],
    lambda_nn_grid: &[f64],
    max_iter: usize,
    tol: f64,
    x: Option<&ArrayView2<f64>>,
) -> TropResult<LoocvGridSearchResult> {
    // Guard: ensure at least one treated cell exists to avoid silent no-op.
    validate_has_treated_units(d)?;

    // Per paper Eq. 5: Q(λ) sums over every D=0 cell.
    let control_obs = get_control_observations(y, control_mask);
    let n_attempted = control_obs.len();

    // Build the pairwise distance cache once; it is reused across every
    // grid point and every control observation within each LOOCV call.
    let dist_cache = UnitDistanceCache::build(y, d);

    // Enumerate all grid combinations.  The C bridge guarantees grid values
    // are finite (sentinel ∞ is pre-converted); no `is_infinite` handling is
    // required here.
    let mut grid_combinations: Vec<(f64, f64, f64)> = Vec::new();
    for &lt in lambda_time_grid {
        for &lu in lambda_unit_grid {
            for &ln in lambda_nn_grid {
                grid_combinations.push((lt, lu, ln));
            }
        }
    }

    // Parallel grid search.  `rayon::collect` preserves insertion order so
    // the later `better_candidate` sweep is deterministic.
    //
    // Task 27: the batch shares a dynamic best score (`atomic_best`, fetch_min)
    // so each combination can prune against the lowest COMPLETE Q(λ) found so
    // far, plus a TIE_TOL margin.  The winning triple is re-evaluated by the
    // full-diagnostic pass below, and pruning only ever drops combinations
    // that are strictly worse than the incumbent beyond TIE_TOL, so the
    // selected triple — and every reported quantity — is identical to the
    // previous `f64::INFINITY` (no-prune) behaviour.  See `AtomicBestScore`.
    let atomic_best = AtomicBestScore::new(f64::INFINITY);
    let results: Vec<LoocvResultTuple> = grid_combinations
        .into_par_iter()
        .with_max_len(1) // Enable fine-grained work-stealing for load balancing
        .map(|(lt, lu, ln)| {
            // Apply the same sentinel conversion as the other search paths
            // so behaviour stays consistent when the grid is supplied
            // directly from Rust rather than through the C bridge.
            let lt_eff = if lt.is_infinite() { 0.0 } else { lt };
            let lu_eff = if lu.is_infinite() { 0.0 } else { lu };
            let ln_eff = if ln.is_infinite() { 1e10 } else { ln };

            let (score, n_valid, first_failed) = loocv_score_for_params(
                y,
                d,
                control_mask,
                time_dist,
                &dist_cache,
                &control_obs,
                lt_eff,
                lu_eff,
                ln_eff,
                max_iter,
                tol,
                atomic_best.pruning_bound(),
                x,
            );
            if n_valid == control_obs.len() {
                atomic_best.observe(score);
            }
            (lt, lu, ln, score, n_valid, first_failed)
        })
        .collect();

    // Reduce with the tie-breaker-aware comparator.
    let mut best_result: LoocvResultTuple = (
        lambda_time_grid.first().copied().unwrap_or(0.0),
        lambda_unit_grid.first().copied().unwrap_or(0.0),
        lambda_nn_grid.first().copied().unwrap_or(0.0),
        f64::INFINITY,
        0usize,
        None,
    );
    for (lt, lu, ln, score, n_valid, first_failed) in results {
        if better_candidate(
            (lt, lu, ln, score),
            (best_result.0, best_result.1, best_result.2, best_result.3),
        ) {
            best_result = (lt, lu, ln, score, n_valid, first_failed);
        }
    }

    // Final diagnostic evaluation at the winning λ triple.  Mirrors
    // `loocv_grid_search`: using the full-diagnostic variant ensures the
    // reported `n_valid` reflects every successful observation, not just
    // "all successes up to the first failure".
    let lt_eff = if best_result.0.is_infinite() { 0.0 } else { best_result.0 };
    let lu_eff = if best_result.1.is_infinite() { 0.0 } else { best_result.1 };
    let ln_eff = if best_result.2.is_infinite() { 1e10 } else { best_result.2 };
    let (best_score, n_valid, failed_obs) = loocv_score_for_params_full_diagnostic(
        y,
        d,
        control_mask,
        time_dist,
        &dist_cache,
        &control_obs,
        lt_eff,
        lu_eff,
        ln_eff,
        max_iter,
        tol,
        x,
    );
    let first_failed = failed_obs.first().copied();

    Ok((
        best_result.0,
        best_result.1,
        best_result.2,
        best_score,
        n_valid,
        n_attempted,
        first_failed,
    ))
}

/// Evaluate the LOOCV criterion for a single (λ_time, λ_unit, λ_nn) triple
/// under the joint method (paper Remark 6.1 aggregation; LOOCV rule
/// inherited from Algorithm 2 step 2).
///
/// Uses global weights δ(λ_time, λ_unit) and a homogeneous treatment
/// effect.  For each control observation (t, i) the weight δ_{t, i} is set
/// to zero before fitting (the "leave one cell out" analogue of Eq. (4)
/// for the shared-weight model), and the pseudo-treatment effect is
/// computed as
///
///   τ̂_{ti} = Y_{ti} − μ̂ − α̂_i − β̂_t − L̂_{ti}.
///
/// Q(λ) is the Eq. (5) criterion restricted to control cells with finite Y.
///
/// Returns `(score, n_valid, first_failed_obs)`.
#[allow(clippy::too_many_arguments)]
pub fn loocv_score_joint(
    y: &ArrayView2<f64>,
    d: &ArrayView2<f64>,
    control_obs: &[(usize, usize)],
    lambda_time: f64,
    lambda_unit: f64,
    lambda_nn: f64,
    treated_periods: usize,
    max_iter: usize,
    tol: f64,
    best_score_so_far: f64,
    x: Option<&ArrayView2<f64>>,
) -> (f64, usize, Option<(usize, usize)>) {
    let n_periods = y.nrows();
    let n_units = y.ncols();

    let mut tau_sq_sum = 0.0;
    let mut n_valid = 0usize;

    // Compute global weights δ once into a reusable buffer.
    let mut delta = Array2::<f64>::zeros((n_periods, n_units));
    compute_joint_weights_into(&mut delta, y, d, lambda_time, lambda_unit, treated_periods);

    // Reuse a single buffer for the leave-one-cell-out copy of δ.
    let mut delta_ex = Array2::<f64>::zeros((n_periods, n_units));

    for &(t_ex, i_ex) in control_obs {
        // Zero out the excluded observation's weight.
        delta_ex.assign(&delta);
        delta_ex[[t_ex, i_ex]] = 0.0;

        // Fit joint model with the modified weights.
        //
        // When λ_nn ≥ 1e10 the low-rank step is skipped (L ≡ 0). τ is not used
        // downstream (the LOOCV score uses the pseudo-residual at the excluded
        // cell directly), so we pass a dummy 0.0 in its slot.
        // B.2 defensive check: delta_ex inherits the (1 − D) mask from δ.
        debug_assert_delta_is_1minus_d_masked(
            d, &delta_ex.view(), "loocv_score_joint/delta_ex",
        );
        let result = if lambda_nn >= 1e10 {
            solve_joint_no_lowrank(y, &delta_ex.view(), x).map(|(mu, alpha, beta, gamma)| {
                let l = Array2::<f64>::zeros((n_periods, n_units));
                (mu, alpha, beta, l, 0.0_f64, 1_usize, true, gamma)
            })
        } else {
            solve_joint_with_lowrank(y, d, &delta_ex.view(), lambda_nn, max_iter, tol, x)
        };

        match result {
            Some((mu, alpha, beta, l, _tau, _n_iters, _converged, gamma)) => {
                // Pseudo-treatment effect: τ̂ = Y − μ − α − β − L − X'γ.
                let y_ti = if y[[t_ex, i_ex]].is_finite() {
                    y[[t_ex, i_ex]]
                } else {
                    continue;
                };
                let mut tau_loocv = y_ti - mu - alpha[i_ex] - beta[t_ex] - l[[t_ex, i_ex]];
                if let Some(x_mat) = x {
                    if let Some(ref g) = gamma {
                        let idx = t_ex * n_units + i_ex;
                        tau_loocv -= x_mat.row(idx).dot(g);
                    }
                }
                tau_sq_sum += tau_loocv * tau_loocv;
                n_valid += 1;

                // Early termination: Q(λ) = Σ τ̂² is a sum of non-negative
                // terms, so the partial sum is monotonically non-decreasing.
                // Once it exceeds the current best score, no subsequent terms
                // can bring it below best — safe to abandon this λ.
                if tau_sq_sum > best_score_so_far {
                    return (tau_sq_sum, n_valid, None);
                }
            }
            None => {
                // Estimation failure invalidates this λ combination.
                return (f64::INFINITY, n_valid, Some((t_ex, i_ex)));
            }
        }
    }

    if n_valid == 0 {
        (f64::INFINITY, 0, None)
    } else {
        (tau_sq_sum, n_valid, None)
    }
}

/// Evaluate the joint LOOCV criterion Q(λ) without short-circuiting on the
/// first failure.
///
/// Mirrors [`loocv_score_joint`] but keeps iterating through all control
/// observations, recording every (t, i) for which the joint estimator
/// returns `None`.  Used for a single "final evaluation" pass at the
/// LOOCV-selected λ so exported diagnostics (`n_valid`, `failed_obs`)
/// report the complete failure pattern rather than just the first failure.
///
/// Score convention follows the paper: if any observation fails, Q = +∞
/// (LOOCV selection should have already excluded such λ).  When every
/// observation succeeds, Q = Σ τ̂² (bit-identical to the short-circuit
/// path).
///
/// Returns `(score, n_valid, failed_obs)` where `failed_obs` is the full
/// list of failing (period, unit) pairs in `control_obs` traversal order.
#[allow(clippy::too_many_arguments)]
pub fn loocv_score_joint_full_diagnostic(
    y: &ArrayView2<f64>,
    d: &ArrayView2<f64>,
    control_obs: &[(usize, usize)],
    lambda_time: f64,
    lambda_unit: f64,
    lambda_nn: f64,
    treated_periods: usize,
    max_iter: usize,
    tol: f64,
    x: Option<&ArrayView2<f64>>,
) -> (f64, usize, Vec<(usize, usize)>) {
    let n_periods = y.nrows();
    let n_units = y.ncols();

    let mut tau_sq_sum = 0.0;
    let mut n_valid = 0usize;
    let mut failed_obs: Vec<(usize, usize)> = Vec::new();

    // Compute global weights δ once into a reusable buffer.
    let mut delta = Array2::<f64>::zeros((n_periods, n_units));
    compute_joint_weights_into(&mut delta, y, d, lambda_time, lambda_unit, treated_periods);

    // Reuse a single buffer for the leave-one-cell-out copy of δ.
    let mut delta_ex = Array2::<f64>::zeros((n_periods, n_units));

    for &(t_ex, i_ex) in control_obs {
        // Zero out the excluded cell's weight (Eq. 4 analogue under shared δ).
        delta_ex.assign(&delta);
        delta_ex[[t_ex, i_ex]] = 0.0;

        // B.2 defensive check: delta_ex inherits the (1 − D) mask from δ.
        debug_assert_delta_is_1minus_d_masked(
            d,
            &delta_ex.view(),
            "loocv_score_joint_full_diagnostic/delta_ex",
        );

        let result = if lambda_nn >= 1e10 {
            solve_joint_no_lowrank(y, &delta_ex.view(), x).map(|(mu, alpha, beta, gamma)| {
                let l = Array2::<f64>::zeros((n_periods, n_units));
                (mu, alpha, beta, l, 0.0_f64, 1_usize, true, gamma)
            })
        } else {
            solve_joint_with_lowrank(y, d, &delta_ex.view(), lambda_nn, max_iter, tol, x)
        };

        match result {
            Some((mu, alpha, beta, l, _tau, _n_iters, _converged, gamma)) => {
                // Pseudo-treatment effect: τ̂ = Y − μ − α − β − L − X'γ.  Skip cells
                // whose outcome is non-finite (these cannot enter Q(λ)).
                let y_ti = if y[[t_ex, i_ex]].is_finite() {
                    y[[t_ex, i_ex]]
                } else {
                    continue;
                };
                let mut tau_loocv = y_ti - mu - alpha[i_ex] - beta[t_ex] - l[[t_ex, i_ex]];
                if let Some(x_mat) = x {
                    if let Some(ref g) = gamma {
                        let idx = t_ex * n_units + i_ex;
                        tau_loocv -= x_mat.row(idx).dot(g);
                    }
                }
                tau_sq_sum += tau_loocv * tau_loocv;
                n_valid += 1;
            }
            None => {
                failed_obs.push((t_ex, i_ex));
            }
        }
    }

    let score = if !failed_obs.is_empty() || n_valid == 0 {
        f64::INFINITY
    } else {
        tau_sq_sum
    };

    (score, n_valid, failed_obs)
}

/// Brute-force LOOCV grid search for the joint method.
///
/// Exhaustively evaluates all (λ_time, λ_unit, λ_nn) combinations in
/// parallel.  Complexity is O(|grid|³) and can be expensive; the
/// coordinate-descent variant `loocv_cycling_search_joint` is preferred
/// for production use.
///
/// # Returns
/// `(best_λ_time, best_λ_unit, best_λ_nn, best_score, n_valid,
///   n_attempted, first_failed_obs)`
#[allow(clippy::too_many_arguments)]
pub fn loocv_grid_search_joint(
    y: &ArrayView2<f64>,
    d: &ArrayView2<f64>,
    control_mask: &ArrayView2<u8>,
    lambda_time_grid: &[f64],
    lambda_unit_grid: &[f64],
    lambda_nn_grid: &[f64],
    max_iter: usize,
    tol: f64,
    x: Option<&ArrayView2<f64>>,
) -> LoocvGridSearchResult {
    // Determine the number of treated periods from D.
    //
    // Safety note: this function is only reached from the Stata plugin through
    // C ABI entries that have already called `check_simultaneous_adoption`.
    // In-process Rust callers (tests, benchmarks) must pass a compliant D.
    let treated_periods = check_simultaneous_adoption(d)
        .expect("joint LOOCV requires simultaneous adoption; caller must pre-validate");

    // Per paper Eq. 5: Q(λ) sums over every D=0 cell.
    let control_obs = get_control_observations(y, control_mask);
    let n_attempted = control_obs.len();

    // Enumerate all grid combinations.
    let mut grid_combinations: Vec<(f64, f64, f64)> = Vec::new();
    for &lt in lambda_time_grid {
        for &lu in lambda_unit_grid {
            for &ln in lambda_nn_grid {
                grid_combinations.push((lt, lu, ln));
            }
        }
    }

    // Parallel grid search.
    //
    // Task 27: share a dynamic best score (`atomic_best`, fetch_min) across the
    // batch so each combination prunes against the lowest COMPLETE Q(λ) found
    // so far plus a TIE_TOL margin.  The winner is re-evaluated by the
    // full-diagnostic pass below and pruning only drops combinations strictly
    // worse than the incumbent beyond TIE_TOL, so the selected triple is
    // identical to the previous no-prune behaviour.  See `AtomicBestScore`.
    let atomic_best = AtomicBestScore::new(f64::INFINITY);
    let results: Vec<LoocvResultTuple> = grid_combinations
        .into_par_iter()
        .with_max_len(1) // Enable fine-grained work-stealing for load balancing
        .map(|(lt, lu, ln)| {
            // Convert infinity values
            let lt_eff = if lt.is_infinite() { 0.0 } else { lt };
            let lu_eff = if lu.is_infinite() { 0.0 } else { lu };
            let ln_eff = if ln.is_infinite() { 1e10 } else { ln };

            let (score, n_valid, first_failed) = loocv_score_joint(
                y,
                d,
                &control_obs,
                lt_eff,
                lu_eff,
                ln_eff,
                treated_periods,
                max_iter,
                tol,
                atomic_best.pruning_bound(),
                x,
            );
            if n_valid == control_obs.len() {
                atomic_best.observe(score);
            }

            (lt, lu, ln, score, n_valid, first_failed)
        })
        .collect();

    // Find best result using the tie-aware comparator.
    //
    // Initial incumbent uses `+Inf` score so any finite-score candidate wins
    // under `better_candidate`.  The (λ_t, λ_u, λ_n) placeholders are the
    // first grid points; they are overwritten as soon as any candidate wins.
    let mut best_result: LoocvResultTuple = (
        lambda_time_grid.first().copied().unwrap_or(0.0),
        lambda_unit_grid.first().copied().unwrap_or(0.0),
        lambda_nn_grid.first().copied().unwrap_or(0.0),
        f64::INFINITY,
        0usize,
        None,
    );

    for (lt, lu, ln, score, n_valid, first_failed) in results {
        if better_candidate(
            (lt, lu, ln, score),
            (best_result.0, best_result.1, best_result.2, best_result.3),
        ) {
            best_result = (lt, lu, ln, score, n_valid, first_failed);
        }
    }

    let (best_lt, best_lu, best_ln, _search_score, _search_n_valid, _search_first_failed) =
        best_result;

    // Final diagnostic evaluation at the winning λ triple.  Mirrors the
    // twostep pattern (see `loocv_grid_search` / `loocv_grid_search_exhaustive`):
    // the search pass uses the short-circuit `loocv_score_joint` for speed,
    // and this pass walks every control observation so the exported `n_valid`
    // counts every successful LOO fit (not "successes up to the first
    // failure").  When the selected λ produces no failures — the overwhelmingly
    // common case — the score is bit-identical to the short-circuit path.
    let lt_eff = if best_lt.is_infinite() { 0.0 } else { best_lt };
    let lu_eff = if best_lu.is_infinite() { 0.0 } else { best_lu };
    let ln_eff = if best_ln.is_infinite() { 1e10 } else { best_ln };
    let (best_score, n_valid, failed_obs) = loocv_score_joint_full_diagnostic(
        y,
        d,
        &control_obs,
        lt_eff,
        lu_eff,
        ln_eff,
        treated_periods,
        max_iter,
        tol,
        x,
    );
    let first_failed = failed_obs.first().copied();

    (
        best_lt,
        best_lu,
        best_ln,
        best_score,
        n_valid,
        n_attempted,
        first_failed,
    )
}

/// Coordinate descent LOOCV search for joint method tuning parameters.
///
/// Paper: Footnote 2 applied to joint method (Remark 6.1).
///
/// Same two-stage approach as Twostep LOOCV:
///   Stage 1: Univariate searches with extreme fixed values for initial estimates
///   Stage 2: Cycling (coordinate descent) until convergence
/// but using joint method's global weights δ and LOOCV score (loocv_score_joint).
///
/// Complexity: O(|grid| × n_cycles) instead of O(|grid|^3) for brute-force.
///
/// Callers that additionally want the Stage-1 univariate initialisation
/// triple (paper Footnote 2) should use
/// [`loocv_cycling_search_joint_with_stage1`] instead.
#[allow(clippy::too_many_arguments)]
pub fn loocv_cycling_search_joint(
    y: &ArrayView2<f64>,
    d: &ArrayView2<f64>,
    control_mask: &ArrayView2<u8>,
    lambda_time_grid: &[f64],
    lambda_unit_grid: &[f64],
    lambda_nn_grid: &[f64],
    max_iter: usize,
    tol: f64,
    max_cycles: usize,
    x: Option<&ArrayView2<f64>>,
) -> LoocvGridSearchResult {
    let (result, _, _, _) = loocv_cycling_search_joint_with_stage1(
        y,
        d,
        control_mask,
        lambda_time_grid,
        lambda_unit_grid,
        lambda_nn_grid,
        max_iter,
        tol,
        max_cycles,
        x,
    );
    result
}

/// Coordinate descent LOOCV search for joint method tuning parameters,
/// returning the Stage-1 univariate initialisation alongside the Stage-2
/// polished result.
///
/// See [`loocv_grid_search_with_stage1`] for the Stage-1 contract; this
/// joint-method variant applies the same Footnote-2 extrema scheme
/// (λ_time = 0, λ_unit = 0 when searching λ_nn; λ_nn = ∞ when searching
/// λ_time or λ_unit) but uses the global weights δ(λ_time, λ_unit) and
/// homogeneous-τ LOOCV score (Remark 6.1).
#[allow(clippy::too_many_arguments)]
pub fn loocv_cycling_search_joint_with_stage1(
    y: &ArrayView2<f64>,
    d: &ArrayView2<f64>,
    control_mask: &ArrayView2<u8>,
    lambda_time_grid: &[f64],
    lambda_unit_grid: &[f64],
    lambda_nn_grid: &[f64],
    max_iter: usize,
    tol: f64,
    max_cycles: usize,
    x: Option<&ArrayView2<f64>>,
) -> LoocvGridSearchResultWithStage1 {
    // Determine treated periods from D matrix.  See `loocv_grid_search_joint`
    // for the simultaneous-adoption precondition.
    let treated_periods = check_simultaneous_adoption(d)
        .expect("joint LOOCV requires simultaneous adoption; caller must pre-validate");

    // Per paper Eq. 5: Q(λ) sums over every D=0 cell.
    let control_obs = get_control_observations(y, control_mask);
    let n_attempted = control_obs.len();

    // P2.1: Score cache shared across Stage-1 and Stage-2 to avoid redundant
    // joint LOOCV evaluations during coordinate descent cycling.
    let joint_score_cache: CyclingScoreCache = Mutex::new(HashMap::new());

    // Helper closure: univariate search over one parameter grid (parallelized)
    // param_idx: 0=lambda_time, 1=lambda_unit, 2=lambda_nn
    // best_score_hint: upper bound for early termination pruning; use
    //   f64::INFINITY when no prior bound is available (Stage 1).
    let univariate_search =
        |grid: &[f64],
         param_idx: usize,
         fixed: (f64, f64, f64),
         best_score_hint: f64|
         -> (f64, f64, usize, Option<(usize, usize)>) {
            // Record the full (λ_time, λ_unit, λ_nn) triple alongside the
            // univariate grid coordinate so that `better_candidate` can apply
            // the tie-breaker deterministically.
            //
            // Task 27: `best_score_hint` seeds a dynamically shared best score
            // (`atomic_best`, fetch_min).  Each candidate prunes against the
            // lowest COMPLETE Q(λ) observed so far plus a TIE_TOL margin,
            // rather than a static snapshot.  Pruning only drops candidates
            // strictly worse than the incumbent beyond TIE_TOL, so the
            // selected (λ*, score) is identical to the snapshot version (and
            // strictly no more aggressive than the previous best_score_hint
            // pruning).  Candidate parallelism is retained here (the joint
            // path has no par_chunks control-obs variant); the joint grids
            // that reach this closure are the coarse per-axis sweeps, which
            // remain correct and merely under-parallelised on tiny grids.
            let atomic_best = AtomicBestScore::new(best_score_hint);
            #[allow(clippy::type_complexity)]
            let results: Vec<(f64, f64, f64, f64, f64, usize, Option<(usize, usize)>)> = grid
                .par_iter()
                .map(|&val| {
                    // Design Issue 38 fix: Convert fixed params upfront.
                    // Fixed params may be f64::INFINITY (Stage 1 uses extreme values).
                    // Grid values (val) are guaranteed finite by C bridge pre-conversion.
                    let f0 = if fixed.0.is_infinite() { 0.0 } else { fixed.0 };
                    let f1 = if fixed.1.is_infinite() { 0.0 } else { fixed.1 };
                    let f2 = if fixed.2.is_infinite() { 1e10 } else { fixed.2 };
                    let (lt, lu, ln) = match param_idx {
                        0 => (val, f1, f2),
                        1 => (f0, val, f2),
                        _ => (f0, f1, val),
                    };

                    // P2.1: Check cache before expensive evaluation
                    let cache_key = (lt.to_bits(), lu.to_bits(), ln.to_bits());
                    if let Ok(cache) = joint_score_cache.lock() {
                        if let Some(&(cached_score, cached_n_valid, cached_failed)) = cache.get(&cache_key) {
                            atomic_best.observe(cached_score);
                            return (val, lt, lu, ln, cached_score, cached_n_valid, cached_failed);
                        }
                    }

                    let (score, n_valid, first_failed) = loocv_score_joint(
                        y,
                        d,
                        &control_obs,
                        lt,
                        lu,
                        ln,
                        treated_periods,
                        max_iter,
                        tol,
                        atomic_best.pruning_bound(),
                        x,
                    );

                    // Cache complete evaluations (not early-terminated)
                    if n_valid == n_attempted || score == f64::INFINITY {
                        if let Ok(mut cache) = joint_score_cache.lock() {
                            cache.insert(cache_key, (score, n_valid, first_failed));
                        }
                    }
                    if n_valid == n_attempted {
                        atomic_best.observe(score);
                    }

                    (val, lt, lu, ln, score, n_valid, first_failed)
                })
                .collect();

            // Incumbent starts at +∞ so any finite candidate wins; the fixed
            // values supply the full triple needed by the tie-breaker.
            let f0_init = if fixed.0.is_infinite() { 0.0 } else { fixed.0 };
            let f1_init = if fixed.1.is_infinite() { 0.0 } else { fixed.1 };
            let f2_init = if fixed.2.is_infinite() { 1e10 } else { fixed.2 };
            let mut best_val = grid.first().copied().unwrap_or(0.0);
            let mut best_lt = f0_init;
            let mut best_lu = f1_init;
            let mut best_ln = f2_init;
            let mut best_score = f64::INFINITY;
            let mut best_n_valid = 0usize;
            let mut best_first_failed: Option<(usize, usize)> = None;
            for (val, lt, lu, ln, score, n_valid, first_failed) in results {
                if better_candidate(
                    (lt, lu, ln, score),
                    (best_lt, best_lu, best_ln, best_score),
                ) {
                    best_val = val;
                    best_lt = lt;
                    best_lu = lu;
                    best_ln = ln;
                    best_score = score;
                    best_n_valid = n_valid;
                    best_first_failed = first_failed;
                }
            }
            (best_val, best_score, best_n_valid, best_first_failed)
        };

    // Stage 1: Univariate searches with fixed values per paper Footnote 2.
    //
    // λ_time search: fix λ_unit = 0, λ_nn = ∞ (factor model off).
    // λ_unit search: fix λ_time = 0, λ_nn = ∞ (factor model off).
    // λ_nn   search: fix λ_time = 0, λ_unit = 0 (uniform weights).
    //
    // Rationale: Footnote 2 fixes the two parameters that are *not* being
    // searched at extrema that do not interfere with the parameter under
    // search.  For λ_nn that means uniform weights (λ_time = 0,
    // λ_unit = 0), because λ_time = ∞ would collapse time weights onto
    // the single target period and mask the factor structure λ_nn is meant
    // to control.  The twostep cycling path (see the twostep branch of
    // `loocv_grid_search`) uses the same seed, keeping the joint and
    // twostep Stage-1 start-points mutually consistent.  Stage-2 cycling
    // only polishes from this seed on a non-convex Q(λ) surface, so a
    // biased initial point would yield a different local optimum.
    // Stage 1: no prior bound available → f64::INFINITY (no pruning).
    let (lt_init, _, _, _) =
        univariate_search(lambda_time_grid, 0, (0.0, 0.0, f64::INFINITY), f64::INFINITY);
    let (ln_init, _, _, _) =
        univariate_search(lambda_nn_grid, 2, (0.0, 0.0, 0.0), f64::INFINITY);
    let (lu_init, _, _, _) =
        univariate_search(lambda_unit_grid, 1, (0.0, 0.0, f64::INFINITY), f64::INFINITY);

    // Stage 2: Cycling refinement (coordinate descent)
    let mut best_lt = lt_init;
    let mut best_lu = lu_init;
    let mut best_ln = ln_init;
    let mut prev_score = f64::INFINITY;

    for _cycle in 0..max_cycles {
        // Optimize λ_unit (fix λ_time, λ_nn)
        // Pass prev_score as the early-termination bound so that candidates
        // whose partial Q(λ) already exceeds the cycling incumbent are pruned.
        let (lu_new, _, _, _) =
            univariate_search(lambda_unit_grid, 1, (best_lt, 0.0, best_ln), prev_score);
        best_lu = lu_new;

        // Optimize λ_time (fix λ_unit, λ_nn)
        let (lt_new, _, _, _) =
            univariate_search(lambda_time_grid, 0, (0.0, best_lu, best_ln), prev_score);
        best_lt = lt_new;

        // Optimize λ_nn (fix λ_unit, λ_time)
        let (ln_new, score, _, _) =
            univariate_search(lambda_nn_grid, 2, (best_lt, best_lu, 0.0), prev_score);
        best_ln = ln_new;

        // Check cycling convergence: |Q_new − Q_old| < 1e-6.
        //
        // This threshold is deliberately separate from the inner-estimation
        // `tol`: Q(λ) varies over a coarse lambda grid, so a hard-coded 1e-6
        // on the aggregate LOOCV score gives predictable cycle termination
        // independent of the user's `tol()` option.
        if (score - prev_score).abs() < 1e-6 {
            break;
        }
        prev_score = score;
    }

    // Final evaluation with best parameters (matching twostep pattern in
    // `loocv_grid_search` / `loocv_grid_search_exhaustive`).
    //
    // Uses the full-diagnostic variant so that the exported `n_valid` counts
    // every successful LOO fit (not "successes up to the first failure").
    // When the selected λ produces no failures — the overwhelmingly common
    // case — this is bit-identical to the short-circuit path.
    //
    // Grid values are guaranteed finite (the C bridge pre-converts Stata
    // missing / ∞ sentinels into effective values); best_lt/lu/ln are drawn
    // from those grid values, so no infinity conversion is needed here.
    let (best_score, final_n_valid, failed_obs) = loocv_score_joint_full_diagnostic(
        y,
        d,
        &control_obs,
        best_lt,
        best_lu,
        best_ln,
        treated_periods,
        max_iter,
        tol,
        x,
    );
    let final_first_failed = failed_obs.first().copied();

    let result = (
        best_lt,
        best_lu,
        best_ln,
        best_score,
        final_n_valid,
        n_attempted,
        final_first_failed,
    );
    (result, lt_init, lu_init, ln_init)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::compute_joint_weights;
    use ndarray::array;

    #[test]
    fn test_check_simultaneous_adoption_accepts_no_treatment() {
        // An all-zero D matrix is degenerate but valid: 0 treated periods.
        let d = array![[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]];
        let res = check_simultaneous_adoption(&d.view());
        assert_eq!(res, Ok(0));
    }

    #[test]
    fn test_check_simultaneous_adoption_accepts_simultaneous() {
        // Units 1 and 2 both enter treatment at period 2; unit 0 is control.
        let d = array![
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ];
        let res = check_simultaneous_adoption(&d.view());
        assert_eq!(res, Ok(2), "n_periods=4, T_1=2 -> treated_periods=2");
    }

    #[test]
    fn test_check_simultaneous_adoption_rejects_staggered_start() {
        // Unit 1 adopts at t=1, unit 2 adopts at t=2 -> staggered.
        let d = array![
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ];
        let res = check_simultaneous_adoption(&d.view());
        assert_eq!(
            res,
            Err(TropError::InvalidDimension),
            "different first-treat periods must be rejected"
        );
    }

    #[test]
    fn test_check_simultaneous_adoption_rejects_treatment_switchoff() {
        // Unit 1 is treated at t=2 but reverts to control at t=3 -> non-absorbing.
        let d = array![
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ];
        let res = check_simultaneous_adoption(&d.view());
        assert_eq!(
            res,
            Err(TropError::InvalidDimension),
            "non-absorbing treatment must be rejected"
        );
    }

    #[test]
    fn test_check_simultaneous_adoption_rejects_pre_t1_treatment() {
        // A single unit has a D=1 pulse before the group-level T_1.
        let d = array![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],  // lone pulse at t=1, i=0
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],  // group T_1 = 3
        ];
        let res = check_simultaneous_adoption(&d.view());
        assert_eq!(res, Err(TropError::InvalidDimension));
    }

    #[test]
    fn test_get_control_observations() {
        let y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let control_mask = array![
            [1u8, 1],
            [1, 1],
            [1, 0] // Unit 1 treated at period 2
        ];

        let obs = get_control_observations(&y.view(), &control_mask.view());

        // Should have 5 control observations (all except (2,1))
        assert_eq!(obs.len(), 5);

        // (2, 1) should not be in the list
        assert!(!obs.contains(&(2, 1)));
    }

    #[test]
    fn test_get_control_observations_returns_all_finite_controls() {
        // Per paper Eq. 5, every finite D=0 cell must enter Q(λ).
        let y = array![[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]];
        let control_mask = array![[1u8, 1, 1], [1, 1, 1], [1, 1, 1]];

        let obs = get_control_observations(&y.view(), &control_mask.view());

        assert_eq!(obs.len(), 9);
    }

    #[test]
    fn test_infinity_conversion() {
        // Test that infinity parameters are correctly converted
        let inf = f64::INFINITY;

        // λ_time/λ_unit = ∞ → 0.0 (uniform weights)
        let time_eff = if inf.is_infinite() { 0.0 } else { inf };
        assert_eq!(time_eff, 0.0);

        // λ_nn = ∞ → 1e10 (disable factor model)
        let nn_eff = if inf.is_infinite() { 1e10 } else { inf };
        assert_eq!(nn_eff, 1e10);
    }

    // ========================================================================
    // Joint LOOCV Property Tests (Story 4.4)
    // ========================================================================

    #[test]
    fn test_loocv_score_joint_basic() {
        // Property 6: LOOCV objective function
        // Q(λ) = Σ (τ̂_{it}^{loocv})²
        let y = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 6.0, 6.0] // Unit 1 treated at period 3
        ];
        let d = array![
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ];

        // Control observations (all except (3,1))
        let control_obs: Vec<(usize, usize)> = vec![
            (0, 0),
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 1),
            (1, 2),
            (2, 0),
            (2, 1),
            (2, 2),
            (3, 0),
            (3, 2),
        ];

        let (score, n_valid, first_failed) = loocv_score_joint(
            &y.view(),
            &d.view(),
            &control_obs,
            0.0,  // lambda_time
            0.0,  // lambda_unit
            1e10, // lambda_nn (no low-rank)
            1,    // treated_periods
            100,
            1e-6,
            f64::INFINITY,
            None,
        );

        // Score should be finite and non-negative
        assert!(score.is_finite(), "Score should be finite");
        assert!(score >= 0.0, "Score should be non-negative");

        // All control observations should be valid
        assert_eq!(
            n_valid,
            control_obs.len(),
            "All control obs should be valid"
        );
        assert!(first_failed.is_none(), "No observation should fail");
    }

    /// P0-2 regression guard: `loocv_score_joint_full_diagnostic` must
    /// produce the same score and `n_valid` as the short-circuit
    /// `loocv_score_joint` on a healthy parameterisation where no control
    /// observation fails.  This pins the "bit-identical when all succeed"
    /// contract that the final-evaluation pass inside
    /// `loocv_grid_search_joint` / `loocv_cycling_search_joint` relies on.
    #[test]
    fn test_loocv_score_joint_full_diagnostic_matches_short_circuit_no_failures() {
        let y = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 6.0, 6.0] // Unit 1 treated at period 3
        ];
        let d = array![
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ];

        let control_obs: Vec<(usize, usize)> = vec![
            (0, 0), (0, 1), (0, 2),
            (1, 0), (1, 1), (1, 2),
            (2, 0), (2, 1), (2, 2),
            (3, 0), (3, 2),
        ];

        let (score_short, n_valid_short, first_failed_short) = loocv_score_joint(
            &y.view(),
            &d.view(),
            &control_obs,
            0.5,  // lambda_time
            0.5,  // lambda_unit
            1e10, // lambda_nn (no low-rank)
            1,    // treated_periods
            100,
            1e-6,
            f64::INFINITY,
            None,
        );

        let (score_full, n_valid_full, failed_full) = loocv_score_joint_full_diagnostic(
            &y.view(),
            &d.view(),
            &control_obs,
            0.5,
            0.5,
            1e10,
            1,
            100,
            1e-6,
            None,
        );

        // When no control fits fail, both paths must agree bit-for-bit on the
        // sum of squared residuals and the success count.
        assert!(
            (score_short - score_full).abs() < 1e-12,
            "full_diagnostic score must match short-circuit when no failures: \
             short={} full={}",
            score_short,
            score_full
        );
        assert_eq!(
            n_valid_short, n_valid_full,
            "n_valid must match between short-circuit and full_diagnostic paths"
        );
        assert!(first_failed_short.is_none(), "short-circuit expected no failure");
        assert!(failed_full.is_empty(), "full_diagnostic expected empty failed_obs list");
    }

    #[test]
    fn test_loocv_grid_search_joint_completeness() {
        // Property 1: Full grid search completeness
        // Should evaluate exactly |λ_time| × |λ_unit| × |λ_nn| combinations
        let y = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 6.0] // Unit 1 treated
        ];
        let d = array![[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0]];
        let control_mask = array![[1u8, 1], [1, 1], [1, 1], [1, 0]];

        let lambda_time_grid = &[0.0, 0.5];
        let lambda_unit_grid = &[0.0, 0.5];
        let lambda_nn_grid = &[1e10]; // No low-rank for speed

        let (
            best_lt,
            best_lu,
            best_ln,
            best_score,
            n_valid,
            n_attempted,
            _,
        ) = loocv_grid_search_joint(
            &y.view(),
            &d.view(),
            &control_mask.view(),
            lambda_time_grid,
            lambda_unit_grid,
            lambda_nn_grid,
            100,
            1e-6,
            None,
        );

        // Best parameters should be from the grid
        assert!(
            lambda_time_grid.contains(&best_lt),
            "Best λ_time should be from grid"
        );
        assert!(
            lambda_unit_grid.contains(&best_lu),
            "Best λ_unit should be from grid"
        );
        assert!(
            lambda_nn_grid.contains(&best_ln),
            "Best λ_nn should be from grid"
        );

        // Score should be finite
        assert!(best_score.is_finite(), "Best score should be finite");

        // n_attempted should equal number of control observations
        assert!(n_attempted > 0, "Should have attempted some observations");
        assert!(n_valid > 0, "Should have valid observations");
    }

    #[test]
    fn test_loocv_joint_determinism() {
        // Full LOOCV is fully deterministic: identical inputs ⇒ identical output.
        let y = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 7.0]
        ];
        let d = array![
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0]
        ];
        let control_mask = array![[1u8, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 0]];

        let lambda_time_grid = &[0.0, 0.5];
        let lambda_unit_grid = &[0.0];
        let lambda_nn_grid = &[1e10];

        let result1 = loocv_grid_search_joint(
            &y.view(),
            &d.view(),
            &control_mask.view(),
            lambda_time_grid,
            lambda_unit_grid,
            lambda_nn_grid,
            100,
            1e-6,
            None,
        );

        let result2 = loocv_grid_search_joint(
            &y.view(),
            &d.view(),
            &control_mask.view(),
            lambda_time_grid,
            lambda_unit_grid,
            lambda_nn_grid,
            100,
            1e-6,
            None,
        );

        assert_eq!(result1.0, result2.0, "λ_time should match");
        assert_eq!(result1.1, result2.1, "λ_unit should match");
        assert_eq!(result1.2, result2.2, "λ_nn should match");
        assert_eq!(result1.3, result2.3, "Score should match");
    }

    #[test]
    fn test_loocv_joint_infinity_handling() {
        // Property 10: Infinity parameter conversion
        // λ=∞ should be converted internally but returned as original value
        let y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 6.0]];
        let d = array![[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0]];
        let control_mask = array![[1u8, 1], [1, 1], [1, 1], [1, 0]];

        // Include infinity in grid
        let lambda_time_grid = &[f64::INFINITY, 0.5];
        let lambda_unit_grid = &[0.0];
        let lambda_nn_grid = &[f64::INFINITY];

        let (best_lt, _best_lu, best_ln, best_score, _, _, _) = loocv_grid_search_joint(
            &y.view(),
            &d.view(),
            &control_mask.view(),
            lambda_time_grid,
            lambda_unit_grid,
            lambda_nn_grid,
            100,
            1e-6,
            None,
        );

        // Score should still be finite (infinity was converted internally)
        assert!(
            best_score.is_finite(),
            "Score should be finite even with ∞ params"
        );

        // If infinity was selected, it should be returned as infinity
        if best_lt.is_infinite() {
            assert!(best_lt == f64::INFINITY, "Should return original ∞ value");
        }
        if best_ln.is_infinite() {
            assert!(best_ln == f64::INFINITY, "Should return original ∞ value");
        }
    }

    #[test]
    fn test_joint_weights_non_normalization() {
        // Property 13: Weight non-normalization
        // Weights should NOT sum to 1 (per paper specification)
        let y = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 7.0]
        ];
        let d = array![
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0]
        ];

        let lambda_time = 0.5;
        let lambda_unit = 0.5;
        let treated_periods = 1;

        let delta = compute_joint_weights(
            &y.view(),
            &d.view(),
            lambda_time,
            lambda_unit,
            treated_periods,
        );

        // Sum of weights
        let weight_sum: f64 = delta.iter().sum();

        // Weights should NOT be normalized to 1
        // With exponential decay, sum will typically be > 1 or < 1
        assert!(
            (weight_sum - 1.0).abs() > 1e-6,
            "Weights should NOT sum to 1, got sum = {}",
            weight_sum
        );

        // All weights should be non-negative
        for &w in delta.iter() {
            assert!(w >= 0.0, "All weights should be non-negative");
        }
    }

    #[test]
    fn test_joint_model_includes_intercept() {
        // Property 14: Model includes global intercept μ
        // Joint model: Y = μ + α + β + L + τD + ε
        // Pseudo treatment effect: τ̂ = Y - μ - α - β - L
        let y = array![
            [10.0, 12.0], // Shifted by constant
            [11.0, 13.0],
            [12.0, 14.0],
            [13.0, 16.0] // Unit 1 treated
        ];
        let d = array![[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0]];

        let control_obs: Vec<(usize, usize)> =
            vec![(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0)];

        // Compute LOOCV score with no low-rank (λ_nn = ∞)
        let (score, n_valid, _) = loocv_score_joint(
            &y.view(),
            &d.view(),
            &control_obs,
            0.0,  // lambda_time (uniform)
            0.0,  // lambda_unit (uniform)
            1e10, // lambda_nn (no low-rank)
            1,    // treated_periods
            100,
            1e-6,
            f64::INFINITY,
            None,
        );

        // Score should be finite (model should fit)
        assert!(score.is_finite(), "Score should be finite with intercept");
        assert!(n_valid == control_obs.len(), "All obs should be valid");

        // The score should be small for well-structured data
        // (intercept absorbs the constant shift)
        assert!(
            score < 100.0,
            "Score should be reasonable with intercept, got {}",
            score
        );
    }

    // ========================================================================
    // Story 4.5: Diagnostic Information Tests
    // ========================================================================

    #[test]
    fn test_loocv_twostep_diagnostic_completeness() {
        // Property 7: Diagnostic Information Completeness
        // All diagnostic fields should be populated correctly
        let y = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 7.0]
        ];
        let d = array![
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0]
        ];
        let control_mask = array![[1u8, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 0]];
        let time_dist = array![[0i64, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]];

        let lambda_time_grid = &[0.0, 0.5];
        let lambda_unit_grid = &[0.0];
        let lambda_nn_grid = &[1e10];

        let (_, _, _, best_score, n_valid, n_attempted, _) = loocv_grid_search(
            &y.view(),
            &d.view(),
            &control_mask.view(),
            &time_dist.view(),
            lambda_time_grid,
            lambda_unit_grid,
            lambda_nn_grid,
            100,
            1e-6,
            None,
        ).unwrap();

        // Verify diagnostic constraints
        assert!(
            best_score.is_finite() && best_score >= 0.0,
            "Score should be finite and non-negative"
        );
        assert!(
            n_valid <= n_attempted,
            "n_valid ({}) should be <= n_attempted ({})",
            n_valid,
            n_attempted
        );
        // Without subsampling, n_attempted equals the total control count.
        assert_eq!(n_attempted, 11, "n_attempted should cover every D=0 cell");
    }

    #[test]
    fn test_loocv_twostep_uses_all_control_observations() {
        // Paper Eq. 5 requires Q(λ) to sum over every D=0 cell.
        // n_attempted must equal the total finite control count, with no
        // subsampling path available.
        let y = array![
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 3.0, 4.0, 5.0, 6.0],
            [3.0, 4.0, 5.0, 6.0, 7.0],
            [4.0, 5.0, 6.0, 7.0, 9.0]
        ];
        let d = array![
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0]
        ];
        let control_mask = array![
            [1u8, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0]
        ];
        let time_dist = array![[0i64, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]];

        let lambda_time_grid = &[0.0];
        let lambda_unit_grid = &[0.0];
        let lambda_nn_grid = &[1e10];

        let (_, _, _, _, n_valid, n_attempted, _) = loocv_grid_search(
            &y.view(),
            &d.view(),
            &control_mask.view(),
            &time_dist.view(),
            lambda_time_grid,
            lambda_unit_grid,
            lambda_nn_grid,
            100,
            1e-6,
            None,
        ).unwrap();

        assert_eq!(n_attempted, 19, "Must enumerate all D=0 cells");
        assert!(n_valid <= n_attempted, "n_valid should be <= n_attempted");
    }

    // ========================================================================
    // LOOCV Score Precision Tests (Story 5.1, Task 3.6)
    // ========================================================================

    #[test]
    fn test_loocv_score_precision() {
        // Test LOOCV score numerical precision (tolerance < 1e-8)
        // Run same computation twice and verify identical results
        let y = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 7.0]
        ];
        let d = array![
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0]
        ];
        let control_mask = array![[1u8, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 0]];
        let time_dist = array![[0i64, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]];

        let control_obs: Vec<(usize, usize)> = vec![
            (0, 0),
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 1),
            (1, 2),
            (2, 0),
            (2, 1),
            (2, 2),
            (3, 0),
            (3, 1),
        ];

        // Compute score twice
        let dist_cache = UnitDistanceCache::build(&y.view(), &d.view());
        let (score1, n_valid1, _) = loocv_score_for_params(
            &y.view(),
            &d.view(),
            &control_mask.view(),
            &time_dist.view(),
            &dist_cache,
            &control_obs,
            0.5, // lambda_time
            0.5, // lambda_unit
            0.1, // lambda_nn
            100,
            1e-6,
            f64::INFINITY,
            None,
        );

        let (score2, n_valid2, _) = loocv_score_for_params(
            &y.view(),
            &d.view(),
            &control_mask.view(),
            &time_dist.view(),
            &dist_cache,
            &control_obs,
            0.5,
            0.5,
            0.1,
            100,
            1e-6,
            f64::INFINITY,
            None,
        );

        // Scores should be identical (deterministic computation)
        let diff = (score1 - score2).abs();
        assert!(
            diff < 1e-8,
            "LOOCV score should be deterministic: score1={}, score2={}, diff={}",
            score1,
            score2,
            diff
        );
        assert_eq!(n_valid1, n_valid2, "n_valid should be identical");
    }

    #[test]
    fn test_loocv_score_finite_and_nonnegative() {
        // Test that LOOCV scores are always finite and non-negative
        let y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 6.0]];
        let d = array![[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0]];
        let control_mask = array![[1u8, 1], [1, 1], [1, 1], [1, 0]];
        let time_dist = array![[0i64, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]];

        let control_obs: Vec<(usize, usize)> =
            vec![(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0)];

        // Test with various parameter combinations
        let test_params = vec![
            (0.0, 0.0, 0.0),
            (0.5, 0.0, 0.0),
            (0.0, 0.5, 0.0),
            (0.0, 0.0, 0.1),
            (0.5, 0.5, 0.1),
            (1.0, 1.0, 1.0),
        ];

        let dist_cache = UnitDistanceCache::build(&y.view(), &d.view());
        for (lt, lu, ln) in test_params {
            let (score, n_valid, first_failed) = loocv_score_for_params(
                &y.view(),
                &d.view(),
                &control_mask.view(),
                &time_dist.view(),
                &dist_cache,
                &control_obs,
                lt,
                lu,
                ln,
                100,
                1e-6,
                f64::INFINITY,
                None,
            );

            if first_failed.is_none() {
                assert!(
                    score.is_finite(),
                    "Score should be finite for λ_t={}, λ_u={}, λ_nn={}",
                    lt,
                    lu,
                    ln
                );
                assert!(
                    score >= 0.0,
                    "Score should be non-negative for λ_t={}, λ_u={}, λ_nn={}, got {}",
                    lt,
                    lu,
                    ln,
                    score
                );
                assert!(
                    n_valid > 0,
                    "Should have valid observations for λ_t={}, λ_u={}, λ_nn={}",
                    lt,
                    lu,
                    ln
                );
            }
        }
    }

    /// The diagnostic variant agrees with the short-circuit variant on
    /// both score and n_valid when no observation fails.  When at least
    /// one observation fails, the diagnostic variant still iterates
    /// through every observation and the reported `failed_obs` contains
    /// `first_failed` as its first entry.
    #[test]
    fn test_loocv_score_diagnostic_matches_short_circuit() {
        let y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 6.0]];
        let d = array![[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0]];
        let control_mask = array![[1u8, 1], [1, 1], [1, 1], [1, 0]];
        let time_dist = array![[0i64, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]];
        let control_obs: Vec<(usize, usize)> =
            vec![(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0)];

        let dist_cache = UnitDistanceCache::build(&y.view(), &d.view());

        // Healthy parameterisation — expect no failures and matching scores.
        let (score_short, n_valid_short, first_failed_short) = loocv_score_for_params(
            &y.view(),
            &d.view(),
            &control_mask.view(),
            &time_dist.view(),
            &dist_cache,
            &control_obs,
            0.5, 0.5, 0.1, 100, 1e-6,
            f64::INFINITY,
            None,
        );
        let (score_full, n_valid_full, failed_full) = loocv_score_for_params_full_diagnostic(
            &y.view(),
            &d.view(),
            &control_mask.view(),
            &time_dist.view(),
            &dist_cache,
            &control_obs,
            0.5, 0.5, 0.1, 100, 1e-6,
            None,
        );

        assert!(first_failed_short.is_none(), "expected zero failures at healthy λ");
        assert!(failed_full.is_empty(), "diagnostic variant should report no failures");
        assert_eq!(n_valid_short, n_valid_full);
        assert!((score_short - score_full).abs() < 1e-12);
    }

    #[test]
    fn test_loocv_parameter_selection_in_grid() {
        // Test that selected parameters are always from the provided grid
        let y = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 7.0]
        ];
        let d = array![
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0]
        ];
        let control_mask = array![[1u8, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 0]];
        let time_dist = array![[0i64, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]];

        let lambda_time_grid = vec![0.0, 0.5, 1.0, 2.0];
        let lambda_unit_grid = vec![0.0, 0.5, 1.0];
        let lambda_nn_grid = vec![0.0, 0.1, 1.0];

        let (best_lt, best_lu, best_ln, _, _, _, _) = loocv_grid_search(
            &y.view(),
            &d.view(),
            &control_mask.view(),
            &time_dist.view(),
            &lambda_time_grid,
            &lambda_unit_grid,
            &lambda_nn_grid,
            100,
            1e-6,
            None,
        ).unwrap();

        assert!(
            lambda_time_grid.contains(&best_lt),
            "Selected λ_time={} should be in grid {:?}",
            best_lt,
            lambda_time_grid
        );
        assert!(
            lambda_unit_grid.contains(&best_lu),
            "Selected λ_unit={} should be in grid {:?}",
            best_lu,
            lambda_unit_grid
        );
        assert!(
            lambda_nn_grid.contains(&best_ln),
            "Selected λ_nn={} should be in grid {:?}",
            best_ln,
            lambda_nn_grid
        );
    }

    // ========================================================================
    // Phase 1-A2: tie-breaker (`better_candidate`) property tests
    // ========================================================================

    #[test]
    fn test_better_candidate_prefers_strictly_lower_score() {
        // Outside `TIE_TOL`, the lower score always wins regardless of the
        // structural coordinates.
        let new = (0.1, 0.1, 10.0, 1.0);        // low λ_nn but lower score
        let best = (1.0, 1.0, 0.01, 2.0);        // high λ_nn but higher score
        assert!(better_candidate(new, best));
        assert!(!better_candidate(best, new));
    }

    #[test]
    fn test_better_candidate_score_tie_prefers_larger_lambda_nn() {
        // Scores equal within `TIE_TOL` → prefer larger λ_nn.
        let best = (0.5, 0.5, 0.01, 1.234567);
        let new = (0.5, 0.5, 10.0, 1.234567 + TIE_TOL / 2.0);
        assert!(better_candidate(new, best));
    }

    #[test]
    fn test_better_candidate_score_and_nn_tie_prefers_smaller_lambda_time() {
        // Score + λ_nn tied → prefer smaller λ_time.
        let best = (2.0, 0.5, 0.1, 0.5);
        let new = (0.1, 0.5, 0.1, 0.5);
        assert!(better_candidate(new, best));
    }

    #[test]
    fn test_better_candidate_full_coord_tie_prefers_smaller_lambda_unit() {
        // Score + λ_nn + λ_time tied → prefer smaller λ_unit.
        let best = (0.5, 3.0, 0.1, 0.5);
        let new = (0.5, 0.1, 0.1, 0.5);
        assert!(better_candidate(new, best));
    }

    #[test]
    fn test_better_candidate_exact_duplicate_rejected() {
        // Identical triples → do NOT replace; this guarantees the first
        // encountered candidate is kept and parallel reduction is
        // deterministic.
        let triple = (0.5, 0.5, 0.1, 1.0);
        assert!(!better_candidate(triple, triple));
    }

    #[test]
    fn test_better_candidate_handles_infinite_incumbent() {
        // Finite candidate should always beat a +∞ incumbent (matches the
        // previous `< f64::INFINITY` behaviour for the initial value).
        let best = (0.0, 0.0, 0.0, f64::INFINITY);
        let new = (1.0, 1.0, 0.01, 42.0);
        assert!(better_candidate(new, best));
        // And a +∞ candidate should never beat a finite incumbent.
        assert!(!better_candidate(best, new));
    }

    #[test]
    fn test_better_candidate_nan_score_never_wins() {
        // NaN scores arise from degenerate LOOCV fits; they must not
        // displace a finite incumbent nor be chosen as the winner.
        let best = (0.5, 0.5, 0.1, 1.0);
        let new = (0.1, 0.1, 10.0, f64::NAN);
        assert!(!better_candidate(new, best));
    }

    // ========================================================================
    // Phase 1-A1: twostep exhaustive search property tests
    // ========================================================================

    #[test]
    fn test_loocv_grid_search_exhaustive_twostep_agrees_with_cycling() {
        // On a small well-conditioned panel, cycling and exhaustive should
        // select the same (λ_time, λ_unit, λ_nn) triple up to TIE_TOL.
        let y = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 7.0]
        ];
        let d = array![
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0]
        ];
        let control_mask = array![[1u8, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 0]];
        let time_dist = array![[0i64, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]];

        let lambda_time_grid = vec![0.0, 0.1, 0.5, 1.0];
        let lambda_unit_grid = vec![0.0, 0.1, 0.5];
        let lambda_nn_grid = vec![0.01, 0.1, 1.0];

        let cycling = loocv_grid_search(
            &y.view(),
            &d.view(),
            &control_mask.view(),
            &time_dist.view(),
            &lambda_time_grid,
            &lambda_unit_grid,
            &lambda_nn_grid,
            100,
            1e-6,
            None,
        ).unwrap();
        let exhaustive = loocv_grid_search_exhaustive(
            &y.view(),
            &d.view(),
            &control_mask.view(),
            &time_dist.view(),
            &lambda_time_grid,
            &lambda_unit_grid,
            &lambda_nn_grid,
            100,
            1e-6,
            None,
        ).unwrap();

        // Exhaustive is guaranteed to find the global grid minimum; cycling
        // may converge to a local minimum.  In both cases the selected λ
        // must come from the supplied grid.
        for (name, val, grid) in [
            ("λ_time exhaustive", exhaustive.0, &lambda_time_grid),
            ("λ_time cycling", cycling.0, &lambda_time_grid),
            ("λ_unit exhaustive", exhaustive.1, &lambda_unit_grid),
            ("λ_unit cycling", cycling.1, &lambda_unit_grid),
            ("λ_nn exhaustive", exhaustive.2, &lambda_nn_grid),
            ("λ_nn cycling", cycling.2, &lambda_nn_grid),
        ] {
            assert!(grid.contains(&val), "{name} value {val} not in grid");
        }

        // Exhaustive score must be ≤ cycling score (up to numerical jitter).
        // This is the entire reason for having the exhaustive path.
        assert!(
            exhaustive.3 <= cycling.3 + 1e-9,
            "exhaustive score {} should be ≤ cycling score {}",
            exhaustive.3,
            cycling.3,
        );

        // Diagnostic counters should match since both paths enumerate the
        // same set of control observations for Q(λ).
        assert_eq!(exhaustive.5, cycling.5, "n_attempted should match");
    }

    #[test]
    fn test_loocv_grid_search_exhaustive_twostep_determinism() {
        // Identical inputs ⇒ identical output, independent of rayon worker
        // scheduling (the tie-breaker removes any order dependence).
        let y = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 7.0]
        ];
        let d = array![
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0]
        ];
        let control_mask = array![[1u8, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 0]];
        let time_dist = array![[0i64, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]];

        let lambda_time_grid = vec![0.0, 0.5];
        let lambda_unit_grid = vec![0.0, 0.5];
        let lambda_nn_grid = vec![0.01, 0.1, 1.0];

        let r1 = loocv_grid_search_exhaustive(
            &y.view(),
            &d.view(),
            &control_mask.view(),
            &time_dist.view(),
            &lambda_time_grid,
            &lambda_unit_grid,
            &lambda_nn_grid,
            100,
            1e-6,
            None,
        ).unwrap();
        let r2 = loocv_grid_search_exhaustive(
            &y.view(),
            &d.view(),
            &control_mask.view(),
            &time_dist.view(),
            &lambda_time_grid,
            &lambda_unit_grid,
            &lambda_nn_grid,
            100,
            1e-6,
            None,
        ).unwrap();

        assert_eq!(r1.0, r2.0, "λ_time must match");
        assert_eq!(r1.1, r2.1, "λ_unit must match");
        assert_eq!(r1.2, r2.2, "λ_nn must match");
        assert!((r1.3 - r2.3).abs() < 1e-12, "score must be bit-identical");
    }

    #[test]
    fn test_loocv_joint_score_precision() {
        // Test Joint LOOCV score precision (tolerance < 1e-8)
        let y = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 6.0, 6.0]
        ];
        let d = array![
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ];

        let control_obs: Vec<(usize, usize)> = vec![
            (0, 0),
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 1),
            (1, 2),
            (2, 0),
            (2, 1),
            (2, 2),
            (3, 0),
            (3, 2),
        ];

        // Compute score twice
        let (score1, _, _) = loocv_score_joint(
            &y.view(),
            &d.view(),
            &control_obs,
            0.5,
            0.5,
            1e10,
            1,
            100,
            1e-6,
            f64::INFINITY,
            None,
        );

        let (score2, _, _) = loocv_score_joint(
            &y.view(),
            &d.view(),
            &control_obs,
            0.5,
            0.5,
            1e10,
            1,
            100,
            1e-6,
            f64::INFINITY,
            None,
        );

        let diff = (score1 - score2).abs();
        assert!(
            diff < 1e-8,
            "Joint LOOCV score should be deterministic: diff={}",
            diff
        );
    }
}

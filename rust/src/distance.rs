//! Unit distance computation for the TROP estimator.
//!
//! Computes pairwise unit distances as defined in Equation (3):
//!
//!   dist^unit_{-t}(j, i) = ( Σ_u 1{u≠t}(1-W_iu)(1-W_ju)(Y_iu - Y_ju)^2
//!                            / Σ_u 1{u≠t}(1-W_iu)(1-W_ju) )^{1/2}
//!
//! These distances parameterize the exponential unit weights
//!   ω_j^{i,t}(λ) = exp(-λ_unit · dist^unit_{-t}(j, i))
//! which down-weight control units whose outcome trajectories diverge from
//! the target unit i over jointly observed control periods.

use ndarray::{Array2, ArrayView1, ArrayView2};
use rayon::prelude::*;

/// Minimum row count per parallel chunk.
/// Avoids excessive thread-scheduling overhead when the unit dimension is small.
const MIN_CHUNK_SIZE: usize = 16;

/// Computes the full pairwise unit distance matrix (no period exclusion).
///
/// For each pair (j, i) the RMSE distance is computed over all periods where
/// both units are untreated (W = 0) and have finite outcomes:
///
///   dist(j, i) = ( Σ_u (1-W_iu)(1-W_ju)(Y_iu - Y_ju)^2
///                  / Σ_u (1-W_iu)(1-W_ju) )^{1/2}
///
/// This is the global variant of Equation (3) without the 1{u ≠ t} exclusion,
/// useful for diagnostics and pre-screening.
///
/// # Arguments
/// * `y` - Outcome matrix, shape (T × N), column-per-unit.
/// * `d` - Treatment indicator matrix, shape (T × N), 0 = control, 1 = treated.
///
/// # Returns
/// Symmetric distance matrix (N × N). Diagonal entries are zero; pairs with
/// no jointly observed control periods receive infinity.
pub fn compute_unit_distance_matrix_internal(
    y: &ArrayView2<f64>,
    d: &ArrayView2<f64>,
) -> Array2<f64> {
    let n_periods = y.nrows();
    let n_units = y.ncols();

    // Validity mask: a period is usable for unit i iff W_{it} = 0 and Y_{it} is finite.
    let valid_mask: Array2<bool> = Array2::from_shape_fn((n_periods, n_units), |(t, i)| {
        d[[t, i]] == 0.0 && y[[t, i]].is_finite()
    });

    // Replace invalid entries with 0.0 so that branchless mask multiplication
    // (0.0 * diff * diff) yields 0.0 rather than NaN (IEEE 754: 0.0 * NaN = NaN).
    let y_masked: Array2<f64> = Array2::from_shape_fn((n_periods, n_units), |(t, i)| {
        if valid_mask[[t, i]] {
            y[[t, i]]
        } else {
            0.0
        }
    });

    // Transpose to (N × T) so each row is one unit's time series (row-major access).
    let y_t = y_masked.t();
    let valid_t = valid_mask.t();

    // Initialize to infinity; finite values are filled below.
    let mut dist_matrix = Array2::<f64>::from_elem((n_units, n_units), f64::INFINITY);

    // Set diagonal to zero (distance from a unit to itself is trivially zero).
    for i in 0..n_units {
        dist_matrix[[i, i]] = 0.0;
    }

    // Upper triangle computed in parallel; lower triangle mirrored (symmetry).
    let row_results: Vec<Vec<(usize, f64)>> = (0..n_units)
        .into_par_iter()
        .with_min_len(MIN_CHUNK_SIZE)
        .map(|j| {
            let mut pairs = Vec::with_capacity(n_units - j - 1);

            for i in (j + 1)..n_units {
                let dist = compute_pair_distance(
                    &y_t.row(j),
                    &y_t.row(i),
                    &valid_t.row(j),
                    &valid_t.row(i),
                );
                pairs.push((i, dist));
            }

            pairs
        })
        .collect();

    // Fill the full matrix from the upper-triangle results.
    for (j, pairs) in row_results.into_iter().enumerate() {
        for (i, dist) in pairs {
            dist_matrix[[j, i]] = dist;
            dist_matrix[[i, j]] = dist;
        }
    }

    dist_matrix
}

/// Computes the RMSE distance between two units over jointly valid periods.
///
/// A period is valid for the pair (j, i) when both validity flags are true.
/// Returns infinity when no valid period exists.
///
/// Implementation uses branchless mask multiplication (bool → 0.0/1.0) to
/// eliminate conditional branches in the hot loop, enabling LLVM auto-vectorization.
#[inline]
pub fn compute_pair_distance(
    y_j: &ArrayView1<f64>,
    y_i: &ArrayView1<f64>,
    valid_j: &ArrayView1<bool>,
    valid_i: &ArrayView1<bool>,
) -> f64 {
    let n_periods = y_j.len();

    // Fast path: if all arrays are contiguous in memory, iterate over slices
    // for better cache locality and auto-vectorization.
    if let (Some(sj), Some(si), Some(vj), Some(vi)) =
        (y_j.as_slice(), y_i.as_slice(), valid_j.as_slice(), valid_i.as_slice())
    {
        let mut sum_sq = 0.0_f64;
        let mut n_valid = 0.0_f64;

        for t in 0..n_periods {
            // Convert bool pair to a 0.0/1.0 mask — branchless.
            let mask = (unsafe { *vj.get_unchecked(t) } as u8 as f64)
                * (unsafe { *vi.get_unchecked(t) } as u8 as f64);
            let diff = unsafe { *si.get_unchecked(t) - *sj.get_unchecked(t) };
            sum_sq += mask * diff * diff;
            n_valid += mask;
        }

        return if n_valid > 0.0 {
            (sum_sq / n_valid).sqrt()
        } else {
            f64::INFINITY
        };
    }

    // Fallback: non-contiguous views — use branchless mask on indexed access.
    let mut sum_sq = 0.0_f64;
    let mut n_valid = 0.0_f64;

    for t in 0..n_periods {
        let mask = (valid_j[t] as u8 as f64) * (valid_i[t] as u8 as f64);
        let diff = y_i[t] - y_j[t];
        sum_sq += mask * diff * diff;
        n_valid += mask;
    }

    if n_valid > 0.0 {
        (sum_sq / n_valid).sqrt()
    } else {
        f64::INFINITY
    }
}

/// Computes the observation-specific unit distance, excluding the target period.
///
/// Implements Equation (3) with the leave-one-period-out indicator 1{u ≠ t}:
///
///   dist^unit_{-t}(j, i) = ( Σ_{u≠t} (1-W_iu)(1-W_ju)(Y_iu - Y_ju)^2
///                            / Σ_{u≠t} (1-W_iu)(1-W_ju) )^{1/2}
///
/// Excluding the target period prevents the distance used to construct
/// weights for observation (i, t) from depending on the outcome at time t.
///
/// # Arguments
/// * `y` - Outcome matrix, shape (T × N).
/// * `d` - Treatment indicator matrix, shape (T × N).
/// * `j` - Source (donor) unit index.
/// * `i` - Target unit index.
/// * `target_period` - Period index to exclude from the summation.
///
/// # Returns
/// RMSE distance over eligible periods, or infinity if none exist.
pub fn compute_unit_distance_for_obs(
    y: &ArrayView2<f64>,
    d: &ArrayView2<f64>,
    j: usize,
    i: usize,
    target_period: usize,
) -> f64 {
    let n_periods = y.nrows();
    let mut sum_sq = 0.0;
    let mut n_valid = 0usize;

    for t in 0..n_periods {
        // Leave out the target period (1{u ≠ t} in Equation (3)).
        if t == target_period {
            continue;
        }
        // Both units must be in the control state with finite outcomes.
        if d[[t, i]] == 0.0 && d[[t, j]] == 0.0 && y[[t, i]].is_finite() && y[[t, j]].is_finite() {
            let diff = y[[t, i]] - y[[t, j]];
            sum_sq += diff * diff;
            n_valid += 1;
        }
    }

    if n_valid > 0 {
        (sum_sq / n_valid as f64).sqrt()
    } else {
        f64::INFINITY
    }
}

/// On-demand pairwise unit distance cache with O(N·T) memory footprint.
///
/// The per-observation distance in Equation (3),
///
/// ```text
/// dist_{-t}(j, i) = sqrt( Σ_{u ≠ t} 1_{valid(u,i,j)} · (Y_{ui} − Y_{uj})²
///                         / Σ_{u ≠ t} 1_{valid(u,i,j)} )
/// ```
///
/// with `valid(u, i, j) = (D_{ui} = 0) ∧ (D_{uj} = 0) ∧ Y_{ui}, Y_{uj}
/// finite`, is computed on demand by iterating over T periods and
/// optionally excluding a single target period.
///
/// Memory usage is O(N·T) (validity mask + Y copy) instead of the previous
/// O(N² + N·T) (pairwise sums). Build time drops from O(N²·T) to O(N·T).
/// Each query costs O(T) instead of O(1); this is acceptable because the
/// weight computation is itself O(N·T) per observation.
#[derive(Clone, Debug)]
pub struct UnitDistanceCache {
    /// Outcome matrix, shape (T × N). Stored as owned copy so that
    /// `compute_distance` does not need a Y reference from the caller.
    y: Array2<f64>,
    /// Per-period validity mask, column-major: `unit_valid[t + i * n_periods]`.
    /// True when both `D[t, i] = 0` and `Y[t, i]` is finite.
    unit_valid: Vec<bool>,
    n_units: usize,
    n_periods: usize,
}

impl UnitDistanceCache {
    /// Build the cache from the full panel.
    ///
    /// Only the per-unit validity mask and a copy of Y are stored.
    /// Cost is O(N·T) arithmetic and memory — down from O(N²·T).
    pub fn build(y: &ArrayView2<f64>, d: &ArrayView2<f64>) -> Self {
        let n_periods = y.nrows();
        let n_units = y.ncols();

        // Per-unit validity mask: cell (t, i) is usable iff control & finite.
        let mut unit_valid = vec![false; n_periods * n_units];
        for i in 0..n_units {
            for t in 0..n_periods {
                unit_valid[t + i * n_periods] =
                    d[[t, i]] == 0.0 && y[[t, i]].is_finite();
            }
        }

        // Replace non-finite values with 0.0 in the stored Y copy so that
        // branchless mask multiplication yields 0.0 (not NaN) for invalid periods.
        let y_sanitized = Array2::from_shape_fn((n_periods, n_units), |(t, i)| {
            let val = y[[t, i]];
            if val.is_finite() { val } else { 0.0 }
        });

        UnitDistanceCache {
            y: y_sanitized,
            unit_valid,
            n_units,
            n_periods,
        }
    }

    /// Compute the Eq. (3) distance on demand in O(T).
    ///
    /// Iterates over all periods, optionally excluding `exclude_t`.
    /// For each period u where both units are valid (control + finite)
    /// and u ≠ exclude_t, accumulates (Y_{ui} − Y_{uj})² and the count.
    ///
    /// Returns `sqrt(sum_sq / count)`, or `f64::INFINITY` when count == 0.
    ///
    /// Symmetry guarantee: `compute_distance(i, j, t) == compute_distance(j, i, t)`.
    ///
    /// Uses branchless mask multiplication for the validity check to enable
    /// LLVM auto-vectorization of the inner loop.
    #[inline]
    pub fn compute_distance(
        &self,
        i: usize,
        j: usize,
        exclude_t: Option<usize>,
    ) -> f64 {
        debug_assert!(i < self.n_units && j < self.n_units);
        if i == j {
            return 0.0;
        }

        let n_periods = self.n_periods;
        let i_offset = i * n_periods;
        let j_offset = j * n_periods;

        let mut sum_sq = 0.0_f64;
        let mut count = 0.0_f64;

        // Unwrap exclude_t once to avoid repeated Option comparison inside the loop.
        let exclude = exclude_t.unwrap_or(usize::MAX);

        for t in 0..n_periods {
            // Branchless: exclude_mask is 0.0 when t == exclude, 1.0 otherwise.
            let exclude_mask = (t != exclude) as u8 as f64;
            // Branchless: valid_mask is 1.0 when both units are valid at t.
            let valid_mask = (self.unit_valid[t + i_offset] as u8 as f64)
                * (self.unit_valid[t + j_offset] as u8 as f64);
            let mask = exclude_mask * valid_mask;
            let diff = self.y[[t, i]] - self.y[[t, j]];
            sum_sq += mask * diff * diff;
            count += mask;
        }

        if count > 0.0 {
            (sum_sq / count).sqrt()
        } else {
            f64::INFINITY
        }
    }

    /// Backward-compatible wrapper around [`compute_distance`].
    ///
    /// The `y` parameter is retained for API compatibility but is **not used** —
    /// the internally stored copy of Y is consulted instead.
    #[inline]
    pub fn distance(
        &self,
        _y: &ArrayView2<f64>,
        j: usize,
        i: usize,
        target_period: usize,
    ) -> f64 {
        self.compute_distance(j, i, Some(target_period))
    }

    /// Returns `true` iff `(target_period, unit)` is in the control state
    /// with a finite outcome.  Convenience accessor for the weights layer,
    /// which needs the same flag when deciding whether a unit participates
    /// in the kernel at a given period.
    #[inline]
    pub fn unit_is_control_at(&self, target_period: usize, unit: usize) -> bool {
        self.unit_valid[target_period + unit * self.n_periods]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_compute_pair_distance_constant_diff() {
        // Constant difference of 0.5 across all periods; RMSE should equal 0.5.
        let y_j = array![1.0, 2.0, 3.0, 4.0];
        let y_i = array![1.5, 2.5, 3.5, 4.5];
        let valid_j = array![true, true, true, true];
        let valid_i = array![true, true, true, true];

        let dist =
            compute_pair_distance(&y_j.view(), &y_i.view(), &valid_j.view(), &valid_i.view());

        // RMSE of a constant difference 0.5 is exactly 0.5.
        assert!((dist - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_compute_pair_distance_partial_overlap() {
        // Only period 0 has both validity flags true; RMSE = |0.5| = 0.5.
        let y_j = array![1.0, 2.0, 3.0, 4.0];
        let y_i = array![1.5, 2.5, 3.5, 4.5];
        let valid_j = array![true, true, false, false];
        let valid_i = array![true, false, true, false];

        // Single overlapping period yields RMSE = |0.5| = 0.5.
        let dist =
            compute_pair_distance(&y_j.view(), &y_i.view(), &valid_j.view(), &valid_i.view());

        assert!((dist - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_compute_pair_distance_no_overlap() {
        // Disjoint validity sets yield no overlapping period; expect infinity.
        let y_j = array![1.0, 2.0, 3.0, 4.0];
        let y_i = array![1.5, 2.5, 3.5, 4.5];
        let valid_j = array![true, true, false, false];
        let valid_i = array![false, false, true, true];

        let dist =
            compute_pair_distance(&y_j.view(), &y_i.view(), &valid_j.view(), &valid_i.view());

        assert!(dist.is_infinite());
    }

    #[test]
    fn test_unit_distance_matrix_diagonal_zero() {
        // Self-distance is zero for every unit.
        let y = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let d = array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];

        let dist = compute_unit_distance_matrix_internal(&y.view(), &d.view());

        for i in 0..3 {
            assert!((dist[[i, i]]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_unit_distance_matrix_symmetric() {
        // dist(j, i) == dist(i, j) for all pairs.
        let y = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let d = array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];

        let dist = compute_unit_distance_matrix_internal(&y.view(), &d.view());

        for i in 0..3 {
            for j in 0..3 {
                assert!((dist[[i, j]] - dist[[j, i]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_unit_distance_for_obs_excludes_target() {
        // Panel: T=4, N=2, all control. Constant difference 0.5.
        let y = array![[1.0, 1.5], [2.0, 2.5], [3.0, 3.5], [4.0, 4.5]];
        let d = array![[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]];

        // Excluding period 0: uses periods 1, 2, 3. RMSE = 0.5.
        let dist_ex0 = compute_unit_distance_for_obs(&y.view(), &d.view(), 0, 1, 0);
        assert!((dist_ex0 - 0.5).abs() < 1e-10);

        // Excluding period 2: uses periods 0, 1, 3. RMSE = 0.5.
        let dist_ex2 = compute_unit_distance_for_obs(&y.view(), &d.view(), 0, 1, 2);
        assert!((dist_ex2 - 0.5).abs() < 1e-10);

        // Out-of-range exclusion index: all four periods used. RMSE = 0.5.
        let dist_full = compute_unit_distance_for_obs(&y.view(), &d.view(), 0, 1, 999);
        assert!((dist_full - 0.5).abs() < 1e-10);
    }

    /// Numerical regression test against independently computed reference values.
    /// Panel: T=5, N=4; unit 2 treated at periods 3-4.
    #[test]
    fn test_distance_numerical_reference() {
        // Y matrix: T=5, N=4, seeded random data with column offsets.
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

        // Reference values computed independently to 15 significant digits.
        let dist_02_ex3 = compute_unit_distance_for_obs(&y.view(), &d.view(), 0, 2, 3);
        assert!(
            (dist_02_ex3 - 1.848595992029445).abs() < 1e-10,
            "dist(0, 2, ex_t=3): got {:.15}, expected 1.848595992029445",
            dist_02_ex3
        );

        let dist_12_ex3 = compute_unit_distance_for_obs(&y.view(), &d.view(), 1, 2, 3);
        assert!(
            (dist_12_ex3 - 1.555771549669933).abs() < 1e-10,
            "dist(1, 2, ex_t=3): got {:.15}, expected 1.555771549669933",
            dist_12_ex3
        );

        let dist_01_ex2 = compute_unit_distance_for_obs(&y.view(), &d.view(), 0, 1, 2);
        assert!(
            (dist_01_ex2 - 1.259591110421568).abs() < 1e-10,
            "dist(0, 1, ex_t=2): got {:.15}, expected 1.259591110421568",
            dist_01_ex2
        );
    }

    #[test]
    fn test_unit_distance_for_obs_with_treated_periods() {
        // Unit 0: all control. Unit 1: treated from period 2 onward.
        // Only periods 0 and 1 are jointly control; both have diff = 0.5.
        let y = array![[1.0, 1.5], [2.0, 2.5], [3.0, 3.5], [4.0, 4.5]];
        let d = array![
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0]
        ];

        // No period excluded (out-of-range index): periods 0, 1 used.
        let dist = compute_unit_distance_for_obs(&y.view(), &d.view(), 0, 1, 999);
        assert!((dist - 0.5).abs() < 1e-10);

        // Exclude period 0: only period 1 remains.
        let dist_ex0 = compute_unit_distance_for_obs(&y.view(), &d.view(), 0, 1, 0);
        assert!((dist_ex0 - 0.5).abs() < 1e-10);

        // Exclude period 1: only period 0 remains.
        let dist_ex1 = compute_unit_distance_for_obs(&y.view(), &d.view(), 0, 1, 1);
        assert!((dist_ex1 - 0.5).abs() < 1e-10);
    }

    /// UnitDistanceCache produces the same distance as the direct computation
    /// for every (target_period, i, j) triple on a fixed reference panel.
    #[test]
    fn test_unit_distance_cache_matches_direct() {
        let y = array![
            [0.496714153011233,  0.361735698828815,  1.647688538100692,  3.023029856408026],
            [-0.234153374723336, 0.265863043050819,  2.579212815507391,  2.267434729152909],
            [-0.469474385934952, 1.042560043585965,  0.536582307187538,  1.034270246429743],
            [0.241962271566034, -1.413280244657798, -0.724917832513033,  0.937712470759027],
            [-1.012831120334424, 0.814247332595274,  0.091975924478789,  0.087696298664708]
        ];
        let d = array![
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ];

        let cache = UnitDistanceCache::build(&y.view(), &d.view());

        for target_period in 0..5 {
            for i in 0..4 {
                for j in 0..4 {
                    let direct = compute_unit_distance_for_obs(
                        &y.view(),
                        &d.view(),
                        j,
                        i,
                        target_period,
                    );
                    let cached = cache.distance(&y.view(), j, i, target_period);
                    if direct.is_infinite() {
                        assert!(
                            cached.is_infinite(),
                            "Cached distance at (t={}, i={}, j={}) should be inf, got {}",
                            target_period, i, j, cached
                        );
                    } else {
                        assert!(
                            (direct - cached).abs() < 1e-10,
                            "Cached distance mismatch at (t={}, i={}, j={}): direct={}, cached={}, diff={}",
                            target_period, i, j, direct, cached, (direct - cached).abs()
                        );
                    }
                }
            }
        }
    }

    /// Cache respects the `d == 1` mask: treated cells are excluded from
    /// the pairwise sum regardless of target period.
    #[test]
    fn test_unit_distance_cache_respects_treatment_mask() {
        // Units 0, 1 always control; unit 2 treated from period 2 onward.
        let y = array![[1.0, 1.5, 0.0], [2.0, 2.5, 0.0], [3.0, 3.5, 0.0], [4.0, 4.5, 0.0]];
        let d = array![
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0]
        ];

        let cache = UnitDistanceCache::build(&y.view(), &d.view());

        // Pairs (0, 1) use every period. Difference is constant 0.5 ⇒ RMSE 0.5.
        for t in 0..4 {
            let direct = compute_unit_distance_for_obs(&y.view(), &d.view(), 0, 1, t);
            let cached = cache.distance(&y.view(), 0, 1, t);
            assert!((direct - cached).abs() < 1e-10);
            assert!((direct - 0.5).abs() < 1e-10);
        }

        // Pair (0, 2) uses only periods 0, 1 (periods 2, 3 have unit 2
        // treated).  Excluding period 0 leaves only period 1 ⇒ finite.
        let direct_ex0 = compute_unit_distance_for_obs(&y.view(), &d.view(), 0, 2, 0);
        let cached_ex0 = cache.distance(&y.view(), 0, 2, 0);
        assert!((direct_ex0 - cached_ex0).abs() < 1e-10);
    }

    /// Catastrophic cancellation guard: removing the sole valid period's
    /// contribution must not yield a negative sum_sq (which would produce NaN).
    /// With on-demand computation this case simply yields zero valid periods → infinity.
    #[test]
    fn test_unit_distance_cache_sole_valid_period() {
        // Units 0, 1 both finite only at t = 2.
        let y = array![
            [f64::NAN, f64::NAN],
            [f64::NAN, f64::NAN],
            [1.0, 2.0],
            [f64::NAN, f64::NAN]
        ];
        let d = array![[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]];

        let cache = UnitDistanceCache::build(&y.view(), &d.view());

        // Excluding t=2 leaves no valid periods → infinity
        let cached_ex2 = cache.distance(&y.view(), 0, 1, 2);
        assert!(cached_ex2.is_infinite());

        // Sanity: at t=0 the only valid period (t=2) is retained
        let cached_ex0 = cache.distance(&y.view(), 0, 1, 0);
        assert!((cached_ex0 - 1.0).abs() < 1e-10);
    }

    /// `compute_distance` matches the direct computation for every
    /// (exclude_t, i, j) triple on a fixed reference panel.
    #[test]
    fn test_compute_distance_matches_direct() {
        let y = array![
            [0.496714153011233,  0.361735698828815,  1.647688538100692,  3.023029856408026],
            [-0.234153374723336, 0.265863043050819,  2.579212815507391,  2.267434729152909],
            [-0.469474385934952, 1.042560043585965,  0.536582307187538,  1.034270246429743],
            [0.241962271566034, -1.413280244657798, -0.724917832513033,  0.937712470759027],
            [-1.012831120334424, 0.814247332595274,  0.091975924478789,  0.087696298664708]
        ];
        let d = array![
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ];

        let cache = UnitDistanceCache::build(&y.view(), &d.view());

        for target_period in 0..5 {
            for i in 0..4 {
                for j in 0..4 {
                    let direct = compute_unit_distance_for_obs(
                        &y.view(), &d.view(), j, i, target_period,
                    );
                    let on_demand = cache.compute_distance(i, j, Some(target_period));
                    if direct.is_infinite() {
                        assert!(
                            on_demand.is_infinite(),
                            "compute_distance at (t={}, i={}, j={}) should be inf, got {}",
                            target_period, i, j, on_demand
                        );
                    } else {
                        assert!(
                            (direct - on_demand).abs() < 1e-14,
                            "compute_distance mismatch at (t={}, i={}, j={}): direct={}, on_demand={}, diff={}",
                            target_period, i, j, direct, on_demand, (direct - on_demand).abs()
                        );
                    }
                }
            }
        }
    }

    /// `compute_distance` with exclude_t=None matches the full (no-exclusion) distance.
    #[test]
    fn test_compute_distance_no_exclusion() {
        let y = array![[1.0, 1.5], [2.0, 2.5], [3.0, 3.5], [4.0, 4.5]];
        let d = array![[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]];

        let cache = UnitDistanceCache::build(&y.view(), &d.view());

        // No exclusion: use all 4 periods. Constant diff 0.5 → RMSE 0.5.
        let dist = cache.compute_distance(0, 1, None);
        assert!((dist - 0.5).abs() < 1e-14);
    }

    /// Symmetry: compute_distance(i, j, t) == compute_distance(j, i, t).
    #[test]
    fn test_compute_distance_symmetry() {
        let y = array![
            [0.496714153011233,  0.361735698828815,  1.647688538100692],
            [-0.234153374723336, 0.265863043050819,  2.579212815507391],
            [-0.469474385934952, 1.042560043585965,  0.536582307187538],
        ];
        let d = array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]];

        let cache = UnitDistanceCache::build(&y.view(), &d.view());

        for t in 0..3 {
            for i in 0..3 {
                for j in 0..3 {
                    let a = cache.compute_distance(i, j, Some(t));
                    let b = cache.compute_distance(j, i, Some(t));
                    if a.is_infinite() {
                        assert!(b.is_infinite(), "symmetry broken at (i={}, j={}, t={})", i, j, t);
                    } else {
                        assert!((a - b).abs() < 1e-14,
                            "symmetry broken at (i={}, j={}, t={}): {} vs {}", i, j, t, a, b);
                    }
                }
            }
        }
    }

    /// Self-distance is always zero via compute_distance.
    #[test]
    fn test_compute_distance_self_is_zero() {
        let y = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let d = array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];

        let cache = UnitDistanceCache::build(&y.view(), &d.view());

        for i in 0..3 {
            assert!((cache.compute_distance(i, i, None)).abs() < 1e-14);
            assert!((cache.compute_distance(i, i, Some(0))).abs() < 1e-14);
        }
    }
}

/// Property-based tests for distance matrix invariants.
#[cfg(test)]
mod proptests {
    use super::*;
    use ndarray::Array2;
    use proptest::prelude::*;

    /// Generates a random outcome matrix Y of shape (T × N) with standard-normal entries.
    fn y_matrix_strategy(n_periods: usize, n_units: usize) -> impl Strategy<Value = Array2<f64>> {
        prop::collection::vec(prop::num::f64::NORMAL, n_periods * n_units)
            .prop_map(move |v| Array2::from_shape_vec((n_periods, n_units), v).unwrap())
    }

    /// Returns an all-control treatment matrix (all zeros).
    fn d_matrix_all_control(n_periods: usize, n_units: usize) -> Array2<f64> {
        Array2::zeros((n_periods, n_units))
    }

    /// Generates a random binary treatment matrix.
    fn d_matrix_strategy(n_periods: usize, n_units: usize) -> impl Strategy<Value = Array2<f64>> {
        prop::collection::vec(prop::bool::ANY, n_periods * n_units).prop_map(move |v| {
            Array2::from_shape_fn((n_periods, n_units), |(t, i)| {
                if v[t * n_units + i] {
                    1.0
                } else {
                    0.0
                }
            })
        })
    }

    proptest! {
        /// dist(i, i) = 0 for every unit (all-control panel).
        #[test]
        fn prop_diagonal_zero(
            y in y_matrix_strategy(5, 4)
        ) {
            let d = d_matrix_all_control(5, 4);
            let dist = compute_unit_distance_matrix_internal(&y.view(), &d.view());

            for i in 0..4 {
                prop_assert!((dist[[i, i]]).abs() < 1e-10,
                    "diagonal [{}, {}] = {}, expected 0", i, i, dist[[i, i]]);
            }
        }

        /// dist(i, j) = dist(j, i) for all pairs (all-control panel).
        #[test]
        fn prop_symmetric(
            y in y_matrix_strategy(5, 4)
        ) {
            let d = d_matrix_all_control(5, 4);
            let dist = compute_unit_distance_matrix_internal(&y.view(), &d.view());

            for i in 0..4 {
                for j in 0..4 {
                    let a = dist[[i, j]];
                    let b = dist[[j, i]];
                    let ok = if a.is_infinite() && b.is_infinite() {
                        a.signum() == b.signum()
                    } else if a.is_nan() || b.is_nan() {
                        false
                    } else {
                        (a - b).abs() < 1e-10
                    };
                    prop_assert!(ok,
                        "dist[{}, {}] = {} != dist[{}, {}] = {}",
                        i, j, a, j, i, b);
                }
            }
        }

        /// All entries are non-negative (or +infinity).
        #[test]
        fn prop_non_negative(
            y in y_matrix_strategy(5, 4),
            d in d_matrix_strategy(5, 4)
        ) {
            let dist = compute_unit_distance_matrix_internal(&y.view(), &d.view());

            for i in 0..4 {
                for j in 0..4 {
                    let val = dist[[i, j]];
                    prop_assert!(val >= 0.0 || val.is_infinite(),
                        "dist[{}, {}] = {}, expected >= 0 or inf", i, j, val);
                }
            }
        }

        /// Self-distance is zero regardless of treatment pattern.
        #[test]
        fn prop_self_distance_zero(
            y in y_matrix_strategy(6, 5),
            d in d_matrix_strategy(6, 5)
        ) {
            let dist = compute_unit_distance_matrix_internal(&y.view(), &d.view());

            for i in 0..5 {
                prop_assert!((dist[[i, i]]).abs() < 1e-10,
                    "self-distance [{}, {}] = {}, expected 0", i, i, dist[[i, i]]);
            }
        }

        /// Symmetry holds under arbitrary treatment patterns.
        #[test]
        fn prop_symmetric_with_treatment(
            y in y_matrix_strategy(6, 5),
            d in d_matrix_strategy(6, 5)
        ) {
            let dist = compute_unit_distance_matrix_internal(&y.view(), &d.view());

            for i in 0..5 {
                for j in 0..5 {
                    let ok = if dist[[i, j]].is_infinite() && dist[[j, i]].is_infinite() {
                        true
                    } else {
                        (dist[[i, j]] - dist[[j, i]]).abs() < 1e-10
                    };
                    prop_assert!(ok,
                        "dist[{}, {}] = {} != dist[{}, {}] = {}",
                        i, j, dist[[i, j]], j, i, dist[[j, i]]);
                }
            }
        }

        /// All three invariants hold on a larger panel (T=10, N=8).
        #[test]
        fn prop_larger_panel(
            y in y_matrix_strategy(10, 8)
        ) {
            let d = d_matrix_all_control(10, 8);
            let dist = compute_unit_distance_matrix_internal(&y.view(), &d.view());

            for i in 0..8 {
                prop_assert!((dist[[i, i]]).abs() < 1e-10);
            }

            for i in 0..8 {
                for j in 0..8 {
                    let a = dist[[i, j]];
                    let b = dist[[j, i]];
                    let ok = if a.is_infinite() && b.is_infinite() {
                        a.signum() == b.signum()
                    } else {
                        (a - b).abs() < 1e-10
                    };
                    prop_assert!(ok);
                }
            }

            for i in 0..8 {
                for j in 0..8 {
                    prop_assert!(dist[[i, j]] >= 0.0 || dist[[i, j]].is_infinite());
                }
            }
        }
    }
}

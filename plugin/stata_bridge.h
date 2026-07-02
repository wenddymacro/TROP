/**
 * stata_bridge.h -- C bridge between Stata and the TROP core library.
 *
 * Declares error codes, command enumeration, FFI function prototypes
 * for the compiled core library, and helper routines for Stata data
 * transfer and missing-value conversion.
 *
 * All matrices exchanged through this interface use column-major
 * (Fortran) storage order, consistent with Stata matrix conventions.
 */

#ifndef STATA_BRIDGE_H
#define STATA_BRIDGE_H

#include "stplugin.h"
#include <stdint.h>
#include <math.h>

/* ============================================================================
 * Error Codes — Core Library
 *
 * Codes 0--11 are returned by the compiled core library.
 * ============================================================================ */

#define TROP_SUCCESS            0
#define TROP_ERR_NULL_POINTER   1
#define TROP_ERR_INVALID_DIM    2
#define TROP_ERR_NO_CONTROL     3
#define TROP_ERR_NO_TREATED     4
#define TROP_ERR_CONVERGENCE    5
#define TROP_ERR_SINGULAR       6
#define TROP_ERR_MEMORY         7
#define TROP_ERR_RUST_PANIC     8
#define TROP_ERR_LOOCV_FAIL     9
#define TROP_ERR_BOOTSTRAP_FAIL 10
#define TROP_ERR_COMPUTATION    11
#define TROP_ERR_INVALID_FPC    12
#define TROP_ERR_SINGLETON_PSU  13

/* ============================================================================
 * Error Codes — Bridge Layer
 *
 * Codes 100+ originate in this C bridge when reading Stata objects.
 * ============================================================================ */

#define TROP_ERR_INVALID_ARGS   100
#define TROP_ERR_VAR_NOT_FOUND  101
#define TROP_ERR_MAT_NOT_FOUND  102
#define TROP_ERR_SCALAR_FAIL    103

/* ============================================================================
 * Command Enumeration
 * ============================================================================ */

typedef enum {
    CMD_LOOCV_TWOSTEP,
    CMD_LOOCV_TWOSTEP_EXHAUSTIVE,
    CMD_LOOCV_JOINT,
    CMD_LOOCV_JOINT_EXHAUSTIVE,
    CMD_ESTIMATE_TWOSTEP,
    CMD_ESTIMATE_JOINT,
    CMD_BOOTSTRAP_TWOSTEP,
    CMD_BOOTSTRAP_JOINT,
    CMD_BOOTSTRAP_RAO_WU_TWOSTEP,
    CMD_BOOTSTRAP_RAO_WU_JOINT,
    CMD_DISTANCE_MATRIX,
    CMD_UNKNOWN
} TropCommand;

/* ============================================================================
 * Verbosity Control
 *
 * Five levels of output detail:
 *   0 (QUIET)    - errors only
 *   1 (NORMAL)   - default: progress milestones (start/complete)
 *   2 (DETAILED) - per-stage summaries and grid-point counts
 *   3 (DEBUG)    - internal dispatch traces and parameter dumps
 *   4 (DEV)      - developer-only: memory layouts and raw buffers
 * ============================================================================ */

#define TROP_VERBOSE_QUIET    0
#define TROP_VERBOSE_NORMAL   1
#define TROP_VERBOSE_DETAILED 2
#define TROP_VERBOSE_DEBUG    3
#define TROP_VERBOSE_DEV      4

extern int g_verbose_level;

/* ============================================================================
 * Logging
 * ============================================================================ */

void trop_log(int level, const char *tag, const char *fmt, ...);

#define TROP_LOG_ERROR(fmt, ...)  trop_log(0, "ERROR", fmt, ##__VA_ARGS__)
#define TROP_LOG_INFO(fmt, ...)   trop_log(1, "INFO", fmt, ##__VA_ARGS__)
#define TROP_LOG_DETAIL(fmt, ...) trop_log(2, "DETAIL", fmt, ##__VA_ARGS__)
#define TROP_LOG_DEBUG(fmt, ...)  trop_log(3, "DEBUG", fmt, ##__VA_ARGS__)
#define TROP_LOG_DEV(fmt, ...)    trop_log(4, "DEV", fmt, ##__VA_ARGS__)

/* ============================================================================
 * Command Dispatch Macros
 *
 * PARSE_CMD_ENTRY: maps a command string to its enum value inside
 *                  parse_command().  Reduces boilerplate strcmp chains.
 *
 * DISPATCH_CMD_CASE: maps a command enum to its handler function inside
 *                    the stata_call switch block.
 * ============================================================================ */

#define PARSE_CMD_ENTRY(str, enum_val) \
    if (strcmp(cmd, str) == 0) return enum_val;

#define DISPATCH_CMD_CASE(cmd_enum, handler) \
    case cmd_enum: rc = handler(); break;

/* ============================================================================
 * ABI Version Handshake (P3.3)
 *
 * The bridge verifies at first call that the loaded core library exports
 * the expected ABI version.  A mismatch emits a warning but does not
 * abort execution (forward-compatible soft check).
 * ============================================================================ */

#define TROP_EXPECTED_ABI_VERSION 2

/* ============================================================================
 * Core Library FFI Declarations
 *
 * Each function below is implemented in the compiled core library and
 * linked at build time.  Parameter documentation uses the notation:
 *
 *   T = number of time periods,  N = number of cross-sectional units.
 *
 * Matrices are T x N in column-major order unless stated otherwise.
 * Output parameters marked [out] are written by the callee; those
 * additionally marked (nullable) may be passed as NULL.
 * ============================================================================ */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Returns the ABI version of the compiled core library.
 * Used for mismatch detection at plugin load time.
 */
extern int trop_abi_version(void);

/**
 * LOOCV grid search for the twostep estimator.
 *
 * Evaluates every combination in the Cartesian product of the three
 * lambda grids via leave-one-out cross-validation and returns the
 * triple that minimises the prediction error.
 *
 * @param y_ptr                 Outcome matrix Y (T x N, column-major)
 * @param d_ptr                 Treatment indicator D (T x N, column-major)
 * @param control_mask_ptr      Control mask (T x N, column-major; 1 = control)
 * @param time_dist_ptr         Time distance matrix (T x T, column-major)
 * @param n_periods             T
 * @param n_units               N
 * @param lambda_time_grid_ptr  Candidate values for lambda_time
 * @param lambda_time_grid_len  Length of lambda_time grid
 * @param lambda_unit_grid_ptr  Candidate values for lambda_unit
 * @param lambda_unit_grid_len  Length of lambda_unit grid
 * @param lambda_nn_grid_ptr    Candidate values for lambda_nn
 * @param lambda_nn_grid_len    Length of lambda_nn grid
 * @param max_iter              Maximum iterations per fit
 * @param tol                   Convergence tolerance
 * @param best_lambda_time_out  [out] Selected lambda_time
 * @param best_lambda_unit_out  [out] Selected lambda_unit
 * @param best_lambda_nn_out    [out] Selected lambda_nn
 * @param best_score_out        [out] Minimum LOOCV score
 * @param n_valid_out           [out] Successful LOOCV fits (nullable)
 * @param n_attempted_out       [out] Total finite control observations (nullable)
 * @param first_failed_t_out    [out] Period of first failed fit, -1 if none (nullable)
 * @param first_failed_i_out    [out] Unit of first failed fit, -1 if none (nullable)
 * @param stage1_lambda_time_out [out] Stage-1 univariate initial lambda_time (nullable; Footnote 2)
 * @param stage1_lambda_unit_out [out] Stage-1 univariate initial lambda_unit (nullable)
 * @param stage1_lambda_nn_out   [out] Stage-1 univariate initial lambda_nn (nullable)
 * @return 0 on success, nonzero error code otherwise
 */
extern int stata_loocv_grid_search(
    const double *y_ptr,
    const double *d_ptr,
    const unsigned char *control_mask_ptr,
    const int64_t *time_dist_ptr,
    int n_periods,
    int n_units,
    const double *lambda_time_grid_ptr,
    int lambda_time_grid_len,
    const double *lambda_unit_grid_ptr,
    int lambda_unit_grid_len,
    const double *lambda_nn_grid_ptr,
    int lambda_nn_grid_len,
    int max_iter,
    double tol,
    double *best_lambda_time_out,
    double *best_lambda_unit_out,
    double *best_lambda_nn_out,
    double *best_score_out,
    int *n_valid_out,
    int *n_attempted_out,
    int *first_failed_t_out,
    int *first_failed_i_out,
    double *stage1_lambda_time_out,
    double *stage1_lambda_unit_out,
    double *stage1_lambda_nn_out
);

/**
 * Exhaustive (Cartesian) LOOCV grid search for the Twostep estimator.
 *
 * Evaluates every (lambda_time, lambda_unit, lambda_nn) combination in
 * parallel and returns the global grid minimum under the LOOCV criterion
 * Q(lambda).  Complexity is O(|grid|^3); for large grids the
 * coordinate-descent variant stata_loocv_grid_search is typically preferred.
 *
 * Parameters and return value are identical to stata_loocv_grid_search().
 * On small panels (N or T small) the exhaustive path is immune to the local
 * minima that the cycling path can encounter when Q(lambda) is non-convex,
 * which is the source of platform- and BLAS-dependent lambda drift.
 */
extern int stata_loocv_grid_search_exhaustive(
    const double *y_ptr,
    const double *d_ptr,
    const unsigned char *control_mask_ptr,
    const int64_t *time_dist_ptr,
    int n_periods,
    int n_units,
    const double *lambda_time_grid_ptr,
    int lambda_time_grid_len,
    const double *lambda_unit_grid_ptr,
    int lambda_unit_grid_len,
    const double *lambda_nn_grid_ptr,
    int lambda_nn_grid_len,
    int max_iter,
    double tol,
    double *best_lambda_time_out,
    double *best_lambda_unit_out,
    double *best_lambda_nn_out,
    double *best_score_out,
    int *n_valid_out,
    int *n_attempted_out,
    int *first_failed_t_out,
    int *first_failed_i_out
);

/**
 * Coordinate-descent LOOCV search for the joint estimator.
 *
 * Uses a two-stage strategy to avoid the cubic cost of a full grid:
 *   Stage 1 — univariate sweeps with remaining parameters at extremes;
 *   Stage 2 — cyclic coordinate descent until convergence.
 * Complexity is O(|grid| x max_cycles) rather than O(|grid|^3).
 *
 * @param max_cycles  Maximum coordinate-descent cycles
 *
 * Remaining parameters and return value are identical to
 * stata_loocv_grid_search(), except that time_dist_ptr is absent
 * (the joint estimator does not use a time distance matrix).
 */
extern int stata_loocv_cycling_search_joint(
    const double *y_ptr,
    const double *d_ptr,
    const unsigned char *control_mask_ptr,
    int n_periods,
    int n_units,
    const double *lambda_time_grid_ptr,
    int lambda_time_grid_len,
    const double *lambda_unit_grid_ptr,
    int lambda_unit_grid_len,
    const double *lambda_nn_grid_ptr,
    int lambda_nn_grid_len,
    int max_iter,
    double tol,
    int max_cycles,
    double *best_lambda_time_out,
    double *best_lambda_unit_out,
    double *best_lambda_nn_out,
    double *best_score_out,
    int *n_valid_out,
    int *n_attempted_out,
    int *first_failed_t_out,
    int *first_failed_i_out,
    double *stage1_lambda_time_out,
    double *stage1_lambda_unit_out,
    double *stage1_lambda_nn_out,
    const double *x_ptr,
    int n_covariates
);

/**
 * Exhaustive (Cartesian) LOOCV grid search for the joint estimator.
 *
 * Evaluates all |grid|^3 (lambda_time, lambda_unit, lambda_nn) combinations
 * in parallel and returns the triple minimising the LOOCV criterion Q(lambda).
 * Matches the Python reference (diff_diff.trop_global, v3.1.1) exactly.
 *
 * Use when the Cartesian product is affordable; prefer the cycling variant
 * for large grids.
 *
 * Parameters and return value mirror stata_loocv_cycling_search_joint(),
 * except that max_cycles is absent (no coordinate descent is performed).
 */
extern int stata_loocv_grid_search_joint(
    const double *y_ptr,
    const double *d_ptr,
    const unsigned char *control_mask_ptr,
    int n_periods,
    int n_units,
    const double *lambda_time_grid_ptr,
    int lambda_time_grid_len,
    const double *lambda_unit_grid_ptr,
    int lambda_unit_grid_len,
    const double *lambda_nn_grid_ptr,
    int lambda_nn_grid_len,
    int max_iter,
    double tol,
    double *best_lambda_time_out,
    double *best_lambda_unit_out,
    double *best_lambda_nn_out,
    double *best_score_out,
    int *n_valid_out,
    int *n_attempted_out,
    int *first_failed_t_out,
    int *first_failed_i_out,
    const double *x_ptr,
    int n_covariates
);

/**
 * Twostep estimation with fixed regularization parameters.
 *
 * Fits the model Y = mu + alpha_i + beta_t + L_{it} + tau_{it} D_{it}
 * for each treated observation separately, then averages the individual
 * treatment effects to obtain the ATT.
 *
 * @param y_ptr              Y (T x N, column-major)
 * @param d_ptr              D (T x N, column-major)
 * @param control_mask_ptr   Control mask (T x N, column-major)
 * @param time_dist_ptr      Time distance matrix (T x T, column-major)
 * @param n_periods          T
 * @param n_units            N
 * @param lambda_time        Time regularization parameter
 * @param lambda_unit        Unit regularization parameter
 * @param lambda_nn          Nuclear norm regularization parameter
 * @param max_iter           Maximum iterations
 * @param tol                Convergence tolerance
 * @param att_out            [out] Average treatment effect on the treated
 * @param tau_ptr            [out] Individual treatment effects (n_treated)
 * @param alpha_ptr          [out] Unit fixed effects (N)
 * @param beta_ptr           [out] Time fixed effects (T)
 * @param l_ptr              [out] Low-rank component L (T x N, column-major)
 * @param n_treated_out      [out] Number of treated observations
 * @param n_iterations_out   [out] Maximum iterations across observations
 * @param converged_out      [out] 1 if converged, 0 otherwise
 * @param converged_by_obs_ptr [out, nullable] 0/1 per treated (t,i); -1 on
 *                             solver failure.  Must be N_treated * int if
 *                             non-null.  Ordering matches the Rust-side
 *                             iteration `for t { for i { if D=1 ... } }`.
 * @param n_iters_by_obs_ptr   [out, nullable] iterations used per treated
 *                             (t,i); -1 on solver failure.
 * @return 0 on success
 */
extern int stata_estimate_twostep(
    const double *y_ptr,
    const double *d_ptr,
    const unsigned char *control_mask_ptr,
    const int64_t *time_dist_ptr,
    int n_periods,
    int n_units,
    double lambda_time,
    double lambda_unit,
    double lambda_nn,
    int max_iter,
    double tol,
    double *att_out,
    double *tau_ptr,
    double *alpha_ptr,
    double *beta_ptr,
    double *l_ptr,
    int *n_treated_out,
    int *n_iterations_out,
    int *converged_out,
    int *converged_by_obs_ptr,
    int *n_iters_by_obs_ptr
);

/**
 * Joint estimation with fixed regularization parameters.
 *
 * Solves a single weighted least squares problem for a scalar
 * treatment effect tau, without requiring a time distance matrix.
 *
 * @param y_ptr              Y (T x N, column-major)
 * @param d_ptr              D (T x N, column-major)
 * @param n_periods          T
 * @param n_units            N
 * @param lambda_time        Time regularization parameter
 * @param lambda_unit        Unit regularization parameter
 * @param lambda_nn          Nuclear norm regularization parameter
 * @param max_iter           Maximum iterations
 * @param tol                Convergence tolerance
 * @param tau_out            [out] Scalar treatment effect (mean of per-cell τ)
 * @param mu_out             [out] Intercept
 * @param alpha_ptr          [out] Unit fixed effects (N)
 * @param beta_ptr           [out] Time fixed effects (T)
 * @param l_ptr              [out] Low-rank component L (T x N, column-major)
 * @param n_iterations_out   [out] Iterations performed
 * @param converged_out      [out] Convergence flag
 * @param tau_vec_ptr        [out] Per-cell τ_it for treated (i,t) (nullable);
 *                                 if non-null, must be pre-allocated to hold
 *                                 at least `n_treated_out` doubles
 * @param n_treated_out      [out] Number of treated cells written (nullable)
 * @return 0 on success
 */
extern int stata_estimate_joint(
    const double *y_ptr,
    const double *d_ptr,
    int n_periods,
    int n_units,
    double lambda_time,
    double lambda_unit,
    double lambda_nn,
    int max_iter,
    double tol,
    double *tau_out,
    double *mu_out,
    double *alpha_ptr,
    double *beta_ptr,
    double *l_ptr,
    int *n_iterations_out,
    int *converged_out,
    double *tau_vec_ptr,
    int *n_treated_out
);

/**
 * Bootstrap variance estimation for the twostep estimator.
 *
 * Resamples units with replacement and re-estimates the ATT on each
 * bootstrap sample.  Returns the standard error and percentile-based
 * confidence bounds.
 *
 * @param y_ptr              Y (T x N, column-major)
 * @param d_ptr              D (T x N, column-major)
 * @param control_mask_ptr   Control mask (T x N, column-major)
 * @param time_dist_ptr      Time distance matrix (T x T, column-major)
 * @param n_periods          T
 * @param n_units            N
 * @param lambda_time        Time regularization parameter
 * @param lambda_unit        Unit regularization parameter
 * @param lambda_nn          Nuclear norm regularization parameter
 * @param n_bootstrap        Number of bootstrap replications
 * @param max_iter           Maximum iterations per replication
 * @param tol                Convergence tolerance
 * @param seed               Random seed
 * @param alpha              Significance level (e.g. 0.05 for 95% CI)
 * @param ddof               Variance denominator selector: 1 = sample
 *                           variance 1/(B-1) (default); 0 = paper
 *                           Algorithm 3 population variance 1/B.
 *                           Any other value collapses to 1.
 * @param estimates_ptr      [out] Bootstrap ATT estimates (n_bootstrap)
 * @param se_out             [out] Bootstrap standard error
 * @param ci_lower_out       [out] Lower percentile bound (nullable)
 * @param ci_upper_out       [out] Upper percentile bound (nullable)
 * @param n_valid_out        [out] Valid replications (nullable)
 * @return 0 on success
 */
extern int stata_bootstrap_trop_variance(
    const double *y_ptr,
    const double *d_ptr,
    const unsigned char *control_mask_ptr,
    const int64_t *time_dist_ptr,
    int n_periods,
    int n_units,
    double lambda_time,
    double lambda_unit,
    double lambda_nn,
    int n_bootstrap,
    int max_iter,
    double tol,
    uint64_t seed,
    double alpha,
    int ddof,
    double *estimates_ptr,
    double *se_out,
    double *ci_lower_out,
    double *ci_upper_out,
    int *n_valid_out
);

/**
 * Bootstrap variance estimation for the joint estimator.
 *
 * Same resampling scheme as the twostep bootstrap, applied to the
 * joint estimator.  Does not require a time distance matrix.
 *
 * Parameters and return value follow stata_bootstrap_trop_variance(),
 * minus control_mask_ptr and time_dist_ptr.
 */
extern int stata_bootstrap_trop_variance_joint(
    const double *y_ptr,
    const double *d_ptr,
    int n_periods,
    int n_units,
    double lambda_time,
    double lambda_unit,
    double lambda_nn,
    int n_bootstrap,
    int max_iter,
    double tol,
    uint64_t seed,
    double alpha,
    int ddof,
    double *estimates_ptr,
    double *se_out,
    double *ci_lower_out,
    double *ci_upper_out,
    int *n_valid_out
);

/**
 * Rao-Wu bootstrap variance estimation for the twostep estimator
 * with complex survey design (strata, PSU, FPC).
 *
 * Fits the model once, then rescales unit weights for each replicate
 * according to the Rao-Wu (1988) scheme.
 *
 * @param fpc_ptr    FPC values per unit (nullable; NULL = no FPC)
 */
extern int stata_bootstrap_trop_variance_rao_wu(
    const double *y_ptr,
    const double *d_ptr,
    const unsigned char *control_mask_ptr,
    const int64_t *time_dist_ptr,
    int n_periods,
    int n_units,
    double lambda_time,
    double lambda_unit,
    double lambda_nn,
    int n_bootstrap,
    int max_iter,
    double tol,
    uint64_t seed,
    double alpha,
    int ddof,
    const int64_t *strata_ptr,
    const int64_t *psu_ptr,
    const double *fpc_ptr,
    const double *unit_weights_ptr,
    int lonely_psu_code,
    double *estimates_ptr,
    double *se_out,
    double *ci_lower_out,
    double *ci_upper_out,
    int *n_valid_out
);

/**
 * Rao-Wu bootstrap variance estimation for the joint estimator
 * with complex survey design (strata, PSU, FPC).
 *
 * Same Rao-Wu reweighting scheme applied to the joint estimator.
 * Does not require control_mask_ptr or time_dist_ptr.
 *
 * @param fpc_ptr    FPC values per unit (nullable; NULL = no FPC)
 */
extern int stata_bootstrap_trop_variance_rao_wu_joint(
    const double *y_ptr,
    const double *d_ptr,
    int n_periods,
    int n_units,
    double lambda_time,
    double lambda_unit,
    double lambda_nn,
    int n_bootstrap,
    int max_iter,
    double tol,
    uint64_t seed,
    double alpha,
    int ddof,
    const int64_t *strata_ptr,
    const int64_t *psu_ptr,
    const double *fpc_ptr,
    const double *unit_weights_ptr,
    int lonely_psu_code,
    double *estimates_ptr,
    double *se_out,
    double *ci_lower_out,
    double *ci_upper_out,
    int *n_valid_out
);

/**
 * Compute the pairwise unit distance matrix.
 *
 * For each pair of units (i, j), computes the Euclidean distance
 * over their pre-treatment outcome trajectories.
 *
 * @param y_ptr      Y (T x N, column-major)
 * @param d_ptr      D (T x N, column-major)
 * @param n_periods  T
 * @param n_units    N
 * @param dist_ptr   [out] Distance matrix (N x N, column-major)
 * @return 0 on success
 */
extern int stata_compute_unit_distance_matrix(
    const double *y_ptr,
    const double *d_ptr,
    int n_periods,
    int n_units,
    double *dist_ptr
);

/**
 * Twostep estimation with per-unit probability weights.
 *
 * Identical to stata_estimate_twostep() but aggregates the per-cell tau
 * into the ATT as tau_hat = sum_i w_i * tau_{t,i} / sum_i w_i, where w_i
 * is the pweight attached to the original unit index.  Per-cell estimation
 * (alpha, beta, L) is unchanged.
 *
 * @param unit_weights_ptr  Per-unit pweights (N values); must be non-null
 *                          and strictly positive (validated by caller).
 */
extern int stata_estimate_twostep_weighted(
    const double *y_ptr,
    const double *d_ptr,
    const unsigned char *control_mask_ptr,
    const int64_t *time_dist_ptr,
    int n_periods,
    int n_units,
    double lambda_time,
    double lambda_unit,
    double lambda_nn,
    int max_iter,
    double tol,
    double *att_out,
    double *tau_ptr,
    double *alpha_ptr,
    double *beta_ptr,
    double *l_ptr,
    int *n_treated_out,
    int *n_iterations_out,
    int *converged_out,
    int *converged_by_obs_ptr,
    int *n_iters_by_obs_ptr,
    const double *unit_weights_ptr
);

/**
 * Joint estimation with per-unit probability weights.
 *
 * Identical to stata_estimate_joint() but computes the post-hoc ATT as
 * tau_hat = sum_i w_i * (Y - mu - alpha - beta - L) / sum_i w_i.
 * The joint estimation of (mu, alpha, beta, L) is unchanged.
 */
extern int stata_estimate_joint_weighted(
    const double *y_ptr,
    const double *d_ptr,
    int n_periods,
    int n_units,
    double lambda_time,
    double lambda_unit,
    double lambda_nn,
    int max_iter,
    double tol,
    double *tau_out,
    double *mu_out,
    double *alpha_ptr,
    double *beta_ptr,
    double *l_ptr,
    int *n_iterations_out,
    int *converged_out,
    double *tau_vec_ptr,
    int *n_treated_out,
    const double *unit_weights_ptr
);

/**
 * Twostep bootstrap with per-unit probability weights.
 *
 * Same resampling scheme as stata_bootstrap_trop_variance() but aggregates
 * each replicate's ATT using the pweight inherited from each resampled
 * unit's original panel column.
 */
extern int stata_bootstrap_trop_variance_weighted(
    const double *y_ptr,
    const double *d_ptr,
    const unsigned char *control_mask_ptr,
    const int64_t *time_dist_ptr,
    int n_periods,
    int n_units,
    double lambda_time,
    double lambda_unit,
    double lambda_nn,
    int n_bootstrap,
    int max_iter,
    double tol,
    uint64_t seed,
    double alpha,
    int ddof,
    double *estimates_ptr,
    double *se_out,
    double *ci_lower_out,
    double *ci_upper_out,
    int *n_valid_out,
    const double *unit_weights_ptr
);

/**
 * Joint bootstrap with per-unit probability weights.
 *
 * Same resampling scheme as stata_bootstrap_trop_variance_joint() but
 * aggregates each replicate's post-hoc ATT using the pweight inherited
 * from each resampled unit's original panel column.
 */
extern int stata_bootstrap_trop_variance_joint_weighted(
    const double *y_ptr,
    const double *d_ptr,
    int n_periods,
    int n_units,
    double lambda_time,
    double lambda_unit,
    double lambda_nn,
    int n_bootstrap,
    int max_iter,
    double tol,
    uint64_t seed,
    double alpha,
    int ddof,
    double *estimates_ptr,
    double *se_out,
    double *ci_lower_out,
    double *ci_upper_out,
    int *n_valid_out,
    const double *unit_weights_ptr
);

/**
 * Compute twostep weight vectors for a given treated observation.
 *
 * Returns the time weights theta (T x 1) and unit weights omega (N x 1)
 * for the specified (unit, period) pair.
 *
 * @param y_ptr           Y (T x N, column-major)
 * @param d_ptr           D (T x N, column-major)
 * @param time_dist_ptr   Time distance matrix (T x T, column-major)
 * @param n_periods       T
 * @param n_units         N
 * @param target_unit     Target unit index (0-based)
 * @param target_period   Target period index (0-based)
 * @param lambda_time     Time weight decay parameter
 * @param lambda_unit     Unit weight decay parameter
 * @param theta_out       [out] Time weights (T values)
 * @param omega_out       [out] Unit weights (N values)
 * @return 0 on success
 */
extern int stata_compute_twostep_weight_vectors(
    const double *y_ptr,
    const double *d_ptr,
    const int64_t *time_dist_ptr,
    int n_periods,
    int n_units,
    int target_unit,
    int target_period,
    double lambda_time,
    double lambda_unit,
    double *theta_out,
    double *omega_out
);

/**
 * Compute joint weight vectors.
 *
 * Returns the global time weights delta_time (T x 1) and unit weights
 * delta_unit (N x 1) used by the joint estimator.
 *
 * @param y_ptr           Y (T x N, column-major)
 * @param d_ptr           D (T x N, column-major)
 * @param n_periods       T
 * @param n_units         N
 * @param lambda_time     Time weight decay parameter
 * @param lambda_unit     Unit weight decay parameter
 * @param delta_time_out  [out] Time weights (T values)
 * @param delta_unit_out  [out] Unit weights (N values)
 * @return 0 on success
 */
extern int stata_compute_joint_weight_vectors(
    const double *y_ptr,
    const double *d_ptr,
    int n_periods,
    int n_units,
    double lambda_time,
    double lambda_unit,
    double *delta_time_out,
    double *delta_unit_out
);

/* ============================================================================
 * Covariate-aware variants
 *
 * These mirror the base functions above but accept an additional covariate
 * matrix X (T*N x p, column-major) and, where applicable, a gamma output
 * buffer (p x 1).
 * ============================================================================ */

extern int stata_loocv_grid_search_with_covariates(
    const double *y_ptr,
    const double *d_ptr,
    const unsigned char *control_mask_ptr,
    const int64_t *time_dist_ptr,
    int n_periods,
    int n_units,
    const double *lambda_time_grid_ptr,
    int lambda_time_grid_len,
    const double *lambda_unit_grid_ptr,
    int lambda_unit_grid_len,
    const double *lambda_nn_grid_ptr,
    int lambda_nn_grid_len,
    int max_iter,
    double tol,
    double *best_lambda_time_out,
    double *best_lambda_unit_out,
    double *best_lambda_nn_out,
    double *best_score_out,
    int *n_valid_out,
    int *n_attempted_out,
    int *first_failed_t_out,
    int *first_failed_i_out,
    double *stage1_lambda_time_out,
    double *stage1_lambda_unit_out,
    double *stage1_lambda_nn_out,
    const double *x_ptr,
    int n_covariates
);

extern int stata_loocv_grid_search_exhaustive_with_covariates(
    const double *y_ptr,
    const double *d_ptr,
    const unsigned char *control_mask_ptr,
    const int64_t *time_dist_ptr,
    int n_periods,
    int n_units,
    const double *lambda_time_grid_ptr,
    int lambda_time_grid_len,
    const double *lambda_unit_grid_ptr,
    int lambda_unit_grid_len,
    const double *lambda_nn_grid_ptr,
    int lambda_nn_grid_len,
    int max_iter,
    double tol,
    double *best_lambda_time_out,
    double *best_lambda_unit_out,
    double *best_lambda_nn_out,
    double *best_score_out,
    int *n_valid_out,
    int *n_attempted_out,
    int *first_failed_t_out,
    int *first_failed_i_out,
    const double *x_ptr,
    int n_covariates
);

extern int stata_estimate_twostep_with_covariates(
    const double *y_ptr,
    const double *d_ptr,
    const unsigned char *control_mask_ptr,
    const int64_t *time_dist_ptr,
    int n_periods,
    int n_units,
    double lambda_time,
    double lambda_unit,
    double lambda_nn,
    int max_iter,
    double tol,
    double *att_out,
    double *tau_ptr,
    double *alpha_ptr,
    double *beta_ptr,
    double *l_ptr,
    int *n_treated_out,
    int *n_iterations_out,
    int *converged_out,
    int *converged_by_obs_ptr,
    int *n_iters_by_obs_ptr,
    const double *x_ptr,
    int n_covariates,
    double *gamma_out
);

extern int stata_bootstrap_trop_variance_with_covariates(
    const double *y_ptr,
    const double *d_ptr,
    const unsigned char *control_mask_ptr,
    const int64_t *time_dist_ptr,
    int n_periods,
    int n_units,
    double lambda_time,
    double lambda_unit,
    double lambda_nn,
    int n_bootstrap,
    int max_iter,
    double tol,
    uint64_t seed,
    double alpha,
    int ddof,
    double *estimates_ptr,
    double *se_out,
    double *ci_lower_out,
    double *ci_upper_out,
    int *n_valid_out,
    const double *x_ptr,
    int n_covariates
);

extern int stata_estimate_joint_with_covariates(
    const double *y_ptr,
    const double *d_ptr,
    int n_periods,
    int n_units,
    double lambda_time,
    double lambda_unit,
    double lambda_nn,
    int max_iter,
    double tol,
    double *tau_out,
    double *mu_out,
    double *alpha_ptr,
    double *beta_ptr,
    double *l_ptr,
    int *n_iterations_out,
    int *converged_out,
    double *tau_vec_ptr,
    int *n_treated_out,
    const double *x_ptr,
    int n_covariates,
    double *gamma_out
);

extern int stata_bootstrap_trop_variance_joint_with_covariates(
    const double *y_ptr,
    const double *d_ptr,
    int n_periods,
    int n_units,
    double lambda_time,
    double lambda_unit,
    double lambda_nn,
    int n_bootstrap,
    int max_iter,
    double tol,
    uint64_t seed,
    double alpha,
    int ddof,
    double *estimates_ptr,
    double *se_out,
    double *ci_lower_out,
    double *ci_upper_out,
    int *n_valid_out,
    const double *x_ptr,
    int n_covariates
);

/**
 * Compute survey diagnostics: Kish DEFF and high-FPC stratum detection.
 *
 * Pure diagnostic function -- does not alter any computation.
 * Call after a successful Rao-Wu bootstrap to obtain diagnostic scalars.
 *
 * @param strata_ptr         Stratum labels per unit (N values)
 * @param psu_ptr            PSU labels per unit (N values)
 * @param fpc_ptr            FPC values per unit (nullable; NULL = no FPC)
 * @param unit_weights_ptr   Per-unit survey weights (N values)
 * @param n_units            N
 * @param deff_weights_out   [out] Kish design effect
 * @param max_fh_out         [out] Maximum sampling fraction
 * @param n_high_fpc_out     [out] Number of strata with f_h > 0.5
 * @param high_fpc_fh_ptr    [out, nullable] f_h values for high-FPC strata
 * @param high_fpc_max_elements  Max elements to write to high_fpc_fh_ptr
 * @return 0 on success
 */
extern int stata_compute_survey_diagnostics(
    const int64_t *strata_ptr,
    const int64_t *psu_ptr,
    const double *fpc_ptr,
    const double *unit_weights_ptr,
    int n_units,
    double *deff_weights_out,
    double *max_fh_out,
    int *n_high_fpc_out,
    double *high_fpc_fh_ptr,
    int high_fpc_max_elements
);

/**
 * Retrieve the last Rust panic message (P1.1).
 *
 * Copies at most buf_len-1 bytes of the message into buf and
 * null-terminates.  Returns number of bytes written (excl. NUL),
 * or 0 if buf is NULL / buf_len <= 0 / no panic occurred.
 */
extern int trop_get_last_panic_message(char *buf, int buf_len);

#ifdef __cplusplus
}
#endif

/* ============================================================================
 * Helper Functions
 * ============================================================================ */

/**
 * Parse a command string into a TropCommand enum value.
 */
TropCommand parse_command(const char *cmd);

/**
 * Map a core library error code to a Stata error message and display it.
 */
void translate_error_code(int rust_code);

/**
 * Convert a Stata missing value (SV_missval) to IEEE 754 NaN.
 */
static inline double stata_to_rust_value(double stata_val) {
    if (SF_is_missing(stata_val)) {
        return NAN;
    }
    return stata_val;
}

/**
 * Convert an IEEE 754 NaN to the Stata missing value (SV_missval).
 */
static inline double rust_to_stata_value(double rust_val) {
    if (isnan(rust_val)) {
        return SV_missval;
    }
    return rust_val;
}

/* ============================================================================
 * Data Reading Functions
 * ============================================================================ */

/**
 * Read panel dimensions from Stata scalars __trop_n_units and
 * __trop_n_periods.
 *
 * @param n_units_out    [out] N
 * @param n_periods_out  [out] T
 * @return Stata return code (0 = success)
 */
ST_retcode read_dimensions(ST_int *n_units_out, ST_int *n_periods_out);

/**
 * Read a panel variable into a T x N column-major matrix.
 *
 * Observation-to-cell mapping uses the panel and time index variables
 * whose positions are stored in __trop_panel_varindex and
 * __trop_time_varindex.  Cells without a corresponding observation
 * are set to NaN.
 *
 * @param varindex       Variable position in the plugin varlist (1-based)
 * @param n_periods      T
 * @param n_units        N
 * @param out_matrix     [out] T x N column-major matrix
 * @return Stata return code (0 = success)
 */
ST_retcode read_panel_to_matrix(
    ST_int varindex,
    ST_int n_periods,
    ST_int n_units,
    double *out_matrix
);

/**
 * Read a control mask from a Stata variable.
 *
 * Produces a T x N uint8 matrix where 1 = control (D == 0) and
 * 0 = treated or missing.
 *
 * @param varindex       Variable position in the plugin varlist (1-based)
 * @param n_periods      T
 * @param n_units        N
 * @param out_mask       [out] T x N column-major mask
 * @return Stata return code (0 = success)
 */
ST_retcode read_control_mask(
    ST_int varindex,
    ST_int n_periods,
    ST_int n_units,
    unsigned char *out_mask
);

/**
 * Read a lambda grid from a Stata matrix (row or column vector).
 *
 * The caller is responsible for freeing *out_grid.
 *
 * @param matname        Stata matrix name
 * @param out_grid       [out] Allocated grid values
 * @param out_len        [out] Grid length
 * @return Stata return code (0 = success)
 */
ST_retcode read_lambda_grid(
    const char *matname,
    double **out_grid,
    int *out_len
);

/**
 * Replace sentinel infinity values in a lambda grid with operational
 * defaults.
 *
 * Values >= 1e99 are treated as infinity and mapped to:
 *   lambda_time  ->  0.0   (uniform time weights)
 *   lambda_unit  ->  0.0   (uniform unit weights)
 *   lambda_nn    ->  1e10  (nuclear norm penalty large enough to
 *                           drive L toward zero)
 *
 * The threshold (1e99) and replacement for lambda_nn (1e10) must stay
 * in sync with the corresponding Mata constants.
 *
 * @param grid           Grid values (modified in place)
 * @param len            Grid length
 * @param param_type     "time", "unit", or "nn"
 */
void convert_lambda_infinity(
    double *grid,
    int len,
    const char *param_type
);

/**
 * Read a time distance matrix from a Stata matrix.
 *
 * @param matname        Stata matrix name
 * @param n_periods      T
 * @param out_matrix     [out] T x T column-major int64 matrix
 * @return Stata return code (0 = success)
 */
ST_retcode read_time_dist_matrix(
    const char *matname,
    ST_int n_periods,
    int64_t *out_matrix
);

/* ============================================================================
 * Result Writing Functions
 * ============================================================================ */

/**
 * Write a vector to a Stata matrix (row or column).
 *
 * NaN values are converted to Stata missing before storage.
 *
 * @param matname        Stata matrix name
 * @param data           Vector data
 * @param len            Vector length
 * @param is_row         1 for row vector, 0 for column vector
 * @return Stata return code (0 = success)
 */
ST_retcode write_vector_to_matrix(
    const char *matname,
    const double *data,
    int len,
    int is_row
);

/**
 * Write a column-major matrix to a Stata matrix.
 *
 * NaN values are converted to Stata missing before storage.
 *
 * @param matname        Stata matrix name
 * @param data           Column-major matrix data
 * @param nrows          Number of rows
 * @param ncols          Number of columns
 * @return Stata return code (0 = success)
 */
ST_retcode write_matrix_to_stata(
    const char *matname,
    const double *data,
    int nrows,
    int ncols
);

/**
 * stata_get_last_condition_number
 *
 * Returns the condition number from the most recent SVD solve in the
 * covariate WLS step.  Used to populate e(condition_number).
 *
 * @return Condition number (kappa), or NaN if no SVD has been performed.
 */
extern double stata_get_last_condition_number(void);

/**
 * stata_get_last_covariate_rcond
 *
 * Returns the condition number from the most recent covariate WLS solve
 * (X'WX system).  Set on both Cholesky and SVD fallback paths, making it
 * the preferred source for e(condition_number).
 *
 * @return Condition number (kappa), or NaN if no covariate solve occurred.
 */
extern double stata_get_last_covariate_rcond(void);

#endif /* STATA_BRIDGE_H */

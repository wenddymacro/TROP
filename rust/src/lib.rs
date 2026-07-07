//! C-ABI entry points for the TROP (Triply Robust Panel) estimator.
//!
//! Provides `extern "C"` functions that a Stata plugin can call to execute
//! leave-one-out cross-validation, point estimation, and bootstrap inference
//! for both the Twostep and Joint estimation methods.
//!
//! All matrix arguments follow column-major (Fortran) layout so that Stata
//! matrices can be passed without transposition.

#[cfg(not(target_os = "windows"))]
extern crate lapack_src;

pub mod bootstrap;
pub mod distance;
pub mod error;
pub mod estimation;
pub mod loocv;
#[cfg(target_os = "macos")]
pub mod newlapack;
pub mod weights;

pub use error::{TropError, TropResult};

use ndarray::{Array1, Array2, ArrayView2, ShapeBuilder};
use rayon::prelude::*;
use std::cell::RefCell;
use std::slice;

// ---------------------------------------------------------------------------
// Thread-local panic message buffer (P1.1)
// ---------------------------------------------------------------------------

thread_local! {
    static LAST_PANIC_MESSAGE: RefCell<String> = RefCell::new(String::new());
}

// ---------------------------------------------------------------------------
// BLAS thread configuration (Task 28)
// ---------------------------------------------------------------------------

use std::sync::Once;

static BLAS_THREADING_INIT: Once = Once::new();

/// Constrain the platform BLAS/LAPACK backend to a single thread so it does
/// not oversubscribe the CPU against rayon's parallelism (Task 28).
///
/// After Task 27 the LOOCV hot path drives exactly one level of rayon
/// parallelism (either across λ candidates or across chunks of control
/// observations).  Each parallel worker then calls the platform BLAS backend
/// for the per-cell SVD / least-squares solves.  If that backend also spins
/// up its own thread pool the two layers oversubscribe the physical cores and
/// hurt throughput, so we request a single BLAS thread and leave all
/// parallelism to rayon.
///
/// Mechanism / reliability:
///   * macOS (Accelerate / vecLib): `VECLIB_MAXIMUM_THREADS=1`.
///   * Linux (OpenBLAS): `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1`.
///   * Windows (faer, pure-Rust): no external BLAS pool — effectively a no-op.
///
/// These backends read the variables when they lazily initialise their thread
/// pools, which for the Stata plugin happens on the first BLAS call — i.e.
/// inside the first heavy entry point, *after* this `Once` has run.  Setting
/// them here is therefore honoured in the common case.  It is not
/// bullet-proof: if a backend already initialised its pool in the host
/// process before the plugin was first invoked, the variable is ignored.  In
/// that residual case the per-cell SVD dimensions (T × N) are small enough
/// that BLAS-internal threading brings negligible benefit and the
/// oversubscription cost is bounded.  For a hard guarantee users can export
/// `VECLIB_MAXIMUM_THREADS=1` / `OPENBLAS_NUM_THREADS=1` in the shell that
/// launches Stata.
fn configure_blas_threading() {
    BLAS_THREADING_INIT.call_once(|| {
        // Edition 2021: `std::env::set_var` is a safe function.
        #[cfg(target_os = "macos")]
        std::env::set_var("VECLIB_MAXIMUM_THREADS", "1");
        #[cfg(target_os = "linux")]
        {
            std::env::set_var("OPENBLAS_NUM_THREADS", "1");
            std::env::set_var("OMP_NUM_THREADS", "1");
        }
    });
}

// ---------------------------------------------------------------------------
// Column-major pointer conversion helpers
// ---------------------------------------------------------------------------

/// Constructs a 2-D array view from a raw `f64` pointer in column-major order.
///
/// # Safety
/// `ptr` must be non-null and point to at least `rows * cols` contiguous `f64` values
/// whose memory remains valid for the lifetime of the returned view.
#[inline]
unsafe fn ptr_to_array2<'a>(ptr: *const f64, rows: usize, cols: usize) -> ArrayView2<'a, f64> {
    let slice = slice::from_raw_parts(ptr, rows * cols);
    ArrayView2::from_shape((rows, cols).f(), slice).unwrap()
}

/// Constructs a 2-D array view from a raw `i64` pointer in column-major order.
///
/// # Safety
/// Same requirements as [`ptr_to_array2`].
#[inline]
unsafe fn ptr_to_array2_i64<'a>(ptr: *const i64, rows: usize, cols: usize) -> ArrayView2<'a, i64> {
    let slice = slice::from_raw_parts(ptr, rows * cols);
    ArrayView2::from_shape((rows, cols).f(), slice).unwrap()
}

/// Constructs a 2-D array view from a raw `u8` pointer in column-major order.
///
/// # Safety
/// Same requirements as [`ptr_to_array2`].
#[inline]
unsafe fn ptr_to_array2_u8<'a>(ptr: *const u8, rows: usize, cols: usize) -> ArrayView2<'a, u8> {
    let slice = slice::from_raw_parts(ptr, rows * cols);
    ArrayView2::from_shape((rows, cols).f(), slice).unwrap()
}

/// Writes a 2-D `f64` array to a raw pointer in column-major order.
///
/// # Safety
/// `ptr` must be non-null and point to a buffer of at least `arr.len()` elements.
#[inline]
unsafe fn array2_to_ptr(arr: &Array2<f64>, ptr: *mut f64) {
    let slice = slice::from_raw_parts_mut(ptr, arr.len());
    for ((t, i), val) in arr.indexed_iter() {
        let idx = i * arr.nrows() + t;
        slice[idx] = *val;
    }
}

/// Catches any unwinding panic inside `$body` and converts it to an error code.
/// Stores the panic message in a thread-local buffer accessible via
/// `trop_get_last_panic_message` (P1.1).
macro_rules! catch_panic {
    ($body:expr) => {{
        // Clear previous panic message at entry
        LAST_PANIC_MESSAGE.with(|m| m.borrow_mut().clear());
        // Task 28: pin the BLAS backend to a single thread before any solve so
        // it does not oversubscribe against rayon (idempotent, `Once`-guarded).
        configure_blas_threading();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| $body));
        match result {
            Ok(r) => r,
            Err(panic_info) => {
                let msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
                    s.to_string()
                } else if let Some(s) = panic_info.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "unknown panic".to_string()
                };
                LAST_PANIC_MESSAGE.with(|m| {
                    *m.borrow_mut() = msg;
                });
                TropError::RustPanic.code()
            }
        }
    }};
}

// ---------------------------------------------------------------------------
// ABI version handshake (P3.3)
// ---------------------------------------------------------------------------

/// Returns the ABI version of this compiled library.
///
/// The C bridge calls this at first invocation to verify that the plugin and
/// dynamic library were compiled from the same source revision.  The check is
/// soft — a mismatch emits a warning but does not abort, so stale
/// `.dylib`/`.so` files produce a diagnostic rather than a hard failure.
#[no_mangle]
pub extern "C" fn trop_abi_version() -> i32 {
    2
}

// ---------------------------------------------------------------------------
// Panic message retrieval (P1.1 FFI export)
// ---------------------------------------------------------------------------

/// Retrieve the last Rust panic message captured by `catch_panic!`.
///
/// Copies at most `buf_len - 1` bytes of the message into `buf` and
/// null-terminates it.  Returns the actual number of bytes written
/// (excluding the null terminator), or 0 if `buf` is null / `buf_len <= 0`.
///
/// An empty string (return value 0 with valid buffer) indicates that no
/// panic has occurred on this thread since the last invocation.
///
/// # Safety
/// `buf` must point to a writable buffer of at least `buf_len` bytes.
#[no_mangle]
pub unsafe extern "C" fn trop_get_last_panic_message(
    buf: *mut std::os::raw::c_char,
    buf_len: i32,
) -> i32 {
    if buf.is_null() || buf_len <= 0 {
        return 0;
    }
    LAST_PANIC_MESSAGE.with(|m| {
        let msg = m.borrow();
        let bytes = msg.as_bytes();
        let copy_len = std::cmp::min(bytes.len(), (buf_len - 1) as usize);
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf as *mut u8, copy_len);
        *buf.add(copy_len) = 0; // null terminator
        copy_len as i32
    })
}

// ---------------------------------------------------------------------------
// Twostep method — C ABI exports
// ---------------------------------------------------------------------------

/// Leave-one-out cross-validation grid search for Twostep tuning parameters
/// (λ_time, λ_unit, λ_nn).
///
/// Searches over the Cartesian product of the three grids and writes the
/// best triple, its LOOCV score, and diagnostic counters to the output
/// pointers.
///
/// # Returns
/// `0` on success; a non-zero `TropError` code otherwise.
///
/// # Safety
/// All pointers must be non-null and point to properly sized buffers.
#[no_mangle]
pub unsafe extern "C" fn stata_loocv_grid_search(
    y_ptr: *const f64,
    d_ptr: *const f64,
    control_mask_ptr: *const u8,
    time_dist_ptr: *const i64,
    n_periods: i32,
    n_units: i32,
    lambda_time_grid_ptr: *const f64,
    lambda_time_grid_len: i32,
    lambda_unit_grid_ptr: *const f64,
    lambda_unit_grid_len: i32,
    lambda_nn_grid_ptr: *const f64,
    lambda_nn_grid_len: i32,
    max_iter: i32,
    tol: f64,
    best_lambda_time_out: *mut f64,
    best_lambda_unit_out: *mut f64,
    best_lambda_nn_out: *mut f64,
    best_score_out: *mut f64,
    n_valid_out: *mut i32,
    n_attempted_out: *mut i32,
    first_failed_t_out: *mut i32,
    first_failed_i_out: *mut i32,
    stage1_lambda_time_out: *mut f64,
    stage1_lambda_unit_out: *mut f64,
    stage1_lambda_nn_out: *mut f64,
    // Covariate support
    x_ptr: *const f64,
    n_covariates: i32,
) -> i32 {
    catch_panic!({
        if y_ptr.is_null()
            || d_ptr.is_null()
            || control_mask_ptr.is_null()
            || time_dist_ptr.is_null()
            || lambda_time_grid_ptr.is_null()
            || lambda_unit_grid_ptr.is_null()
            || lambda_nn_grid_ptr.is_null()
            || best_lambda_time_out.is_null()
            || best_lambda_unit_out.is_null()
            || best_lambda_nn_out.is_null()
            || best_score_out.is_null()
        {
            return TropError::NullPointer.code();
        }

        if n_periods <= 0 || n_units <= 0 {
            return TropError::InvalidDimension.code();
        }

        let np = n_periods as usize;
        let nu = n_units as usize;

        // Construct covariate matrix view (column-major)
        let x_view: Option<ArrayView2<f64>> = if n_covariates > 0 && !x_ptr.is_null() {
            let n_obs = np * nu;
            let n_cov = n_covariates as usize;
            let x_slice = std::slice::from_raw_parts(x_ptr, n_obs * n_cov);
            Some(ArrayView2::from_shape((n_obs, n_cov).f(), x_slice).unwrap())
        } else {
            None
        };

        let y = ptr_to_array2(y_ptr, np, nu);
        let d = ptr_to_array2(d_ptr, np, nu);
        let control_mask = ptr_to_array2_u8(control_mask_ptr, np, nu);
        let time_dist = ptr_to_array2_i64(time_dist_ptr, np, np);

        let lambda_time_grid =
            slice::from_raw_parts(lambda_time_grid_ptr, lambda_time_grid_len as usize);
        let lambda_unit_grid =
            slice::from_raw_parts(lambda_unit_grid_ptr, lambda_unit_grid_len as usize);
        let lambda_nn_grid = slice::from_raw_parts(lambda_nn_grid_ptr, lambda_nn_grid_len as usize);

        let (
            (
                best_time,
                best_unit,
                best_nn,
                best_score,
                n_valid,
                n_attempted,
                first_failed,
            ),
            stage1_time,
            stage1_unit,
            stage1_nn,
        ) = match loocv::loocv_grid_search_with_stage1(
            &y,
            &d,
            &control_mask,
            &time_dist,
            lambda_time_grid,
            lambda_unit_grid,
            lambda_nn_grid,
            max_iter as usize,
            tol,
            x_view.as_ref(),
        ) {
            Ok(v) => v,
            Err(e) => return e.code(),
        };

        *best_lambda_time_out = best_time;
        *best_lambda_unit_out = best_unit;
        *best_lambda_nn_out = best_nn;
        *best_score_out = best_score;

        if !n_valid_out.is_null() {
            *n_valid_out = n_valid as i32;
        }
        if !n_attempted_out.is_null() {
            *n_attempted_out = n_attempted as i32;
        }
        if !first_failed_t_out.is_null() {
            *first_failed_t_out = match first_failed {
                Some((t, _)) => t as i32,
                None => -1,
            };
        }
        if !first_failed_i_out.is_null() {
            *first_failed_i_out = match first_failed {
                Some((_, i)) => i as i32,
                None => -1,
            };
        }

        // Stage-1 univariate initialisation (paper Footnote 2).  Each out
        // pointer is independently NULL-safe so callers that do not care
        // about Stage-1 diagnostics can pass NULL for any subset.
        if !stage1_lambda_time_out.is_null() {
            *stage1_lambda_time_out = stage1_time;
        }
        if !stage1_lambda_unit_out.is_null() {
            *stage1_lambda_unit_out = stage1_unit;
        }
        if !stage1_lambda_nn_out.is_null() {
            *stage1_lambda_nn_out = stage1_nn;
        }

        TropError::Success.code()
    })
}

/// Exhaustive (Cartesian) LOOCV grid search for Twostep tuning parameters.
///
/// Evaluates every (λ_time, λ_unit, λ_nn) combination in parallel and
/// returns the global grid minimum under the tie-breaker rules used by
/// [`stata_loocv_grid_search`].  On small panels where the coordinate-descent
/// (cycling) path can converge to a local minimum, the exhaustive search
/// gives a platform- and BLAS-independent λ selection at the cost of
/// O(|Λ_time| · |Λ_unit| · |Λ_nn|) LOOCV evaluations.
///
/// # Returns
/// `0` on success; a non-zero `TropError` code otherwise.
///
/// # Safety
/// All pointers must be non-null and point to properly sized buffers.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn stata_loocv_grid_search_exhaustive(
    y_ptr: *const f64,
    d_ptr: *const f64,
    control_mask_ptr: *const u8,
    time_dist_ptr: *const i64,
    n_periods: i32,
    n_units: i32,
    lambda_time_grid_ptr: *const f64,
    lambda_time_grid_len: i32,
    lambda_unit_grid_ptr: *const f64,
    lambda_unit_grid_len: i32,
    lambda_nn_grid_ptr: *const f64,
    lambda_nn_grid_len: i32,
    max_iter: i32,
    tol: f64,
    best_lambda_time_out: *mut f64,
    best_lambda_unit_out: *mut f64,
    best_lambda_nn_out: *mut f64,
    best_score_out: *mut f64,
    n_valid_out: *mut i32,
    n_attempted_out: *mut i32,
    first_failed_t_out: *mut i32,
    first_failed_i_out: *mut i32,
    // Covariate support
    x_ptr: *const f64,
    n_covariates: i32,
) -> i32 {
    catch_panic!({
        if y_ptr.is_null()
            || d_ptr.is_null()
            || control_mask_ptr.is_null()
            || time_dist_ptr.is_null()
            || lambda_time_grid_ptr.is_null()
            || lambda_unit_grid_ptr.is_null()
            || lambda_nn_grid_ptr.is_null()
            || best_lambda_time_out.is_null()
            || best_lambda_unit_out.is_null()
            || best_lambda_nn_out.is_null()
            || best_score_out.is_null()
        {
            return TropError::NullPointer.code();
        }

        let np = n_periods as usize;
        let nu = n_units as usize;

        // Construct covariate matrix view (column-major)
        let x_view: Option<ArrayView2<f64>> = if n_covariates > 0 && !x_ptr.is_null() {
            let n_obs = np * nu;
            let n_cov = n_covariates as usize;
            let x_slice = std::slice::from_raw_parts(x_ptr, n_obs * n_cov);
            Some(ArrayView2::from_shape((n_obs, n_cov).f(), x_slice).unwrap())
        } else {
            None
        };

        let y = ptr_to_array2(y_ptr, np, nu);
        let d = ptr_to_array2(d_ptr, np, nu);
        let control_mask = ptr_to_array2_u8(control_mask_ptr, np, nu);
        let time_dist = ptr_to_array2_i64(time_dist_ptr, np, np);

        let lambda_time_grid =
            slice::from_raw_parts(lambda_time_grid_ptr, lambda_time_grid_len as usize);
        let lambda_unit_grid =
            slice::from_raw_parts(lambda_unit_grid_ptr, lambda_unit_grid_len as usize);
        let lambda_nn_grid = slice::from_raw_parts(lambda_nn_grid_ptr, lambda_nn_grid_len as usize);

        let (
            best_time,
            best_unit,
            best_nn,
            best_score,
            n_valid,
            n_attempted,
            first_failed,
        ) = match loocv::loocv_grid_search_exhaustive(
            &y,
            &d,
            &control_mask,
            &time_dist,
            lambda_time_grid,
            lambda_unit_grid,
            lambda_nn_grid,
            max_iter as usize,
            tol,
            x_view.as_ref(),
        ) {
            Ok(v) => v,
            Err(e) => return e.code(),
        };

        *best_lambda_time_out = best_time;
        *best_lambda_unit_out = best_unit;
        *best_lambda_nn_out = best_nn;
        *best_score_out = best_score;

        if !n_valid_out.is_null() {
            *n_valid_out = n_valid as i32;
        }
        if !n_attempted_out.is_null() {
            *n_attempted_out = n_attempted as i32;
        }
        if !first_failed_t_out.is_null() {
            *first_failed_t_out = match first_failed {
                Some((t, _)) => t as i32,
                None => -1,
            };
        }
        if !first_failed_i_out.is_null() {
            *first_failed_i_out = match first_failed {
                Some((_, i)) => i as i32,
                None => -1,
            };
        }

        TropError::Success.code()
    })
}

/// Twostep point estimation with fixed tuning parameters.
///
/// For each treated observation (t, i), computes observation-specific weights,
/// fits the additive model Y = α_i + β_t + L_{ti} + τ_{ti} D_{ti} via
/// penalised SVD, and returns the ATT (average of individual τ values) together
/// with the averaged parameter matrices.
///
/// # Returns
/// `0` on success; a non-zero `TropError` code otherwise.
///
/// # Safety
/// All pointers must be non-null and point to properly sized buffers.
#[no_mangle]
pub unsafe extern "C" fn stata_estimate_twostep(
    y_ptr: *const f64,
    d_ptr: *const f64,
    control_mask_ptr: *const u8,
    time_dist_ptr: *const i64,
    n_periods: i32,
    n_units: i32,
    lambda_time: f64,
    lambda_unit: f64,
    lambda_nn: f64,
    max_iter: i32,
    tol: f64,
    att_out: *mut f64,
    tau_ptr: *mut f64,
    alpha_ptr: *mut f64,
    beta_ptr: *mut f64,
    l_ptr: *mut f64,
    n_treated_out: *mut i32,
    n_iterations_out: *mut i32,
    converged_out: *mut i32,
    // Optional per-observation diagnostics.  Both nullable; if non-null each
    // must be pre-allocated to hold N_treated i32 values.  The ordering
    // matches the outer iteration `for t in 0..T { for i in 0..N { if D=1 } }`
    // so Mata can reconstruct (t,i) indices by iterating D in the same way.
    converged_by_obs_ptr: *mut i32,
    n_iters_by_obs_ptr: *mut i32,
    // Covariate support
    x_ptr: *const f64,
    n_covariates: i32,
    gamma_out: *mut f64,
) -> i32 {
    catch_panic!({
        if y_ptr.is_null()
            || d_ptr.is_null()
            || control_mask_ptr.is_null()
            || time_dist_ptr.is_null()
            || att_out.is_null()
        {
            return TropError::NullPointer.code();
        }

        if n_periods <= 0 || n_units <= 0 {
            return TropError::InvalidDimension.code();
        }

        let np = n_periods as usize;
        let nu = n_units as usize;

        // Construct covariate matrix view (column-major)
        let x_view: Option<ArrayView2<f64>> = if n_covariates > 0 && !x_ptr.is_null() {
            let n_obs = np * nu;
            let n_cov = n_covariates as usize;
            let x_slice = std::slice::from_raw_parts(x_ptr, n_obs * n_cov);
            Some(ArrayView2::from_shape((n_obs, n_cov).f(), x_slice).unwrap())
        } else {
            None
        };

        let y = ptr_to_array2(y_ptr, np, nu);
        let d = ptr_to_array2(d_ptr, np, nu);
        let control_mask = ptr_to_array2_u8(control_mask_ptr, np, nu);
        let time_dist = ptr_to_array2_i64(time_dist_ptr, np, np);

        // Collect (period, unit) indices of treated observations.
        let mut treated_obs: Vec<(usize, usize)> = Vec::new();
        for t in 0..np {
            for i in 0..nu {
                if d[[t, i]] == 1.0 {
                    treated_obs.push((t, i));
                }
            }
        }

        if treated_obs.is_empty() {
            return TropError::NoTreated.code();
        }

        // Map +Inf → 0 (no penalty) and +Inf for λ_nn → large finite cap.
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

        // Per-observation estimation (parallelised over treated cells).
        struct ObsResult {
            tau: f64,
            alpha: Array1<f64>,
            beta: Array1<f64>,
            l: Array2<f64>,
            gamma: Option<Array1<f64>>,
            n_iters: usize,
            converged: bool,
        }

        // Share a single UnitDistanceCache across all treated observations.
        // Even with a modest N_treated (~10–30), avoiding the per-call
        // O(T) pairwise distance computation is worthwhile when T is
        // large (e.g. PWT at T=48).
        let dist_cache = distance::UnitDistanceCache::build(&y, &d);

        let obs_results: Vec<Option<ObsResult>> = treated_obs
            .par_iter()
            .map(|(t, i)| {
                let weight_matrix = weights::compute_weight_matrix_cached(
                    &y, &d, &dist_cache, np, nu, *i, *t, lt_eff, lu_eff, &time_dist,
                );

                match estimation::estimate_model(
                    &y,
                    &control_mask,
                    &weight_matrix.view(),
                    ln_eff,
                    np,
                    nu,
                    max_iter as usize,
                    tol,
                    None,
                    None,
                    x_view.as_ref(),
                    None,
                ) {
                    Some((alpha, beta, l, n_iters, did_converge, gamma)) => {
                        let tau = y[[*t, *i]] - alpha[*i] - beta[*t] - l[[*t, *i]];
                        Some(ObsResult {
                            tau,
                            alpha,
                            beta,
                            l,
                            gamma,
                            n_iters,
                            converged: did_converge,
                        })
                    }
                    None => None,
                }
            })
            .collect();

        // Aggregate: average α, β, L across successful observations.  The
        // per-obs diagnostics (`converged_by_obs`, `n_iters_by_obs`) are
        // aligned with `treated_obs`: one entry per (t,i) in the same order,
        // with -1 entries marking observations whose solver returned None.
        let mut tau_values: Vec<f64> = Vec::with_capacity(treated_obs.len());
        let mut alpha_sum = Array1::<f64>::zeros(nu);
        let mut beta_sum = Array1::<f64>::zeros(np);
        let mut l_sum = Array2::<f64>::zeros((np, nu));
        let mut gamma_sum: Option<Array1<f64>> = None;
        let mut n_successful: usize = 0;
        let mut max_iters: usize = 0;
        let mut all_successful_converged = true;
        let mut converged_by_obs: Vec<i32> = Vec::with_capacity(treated_obs.len());
        let mut n_iters_by_obs: Vec<i32> = Vec::with_capacity(treated_obs.len());

        for result in obs_results {
            match result {
                Some(obs) => {
                    tau_values.push(obs.tau);
                    alpha_sum += &obs.alpha;
                    beta_sum += &obs.beta;
                    l_sum += &obs.l;
                    if let Some(ref g) = obs.gamma {
                        let gs = gamma_sum.get_or_insert_with(|| Array1::<f64>::zeros(g.len()));
                        *gs += g;
                    }
                    n_successful += 1;
                    if obs.n_iters > max_iters {
                        max_iters = obs.n_iters;
                    }
                    if !obs.converged {
                        all_successful_converged = false;
                    }
                    converged_by_obs.push(if obs.converged { 1 } else { 0 });
                    n_iters_by_obs.push(obs.n_iters as i32);
                }
                None => {
                    // Solver failed (e.g. SVD error); keep slot alignment with
                    // `treated_obs` and mark as -1 for Mata/Stata side.
                    converged_by_obs.push(-1);
                    n_iters_by_obs.push(-1);
                }
            }
        }

        if tau_values.is_empty() {
            return TropError::Convergence.code();
        }

        let att = tau_values.iter().sum::<f64>() / tau_values.len() as f64;

        let n_succ_f64 = n_successful as f64;
        let all_alpha = alpha_sum / n_succ_f64;
        let all_beta = beta_sum / n_succ_f64;
        let all_l = l_sum / n_succ_f64;
        let all_gamma = gamma_sum.map(|g| g / n_succ_f64);

        *att_out = att;
        *n_treated_out = tau_values.len() as i32;
        *n_iterations_out = max_iters as i32;
        *converged_out = if all_successful_converged { 1 } else { 0 };

        // Write per-obs diagnostics when the caller requests them.
        if !converged_by_obs_ptr.is_null() {
            let slot = slice::from_raw_parts_mut(converged_by_obs_ptr, converged_by_obs.len());
            slot.copy_from_slice(&converged_by_obs);
        }
        if !n_iters_by_obs_ptr.is_null() {
            let slot = slice::from_raw_parts_mut(n_iters_by_obs_ptr, n_iters_by_obs.len());
            slot.copy_from_slice(&n_iters_by_obs);
        }

        if !tau_ptr.is_null() {
            let tau_slice = slice::from_raw_parts_mut(tau_ptr, tau_values.len());
            tau_slice.copy_from_slice(&tau_values);
        }

        if !alpha_ptr.is_null() {
            let alpha_slice = slice::from_raw_parts_mut(alpha_ptr, nu);
            alpha_slice.copy_from_slice(all_alpha.as_slice().unwrap());
        }

        if !beta_ptr.is_null() {
            let beta_slice = slice::from_raw_parts_mut(beta_ptr, np);
            beta_slice.copy_from_slice(all_beta.as_slice().unwrap());
        }

        if !l_ptr.is_null() {
            array2_to_ptr(&all_l, l_ptr);
        }

        // Write averaged covariate coefficients
        if !gamma_out.is_null() && n_covariates > 0 {
            if let Some(ref gamma_vec) = all_gamma {
                let gamma_slice = std::slice::from_raw_parts_mut(gamma_out, n_covariates as usize);
                for (idx, val) in gamma_vec.iter().enumerate() {
                    gamma_slice[idx] = *val;
                }
            } else {
                let gamma_slice = std::slice::from_raw_parts_mut(gamma_out, n_covariates as usize);
                for val in gamma_slice.iter_mut() {
                    *val = 0.0;
                }
            }
        }

        TropError::Success.code()
    })
}

/// Bootstrap variance estimation for the Twostep method.
///
/// Resamples units with replacement `n_bootstrap` times, re-estimates the ATT
/// in each replicate, and returns the standard error, percentile confidence
/// interval, and the full vector of bootstrap estimates.
///
/// `ddof` selects the variance denominator:
///   - `1` (default) → Bessel-corrected sample variance `1/(B−1)`.
///   - `0`           → paper Algorithm 3 population variance `1/B`.
///   - Any other value collapses to `1`.
///
/// # Returns
/// `0` on success; a non-zero `TropError` code otherwise.
///
/// # Safety
/// All pointers must be non-null and point to properly sized buffers.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn stata_bootstrap_trop_variance(
    y_ptr: *const f64,
    d_ptr: *const f64,
    control_mask_ptr: *const u8,
    time_dist_ptr: *const i64,
    n_periods: i32,
    n_units: i32,
    lambda_time: f64,
    lambda_unit: f64,
    lambda_nn: f64,
    n_bootstrap: i32,
    max_iter: i32,
    tol: f64,
    seed: u64,
    alpha: f64,
    ddof: i32,
    estimates_ptr: *mut f64,
    se_out: *mut f64,
    ci_lower_out: *mut f64,
    ci_upper_out: *mut f64,
    n_valid_out: *mut i32,
    // Covariate support
    x_ptr: *const f64,
    n_covariates: i32,
) -> i32 {
    catch_panic!({
        if y_ptr.is_null()
            || d_ptr.is_null()
            || control_mask_ptr.is_null()
            || time_dist_ptr.is_null()
            || se_out.is_null()
        {
            return TropError::NullPointer.code();
        }

        if n_periods <= 0 || n_units <= 0 {
            return TropError::InvalidDimension.code();
        }

        let np = n_periods as usize;
        let nu = n_units as usize;

        // Construct covariate matrix view (column-major)
        let x_view: Option<ArrayView2<f64>> = if n_covariates > 0 && !x_ptr.is_null() {
            let n_obs = np * nu;
            let n_cov = n_covariates as usize;
            let x_slice = std::slice::from_raw_parts(x_ptr, n_obs * n_cov);
            Some(ArrayView2::from_shape((n_obs, n_cov).f(), x_slice).unwrap())
        } else {
            None
        };

        let y = ptr_to_array2(y_ptr, np, nu);
        let d = ptr_to_array2(d_ptr, np, nu);
        let control_mask = ptr_to_array2_u8(control_mask_ptr, np, nu);
        let time_dist = ptr_to_array2_i64(time_dist_ptr, np, np);

        let alpha_eff = if alpha <= 0.0 || alpha >= 1.0 {
            0.05
        } else {
            alpha
        };

        // Clamp ddof into the {0, 1} set; any out-of-range value defaults to 1.
        let ddof_u8: u8 = if ddof == 0 { 0 } else { 1 };

        let result = bootstrap::bootstrap_trop_variance_full(
            &y,
            &d,
            &control_mask,
            &time_dist,
            lambda_time,
            lambda_unit,
            lambda_nn,
            n_bootstrap as usize,
            max_iter as usize,
            tol,
            seed,
            alpha_eff,
            ddof_u8,
            x_view.as_ref(),
        );

        *se_out = result.se;

        if !ci_lower_out.is_null() {
            *ci_lower_out = result.ci_lower;
        }
        if !ci_upper_out.is_null() {
            *ci_upper_out = result.ci_upper;
        }
        if !n_valid_out.is_null() {
            *n_valid_out = result.n_valid as i32;
        }

        if !estimates_ptr.is_null() {
            let est_slice = slice::from_raw_parts_mut(estimates_ptr, result.estimates.len());
            est_slice.copy_from_slice(&result.estimates);
        }

        TropError::Success.code()
    })
}

// ---------------------------------------------------------------------------
// Joint method — C ABI exports
// ---------------------------------------------------------------------------

/// Coordinate-descent LOOCV search for Joint method tuning parameters
/// (λ_time, λ_unit, λ_nn).
///
/// Performs univariate sweeps along each parameter axis, then cycles until
/// the selected triple stabilises or `max_cycles` is reached.
///
/// # Returns
/// `0` on success; a non-zero `TropError` code otherwise.
///
/// # Safety
/// All pointers must be non-null and point to properly sized buffers.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn stata_loocv_cycling_search_joint(
    y_ptr: *const f64,
    d_ptr: *const f64,
    control_mask_ptr: *const u8,
    n_periods: i32,
    n_units: i32,
    lambda_time_grid_ptr: *const f64,
    lambda_time_grid_len: i32,
    lambda_unit_grid_ptr: *const f64,
    lambda_unit_grid_len: i32,
    lambda_nn_grid_ptr: *const f64,
    lambda_nn_grid_len: i32,
    max_iter: i32,
    tol: f64,
    max_cycles: i32,
    best_lambda_time_out: *mut f64,
    best_lambda_unit_out: *mut f64,
    best_lambda_nn_out: *mut f64,
    best_score_out: *mut f64,
    n_valid_out: *mut i32,
    n_attempted_out: *mut i32,
    first_failed_t_out: *mut i32,
    first_failed_i_out: *mut i32,
    stage1_lambda_time_out: *mut f64,
    stage1_lambda_unit_out: *mut f64,
    stage1_lambda_nn_out: *mut f64,
    // Covariate support
    x_ptr: *const f64,
    n_covariates: i32,
) -> i32 {
    catch_panic!({
        if y_ptr.is_null()
            || d_ptr.is_null()
            || control_mask_ptr.is_null()
            || lambda_time_grid_ptr.is_null()
            || lambda_unit_grid_ptr.is_null()
            || lambda_nn_grid_ptr.is_null()
            || best_lambda_time_out.is_null()
            || best_lambda_unit_out.is_null()
            || best_lambda_nn_out.is_null()
            || best_score_out.is_null()
        {
            return TropError::NullPointer.code();
        }

        let np = n_periods as usize;
        let nu = n_units as usize;

        // Construct covariate matrix view (column-major)
        let x_view: Option<ArrayView2<f64>> = if n_covariates > 0 && !x_ptr.is_null() {
            let n_obs = np * nu;
            let n_cov = n_covariates as usize;
            let x_slice = std::slice::from_raw_parts(x_ptr, n_obs * n_cov);
            Some(ArrayView2::from_shape((n_obs, n_cov).f(), x_slice).unwrap())
        } else {
            None
        };

        let y = ptr_to_array2(y_ptr, np, nu);
        let d = ptr_to_array2(d_ptr, np, nu);
        let control_mask = ptr_to_array2_u8(control_mask_ptr, np, nu);

        // Defence-in-depth: the joint estimator's global weight matrix is
        // only well-defined when treatment is simultaneously adopted.  The
        // Stata layer already rejects staggered D, but if we arrive here
        // with a non-conforming matrix we prefer a clean error over a
        // silently wrong `treated_periods`.
        if let Err(err) = loocv::check_simultaneous_adoption(&d) {
            return err.code();
        }

        let lambda_time_grid =
            slice::from_raw_parts(lambda_time_grid_ptr, lambda_time_grid_len as usize);
        let lambda_unit_grid =
            slice::from_raw_parts(lambda_unit_grid_ptr, lambda_unit_grid_len as usize);
        let lambda_nn_grid = slice::from_raw_parts(lambda_nn_grid_ptr, lambda_nn_grid_len as usize);

        let (
            (
                best_time,
                best_unit,
                best_nn,
                best_score,
                n_valid,
                n_attempted,
                first_failed,
            ),
            stage1_time,
            stage1_unit,
            stage1_nn,
        ) = loocv::loocv_cycling_search_joint_with_stage1(
            &y,
            &d,
            &control_mask,
            lambda_time_grid,
            lambda_unit_grid,
            lambda_nn_grid,
            max_iter as usize,
            tol,
            max_cycles as usize,
            x_view.as_ref(),
        );

        *best_lambda_time_out = best_time;
        *best_lambda_unit_out = best_unit;
        *best_lambda_nn_out = best_nn;
        *best_score_out = best_score;

        if !n_valid_out.is_null() {
            *n_valid_out = n_valid as i32;
        }
        if !n_attempted_out.is_null() {
            *n_attempted_out = n_attempted as i32;
        }
        if !first_failed_t_out.is_null() {
            *first_failed_t_out = match first_failed {
                Some((t, _)) => t as i32,
                None => -1,
            };
        }
        if !first_failed_i_out.is_null() {
            *first_failed_i_out = match first_failed {
                Some((_, i)) => i as i32,
                None => -1,
            };
        }

        // Stage-1 univariate initialisation (paper Footnote 2).  NULL-safe.
        if !stage1_lambda_time_out.is_null() {
            *stage1_lambda_time_out = stage1_time;
        }
        if !stage1_lambda_unit_out.is_null() {
            *stage1_lambda_unit_out = stage1_unit;
        }
        if !stage1_lambda_nn_out.is_null() {
            *stage1_lambda_nn_out = stage1_nn;
        }

        TropError::Success.code()
    })
}

/// Exhaustive (Cartesian) LOOCV grid search for Joint method tuning parameters
/// (λ_time, λ_unit, λ_nn).
///
/// Evaluates every combination in the Cartesian product of the three grids in
/// parallel and returns the triple that minimises the LOOCV criterion Q(λ).
/// Complexity is O(|Λ_time| · |Λ_unit| · |Λ_nn|).  Prefer this over the
/// coordinate-descent variant when the grid is small enough to afford it,
/// as it returns the exact grid argmin of Q(λ) (paper Algorithm 2 step 5)
/// rather than a coordinate-descent polish of Q(λ) on a possibly non-convex
/// surface.
///
/// The signature mirrors [`stata_loocv_cycling_search_joint`] with the sole
/// exception that `max_cycles` is absent.
///
/// # Returns
/// `0` on success; a non-zero `TropError` code otherwise.
///
/// # Safety
/// All pointers must be non-null and point to properly sized buffers.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn stata_loocv_grid_search_joint(
    y_ptr: *const f64,
    d_ptr: *const f64,
    control_mask_ptr: *const u8,
    n_periods: i32,
    n_units: i32,
    lambda_time_grid_ptr: *const f64,
    lambda_time_grid_len: i32,
    lambda_unit_grid_ptr: *const f64,
    lambda_unit_grid_len: i32,
    lambda_nn_grid_ptr: *const f64,
    lambda_nn_grid_len: i32,
    max_iter: i32,
    tol: f64,
    best_lambda_time_out: *mut f64,
    best_lambda_unit_out: *mut f64,
    best_lambda_nn_out: *mut f64,
    best_score_out: *mut f64,
    n_valid_out: *mut i32,
    n_attempted_out: *mut i32,
    first_failed_t_out: *mut i32,
    first_failed_i_out: *mut i32,
    // Covariate support
    x_ptr: *const f64,
    n_covariates: i32,
) -> i32 {
    catch_panic!({
        if y_ptr.is_null()
            || d_ptr.is_null()
            || control_mask_ptr.is_null()
            || lambda_time_grid_ptr.is_null()
            || lambda_unit_grid_ptr.is_null()
            || lambda_nn_grid_ptr.is_null()
            || best_lambda_time_out.is_null()
            || best_lambda_unit_out.is_null()
            || best_lambda_nn_out.is_null()
            || best_score_out.is_null()
        {
            return TropError::NullPointer.code();
        }

        if n_periods <= 0 || n_units <= 0 {
            return TropError::InvalidDimension.code();
        }

        let np = n_periods as usize;
        let nu = n_units as usize;

        // Construct covariate matrix view (column-major)
        let x_view: Option<ArrayView2<f64>> = if n_covariates > 0 && !x_ptr.is_null() {
            let n_obs = np * nu;
            let n_cov = n_covariates as usize;
            let x_slice = std::slice::from_raw_parts(x_ptr, n_obs * n_cov);
            Some(ArrayView2::from_shape((n_obs, n_cov).f(), x_slice).unwrap())
        } else {
            None
        };

        let y = ptr_to_array2(y_ptr, np, nu);
        let d = ptr_to_array2(d_ptr, np, nu);
        let control_mask = ptr_to_array2_u8(control_mask_ptr, np, nu);

        // Defence-in-depth: joint LOOCV is only well-defined under
        // simultaneous adoption; mirror the guard in the cycling entry.
        if let Err(err) = loocv::check_simultaneous_adoption(&d) {
            return err.code();
        }

        let lambda_time_grid =
            slice::from_raw_parts(lambda_time_grid_ptr, lambda_time_grid_len as usize);
        let lambda_unit_grid =
            slice::from_raw_parts(lambda_unit_grid_ptr, lambda_unit_grid_len as usize);
        let lambda_nn_grid = slice::from_raw_parts(lambda_nn_grid_ptr, lambda_nn_grid_len as usize);

        let (
            best_time,
            best_unit,
            best_nn,
            best_score,
            n_valid,
            n_attempted,
            first_failed,
        ) = loocv::loocv_grid_search_joint(
            &y,
            &d,
            &control_mask,
            lambda_time_grid,
            lambda_unit_grid,
            lambda_nn_grid,
            max_iter as usize,
            tol,
            x_view.as_ref(),
        );

        *best_lambda_time_out = best_time;
        *best_lambda_unit_out = best_unit;
        *best_lambda_nn_out = best_nn;
        *best_score_out = best_score;

        if !n_valid_out.is_null() {
            *n_valid_out = n_valid as i32;
        }
        if !n_attempted_out.is_null() {
            *n_attempted_out = n_attempted as i32;
        }
        if !first_failed_t_out.is_null() {
            *first_failed_t_out = match first_failed {
                Some((t, _)) => t as i32,
                None => -1,
            };
        }
        if !first_failed_i_out.is_null() {
            *first_failed_i_out = match first_failed {
                Some((_, i)) => i as i32,
                None => -1,
            };
        }

        TropError::Success.code()
    })
}

/// Joint point estimation with fixed tuning parameters.
///
/// Computes global weights δ, then solves the weighted least-squares problem
/// Y = μ + α_i + β_t + L_{ti} + τ D_{ti}.  When λ_nn is effectively
/// infinite the low-rank component L is dropped and a direct WLS solve is
/// used; otherwise coordinate descent alternates between (μ, α, β, τ) and
/// the nuclear-norm–penalised L.
///
/// # Returns
/// `0` on success; a non-zero `TropError` code otherwise.
///
/// # Safety
/// All pointers must be non-null and point to properly sized buffers.
#[no_mangle]
pub unsafe extern "C" fn stata_estimate_joint(
    y_ptr: *const f64,
    d_ptr: *const f64,
    n_periods: i32,
    n_units: i32,
    lambda_time: f64,
    lambda_unit: f64,
    lambda_nn: f64,
    max_iter: i32,
    tol: f64,
    tau_out: *mut f64,
    mu_out: *mut f64,
    alpha_ptr: *mut f64,
    beta_ptr: *mut f64,
    l_ptr: *mut f64,
    n_iterations_out: *mut i32,
    converged_out: *mut i32,
    tau_vec_ptr: *mut f64,
    n_treated_out: *mut i32,
    // Covariate support
    x_ptr: *const f64,
    n_covariates: i32,
    gamma_out: *mut f64,
) -> i32 {
    catch_panic!({
        if y_ptr.is_null() || d_ptr.is_null() || tau_out.is_null() || mu_out.is_null() {
            return TropError::NullPointer.code();
        }

        if n_periods <= 0 || n_units <= 0 {
            return TropError::InvalidDimension.code();
        }

        let np = n_periods as usize;
        let nu = n_units as usize;

        // Construct covariate matrix view (column-major)
        let x_view: Option<ArrayView2<f64>> = if n_covariates > 0 && !x_ptr.is_null() {
            let n_obs = np * nu;
            let n_cov = n_covariates as usize;
            let x_slice = std::slice::from_raw_parts(x_ptr, n_obs * n_cov);
            Some(ArrayView2::from_shape((n_obs, n_cov).f(), x_slice).unwrap())
        } else {
            None
        };

        let y = ptr_to_array2(y_ptr, np, nu);
        let d = ptr_to_array2(d_ptr, np, nu);

        // Map +Inf → 0 (no penalty) and +Inf for λ_nn → large finite cap.
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

        // Joint method is only well-defined under simultaneous adoption
        // (paper Remark 6.1).  Short-circuit with a clean error code if the
        // upstream Stata validation was bypassed.  On success, this yields
        // the shared `treated_periods` count used by δ.
        let treated_periods = match loocv::check_simultaneous_adoption(&d) {
            Ok(tp) => tp,
            Err(err) => return err.code(),
        };

        let delta = weights::compute_joint_weights(&y, &d, lt_eff, lu_eff, treated_periods);

        // B.2 defensive check: compute_joint_weights always (1 − D)-masks
        // δ, but wiring regressions at the FFI boundary are exactly what this
        // debug_assert is designed to catch.
        estimation::debug_assert_delta_is_1minus_d_masked(
            &d, &delta.view(), "stata_estimate_joint/delta",
        );

        // When λ_nn is large enough, skip the low-rank component entirely.
        // τ is post-hoc: mean residual over treated cells (L ≡ 0 here).
        let result = if ln_eff >= 1e10 {
            estimation::solve_joint_no_lowrank(&y, &delta.view(), x_view.as_ref()).map(
                |(mu, alpha, beta, gamma)| {
                    let mut tau_sum = 0.0_f64;
                    let mut tau_count = 0usize;
                    for t in 0..np {
                        for i in 0..nu {
                            if d[[t, i]] == 1.0 && y[[t, i]].is_finite() {
                                tau_sum += y[[t, i]] - mu - alpha[i] - beta[t];
                                tau_count += 1;
                            }
                        }
                    }
                    let tau = if tau_count > 0 { tau_sum / tau_count as f64 } else { 0.0 };
                    let l = Array2::<f64>::zeros((np, nu));
                    (mu, alpha, beta, l, tau, 1_usize, true, gamma)
                },
            )
        } else {
            estimation::solve_joint_with_lowrank(
                &y,
                &d,
                &delta.view(),
                ln_eff,
                max_iter as usize,
                tol,
                x_view.as_ref(),
            )
        };

        match result {
            Some((mu, alpha, beta, l, tau, n_iters, did_converge, gamma)) => {
                *tau_out = tau;
                *mu_out = mu;
                *n_iterations_out = n_iters as i32;
                *converged_out = if did_converge { 1 } else { 0 };

                if !alpha_ptr.is_null() {
                    let alpha_slice = slice::from_raw_parts_mut(alpha_ptr, nu);
                    alpha_slice.copy_from_slice(alpha.as_slice().unwrap());
                }

                if !beta_ptr.is_null() {
                    let beta_slice = slice::from_raw_parts_mut(beta_ptr, np);
                    beta_slice.copy_from_slice(beta.as_slice().unwrap());
                }

                if !l_ptr.is_null() {
                    array2_to_ptr(&l, l_ptr);
                }

                // Paper Eq 13: the joint method assembles τ_it for every treated
                // (i,t) cell as the post-hoc residual Y − μ − α_i − β_t − L_{ti}.
                // The scalar tau_out equals the mean of these values (Eq 1 ATT).
                // Exposing the vector lets users inspect cell-level heterogeneity.
                let mut n_treated_cells: i32 = 0;
                if !tau_vec_ptr.is_null() {
                    let mut tau_values: Vec<f64> = Vec::new();
                    for t in 0..np {
                        for i in 0..nu {
                            if d[[t, i]] == 1.0 && y[[t, i]].is_finite() {
                                tau_values.push(
                                    y[[t, i]] - mu - alpha[i] - beta[t] - l[[t, i]],
                                );
                            }
                        }
                    }
                    n_treated_cells = tau_values.len() as i32;
                    let tau_slice = slice::from_raw_parts_mut(tau_vec_ptr, tau_values.len());
                    tau_slice.copy_from_slice(&tau_values);
                } else {
                    for t in 0..np {
                        for i in 0..nu {
                            if d[[t, i]] == 1.0 && y[[t, i]].is_finite() {
                                n_treated_cells += 1;
                            }
                        }
                    }
                }
                if !n_treated_out.is_null() {
                    *n_treated_out = n_treated_cells;
                }

                // Write covariate coefficients
                if !gamma_out.is_null() && n_covariates > 0 {
                    if let Some(ref gamma_vec) = gamma {
                        let gamma_slice = std::slice::from_raw_parts_mut(gamma_out, n_covariates as usize);
                        for (idx, val) in gamma_vec.iter().enumerate() {
                            gamma_slice[idx] = *val;
                        }
                    } else {
                        let gamma_slice = std::slice::from_raw_parts_mut(gamma_out, n_covariates as usize);
                        for val in gamma_slice.iter_mut() {
                            *val = 0.0;
                        }
                    }
                }

                TropError::Success.code()
            }
            None => TropError::Convergence.code(),
        }
    })
}

/// Bootstrap variance estimation for the Joint method.
///
/// Resamples units with replacement `n_bootstrap` times, re-estimates τ in
/// each replicate, and returns the standard error, percentile confidence
/// interval, and the full vector of bootstrap estimates.
///
/// `ddof` selects the variance denominator:
///   - `1` (default) → Bessel-corrected sample variance `1/(B−1)`.
///   - `0`           → paper Algorithm 3 population variance `1/B`.
///   - Any other value collapses to `1`.
///
/// # Returns
/// `0` on success; a non-zero `TropError` code otherwise.
///
/// # Safety
/// All pointers must be non-null and point to properly sized buffers.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn stata_bootstrap_trop_variance_joint(
    y_ptr: *const f64,
    d_ptr: *const f64,
    n_periods: i32,
    n_units: i32,
    lambda_time: f64,
    lambda_unit: f64,
    lambda_nn: f64,
    n_bootstrap: i32,
    max_iter: i32,
    tol: f64,
    seed: u64,
    alpha: f64,
    ddof: i32,
    estimates_ptr: *mut f64,
    se_out: *mut f64,
    ci_lower_out: *mut f64,
    ci_upper_out: *mut f64,
    n_valid_out: *mut i32,
    // Covariate support
    x_ptr: *const f64,
    n_covariates: i32,
) -> i32 {
    catch_panic!({
        if y_ptr.is_null() || d_ptr.is_null() || se_out.is_null() {
            return TropError::NullPointer.code();
        }

        if n_periods <= 0 || n_units <= 0 {
            return TropError::InvalidDimension.code();
        }

        let np = n_periods as usize;
        let nu = n_units as usize;

        // Construct covariate matrix view (column-major)
        let x_view: Option<ArrayView2<f64>> = if n_covariates > 0 && !x_ptr.is_null() {
            let n_obs = np * nu;
            let n_cov = n_covariates as usize;
            let x_slice = std::slice::from_raw_parts(x_ptr, n_obs * n_cov);
            Some(ArrayView2::from_shape((n_obs, n_cov).f(), x_slice).unwrap())
        } else {
            None
        };

        let y = ptr_to_array2(y_ptr, np, nu);
        let d = ptr_to_array2(d_ptr, np, nu);

        // Joint bootstrap requires simultaneous adoption (each replicate calls
        // compute_joint_weights internally).  Short-circuit on violation.
        if let Err(err) = loocv::check_simultaneous_adoption(&d) {
            return err.code();
        }

        let alpha_eff = if alpha <= 0.0 || alpha >= 1.0 {
            0.05
        } else {
            alpha
        };

        // Clamp ddof into the {0, 1} set; any out-of-range value defaults to 1.
        let ddof_u8: u8 = if ddof == 0 { 0 } else { 1 };

        let result = bootstrap::bootstrap_trop_variance_joint_full(
            &y,
            &d,
            lambda_time,
            lambda_unit,
            lambda_nn,
            n_bootstrap as usize,
            max_iter as usize,
            tol,
            seed,
            alpha_eff,
            ddof_u8,
            x_view.as_ref(),
        );

        *se_out = result.se;

        if !ci_lower_out.is_null() {
            *ci_lower_out = result.ci_lower;
        }
        if !ci_upper_out.is_null() {
            *ci_upper_out = result.ci_upper;
        }
        if !n_valid_out.is_null() {
            *n_valid_out = result.n_valid as i32;
        }

        if !estimates_ptr.is_null() {
            let est_slice = slice::from_raw_parts_mut(estimates_ptr, result.estimates.len());
            est_slice.copy_from_slice(&result.estimates);
        }

        TropError::Success.code()
    })
}

// ---------------------------------------------------------------------------
// pweight-aware variants — C ABI exports
//
// These variants accept a non-null pointer `unit_weights_ptr` of length
// `n_units` with strictly positive pweights (validated by the caller) and
// aggregate the ATT as τ̂ = Σ w_i τ_{t,i} / Σ w_i.  Per-cell estimation and
// pre/post-hoc parameters remain unchanged — pweight enters only the
// aggregation step, matching the pweight-only (no strata/PSU/FPC) survey
// design used by the reference Python implementation.
// ---------------------------------------------------------------------------

/// Twostep point estimation with per-unit pweights.
///
/// Same as [`stata_estimate_twostep`] but aggregates the per-cell τ as the
/// weighted mean `τ̂ = Σ w_i τ_{t,i} / Σ w_i`.  The parameters (α, β, L)
/// are averaged unweighted across treated observations (unchanged).
///
/// # Arguments
/// All of [`stata_estimate_twostep`] plus `unit_weights_ptr` of length
/// `n_units`.  Weights are indexed by the original panel column order and
/// must be strictly positive (validated by the caller).
///
/// # Returns
/// `0` on success; a non-zero `TropError` code otherwise.
///
/// # Safety
/// All pointers must be non-null and point to properly sized buffers.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn stata_estimate_twostep_weighted(
    y_ptr: *const f64,
    d_ptr: *const f64,
    control_mask_ptr: *const u8,
    time_dist_ptr: *const i64,
    n_periods: i32,
    n_units: i32,
    lambda_time: f64,
    lambda_unit: f64,
    lambda_nn: f64,
    max_iter: i32,
    tol: f64,
    att_out: *mut f64,
    tau_ptr: *mut f64,
    alpha_ptr: *mut f64,
    beta_ptr: *mut f64,
    l_ptr: *mut f64,
    n_treated_out: *mut i32,
    n_iterations_out: *mut i32,
    converged_out: *mut i32,
    converged_by_obs_ptr: *mut i32,
    n_iters_by_obs_ptr: *mut i32,
    unit_weights_ptr: *const f64,
) -> i32 {
    catch_panic!({
        if y_ptr.is_null()
            || d_ptr.is_null()
            || control_mask_ptr.is_null()
            || time_dist_ptr.is_null()
            || att_out.is_null()
            || unit_weights_ptr.is_null()
        {
            return TropError::NullPointer.code();
        }

        let np = n_periods as usize;
        let nu = n_units as usize;

        let y = ptr_to_array2(y_ptr, np, nu);
        let d = ptr_to_array2(d_ptr, np, nu);
        let control_mask = ptr_to_array2_u8(control_mask_ptr, np, nu);
        let time_dist = ptr_to_array2_i64(time_dist_ptr, np, np);
        let unit_weights = slice::from_raw_parts(unit_weights_ptr, nu);

        let mut treated_obs: Vec<(usize, usize)> = Vec::new();
        for t in 0..np {
            for i in 0..nu {
                if d[[t, i]] == 1.0 {
                    treated_obs.push((t, i));
                }
            }
        }

        if treated_obs.is_empty() {
            return TropError::NoTreated.code();
        }

        let lt_eff = if lambda_time.is_infinite() { 0.0 } else { lambda_time };
        let lu_eff = if lambda_unit.is_infinite() { 0.0 } else { lambda_unit };
        let ln_eff = if lambda_nn.is_infinite() { 1e10 } else { lambda_nn };

        struct ObsResult {
            tau: f64,
            alpha: Array1<f64>,
            beta: Array1<f64>,
            l: Array2<f64>,
            n_iters: usize,
            converged: bool,
        }

        let dist_cache = distance::UnitDistanceCache::build(&y, &d);

        let obs_results: Vec<Option<ObsResult>> = treated_obs
            .par_iter()
            .map(|(t, i)| {
                let weight_matrix = weights::compute_weight_matrix_cached(
                    &y, &d, &dist_cache, np, nu, *i, *t, lt_eff, lu_eff, &time_dist,
                );

                match estimation::estimate_model(
                    &y,
                    &control_mask,
                    &weight_matrix.view(),
                    ln_eff,
                    np,
                    nu,
                    max_iter as usize,
                    tol,
                    None,
                    None,
                    None,
                    None,
                ) {
                    Some((alpha, beta, l, n_iters, did_converge, _gamma)) => {
                        let tau = y[[*t, *i]] - alpha[*i] - beta[*t] - l[[*t, *i]];
                        Some(ObsResult {
                            tau,
                            alpha,
                            beta,
                            l,
                            n_iters,
                            converged: did_converge,
                        })
                    }
                    None => None,
                }
            })
            .collect();

        // Collect (tau, unit_index) pairs for weighted aggregation; aggregate
        // (α, β, L) with an unweighted mean (unchanged vs. unweighted path).
        let mut tau_values: Vec<f64> = Vec::with_capacity(treated_obs.len());
        let mut tau_units: Vec<usize> = Vec::with_capacity(treated_obs.len());
        let mut alpha_sum = Array1::<f64>::zeros(nu);
        let mut beta_sum = Array1::<f64>::zeros(np);
        let mut l_sum = Array2::<f64>::zeros((np, nu));
        let mut n_successful: usize = 0;
        let mut max_iters: usize = 0;
        let mut all_successful_converged = true;
        let mut converged_by_obs: Vec<i32> = Vec::with_capacity(treated_obs.len());
        let mut n_iters_by_obs: Vec<i32> = Vec::with_capacity(treated_obs.len());

        for ((_t, i), result) in treated_obs.iter().zip(obs_results.into_iter()) {
            match result {
                Some(obs) => {
                    tau_values.push(obs.tau);
                    tau_units.push(*i);
                    alpha_sum += &obs.alpha;
                    beta_sum += &obs.beta;
                    l_sum += &obs.l;
                    n_successful += 1;
                    if obs.n_iters > max_iters {
                        max_iters = obs.n_iters;
                    }
                    if !obs.converged {
                        all_successful_converged = false;
                    }
                    converged_by_obs.push(if obs.converged { 1 } else { 0 });
                    n_iters_by_obs.push(obs.n_iters as i32);
                }
                None => {
                    converged_by_obs.push(-1);
                    n_iters_by_obs.push(-1);
                }
            }
        }

        if tau_values.is_empty() {
            return TropError::Convergence.code();
        }

        // Weighted ATT: τ̂ = Σ w_i τ_{t,i} / Σ w_i.
        let mut num = 0.0_f64;
        let mut den = 0.0_f64;
        for (tau_val, unit_idx) in tau_values.iter().zip(tau_units.iter()) {
            let wi = unit_weights[*unit_idx];
            if !wi.is_finite() || wi <= 0.0 {
                continue;
            }
            num += wi * tau_val;
            den += wi;
        }
        let att = if den > 0.0 { num / den } else { return TropError::Convergence.code(); };

        let n_succ_f64 = n_successful as f64;
        let all_alpha = alpha_sum / n_succ_f64;
        let all_beta = beta_sum / n_succ_f64;
        let all_l = l_sum / n_succ_f64;

        *att_out = att;
        *n_treated_out = tau_values.len() as i32;
        *n_iterations_out = max_iters as i32;
        *converged_out = if all_successful_converged { 1 } else { 0 };

        if !converged_by_obs_ptr.is_null() {
            let slot = slice::from_raw_parts_mut(converged_by_obs_ptr, converged_by_obs.len());
            slot.copy_from_slice(&converged_by_obs);
        }
        if !n_iters_by_obs_ptr.is_null() {
            let slot = slice::from_raw_parts_mut(n_iters_by_obs_ptr, n_iters_by_obs.len());
            slot.copy_from_slice(&n_iters_by_obs);
        }

        if !tau_ptr.is_null() {
            let tau_slice = slice::from_raw_parts_mut(tau_ptr, tau_values.len());
            tau_slice.copy_from_slice(&tau_values);
        }

        if !alpha_ptr.is_null() {
            let alpha_slice = slice::from_raw_parts_mut(alpha_ptr, nu);
            alpha_slice.copy_from_slice(all_alpha.as_slice().unwrap());
        }

        if !beta_ptr.is_null() {
            let beta_slice = slice::from_raw_parts_mut(beta_ptr, np);
            beta_slice.copy_from_slice(all_beta.as_slice().unwrap());
        }

        if !l_ptr.is_null() {
            array2_to_ptr(&all_l, l_ptr);
        }

        TropError::Success.code()
    })
}

/// Twostep bootstrap with per-unit pweights.
///
/// Same as [`stata_bootstrap_trop_variance`] but aggregates the bootstrap
/// ATT as `τ̂_b = Σ w_i τ_{t,i} / Σ w_i`, where each resampled column
/// inherits the pweight of the original unit it represents.
///
/// # Safety
/// All pointers must be non-null and point to properly sized buffers.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn stata_bootstrap_trop_variance_weighted(
    y_ptr: *const f64,
    d_ptr: *const f64,
    control_mask_ptr: *const u8,
    time_dist_ptr: *const i64,
    n_periods: i32,
    n_units: i32,
    lambda_time: f64,
    lambda_unit: f64,
    lambda_nn: f64,
    n_bootstrap: i32,
    max_iter: i32,
    tol: f64,
    seed: u64,
    alpha: f64,
    ddof: i32,
    estimates_ptr: *mut f64,
    se_out: *mut f64,
    ci_lower_out: *mut f64,
    ci_upper_out: *mut f64,
    n_valid_out: *mut i32,
    unit_weights_ptr: *const f64,
) -> i32 {
    catch_panic!({
        if y_ptr.is_null()
            || d_ptr.is_null()
            || control_mask_ptr.is_null()
            || time_dist_ptr.is_null()
            || se_out.is_null()
            || unit_weights_ptr.is_null()
        {
            return TropError::NullPointer.code();
        }

        let np = n_periods as usize;
        let nu = n_units as usize;

        let y = ptr_to_array2(y_ptr, np, nu);
        let d = ptr_to_array2(d_ptr, np, nu);
        let control_mask = ptr_to_array2_u8(control_mask_ptr, np, nu);
        let time_dist = ptr_to_array2_i64(time_dist_ptr, np, np);
        let unit_weights = slice::from_raw_parts(unit_weights_ptr, nu);

        let alpha_eff = if alpha <= 0.0 || alpha >= 1.0 { 0.05 } else { alpha };
        let ddof_u8: u8 = if ddof == 0 { 0 } else { 1 };

        let result = bootstrap::bootstrap_trop_variance_full_weighted(
            &y,
            &d,
            &control_mask,
            &time_dist,
            lambda_time,
            lambda_unit,
            lambda_nn,
            n_bootstrap as usize,
            max_iter as usize,
            tol,
            seed,
            alpha_eff,
            ddof_u8,
            unit_weights,
            None,
        );

        *se_out = result.se;

        if !ci_lower_out.is_null() {
            *ci_lower_out = result.ci_lower;
        }
        if !ci_upper_out.is_null() {
            *ci_upper_out = result.ci_upper;
        }
        if !n_valid_out.is_null() {
            *n_valid_out = result.n_valid as i32;
        }

        if !estimates_ptr.is_null() {
            let est_slice = slice::from_raw_parts_mut(estimates_ptr, result.estimates.len());
            est_slice.copy_from_slice(&result.estimates);
        }

        TropError::Success.code()
    })
}

/// Joint point estimation with per-unit pweights.
///
/// Same as [`stata_estimate_joint`] but aggregates the post-hoc τ as the
/// weighted mean `τ̂ = Σ w_i (Y_{t,i} − μ − α_i − β_t − L_{t,i}) / Σ w_i`.
/// The joint estimation of (μ, α, β, L) is unchanged.
///
/// # Safety
/// All pointers must be non-null and point to properly sized buffers.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn stata_estimate_joint_weighted(
    y_ptr: *const f64,
    d_ptr: *const f64,
    n_periods: i32,
    n_units: i32,
    lambda_time: f64,
    lambda_unit: f64,
    lambda_nn: f64,
    max_iter: i32,
    tol: f64,
    tau_out: *mut f64,
    mu_out: *mut f64,
    alpha_ptr: *mut f64,
    beta_ptr: *mut f64,
    l_ptr: *mut f64,
    n_iterations_out: *mut i32,
    converged_out: *mut i32,
    tau_vec_ptr: *mut f64,
    n_treated_out: *mut i32,
    unit_weights_ptr: *const f64,
) -> i32 {
    catch_panic!({
        if y_ptr.is_null()
            || d_ptr.is_null()
            || tau_out.is_null()
            || mu_out.is_null()
            || unit_weights_ptr.is_null()
        {
            return TropError::NullPointer.code();
        }

        let np = n_periods as usize;
        let nu = n_units as usize;

        let y = ptr_to_array2(y_ptr, np, nu);
        let d = ptr_to_array2(d_ptr, np, nu);
        let unit_weights = slice::from_raw_parts(unit_weights_ptr, nu);

        let lt_eff = if lambda_time.is_infinite() { 0.0 } else { lambda_time };
        let lu_eff = if lambda_unit.is_infinite() { 0.0 } else { lambda_unit };
        let ln_eff = if lambda_nn.is_infinite() { 1e10 } else { lambda_nn };

        // Joint method requires simultaneous adoption; see stata_estimate_joint.
        let treated_periods = match loocv::check_simultaneous_adoption(&d) {
            Ok(tp) => tp,
            Err(err) => return err.code(),
        };

        let delta = weights::compute_joint_weights(&y, &d, lt_eff, lu_eff, treated_periods);

        estimation::debug_assert_delta_is_1minus_d_masked(
            &d, &delta.view(), "stata_estimate_joint_weighted/delta",
        );

        // Joint WLS produces (μ, α, β, L).  We then compute the weighted ATT
        // post-hoc instead of the unweighted mean returned by the unweighted
        // path.
        let fit = if ln_eff >= 1e10 {
            estimation::solve_joint_no_lowrank(&y, &delta.view(), None).map(
                |(mu, alpha, beta, _gamma)| {
                    let l = Array2::<f64>::zeros((np, nu));
                    (mu, alpha, beta, l, 1_usize, true)
                },
            )
        } else {
            estimation::solve_joint_with_lowrank(
                &y,
                &d,
                &delta.view(),
                ln_eff,
                max_iter as usize,
                tol,
                None,
            )
            .map(|(mu, alpha, beta, l, _tau, n_iters, did_converge, _gamma)| {
                (mu, alpha, beta, l, n_iters, did_converge)
            })
        };

        match fit {
            Some((mu, alpha, beta, l, n_iters, did_converge)) => {
                // Weighted post-hoc ATT + per-cell τ vector.
                let mut tau_values: Vec<f64> = Vec::new();
                let mut num = 0.0_f64;
                let mut den = 0.0_f64;
                for t in 0..np {
                    for i in 0..nu {
                        if d[[t, i]] == 1.0 && y[[t, i]].is_finite() {
                            let tau_cell =
                                y[[t, i]] - mu - alpha[i] - beta[t] - l[[t, i]];
                            tau_values.push(tau_cell);
                            let wi = unit_weights[i];
                            if wi.is_finite() && wi > 0.0 {
                                num += wi * tau_cell;
                                den += wi;
                            }
                        }
                    }
                }
                let tau_scalar = if den > 0.0 { num / den } else { f64::NAN };

                *tau_out = tau_scalar;
                *mu_out = mu;
                *n_iterations_out = n_iters as i32;
                *converged_out = if did_converge { 1 } else { 0 };

                if !alpha_ptr.is_null() {
                    let alpha_slice = slice::from_raw_parts_mut(alpha_ptr, nu);
                    alpha_slice.copy_from_slice(alpha.as_slice().unwrap());
                }
                if !beta_ptr.is_null() {
                    let beta_slice = slice::from_raw_parts_mut(beta_ptr, np);
                    beta_slice.copy_from_slice(beta.as_slice().unwrap());
                }
                if !l_ptr.is_null() {
                    array2_to_ptr(&l, l_ptr);
                }

                if !tau_vec_ptr.is_null() {
                    let tau_slice = slice::from_raw_parts_mut(tau_vec_ptr, tau_values.len());
                    tau_slice.copy_from_slice(&tau_values);
                }
                if !n_treated_out.is_null() {
                    *n_treated_out = tau_values.len() as i32;
                }

                TropError::Success.code()
            }
            None => TropError::Convergence.code(),
        }
    })
}

/// Joint bootstrap with per-unit pweights.
///
/// Same as [`stata_bootstrap_trop_variance_joint`] but aggregates each
/// bootstrap replicate's ATT as the weighted post-hoc mean.
///
/// # Safety
/// All pointers must be non-null and point to properly sized buffers.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn stata_bootstrap_trop_variance_joint_weighted(
    y_ptr: *const f64,
    d_ptr: *const f64,
    n_periods: i32,
    n_units: i32,
    lambda_time: f64,
    lambda_unit: f64,
    lambda_nn: f64,
    n_bootstrap: i32,
    max_iter: i32,
    tol: f64,
    seed: u64,
    alpha: f64,
    ddof: i32,
    estimates_ptr: *mut f64,
    se_out: *mut f64,
    ci_lower_out: *mut f64,
    ci_upper_out: *mut f64,
    n_valid_out: *mut i32,
    unit_weights_ptr: *const f64,
) -> i32 {
    catch_panic!({
        if y_ptr.is_null()
            || d_ptr.is_null()
            || se_out.is_null()
            || unit_weights_ptr.is_null()
        {
            return TropError::NullPointer.code();
        }

        let np = n_periods as usize;
        let nu = n_units as usize;

        let y = ptr_to_array2(y_ptr, np, nu);
        let d = ptr_to_array2(d_ptr, np, nu);
        let unit_weights = slice::from_raw_parts(unit_weights_ptr, nu);

        // Joint bootstrap requires simultaneous adoption; see the unweighted
        // entry for the rationale.
        if let Err(err) = loocv::check_simultaneous_adoption(&d) {
            return err.code();
        }

        let alpha_eff = if alpha <= 0.0 || alpha >= 1.0 { 0.05 } else { alpha };
        let ddof_u8: u8 = if ddof == 0 { 0 } else { 1 };

        let result = bootstrap::bootstrap_trop_variance_joint_full_weighted(
            &y,
            &d,
            lambda_time,
            lambda_unit,
            lambda_nn,
            n_bootstrap as usize,
            max_iter as usize,
            tol,
            seed,
            alpha_eff,
            ddof_u8,
            unit_weights,
            None,
        );

        *se_out = result.se;

        if !ci_lower_out.is_null() {
            *ci_lower_out = result.ci_lower;
        }
        if !ci_upper_out.is_null() {
            *ci_upper_out = result.ci_upper;
        }
        if !n_valid_out.is_null() {
            *n_valid_out = result.n_valid as i32;
        }

        if !estimates_ptr.is_null() {
            let est_slice = slice::from_raw_parts_mut(estimates_ptr, result.estimates.len());
            est_slice.copy_from_slice(&result.estimates);
        }

        TropError::Success.code()
    })
}

// ---------------------------------------------------------------------------
// Rao-Wu survey bootstrap — C ABI exports
// ---------------------------------------------------------------------------

/// Rao-Wu bootstrap variance estimation for the Twostep method with complex
/// survey design (strata, PSU, FPC).
///
/// Fits the model once, then for each bootstrap replicate rescales unit weights
/// according to the Rao-Wu (1988) scheme and recomputes the weighted ATT.
///
/// # Safety
/// All pointers must be non-null and point to properly sized buffers.
/// `fpc_ptr` may be null if no finite population correction is applied.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn stata_bootstrap_trop_variance_rao_wu(
    y_ptr: *const f64,
    d_ptr: *const f64,
    control_mask_ptr: *const u8,
    time_dist_ptr: *const i64,
    n_periods: i32,
    n_units: i32,
    lambda_time: f64,
    lambda_unit: f64,
    lambda_nn: f64,
    n_bootstrap: i32,
    max_iter: i32,
    tol: f64,
    seed: u64,
    alpha: f64,
    ddof: i32,
    strata_ptr: *const i64,
    psu_ptr: *const i64,
    fpc_ptr: *const f64,
    unit_weights_ptr: *const f64,
    lonely_psu_code: i32,
    estimates_ptr: *mut f64,
    se_out: *mut f64,
    ci_lower_out: *mut f64,
    ci_upper_out: *mut f64,
    n_valid_out: *mut i32,
) -> i32 {
    catch_panic!({
        if y_ptr.is_null()
            || d_ptr.is_null()
            || control_mask_ptr.is_null()
            || time_dist_ptr.is_null()
            || strata_ptr.is_null()
            || psu_ptr.is_null()
            || unit_weights_ptr.is_null()
            || se_out.is_null()
        {
            return TropError::NullPointer.code();
        }

        let np = n_periods as usize;
        let nu = n_units as usize;

        let y = ptr_to_array2(y_ptr, np, nu);
        let d = ptr_to_array2(d_ptr, np, nu);
        let control_mask = ptr_to_array2_u8(control_mask_ptr, np, nu);
        let time_dist = ptr_to_array2_i64(time_dist_ptr, np, np);

        let strata = slice::from_raw_parts(strata_ptr, nu);
        let psu = slice::from_raw_parts(psu_ptr, nu);
        let fpc: Option<&[f64]> = if fpc_ptr.is_null() {
            None
        } else {
            Some(slice::from_raw_parts(fpc_ptr, nu))
        };
        let unit_weights = slice::from_raw_parts(unit_weights_ptr, nu);

        let alpha_eff = if alpha <= 0.0 || alpha >= 1.0 { 0.05 } else { alpha };
        let ddof_u8: u8 = if ddof == 0 { 0 } else { 1 };

        let result = match bootstrap::bootstrap_trop_variance_rao_wu(
            &y,
            &d,
            &control_mask,
            &time_dist,
            lambda_time,
            lambda_unit,
            lambda_nn,
            n_bootstrap as usize,
            max_iter as usize,
            tol,
            seed,
            alpha_eff,
            ddof_u8,
            strata,
            psu,
            fpc,
            unit_weights,
            None,
            bootstrap::LonelyPsuStrategy::from_code(lonely_psu_code),
        ) {
            Ok(r) => r,
            Err(e) => return e.code(),
        };

        *se_out = result.se;

        if !ci_lower_out.is_null() {
            *ci_lower_out = result.ci_lower;
        }
        if !ci_upper_out.is_null() {
            *ci_upper_out = result.ci_upper;
        }
        if !n_valid_out.is_null() {
            *n_valid_out = result.n_valid as i32;
        }

        if !estimates_ptr.is_null() {
            let est_slice = slice::from_raw_parts_mut(estimates_ptr, result.estimates.len());
            est_slice.copy_from_slice(&result.estimates);
        }

        TropError::Success.code()
    })
}

/// Rao-Wu bootstrap variance estimation for the Joint method with complex
/// survey design (strata, PSU, FPC).
///
/// Fits the joint model once, then for each bootstrap replicate rescales unit
/// weights according to the Rao-Wu (1988) scheme and recomputes the weighted ATT.
///
/// # Safety
/// All pointers must be non-null and point to properly sized buffers.
/// `fpc_ptr` may be null if no finite population correction is applied.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn stata_bootstrap_trop_variance_rao_wu_joint(
    y_ptr: *const f64,
    d_ptr: *const f64,
    n_periods: i32,
    n_units: i32,
    lambda_time: f64,
    lambda_unit: f64,
    lambda_nn: f64,
    n_bootstrap: i32,
    max_iter: i32,
    tol: f64,
    seed: u64,
    alpha: f64,
    ddof: i32,
    strata_ptr: *const i64,
    psu_ptr: *const i64,
    fpc_ptr: *const f64,
    unit_weights_ptr: *const f64,
    lonely_psu_code: i32,
    estimates_ptr: *mut f64,
    se_out: *mut f64,
    ci_lower_out: *mut f64,
    ci_upper_out: *mut f64,
    n_valid_out: *mut i32,
) -> i32 {
    catch_panic!({
        if y_ptr.is_null()
            || d_ptr.is_null()
            || strata_ptr.is_null()
            || psu_ptr.is_null()
            || unit_weights_ptr.is_null()
            || se_out.is_null()
        {
            return TropError::NullPointer.code();
        }

        let np = n_periods as usize;
        let nu = n_units as usize;

        let y = ptr_to_array2(y_ptr, np, nu);
        let d = ptr_to_array2(d_ptr, np, nu);

        // Joint bootstrap requires simultaneous adoption.
        if let Err(err) = loocv::check_simultaneous_adoption(&d) {
            return err.code();
        }

        let strata = slice::from_raw_parts(strata_ptr, nu);
        let psu = slice::from_raw_parts(psu_ptr, nu);
        let fpc: Option<&[f64]> = if fpc_ptr.is_null() {
            None
        } else {
            Some(slice::from_raw_parts(fpc_ptr, nu))
        };
        let unit_weights = slice::from_raw_parts(unit_weights_ptr, nu);

        let alpha_eff = if alpha <= 0.0 || alpha >= 1.0 { 0.05 } else { alpha };
        let ddof_u8: u8 = if ddof == 0 { 0 } else { 1 };

        let result = match bootstrap::bootstrap_trop_variance_rao_wu_joint(
            &y,
            &d,
            lambda_time,
            lambda_unit,
            lambda_nn,
            n_bootstrap as usize,
            max_iter as usize,
            tol,
            seed,
            alpha_eff,
            ddof_u8,
            strata,
            psu,
            fpc,
            unit_weights,
            None,
            bootstrap::LonelyPsuStrategy::from_code(lonely_psu_code),
        ) {
            Ok(r) => r,
            Err(e) => return e.code(),
        };

        *se_out = result.se;

        if !ci_lower_out.is_null() {
            *ci_lower_out = result.ci_lower;
        }
        if !ci_upper_out.is_null() {
            *ci_upper_out = result.ci_upper;
        }
        if !n_valid_out.is_null() {
            *n_valid_out = result.n_valid as i32;
        }

        if !estimates_ptr.is_null() {
            let est_slice = slice::from_raw_parts_mut(estimates_ptr, result.estimates.len());
            est_slice.copy_from_slice(&result.estimates);
        }

        TropError::Success.code()
    })
}

// ---------------------------------------------------------------------------
// Utility — C ABI exports
// ---------------------------------------------------------------------------

/// Computes the N×N unit distance matrix based on pre-treatment outcomes.
///
/// Entry (i, j) measures the Euclidean distance between units i and j over
/// their shared control periods.  The result is written in column-major order.
///
/// # Returns
/// `0` on success; a non-zero `TropError` code otherwise.
///
/// # Safety
/// All pointers must be non-null and point to properly sized buffers.
#[no_mangle]
pub unsafe extern "C" fn stata_compute_unit_distance_matrix(
    y_ptr: *const f64,
    d_ptr: *const f64,
    n_periods: i32,
    n_units: i32,
    dist_ptr: *mut f64,
) -> i32 {
    catch_panic!({
        if y_ptr.is_null() || d_ptr.is_null() || dist_ptr.is_null() {
            return TropError::NullPointer.code();
        }

        let np = n_periods as usize;
        let nu = n_units as usize;

        let y = ptr_to_array2(y_ptr, np, nu);
        let d = ptr_to_array2(d_ptr, np, nu);

        let dist_matrix = distance::compute_unit_distance_matrix_internal(&y, &d);

        let dist_slice = slice::from_raw_parts_mut(dist_ptr, nu * nu);
        for i in 0..nu {
            for j in 0..nu {
                let idx = j * nu + i;
                dist_slice[idx] = dist_matrix[[i, j]];
            }
        }

        TropError::Success.code()
    })
}

/// Returns Twostep weight component vectors for a single treated observation.
///
/// Writes the time weight vector θ (T×1) and the unit weight vector ω (N×1)
/// that would be applied when estimating the model centred on
/// (`target_period`, `target_unit`).
///
/// # Returns
/// `0` on success; a non-zero `TropError` code otherwise.
///
/// # Safety
/// All pointers must be non-null and point to properly sized buffers.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn stata_compute_twostep_weight_vectors(
    y_ptr: *const f64,
    d_ptr: *const f64,
    time_dist_ptr: *const i64,
    n_periods: i32,
    n_units: i32,
    target_unit: i32,
    target_period: i32,
    lambda_time: f64,
    lambda_unit: f64,
    theta_out: *mut f64,
    omega_out: *mut f64,
) -> i32 {
    catch_panic!({
        if y_ptr.is_null()
            || d_ptr.is_null()
            || time_dist_ptr.is_null()
            || theta_out.is_null()
            || omega_out.is_null()
        {
            return TropError::NullPointer.code();
        }

        let np = n_periods as usize;
        let nu = n_units as usize;
        let tu = target_unit as usize;
        let tp = target_period as usize;

        let y = ptr_to_array2(y_ptr, np, nu);
        let d = ptr_to_array2(d_ptr, np, nu);
        let time_dist = ptr_to_array2_i64(time_dist_ptr, np, np);

        // Map +Inf → 0 (no penalty), consistent with estimation functions.
        let lt_eff = if lambda_time.is_infinite() { 0.0 } else { lambda_time };
        let lu_eff = if lambda_unit.is_infinite() { 0.0 } else { lambda_unit };

        let (time_weights, unit_weights) = weights::compute_twostep_weight_vectors(
            &y, &d, np, nu, tu, tp, lt_eff, lu_eff, &time_dist,
        );

        // Write output vectors.
        let theta_slice = slice::from_raw_parts_mut(theta_out, np);
        theta_slice.copy_from_slice(time_weights.as_slice().unwrap());

        let omega_slice = slice::from_raw_parts_mut(omega_out, nu);
        omega_slice.copy_from_slice(unit_weights.as_slice().unwrap());

        TropError::Success.code()
    })
}

/// Returns Joint weight component vectors.
///
/// Writes the global time weight vector δ_time (T×1) and the unit weight
/// vector δ_unit (N×1) used in the Joint estimation objective.
///
/// # Returns
/// `0` on success; a non-zero `TropError` code otherwise.
///
/// # Safety
/// All pointers must be non-null and point to properly sized buffers.
#[no_mangle]
pub unsafe extern "C" fn stata_compute_joint_weight_vectors(
    y_ptr: *const f64,
    d_ptr: *const f64,
    n_periods: i32,
    n_units: i32,
    lambda_time: f64,
    lambda_unit: f64,
    delta_time_out: *mut f64,
    delta_unit_out: *mut f64,
) -> i32 {
    catch_panic!({
        if y_ptr.is_null()
            || d_ptr.is_null()
            || delta_time_out.is_null()
            || delta_unit_out.is_null()
        {
            return TropError::NullPointer.code();
        }

        let np = n_periods as usize;
        let nu = n_units as usize;

        let y = ptr_to_array2(y_ptr, np, nu);
        let d = ptr_to_array2(d_ptr, np, nu);

        // Map +Inf → 0 (no penalty).
        let lt_eff = if lambda_time.is_infinite() { 0.0 } else { lambda_time };
        let lu_eff = if lambda_unit.is_infinite() { 0.0 } else { lambda_unit };

        // Joint weight construction requires simultaneous adoption; see
        // stata_estimate_joint for the rationale.
        let treated_periods = match loocv::check_simultaneous_adoption(&d) {
            Ok(tp) => tp,
            Err(err) => return err.code(),
        };

        let (delta_time, delta_unit) = weights::compute_joint_weight_vectors(
            &y, &d, lt_eff, lu_eff, treated_periods,
        );

        // Write output vectors.
        let dt_slice = slice::from_raw_parts_mut(delta_time_out, np);
        dt_slice.copy_from_slice(delta_time.as_slice().unwrap());

        let du_slice = slice::from_raw_parts_mut(delta_unit_out, nu);
        du_slice.copy_from_slice(delta_unit.as_slice().unwrap());

        TropError::Success.code()
    })
}

// ---------------------------------------------------------------------------
// Covariate-aware variants — C ABI exports
//
// These variants accept an additional X matrix (T*N × p, column-major) and
// pass it through to the internal estimation/loocv/bootstrap functions.
// The non-covariate entry points remain unchanged (backward-compatible).
// ---------------------------------------------------------------------------

/// Helper: convert a column-major X pointer into a row-major Array2<f64>.
///
/// Stata sends X as column-major (n_obs × p) where n_obs = n_periods * n_units.
/// Internally Rust expects row-major with row index = t * n_units + i.
#[inline]
unsafe fn ptr_to_x_matrix(
    x_ptr: *const f64,
    n_periods: usize,
    n_units: usize,
    n_covariates: usize,
) -> Option<Array2<f64>> {
    if n_covariates == 0 || x_ptr.is_null() {
        return None;
    }
    let n_obs = n_periods * n_units;
    let p = n_covariates;
    let x_slice = slice::from_raw_parts(x_ptr, n_obs * p);
    let mut x_arr = Array2::<f64>::zeros((n_obs, p));
    for col in 0..p {
        for row in 0..n_obs {
            x_arr[[row, col]] = x_slice[row + col * n_obs];
        }
    }
    Some(x_arr)
}

/// LOOCV grid search for Twostep with covariates.
///
/// # Safety
///
/// All pointer arguments must be non-null and point to properly sized,
/// aligned buffers. The caller must guarantee that buffer lengths match
/// the declared dimensions.
///
/// # Safety
///
/// All pointer arguments must be non-null and point to properly sized,
/// aligned buffers. The caller must guarantee that buffer lengths match
/// the declared dimensions.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn stata_loocv_grid_search_with_covariates(
    y_ptr: *const f64,
    d_ptr: *const f64,
    control_mask_ptr: *const u8,
    time_dist_ptr: *const i64,
    n_periods: i32,
    n_units: i32,
    lambda_time_grid_ptr: *const f64,
    lambda_time_grid_len: i32,
    lambda_unit_grid_ptr: *const f64,
    lambda_unit_grid_len: i32,
    lambda_nn_grid_ptr: *const f64,
    lambda_nn_grid_len: i32,
    max_iter: i32,
    tol: f64,
    best_lambda_time_out: *mut f64,
    best_lambda_unit_out: *mut f64,
    best_lambda_nn_out: *mut f64,
    best_score_out: *mut f64,
    n_valid_out: *mut i32,
    n_attempted_out: *mut i32,
    first_failed_t_out: *mut i32,
    first_failed_i_out: *mut i32,
    stage1_lambda_time_out: *mut f64,
    stage1_lambda_unit_out: *mut f64,
    stage1_lambda_nn_out: *mut f64,
    x_ptr: *const f64,
    n_covariates: i32,
) -> i32 {
    catch_panic!({
        if y_ptr.is_null()
            || d_ptr.is_null()
            || control_mask_ptr.is_null()
            || time_dist_ptr.is_null()
            || lambda_time_grid_ptr.is_null()
            || lambda_unit_grid_ptr.is_null()
            || lambda_nn_grid_ptr.is_null()
            || best_lambda_time_out.is_null()
            || best_lambda_unit_out.is_null()
            || best_lambda_nn_out.is_null()
            || best_score_out.is_null()
        {
            return TropError::NullPointer.code();
        }

        if n_covariates > 0 && x_ptr.is_null() {
            return TropError::NullPointer.code();
        }

        let np = n_periods as usize;
        let nu = n_units as usize;

        let y = ptr_to_array2(y_ptr, np, nu);
        let d = ptr_to_array2(d_ptr, np, nu);
        let control_mask = ptr_to_array2_u8(control_mask_ptr, np, nu);
        let time_dist = ptr_to_array2_i64(time_dist_ptr, np, np);

        let lambda_time_grid =
            slice::from_raw_parts(lambda_time_grid_ptr, lambda_time_grid_len as usize);
        let lambda_unit_grid =
            slice::from_raw_parts(lambda_unit_grid_ptr, lambda_unit_grid_len as usize);
        let lambda_nn_grid = slice::from_raw_parts(lambda_nn_grid_ptr, lambda_nn_grid_len as usize);

        let x_opt = ptr_to_x_matrix(x_ptr, np, nu, n_covariates as usize);
        let x_view = x_opt.as_ref().map(|a| a.view());

        let (
            (
                best_time,
                best_unit,
                best_nn,
                best_score,
                n_valid,
                n_attempted,
                first_failed,
            ),
            stage1_time,
            stage1_unit,
            stage1_nn,
        ) = match loocv::loocv_grid_search_with_stage1(
            &y,
            &d,
            &control_mask,
            &time_dist,
            lambda_time_grid,
            lambda_unit_grid,
            lambda_nn_grid,
            max_iter as usize,
            tol,
            x_view.as_ref(),
        ) {
            Ok(v) => v,
            Err(e) => return e.code(),
        };

        *best_lambda_time_out = best_time;
        *best_lambda_unit_out = best_unit;
        *best_lambda_nn_out = best_nn;
        *best_score_out = best_score;

        if !n_valid_out.is_null() {
            *n_valid_out = n_valid as i32;
        }
        if !n_attempted_out.is_null() {
            *n_attempted_out = n_attempted as i32;
        }
        if !first_failed_t_out.is_null() {
            *first_failed_t_out = match first_failed {
                Some((t, _)) => t as i32,
                None => -1,
            };
        }
        if !first_failed_i_out.is_null() {
            *first_failed_i_out = match first_failed {
                Some((_, i)) => i as i32,
                None => -1,
            };
        }
        if !stage1_lambda_time_out.is_null() {
            *stage1_lambda_time_out = stage1_time;
        }
        if !stage1_lambda_unit_out.is_null() {
            *stage1_lambda_unit_out = stage1_unit;
        }
        if !stage1_lambda_nn_out.is_null() {
            *stage1_lambda_nn_out = stage1_nn;
        }

        TropError::Success.code()
    })
}

/// Exhaustive LOOCV grid search for Twostep with covariates.
///
/// # Safety
///
/// All pointer arguments must be non-null and point to properly sized,
/// aligned buffers. The caller must guarantee that buffer lengths match
/// the declared dimensions.
///
/// # Safety
///
/// All pointer arguments must be non-null and point to properly sized,
/// aligned buffers. The caller must guarantee that buffer lengths match
/// the declared dimensions.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn stata_loocv_grid_search_exhaustive_with_covariates(
    y_ptr: *const f64,
    d_ptr: *const f64,
    control_mask_ptr: *const u8,
    time_dist_ptr: *const i64,
    n_periods: i32,
    n_units: i32,
    lambda_time_grid_ptr: *const f64,
    lambda_time_grid_len: i32,
    lambda_unit_grid_ptr: *const f64,
    lambda_unit_grid_len: i32,
    lambda_nn_grid_ptr: *const f64,
    lambda_nn_grid_len: i32,
    max_iter: i32,
    tol: f64,
    best_lambda_time_out: *mut f64,
    best_lambda_unit_out: *mut f64,
    best_lambda_nn_out: *mut f64,
    best_score_out: *mut f64,
    n_valid_out: *mut i32,
    n_attempted_out: *mut i32,
    first_failed_t_out: *mut i32,
    first_failed_i_out: *mut i32,
    x_ptr: *const f64,
    n_covariates: i32,
) -> i32 {
    catch_panic!({
        if y_ptr.is_null()
            || d_ptr.is_null()
            || control_mask_ptr.is_null()
            || time_dist_ptr.is_null()
            || lambda_time_grid_ptr.is_null()
            || lambda_unit_grid_ptr.is_null()
            || lambda_nn_grid_ptr.is_null()
            || best_lambda_time_out.is_null()
            || best_lambda_unit_out.is_null()
            || best_lambda_nn_out.is_null()
            || best_score_out.is_null()
        {
            return TropError::NullPointer.code();
        }

        if n_covariates > 0 && x_ptr.is_null() {
            return TropError::NullPointer.code();
        }

        let np = n_periods as usize;
        let nu = n_units as usize;

        let y = ptr_to_array2(y_ptr, np, nu);
        let d = ptr_to_array2(d_ptr, np, nu);
        let control_mask = ptr_to_array2_u8(control_mask_ptr, np, nu);
        let time_dist = ptr_to_array2_i64(time_dist_ptr, np, np);

        let lambda_time_grid =
            slice::from_raw_parts(lambda_time_grid_ptr, lambda_time_grid_len as usize);
        let lambda_unit_grid =
            slice::from_raw_parts(lambda_unit_grid_ptr, lambda_unit_grid_len as usize);
        let lambda_nn_grid = slice::from_raw_parts(lambda_nn_grid_ptr, lambda_nn_grid_len as usize);

        let x_opt = ptr_to_x_matrix(x_ptr, np, nu, n_covariates as usize);
        let x_view = x_opt.as_ref().map(|a| a.view());

        let (
            best_time,
            best_unit,
            best_nn,
            best_score,
            n_valid,
            n_attempted,
            first_failed,
        ) = match loocv::loocv_grid_search_exhaustive(
            &y,
            &d,
            &control_mask,
            &time_dist,
            lambda_time_grid,
            lambda_unit_grid,
            lambda_nn_grid,
            max_iter as usize,
            tol,
            x_view.as_ref(),
        ) {
            Ok(v) => v,
            Err(e) => return e.code(),
        };

        *best_lambda_time_out = best_time;
        *best_lambda_unit_out = best_unit;
        *best_lambda_nn_out = best_nn;
        *best_score_out = best_score;

        if !n_valid_out.is_null() {
            *n_valid_out = n_valid as i32;
        }
        if !n_attempted_out.is_null() {
            *n_attempted_out = n_attempted as i32;
        }
        if !first_failed_t_out.is_null() {
            *first_failed_t_out = match first_failed {
                Some((t, _)) => t as i32,
                None => -1,
            };
        }
        if !first_failed_i_out.is_null() {
            *first_failed_i_out = match first_failed {
                Some((_, i)) => i as i32,
                None => -1,
            };
        }

        TropError::Success.code()
    })
}

/// Twostep point estimation with covariates.
///
/// # Safety
///
/// All pointer arguments must be non-null and point to properly sized,
/// aligned buffers. The caller must guarantee that buffer lengths match
/// the declared dimensions.
///
/// # Safety
///
/// All pointer arguments must be non-null and point to properly sized,
/// aligned buffers. The caller must guarantee that buffer lengths match
/// the declared dimensions.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn stata_estimate_twostep_with_covariates(
    y_ptr: *const f64,
    d_ptr: *const f64,
    control_mask_ptr: *const u8,
    time_dist_ptr: *const i64,
    n_periods: i32,
    n_units: i32,
    lambda_time: f64,
    lambda_unit: f64,
    lambda_nn: f64,
    max_iter: i32,
    tol: f64,
    att_out: *mut f64,
    tau_ptr: *mut f64,
    alpha_ptr: *mut f64,
    beta_ptr: *mut f64,
    l_ptr: *mut f64,
    n_treated_out: *mut i32,
    n_iterations_out: *mut i32,
    converged_out: *mut i32,
    converged_by_obs_ptr: *mut i32,
    n_iters_by_obs_ptr: *mut i32,
    x_ptr: *const f64,
    n_covariates: i32,
    gamma_out: *mut f64,
) -> i32 {
    catch_panic!({
        if y_ptr.is_null()
            || d_ptr.is_null()
            || control_mask_ptr.is_null()
            || time_dist_ptr.is_null()
            || att_out.is_null()
        {
            return TropError::NullPointer.code();
        }

        if n_covariates > 0 && x_ptr.is_null() {
            return TropError::NullPointer.code();
        }

        let np = n_periods as usize;
        let nu = n_units as usize;
        let p = n_covariates as usize;

        let y = ptr_to_array2(y_ptr, np, nu);
        let d = ptr_to_array2(d_ptr, np, nu);
        let control_mask = ptr_to_array2_u8(control_mask_ptr, np, nu);
        let time_dist = ptr_to_array2_i64(time_dist_ptr, np, np);

        let x_opt = ptr_to_x_matrix(x_ptr, np, nu, p);
        let x_view = x_opt.as_ref().map(|a| a.view());

        let mut treated_obs: Vec<(usize, usize)> = Vec::new();
        for t in 0..np {
            for i in 0..nu {
                if d[[t, i]] == 1.0 {
                    treated_obs.push((t, i));
                }
            }
        }

        if treated_obs.is_empty() {
            return TropError::NoTreated.code();
        }

        let lt_eff = if lambda_time.is_infinite() { 0.0 } else { lambda_time };
        let lu_eff = if lambda_unit.is_infinite() { 0.0 } else { lambda_unit };
        let ln_eff = if lambda_nn.is_infinite() { 1e10 } else { lambda_nn };

        struct ObsResult {
            tau: f64,
            alpha: Array1<f64>,
            beta: Array1<f64>,
            l: Array2<f64>,
            n_iters: usize,
            converged: bool,
            gamma: Option<Array1<f64>>,
        }

        let dist_cache = distance::UnitDistanceCache::build(&y, &d);

        let obs_results: Vec<Option<ObsResult>> = treated_obs
            .par_iter()
            .map(|(t, i)| {
                let weight_matrix = weights::compute_weight_matrix_cached(
                    &y, &d, &dist_cache, np, nu, *i, *t, lt_eff, lu_eff, &time_dist,
                );

                match estimation::estimate_model(
                    &y,
                    &control_mask,
                    &weight_matrix.view(),
                    ln_eff,
                    np,
                    nu,
                    max_iter as usize,
                    tol,
                    None,
                    None,
                    x_view.as_ref(),
                    None,
                ) {
                    Some((alpha, beta, l, n_iters, did_converge, gamma)) => {
                        let x_gamma = match (&gamma, &x_view) {
                            (Some(g), Some(xm)) => {
                                let obs_idx = *t * nu + *i;
                                xm.row(obs_idx).dot(g)
                            }
                            _ => 0.0,
                        };
                        let tau = y[[*t, *i]] - alpha[*i] - beta[*t] - l[[*t, *i]] - x_gamma;
                        Some(ObsResult {
                            tau,
                            alpha,
                            beta,
                            l,
                            n_iters,
                            converged: did_converge,
                            gamma,
                        })
                    }
                    None => None,
                }
            })
            .collect();

        let mut tau_values: Vec<f64> = Vec::with_capacity(treated_obs.len());
        let mut alpha_sum = Array1::<f64>::zeros(nu);
        let mut beta_sum = Array1::<f64>::zeros(np);
        let mut l_sum = Array2::<f64>::zeros((np, nu));
        let mut gamma_sum: Option<Array1<f64>> = if p > 0 { Some(Array1::zeros(p)) } else { None };
        let mut n_successful: usize = 0;
        let mut max_iters: usize = 0;
        let mut all_successful_converged = true;
        let mut converged_by_obs: Vec<i32> = Vec::with_capacity(treated_obs.len());
        let mut n_iters_by_obs: Vec<i32> = Vec::with_capacity(treated_obs.len());

        for result in obs_results {
            match result {
                Some(obs) => {
                    tau_values.push(obs.tau);
                    alpha_sum += &obs.alpha;
                    beta_sum += &obs.beta;
                    l_sum += &obs.l;
                    if let (Some(ref mut gs), Some(ref g)) = (&mut gamma_sum, &obs.gamma) {
                        *gs += g;
                    }
                    n_successful += 1;
                    if obs.n_iters > max_iters {
                        max_iters = obs.n_iters;
                    }
                    if !obs.converged {
                        all_successful_converged = false;
                    }
                    converged_by_obs.push(if obs.converged { 1 } else { 0 });
                    n_iters_by_obs.push(obs.n_iters as i32);
                }
                None => {
                    converged_by_obs.push(-1);
                    n_iters_by_obs.push(-1);
                }
            }
        }

        if tau_values.is_empty() {
            return TropError::Convergence.code();
        }

        let att = tau_values.iter().sum::<f64>() / tau_values.len() as f64;

        let n_succ_f64 = n_successful as f64;
        let all_alpha = alpha_sum / n_succ_f64;
        let all_beta = beta_sum / n_succ_f64;
        let all_l = l_sum / n_succ_f64;

        *att_out = att;
        *n_treated_out = tau_values.len() as i32;
        *n_iterations_out = max_iters as i32;
        *converged_out = if all_successful_converged { 1 } else { 0 };

        if !converged_by_obs_ptr.is_null() {
            let slot = slice::from_raw_parts_mut(converged_by_obs_ptr, converged_by_obs.len());
            slot.copy_from_slice(&converged_by_obs);
        }
        if !n_iters_by_obs_ptr.is_null() {
            let slot = slice::from_raw_parts_mut(n_iters_by_obs_ptr, n_iters_by_obs.len());
            slot.copy_from_slice(&n_iters_by_obs);
        }

        if !tau_ptr.is_null() {
            let tau_slice = slice::from_raw_parts_mut(tau_ptr, tau_values.len());
            tau_slice.copy_from_slice(&tau_values);
        }

        if !alpha_ptr.is_null() {
            let alpha_slice = slice::from_raw_parts_mut(alpha_ptr, nu);
            alpha_slice.copy_from_slice(all_alpha.as_slice().unwrap());
        }

        if !beta_ptr.is_null() {
            let beta_slice = slice::from_raw_parts_mut(beta_ptr, np);
            beta_slice.copy_from_slice(all_beta.as_slice().unwrap());
        }

        if !l_ptr.is_null() {
            array2_to_ptr(&all_l, l_ptr);
        }

        // Write averaged gamma to output buffer.
        if !gamma_out.is_null() && p > 0 {
            if let Some(ref gs) = gamma_sum {
                let avg_gamma = gs / n_succ_f64;
                let gamma_slice = slice::from_raw_parts_mut(gamma_out, p);
                for j in 0..p {
                    gamma_slice[j] = avg_gamma[j];
                }
            }
        }

        TropError::Success.code()
    })
}

/// Bootstrap variance estimation for Twostep with covariates.
///
/// # Safety
///
/// All pointer arguments must be non-null and point to properly sized,
/// aligned buffers. The caller must guarantee that buffer lengths match
/// the declared dimensions.
///
/// # Safety
///
/// All pointer arguments must be non-null and point to properly sized,
/// aligned buffers. The caller must guarantee that buffer lengths match
/// the declared dimensions.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn stata_bootstrap_trop_variance_with_covariates(
    y_ptr: *const f64,
    d_ptr: *const f64,
    control_mask_ptr: *const u8,
    time_dist_ptr: *const i64,
    n_periods: i32,
    n_units: i32,
    lambda_time: f64,
    lambda_unit: f64,
    lambda_nn: f64,
    n_bootstrap: i32,
    max_iter: i32,
    tol: f64,
    seed: u64,
    alpha: f64,
    ddof: i32,
    estimates_ptr: *mut f64,
    se_out: *mut f64,
    ci_lower_out: *mut f64,
    ci_upper_out: *mut f64,
    n_valid_out: *mut i32,
    x_ptr: *const f64,
    n_covariates: i32,
) -> i32 {
    catch_panic!({
        if y_ptr.is_null()
            || d_ptr.is_null()
            || control_mask_ptr.is_null()
            || time_dist_ptr.is_null()
            || se_out.is_null()
        {
            return TropError::NullPointer.code();
        }

        if n_covariates > 0 && x_ptr.is_null() {
            return TropError::NullPointer.code();
        }

        let np = n_periods as usize;
        let nu = n_units as usize;

        let y = ptr_to_array2(y_ptr, np, nu);
        let d = ptr_to_array2(d_ptr, np, nu);
        let control_mask = ptr_to_array2_u8(control_mask_ptr, np, nu);
        let time_dist = ptr_to_array2_i64(time_dist_ptr, np, np);

        let x_opt = ptr_to_x_matrix(x_ptr, np, nu, n_covariates as usize);
        let x_view = x_opt.as_ref().map(|a| a.view());

        let alpha_eff = if alpha <= 0.0 || alpha >= 1.0 { 0.05 } else { alpha };
        let ddof_u8: u8 = if ddof == 0 { 0 } else { 1 };

        let result = bootstrap::bootstrap_trop_variance_full(
            &y,
            &d,
            &control_mask,
            &time_dist,
            lambda_time,
            lambda_unit,
            lambda_nn,
            n_bootstrap as usize,
            max_iter as usize,
            tol,
            seed,
            alpha_eff,
            ddof_u8,
            x_view.as_ref(),
        );

        *se_out = result.se;

        if !ci_lower_out.is_null() {
            *ci_lower_out = result.ci_lower;
        }
        if !ci_upper_out.is_null() {
            *ci_upper_out = result.ci_upper;
        }
        if !n_valid_out.is_null() {
            *n_valid_out = result.n_valid as i32;
        }

        if !estimates_ptr.is_null() {
            let est_slice = slice::from_raw_parts_mut(estimates_ptr, result.estimates.len());
            est_slice.copy_from_slice(&result.estimates);
        }

        TropError::Success.code()
    })
}

/// Joint point estimation with covariates.
///
/// # Safety
///
/// All pointer arguments must be non-null and point to properly sized,
/// aligned buffers. The caller must guarantee that buffer lengths match
/// the declared dimensions.
///
/// # Safety
///
/// All pointer arguments must be non-null and point to properly sized,
/// aligned buffers. The caller must guarantee that buffer lengths match
/// the declared dimensions.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn stata_estimate_joint_with_covariates(
    y_ptr: *const f64,
    d_ptr: *const f64,
    n_periods: i32,
    n_units: i32,
    lambda_time: f64,
    lambda_unit: f64,
    lambda_nn: f64,
    max_iter: i32,
    tol: f64,
    tau_out: *mut f64,
    mu_out: *mut f64,
    alpha_ptr: *mut f64,
    beta_ptr: *mut f64,
    l_ptr: *mut f64,
    n_iterations_out: *mut i32,
    converged_out: *mut i32,
    tau_vec_ptr: *mut f64,
    n_treated_out: *mut i32,
    x_ptr: *const f64,
    n_covariates: i32,
    gamma_out: *mut f64,
) -> i32 {
    catch_panic!({
        if y_ptr.is_null() || d_ptr.is_null() || tau_out.is_null() || mu_out.is_null() {
            return TropError::NullPointer.code();
        }

        if n_covariates > 0 && x_ptr.is_null() {
            return TropError::NullPointer.code();
        }

        let np = n_periods as usize;
        let nu = n_units as usize;
        let p = n_covariates as usize;

        let y = ptr_to_array2(y_ptr, np, nu);
        let d = ptr_to_array2(d_ptr, np, nu);

        let x_opt = ptr_to_x_matrix(x_ptr, np, nu, p);
        let x_view = x_opt.as_ref().map(|a| a.view());

        let lt_eff = if lambda_time.is_infinite() { 0.0 } else { lambda_time };
        let lu_eff = if lambda_unit.is_infinite() { 0.0 } else { lambda_unit };
        let ln_eff = if lambda_nn.is_infinite() { 1e10 } else { lambda_nn };

        let treated_periods = match loocv::check_simultaneous_adoption(&d) {
            Ok(tp) => tp,
            Err(err) => return err.code(),
        };

        let delta = weights::compute_joint_weights(&y, &d, lt_eff, lu_eff, treated_periods);

        estimation::debug_assert_delta_is_1minus_d_masked(
            &d, &delta.view(), "stata_estimate_joint_with_covariates/delta",
        );

        let result = if ln_eff >= 1e10 {
            estimation::solve_joint_no_lowrank(&y, &delta.view(), x_view.as_ref()).map(
                |(mu, alpha, beta, gamma)| {
                    let mut tau_sum = 0.0_f64;
                    let mut tau_count = 0usize;
                    for t in 0..np {
                        for i in 0..nu {
                            if d[[t, i]] == 1.0 && y[[t, i]].is_finite() {
                                let x_gamma_val = match (&gamma, &x_view) {
                                    (Some(g), Some(xm)) => {
                                        let obs_idx = t * nu + i;
                                        xm.row(obs_idx).dot(g)
                                    }
                                    _ => 0.0,
                                };
                                tau_sum += y[[t, i]] - mu - alpha[i] - beta[t] - x_gamma_val;
                                tau_count += 1;
                            }
                        }
                    }
                    let tau = if tau_count > 0 { tau_sum / tau_count as f64 } else { 0.0 };
                    let l = Array2::<f64>::zeros((np, nu));
                    (mu, alpha, beta, l, tau, 1_usize, true, gamma)
                },
            )
        } else {
            estimation::solve_joint_with_lowrank(
                &y,
                &d,
                &delta.view(),
                ln_eff,
                max_iter as usize,
                tol,
                x_view.as_ref(),
            )
        };

        match result {
            Some((mu, alpha, beta, l, tau, n_iters, did_converge, gamma)) => {
                *tau_out = tau;
                *mu_out = mu;
                *n_iterations_out = n_iters as i32;
                *converged_out = if did_converge { 1 } else { 0 };

                if !alpha_ptr.is_null() {
                    let alpha_slice = slice::from_raw_parts_mut(alpha_ptr, nu);
                    alpha_slice.copy_from_slice(alpha.as_slice().unwrap());
                }
                if !beta_ptr.is_null() {
                    let beta_slice = slice::from_raw_parts_mut(beta_ptr, np);
                    beta_slice.copy_from_slice(beta.as_slice().unwrap());
                }
                if !l_ptr.is_null() {
                    array2_to_ptr(&l, l_ptr);
                }

                let mut n_treated_cells: i32 = 0;
                if !tau_vec_ptr.is_null() {
                    let mut tau_values: Vec<f64> = Vec::new();
                    for t in 0..np {
                        for i in 0..nu {
                            if d[[t, i]] == 1.0 && y[[t, i]].is_finite() {
                                let x_gamma_val = match (&gamma, &x_view) {
                                    (Some(g), Some(xm)) => {
                                        let obs_idx = t * nu + i;
                                        xm.row(obs_idx).dot(g)
                                    }
                                    _ => 0.0,
                                };
                                tau_values.push(
                                    y[[t, i]] - mu - alpha[i] - beta[t] - l[[t, i]] - x_gamma_val,
                                );
                            }
                        }
                    }
                    n_treated_cells = tau_values.len() as i32;
                    let tau_slice = slice::from_raw_parts_mut(tau_vec_ptr, tau_values.len());
                    tau_slice.copy_from_slice(&tau_values);
                } else {
                    for t in 0..np {
                        for i in 0..nu {
                            if d[[t, i]] == 1.0 && y[[t, i]].is_finite() {
                                n_treated_cells += 1;
                            }
                        }
                    }
                }
                if !n_treated_out.is_null() {
                    *n_treated_out = n_treated_cells;
                }

                // Write gamma to output buffer.
                if !gamma_out.is_null() && p > 0 {
                    if let Some(ref g) = gamma {
                        let gamma_slice = slice::from_raw_parts_mut(gamma_out, p);
                        for j in 0..p {
                            gamma_slice[j] = g[j];
                        }
                    }
                }

                TropError::Success.code()
            }
            None => TropError::Convergence.code(),
        }
    })
}

/// Bootstrap variance estimation for Joint with covariates.
///
/// # Safety
///
/// All pointer arguments must be non-null and point to properly sized,
/// aligned buffers. The caller must guarantee that buffer lengths match
/// the declared dimensions.
///
/// # Safety
///
/// All pointer arguments must be non-null and point to properly sized,
/// aligned buffers. The caller must guarantee that buffer lengths match
/// the declared dimensions.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn stata_bootstrap_trop_variance_joint_with_covariates(
    y_ptr: *const f64,
    d_ptr: *const f64,
    n_periods: i32,
    n_units: i32,
    lambda_time: f64,
    lambda_unit: f64,
    lambda_nn: f64,
    n_bootstrap: i32,
    max_iter: i32,
    tol: f64,
    seed: u64,
    alpha: f64,
    ddof: i32,
    estimates_ptr: *mut f64,
    se_out: *mut f64,
    ci_lower_out: *mut f64,
    ci_upper_out: *mut f64,
    n_valid_out: *mut i32,
    x_ptr: *const f64,
    n_covariates: i32,
) -> i32 {
    catch_panic!({
        if y_ptr.is_null() || d_ptr.is_null() || se_out.is_null() {
            return TropError::NullPointer.code();
        }

        if n_covariates > 0 && x_ptr.is_null() {
            return TropError::NullPointer.code();
        }

        let np = n_periods as usize;
        let nu = n_units as usize;

        let y = ptr_to_array2(y_ptr, np, nu);
        let d = ptr_to_array2(d_ptr, np, nu);

        if let Err(err) = loocv::check_simultaneous_adoption(&d) {
            return err.code();
        }

        let x_opt = ptr_to_x_matrix(x_ptr, np, nu, n_covariates as usize);
        let x_view = x_opt.as_ref().map(|a| a.view());

        let alpha_eff = if alpha <= 0.0 || alpha >= 1.0 { 0.05 } else { alpha };
        let ddof_u8: u8 = if ddof == 0 { 0 } else { 1 };

        let result = bootstrap::bootstrap_trop_variance_joint_full(
            &y,
            &d,
            lambda_time,
            lambda_unit,
            lambda_nn,
            n_bootstrap as usize,
            max_iter as usize,
            tol,
            seed,
            alpha_eff,
            ddof_u8,
            x_view.as_ref(),
        );

        *se_out = result.se;

        if !ci_lower_out.is_null() {
            *ci_lower_out = result.ci_lower;
        }
        if !ci_upper_out.is_null() {
            *ci_upper_out = result.ci_upper;
        }
        if !n_valid_out.is_null() {
            *n_valid_out = result.n_valid as i32;
        }

        if !estimates_ptr.is_null() {
            let est_slice = slice::from_raw_parts_mut(estimates_ptr, result.estimates.len());
            est_slice.copy_from_slice(&result.estimates);
        }

        TropError::Success.code()
    })
}

// ---------------------------------------------------------------------------
// Survey Diagnostics — C ABI exports
// ---------------------------------------------------------------------------

/// Compute survey diagnostics: Kish DEFF and high-FPC stratum detection.
///
/// This is a pure diagnostic function that does not alter any computation.
/// Call it after a successful Rao-Wu bootstrap to obtain diagnostic scalars.
///
/// # Output parameters
/// * `deff_weights_out` — Kish (1965) design effect due to unequal weighting.
/// * `max_fh_out` — Maximum sampling fraction across all strata (NaN if no FPC).
/// * `n_high_fpc_out` — Number of strata with f_h > 0.5.
/// * `high_fpc_fh_ptr` — If non-null, filled with f_h values for high-FPC strata
///   (caller must allocate at least `n_high_fpc_out` doubles).
///
/// # Safety
/// All pointers must be non-null and point to properly sized buffers.
/// `fpc_ptr` may be null if no finite population correction is applied.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn stata_compute_survey_diagnostics(
    strata_ptr: *const i64,
    psu_ptr: *const i64,
    fpc_ptr: *const f64,
    unit_weights_ptr: *const f64,
    n_units: i32,
    deff_weights_out: *mut f64,
    max_fh_out: *mut f64,
    n_high_fpc_out: *mut i32,
    high_fpc_fh_ptr: *mut f64,
    high_fpc_max_elements: i32,
) -> i32 {
    catch_panic!({
        if strata_ptr.is_null()
            || psu_ptr.is_null()
            || unit_weights_ptr.is_null()
            || deff_weights_out.is_null()
            || max_fh_out.is_null()
            || n_high_fpc_out.is_null()
        {
            return TropError::NullPointer.code();
        }

        let nu = n_units as usize;
        let strata = slice::from_raw_parts(strata_ptr, nu);
        let psu = slice::from_raw_parts(psu_ptr, nu);
        let fpc: Option<&[f64]> = if fpc_ptr.is_null() {
            None
        } else {
            Some(slice::from_raw_parts(fpc_ptr, nu))
        };
        let unit_weights = slice::from_raw_parts(unit_weights_ptr, nu);

        let diag = bootstrap::compute_survey_diagnostics(strata, psu, fpc, unit_weights);

        *deff_weights_out = diag.deff_weights;
        *max_fh_out = diag.max_fh;
        *n_high_fpc_out = diag.n_high_fpc as i32;

        // Write high-FPC f_h values if buffer provided.
        if !high_fpc_fh_ptr.is_null() && high_fpc_max_elements > 0 {
            let max_write = (high_fpc_max_elements as usize).min(diag.high_fpc_strata.len());
            let out_slice = slice::from_raw_parts_mut(high_fpc_fh_ptr, max_write);
            for (i, s) in diag.high_fpc_strata.iter().take(max_write).enumerate() {
                out_slice[i] = s.f_h;
            }
        }

        TropError::Success.code()
    })
}

/// Returns the condition number from the most recent SVD solve in
/// [`estimation::solve_lstsq_small`].  This allows the Mata/C layer to
/// expose `e(condition_number)` for covariate diagnostics without passing
/// the value through every intermediate function signature.
///
/// Returns NaN if no SVD solve has been performed on the current thread.
#[no_mangle]
pub extern "C" fn stata_get_last_condition_number() -> f64 {
    estimation::LAST_CONDITION_NUMBER.with(|c| c.get())
}

/// Returns the inverse condition number (rcond) from the most recent
/// covariate WLS solve (X'WX system).
///
/// Only set to a finite value when the SVD fallback path is triggered
/// (i.e., Cholesky factorization of X'WX failed due to near-singularity).
/// On the Cholesky success path this returns `NaN` because the singular
/// values are not computed and no reliable condition estimate is available.
///
/// Returns `NaN` if:
/// - No covariate solve has been performed on the current thread, or
/// - The solve succeeded via Cholesky (condition number not computed), or
/// - No covariates were specified in the model.
///
/// **Distinction from [`stata_get_last_condition_number`]:** that function
/// records the condition number of the *main model* SVD in `solve_lstsq_small`;
/// this function tracks only the covariate sub-problem (γ WLS).
#[no_mangle]
pub extern "C" fn stata_get_last_covariate_rcond() -> f64 {
    estimation::LAST_COVARIATE_RCOND.with(|c| c.get())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_codes() {
        assert_eq!(TropError::Success.code(), 0);
        assert_eq!(TropError::NullPointer.code(), 1);
        assert_eq!(TropError::RustPanic.code(), 8);
    }
}

#[cfg(test)]
mod ffi_error_propagation_tests {
    use super::*;
    use std::ptr;

    /// Test: Null pointer to `stata_estimate_twostep` returns NullPointer error code.
    #[test]
    fn test_ffi_null_pointer_twostep() {
        unsafe {
            let code = stata_estimate_twostep(
                ptr::null(),     // y_ptr (null)
                ptr::null(),     // d_ptr
                ptr::null(),     // control_mask_ptr
                ptr::null(),     // time_dist_ptr
                0, 0,            // n_periods, n_units
                0.0, 0.0, 0.0,  // lambdas
                100, 1e-4,       // max_iter, tol
                ptr::null_mut(), // att_out
                ptr::null_mut(), // tau_ptr
                ptr::null_mut(), // alpha_ptr
                ptr::null_mut(), // beta_ptr
                ptr::null_mut(), // l_ptr
                ptr::null_mut(), // n_treated_out
                ptr::null_mut(), // n_iterations_out
                ptr::null_mut(), // converged_out
                ptr::null_mut(), // converged_by_obs_ptr
                ptr::null_mut(), // n_iters_by_obs_ptr
                ptr::null(),     // x_ptr
                0,               // n_covariates
                ptr::null_mut(), // gamma_out
            );
            assert_eq!(
                code,
                TropError::NullPointer.code(),
                "Null y_ptr should return NullPointer, got {}",
                code
            );
        }
    }

    /// Test: Null pointer to `stata_estimate_joint` returns NullPointer error code.
    #[test]
    fn test_ffi_null_pointer_joint() {
        unsafe {
            let code = stata_estimate_joint(
                ptr::null(),     // y_ptr (null)
                ptr::null(),     // d_ptr
                0, 0,            // n_periods, n_units
                0.0, 0.0, 0.0,  // lambdas
                100, 1e-4,       // max_iter, tol
                ptr::null_mut(), // tau_out
                ptr::null_mut(), // mu_out
                ptr::null_mut(), // alpha_ptr
                ptr::null_mut(), // beta_ptr
                ptr::null_mut(), // l_ptr
                ptr::null_mut(), // n_iterations_out
                ptr::null_mut(), // converged_out
                ptr::null_mut(), // tau_vec_ptr
                ptr::null_mut(), // n_treated_out
                ptr::null(),     // x_ptr
                0,               // n_covariates
                ptr::null_mut(), // gamma_out
            );
            assert_eq!(
                code,
                TropError::NullPointer.code(),
                "Null y_ptr should return NullPointer, got {}",
                code
            );
        }
    }

    /// Test: Null pointer to `stata_bootstrap_trop_variance` returns NullPointer.
    #[test]
    fn test_ffi_null_pointer_bootstrap() {
        unsafe {
            let code = stata_bootstrap_trop_variance(
                ptr::null(),     // y_ptr (null)
                ptr::null(),     // d_ptr
                ptr::null(),     // control_mask_ptr
                ptr::null(),     // time_dist_ptr
                0, 0,            // n_periods, n_units
                0.0, 0.0, 0.0,  // lambdas
                50,              // n_bootstrap
                100, 1e-4,       // max_iter, tol
                42,              // seed
                0.05,            // alpha
                1,               // ddof
                ptr::null_mut(), // estimates_ptr
                ptr::null_mut(), // se_out
                ptr::null_mut(), // ci_lower_out
                ptr::null_mut(), // ci_upper_out
                ptr::null_mut(), // n_valid_out
                ptr::null(),     // x_ptr
                0,               // n_covariates
            );
            assert_eq!(
                code,
                TropError::NullPointer.code(),
                "Null y_ptr should return NullPointer, got {}",
                code
            );
        }
    }

    /// Test: Null pointer to LOOCV grid search returns NullPointer.
    #[test]
    fn test_ffi_null_pointer_loocv() {
        unsafe {
            let code = stata_loocv_grid_search(
                ptr::null(),     // y_ptr (null)
                ptr::null(),     // d_ptr
                ptr::null(),     // control_mask_ptr
                ptr::null(),     // time_dist_ptr
                0, 0,            // n_periods, n_units
                ptr::null(),     // lambda_time_grid_ptr
                0,               // lambda_time_grid_len
                ptr::null(),     // lambda_unit_grid_ptr
                0,               // lambda_unit_grid_len
                ptr::null(),     // lambda_nn_grid_ptr
                0,               // lambda_nn_grid_len
                100, 1e-4,       // max_iter, tol
                ptr::null_mut(), // best_lambda_time_out
                ptr::null_mut(), // best_lambda_unit_out
                ptr::null_mut(), // best_lambda_nn_out
                ptr::null_mut(), // best_score_out
                ptr::null_mut(), // n_valid_out
                ptr::null_mut(), // n_attempted_out
                ptr::null_mut(), // first_failed_t_out
                ptr::null_mut(), // first_failed_i_out
                ptr::null_mut(), // stage1_lambda_time_out
                ptr::null_mut(), // stage1_lambda_unit_out
                ptr::null_mut(), // stage1_lambda_nn_out
                ptr::null(),     // x_ptr
                0,               // n_covariates
            );
            assert_eq!(
                code,
                TropError::NullPointer.code(),
                "Null y_ptr should return NullPointer, got {}",
                code
            );
        }
    }

    /// Test: Invalid dimensions (zero) to estimation function.
    /// With valid pointers but zero dimensions, should return an error (not crash).
    #[test]
    fn test_ffi_invalid_dimension_twostep() {
        let y = [0.0_f64; 1];
        let d = [0.0_f64; 1];
        let control_mask = [0u8; 1];
        let time_dist = [0i64; 1];
        let mut att = 0.0_f64;
        let mut tau = 0.0_f64;
        let mut alpha = 0.0_f64;
        let mut beta = 0.0_f64;
        let mut l = 0.0_f64;
        let mut n_treated = 0i32;
        let mut n_iterations = 0i32;
        let mut converged = 0i32;
        let mut converged_by_obs = 0i32;
        let mut n_iters_by_obs = 0i32;
        let mut gamma = 0.0_f64;

        unsafe {
            let code = stata_estimate_twostep(
                y.as_ptr(),
                d.as_ptr(),
                control_mask.as_ptr(),
                time_dist.as_ptr(),
                0, 0,  // zero dimensions
                1.0, 1.0, 1.0,
                100, 1e-4,
                &mut att,
                &mut tau,
                &mut alpha,
                &mut beta,
                &mut l,
                &mut n_treated,
                &mut n_iterations,
                &mut converged,
                &mut converged_by_obs,
                &mut n_iters_by_obs,
                ptr::null(),  // x_ptr
                0,            // n_covariates
                &mut gamma,   // gamma_out
            );
            // Should return an error code (non-zero), not crash
            assert_ne!(
                code, 0,
                "Zero dimensions should return an error code, not success"
            );
        }
    }
}

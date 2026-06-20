/**
 * stata_bridge.c
 *
 * C bridge between Stata's plugin interface and the Rust core library.
 * Responsibilities:
 *   - Plugin entry point and command dispatch
 *   - Reading panel data from Stata variables into column-major matrices
 *   - Stata missing value / IEEE NaN conversion
 *   - Writing estimation results back to Stata scalars and matrices
 *   - Structured logging and error translation
 */

#include "stata_bridge.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>

/* ============================================================================
 * Global Variables
 * ============================================================================ */

int g_verbose_level = TROP_VERBOSE_NORMAL;

/* ============================================================================
 * Logging Implementation
 * ============================================================================ */

void trop_log(int level, const char *tag, const char *fmt, ...) {
    if (level > g_verbose_level) return;
    
    char buffer[1024];
    va_list args;
    va_start(args, fmt);
    
    int prefix_len = snprintf(buffer, sizeof(buffer), "[trop %s] ", tag);
    vsnprintf(buffer + prefix_len, sizeof(buffer) - prefix_len, fmt, args);
    
    va_end(args);
    
    /* Add newline if not present */
    size_t len = strlen(buffer);
    if (len > 0 && buffer[len-1] != '\n') {
        if (len < sizeof(buffer) - 1) {
            buffer[len] = '\n';
            buffer[len+1] = '\0';
        }
    }
    
    SF_display(buffer);
}

/* ============================================================================
 * Command Parsing
 * ============================================================================ */

TropCommand parse_command(const char *cmd) {
    if (cmd == NULL) return CMD_UNKNOWN;
    
    if (strcmp(cmd, "loocv_twostep") == 0) return CMD_LOOCV_TWOSTEP;
    if (strcmp(cmd, "loocv_twostep_exhaustive") == 0) return CMD_LOOCV_TWOSTEP_EXHAUSTIVE;
    if (strcmp(cmd, "loocv_joint") == 0) return CMD_LOOCV_JOINT;
    if (strcmp(cmd, "loocv_joint_exhaustive") == 0) return CMD_LOOCV_JOINT_EXHAUSTIVE;
    if (strcmp(cmd, "estimate_twostep") == 0) return CMD_ESTIMATE_TWOSTEP;
    if (strcmp(cmd, "estimate_joint") == 0) return CMD_ESTIMATE_JOINT;
    if (strcmp(cmd, "bootstrap_twostep") == 0) return CMD_BOOTSTRAP_TWOSTEP;
    if (strcmp(cmd, "bootstrap_joint") == 0) return CMD_BOOTSTRAP_JOINT;
    if (strcmp(cmd, "bootstrap_rao_wu_twostep") == 0) return CMD_BOOTSTRAP_RAO_WU_TWOSTEP;
    if (strcmp(cmd, "bootstrap_rao_wu_joint") == 0) return CMD_BOOTSTRAP_RAO_WU_JOINT;
    if (strcmp(cmd, "distance_matrix") == 0) return CMD_DISTANCE_MATRIX;
    
    return CMD_UNKNOWN;
}

/* ============================================================================
 * Error Code Translation
 * ============================================================================ */

void translate_error_code(int rust_code) {
    char msg[256];
    
    switch (rust_code) {
        case TROP_SUCCESS:
            return; /* No error */
        case TROP_ERR_NULL_POINTER:
            snprintf(msg, sizeof(msg), "internal error: null pointer\n");
            break;
        case TROP_ERR_INVALID_DIM:
            snprintf(msg, sizeof(msg), "invalid matrix dimensions\n");
            break;
        case TROP_ERR_NO_CONTROL:
            snprintf(msg, sizeof(msg), "no control observations found in data\n");
            break;
        case TROP_ERR_NO_TREATED:
            snprintf(msg, sizeof(msg), "no treated observations found in data\n");
            break;
        case TROP_ERR_CONVERGENCE:
            snprintf(msg, sizeof(msg), "algorithm did not converge\n");
            break;
        case TROP_ERR_SINGULAR:
            snprintf(msg, sizeof(msg), "singular matrix encountered\n");
            break;
        case TROP_ERR_MEMORY:
            snprintf(msg, sizeof(msg), "memory allocation failure\n");
            break;
        case TROP_ERR_RUST_PANIC:
            snprintf(msg, sizeof(msg), "internal error in Rust library\n");
            break;
        case TROP_ERR_LOOCV_FAIL:
            snprintf(msg, sizeof(msg), "LOOCV search failed\n");
            break;
        case TROP_ERR_BOOTSTRAP_FAIL:
            snprintf(msg, sizeof(msg), "bootstrap estimation failed\n");
            break;
        case TROP_ERR_COMPUTATION:
            snprintf(msg, sizeof(msg), "computation error\n");
            break;
        default:
            snprintf(msg, sizeof(msg), "unknown error (code %d)\n", rust_code);
            break;
    }
    
    SF_error(msg);
}

/* ============================================================================
 * Dimension Reading
 * ============================================================================ */

ST_retcode read_dimensions(ST_int *n_units_out, ST_int *n_periods_out) {
    double n_units_d, n_periods_d;
    ST_retcode rc;
    
    rc = SF_scal_use("__trop_n_units", &n_units_d);
    if (rc != 0) {
        TROP_LOG_ERROR("failed to read scalar __trop_n_units");
        return TROP_ERR_SCALAR_FAIL;
    }
    
    rc = SF_scal_use("__trop_n_periods", &n_periods_d);
    if (rc != 0) {
        TROP_LOG_ERROR("failed to read scalar __trop_n_periods");
        return TROP_ERR_SCALAR_FAIL;
    }
    
    *n_units_out = (ST_int)n_units_d;
    *n_periods_out = (ST_int)n_periods_d;
    
    if (*n_units_out <= 0 || *n_periods_out <= 0) {
        TROP_LOG_ERROR("invalid dimensions: n_units=%d, n_periods=%d", 
                       *n_units_out, *n_periods_out);
        return TROP_ERR_INVALID_DIM;
    }
    
    TROP_LOG_DEBUG("dimensions: n_units=%d, n_periods=%d", 
                   *n_units_out, *n_periods_out);
    
    return TROP_SUCCESS;
}

/* ============================================================================
 * Panel Data Reading
 * ============================================================================ */

ST_retcode read_panel_to_matrix(
    ST_int varindex,
    ST_int n_periods,
    ST_int n_units,
    double *out_matrix
) {
    ST_int obs, t, i, idx;
    ST_int in1, in2, nobs;
    double val;
    ST_retcode rc;
    
    /* Read panel/time variable indices from scalars.
     * Each observation carries its own (unit, period) coordinate so that
     * unbalanced panels are mapped correctly into the N×T matrix, with
     * missing cells left as NaN.
     */
    double panel_vidx_d, time_vidx_d;
    ST_int panel_varindex, time_varindex;
    double panel_val, time_val;
    
    rc = SF_scal_use("__trop_panel_varindex", &panel_vidx_d);
    if (rc != 0) {
        TROP_LOG_ERROR("failed to read scalar __trop_panel_varindex");
        return TROP_ERR_SCALAR_FAIL;
    }
    rc = SF_scal_use("__trop_time_varindex", &time_vidx_d);
    if (rc != 0) {
        TROP_LOG_ERROR("failed to read scalar __trop_time_varindex");
        return TROP_ERR_SCALAR_FAIL;
    }
    panel_varindex = (ST_int)panel_vidx_d;
    time_varindex = (ST_int)time_vidx_d;
    
    in1 = SF_in1();
    in2 = SF_in2();
    nobs = SF_nobs();
    
    TROP_LOG_DEBUG("read_panel_to_matrix: varindex=%d, panel_vi=%d, time_vi=%d, in1=%d, in2=%d, nobs=%d", 
                   varindex, panel_varindex, time_varindex, in1, in2, nobs);
    
    /* Initialize matrix with NaN; cells without observations remain missing */
    for (idx = 0; idx < n_units * n_periods; idx++) {
        out_matrix[idx] = NAN;
    }
    
    /* 
     * Place each observation at its (unit, period) coordinate.
     * Unbalanced panels are handled naturally: cells without an
     * observation remain NaN.
     *
     * Column-major storage: index = i * n_periods + t
     */
    for (obs = in1; obs <= in2; obs++) {
        if (!SF_ifobs(obs)) continue;
        
        /* Read panel_idx and time_idx for this observation */
        rc = SF_vdata(panel_varindex, obs, &panel_val);
        if (rc != 0) {
            TROP_LOG_ERROR("SF_vdata failed for panel_idx: obs=%d, rc=%d", obs, rc);
            return rc;
        }
        rc = SF_vdata(time_varindex, obs, &time_val);
        if (rc != 0) {
            TROP_LOG_ERROR("SF_vdata failed for time_idx: obs=%d, rc=%d", obs, rc);
            return rc;
        }
        
        /* Convert from 1-based (Stata/egen group) to 0-based */
        i = (ST_int)panel_val - 1;
        t = (ST_int)time_val - 1;
        
        /* Bounds check */
        if (i < 0 || i >= n_units || t < 0 || t >= n_periods) {
            TROP_LOG_DEBUG("observation %d beyond panel bounds: i=%d, t=%d (n_units=%d, n_periods=%d)",
                           obs, i, t, n_units, n_periods);
            continue;
        }
        
        /* Read the actual variable value */
        rc = SF_vdata(varindex, obs, &val);
        if (rc != 0) {
            TROP_LOG_ERROR("SF_vdata failed: varindex=%d, obs=%d, rc=%d", varindex, obs, rc);
            return rc;
        }
        
        /* Convert Stata missing to NaN */
        val = stata_to_rust_value(val);
        
        /* Column-major storage: index = i * n_periods + t */
        out_matrix[i * n_periods + t] = val;
    }
    
    return TROP_SUCCESS;
}

/* ============================================================================
 * Control Mask Reading
 * ============================================================================ */

ST_retcode read_control_mask(
    ST_int varindex,
    ST_int n_periods,
    ST_int n_units,
    unsigned char *out_mask
) {
    ST_int obs, t, i;
    ST_int in1, in2;
    double val;
    ST_retcode rc;
    
    /* Use panel/time variable indices for coordinate mapping, same
     * approach as read_panel_to_matrix.  Missing (unit, period) cells
     * stay 0 (treated) so that the control mask is correct for
     * unbalanced panels.
     */
    double panel_vidx_d, time_vidx_d;
    ST_int panel_varindex, time_varindex;
    double panel_val, time_val;
    
    rc = SF_scal_use("__trop_panel_varindex", &panel_vidx_d);
    if (rc != 0) {
        TROP_LOG_ERROR("failed to read scalar __trop_panel_varindex");
        return TROP_ERR_SCALAR_FAIL;
    }
    rc = SF_scal_use("__trop_time_varindex", &time_vidx_d);
    if (rc != 0) {
        TROP_LOG_ERROR("failed to read scalar __trop_time_varindex");
        return TROP_ERR_SCALAR_FAIL;
    }
    panel_varindex = (ST_int)panel_vidx_d;
    time_varindex = (ST_int)time_vidx_d;
    
    in1 = SF_in1();
    in2 = SF_in2();
    
    /* Initialize mask: 0 = treated/missing.
     * Missing D is treated as D=0 for the control mask, but since Y is
     * NaN for those cells, the (1-W) mask in the estimator effectively
     * excludes them from the objective. */
    memset(out_mask, 0, n_units * n_periods);
    
    for (obs = in1; obs <= in2; obs++) {
        if (!SF_ifobs(obs)) continue;
        
        /* Read panel_idx and time_idx for this observation */
        rc = SF_vdata(panel_varindex, obs, &panel_val);
        if (rc != 0) {
            TROP_LOG_ERROR("SF_vdata failed for panel_idx: obs=%d, rc=%d", obs, rc);
            return rc;
        }
        rc = SF_vdata(time_varindex, obs, &time_val);
        if (rc != 0) {
            TROP_LOG_ERROR("SF_vdata failed for time_idx: obs=%d, rc=%d", obs, rc);
            return rc;
        }
        
        /* Convert from 1-based to 0-based */
        i = (ST_int)panel_val - 1;
        t = (ST_int)time_val - 1;
        
        if (i < 0 || i >= n_units || t < 0 || t >= n_periods) continue;
        
        rc = SF_vdata(varindex, obs, &val);
        if (rc != 0) {
            TROP_LOG_ERROR("failed to read control mask at obs %d", obs);
            return rc;
        }
        
        /* Column-major storage */
        /* Control = 1 if D == 0 (untreated), otherwise 0 */
        out_mask[i * n_periods + t] = (val == 0.0) ? 1 : 0;
    }
    
    return TROP_SUCCESS;
}

/* ============================================================================
 * Lambda Grid Reading
 * ============================================================================ */

ST_retcode read_lambda_grid(
    const char *matname,
    double **out_grid,
    int *out_len
) {
    ST_int nrows, ncols, len;
    ST_int j;
    double val;
    double *grid;
    ST_retcode rc;
    
    nrows = SF_row((char *)matname);
    ncols = SF_col((char *)matname);
    
    if (nrows <= 0 || ncols <= 0) {
        TROP_LOG_ERROR("matrix %s not found or empty", matname);
        return TROP_ERR_MAT_NOT_FOUND;
    }
    
    /* Grid can be row or column vector */
    len = (nrows == 1) ? ncols : nrows;
    
    grid = (double *)malloc(len * sizeof(double));
    if (grid == NULL) {
        TROP_LOG_ERROR("failed to allocate memory for lambda grid");
        return TROP_ERR_MEMORY;
    }
    
    for (j = 0; j < len; j++) {
        if (nrows == 1) {
            rc = SF_mat_el((char *)matname, 1, j + 1, &val);
        } else {
            rc = SF_mat_el((char *)matname, j + 1, 1, &val);
        }
        
        if (rc != 0) {
            TROP_LOG_ERROR("failed to read matrix %s element %d", matname, j + 1);
            free(grid);
            return rc;
        }
        
        grid[j] = val;
    }
    
    *out_grid = grid;
    *out_len = len;
    
    TROP_LOG_DEBUG("read lambda grid %s: length=%d", matname, len);
    
    return TROP_SUCCESS;
}

/* ============================================================================
 * Lambda Infinity Conversion
 *
 * Converts sentinel infinity values (>=1e99 or NaN) in lambda grids to
 * finite values understood by the Rust core.  The threshold (1e99) and
 * replacement (1e10) must stay in sync with the corresponding Mata
 * constants _TROP_LAMBDA_INF_THRESHOLD and _TROP_LAMBDA_NN_INF_VALUE.
 *
 * This function is the sole conversion point for grid values; the Rust
 * library assumes all grid entries are finite after this step.
 *
 * Conversion rules:
 *   lambda_time : inf -> 0.0   (uniform time weights, exp(-0*d) = 1)
 *   lambda_unit : inf -> 0.0   (uniform unit weights, exp(-0*d) = 1)
 *   lambda_nn   : inf -> 1e10  (strong nuclear-norm penalty, L ~ 0)
 *
 * Note: lambda_nn = 0 means NO regularisation (full-rank L), which is
 * the opposite of lambda_nn = inf (maximum regularisation, L ~ 0).
 * ============================================================================ */

void convert_lambda_infinity(
    double *grid,
    int len,
    const char *param_type
) {
    double inf_replacement;
    int i;
    
    /* Determine replacement value based on parameter type */
    if (strcmp(param_type, "nn") == 0) {
    /* lambda_nn = inf -> 1e10 (strong nuclear-norm penalty, L ~ 0) */
        inf_replacement = 1e10;
    } else {
        /* lambda_time / lambda_unit = inf -> 0.0 (uniform weights) */
        inf_replacement = 0.0;
    }
    
    /* Convert infinity values */
    for (i = 0; i < len; i++) {
        /* Check for Stata missing (NaN) or large values (≥1e99) */
        if (isnan(grid[i]) || grid[i] >= 1e99) {
            TROP_LOG_DEBUG("converting lambda_%s[%d] from %g to %g (infinity)",
                           param_type, i, grid[i], inf_replacement);
            grid[i] = inf_replacement;
        }
    }
}

/* ============================================================================
 * Time Distance Matrix Reading
 * ============================================================================ */

ST_retcode read_time_dist_matrix(
    const char *matname,
    ST_int n_periods,
    int64_t *out_matrix
) {
    ST_int nrows, ncols;
    ST_int t1, t2;
    double val;
    ST_retcode rc;
    
    nrows = SF_row((char *)matname);
    ncols = SF_col((char *)matname);
    
    if (nrows != n_periods || ncols != n_periods) {
        TROP_LOG_ERROR("time distance matrix %s has wrong dimensions: %dx%d, expected %dx%d",
                       matname, nrows, ncols, n_periods, n_periods);
        return TROP_ERR_INVALID_DIM;
    }
    
    /* Read matrix in column-major order */
    for (t2 = 0; t2 < n_periods; t2++) {
        for (t1 = 0; t1 < n_periods; t1++) {
            rc = SF_mat_el((char *)matname, t1 + 1, t2 + 1, &val);
            if (rc != 0) {
                TROP_LOG_ERROR("failed to read time_dist[%d,%d]", t1 + 1, t2 + 1);
                return rc;
            }
            
            /* Column-major: index = t2 * n_periods + t1 */
            out_matrix[t2 * n_periods + t1] = (int64_t)val;
        }
    }
    
    return TROP_SUCCESS;
}


/* ============================================================================
 * Result Writing Functions
 * ============================================================================ */

ST_retcode write_vector_to_matrix(
    const char *matname,
    const double *data,
    int len,
    int is_row
) {
    ST_int j;
    double val;
    ST_retcode rc;
    
    for (j = 0; j < len; j++) {
        val = rust_to_stata_value(data[j]);
        
        if (is_row) {
            rc = SF_mat_store((char *)matname, 1, j + 1, val);
        } else {
            rc = SF_mat_store((char *)matname, j + 1, 1, val);
        }
        
        if (rc != 0) {
            TROP_LOG_ERROR("failed to write to matrix %s at position %d", matname, j + 1);
            return rc;
        }
    }
    
    return TROP_SUCCESS;
}

ST_retcode write_matrix_to_stata(
    const char *matname,
    const double *data,
    int nrows,
    int ncols
) {
    ST_int r, c;
    double val;
    ST_retcode rc;
    
    /* Data is in column-major order */
    for (c = 0; c < ncols; c++) {
        for (r = 0; r < nrows; r++) {
            val = rust_to_stata_value(data[c * nrows + r]);
            
            rc = SF_mat_store((char *)matname, r + 1, c + 1, val);
            if (rc != 0) {
                TROP_LOG_ERROR("failed to write to matrix %s at [%d,%d]", 
                               matname, r + 1, c + 1);
                return rc;
            }
        }
    }
    
    return TROP_SUCCESS;
}

/* ============================================================================
 * Command Handler: LOOCV Twostep
 * ============================================================================ */

static ST_retcode handle_loocv_twostep(void) {
    ST_int n_units, n_periods;
    double *y_matrix = NULL;
    double *d_matrix = NULL;
    unsigned char *control_mask = NULL;
    int64_t *time_dist = NULL;
    double *lambda_time_grid = NULL;
    double *lambda_unit_grid = NULL;
    double *lambda_nn_grid = NULL;
    double *x_buf = NULL;
    int lambda_time_len, lambda_unit_len, lambda_nn_len;
    int n_covariates = 0;
    double max_iter_d, tol;
    int max_iter;
    double best_time, best_unit, best_nn, best_score;
    int n_valid, n_attempted;
    int first_failed_t, first_failed_i;  /* first failed LOOCV observation indices */
    double stage1_time, stage1_unit, stage1_nn;  /* Stage-1 univariate init (Footnote 2) */
    ST_retcode rc;
    int rust_rc;
    
    first_failed_t = -1;
    first_failed_i = -1;
    stage1_time = 0.0;
    stage1_unit = 0.0;
    stage1_nn = 0.0;
    
    TROP_LOG_INFO("starting LOOCV grid search (twostep)");
    
    /* Read dimensions */
    rc = read_dimensions(&n_units, &n_periods);
    if (rc != TROP_SUCCESS) goto cleanup;
    
    /* Allocate memory */
    y_matrix = (double *)malloc(n_units * n_periods * sizeof(double));
    d_matrix = (double *)malloc(n_units * n_periods * sizeof(double));
    control_mask = (unsigned char *)malloc(n_units * n_periods);
    time_dist = (int64_t *)malloc(n_periods * n_periods * sizeof(int64_t));
    
    if (!y_matrix || !d_matrix || !control_mask || !time_dist) {
        TROP_LOG_ERROR("memory allocation failed");
        rc = TROP_ERR_MEMORY;
        goto cleanup;
    }
    
    /* Read data from Stata variables (indices from scalars) */
    double y_idx_d, d_idx_d, ctrl_idx_d;
    SF_scal_use("__trop_y_varindex", &y_idx_d);
    SF_scal_use("__trop_d_varindex", &d_idx_d);
    SF_scal_use("__trop_ctrl_varindex", &ctrl_idx_d);
    
    rc = read_panel_to_matrix((ST_int)y_idx_d, n_periods, n_units, y_matrix);
    if (rc != TROP_SUCCESS) goto cleanup;
    
    rc = read_panel_to_matrix((ST_int)d_idx_d, n_periods, n_units, d_matrix);
    if (rc != TROP_SUCCESS) goto cleanup;
    
    rc = read_control_mask((ST_int)ctrl_idx_d, n_periods, n_units, control_mask);
    if (rc != TROP_SUCCESS) goto cleanup;
    
    /* Read time distance matrix */
    rc = read_time_dist_matrix("__trop_time_dist", n_periods, time_dist);
    if (rc != TROP_SUCCESS) goto cleanup;
    
    /* Read lambda grids */
    rc = read_lambda_grid("__trop_lambda_time_grid", &lambda_time_grid, &lambda_time_len);
    if (rc != TROP_SUCCESS) goto cleanup;
    
    rc = read_lambda_grid("__trop_lambda_unit_grid", &lambda_unit_grid, &lambda_unit_len);
    if (rc != TROP_SUCCESS) goto cleanup;
    
    rc = read_lambda_grid("__trop_lambda_nn_grid", &lambda_nn_grid, &lambda_nn_len);
    if (rc != TROP_SUCCESS) goto cleanup;
    
    /* Convert infinity sentinel values in grids */
    convert_lambda_infinity(lambda_time_grid, lambda_time_len, "time");
    convert_lambda_infinity(lambda_unit_grid, lambda_unit_len, "unit");
    convert_lambda_infinity(lambda_nn_grid, lambda_nn_len, "nn");
    
    /* Read algorithm parameters */
    SF_scal_use("__trop_max_iter", &max_iter_d);
    SF_scal_use("__trop_tol", &tol);
    
    max_iter = (int)max_iter_d;

    /* --- Covariate support --- */
    {
        double nc_val = 0;
        if (SF_scal_use("__trop_n_covariates", &nc_val) == 0 && nc_val > 0) {
            n_covariates = (int)nc_val;
        }
    }
    if (n_covariates > 0) {
        int n_obs = (int)n_periods * (int)n_units;
        x_buf = (double *)malloc((size_t)n_obs * (size_t)n_covariates * sizeof(double));
        if (!x_buf) {
            TROP_LOG_ERROR("covariate memory allocation failed");
            rc = TROP_ERR_MEMORY;
            goto cleanup;
        }
        {
            int row, col;
            double val;
            for (col = 0; col < n_covariates; col++) {
                for (row = 0; row < n_obs; row++) {
                    if (SF_mat_el("__trop_covariates", row + 1, col + 1, &val) != 0) {
                        val = 0.0;
                    }
                    x_buf[row + col * n_obs] = val;
                }
            }
        }
    }

    if (n_covariates > 0) {
        rust_rc = stata_loocv_grid_search_with_covariates(
            y_matrix, d_matrix, control_mask, time_dist,
            n_periods, n_units,
            lambda_time_grid, lambda_time_len,
            lambda_unit_grid, lambda_unit_len,
            lambda_nn_grid, lambda_nn_len,
            max_iter, tol,
            &best_time, &best_unit, &best_nn, &best_score,
            &n_valid, &n_attempted,
            &first_failed_t, &first_failed_i,
            &stage1_time, &stage1_unit, &stage1_nn,
            x_buf, n_covariates
        );
    } else {
        rust_rc = stata_loocv_grid_search(
            y_matrix, d_matrix, control_mask, time_dist,
            n_periods, n_units,
            lambda_time_grid, lambda_time_len,
            lambda_unit_grid, lambda_unit_len,
            lambda_nn_grid, lambda_nn_len,
            max_iter, tol,
            &best_time, &best_unit, &best_nn, &best_score,
            &n_valid, &n_attempted,
            &first_failed_t, &first_failed_i,
            &stage1_time, &stage1_unit, &stage1_nn
        );
    }
    
    if (rust_rc != TROP_SUCCESS) {
        translate_error_code(rust_rc);
        rc = rust_rc;
        goto cleanup;
    }
    
    /* Write results to Stata scalars */
    SF_scal_save("__trop_lambda_time", best_time);
    SF_scal_save("__trop_lambda_unit", best_unit);
    SF_scal_save("__trop_lambda_nn", best_nn);
    SF_scal_save("__trop_loocv_score", best_score);
    SF_scal_save("__trop_loocv_n_valid", (double)n_valid);
    SF_scal_save("__trop_loocv_n_attempted", (double)n_attempted);
    /* First failed observation indices (for diagnostics) */
    SF_scal_save("__trop_loocv_first_failed_t", (double)first_failed_t);
    SF_scal_save("__trop_loocv_first_failed_i", (double)first_failed_i);
    /* Stage-1 univariate init (paper Footnote 2); cycling path only. */
    SF_scal_save("__trop_stage1_lambda_time", stage1_time);
    SF_scal_save("__trop_stage1_lambda_unit", stage1_unit);
    SF_scal_save("__trop_stage1_lambda_nn", stage1_nn);
    
    TROP_LOG_INFO("LOOCV complete: lambda_time=%g, lambda_unit=%g, lambda_nn=%g, score=%g, n_valid=%d, n_attempted=%d, first_failed=(%d,%d), stage1=(%g,%g,%g)",
                  best_time, best_unit, best_nn, best_score, n_valid, n_attempted, first_failed_t, first_failed_i,
                  stage1_time, stage1_unit, stage1_nn);
    
    rc = TROP_SUCCESS;
    
cleanup:
    free(y_matrix);
    free(d_matrix);
    free(control_mask);
    free(time_dist);
    free(lambda_time_grid);
    free(lambda_unit_grid);
    free(lambda_nn_grid);
    free(x_buf);
    
    return rc;
}

/* ============================================================================
 * Command Handler: LOOCV Twostep Exhaustive
 * ============================================================================ */

/**
 * Exhaustive (Cartesian) grid search variant of handle_loocv_twostep.
 *
 * Reads the same Stata scalars/matrices as handle_loocv_twostep, then calls
 * stata_loocv_grid_search_exhaustive which enumerates all |grid|^3 triples in
 * parallel.  Writes identical output scalars so the downstream Mata/ADO layers
 * are agnostic to which strategy was used.
 */
static ST_retcode handle_loocv_twostep_exhaustive(void) {
    ST_int n_units, n_periods;
    double *y_matrix = NULL;
    double *d_matrix = NULL;
    unsigned char *control_mask = NULL;
    int64_t *time_dist = NULL;
    double *lambda_time_grid = NULL;
    double *lambda_unit_grid = NULL;
    double *lambda_nn_grid = NULL;
    double *x_buf = NULL;
    int lambda_time_len, lambda_unit_len, lambda_nn_len;
    int n_covariates = 0;
    double max_iter_d, tol;
    int max_iter;
    double best_time, best_unit, best_nn, best_score;
    int n_valid, n_attempted;
    int first_failed_t, first_failed_i;
    ST_retcode rc;
    int rust_rc;

    first_failed_t = -1;
    first_failed_i = -1;

    TROP_LOG_INFO("starting LOOCV exhaustive grid search (twostep)");

    /* Read dimensions */
    rc = read_dimensions(&n_units, &n_periods);
    if (rc != TROP_SUCCESS) goto cleanup;

    /* Allocate memory */
    y_matrix = (double *)malloc(n_units * n_periods * sizeof(double));
    d_matrix = (double *)malloc(n_units * n_periods * sizeof(double));
    control_mask = (unsigned char *)malloc(n_units * n_periods);
    time_dist = (int64_t *)malloc(n_periods * n_periods * sizeof(int64_t));

    if (!y_matrix || !d_matrix || !control_mask || !time_dist) {
        TROP_LOG_ERROR("memory allocation failed");
        rc = TROP_ERR_MEMORY;
        goto cleanup;
    }

    /* Read data from Stata variables (indices from scalars) */
    double y_idx_d, d_idx_d, ctrl_idx_d;
    SF_scal_use("__trop_y_varindex", &y_idx_d);
    SF_scal_use("__trop_d_varindex", &d_idx_d);
    SF_scal_use("__trop_ctrl_varindex", &ctrl_idx_d);

    rc = read_panel_to_matrix((ST_int)y_idx_d, n_periods, n_units, y_matrix);
    if (rc != TROP_SUCCESS) goto cleanup;

    rc = read_panel_to_matrix((ST_int)d_idx_d, n_periods, n_units, d_matrix);
    if (rc != TROP_SUCCESS) goto cleanup;

    rc = read_control_mask((ST_int)ctrl_idx_d, n_periods, n_units, control_mask);
    if (rc != TROP_SUCCESS) goto cleanup;

    /* Read time distance matrix */
    rc = read_time_dist_matrix("__trop_time_dist", n_periods, time_dist);
    if (rc != TROP_SUCCESS) goto cleanup;

    /* Read lambda grids */
    rc = read_lambda_grid("__trop_lambda_time_grid", &lambda_time_grid, &lambda_time_len);
    if (rc != TROP_SUCCESS) goto cleanup;

    rc = read_lambda_grid("__trop_lambda_unit_grid", &lambda_unit_grid, &lambda_unit_len);
    if (rc != TROP_SUCCESS) goto cleanup;

    rc = read_lambda_grid("__trop_lambda_nn_grid", &lambda_nn_grid, &lambda_nn_len);
    if (rc != TROP_SUCCESS) goto cleanup;

    /* Convert infinity sentinel values in grids */
    convert_lambda_infinity(lambda_time_grid, lambda_time_len, "time");
    convert_lambda_infinity(lambda_unit_grid, lambda_unit_len, "unit");
    convert_lambda_infinity(lambda_nn_grid, lambda_nn_len, "nn");

    /* Read algorithm parameters */
    SF_scal_use("__trop_max_iter", &max_iter_d);
    SF_scal_use("__trop_tol", &tol);

    max_iter = (int)max_iter_d;

    /* --- Covariate support --- */
    {
        double nc_val = 0;
        if (SF_scal_use("__trop_n_covariates", &nc_val) == 0 && nc_val > 0) {
            n_covariates = (int)nc_val;
        }
    }
    if (n_covariates > 0) {
        int n_obs = (int)n_periods * (int)n_units;
        x_buf = (double *)malloc((size_t)n_obs * (size_t)n_covariates * sizeof(double));
        if (!x_buf) {
            TROP_LOG_ERROR("covariate memory allocation failed");
            rc = TROP_ERR_MEMORY;
            goto cleanup;
        }
        {
            int row, col;
            double val;
            for (col = 0; col < n_covariates; col++) {
                for (row = 0; row < n_obs; row++) {
                    if (SF_mat_el("__trop_covariates", row + 1, col + 1, &val) != 0) {
                        val = 0.0;
                    }
                    x_buf[row + col * n_obs] = val;
                }
            }
        }
    }

    /* Call Rust: exhaustive Cartesian search over the full grid. */
    if (n_covariates > 0) {
        rust_rc = stata_loocv_grid_search_exhaustive_with_covariates(
            y_matrix, d_matrix, control_mask, time_dist,
            n_periods, n_units,
            lambda_time_grid, lambda_time_len,
            lambda_unit_grid, lambda_unit_len,
            lambda_nn_grid, lambda_nn_len,
            max_iter, tol,
            &best_time, &best_unit, &best_nn, &best_score,
            &n_valid, &n_attempted,
            &first_failed_t, &first_failed_i,
            x_buf, n_covariates
        );
    } else {
        rust_rc = stata_loocv_grid_search_exhaustive(
            y_matrix, d_matrix, control_mask, time_dist,
            n_periods, n_units,
            lambda_time_grid, lambda_time_len,
            lambda_unit_grid, lambda_unit_len,
            lambda_nn_grid, lambda_nn_len,
            max_iter, tol,
            &best_time, &best_unit, &best_nn, &best_score,
            &n_valid, &n_attempted,
            &first_failed_t, &first_failed_i
        );
    }

    if (rust_rc != TROP_SUCCESS) {
        translate_error_code(rust_rc);
        rc = rust_rc;
        goto cleanup;
    }

    /* Write results (identical schema to handle_loocv_twostep) */
    SF_scal_save("__trop_lambda_time", best_time);
    SF_scal_save("__trop_lambda_unit", best_unit);
    SF_scal_save("__trop_lambda_nn", best_nn);
    SF_scal_save("__trop_loocv_score", best_score);
    SF_scal_save("__trop_loocv_n_valid", (double)n_valid);
    SF_scal_save("__trop_loocv_n_attempted", (double)n_attempted);
    SF_scal_save("__trop_loocv_first_failed_t", (double)first_failed_t);
    SF_scal_save("__trop_loocv_first_failed_i", (double)first_failed_i);

    TROP_LOG_INFO("LOOCV exhaustive complete: lambda_time=%g, lambda_unit=%g, lambda_nn=%g, score=%g, n_valid=%d, n_attempted=%d, first_failed=(%d,%d)",
                  best_time, best_unit, best_nn, best_score, n_valid, n_attempted, first_failed_t, first_failed_i);

    rc = TROP_SUCCESS;

cleanup:
    free(y_matrix);
    free(d_matrix);
    free(control_mask);
    free(time_dist);
    free(lambda_time_grid);
    free(lambda_unit_grid);
    free(lambda_nn_grid);
    free(x_buf);

    return rc;
}


/* ============================================================================
 * Command Handler: LOOCV Joint
 * ============================================================================ */

static ST_retcode handle_loocv_joint(void) {
    ST_int n_units, n_periods;
    double *y_matrix = NULL;
    double *d_matrix = NULL;
    unsigned char *control_mask = NULL;
    double *lambda_time_grid = NULL;
    double *lambda_unit_grid = NULL;
    double *lambda_nn_grid = NULL;
    int lambda_time_len, lambda_unit_len, lambda_nn_len;
    double max_iter_d, tol;
    int max_iter;
    double best_time, best_unit, best_nn, best_score;
    int n_valid, n_attempted;
    int first_failed_t, first_failed_i;  /* first failed LOOCV observation indices */
    double stage1_time, stage1_unit, stage1_nn;  /* Stage-1 univariate init (Footnote 2) */
    ST_retcode rc;
    int rust_rc;
    
    first_failed_t = -1;
    first_failed_i = -1;
    stage1_time = 0.0;
    stage1_unit = 0.0;
    stage1_nn = 0.0;
    
    TROP_LOG_INFO("starting LOOCV cycling search (joint)");
    
    /* Read dimensions */
    rc = read_dimensions(&n_units, &n_periods);
    if (rc != TROP_SUCCESS) goto cleanup;
    
    /* Allocate memory (no time_dist for joint method) */
    y_matrix = (double *)malloc(n_units * n_periods * sizeof(double));
    d_matrix = (double *)malloc(n_units * n_periods * sizeof(double));
    control_mask = (unsigned char *)malloc(n_units * n_periods);
    
    if (!y_matrix || !d_matrix || !control_mask) {
        TROP_LOG_ERROR("memory allocation failed");
        rc = TROP_ERR_MEMORY;
        goto cleanup;
    }
    
    /* Read data */
    double y_idx_d, d_idx_d, ctrl_idx_d;
    SF_scal_use("__trop_y_varindex", &y_idx_d);
    SF_scal_use("__trop_d_varindex", &d_idx_d);
    SF_scal_use("__trop_ctrl_varindex", &ctrl_idx_d);
    
    rc = read_panel_to_matrix((ST_int)y_idx_d, n_periods, n_units, y_matrix);
    if (rc != TROP_SUCCESS) goto cleanup;
    
    rc = read_panel_to_matrix((ST_int)d_idx_d, n_periods, n_units, d_matrix);
    if (rc != TROP_SUCCESS) goto cleanup;
    
    rc = read_control_mask((ST_int)ctrl_idx_d, n_periods, n_units, control_mask);
    if (rc != TROP_SUCCESS) goto cleanup;
    
    /* Read lambda grids */
    rc = read_lambda_grid("__trop_lambda_time_grid", &lambda_time_grid, &lambda_time_len);
    if (rc != TROP_SUCCESS) goto cleanup;
    
    rc = read_lambda_grid("__trop_lambda_unit_grid", &lambda_unit_grid, &lambda_unit_len);
    if (rc != TROP_SUCCESS) goto cleanup;
    
    rc = read_lambda_grid("__trop_lambda_nn_grid", &lambda_nn_grid, &lambda_nn_len);
    if (rc != TROP_SUCCESS) goto cleanup;
    
    /* Convert infinity sentinel values in grids */
    convert_lambda_infinity(lambda_time_grid, lambda_time_len, "time");
    convert_lambda_infinity(lambda_unit_grid, lambda_unit_len, "unit");
    convert_lambda_infinity(lambda_nn_grid, lambda_nn_len, "nn");
    
    /* Read algorithm parameters */
    SF_scal_use("__trop_max_iter", &max_iter_d);
    SF_scal_use("__trop_tol", &tol);
    
    max_iter = (int)max_iter_d;
    
    /* Call Rust: two-stage coordinate descent over lambda grid.
     * max_cycles=10 controls the number of coordinate descent iterations. */
    rust_rc = stata_loocv_cycling_search_joint(
        y_matrix, d_matrix, control_mask,
        n_periods, n_units,
        lambda_time_grid, lambda_time_len,
        lambda_unit_grid, lambda_unit_len,
        lambda_nn_grid, lambda_nn_len,
        max_iter, tol,
        10, /* max_cycles: coordinate descent iterations */
        &best_time, &best_unit, &best_nn, &best_score,
        &n_valid, &n_attempted,
        &first_failed_t, &first_failed_i,
        &stage1_time, &stage1_unit, &stage1_nn
    );
    
    if (rust_rc != TROP_SUCCESS) {
        translate_error_code(rust_rc);
        rc = rust_rc;
        goto cleanup;
    }
    
    /* Write results */
    SF_scal_save("__trop_lambda_time", best_time);
    SF_scal_save("__trop_lambda_unit", best_unit);
    SF_scal_save("__trop_lambda_nn", best_nn);
    SF_scal_save("__trop_loocv_score", best_score);
    /* Save LOOCV diagnostic scalars */
    SF_scal_save("__trop_loocv_n_valid", (double)n_valid);
    SF_scal_save("__trop_loocv_n_attempted", (double)n_attempted);
    /* First failed observation indices (for diagnostics) */
    SF_scal_save("__trop_loocv_first_failed_t", (double)first_failed_t);
    SF_scal_save("__trop_loocv_first_failed_i", (double)first_failed_i);
    /* Stage-1 univariate init (paper Footnote 2); cycling path only. */
    SF_scal_save("__trop_stage1_lambda_time", stage1_time);
    SF_scal_save("__trop_stage1_lambda_unit", stage1_unit);
    SF_scal_save("__trop_stage1_lambda_nn", stage1_nn);
    
    TROP_LOG_INFO("LOOCV complete: lambda_time=%g, lambda_unit=%g, lambda_nn=%g, score=%g, n_valid=%d, n_attempted=%d, first_failed=(%d,%d), stage1=(%g,%g,%g)",
                  best_time, best_unit, best_nn, best_score, n_valid, n_attempted, first_failed_t, first_failed_i,
                  stage1_time, stage1_unit, stage1_nn);
    
    rc = TROP_SUCCESS;
    
cleanup:
    free(y_matrix);
    free(d_matrix);
    free(control_mask);
    free(lambda_time_grid);
    free(lambda_unit_grid);
    free(lambda_nn_grid);
    
    return rc;
}


/* ============================================================================
 * Command Handler: LOOCV Joint (Exhaustive / Cartesian)
 * ============================================================================ */

/**
 * Exhaustive (Cartesian) grid search variant of handle_loocv_joint.
 *
 * Reads the same Stata scalars/matrices as handle_loocv_joint (no max_cycles
 * needed), then calls stata_loocv_grid_search_joint which enumerates all
 * |grid|^3 triples in parallel.  Writes identical output scalars so the
 * downstream Mata/ADO layers are agnostic to which strategy was used.
 */
static ST_retcode handle_loocv_joint_exhaustive(void) {
    ST_int n_units, n_periods;
    double *y_matrix = NULL;
    double *d_matrix = NULL;
    unsigned char *control_mask = NULL;
    double *lambda_time_grid = NULL;
    double *lambda_unit_grid = NULL;
    double *lambda_nn_grid = NULL;
    int lambda_time_len, lambda_unit_len, lambda_nn_len;
    double max_iter_d, tol;
    int max_iter;
    double best_time, best_unit, best_nn, best_score;
    int n_valid, n_attempted;
    int first_failed_t, first_failed_i;
    ST_retcode rc;
    int rust_rc;
    
    first_failed_t = -1;
    first_failed_i = -1;
    
    TROP_LOG_INFO("starting LOOCV exhaustive grid search (joint)");
    
    /* Read dimensions */
    rc = read_dimensions(&n_units, &n_periods);
    if (rc != TROP_SUCCESS) goto cleanup;
    
    /* Allocate memory (no time_dist for joint method) */
    y_matrix = (double *)malloc(n_units * n_periods * sizeof(double));
    d_matrix = (double *)malloc(n_units * n_periods * sizeof(double));
    control_mask = (unsigned char *)malloc(n_units * n_periods);
    
    if (!y_matrix || !d_matrix || !control_mask) {
        TROP_LOG_ERROR("memory allocation failed");
        rc = TROP_ERR_MEMORY;
        goto cleanup;
    }
    
    /* Read data */
    double y_idx_d, d_idx_d, ctrl_idx_d;
    SF_scal_use("__trop_y_varindex", &y_idx_d);
    SF_scal_use("__trop_d_varindex", &d_idx_d);
    SF_scal_use("__trop_ctrl_varindex", &ctrl_idx_d);
    
    rc = read_panel_to_matrix((ST_int)y_idx_d, n_periods, n_units, y_matrix);
    if (rc != TROP_SUCCESS) goto cleanup;
    
    rc = read_panel_to_matrix((ST_int)d_idx_d, n_periods, n_units, d_matrix);
    if (rc != TROP_SUCCESS) goto cleanup;
    
    rc = read_control_mask((ST_int)ctrl_idx_d, n_periods, n_units, control_mask);
    if (rc != TROP_SUCCESS) goto cleanup;
    
    /* Read lambda grids */
    rc = read_lambda_grid("__trop_lambda_time_grid", &lambda_time_grid, &lambda_time_len);
    if (rc != TROP_SUCCESS) goto cleanup;
    
    rc = read_lambda_grid("__trop_lambda_unit_grid", &lambda_unit_grid, &lambda_unit_len);
    if (rc != TROP_SUCCESS) goto cleanup;
    
    rc = read_lambda_grid("__trop_lambda_nn_grid", &lambda_nn_grid, &lambda_nn_len);
    if (rc != TROP_SUCCESS) goto cleanup;
    
    /* Convert infinity sentinel values in grids */
    convert_lambda_infinity(lambda_time_grid, lambda_time_len, "time");
    convert_lambda_infinity(lambda_unit_grid, lambda_unit_len, "unit");
    convert_lambda_infinity(lambda_nn_grid, lambda_nn_len, "nn");
    
    /* Read algorithm parameters */
    SF_scal_use("__trop_max_iter", &max_iter_d);
    SF_scal_use("__trop_tol", &tol);
    
    max_iter = (int)max_iter_d;
    
    /* Call Rust: exhaustive Cartesian search over the full grid. */
    rust_rc = stata_loocv_grid_search_joint(
        y_matrix, d_matrix, control_mask,
        n_periods, n_units,
        lambda_time_grid, lambda_time_len,
        lambda_unit_grid, lambda_unit_len,
        lambda_nn_grid, lambda_nn_len,
        max_iter, tol,
        &best_time, &best_unit, &best_nn, &best_score,
        &n_valid, &n_attempted,
        &first_failed_t, &first_failed_i
    );
    
    if (rust_rc != TROP_SUCCESS) {
        translate_error_code(rust_rc);
        rc = rust_rc;
        goto cleanup;
    }
    
    /* Write results (identical schema to handle_loocv_joint) */
    SF_scal_save("__trop_lambda_time", best_time);
    SF_scal_save("__trop_lambda_unit", best_unit);
    SF_scal_save("__trop_lambda_nn", best_nn);
    SF_scal_save("__trop_loocv_score", best_score);
    SF_scal_save("__trop_loocv_n_valid", (double)n_valid);
    SF_scal_save("__trop_loocv_n_attempted", (double)n_attempted);
    SF_scal_save("__trop_loocv_first_failed_t", (double)first_failed_t);
    SF_scal_save("__trop_loocv_first_failed_i", (double)first_failed_i);
    
    TROP_LOG_INFO("LOOCV (exhaustive) complete: lambda_time=%g, lambda_unit=%g, lambda_nn=%g, score=%g, n_valid=%d, n_attempted=%d, first_failed=(%d,%d)",
                  best_time, best_unit, best_nn, best_score, n_valid, n_attempted, first_failed_t, first_failed_i);
    
    rc = TROP_SUCCESS;
    
cleanup:
    free(y_matrix);
    free(d_matrix);
    free(control_mask);
    free(lambda_time_grid);
    free(lambda_unit_grid);
    free(lambda_nn_grid);
    
    return rc;
}


/* ============================================================================
 * Command Handler: Estimate Twostep
 * ============================================================================ */

static ST_retcode handle_estimate_twostep(void) {
    ST_int n_units, n_periods;
    double *y_matrix = NULL;
    double *d_matrix = NULL;
    unsigned char *control_mask = NULL;
    int64_t *time_dist = NULL;
    double *tau = NULL;
    double *alpha = NULL;
    double *beta = NULL;
    double *l_matrix = NULL;
    int *converged_by_obs = NULL;
    int *n_iters_by_obs = NULL;
    double *unit_weights = NULL;
    double *x_buf = NULL;
    double *gamma_buf = NULL;
    int unit_weights_len = 0;
    int use_weights = 0;
    int n_covariates = 0;
    double use_weights_d = 0.0;
    double lambda_time, lambda_unit, lambda_nn;
    double max_iter_d, tol;
    int max_iter;
    double att;
    int n_treated, n_iterations, converged;
    ST_retcode rc;
    int rust_rc;
    
    TROP_LOG_INFO("starting estimation (twostep)");
    
    /* Read dimensions */
    rc = read_dimensions(&n_units, &n_periods);
    if (rc != TROP_SUCCESS) goto cleanup;
    
    /* Allocate input memory */
    y_matrix = (double *)malloc(n_units * n_periods * sizeof(double));
    d_matrix = (double *)malloc(n_units * n_periods * sizeof(double));
    control_mask = (unsigned char *)malloc(n_units * n_periods);
    time_dist = (int64_t *)malloc(n_periods * n_periods * sizeof(int64_t));
    
    /* Allocate output memory (max possible sizes) */
    tau = (double *)malloc(n_units * n_periods * sizeof(double));  /* Max treated */
    alpha = (double *)malloc(n_units * sizeof(double));
    beta = (double *)malloc(n_periods * sizeof(double));
    l_matrix = (double *)malloc(n_units * n_periods * sizeof(double));
    /* Per-obs diagnostics: sized at the N*T upper bound on treated cells. */
    converged_by_obs = (int *)malloc((size_t)n_units * (size_t)n_periods * sizeof(int));
    n_iters_by_obs   = (int *)malloc((size_t)n_units * (size_t)n_periods * sizeof(int));
    
    if (!y_matrix || !d_matrix || !control_mask || !time_dist ||
        !tau || !alpha || !beta || !l_matrix ||
        !converged_by_obs || !n_iters_by_obs) {
        TROP_LOG_ERROR("memory allocation failed");
        rc = TROP_ERR_MEMORY;
        goto cleanup;
    }
    
    /* Read data */
    double y_idx_d, d_idx_d, ctrl_idx_d;
    SF_scal_use("__trop_y_varindex", &y_idx_d);
    SF_scal_use("__trop_d_varindex", &d_idx_d);
    SF_scal_use("__trop_ctrl_varindex", &ctrl_idx_d);
    
    rc = read_panel_to_matrix((ST_int)y_idx_d, n_periods, n_units, y_matrix);
    if (rc != TROP_SUCCESS) goto cleanup;
    
    rc = read_panel_to_matrix((ST_int)d_idx_d, n_periods, n_units, d_matrix);
    if (rc != TROP_SUCCESS) goto cleanup;
    
    rc = read_control_mask((ST_int)ctrl_idx_d, n_periods, n_units, control_mask);
    if (rc != TROP_SUCCESS) goto cleanup;
    
    rc = read_time_dist_matrix("__trop_time_dist", n_periods, time_dist);
    if (rc != TROP_SUCCESS) goto cleanup;
    
    /* Read lambda parameters */
    SF_scal_use("__trop_lambda_time", &lambda_time);
    SF_scal_use("__trop_lambda_unit", &lambda_unit);
    SF_scal_use("__trop_lambda_nn", &lambda_nn);
    SF_scal_use("__trop_max_iter", &max_iter_d);
    SF_scal_use("__trop_tol", &tol);
    
    max_iter = (int)max_iter_d;
    
    TROP_LOG_DEBUG("estimate params: lambda_time=%g, lambda_unit=%g, lambda_nn=%g",
                   lambda_time, lambda_unit, lambda_nn);

    /* Optional pweight path.  __trop_use_weights == 1 triggers the weighted
     * ATT aggregation; __trop_unit_weights is a N×1 column matrix with
     * strictly positive per-unit pweights (validated Mata-side). */
    if (SF_scal_use("__trop_use_weights", &use_weights_d) == 0) {
        use_weights = ((int)use_weights_d != 0) ? 1 : 0;
    }
    if (use_weights) {
        rc = read_lambda_grid("__trop_unit_weights", &unit_weights, &unit_weights_len);
        if (rc != TROP_SUCCESS) goto cleanup;
        if (unit_weights_len != (int)n_units) {
            TROP_LOG_ERROR("unit_weights length %d != n_units %d",
                           unit_weights_len, (int)n_units);
            rc = TROP_ERR_INVALID_DIM;
            goto cleanup;
        }
    }

    /* --- Covariate support --- */
    {
        double nc_val = 0;
        if (SF_scal_use("__trop_n_covariates", &nc_val) == 0 && nc_val > 0) {
            n_covariates = (int)nc_val;
        }
    }
    if (n_covariates > 0) {
        int n_obs = (int)n_periods * (int)n_units;
        x_buf = (double *)malloc((size_t)n_obs * (size_t)n_covariates * sizeof(double));
        gamma_buf = (double *)calloc((size_t)n_covariates, sizeof(double));
        if (!x_buf || !gamma_buf) {
            TROP_LOG_ERROR("covariate memory allocation failed");
            rc = TROP_ERR_MEMORY;
            goto cleanup;
        }
        {
            int row, col;
            double val;
            for (col = 0; col < n_covariates; col++) {
                for (row = 0; row < n_obs; row++) {
                    if (SF_mat_el("__trop_covariates", row + 1, col + 1, &val) != 0) {
                        val = 0.0;
                    }
                    x_buf[row + col * n_obs] = val;
                }
            }
        }
    }

    /* Call Rust function (weighted, covariate, or plain) */
    if (use_weights) {
        rust_rc = stata_estimate_twostep_weighted(
            y_matrix, d_matrix, control_mask, time_dist,
            n_periods, n_units,
            lambda_time, lambda_unit, lambda_nn,
            max_iter, tol,
            &att, tau, alpha, beta, l_matrix,
            &n_treated, &n_iterations, &converged,
            converged_by_obs, n_iters_by_obs,
            unit_weights
        );
    } else if (n_covariates > 0) {
        rust_rc = stata_estimate_twostep_with_covariates(
            y_matrix, d_matrix, control_mask, time_dist,
            n_periods, n_units,
            lambda_time, lambda_unit, lambda_nn,
            max_iter, tol,
            &att, tau, alpha, beta, l_matrix,
            &n_treated, &n_iterations, &converged,
            converged_by_obs, n_iters_by_obs,
            x_buf, n_covariates, gamma_buf
        );
    } else {
        rust_rc = stata_estimate_twostep(
            y_matrix, d_matrix, control_mask, time_dist,
            n_periods, n_units,
            lambda_time, lambda_unit, lambda_nn,
            max_iter, tol,
            &att, tau, alpha, beta, l_matrix,
            &n_treated, &n_iterations, &converged,
            converged_by_obs, n_iters_by_obs
        );
    }
    
    if (rust_rc != TROP_SUCCESS) {
        translate_error_code(rust_rc);
        rc = rust_rc;
        goto cleanup;
    }
    
    /* Write results to Stata */
    SF_scal_save("__trop_att", att);
    SF_scal_save("__trop_n_treated", (double)n_treated);
    /* n_iterations: maximum iterations across all treated observations */
    SF_scal_save("__trop_n_iterations", (double)n_iterations);
    SF_scal_save("__trop_converged", (double)converged);

    /* Per-obs diagnostics (N_treated × 1) — always written so Mata can
     * decide whether to surface a message.  Converted to double for Stata. */
    if (n_treated > 0) {
        double *tmp = (double *)malloc((size_t)n_treated * sizeof(double));
        if (tmp != NULL) {
            int k;
            for (k = 0; k < n_treated; k++) tmp[k] = (double)converged_by_obs[k];
            rc = write_vector_to_matrix("__trop_converged_by_obs", tmp, n_treated, 0);
            if (rc == TROP_SUCCESS) {
                for (k = 0; k < n_treated; k++) tmp[k] = (double)n_iters_by_obs[k];
                rc = write_vector_to_matrix("__trop_n_iters_by_obs", tmp, n_treated, 0);
            }
            free(tmp);
            if (rc != TROP_SUCCESS) goto cleanup;
        }
    }
    
    /* Count ever-treated units vs treated observations (unit-period pairs).
     * __trop_n_treated      = treated observations (for degrees of freedom).
     * __trop_n_treated_units = ever-treated units (for reporting). */
    {
        int n_treated_units = 0;
        int n_treated_total = 0;
        ST_int ii, tt;
        for (ii = 0; ii < n_units; ii++) {
            int unit_treated = 0;
            for (tt = 0; tt < n_periods; tt++) {
                if (d_matrix[ii * n_periods + tt] == 1.0) {
                    n_treated_total++;
                    unit_treated = 1;
                }
            }
            if (unit_treated) n_treated_units++;
        }
        SF_scal_save("__trop_n_treated_units", (double)n_treated_units);
        /* Per-observation diagnostics for twostep:
         * n_obs_estimated = successfully estimated treated observations.
         * n_obs_failed    = treated observations that failed estimation. */
        SF_scal_save("__trop_n_obs_estimated", (double)n_treated);
        SF_scal_save("__trop_n_obs_failed", (double)(n_treated_total - n_treated));
    }
    
    /* Write tau vector to matrix */
    rc = write_vector_to_matrix("__trop_tau", tau, n_treated, 0);
    if (rc != TROP_SUCCESS) goto cleanup;
    
    /* Write alpha (unit fixed effects) */
    rc = write_vector_to_matrix("__trop_alpha", alpha, n_units, 0);
    if (rc != TROP_SUCCESS) goto cleanup;
    
    /* Write beta (time fixed effects) */
    rc = write_vector_to_matrix("__trop_beta", beta, n_periods, 0);
    if (rc != TROP_SUCCESS) goto cleanup;
    
    /* Write L matrix (low-rank factor matrix) */
    rc = write_matrix_to_stata("__trop_factor_matrix", l_matrix, n_periods, n_units);
    if (rc != TROP_SUCCESS) goto cleanup;

    /* Write gamma (covariate coefficients) if estimated.
       __trop_gamma is pre-allocated as 1 x p (row vector, Stata convention).
       Store into row=1, col=j+1. */
    if (n_covariates > 0 && gamma_buf != NULL) {
        int j;
        for (j = 0; j < n_covariates; j++) {
            SF_mat_store("__trop_gamma", 1, j + 1, gamma_buf[j]);
        }
    }
    
    TROP_LOG_INFO("estimation complete: ATT=%g, n_treated=%d, converged=%d",
                  att, n_treated, converged);
    
    /* Compute and store weight vectors for post-estimation diagnostics.
     * For twostep, weights vary per treated observation; compute for the
     * first treated observation as a representative example. */
    {
        double *theta_vec = NULL;
        double *omega_vec = NULL;
        int first_target_unit = -1;
        int first_target_period = -1;
        ST_int t_idx, i_idx;
        
        /* Find first treated observation from D matrix */
        for (t_idx = 0; t_idx < n_periods && first_target_unit < 0; t_idx++) {
            for (i_idx = 0; i_idx < n_units; i_idx++) {
                if (d_matrix[i_idx * n_periods + t_idx] == 1.0) {
                    first_target_unit = (int)i_idx;
                    first_target_period = (int)t_idx;
                    break;
                }
            }
        }
        
        if (first_target_unit >= 0) {
            theta_vec = (double *)malloc(n_periods * sizeof(double));
            omega_vec = (double *)malloc(n_units * sizeof(double));
            
            if (theta_vec && omega_vec) {
                rust_rc = stata_compute_twostep_weight_vectors(
                    y_matrix, d_matrix, time_dist,
                    n_periods, n_units,
                    first_target_unit, first_target_period,
                    lambda_time, lambda_unit,
                    theta_vec, omega_vec
                );
                
                if (rust_rc == TROP_SUCCESS) {
                    rc = write_vector_to_matrix("__trop_theta", theta_vec, n_periods, 0);
                    if (rc == TROP_SUCCESS) {
                        rc = write_vector_to_matrix("__trop_omega", omega_vec, n_units, 0);
                    }
                    if (rc != TROP_SUCCESS) {
                        TROP_LOG_DEBUG("warning: failed to write weight vectors (non-fatal)");
                        rc = TROP_SUCCESS;  /* Non-fatal: estimation succeeded */
                    }
                } else {
                    TROP_LOG_DEBUG("warning: weight vector computation failed (non-fatal)");
                }
            }
            
            free(theta_vec);
            free(omega_vec);
        }
    }
    
    rc = TROP_SUCCESS;
    
cleanup:
    free(y_matrix);
    free(d_matrix);
    free(control_mask);
    free(time_dist);
    free(tau);
    free(alpha);
    free(beta);
    free(l_matrix);
    free(converged_by_obs);
    free(n_iters_by_obs);
    free(unit_weights);
    free(x_buf);
    free(gamma_buf);
    
    return rc;
}

/* ============================================================================
 * Command Handler: Estimate Joint
 * ============================================================================ */

static ST_retcode handle_estimate_joint(void) {
    ST_int n_units, n_periods;
    double *y_matrix = NULL;
    double *d_matrix = NULL;
    double *alpha = NULL;
    double *beta = NULL;
    double *l_matrix = NULL;
    double *tau_vec = NULL;
    double *unit_weights = NULL;
    double *x_buf = NULL;
    double *gamma_buf = NULL;
    int unit_weights_len = 0;
    int use_weights = 0;
    int n_covariates = 0;
    double use_weights_d = 0.0;
    double lambda_time, lambda_unit, lambda_nn;
    double max_iter_d, tol;
    int max_iter;
    double tau, mu;
    int n_iterations, converged;
    int n_treated_cells = 0;
    ST_retcode rc;
    int rust_rc;
    
    TROP_LOG_INFO("starting estimation (joint)");
    
    /* Read dimensions */
    rc = read_dimensions(&n_units, &n_periods);
    if (rc != TROP_SUCCESS) goto cleanup;
    
    /* Allocate memory (no control_mask or time_dist for joint) */
    y_matrix = (double *)malloc(n_units * n_periods * sizeof(double));
    d_matrix = (double *)malloc(n_units * n_periods * sizeof(double));
    alpha = (double *)malloc(n_units * sizeof(double));
    beta = (double *)malloc(n_periods * sizeof(double));
    l_matrix = (double *)malloc(n_units * n_periods * sizeof(double));
    /* Upper bound for treated cells: N*T; caller writes ≤ n_treated_cells. */
    tau_vec = (double *)malloc((size_t)n_units * (size_t)n_periods * sizeof(double));
    
    if (!y_matrix || !d_matrix || !alpha || !beta || !l_matrix || !tau_vec) {
        TROP_LOG_ERROR("memory allocation failed");
        rc = TROP_ERR_MEMORY;
        goto cleanup;
    }
    
    /* Read data */
    double y_idx_d, d_idx_d;
    SF_scal_use("__trop_y_varindex", &y_idx_d);
    SF_scal_use("__trop_d_varindex", &d_idx_d);
    
    rc = read_panel_to_matrix((ST_int)y_idx_d, n_periods, n_units, y_matrix);
    if (rc != TROP_SUCCESS) goto cleanup;
    
    rc = read_panel_to_matrix((ST_int)d_idx_d, n_periods, n_units, d_matrix);
    if (rc != TROP_SUCCESS) goto cleanup;
    
    /* Read lambda parameters */
    SF_scal_use("__trop_lambda_time", &lambda_time);
    SF_scal_use("__trop_lambda_unit", &lambda_unit);
    SF_scal_use("__trop_lambda_nn", &lambda_nn);
    SF_scal_use("__trop_max_iter", &max_iter_d);
    SF_scal_use("__trop_tol", &tol);
    
    max_iter = (int)max_iter_d;

    /* Optional pweight path — see handle_estimate_twostep for protocol. */
    if (SF_scal_use("__trop_use_weights", &use_weights_d) == 0) {
        use_weights = ((int)use_weights_d != 0) ? 1 : 0;
    }
    if (use_weights) {
        rc = read_lambda_grid("__trop_unit_weights", &unit_weights, &unit_weights_len);
        if (rc != TROP_SUCCESS) goto cleanup;
        if (unit_weights_len != (int)n_units) {
            TROP_LOG_ERROR("unit_weights length %d != n_units %d",
                           unit_weights_len, (int)n_units);
            rc = TROP_ERR_INVALID_DIM;
            goto cleanup;
        }
    }

    /* --- Covariate support --- */
    {
        double nc_val = 0;
        if (SF_scal_use("__trop_n_covariates", &nc_val) == 0 && nc_val > 0) {
            n_covariates = (int)nc_val;
        }
    }
    if (n_covariates > 0) {
        int n_obs = (int)n_periods * (int)n_units;
        x_buf = (double *)malloc((size_t)n_obs * (size_t)n_covariates * sizeof(double));
        gamma_buf = (double *)calloc((size_t)n_covariates, sizeof(double));
        if (!x_buf || !gamma_buf) {
            TROP_LOG_ERROR("covariate memory allocation failed");
            rc = TROP_ERR_MEMORY;
            goto cleanup;
        }
        {
            int row, col;
            double val;
            for (col = 0; col < n_covariates; col++) {
                for (row = 0; row < n_obs; row++) {
                    if (SF_mat_el("__trop_covariates", row + 1, col + 1, &val) != 0) {
                        val = 0.0;
                    }
                    x_buf[row + col * n_obs] = val;
                }
            }
        }
    }

    /* Call Rust function (weighted, covariate, or plain) */
    if (use_weights) {
        rust_rc = stata_estimate_joint_weighted(
            y_matrix, d_matrix,
            n_periods, n_units,
            lambda_time, lambda_unit, lambda_nn,
            max_iter, tol,
            &tau, &mu, alpha, beta, l_matrix,
            &n_iterations, &converged,
            tau_vec, &n_treated_cells,
            unit_weights
        );
    } else if (n_covariates > 0) {
        rust_rc = stata_estimate_joint_with_covariates(
            y_matrix, d_matrix,
            n_periods, n_units,
            lambda_time, lambda_unit, lambda_nn,
            max_iter, tol,
            &tau, &mu, alpha, beta, l_matrix,
            &n_iterations, &converged,
            tau_vec, &n_treated_cells,
            x_buf, n_covariates, gamma_buf
        );
    } else {
        rust_rc = stata_estimate_joint(
            y_matrix, d_matrix,
            n_periods, n_units,
            lambda_time, lambda_unit, lambda_nn,
            max_iter, tol,
            &tau, &mu, alpha, beta, l_matrix,
            &n_iterations, &converged,
            tau_vec, &n_treated_cells
        );
    }
    
    if (rust_rc != TROP_SUCCESS) {
        translate_error_code(rust_rc);
        rc = rust_rc;
        goto cleanup;
    }
    
    /* n_treated_cells was written by Rust; fall back to a scan in the unlikely
     * event it is non-positive (e.g. null pointer path). */
    int n_treated_obs = n_treated_cells;
    if (n_treated_obs <= 0) {
        for (ST_int i = 0; i < n_units; i++) {
            for (ST_int t = 0; t < n_periods; t++) {
                if (d_matrix[i * n_periods + t] == 1.0) {
                    n_treated_obs++;
                }
            }
        }
    }
    
    /* Write results.
     *
     * Paper Eq 13 defines τ per treated (i,t) cell; Eq 1 aggregates them to
     * ATT = mean(τ_it).  __trop_att carries the scalar; __trop_tau becomes a
     * N_treated × 1 matrix so Mata-side consumers (e(tau)) see the same
     * representation as for method("twostep"). */
    SF_scal_save("__trop_att", tau);
    SF_scal_save("__trop_mu", mu);
    SF_scal_save("__trop_n_treated", (double)n_treated_obs);
    SF_scal_save("__trop_n_iterations", (double)n_iterations);
    SF_scal_save("__trop_converged", (double)converged);
    
    /* Write per-cell τ vector as a Stata matrix (N_treated × 1). */
    if (n_treated_obs > 0) {
        rc = write_vector_to_matrix("__trop_tau", tau_vec, n_treated_obs, 0);
        if (rc != TROP_SUCCESS) goto cleanup;
    }
    
    /* Count ever-treated units (units with at least one D==1 cell). */
    {
        int n_treated_units = 0;
        ST_int ii, tt;
        for (ii = 0; ii < n_units; ii++) {
            for (tt = 0; tt < n_periods; tt++) {
                if (d_matrix[ii * n_periods + tt] == 1.0) {
                    n_treated_units++;
                    break;
                }
            }
        }
        SF_scal_save("__trop_n_treated_units", (double)n_treated_units);
    }
    
    rc = write_vector_to_matrix("__trop_alpha", alpha, n_units, 0);
    if (rc != TROP_SUCCESS) goto cleanup;
    
    rc = write_vector_to_matrix("__trop_beta", beta, n_periods, 0);
    if (rc != TROP_SUCCESS) goto cleanup;
    
    rc = write_matrix_to_stata("__trop_factor_matrix", l_matrix, n_periods, n_units);
    if (rc != TROP_SUCCESS) goto cleanup;

    /* Write gamma (covariate coefficients) if estimated.
       __trop_gamma is pre-allocated as 1 x p (row vector, Stata convention).
       Store into row=1, col=j+1. */
    if (n_covariates > 0 && gamma_buf != NULL) {
        int j;
        for (j = 0; j < n_covariates; j++) {
            SF_mat_store("__trop_gamma", 1, j + 1, gamma_buf[j]);
        }
    }
    
    TROP_LOG_INFO("estimation complete: tau=%g, mu=%g, converged=%d",
                  tau, mu, converged);
    
    /* Compute and store global weight vectors for post-estimation diagnostics.
     * Joint weights are global (not per-observation): delta_time (T×1) and
     * delta_unit (N×1). */
    {
        double *delta_time_vec = NULL;
        double *delta_unit_vec = NULL;
        
        delta_time_vec = (double *)malloc(n_periods * sizeof(double));
        delta_unit_vec = (double *)malloc(n_units * sizeof(double));
        
        if (delta_time_vec && delta_unit_vec) {
            rust_rc = stata_compute_joint_weight_vectors(
                y_matrix, d_matrix,
                n_periods, n_units,
                lambda_time, lambda_unit,
                delta_time_vec, delta_unit_vec
            );
            
            if (rust_rc == TROP_SUCCESS) {
                rc = write_vector_to_matrix("__trop_delta_time", delta_time_vec, n_periods, 0);
                if (rc == TROP_SUCCESS) {
                    rc = write_vector_to_matrix("__trop_delta_unit", delta_unit_vec, n_units, 0);
                }
                if (rc != TROP_SUCCESS) {
                    TROP_LOG_DEBUG("warning: failed to write weight vectors (non-fatal)");
                    rc = TROP_SUCCESS;  /* Non-fatal: estimation succeeded */
                }
            } else {
                TROP_LOG_DEBUG("warning: weight vector computation failed (non-fatal)");
            }
        }
        
        free(delta_time_vec);
        free(delta_unit_vec);
    }
    
    rc = TROP_SUCCESS;
    
cleanup:
    free(y_matrix);
    free(d_matrix);
    free(alpha);
    free(beta);
    free(l_matrix);
    free(tau_vec);
    free(unit_weights);
    
    return rc;
}


/* ============================================================================
 * Command Handler: Bootstrap Twostep
 * ============================================================================ */

static ST_retcode handle_bootstrap_twostep(void) {
    ST_int n_units, n_periods;
    double *y_matrix = NULL;
    double *d_matrix = NULL;
    unsigned char *control_mask = NULL;
    int64_t *time_dist = NULL;
    double *estimates = NULL;
    double *unit_weights = NULL;
    double *x_buf = NULL;
    int unit_weights_len = 0;
    int use_weights = 0;
    int n_covariates = 0;
    double use_weights_d = 0.0;
    double lambda_time, lambda_unit, lambda_nn;
    double n_bootstrap_d, max_iter_d, tol, seed_d, alpha_d, ddof_d;
    int n_bootstrap, max_iter, ddof;
    uint64_t seed;
    double se, alpha;
    double ci_lower_pct, ci_upper_pct;  /* percentile CI from bootstrap distribution */
    int n_valid;
    ST_retcode rc;
    int rust_rc;
    
    TROP_LOG_INFO("starting bootstrap variance estimation (twostep)");
    
    /* Read dimensions */
    rc = read_dimensions(&n_units, &n_periods);
    if (rc != TROP_SUCCESS) goto cleanup;
    
    /* Read bootstrap parameters */
    SF_scal_use("__trop_n_bootstrap", &n_bootstrap_d);
    n_bootstrap = (int)n_bootstrap_d;
    
    /* Read significance level; __trop_bs_alpha is used to avoid name
     * collision with the __trop_alpha matrix (unit fixed effects). */
    if (SF_scal_use("__trop_bs_alpha", &alpha_d) != 0) {
        alpha = 0.05;  /* Default to 95% CI */
    } else {
        alpha = alpha_d;
    }

    /* Variance-denominator selector: 1 = sample (1/(B-1)), 0 = paper Alg 3
     * population (1/B).  Absent scalar defaults to 1 for backward compat. */
    if (SF_scal_use("__trop_bs_ddof", &ddof_d) != 0) {
        ddof = 1;
    } else {
        ddof = ((int)ddof_d == 0) ? 0 : 1;
    }
    
    /* Allocate memory */
    y_matrix = (double *)malloc(n_units * n_periods * sizeof(double));
    d_matrix = (double *)malloc(n_units * n_periods * sizeof(double));
    control_mask = (unsigned char *)malloc(n_units * n_periods);
    time_dist = (int64_t *)malloc(n_periods * n_periods * sizeof(int64_t));
    estimates = (double *)malloc(n_bootstrap * sizeof(double));
    
    if (!y_matrix || !d_matrix || !control_mask || !time_dist || !estimates) {
        TROP_LOG_ERROR("memory allocation failed");
        rc = TROP_ERR_MEMORY;
        goto cleanup;
    }
    
    /* Read data */
    double y_idx_d, d_idx_d, ctrl_idx_d;
    SF_scal_use("__trop_y_varindex", &y_idx_d);
    SF_scal_use("__trop_d_varindex", &d_idx_d);
    SF_scal_use("__trop_ctrl_varindex", &ctrl_idx_d);
    
    rc = read_panel_to_matrix((ST_int)y_idx_d, n_periods, n_units, y_matrix);
    if (rc != TROP_SUCCESS) goto cleanup;
    
    rc = read_panel_to_matrix((ST_int)d_idx_d, n_periods, n_units, d_matrix);
    if (rc != TROP_SUCCESS) goto cleanup;
    
    rc = read_control_mask((ST_int)ctrl_idx_d, n_periods, n_units, control_mask);
    if (rc != TROP_SUCCESS) goto cleanup;
    
    rc = read_time_dist_matrix("__trop_time_dist", n_periods, time_dist);
    if (rc != TROP_SUCCESS) goto cleanup;
    
    /* Read parameters */
    SF_scal_use("__trop_lambda_time", &lambda_time);
    SF_scal_use("__trop_lambda_unit", &lambda_unit);
    SF_scal_use("__trop_lambda_nn", &lambda_nn);
    SF_scal_use("__trop_max_iter", &max_iter_d);
    SF_scal_use("__trop_tol", &tol);
    SF_scal_use("__trop_seed", &seed_d);
    
    max_iter = (int)max_iter_d;
    seed = (uint64_t)seed_d;
    
    TROP_LOG_DEBUG("bootstrap params: n_bootstrap=%d, seed=%llu, alpha=%g, ddof=%d",
                   n_bootstrap, (unsigned long long)seed, alpha, ddof);

    /* Optional pweight path — see handle_estimate_twostep for protocol. */
    if (SF_scal_use("__trop_use_weights", &use_weights_d) == 0) {
        use_weights = ((int)use_weights_d != 0) ? 1 : 0;
    }
    if (use_weights) {
        rc = read_lambda_grid("__trop_unit_weights", &unit_weights, &unit_weights_len);
        if (rc != TROP_SUCCESS) goto cleanup;
        if (unit_weights_len != (int)n_units) {
            TROP_LOG_ERROR("unit_weights length %d != n_units %d",
                           unit_weights_len, (int)n_units);
            rc = TROP_ERR_INVALID_DIM;
            goto cleanup;
        }
    }

    /* --- Covariate support --- */
    {
        double nc_val = 0;
        if (SF_scal_use("__trop_n_covariates", &nc_val) == 0 && nc_val > 0) {
            n_covariates = (int)nc_val;
        }
    }
    if (n_covariates > 0) {
        int n_obs = (int)n_periods * (int)n_units;
        x_buf = (double *)malloc((size_t)n_obs * (size_t)n_covariates * sizeof(double));
        if (!x_buf) {
            TROP_LOG_ERROR("covariate memory allocation failed");
            rc = TROP_ERR_MEMORY;
            goto cleanup;
        }
        {
            int row, col;
            double val;
            for (col = 0; col < n_covariates; col++) {
                for (row = 0; row < n_obs; row++) {
                    if (SF_mat_el("__trop_covariates", row + 1, col + 1, &val) != 0) {
                        val = 0.0;
                    }
                    x_buf[row + col * n_obs] = val;
                }
            }
        }
    }

    /* Call Rust bootstrap function (weighted, covariate, or plain) */
    if (use_weights) {
        rust_rc = stata_bootstrap_trop_variance_weighted(
            y_matrix, d_matrix, control_mask, time_dist,
            n_periods, n_units,
            lambda_time, lambda_unit, lambda_nn,
            n_bootstrap, max_iter, tol, seed, alpha, ddof,
            estimates, &se, &ci_lower_pct, &ci_upper_pct, &n_valid,
            unit_weights
        );
    } else if (n_covariates > 0) {
        rust_rc = stata_bootstrap_trop_variance_with_covariates(
            y_matrix, d_matrix, control_mask, time_dist,
            n_periods, n_units,
            lambda_time, lambda_unit, lambda_nn,
            n_bootstrap, max_iter, tol, seed, alpha, ddof,
            estimates, &se, &ci_lower_pct, &ci_upper_pct, &n_valid,
            x_buf, n_covariates
        );
    } else {
        rust_rc = stata_bootstrap_trop_variance(
            y_matrix, d_matrix, control_mask, time_dist,
            n_periods, n_units,
            lambda_time, lambda_unit, lambda_nn,
            n_bootstrap, max_iter, tol, seed, alpha, ddof,
            estimates, &se, &ci_lower_pct, &ci_upper_pct, &n_valid
        );
    }
    
    if (rust_rc != TROP_SUCCESS) {
        translate_error_code(rust_rc);
        rc = rust_rc;
        goto cleanup;
    }
    
    /* Write results */
    SF_scal_save("__trop_se", se);
    SF_scal_save("__trop_n_bootstrap_valid", (double)n_valid);
    SF_scal_save("__trop_level", 1.0 - alpha);
    
    /* Save percentile CI as diagnostics.  The authoritative CI is computed
     * by Mata using the t-distribution; the percentile CI is useful for
     * assessing bootstrap distribution asymmetry. */
    SF_scal_save("__trop_ci_lower_percentile", ci_lower_pct);
    SF_scal_save("__trop_ci_upper_percentile", ci_upper_pct);
    
    /* Write bootstrap estimates to matrix (only valid estimates) */
    rc = write_vector_to_matrix("__trop_bootstrap_estimates", estimates, n_valid, 0);
    if (rc != TROP_SUCCESS) goto cleanup;
    
    /* Check failure rate and warn if high */
    double failure_rate = 1.0 - (double)n_valid / (double)n_bootstrap;
    if (failure_rate > 0.1) {
        char warn_msg[256];
        snprintf(warn_msg, sizeof(warn_msg), 
                 "warning: %.1f%% of bootstrap iterations failed (%d/%d valid)\n",
                 failure_rate * 100.0, n_valid, n_bootstrap);
        SF_display(warn_msg);
    }
    
    TROP_LOG_INFO("bootstrap complete: SE=%g, n_valid=%d/%d",
                  se, n_valid, n_bootstrap);
    
    rc = TROP_SUCCESS;
    
cleanup:
    free(y_matrix);
    free(d_matrix);
    free(control_mask);
    free(time_dist);
    free(estimates);
    free(unit_weights);
    free(x_buf);
    
    return rc;
}

/* ============================================================================
 * Command Handler: Bootstrap Joint
 * ============================================================================ */

static ST_retcode handle_bootstrap_joint(void) {
    ST_int n_units, n_periods;
    double *y_matrix = NULL;
    double *d_matrix = NULL;
    double *estimates = NULL;
    double *unit_weights = NULL;
    double *x_buf = NULL;
    int n_covariates = 0;
    int unit_weights_len = 0;
    int use_weights = 0;
    double use_weights_d = 0.0;
    double lambda_time, lambda_unit, lambda_nn;
    double n_bootstrap_d, max_iter_d, tol, seed_d, alpha_d, ddof_d;
    int n_bootstrap, max_iter, ddof;
    uint64_t seed;
    double se, alpha;
    double ci_lower_pct, ci_upper_pct;  /* percentile CI from bootstrap distribution */
    int n_valid;
    ST_retcode rc;
    int rust_rc;
    
    TROP_LOG_INFO("starting bootstrap variance estimation (joint)");
    
    /* Read dimensions */
    rc = read_dimensions(&n_units, &n_periods);
    if (rc != TROP_SUCCESS) goto cleanup;
    
    /* Read bootstrap parameters */
    SF_scal_use("__trop_n_bootstrap", &n_bootstrap_d);
    n_bootstrap = (int)n_bootstrap_d;
    
    /* Read significance level; __trop_bs_alpha is used to avoid name
     * collision with the __trop_alpha matrix (unit fixed effects). */
    if (SF_scal_use("__trop_bs_alpha", &alpha_d) != 0) {
        alpha = 0.05;  /* Default to 95% CI */
    } else {
        alpha = alpha_d;
    }

    /* Variance-denominator selector: 1 = sample (1/(B-1)), 0 = paper Alg 3
     * population (1/B).  Absent scalar defaults to 1 for backward compat. */
    if (SF_scal_use("__trop_bs_ddof", &ddof_d) != 0) {
        ddof = 1;
    } else {
        ddof = ((int)ddof_d == 0) ? 0 : 1;
    }
    
    /* Allocate memory */
    y_matrix = (double *)malloc(n_units * n_periods * sizeof(double));
    d_matrix = (double *)malloc(n_units * n_periods * sizeof(double));
    estimates = (double *)malloc(n_bootstrap * sizeof(double));
    
    if (!y_matrix || !d_matrix || !estimates) {
        TROP_LOG_ERROR("memory allocation failed");
        rc = TROP_ERR_MEMORY;
        goto cleanup;
    }
    
    /* Read data */
    double y_idx_d, d_idx_d;
    SF_scal_use("__trop_y_varindex", &y_idx_d);
    SF_scal_use("__trop_d_varindex", &d_idx_d);
    
    rc = read_panel_to_matrix((ST_int)y_idx_d, n_periods, n_units, y_matrix);
    if (rc != TROP_SUCCESS) goto cleanup;
    
    rc = read_panel_to_matrix((ST_int)d_idx_d, n_periods, n_units, d_matrix);
    if (rc != TROP_SUCCESS) goto cleanup;
    
    /* Read parameters */
    SF_scal_use("__trop_lambda_time", &lambda_time);
    SF_scal_use("__trop_lambda_unit", &lambda_unit);
    SF_scal_use("__trop_lambda_nn", &lambda_nn);
    SF_scal_use("__trop_max_iter", &max_iter_d);
    SF_scal_use("__trop_tol", &tol);
    SF_scal_use("__trop_seed", &seed_d);
    
    max_iter = (int)max_iter_d;
    seed = (uint64_t)seed_d;
    
    TROP_LOG_DEBUG("bootstrap params: n_bootstrap=%d, seed=%llu, alpha=%g, ddof=%d",
                   n_bootstrap, (unsigned long long)seed, alpha, ddof);

    /* Optional pweight path — see handle_estimate_twostep for protocol. */
    if (SF_scal_use("__trop_use_weights", &use_weights_d) == 0) {
        use_weights = ((int)use_weights_d != 0) ? 1 : 0;
    }
    if (use_weights) {
        rc = read_lambda_grid("__trop_unit_weights", &unit_weights, &unit_weights_len);
        if (rc != TROP_SUCCESS) goto cleanup;
        if (unit_weights_len != (int)n_units) {
            TROP_LOG_ERROR("unit_weights length %d != n_units %d",
                           unit_weights_len, (int)n_units);
            rc = TROP_ERR_INVALID_DIM;
            goto cleanup;
        }
    }

    /* --- Covariate support --- */
    {
        double nc_val = 0;
        if (SF_scal_use("__trop_n_covariates", &nc_val) == 0 && nc_val > 0) {
            n_covariates = (int)nc_val;
        }
    }
    if (n_covariates > 0) {
        int n_obs = (int)n_periods * (int)n_units;
        x_buf = (double *)malloc((size_t)n_obs * (size_t)n_covariates * sizeof(double));
        if (!x_buf) {
            TROP_LOG_ERROR("covariate memory allocation failed");
            rc = TROP_ERR_MEMORY;
            goto cleanup;
        }
        {
            int row, col;
            double val;
            for (col = 0; col < n_covariates; col++) {
                for (row = 0; row < n_obs; row++) {
                    if (SF_mat_el("__trop_covariates", row + 1, col + 1, &val) != 0) {
                        val = 0.0;
                    }
                    x_buf[row + col * n_obs] = val;
                }
            }
        }
    }

    /* Call Rust bootstrap function (weighted / covariates / plain) */
    if (use_weights) {
        rust_rc = stata_bootstrap_trop_variance_joint_weighted(
            y_matrix, d_matrix,
            n_periods, n_units,
            lambda_time, lambda_unit, lambda_nn,
            n_bootstrap, max_iter, tol, seed, alpha, ddof,
            estimates, &se, &ci_lower_pct, &ci_upper_pct, &n_valid,
            unit_weights
        );
    } else if (n_covariates > 0) {
        rust_rc = stata_bootstrap_trop_variance_joint_with_covariates(
            y_matrix, d_matrix,
            n_periods, n_units,
            lambda_time, lambda_unit, lambda_nn,
            n_bootstrap, max_iter, tol, seed, alpha, ddof,
            estimates, &se, &ci_lower_pct, &ci_upper_pct, &n_valid,
            x_buf, n_covariates
        );
    } else {
        rust_rc = stata_bootstrap_trop_variance_joint(
            y_matrix, d_matrix,
            n_periods, n_units,
            lambda_time, lambda_unit, lambda_nn,
            n_bootstrap, max_iter, tol, seed, alpha, ddof,
            estimates, &se, &ci_lower_pct, &ci_upper_pct, &n_valid
        );
    }
    
    if (rust_rc != TROP_SUCCESS) {
        translate_error_code(rust_rc);
        rc = rust_rc;
        goto cleanup;
    }
    
    /* Write results */
    SF_scal_save("__trop_se", se);
    SF_scal_save("__trop_n_bootstrap_valid", (double)n_valid);
    SF_scal_save("__trop_level", 1.0 - alpha);
    
    /* Save percentile CI as diagnostics.  The authoritative CI is computed
     * by Mata using the t-distribution; the percentile CI is useful for
     * assessing bootstrap distribution asymmetry. */
    SF_scal_save("__trop_ci_lower_percentile", ci_lower_pct);
    SF_scal_save("__trop_ci_upper_percentile", ci_upper_pct);
    
    /* Write bootstrap estimates to matrix (only valid estimates) */
    rc = write_vector_to_matrix("__trop_bootstrap_estimates", estimates, n_valid, 0);
    if (rc != TROP_SUCCESS) goto cleanup;
    
    /* Warn if bootstrap failure rate exceeds 10% */
    double failure_rate = 1.0 - (double)n_valid / (double)n_bootstrap;
    if (failure_rate > 0.1) {
        char warn_msg[256];
        snprintf(warn_msg, sizeof(warn_msg), 
                 "warning: %.1f%% of bootstrap iterations failed (%d/%d valid)\n",
                 failure_rate * 100.0, n_valid, n_bootstrap);
        SF_display(warn_msg);
    }
    
    TROP_LOG_INFO("bootstrap complete: SE=%g, n_valid=%d/%d",
                  se, n_valid, n_bootstrap);
    
    rc = TROP_SUCCESS;
    
cleanup:
    free(y_matrix);
    free(d_matrix);
    free(estimates);
    free(unit_weights);
    free(x_buf);
    
    return rc;
}

/* ============================================================================
 * Command Handler: Bootstrap Rao-Wu Twostep
 * ============================================================================ */

static ST_retcode handle_bootstrap_rao_wu_twostep(void) {
    ST_int n_units, n_periods;
    double *y_matrix = NULL;
    double *d_matrix = NULL;
    unsigned char *control_mask = NULL;
    int64_t *time_dist = NULL;
    double *estimates = NULL;
    int64_t *strata = NULL;
    int64_t *psu = NULL;
    double *fpc = NULL;
    double *unit_weights = NULL;
    int strata_len = 0, psu_len = 0, fpc_len = 0, uw_len = 0;
    double lambda_time, lambda_unit, lambda_nn;
    double n_bootstrap_d, max_iter_d, tol, seed_d, alpha_d, ddof_d;
    int n_bootstrap, max_iter, ddof;
    uint64_t seed;
    double se, alpha;
    double ci_lower_pct, ci_upper_pct;
    int n_valid;
    ST_retcode rc;
    int rust_rc;
    
    TROP_LOG_INFO("starting Rao-Wu bootstrap variance estimation (twostep)");
    
    /* Read dimensions */
    rc = read_dimensions(&n_units, &n_periods);
    if (rc != TROP_SUCCESS) goto cleanup_rw_ts;
    
    /* Read bootstrap parameters */
    SF_scal_use("__trop_n_bootstrap", &n_bootstrap_d);
    n_bootstrap = (int)n_bootstrap_d;
    
    if (SF_scal_use("__trop_bs_alpha", &alpha_d) != 0) {
        alpha = 0.05;
    } else {
        alpha = alpha_d;
    }

    if (SF_scal_use("__trop_bs_ddof", &ddof_d) != 0) {
        ddof = 1;
    } else {
        ddof = ((int)ddof_d == 0) ? 0 : 1;
    }
    
    /* Allocate memory for panel data */
    y_matrix = (double *)malloc(n_units * n_periods * sizeof(double));
    d_matrix = (double *)malloc(n_units * n_periods * sizeof(double));
    control_mask = (unsigned char *)malloc(n_units * n_periods);
    time_dist = (int64_t *)malloc(n_periods * n_periods * sizeof(int64_t));
    estimates = (double *)malloc(n_bootstrap * sizeof(double));
    
    if (!y_matrix || !d_matrix || !control_mask || !time_dist || !estimates) {
        TROP_LOG_ERROR("memory allocation failed");
        rc = TROP_ERR_MEMORY;
        goto cleanup_rw_ts;
    }
    
    /* Read data matrices */
    double y_idx_d, d_idx_d, ctrl_idx_d;
    SF_scal_use("__trop_y_varindex", &y_idx_d);
    SF_scal_use("__trop_d_varindex", &d_idx_d);
    SF_scal_use("__trop_ctrl_varindex", &ctrl_idx_d);
    
    rc = read_panel_to_matrix((ST_int)y_idx_d, n_periods, n_units, y_matrix);
    if (rc != TROP_SUCCESS) goto cleanup_rw_ts;
    
    rc = read_panel_to_matrix((ST_int)d_idx_d, n_periods, n_units, d_matrix);
    if (rc != TROP_SUCCESS) goto cleanup_rw_ts;
    
    rc = read_control_mask((ST_int)ctrl_idx_d, n_periods, n_units, control_mask);
    if (rc != TROP_SUCCESS) goto cleanup_rw_ts;
    
    rc = read_time_dist_matrix("__trop_time_dist", n_periods, time_dist);
    if (rc != TROP_SUCCESS) goto cleanup_rw_ts;
    
    /* Read parameters */
    SF_scal_use("__trop_lambda_time", &lambda_time);
    SF_scal_use("__trop_lambda_unit", &lambda_unit);
    SF_scal_use("__trop_lambda_nn", &lambda_nn);
    SF_scal_use("__trop_max_iter", &max_iter_d);
    SF_scal_use("__trop_tol", &tol);
    SF_scal_use("__trop_seed", &seed_d);
    
    max_iter = (int)max_iter_d;
    seed = (uint64_t)seed_d;
    
    /* Read survey design matrices */
    {
        double *strata_d = NULL;
        double *psu_d = NULL;
        int i;
        
        rc = read_lambda_grid("__trop_strata", &strata_d, &strata_len);
        if (rc != TROP_SUCCESS) goto cleanup_rw_ts;
        
        rc = read_lambda_grid("__trop_psu", &psu_d, &psu_len);
        if (rc != TROP_SUCCESS) { free(strata_d); goto cleanup_rw_ts; }
        
        /* Convert double arrays to int64_t */
        strata = (int64_t *)malloc(strata_len * sizeof(int64_t));
        psu = (int64_t *)malloc(psu_len * sizeof(int64_t));
        if (!strata || !psu) {
            free(strata_d); free(psu_d);
            rc = TROP_ERR_MEMORY;
            goto cleanup_rw_ts;
        }
        for (i = 0; i < strata_len; i++) strata[i] = (int64_t)strata_d[i];
        for (i = 0; i < psu_len; i++) psu[i] = (int64_t)psu_d[i];
        free(strata_d);
        free(psu_d);
    }
    
    /* Read FPC (may be empty/0-row → pass NULL to Rust) */
    {
        double *fpc_tmp = NULL;
        rc = read_lambda_grid("__trop_fpc", &fpc_tmp, &fpc_len);
        if (rc != TROP_SUCCESS || fpc_len == 0) {
            fpc = NULL;
            free(fpc_tmp);
            if (rc != TROP_SUCCESS) rc = TROP_SUCCESS; /* empty FPC is OK */
        } else {
            fpc = fpc_tmp;
        }
    }
    
    /* Read unit weights (Mata layer guarantees this matrix always exists) */
    rc = read_lambda_grid("__trop_unit_weights", &unit_weights, &uw_len);
    if (rc != TROP_SUCCESS || unit_weights == NULL || uw_len == 0) {
        SF_error("internal error: __trop_unit_weights matrix missing or unreadable\n");
        rc = 198;
        goto cleanup_rw_ts;
    }
    if (uw_len != (int)n_units) {
        TROP_LOG_ERROR("unit_weights length %d != n_units %d", uw_len, (int)n_units);
        SF_error("internal error: __trop_unit_weights wrong size\n");
        rc = TROP_ERR_INVALID_DIM;
        goto cleanup_rw_ts;
    }
    
    TROP_LOG_DEBUG("rao_wu_twostep params: n_bootstrap=%d, seed=%llu, alpha=%g",
                   n_bootstrap, (unsigned long long)seed, alpha);

    /* Call Rust Rao-Wu bootstrap */
    rust_rc = stata_bootstrap_trop_variance_rao_wu(
        y_matrix, d_matrix, control_mask, time_dist,
        n_periods, n_units,
        lambda_time, lambda_unit, lambda_nn,
        n_bootstrap, max_iter, tol, seed, alpha, ddof,
        strata, psu, fpc, unit_weights,
        estimates, &se, &ci_lower_pct, &ci_upper_pct, &n_valid
    );
    
    if (rust_rc != TROP_SUCCESS) {
        translate_error_code(rust_rc);
        rc = rust_rc;
        goto cleanup_rw_ts;
    }
    
    /* Write results */
    SF_scal_save("__trop_se", se);
    SF_scal_save("__trop_n_bootstrap_valid", (double)n_valid);
    SF_scal_save("__trop_level", 1.0 - alpha);
    SF_scal_save("__trop_ci_lower_percentile", ci_lower_pct);
    SF_scal_save("__trop_ci_upper_percentile", ci_upper_pct);
    
    rc = write_vector_to_matrix("__trop_bootstrap_estimates", estimates, n_valid, 0);
    if (rc != TROP_SUCCESS) goto cleanup_rw_ts;
    
    double failure_rate = 1.0 - (double)n_valid / (double)n_bootstrap;
    if (failure_rate > 0.1) {
        char warn_msg[256];
        snprintf(warn_msg, sizeof(warn_msg), 
                 "warning: %.1f%% of Rao-Wu bootstrap iterations failed (%d/%d valid)\n",
                 failure_rate * 100.0, n_valid, n_bootstrap);
        SF_display(warn_msg);
    }
    
    TROP_LOG_INFO("Rao-Wu bootstrap (twostep) complete: SE=%g, n_valid=%d/%d",
                  se, n_valid, n_bootstrap);
    
    rc = TROP_SUCCESS;
    
cleanup_rw_ts:
    free(y_matrix);
    free(d_matrix);
    free(control_mask);
    free(time_dist);
    free(estimates);
    free(strata);
    free(psu);
    free(fpc);
    free(unit_weights);
    
    return rc;
}

/* ============================================================================
 * Command Handler: Bootstrap Rao-Wu Joint
 * ============================================================================ */

static ST_retcode handle_bootstrap_rao_wu_joint(void) {
    ST_int n_units, n_periods;
    double *y_matrix = NULL;
    double *d_matrix = NULL;
    double *estimates = NULL;
    int64_t *strata = NULL;
    int64_t *psu = NULL;
    double *fpc = NULL;
    double *unit_weights = NULL;
    int strata_len = 0, psu_len = 0, fpc_len = 0, uw_len = 0;
    double lambda_time, lambda_unit, lambda_nn;
    double n_bootstrap_d, max_iter_d, tol, seed_d, alpha_d, ddof_d;
    int n_bootstrap, max_iter, ddof;
    uint64_t seed;
    double se, alpha;
    double ci_lower_pct, ci_upper_pct;
    int n_valid;
    ST_retcode rc;
    int rust_rc;
    
    TROP_LOG_INFO("starting Rao-Wu bootstrap variance estimation (joint)");
    
    /* Read dimensions */
    rc = read_dimensions(&n_units, &n_periods);
    if (rc != TROP_SUCCESS) goto cleanup_rw_jt;
    
    /* Read bootstrap parameters */
    SF_scal_use("__trop_n_bootstrap", &n_bootstrap_d);
    n_bootstrap = (int)n_bootstrap_d;
    
    if (SF_scal_use("__trop_bs_alpha", &alpha_d) != 0) {
        alpha = 0.05;
    } else {
        alpha = alpha_d;
    }

    if (SF_scal_use("__trop_bs_ddof", &ddof_d) != 0) {
        ddof = 1;
    } else {
        ddof = ((int)ddof_d == 0) ? 0 : 1;
    }
    
    /* Allocate memory for panel data */
    y_matrix = (double *)malloc(n_units * n_periods * sizeof(double));
    d_matrix = (double *)malloc(n_units * n_periods * sizeof(double));
    estimates = (double *)malloc(n_bootstrap * sizeof(double));
    
    if (!y_matrix || !d_matrix || !estimates) {
        TROP_LOG_ERROR("memory allocation failed");
        rc = TROP_ERR_MEMORY;
        goto cleanup_rw_jt;
    }
    
    /* Read data matrices */
    double y_idx_d, d_idx_d;
    SF_scal_use("__trop_y_varindex", &y_idx_d);
    SF_scal_use("__trop_d_varindex", &d_idx_d);
    
    rc = read_panel_to_matrix((ST_int)y_idx_d, n_periods, n_units, y_matrix);
    if (rc != TROP_SUCCESS) goto cleanup_rw_jt;
    
    rc = read_panel_to_matrix((ST_int)d_idx_d, n_periods, n_units, d_matrix);
    if (rc != TROP_SUCCESS) goto cleanup_rw_jt;
    
    /* Read parameters */
    SF_scal_use("__trop_lambda_time", &lambda_time);
    SF_scal_use("__trop_lambda_unit", &lambda_unit);
    SF_scal_use("__trop_lambda_nn", &lambda_nn);
    SF_scal_use("__trop_max_iter", &max_iter_d);
    SF_scal_use("__trop_tol", &tol);
    SF_scal_use("__trop_seed", &seed_d);
    
    max_iter = (int)max_iter_d;
    seed = (uint64_t)seed_d;
    
    /* Read survey design matrices */
    {
        double *strata_d = NULL;
        double *psu_d = NULL;
        int i;
        
        rc = read_lambda_grid("__trop_strata", &strata_d, &strata_len);
        if (rc != TROP_SUCCESS) goto cleanup_rw_jt;
        
        rc = read_lambda_grid("__trop_psu", &psu_d, &psu_len);
        if (rc != TROP_SUCCESS) { free(strata_d); goto cleanup_rw_jt; }
        
        /* Convert double arrays to int64_t */
        strata = (int64_t *)malloc(strata_len * sizeof(int64_t));
        psu = (int64_t *)malloc(psu_len * sizeof(int64_t));
        if (!strata || !psu) {
            free(strata_d); free(psu_d);
            rc = TROP_ERR_MEMORY;
            goto cleanup_rw_jt;
        }
        for (i = 0; i < strata_len; i++) strata[i] = (int64_t)strata_d[i];
        for (i = 0; i < psu_len; i++) psu[i] = (int64_t)psu_d[i];
        free(strata_d);
        free(psu_d);
    }
    
    /* Read FPC (may be empty/0-row → pass NULL to Rust) */
    {
        double *fpc_tmp = NULL;
        rc = read_lambda_grid("__trop_fpc", &fpc_tmp, &fpc_len);
        if (rc != TROP_SUCCESS || fpc_len == 0) {
            fpc = NULL;
            free(fpc_tmp);
            if (rc != TROP_SUCCESS) rc = TROP_SUCCESS; /* empty FPC is OK */
        } else {
            fpc = fpc_tmp;
        }
    }
    
    /* Read unit weights (Mata layer guarantees this matrix always exists) */
    rc = read_lambda_grid("__trop_unit_weights", &unit_weights, &uw_len);
    if (rc != TROP_SUCCESS || unit_weights == NULL || uw_len == 0) {
        SF_error("internal error: __trop_unit_weights matrix missing or unreadable\n");
        rc = 198;
        goto cleanup_rw_jt;
    }
    if (uw_len != (int)n_units) {
        TROP_LOG_ERROR("unit_weights length %d != n_units %d", uw_len, (int)n_units);
        SF_error("internal error: __trop_unit_weights wrong size\n");
        rc = TROP_ERR_INVALID_DIM;
        goto cleanup_rw_jt;
    }
    
    TROP_LOG_DEBUG("rao_wu_joint params: n_bootstrap=%d, seed=%llu, alpha=%g",
                   n_bootstrap, (unsigned long long)seed, alpha);

    /* Call Rust Rao-Wu bootstrap (joint) */
    rust_rc = stata_bootstrap_trop_variance_rao_wu_joint(
        y_matrix, d_matrix,
        n_periods, n_units,
        lambda_time, lambda_unit, lambda_nn,
        n_bootstrap, max_iter, tol, seed, alpha, ddof,
        strata, psu, fpc, unit_weights,
        estimates, &se, &ci_lower_pct, &ci_upper_pct, &n_valid
    );
    
    if (rust_rc != TROP_SUCCESS) {
        translate_error_code(rust_rc);
        rc = rust_rc;
        goto cleanup_rw_jt;
    }
    
    /* Write results */
    SF_scal_save("__trop_se", se);
    SF_scal_save("__trop_n_bootstrap_valid", (double)n_valid);
    SF_scal_save("__trop_level", 1.0 - alpha);
    SF_scal_save("__trop_ci_lower_percentile", ci_lower_pct);
    SF_scal_save("__trop_ci_upper_percentile", ci_upper_pct);
    
    rc = write_vector_to_matrix("__trop_bootstrap_estimates", estimates, n_valid, 0);
    if (rc != TROP_SUCCESS) goto cleanup_rw_jt;
    
    double failure_rate = 1.0 - (double)n_valid / (double)n_bootstrap;
    if (failure_rate > 0.1) {
        char warn_msg[256];
        snprintf(warn_msg, sizeof(warn_msg), 
                 "warning: %.1f%% of Rao-Wu bootstrap iterations failed (%d/%d valid)\n",
                 failure_rate * 100.0, n_valid, n_bootstrap);
        SF_display(warn_msg);
    }
    
    TROP_LOG_INFO("Rao-Wu bootstrap (joint) complete: SE=%g, n_valid=%d/%d",
                  se, n_valid, n_bootstrap);
    
    rc = TROP_SUCCESS;
    
cleanup_rw_jt:
    free(y_matrix);
    free(d_matrix);
    free(estimates);
    free(strata);
    free(psu);
    free(fpc);
    free(unit_weights);
    
    return rc;
}

/* ============================================================================
 * Command Handler: Distance Matrix
 * ============================================================================ */

static ST_retcode handle_distance_matrix(void) {
    ST_int n_units, n_periods;
    double *y_matrix = NULL;
    double *d_matrix = NULL;
    double *dist_matrix = NULL;
    ST_retcode rc;
    int rust_rc;
    
    TROP_LOG_INFO("computing unit distance matrix");
    
    /* Read dimensions */
    rc = read_dimensions(&n_units, &n_periods);
    if (rc != TROP_SUCCESS) goto cleanup;
    
    /* Allocate memory */
    y_matrix = (double *)malloc(n_units * n_periods * sizeof(double));
    d_matrix = (double *)malloc(n_units * n_periods * sizeof(double));
    dist_matrix = (double *)malloc(n_units * n_units * sizeof(double));
    
    if (!y_matrix || !d_matrix || !dist_matrix) {
        TROP_LOG_ERROR("memory allocation failed");
        rc = TROP_ERR_MEMORY;
        goto cleanup;
    }
    
    /* Read data */
    double y_idx_d, d_idx_d;
    SF_scal_use("__trop_y_varindex", &y_idx_d);
    SF_scal_use("__trop_d_varindex", &d_idx_d);
    
    rc = read_panel_to_matrix((ST_int)y_idx_d, n_periods, n_units, y_matrix);
    if (rc != TROP_SUCCESS) goto cleanup;
    
    rc = read_panel_to_matrix((ST_int)d_idx_d, n_periods, n_units, d_matrix);
    if (rc != TROP_SUCCESS) goto cleanup;
    
    /* Call Rust function */
    rust_rc = stata_compute_unit_distance_matrix(
        y_matrix, d_matrix,
        n_periods, n_units,
        dist_matrix
    );
    
    if (rust_rc != TROP_SUCCESS) {
        translate_error_code(rust_rc);
        rc = rust_rc;
        goto cleanup;
    }
    
    /* Write distance matrix to Stata */
    rc = write_matrix_to_stata("__trop_unit_dist", dist_matrix, n_units, n_units);
    if (rc != TROP_SUCCESS) goto cleanup;
    
    TROP_LOG_INFO("distance matrix computed: %dx%d", n_units, n_units);
    
    rc = TROP_SUCCESS;
    
cleanup:
    free(y_matrix);
    free(d_matrix);
    free(dist_matrix);
    
    return rc;
}

/* ============================================================================
 * Initialize Verbose Level
 * ============================================================================ */

static void init_verbose_level(void) {
    double verbose_d;
    ST_retcode rc;
    
    rc = SF_scal_use("__trop_verbose", &verbose_d);
    if (rc == 0) {
        g_verbose_level = (int)verbose_d;
    } else {
        g_verbose_level = TROP_VERBOSE_NORMAL;
    }
}

/* ============================================================================
 * Plugin Entry Point
 * ============================================================================ */

STDLL stata_call(int argc, char *argv[]) {
    TropCommand cmd;
    ST_retcode rc;
    
    /* Initialize verbose level */
    init_verbose_level();
    
    /* Check for command argument */
    if (argc < 1 || argv[0] == NULL) {
        SF_error("trop: no command specified\n");
        return TROP_ERR_INVALID_ARGS;
    }
    
    /* Parse command */
    cmd = parse_command(argv[0]);
    
    TROP_LOG_DEBUG("received command: %s", argv[0]);
    
    /* Dispatch to handler */
    switch (cmd) {
        case CMD_LOOCV_TWOSTEP:
            rc = handle_loocv_twostep();
            break;

        case CMD_LOOCV_TWOSTEP_EXHAUSTIVE:
            rc = handle_loocv_twostep_exhaustive();
            break;

        case CMD_LOOCV_JOINT:
            rc = handle_loocv_joint();
            break;
            
        case CMD_LOOCV_JOINT_EXHAUSTIVE:
            rc = handle_loocv_joint_exhaustive();
            break;
            
        case CMD_ESTIMATE_TWOSTEP:
            rc = handle_estimate_twostep();
            break;
            
        case CMD_ESTIMATE_JOINT:
            rc = handle_estimate_joint();
            break;
            
        case CMD_BOOTSTRAP_TWOSTEP:
            rc = handle_bootstrap_twostep();
            break;
            
        case CMD_BOOTSTRAP_JOINT:
            rc = handle_bootstrap_joint();
            break;
            
        case CMD_BOOTSTRAP_RAO_WU_TWOSTEP:
            rc = handle_bootstrap_rao_wu_twostep();
            break;
            
        case CMD_BOOTSTRAP_RAO_WU_JOINT:
            rc = handle_bootstrap_rao_wu_joint();
            break;
            
        case CMD_DISTANCE_MATRIX:
            rc = handle_distance_matrix();
            break;
            
        case CMD_UNKNOWN:
        default:
            SF_error("trop: unknown command '");
            SF_error(argv[0]);
            SF_error("'\n");
            rc = TROP_ERR_INVALID_ARGS;
            break;
    }
    
    return rc;
}

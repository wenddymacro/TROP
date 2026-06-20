/*==============================================================================
  trop_main.mata

  Entry point for the TROP estimator.

  Coordinates the full estimation pipeline:
    1. Validate inputs and resolve optional parameters.
    2. Prepare the N x T outcome matrix Y and treatment matrix W.
    3. Set up the regularization parameter grid
       (lambda_time, lambda_unit, lambda_nn) or apply fixed values.
    4. Invoke the compiled plugin for LOOCV tuning, point estimation,
       and bootstrap variance estimation.
    5. Display the results summary.

  Call chain:  trop.ado -> trop_main() -> _trop_main() -> plugin
==============================================================================*/

version 17
mata:
mata set matastrict on

/*------------------------------------------------------------------------------
  trop_main()

  Entry point called from the ADO layer.

  Arguments
    depvar           -- outcome variable Y_{it}
    treatvar         -- binary treatment indicator W_{it}
    panel_idx_var    -- unit (panel) identifier, i = 1,...,N
    time_idx_var     -- time period identifier, t = 1,...,T
    touse_var        -- estimation-sample marker
    method           -- "twostep" (per-observation tau_{it}, paper
                        Algorithm 2 / Eq. 13) or
                        "joint" (pooled weighted least squares; paper
                        Remark 6.1 homogeneous-tau aggregation; the
                        delta_time / delta_unit kernels are engineering
                        choices not prescribed by Remark 6.1, see
                        rust/src/weights.rs::compute_joint_weights)
    lambda_time_user -- fixed lambda_time; missing triggers LOOCV selection
    lambda_unit_user -- fixed lambda_unit; missing triggers LOOCV selection
    lambda_nn_user   -- fixed lambda_nn (nuclear-norm penalty); missing
                        triggers LOOCV selection
    run_loocv        -- 1 = select (lambda_time, lambda_unit, lambda_nn)
                        via leave-one-out cross-validation
    boot_reps        -- number of bootstrap replications B;
                        0 disables bootstrap inference
    seed             -- RNG seed for bootstrap resampling
    verbose          -- verbosity level (0 = silent)

  Optional arguments
    max_iter_user      -- iteration limit for the SVD solver (default 500)
    tol_user           -- convergence tolerance (default 1e-6)
    alpha_level_user   -- significance level for CIs (default 0.05)
    ddof_user          -- bootstrap-variance denominator selector:
                          1 = sample variance 1/(B-1) (default),
                          0 = paper Algorithm 3 population variance 1/B.
                          Any other value collapses to 1.
    weight_var_user    -- pweight variable name; empty disables pweights.
                          When set, per-unit pweights must be strictly
                          positive and constant within unit.

  Returns
    0 on success; nonzero Stata return code on failure.
------------------------------------------------------------------------------*/

real scalar trop_main(
    string scalar depvar,
    string scalar treatvar,
    string scalar panel_idx_var,
    string scalar time_idx_var,
    string scalar touse_var,
    string scalar method,
    real scalar lambda_time_user,
    real scalar lambda_unit_user,
    real scalar lambda_nn_user,
    real scalar run_loocv,
    real scalar boot_reps,
    real scalar seed,
    real scalar verbose,
    | real scalar max_iter_user,
    real scalar tol_user,
    real scalar alpha_level_user,
    real scalar ddof_user,
    string scalar weight_var_user
)
{
    real scalar rc, do_loocv, do_bootstrap
    real scalar n_units, n_periods
    real colvector panel_idx, time_idx
    real scalar max_iter_eff, tol_eff, alpha_eff, ddof_eff
    real scalar have_ddof
    real scalar has_survey, do_nest
    string scalar strata_var, psu_var, fpc_var
    
    // Resolve optional parameters to effective values
    if (args() >= 14 & max_iter_user < . & max_iter_user > 0) max_iter_eff = max_iter_user
    else max_iter_eff = 500
    if (args() >= 15 & tol_user < . & tol_user > 0) tol_eff = tol_user
    else tol_eff = 1e-6
    if (args() >= 16 & alpha_level_user < . & alpha_level_user > 0 & alpha_level_user < 1) alpha_eff = alpha_level_user
    else alpha_eff = 0.05
    // ddof: forwarded only when the caller supplies a finite value.
    have_ddof = (args() >= 17 & ddof_user < .)
    if (have_ddof) ddof_eff = (ddof_user == 0) ? 0 : 1
    else ddof_eff = 1
    
    // Validate estimation method
    if (method != "twostep" & method != "joint") {
        errprintf("Invalid method: %s. Must be 'twostep' or 'joint'.\n", method)
        return(198)
    }
    
    // Extract panel dimensions N and T
    panel_idx = st_data(., panel_idx_var, touse_var)
    time_idx = st_data(., time_idx_var, touse_var)
    n_units = max(panel_idx)
    n_periods = max(time_idx)
    
    // Count treated cells sum(W_{it}) for tau vector pre-allocation
    real colvector d_vec
    real scalar n_treated
    d_vec = st_data(., treatvar, touse_var)
    n_treated = sum(d_vec :!= 0)
    if (n_treated < 1) n_treated = 1
    
    if (verbose) {
        printf("{txt}\n")
        printf("{txt}TROP Estimation\n")
        printf("{txt}====================================\n")
        printf("{txt}Method: %s\n", method)
        printf("{txt}Data dimensions: N=%g units, T=%g periods\n", n_units, n_periods)
        printf("{txt}Treated observations: %g\n", n_treated)
    }
    
    // Transfer the N x T outcome and treatment matrices to the plugin
    trop_prepare_data(depvar, treatvar, panel_idx_var, time_idx_var, 
                      n_units, n_periods)
    
    // Record the touse variable so the plugin reads only in-sample rows
    st_global("__trop_touse_var", touse_var)

    // Optional pweight: validate + write __trop_unit_weights / __trop_use_weights.
    // Default is 0 (disabled); the plugin only consults the weighted ABI
    // when __trop_use_weights is present and equals 1.
    st_numscalar("__trop_use_weights", 0)
    if (args() >= 18 & weight_var_user != "") {
        rc = trop_prepare_pweights(weight_var_user, panel_idx_var,
                                   touse_var, n_units)
        if (rc != 0) return(rc)
        if (verbose) {
            printf("{txt}Survey weights: enabled (pweight = %s)\n", weight_var_user)
        }
    }

    // Survey design preparation (Rao-Wu bootstrap).
    // Read survey variables from globals set by the ADO layer.
    has_survey = st_numscalar("__trop_has_survey_design")
    if (has_survey >= .) has_survey = 0
    if (has_survey == 1) {
        strata_var = st_global("__trop_strata_var")
        psu_var    = st_global("__trop_psu_var")
        fpc_var    = st_global("__trop_fpc_var")
        do_nest    = st_numscalar("__trop_survey_nest")
        if (do_nest >= .) do_nest = 0

        rc = trop_prepare_survey_design(strata_var, psu_var, fpc_var,
                                        panel_idx_var, touse_var,
                                        n_units, do_nest)
        if (rc != 0) return(rc)

        // Rao-Wu bootstrap requires unit_weights.  When no explicit pweight
        // was supplied by the user, create uniform weights (all 1.0) so the
        // C bridge always finds a valid __trop_unit_weights matrix.
        // This matches the Python reference: unit_weights = np.ones(n_units).
        if (st_numscalar("__trop_use_weights") == 0) {
            st_matrix("__trop_unit_weights", J(n_units, 1, 1))
            st_numscalar("__trop_use_weights", 1)
            if (verbose) {
                printf("{txt}Survey design: no pweight specified; using uniform weights\n")
            }
        }

        if (verbose) {
            printf("{txt}Survey design: Rao-Wu bootstrap active\n")
            printf("{txt}  strata=%s, psu=%s", strata_var, psu_var)
            if (fpc_var != "") printf(", fpc=%s", fpc_var)
            printf("\n")
        }
    }
    
    // Pre-allocate output matrices (ATT, SE, tau vector, etc.)
    trop_prepare_output_matrices(n_units, n_periods, n_treated)
    
    // Configure regularization parameters (lambda_time, lambda_unit, lambda_nn)
    do_loocv = (run_loocv == 1)
    
    if (do_loocv) {
        // Load user-specified grids or fall back to defaults.
        // LOOCV minimizes Q(lambda) = sum_{W_{it}=0} tau_{it}(lambda)^2
        // over a grid of (lambda_time, lambda_unit, lambda_nn).
        real rowvector lambda_time_grid, lambda_unit_grid, lambda_nn_grid
        
        if (rows(st_matrix("__trop_lambda_time_grid")) > 0) {
            lambda_time_grid = st_matrix("__trop_lambda_time_grid")
        }
        else {
            lambda_time_grid = (0, 0.1, 0.5, 1, 2, 5)
        }
        
        if (rows(st_matrix("__trop_lambda_unit_grid")) > 0) {
            lambda_unit_grid = st_matrix("__trop_lambda_unit_grid")
        }
        else {
            lambda_unit_grid = (0, 0.1, 0.5, 1, 2, 5)
        }
        
        if (rows(st_matrix("__trop_lambda_nn_grid")) > 0) {
            lambda_nn_grid = st_matrix("__trop_lambda_nn_grid")
        }
        else {
            // Mirror the default grid advertised by `_trop_set_grid.ado` and
            // `trop_get_lambda_grid`: a five-point log-decade ladder covering
            // the empirically relevant range of the paper's Eq. 2 nuclear-
            // norm penalty.  The DID/TWFE corner (λ_nn = ∞, encoded as
            // Stata missing .) is reserved for `grid_style(extended)` or
            // user-supplied grids.
            lambda_nn_grid = (0, 0.01, 0.1, 1, 10)
        }
        
        trop_prepare_lambda_grids(
            lambda_time_grid,
            lambda_unit_grid,
            lambda_nn_grid
        )
    }
    else {
        // Fixed lambda values; apply defaults when the user omits them
        if (lambda_time_user >= .) lambda_time_user = 1.0
        if (lambda_unit_user >= .) lambda_unit_user = 1.0
        if (lambda_nn_user >= .) lambda_nn_user = 0.1
        
        st_numscalar("__trop_lambda_time", lambda_time_user)
        st_numscalar("__trop_lambda_unit", lambda_unit_user)
        st_numscalar("__trop_lambda_nn", lambda_nn_user)
    }
    
    // Configure bootstrap and solver options
    do_bootstrap = (boot_reps > 0)
    
    trop_prepare_options(
        max_iter_eff, tol_eff, seed, boot_reps, alpha_eff, verbose
    )
    
    if (verbose) {
        if (do_loocv) printf("{txt}LOOCV: enabled\n")
        else printf("{txt}LOOCV: disabled\n")
        if (do_bootstrap) printf("{txt}Bootstrap: enabled")
        else printf("{txt}Bootstrap: disabled")
        if (do_bootstrap) printf(" (B=%g)", boot_reps)
        printf("\n")
    }
    
    // Invoke the compiled plugin
    if (verbose) {
        printf("{txt}\nCalling plugin...\n")
    }
    
    if (have_ddof) rc = _trop_main(method, do_loocv, do_bootstrap, boot_reps, ddof_eff)
    else           rc = _trop_main(method, do_loocv, do_bootstrap, boot_reps)
    
    if (rc != 0) {
        errprintf("TROP estimation failed with error code %g\n", rc)
        return(rc)
    }
    
    // Display estimation summary
    if (verbose) {
        _trop_display_summary(method)
    }
    
    return(0)
}

/*------------------------------------------------------------------------------
  _trop_display_summary()

  Print a summary table of estimation results.  Reads from temporary Stata
  scalars (__trop_*) populated by the plugin, before ereturn post.

  Inference uses the t(N_1 - 1) distribution when at least two treated
  units are available, falling back to the standard normal otherwise.
  N_1 is the cluster count for the stratified bootstrap (Algorithm 3).
------------------------------------------------------------------------------*/

void _trop_display_summary(string scalar method)
{
    real scalar att, se, ci_lower, ci_upper, pvalue, tstat
    real scalar lambda_time, lambda_unit, lambda_nn
    real scalar n_units, n_periods, converged
    real scalar alpha_level, n_treated_units, df_pvalue
    real scalar level_pct
    
    // ATT = (1/sum W) * sum W_{it} * tau_{it}
    att = _trop_safe_read_scalar("__trop_att")
    se = _trop_safe_read_scalar("__trop_se")
    
    // Resolve significance level from available sources
    alpha_level = _trop_safe_read_scalar("__trop_bs_alpha")
    if (alpha_level >= . | alpha_level <= 0 | alpha_level >= 1) {
        alpha_level = _trop_safe_read_scalar("__trop_alpha_level")
    }
    if (alpha_level >= . | alpha_level <= 0 | alpha_level >= 1) {
        alpha_level = 0.05
    }
    level_pct = 100 * (1 - alpha_level)
    
    // Compute t-statistic, p-value, and confidence interval.
    // Reference distribution: t(N_1 - 1) when at least 2 treated units
    // are available (Algorithm 3 resamples units, so N_1 is the cluster
    // count); normal approximation otherwise.
    n_treated_units = _trop_safe_read_scalar("__trop_n_treated_units")
    if (se > 0 && se < .) {
        tstat = att / se
        if (n_treated_units < . && n_treated_units >= 2) {
            df_pvalue = max((1, n_treated_units - 1))
            pvalue = 2 * ttail(df_pvalue, abs(tstat))
            ci_lower = att - invttail(df_pvalue, alpha_level / 2) * se
            ci_upper = att + invttail(df_pvalue, alpha_level / 2) * se
        }
        else {
            pvalue = 2 * normal(-abs(tstat))
            ci_lower = att - invnormal(1 - alpha_level / 2) * se
            ci_upper = att + invnormal(1 - alpha_level / 2) * se
        }
    }
    else {
        tstat = .
        pvalue = .
        ci_lower = .
        ci_upper = .
    }
    
    // Selected regularization parameters (lambda_time, lambda_unit, lambda_nn)
    lambda_time = _trop_safe_read_scalar("__trop_lambda_time")
    lambda_unit = _trop_safe_read_scalar("__trop_lambda_unit")
    lambda_nn = _trop_safe_read_scalar("__trop_lambda_nn")
    
    // Panel dimensions and convergence indicator
    n_units = _trop_safe_read_scalar("__trop_n_units")
    n_periods = _trop_safe_read_scalar("__trop_n_periods")
    converged = _trop_safe_read_scalar("__trop_converged")
    if (converged >= .) converged = 0
    
    printf("{txt}\n")
    printf("{txt}TROP Estimation Results\n")
    printf("{txt}=======================\n")
    printf("{txt}Method: %s\n", method)
    printf("{txt}\n")
    printf("{txt}ATT estimate:     %12.6f\n", att)
    printf("{txt}Std. Error:       %12.6f\n", se)
    printf("{txt}%g%% CI:           [%9.4f, %9.4f]\n", level_pct, ci_lower, ci_upper)
    printf("{txt}p-value:          %12.4f\n", pvalue)
    printf("{txt}\n")
    printf("{txt}Selected lambda:\n")
    printf("{txt}  lambda_time:    %12.6f\n", lambda_time)
    printf("{txt}  lambda_unit:    %12.6f\n", lambda_unit)
    /* lambda_nn: show the internal 1e10 DID/TWFE corner sentinel as "+inf"
       so the printed summary matches the user-facing e(lambda_nn) value.
       _trop_lambda_nn_user_face() returns Stata missing (.) for the corner
       case; printf's %g formatter renders "." as ".", so we branch here to
       emit the human-readable "+inf" marker. */
    if (_trop_lambda_nn_user_face(lambda_nn) >= .) {
        printf("{txt}  lambda_nn:      %12s  (DID/TWFE corner)\n", "+inf")
    }
    else {
        printf("{txt}  lambda_nn:      %12.6f\n", lambda_nn)
    }
    printf("{txt}\n")
    printf("{txt}Sample: N=%g units, T=%g periods\n", n_units, n_periods)
    if (converged) printf("{txt}Converged: Yes\n")
    else printf("{txt}Converged: No\n")
}

end

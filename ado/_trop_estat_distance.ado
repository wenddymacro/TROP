*! _trop_estat_distance.ado
*! Unit distance distribution diagnostics for trop
*!
*! Computes and visualizes the pairwise unit distance matrix distribution.
*! Theory: Paper Eq.3 defines dist_{-t}^unit(j,i) = sqrt[mean_{u!=t}((Y_iu-Y_ju)^2)]
*!
*! Syntax:
*!   _trop_estat_distance [, GRAPH SAVing(string) BINS(integer)]
*!
*! Options:
*!   graph       - Display publication-quality weight diagnostic plots
*!   saving(str) - Save combined graph to file (e.g., saving(diag.gph))
*!   bins(#)     - Number of histogram bins (default: 30)

program define _trop_estat_distance, rclass
    version 17.0
    syntax [, GRAPH SAVing(string) BINS(integer 30)]

    /* ──────────────────────────────────────────────────────────────────────
       1. Pre-checks
    ────────────────────────────────────────────────────────────────────── */

    if "`e(cmd)'" != "trop" {
        di as error "estat distance requires trop estimation results"
        exit 301
    }

    // Retrieve panel dimensions from estimation results
    local N_units   = e(N_units)
    local N_periods = e(N_periods)
    local depvar    "`e(depvar)'"
    local treatvar  "`e(treatvar)'"
    local panelvar  "`e(panelvar)'"
    local timevar   "`e(timevar)'"

    if `N_units' < 2 {
        di as error "Distance computation requires at least 2 panel units"
        exit 459
    }

    /* ──────────────────────────────────────────────────────────────────────
       2. Compute pairwise unit distances via Mata
    ────────────────────────────────────────────────────────────────────── */

    mata: {
        real matrix _ed_Y, _ed_D, _ed_obs_data
        real scalar _ed_N, _ed_T, _ed_nobs, _ed_k
        real scalar _ed_row_t, _ed_col_i
        string scalar _ed_panel_idx_var, _ed_time_idx_var, _ed_touse_var

        _ed_N = `N_units'
        _ed_T = `N_periods'

        // Retrieve global variable names set by trop.ado
        _ed_panel_idx_var = st_global("__trop_panel_idx_var")
        _ed_time_idx_var  = st_global("__trop_time_idx_var")
        _ed_touse_var     = ""

        // Use e(sample) as the estimation sample marker
        // Read depvar, treatvar, panel_idx, time_idx from estimation sample
        if (_ed_panel_idx_var == "" | _ed_time_idx_var == "" | ///
            _st_varindex(_ed_panel_idx_var) >= . | ///
            _st_varindex(_ed_time_idx_var) >= .) {
            errprintf("Panel index variables not found in memory.\n")
            errprintf("Re-run trop estimation before using estat distance.\n")
            st_local("_ed_rc", "459")
        }
        else {
            st_local("_ed_rc", "0")

            // Read data restricted to e(sample)
            _ed_obs_data = st_data(., ///
                ("`depvar'", "`treatvar'", _ed_panel_idx_var, _ed_time_idx_var), ///
                "e(sample)")
            _ed_nobs = rows(_ed_obs_data)

            // Construct Y (T x N) and D (T x N) matrices
            _ed_Y = J(_ed_T, _ed_N, .)
            _ed_D = J(_ed_T, _ed_N, 0)

            for (_ed_k = 1; _ed_k <= _ed_nobs; _ed_k++) {
                _ed_row_t = _ed_obs_data[_ed_k, 4]
                _ed_col_i = _ed_obs_data[_ed_k, 3]
                if (_ed_row_t >= 1 & _ed_row_t <= _ed_T & ///
                    _ed_col_i >= 1 & _ed_col_i <= _ed_N) {
                    _ed_Y[_ed_row_t, _ed_col_i] = _ed_obs_data[_ed_k, 1]
                    _ed_D[_ed_row_t, _ed_col_i] = (_ed_obs_data[_ed_k, 2] != 0 ? 1 : 0)
                }
            }

            // Compute pairwise unit distances (upper triangle)
            // dist(i,j) = sqrt(mean((Y_t_i - Y_t_j)^2)) over control periods
            real scalar _ed_n_pairs, _ed_idx, _ed_i, _ed_j, _ed_t
            real scalar _ed_sum_sq, _ed_count, _ed_diff
            real colvector _ed_distances, _ed_valid_dist

            _ed_n_pairs = _ed_N * (_ed_N - 1) / 2
            _ed_distances = J(_ed_n_pairs, 1, .)

            _ed_idx = 0
            for (_ed_i = 1; _ed_i <= _ed_N; _ed_i++) {
                for (_ed_j = _ed_i + 1; _ed_j <= _ed_N; _ed_j++) {
                    _ed_idx++
                    _ed_sum_sq = 0
                    _ed_count = 0
                    for (_ed_t = 1; _ed_t <= _ed_T; _ed_t++) {
                        // Use only periods where both units are untreated
                        // and outcomes are non-missing
                        if (_ed_D[_ed_t, _ed_i] == 0 & _ed_D[_ed_t, _ed_j] == 0 & ///
                            _ed_Y[_ed_t, _ed_i] < . & _ed_Y[_ed_t, _ed_j] < .) {
                            _ed_diff = _ed_Y[_ed_t, _ed_i] - _ed_Y[_ed_t, _ed_j]
                            _ed_sum_sq = _ed_sum_sq + _ed_diff * _ed_diff
                            _ed_count++
                        }
                    }
                    if (_ed_count > 0) {
                        _ed_distances[_ed_idx] = sqrt(_ed_sum_sq / _ed_count)
                    }
                }
            }

            // Remove missing values (pairs with no valid control periods)
            _ed_valid_dist = select(_ed_distances, _ed_distances :< .)

            if (rows(_ed_valid_dist) == 0) {
                errprintf("No valid unit pairs found for distance computation.\n")
                st_local("_ed_rc", "459")
            }
            else {
                // Sort for percentile computation
                real colvector _ed_sorted
                _ed_sorted = sort(_ed_valid_dist, 1)

                // Store descriptive statistics
                st_numscalar("__ed_N_pairs", rows(_ed_valid_dist))
                st_numscalar("__ed_mean", mean(_ed_valid_dist))
                st_numscalar("__ed_sd", sqrt(variance(_ed_valid_dist)))
                st_numscalar("__ed_min", min(_ed_valid_dist))
                st_numscalar("__ed_max", max(_ed_valid_dist))
                st_numscalar("__ed_p25", ///
                    _trop_interpolate_percentile(_ed_sorted, 0.25))
                st_numscalar("__ed_p50", ///
                    _trop_interpolate_percentile(_ed_sorted, 0.50))
                st_numscalar("__ed_p75", ///
                    _trop_interpolate_percentile(_ed_sorted, 0.75))

                // Store distance vector as matrix for graphing
                st_matrix("__ed_distances", _ed_valid_dist')
            }
        }
    }

    // Check for computation errors
    if `_ed_rc' != 0 {
        exit `_ed_rc'
    }

    /* ──────────────────────────────────────────────────────────────────────
       3. Display results table
    ────────────────────────────────────────────────────────────────────── */

    di as txt _n "{hline 62}"
    di as txt "Unit Distance Distribution (Eq.3: RMSE over control periods)"
    di as txt "{hline 62}"
    di as txt "  Number of unit pairs: " as res %10.0f scalar(__ed_N_pairs)
    di as txt "  Mean distance:        " as res %10.4f scalar(__ed_mean)
    di as txt "  Std. deviation:       " as res %10.4f scalar(__ed_sd)
    di as txt "  Minimum:              " as res %10.4f scalar(__ed_min)
    di as txt "  25th percentile:      " as res %10.4f scalar(__ed_p25)
    di as txt "  Median:               " as res %10.4f scalar(__ed_p50)
    di as txt "  75th percentile:      " as res %10.4f scalar(__ed_p75)
    di as txt "  Maximum:              " as res %10.4f scalar(__ed_max)
    di as txt "{hline 62}"

    /* ──────────────────────────────────────────────────────────────────────
       4. Weight interpretation (if lambda_unit available)
    ────────────────────────────────────────────────────────────────────── */

    capture confirm scalar e(lambda_unit)
    if !_rc & !missing(e(lambda_unit)) {
        local lambda_u = e(lambda_unit)
        local w_median = exp(-`lambda_u' * scalar(__ed_p50))
        local w_p75    = exp(-`lambda_u' * scalar(__ed_p75))
        local pct_p75 : di %4.1f `w_p75' * 100

        di as txt _n "Weight mapping (lambda_unit = " as res %6.4f `lambda_u' as txt "):"
        di as txt "  w(median dist) = exp(-" %4.2f `lambda_u' ///
            " * " %6.4f scalar(__ed_p50) ") = " as res %8.6f `w_median'
        di as txt "  w(75th pctl)   = exp(-" %4.2f `lambda_u' ///
            " * " %6.4f scalar(__ed_p75) ") = " as res %8.6f `w_p75'
        di as txt "  {it:Units beyond 75th pctl contribute <`pct_p75'% weight}"
    }

    /* ──────────────────────────────────────────────────────────────────────
       5. Graphical output (publication quality)
    ────────────────────────────────────────────────────────────────────── */

    if "`graph'" != "" {
        local n_dist = colsof(__ed_distances)

        // Check lambda availability
        capture confirm scalar e(lambda_unit)
        local _has_lambda_u = (!_rc & !missing(e(lambda_unit)))
        capture confirm scalar e(lambda_time)
        local _has_lambda_t = (!_rc & !missing(e(lambda_time)))

        if `_has_lambda_u' {
            local lambda_unit = e(lambda_unit)
        }
        if `_has_lambda_t' {
            local lambda_time = e(lambda_time)
        }

        preserve
        clear
        quietly {
            set obs `n_dist'
            gen double distance = .
            forvalues i = 1/`n_dist' {
                replace distance = __ed_distances[1, `i'] in `i'
            }
        }

        // ── Figure 1: Distance distribution histogram ──────────────────────
        quietly {
            local med_line = scalar(__ed_p50)
            histogram distance, bins(`bins') ///
                color("24 105 175%60") lcolor("24 105 175") ///
                xtitle("Unit Distance (RMSE)", size(medsmall)) ///
                ytitle("Frequency", size(medsmall)) ///
                title("Distribution of Unit Distances", ///
                    size(medium) position(11)) ///
                xline(`med_line', lcolor(cranberry) lpattern(dash) ///
                    lwidth(medium)) ///
                note("Dashed line = median distance", size(vsmall)) ///
                graphregion(color(white)) plotregion(lcolor(none)) ///
                xsize(6) ysize(3.75) ///
                name(__trop_dist_hist, replace)
        }

        // ── Figure 2: Unit weights vs distance ─────────────────────────────
        if `_has_lambda_u' {
            quietly gen double weight = exp(-`lambda_unit' * distance)

            twoway (scatter weight distance, ///
                    mcolor("24 105 175%70") msymbol(o) msize(small)) ///
                (lowess weight distance, ///
                    lcolor(cranberry) lwidth(medium)), ///
                xtitle("Unit Distance (RMSE)", size(medsmall)) ///
                ytitle("Unit Weight {&omega}", size(medsmall)) ///
                title("Unit Weights vs Distance", ///
                    size(medium) position(11)) ///
                note("{&omega} = exp(-{&lambda}{sub:unit} {&times} d); " ///
                    "{&lambda}{sub:unit} = `lambda_unit'", size(vsmall)) ///
                legend(off) ///
                graphregion(color(white)) plotregion(lcolor(none)) ///
                xsize(6) ysize(3.75) ///
                name(__trop_dist_weight, replace)
        }

        // ── Figure 3: Time weight decay ────────────────────────────────────
        if `_has_lambda_t' {
            // Generate time horizon range from -N_periods/2 to N_periods/2
            local max_h = floor(`N_periods' / 2)
            local n_horizon = 2 * `max_h' + 1

            drop _all
            quietly {
                set obs `n_horizon'
                gen int horizon = _n - `max_h' - 1
                gen double time_weight = exp(-`lambda_time' * abs(horizon))
            }

            twoway (connected time_weight horizon, ///
                    lcolor("24 105 175") mcolor("24 105 175") ///
                    msymbol(O) msize(medsmall) lwidth(medium)), ///
                xtitle("Periods from Treatment", size(medsmall)) ///
                ytitle("Time Weight {&theta}", size(medsmall)) ///
                title("Time Weight Decay", ///
                    size(medium) position(11)) ///
                note("{&theta} = exp(-{&lambda}{sub:time} {&times} |t - t{sub:0}|); " ///
                    "{&lambda}{sub:time} = `lambda_time'", size(vsmall)) ///
                yline(0, lcolor(gs12) lwidth(vthin)) ///
                graphregion(color(white)) plotregion(lcolor(none)) ///
                xsize(6) ysize(3.75) ///
                name(__trop_time_weight, replace)
        }

        // ── Figure 4: Combined panel ───────────────────────────────────────
        local combine_list "__trop_dist_hist"
        if `_has_lambda_u' {
            local combine_list "`combine_list' __trop_dist_weight"
        }
        if `_has_lambda_t' {
            local combine_list "`combine_list' __trop_time_weight"
        }

        local n_graphs : word count `combine_list'
        if `n_graphs' > 1 {
            local cols = cond(`n_graphs' == 3, 2, `n_graphs')
            graph combine `combine_list', ///
                cols(`cols') ///
                title("TROP Weight Diagnostics", size(medium)) ///
                graphregion(color(white)) ///
                xsize(10) ysize(6) ///
                name(_trop_weight_diag, replace)

            // Clean up individual sub-graphs (combined panel kept)
            capture graph drop __trop_dist_hist
            capture graph drop __trop_dist_weight
            capture graph drop __trop_time_weight

            // Save combined graph if requested
            if `"`saving'"' != "" {
                graph export `"`saving'"', replace
            }
        }
        else {
            // Only histogram available: rename to canonical name
            graph rename __trop_dist_hist _trop_weight_diag, replace
            if `"`saving'"' != "" {
                graph export `"`saving'"', replace
            }
        }

        restore
    }

    /* ──────────────────────────────────────────────────────────────────────
       6. Store r() results and clean up
    ────────────────────────────────────────────────────────────────────── */

    return scalar N_pairs = scalar(__ed_N_pairs)
    return scalar mean    = scalar(__ed_mean)
    return scalar sd      = scalar(__ed_sd)
    return scalar min     = scalar(__ed_min)
    return scalar max     = scalar(__ed_max)
    return scalar p25     = scalar(__ed_p25)
    return scalar p50     = scalar(__ed_p50)
    return scalar p75     = scalar(__ed_p75)

    // Clean up temporary scalars and matrices
    capture scalar drop __ed_N_pairs
    capture scalar drop __ed_mean
    capture scalar drop __ed_sd
    capture scalar drop __ed_min
    capture scalar drop __ed_max
    capture scalar drop __ed_p25
    capture scalar drop __ed_p50
    capture scalar drop __ed_p75
    capture matrix drop __ed_distances
end

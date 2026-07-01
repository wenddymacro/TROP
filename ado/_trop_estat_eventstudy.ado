*! _trop_estat_eventstudy.ado
*! Event-study analysis for TROP twostep estimation
*!
*! Aggregates individual treatment effects tau_{i,t} by relative event time
*! (horizon h = t - g_i) and optionally performs pre-trend testing.
*!
*! Syntax:
*!   _trop_estat_eventstudy [, Window(numlist) REFerence(integer)
*!       PLACebo PLACebo_periods(integer) GRAPH noGRAPH
*!       Level(cilevel) TItle(string) SAVing(string)
*!       CONnect CIColor(string) MColor(string) MSYmbol(string) MSize(string)]

program define _trop_estat_eventstudy, rclass
    version 17.0

    syntax [, Window(numlist min=2 max=2) ///
        REFerence(integer -1) ///
        PLACebo PLACebo_periods(integer 3) ///
        GRAPH noGRAPH ///
        Level(cilevel) ///
        TItle(string) SAVing(string) ///
        CONnect ///
        CIColor(string) ///
        MColor(string) ///
        MSYmbol(string) ///
        MSize(string)]

    /* ──────────────────────────────────────────────────────────────────────
       1. Pre-checks
    ────────────────────────────────────────────────────────────────────── */

    // Check: must follow trop estimation
    if "`e(cmd)'" != "trop" {
        di as error "estat eventstudy requires trop estimation results"
        exit 301
    }

    // Check: only twostep method is supported
    if "`e(method)'" != "twostep" {
        di as error "estat eventstudy requires method(twostep)"
        di as error "  The joint method estimates a single scalar tau,"
        di as error "  which cannot be decomposed into event-time effects."
        exit 459
    }

    // Check: tau_matrix must exist
    capture confirm matrix e(tau_matrix)
    if _rc {
        di as error "e(tau_matrix) not found. Re-run trop with method(twostep)."
        exit 301
    }

    /* ──────────────────────────────────────────────────────────────────────
       2. Retrieve tau_matrix and reconstruct D_matrix, call Mata
    ────────────────────────────────────────────────────────────────────── */

    tempname tau_mat es_result

    matrix `tau_mat' = e(tau_matrix)

    local depvar   "`e(depvar)'"
    local panelvar "`e(panelvar)'"
    local timevar  "`e(timevar)'"
    local treatvar "`e(treatvar)'"

    // Rebuild 1..N / 1..T index variables (originals cleaned after trop)
    tempvar _es_pidx _es_tidx _es_touse_tmp
    qui gen byte `_es_touse_tmp' = e(sample)
    qui egen `_es_pidx' = group(`panelvar') if `_es_touse_tmp'
    qui egen `_es_tidx' = group(`timevar') if `_es_touse_tmp'
    // Pass tempvar names to Mata via globals
    mata: st_global("__trop_panel_idx_var", "`_es_pidx'")
    mata: st_global("__trop_time_idx_var", "`_es_tidx'")
    mata: st_global("__trop_touse_var", "`_es_touse_tmp'")

    // Call Mata aggregation
    mata: {
        real matrix _es_tau_m, _es_D_m, _es_result
        real scalar _es_nT, _es_nN, _es_nobs, _es_k
        real scalar _es_row_t, _es_col_i
        real matrix _es_obs_data
        string scalar _es_touse_var, _es_panel_idx_var, _es_time_idx_var

        _es_tau_m = st_matrix("`tau_mat'")
        _es_nT = rows(_es_tau_m)
        _es_nN = cols(_es_tau_m)

        /* Reconstruct D_matrix from data */
        _es_touse_var = st_global("__trop_touse_var")
        _es_panel_idx_var = st_global("__trop_panel_idx_var")
        _es_time_idx_var = st_global("__trop_time_idx_var")

        if (_es_touse_var != "" & _es_panel_idx_var != "" & _es_time_idx_var != "" & ///
            _st_varindex(_es_touse_var) < . & _st_varindex(_es_panel_idx_var) < . & ///
            _st_varindex(_es_time_idx_var) < . & _st_varindex("`treatvar'") < .) {

            _es_D_m = J(_es_nT, _es_nN, 0)
            _es_obs_data = st_data(., ("`treatvar'", _es_panel_idx_var, _es_time_idx_var), _es_touse_var)
            _es_nobs = rows(_es_obs_data)
            for (_es_k = 1; _es_k <= _es_nobs; _es_k++) {
                _es_row_t = _es_obs_data[_es_k, 3]
                _es_col_i = _es_obs_data[_es_k, 2]
                if (_es_row_t >= 1 & _es_row_t <= _es_nT & _es_col_i >= 1 & _es_col_i <= _es_nN) {
                    _es_D_m[_es_row_t, _es_col_i] = (_es_obs_data[_es_k, 1] != 0 ? 1 : 0)
                }
            }
        }
        else {
            /* Fallback: infer D from non-missing tau values */
            _es_D_m = (_es_tau_m :< .)
        }

        _es_result = _trop_aggregate_by_horizon(_es_tau_m, _es_D_m, `level')
        st_matrix("__es_result", _es_result)
    }

    // Verify result exists
    capture confirm matrix __es_result
    if _rc {
        di as error "Event-study aggregation produced no results."
        exit 459
    }

    local nrows = rowsof(__es_result)
    if `nrows' == 0 {
        di as error "No valid horizons found for event-study aggregation."
        exit 459
    }

    /* ──────────────────────────────────────────────────────────────────────
       3. Window filtering
    ────────────────────────────────────────────────────────────────────── */

    if "`window'" != "" {
        local wlow : word 1 of `window'
        local whigh : word 2 of `window'

        tempname es_filtered
        local nkeep = 0

        forvalues r = 1/`nrows' {
            local h_val = __es_result[`r', 1]
            if `h_val' >= `wlow' & `h_val' <= `whigh' {
                local nkeep = `nkeep' + 1
            }
        }

        if `nkeep' == 0 {
            di as error "No horizons fall within window(`wlow' `whigh')"
            exit 459
        }

        mata: {
            real matrix _es_filtered, _es_orig
            real scalar _es_r, _es_wlow, _es_whigh, _es_cnt

            _es_orig = st_matrix("__es_result")
            _es_wlow = `wlow'
            _es_whigh = `whigh'
            _es_filtered = J(`nkeep', 6, .)
            _es_cnt = 0

            for (_es_r = 1; _es_r <= rows(_es_orig); _es_r++) {
                if (_es_orig[_es_r, 1] >= _es_wlow & _es_orig[_es_r, 1] <= _es_whigh) {
                    _es_cnt++
                    _es_filtered[_es_cnt, .] = _es_orig[_es_r, .]
                }
            }

            st_matrix("__es_result", _es_filtered)
        }

        local nrows = `nkeep'
    }

    /* ──────────────────────────────────────────────────────────────────────
       4. Placebo / Pre-trend test (optional)
    ────────────────────────────────────────────────────────────────────── */

    if "`placebo'" != "" {
        mata: {
            real matrix _es_placebo_result
            real rowvector _es_pretrend

            _es_placebo_result = _trop_placebo_effects("`depvar'", "`panelvar'", ///
                "`timevar'", "`treatvar'", `placebo_periods', `level')
            st_matrix("__es_placebo", _es_placebo_result)

            _es_pretrend = _trop_pretrend_test(_es_placebo_result)
            st_numscalar("__es_chi2", _es_pretrend[1])
            st_numscalar("__es_df", _es_pretrend[2])
            st_numscalar("__es_pval", _es_pretrend[3])
        }
    }

    /* ──────────────────────────────────────────────────────────────────────
       5. Display results
    ────────────────────────────────────────────────────────────────────── */

    di as txt _n "{hline 70}"
    di as txt "TROP Event Study"
    di as txt "{hline 70}"
    di as txt ""
    di as txt "  Method: twostep (Algorithm 2, heterogeneous tau_it)"
    di as txt "  Aggregation: mean effect by event-time horizon h = t - g_i"
    di as txt ""

    // Table header
    di as txt "{hline 70}"
    di as txt %10s "Horizon" _col(15) %12s "Effect" _col(30) %10s "Std.Err." ///
        _col(43) %12s "[`level'% CI]" _col(63) %6s "N_cells"
    di as txt "{hline 70}"

    // Table rows
    forvalues r = 1/`nrows' {
        local h_val = __es_result[`r', 1]
        local eff   = __es_result[`r', 2]
        local se    = __es_result[`r', 3]
        local ci_l  = __es_result[`r', 4]
        local ci_u  = __es_result[`r', 5]
        local nc    = __es_result[`r', 6]

        // Mark reference period
        local marker = ""
        if `h_val' == `reference' {
            local marker " (ref)"
        }

        di as txt %10.0f `h_val' "`marker'" _col(15) as res %12.6f `eff' ///
            _col(30) %10.6f `se' _col(43) "[" %8.4f `ci_l' ", " %8.4f `ci_u' "]" ///
            _col(63) %6.0f `nc'
    }
    di as txt "{hline 70}"

    // Pre-trend test output
    if "`placebo'" != "" {
        di as txt _n "Pre-trend test (H0: all pre-treatment effects = 0):"
        di as txt "  Chi2(" %3.0f scalar(__es_df) ") = " as res %8.4f scalar(__es_chi2)
        di as txt "  p-value = " as res %8.4f scalar(__es_pval)
        if scalar(__es_pval) > 0.05 {
            di as txt "  {it:Cannot reject parallel trends (p > 0.05)}"
        }
        else {
            di as txt "  {err}Warning: Pre-trends detected (p <= 0.05)"
        }
    }

    /* ──────────────────────────────────────────────────────────────────────
       6. Graph output (default on; nograph turns off)
       Publication-quality figure targeting AER/QJE/Econometrica standards.
    ────────────────────────────────────────────────────────────────────── */

    if "`graph'" != "nograph" {
        // ── 6a. Set publication-quality defaults (user options override) ──
        local _cicolor = cond("`cicolor'" != "", "`cicolor'", `""24 105 175""')
        local _mcolor  = cond("`mcolor'"  != "", "`mcolor'",  `""24 105 175""')
        local _msymbol = cond("`msymbol'" != "", "`msymbol'", "O")
        local _msize   = cond("`msize'"   != "", "`msize'",   "medsmall")
        local _title   = cond(`"`title'"' != "", `"`title'"', "Event Study")

        preserve
        clear
        quietly {
            local nrows_g = rowsof(__es_result)
            set obs `nrows_g'
            gen double horizon = .
            gen double effect = .
            gen double ci_lower = .
            gen double ci_upper = .
            forvalues r = 1/`nrows_g' {
                replace horizon  = __es_result[`r', 1] in `r'
                replace effect   = __es_result[`r', 2] in `r'
                replace ci_lower = __es_result[`r', 4] in `r'
                replace ci_upper = __es_result[`r', 5] in `r'
            }
        }

        // ── 6b. Determine reference line position ──
        // Use treatment onset reference for vertical line
        local xref = `reference'

        // ── 6c. Build twoway plot layers ──
        local plot_layers ""

        // Layer 1: Confidence interval error bars (rcap)
        local plot_layers `"`plot_layers' (rcap ci_upper ci_lower horizon, lcolor(`_cicolor') lwidth(medthin))"'

        // Layer 2: Point estimates (scatter)
        local plot_layers `"`plot_layers' (scatter effect horizon, mcolor(`_mcolor') msymbol(`_msymbol') msize(`_msize'))"'

        // Layer 3: Connecting line (always drawn for visual continuity)
        local plot_layers `"`plot_layers' (line effect horizon, lcolor(`_mcolor') lwidth(thin) lpattern(solid))"'

        // ── 6d. Construct the publication-quality twoway command ──
        twoway `plot_layers' ///
            , ///
            yline(0, lcolor(gs10) lpattern(shortdash) lwidth(thin)) ///
            xline(`xref', lcolor(gs10) lpattern(dash) lwidth(thin)) ///
            xtitle("Periods Relative to Treatment", size(medsmall)) ///
            ytitle("Treatment Effect Estimate", size(medsmall)) ///
            title(`"`_title'"', size(medium) position(11)) ///
            xlabel(, labsize(small) grid glcolor(gs15) glwidth(vthin) glpattern(solid)) ///
            ylabel(, labsize(small) angle(horizontal) grid glcolor(gs15) glwidth(vthin) glpattern(solid)) ///
            legend(off) ///
            plotregion(lcolor(none) margin(small)) ///
            graphregion(color(white) margin(medsmall)) ///
            xsize(6) ysize(3.75) ///
            note("Note: `level'% confidence intervals shown." ///
                 "Vertical dashed line indicates treatment onset.", ///
                 size(vsmall) position(7)) ///
            scheme(s2color) ///
            name(_trop_eventstudy, replace)

        // ── 6e. High-resolution export ──
        if `"`saving'"' != "" {
            // Detect file extension for format-appropriate export
            local _savefile `"`saving'"'
            local _ext = lower(substr(`"`_savefile'"', -4, 4))
            if "`_ext'" == ".pdf" | "`_ext'" == ".eps" {
                graph export `"`_savefile'"', replace
            }
            else if "`_ext'" == ".png" {
                graph export `"`_savefile'"', replace width(2400)
            }
            else if "`_ext'" == ".gph" {
                graph save `"`_savefile'"', replace
            }
            else {
                // Default: try export (supports .pdf/.png/.tif etc.)
                graph export `"`_savefile'"', replace
            }
        }

        restore
    }

    /* ──────────────────────────────────────────────────────────────────────
       7. Store r() results
    ────────────────────────────────────────────────────────────────────── */

    return matrix event_effects = __es_result

    if "`placebo'" != "" {
        capture confirm matrix __es_placebo
        if !_rc {
            return scalar chi2_pretrend = scalar(__es_chi2)
            return scalar df_pretrend   = scalar(__es_df)
            return scalar p_pretrend    = scalar(__es_pval)
            return matrix placebo_effects = __es_placebo
        }
    }

    // Clean up temporary scalars/matrices
    capture matrix drop __es_result
    capture matrix drop __es_placebo
    capture scalar drop __es_chi2
    capture scalar drop __es_df
    capture scalar drop __es_pval
end

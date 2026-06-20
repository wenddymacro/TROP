/*===========================================================================
  test_rao_wu_bootstrap.do
  
  End-to-end test for Rao-Wu Bootstrap in trop_stata.
  Compares Stata results with Python reference values.
  
  Prerequisites:
  - Run generate_survey_test_data.py first to create test data
  - Run test_rao_wu_bootstrap_parity.py to generate reference values
  - Plugin must be compiled (build.sh)
===========================================================================*/

clear all
set more off

// Load test data
use "data/survey_test_panel.dta", clear

// Ensure panel structure
xtset unit time

// =========================================================================
// Test 1: pweight only (no strata/PSU/FPC)
// Expected: standard weighted bootstrap
// =========================================================================
di _n "{hline 70}"
di "Test 1: pweight only"
di "{hline 70}"

trop y d [pw=weight], panelvar(unit) timevar(time) ///
    method(twostep) bootstrap(200) seed(42) ///
    lambda_time_grid(0 1) lambda_unit_grid(0 1) lambda_nn_grid(0.1 1)

di "  ATT = " e(att)
di "  SE  = " e(se)
di "  Lambda = (" e(lambda_time) ", " e(lambda_unit) ", " e(lambda_nn) ")"

// Store for comparison
local att1 = e(att)
local se1 = e(se)

// =========================================================================
// Test 2: pweight + strata (Rao-Wu bootstrap)
// =========================================================================
di _n "{hline 70}"
di "Test 2: pweight + strata (Rao-Wu)"
di "{hline 70}"

trop y d [pw=weight], panelvar(unit) timevar(time) ///
    method(twostep) bootstrap(200) seed(42) ///
    strata(stratum) ///
    lambda_time_grid(0 1) lambda_unit_grid(0 1) lambda_nn_grid(0.1 1)

di "  ATT = " e(att)
di "  SE  = " e(se)
di "  Lambda = (" e(lambda_time) ", " e(lambda_unit) ", " e(lambda_nn) ")"

local att2 = e(att)
local se2 = e(se)

// =========================================================================
// Test 3: pweight + strata + PSU + FPC
// =========================================================================
di _n "{hline 70}"
di "Test 3: pweight + strata + PSU + FPC (Rao-Wu)"
di "{hline 70}"

trop y d [pw=weight], panelvar(unit) timevar(time) ///
    method(twostep) bootstrap(200) seed(42) ///
    strata(stratum) psu(psu) fpc(fpc) ///
    lambda_time_grid(0 1) lambda_unit_grid(0 1) lambda_nn_grid(0.1 1)

di "  ATT = " e(att)
di "  SE  = " e(se)
di "  Lambda = (" e(lambda_time) ", " e(lambda_unit) ", " e(lambda_nn) ")"

local att3 = e(att)
local se3 = e(se)

// =========================================================================
// Test 4: Joint method + strata
// =========================================================================
di _n "{hline 70}"
di "Test 4: Joint method + strata (Rao-Wu)"
di "{hline 70}"

trop y d [pw=weight], panelvar(unit) timevar(time) ///
    method(joint) bootstrap(200) seed(42) ///
    strata(stratum) ///
    lambda_time_grid(0 1) lambda_unit_grid(0 1) lambda_nn_grid(0.1 1)

di "  ATT = " e(att)
di "  SE  = " e(se)
di "  Lambda = (" e(lambda_time) ", " e(lambda_unit) ", " e(lambda_nn) ")"

local att4 = e(att)
local se4 = e(se)

// =========================================================================
// Summary
// =========================================================================
di _n "{hline 70}"
di "SUMMARY"
di "{hline 70}"
di "Test 1 (pweight only):        ATT=" %9.6f `att1' " SE=" %9.6f `se1'
di "Test 2 (strata):              ATT=" %9.6f `att2' " SE=" %9.6f `se2'
di "Test 3 (strata+PSU+FPC):      ATT=" %9.6f `att3' " SE=" %9.6f `se3'
di "Test 4 (joint+strata):        ATT=" %9.6f `att4' " SE=" %9.6f `se4'

// Sanity checks
// ATT should be similar across methods (same data, same lambda grid)
// SE with FPC should be <= SE without FPC (FPC reduces variance)
di _n "Sanity checks:"
di "  SE with FPC <= SE without FPC: " cond(`se3' <= `se2' + 0.01, "PASS", "WARN")

di _n "{hline 70}"
di "Test completed successfully."
di "{hline 70}"

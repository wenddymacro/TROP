*! TROP Mata Library Builder
*! Version: 1.0
*! Date: 2026-02-12
*! Purpose: Compile all TROP Mata source files into ltrop.mlib for distribution
*!
*! Usage:
*!   From trop_stata/ directory:
*!     stata-mp -b do build_mlib.do
*!   Or from project root:
*!     stata-mp -b do trop_stata/build_mlib.do
*!
*! Output:
*!   ltrop.mlib in the trop_stata/ directory
*!   This file should be listed in trop.pkg for distribution.

clear all
set more off

// ============================================================
// Step 1: Locate Mata source files
// ============================================================
// Note: build.sh uses a wrapper do-file that cd's to trop_stata/ before
// calling this file, so c(pwd) should be correct. If profile.do interferes,
// the wrapper's cd overrides it.
local mata_dir ""
local base_dir ""

// Try: current directory is trop_stata/
if "`mata_dir'" == "" {
    capture confirm file "mata/trop_constants.mata"
    if _rc == 0 {
        local mata_dir "mata"
        local base_dir "."
    }
}

// Fallback: Try current directory is project root
if "`mata_dir'" == "" {
    capture confirm file "trop_stata/mata/trop_constants.mata"
    if _rc == 0 {
        local mata_dir "trop_stata/mata"
        local base_dir "trop_stata"
    }
}

// Fallback: Try current directory is trop_stata/mata/
if "`mata_dir'" == "" {
    capture confirm file "trop_constants.mata"
    if _rc == 0 {
        local mata_dir "."
        local base_dir ".."
    }
}

if "`mata_dir'" == "" {
    display as error "Error: Cannot find Mata source files."
    display as error "Run from trop_stata/ or project root directory."
    display as error "c(pwd) = `c(pwd)'"
    display as error "c(filename) = `c(filename)'"
    exit 601
}

display as text ""
display as text "========================================"
display as text "Building ltrop.mlib"
display as text "========================================"
display as text "Mata source directory: `mata_dir'"
display as text ""

// ============================================================
// Step 2: Compile all Mata source files (dependency order)
// ============================================================
// This matches the order in load_mata_once.do and compile_all.do

// File list must match load_mata_once.do / compile_all.do exactly.
local mata_files ///
    "trop_constants.mata" ///
    "trop_rust_interface.mata" ///
    "trop_data_transfer.mata" ///
    "trop_lambda_grid.mata" ///
    "trop_backend_select.mata" ///
    "trop_ereturn_store.mata" ///
    "trop_validation.mata" ///
    "trop_loocv_validation.mata" ///
    "trop_bootstrap_diagnostics.mata" ///
    "trop_estat_helpers.mata" ///
    "trop_eventstudy.mata" ///
    "trop_main.mata"

local error_count = 0
local total = 0

foreach file of local mata_files {
    local total = `total' + 1
    display as text "  Compiling: `file'"
    capture noisily do "`mata_dir'/`file'"
    if _rc != 0 {
        local error_count = `error_count' + 1
        display as error "    FAILED: `file'"
    }
}

if `error_count' > 0 {
    display as error ""
    display as error "Compilation failed for `error_count' file(s)."
    display as error "Cannot create ltrop.mlib."
    exit 198
}

display as text ""
display as text "All `total' Mata files compiled successfully."

// ============================================================
// Step 3: Create ltrop.mlib
// ============================================================
// The mlib file packages all compiled Mata functions into a single
// library that Stata auto-loads from the adopath.
// After `net install`, ltrop.mlib goes to c(sysdir_plus)/l/
// and all TROP Mata functions become available automatically.

display as text ""
display as text "Creating ltrop.mlib..."

// Create the mlib in the base directory (trop_stata/)
mata: mata mlib create ltrop, dir("`base_dir'") replace

// Add all TROP functions using naming convention patterns
// Functions follow these naming patterns:
//   trop_*()     - public API functions
//   _trop_*()    - internal helper functions
//   store_loocv_diagnostics(), display_loocv_verbose(), check_loocv_fail_rate()
//   validate_lambda_vs_table2(), get_table2_lambda(), get_table2_rmse()
//   select_backend(), check_backend(), print_backend_info()
//   _compute_*(), _diagnose_*(), _shapiro_wilk_test()
//   Seed management: GLOBAL_RANDOM_SEED(), set_GLOBAL_RANDOM_SEED()

// Add all functions - after clear all + compile, only TROP functions exist
mata: mata mlib add ltrop *(), complete

display as text ""
display as text "========================================"
display as result "ltrop.mlib created successfully"
display as text "Location: `base_dir'/ltrop.mlib"
display as text "Contains all TROP Mata functions for distribution"
display as text "========================================"

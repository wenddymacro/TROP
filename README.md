# trop

**Triply Robust Panel Estimator for Stata**

[![Stata 17+](https://img.shields.io/badge/Stata-17%2B-blue.svg)](https://www.stata.com/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)
[![Version: 1.2.0](https://img.shields.io/badge/Version-1.2.0-green.svg)]()
![Platforms](https://img.shields.io/badge/platforms-macOS%20|%20Linux%20|%20Windows-blue)

![trop](image/image.png)

## Overview

`trop` implements the **Triply RObust Panel (TROP) estimator** proposed by Athey, Imbens, Qu, and Viviano (2025) for Stata. The estimator combines three components — **unit weights**, **time weights**, and a **nuclear-norm-regularized low-rank regression adjustment** — to estimate average treatment effects on the treated (ATT) in panel data with potentially complex assignment patterns.

In semi-synthetic simulations calibrated to seven real datasets (Table 1 of the paper), TROP achieves the **lowest RMSE in 20 out of 21 specifications**, outperforming DID, SC, SDID, MC, and DIFP estimators across diverse data-generating processes:

| Data set | Outcome | Treatment | N | T | TROP | SDID | SC | DID | MC | DIFP |
|----------|---------|-----------|---|---|------|------|----|-----|-----|------|
| CPS | log-wage | min wage | 50 | 40 | **1.00** | 1.14 | 1.44 | 1.91 | 1.26 | 1.22 |
| CPS | urate | min wage | 50 | 40 | **1.00** | 1.05 | 1.11 | 1.89 | 1.10 | 1.09 |
| PWT | log-GDP | democracy | 111 | 48 | **1.00** | 1.44 | 1.59 | 7.85 | 1.76 | 1.54 |
| Germany | GDP | random | 17 | 44 | **1.00** | 1.46 | 2.82 | 3.58 | 1.56 | 2.46 |
| Basque | GDP | random | 18 | 43 | **1.00** | 1.02 | 4.55 | 9.11 | 1.70 | 2.47 |
| Smoking | packs pc | random | 39 | 31 | **1.00** | 1.22 | 1.48 | 2.16 | 1.14 | 1.45 |
| Boatlift | log-wage | random | 44 | 19 | **1.00** | 1.34 | 1.62 | 1.35 | 1.04 | 1.62 |

<sub>Normalized RMSE from Table 1 of Athey et al. (2025). Full results cover 21 specifications across 7 datasets.</sub>

**Features:**

- **Triple Robust Estimation** — Asymptotically unbiased if *any one of* unit weights, time weights, or the regression adjustment removes biases (Theorem 5.1)
- **Two Estimation Methods** — Twostep (per-observation, heterogeneous effects; Algorithm 2) and Joint (weighted least squares, homogeneous effect)
- **Leave-One-Out Cross-Validation** — Data-driven selection of tuning parameters via coordinate-cycling LOOCV (Algorithm 1)
- **Bootstrap Inference** — Stratified unit block bootstrap for variance estimation and confidence intervals (Algorithm 3)
- **General Assignment Patterns** — Handles staggered adoption, switching treatments, and arbitrary binary treatment matrices
- **Post-Estimation Diagnostics** — 13 `estat` subcommands (including triple-robustness bias decomposition, event-study, pre-trend test, and table export) and 11 `predict` types for comprehensive analysis
- **Covariate Adjustment** — Time-invariant covariates X_i'γ (paper Section 6.2 Eq. 14) with automatic WLS projection
- **Survey Design Support** — Stratification, PSU clustering, and FPC via Rao-Wu rescaled bootstrap
- **High-Performance Backend** — Core computation in Rust via compiled plugin; no external dependencies

## Key Concepts

### Triple Robustness

The TROP estimator combines three components, each targeting a different source of confounding:

| Component | Role | Controlled by |
|-----------|------|--------------|
| **Unit weights** $\omega_j$ | Upweight control units similar to treated units | $\lambda_{\text{unit}}$ |
| **Time weights** $\theta_s$ | Upweight time periods close to the treatment period | $\lambda_{\text{time}}$ |
| **Low-rank factor model** $\mathbf{L}$ | Capture unobserved interactive fixed effects | $\lambda_{nn}$ |

The key insight is that the estimator's bias is bounded by the **product** of three imbalance terms (Theorem 5.1). If any single component successfully removes bias, the overall bias vanishes — hence *triple* robustness. This multiplicative bound is strictly tighter than the additive bounds of DID, SC, or SDID.

### Special Cases

The TROP framework nests existing estimators:

| Setting | Recovers |
|---------|----------|
| $\lambda_{nn} = \infty$, uniform weights | **DID / TWFE** |
| Uniform weights, $\lambda_{nn} < \infty$ | **Matrix Completion** |
| $\lambda_{nn} = \infty$, specific unit/time weights | **SC / SDID** |

## Requirements

- Stata 17.0 or later
- No additional Stata packages required
- Precompiled plugins included for all major platforms (see below)

## Installation

### Supported Platforms

| Platform | Plugin File | Status |
|----------|-------------|--------|
| macOS Apple Silicon (ARM64) | `trop_macos_arm64.plugin` | ✅ Precompiled |
| macOS Intel (x86-64) | `trop_macos_x64.plugin` | ✅ Precompiled |
| Windows x86-64 | `trop_windows_x64.plugin` | ✅ Precompiled |
| Linux x86-64 | `trop_linux_x64.plugin` | ✅ Precompiled |

### Option A: Install from GitHub (recommended)

```stata
net install trop, from("https://raw.githubusercontent.com/gorgeousfish/TROP/main") replace
```

This automatically installs:
- All commands and help files
- Pre-compiled Mata library
- Platform-specific plugin for your system

### Option B: Local Installation

If you have downloaded or cloned the repository:

```stata
net install trop, from("/path/to/TROP") replace
```

### Verify Installation

```stata
trop, version
trop_check
```

## Quick Start with Examples

```stata
* Install
net install trop, from("https://raw.githubusercontent.com/gorgeousfish/TROP/main") replace

* Load example data
trop_data cps_logwage

* Estimate
trop y d, panelvar(id) timevar(t) method(twostep) fixedlambda(0.5 0 0.01)
```

All examples below use the **CPS log-wage dataset** — 50 US states × 40 years (1979–2018) of state-level log wages, where `d` flags state-years in which a minimum wage increase was in effect. This is one of the seven benchmark datasets from Athey et al. (2025).

**Dataset:** N = 50, T = 40, 2,000 observations. `y` = log state-level wage; `d` = minimum wage treatment (8 treated state-year cells, 0.4%); `id` = state identifier; `t` = year.

### Example 1: Fixed Hyperparameters (Twostep)

Using the paper's recommended values for CPS log-wage (Table S.1 of Athey et al. 2025):

```stata
trop_data cps_logwage
trop y d, panelvar(id) timevar(t) fixedlambda(0.1 0 0.9)
```

Output:

```
------------------------------------------------------------------------------
TROP Estimation Results
------------------------------------------------------------------------------
Method:            twostep
Grid style:        default (17 grid points/cycle, coordinate descent)

Panel dimensions:  N = 50, T = 40
Observations:      2000
Treated:           8 ( 0.4%)

Fixed hyperparameters (LOOCV skipped):
  lambda_time =   0.1000
  lambda_unit =   0.0000
  lambda_nn   =   0.9000

Treatment Effect (ATT):
  tau     =     0.031406
```

### Example 2: LOOCV-Selected Hyperparameters

```stata
trop_data cps_logwage
trop y d, panelvar(id) timevar(t)
```

Output:

```
------------------------------------------------------------------------------
TROP Estimation Results
------------------------------------------------------------------------------
Method:            twostep
Grid style:        default (17 grid points/cycle, coordinate descent)

Panel dimensions:  N = 50, T = 40
Observations:      2000
Treated:           8 ( 0.4%)

Selected hyperparameters (via LOOCV):
  lambda_time =   0.5000
  lambda_unit =   5.0000
  lambda_nn   =   1.0000
  Q(lambda_hat) =   3.471727

Treatment Effect (ATT):
  tau     =     0.034188
```

**Note:** LOOCV on N = 50, T = 40 sums over every D = 0 cell (paper Eq. 5)
and can take 20–40 minutes. Use `fixedlambda()` when a full grid search is
not needed.

### Example 3: Bootstrap Inference

```stata
trop_data cps_logwage
trop y d, panelvar(id) timevar(t) ///
    fixedlambda(0.1 0 0.9) bootstrap(200) seed(42)
```

Output:

```
Treatment Effect (ATT):
  tau     =     0.031406
  SE     =     0.015716
  t      =       1.9984
  p-value=       0.0858
  95% CI = [   -0.005756,     0.068568]
```

### Example 4: Joint Method

```stata
trop_data cps_logwage
trop y d, panelvar(id) timevar(t) ///
    method(joint) fixedlambda(0.1 0 0.9)
```

Output:

```
Treatment Effect (ATT):
  tau     =     0.031406

Global intercept:
  mu     =     5.154320
```

### Example 5: Post-Estimation Workflow

```stata
trop_data cps_logwage
trop y d, panelvar(id) timevar(t) fixedlambda(0.1 0 0.9)

* Predict counterfactual outcomes and treatment effects
predict y0_hat, y0
predict te_hat, te

* Diagnostics
estat summarize
estat weights
estat factors
```

`estat summarize` output:

```
-----------------------------------------------------------------
Estimation sample summary
-----------------------------------------------------------------
  Number of observations:        2000    (balanced panel)
  Number of units (N):             50
  Number of periods (T):           40
  Missing rate:                   0.0%

Treatment structure:
  Treated observations:             8    (  0.4%)
  Control observations:          1992    ( 99.6%)
  Treated units:                    8    ( 16.0%)
  Treated periods:                  1    (  2.5%)
  Pattern:                   multiple_treated_simultaneous
-----------------------------------------------------------------
```

To also inspect LOOCV diagnostics, run without `fixedlambda()`:

```stata
trop_data cps_logwage
trop y d, panelvar(id) timevar(t)
estat loocv
```

### Example 6: Standalone Bootstrap (Post-Estimation)

```stata
trop_data cps_logwage

* Estimate without bootstrap first (faster iteration)
trop y d, panelvar(id) timevar(t) fixedlambda(0.1 0 0.9)

* Then add bootstrap inference separately
trop_bootstrap, nreps(200) seed(42)
```

Output:

```
------------------------------------------------------------
TROP Bootstrap Inference Results
------------------------------------------------------------
ATT estimate:       0.031406
Bootstrap SE:       0.015716
95% CI:       [   -0.005756,     0.068568]
p-value:              0.0858

Bootstrap reps:        200
Valid reps:            200
------------------------------------------------------------
```

### Example 7: PWT Log-GDP Panel (111 Countries × 48 Years)

For a large panel, use the Penn World Tables democracy dataset:

```stata
trop_data pwt_loggdp

* Paper's hyperparameters for PWT. Large panels with very small lambda_nn
* may require many iterations; use maxiter(1000) for tighter convergence.
trop y d, panelvar(id) timevar(t) fixedlambda(0.4 0.3 0.006) maxiter(1000)
```

Output:

```
------------------------------------------------------------------------------
TROP Estimation Results
------------------------------------------------------------------------------
Method:            twostep
Grid style:        default (17 grid points/cycle, coordinate descent)

Panel dimensions:  N = 111, T = 48
Observations:      5328
Treated:           29 ( 0.5%)

Fixed hyperparameters (LOOCV skipped):
  lambda_time =   0.4000
  lambda_unit =   0.3000
  lambda_nn   =   0.0060

Treatment Effect (ATT):
  tau     =    -0.014818
```

> **Convergence note.** With a very small `lambda_nn` (e.g. 0.006), the low-rank
> factor matrix L can have many nonzero singular values, and the alternating
> minimization progresses slowly on large panels. The strict `tol(1e-6)` default
> may not be reached within `maxiter(1000)`; the point estimate is nevertheless
> stable to the third decimal place. For faster and fully-converged estimates,
> either (i) increase `lambda_nn` (e.g. `fixedlambda(0.4 0.3 0.1)` converges in
> about 110 iterations to τ = -0.006672) or (ii) relax the tolerance via
> `tol(1e-4)`. The ATT is stable to `|Δτ| < 4e-7` against the released numerical baseline.

**Available datasets** (load via `trop_data`):

| File | Description | N | T |
|------|-------------|---|---|
| `cps_logwage.dta` | CPS state-level log wage (min wage treatment) | 50 | 40 |
| `cps_urate.dta` | CPS state-level unemployment rate (min wage treatment) | 50 | 40 |
| `pwt_loggdp.dta` | Penn World Tables log GDP (democracy transition) | 111 | 48 |
| `germany_gdp.dta` | Abadie & Gardeazabal (2003) West Germany GDP | 17 | 44 |
| `basque_gdp.dta` | Abadie (2003) Basque Country GDP | 18 | 43 |
| `smoking_packs.dta` | California Prop 99 cigarette consumption | 39 | 31 |

> `germany_gdp`, `basque_gdp`, and `smoking_packs` have `d = 0` throughout — they are raw outcome panels designed for semi-synthetic simulation (as in Table 1 of Athey et al. 2025). Assign a treatment indicator before running `trop`.

## Recommended Workflow

```
Step 1: Validate     →  trop_validate depvar treatvar, panelvar() timevar()
Step 2: Estimate     →  trop depvar treatvar, panelvar() timevar()
Step 3: Diagnose     →  estat summarize / estat weights / estat loocv / estat factors
Step 4: Inference    →  trop_bootstrap, nreps(1000)    — or —    bootstrap() in Step 2
```

**Step 1 — Validate data structure.** `trop_validate` checks panel balance, treatment patterns, missing data, and minimum sample requirements before estimation. This step is also performed automatically inside `trop`.

```stata
trop_validate y d, panelvar(id) timevar(t)
```

**Step 2 — Estimate ATT.** Choose `method(twostep)` (default) for heterogeneous effects, or `method(joint)` for faster homogeneous estimation. LOOCV selects hyperparameters automatically.

| Method | Use when | Speed |
|--------|----------|-------|
| `twostep` | Heterogeneous effects; general assignment | Slower |
| `joint` | Homogeneous effect; simultaneous adoption only | Faster |

**Step 3 — Diagnose results.**

| Subcommand | What to check |
|------------|---------------|
| `estat summarize` | Panel dimensions, treatment pattern, balance |
| `estat weights` | Whether weights concentrate on few units/periods |
| `estat loocv` | Convergence, failure rate, selected lambdas |
| `estat factors` | Effective rank, explained variance by top singular values |

**Step 4 — Conduct inference.** Use `bootstrap()` inline or `trop_bootstrap` post-estimation. The bootstrap resamples units within treatment strata (Algorithm 3).

```stata
* Inline (re-runs estimation + bootstrap together)
trop y d, panelvar(id) timevar(t) bootstrap(1000) seed(42)

* Post-estimation (uses stored estimation results)
trop_bootstrap, nreps(1000) seed(42)
```

## Tutorial

An interactive Jupyter notebook tutorial is included as ancillary material. After installation, copy the tutorial notebook to your working directory:

```stata
trop_data 10_trop_stata, type(ancillary)
```

The file `10_trop_stata.ipynb` covers data generation, estimation with both methods, all `estat` diagnostics, prediction, and a real-data CPS example.

## Commands

| Command          | Description                                       |
| ---------------- | ------------------------------------------------- |
| `trop`           | Main estimation command (twostep or joint method)  |
| `trop_bootstrap` | Standalone bootstrap inference (post-estimation)   |
| `trop_validate`  | Panel data structure validation                    |
| `trop_check`     | Environment and installation check                 |

Post-estimation (available after `trop`):

| Command          | Description                                       |
| ---------------- | ------------------------------------------------- |
| `estat`          | Diagnostics dispatcher (13 subcommands; see below) |
| `predict`        | Prediction dispatcher (11 types; see below)       |

## Options

### trop Options

**Required:**

| Option              | Description                      |
| ------------------- | -------------------------------- |
| `panelvar(varname)` | Unit (panel) identifier variable |
| `timevar(varname)`  | Time identifier variable         |

**Optional:**

| Option                       | Description                                                     | Default    |
| ---------------------------- | --------------------------------------------------------------- | ---------- |
| `method(string)`             | Estimation method: `twostep` / `joint` (or aliases `local` / `global`) | `twostep`  |
| `grid_style(string)`         | Lambda grid style: `default`, `fine`, or `extended`             | `default`  |
| `lambda_time_grid(numlist)`  | User-specified grid for lambda_time                             | auto       |
| `lambda_unit_grid(numlist)`  | User-specified grid for lambda_unit                             | auto       |
| `lambda_nn_grid(numlist)`    | User-specified grid for lambda_nn                               | auto       |
| `fixedlambda(numlist)`       | Fix (lambda_time lambda_unit lambda_nn); skip LOOCV             | —          |
| `tol(real)`                  | Convergence tolerance for iterative estimation                  | `1e-6`     |
| `maxiter(integer)`           | Maximum number of iterations                                    | `500`      |
| `bootstrap(integer)`         | Number of bootstrap replications (0 = skip inference)           | `200`      |
| `bsvariance(string)`         | Bootstrap variance denominator: `sample` (1/(B-1)) or `paper` (1/B, Alg 3) | `sample`   |
| `cimethod(string)`           | Primary confidence interval: `percentile` (Alg 3 step 6), `t`, or `normal` | `percentile` if `bootstrap > 0`, else `t` |
| `seed(integer)`              | Random number generator seed                                    | `42`       |
| `level(cilevel)`             | Confidence level for intervals                                  | `c(level)` |
| `verbose`                    | Display detailed diagnostic output                              | off        |

**Grid styles:**
- `default` — 6 × 6 × 5 = 180 grid combinations, 17 evaluations per coordinate-descent cycle
- `fine` — 7 × 7 × 7 = 343 combinations, 21 evaluations per cycle (intermediate resolution)
- `extended` — 14 × 16 × 19 = 4,256 combinations, 49 evaluations per cycle (finer search, slower; includes DID/TWFE corner)

| `covariates(varlist)`        | Time-invariant covariates for X_i'γ adjustment (paper Section 6.2 Eq. 14) | —          |
| `twostep_loocv(string)`      | Twostep LOOCV strategy: `cycling` (default) or `exhaustive`    | `cycling`  |
| `joint_loocv(string)`        | Joint LOOCV strategy: `cycling` or `exhaustive` (default)      | `exhaustive` |
| `vlevel(integer)`            | Verbosity level (0-4): 0=silent, 1=minimal, 2=detailed, 3=debug, 4=trace | `0`        |
| `singleunit(string)`         | Single-PSU stratum handling: `skip` (omit), `centered` (grand-mean correction) | `skip`     |
| `strata(varname)`            | Stratification variable for Rao-Wu bootstrap                   | —          |
| `psu(varname)`               | Primary sampling unit variable                                  | —          |
| `fpc(varname)`               | Finite population correction variable                           | —          |
| `nest`                       | Declare PSUs nested within strata                              | off        |
| `notiming`                   | Suppress elapsed-time display                                   | off        |

**Grid notes:**
- `lambda_time_grid()` and `lambda_unit_grid()` must be finite, non-negative
  numlists; Stata missing (`.`) is rejected at parse time.
- `lambda_nn_grid()` and the third slot of `fixedlambda()` accept `.` as
  +infinity (DID/TWFE corner, L ≡ 0).  The `default` grid does **not**
  include this corner; use `grid_style(extended)` or add `.` to a custom
  `lambda_nn_grid()` to let LOOCV select the "no factor structure" regime
  (classical DID / synthetic control).

### trop_bootstrap Options

| Option              | Description                                                     | Default    |
| ------------------- | --------------------------------------------------------------- | ---------- |
| `nreps(integer)`    | Number of bootstrap replications                                | `1000`     |
| `level(real)`       | Confidence level in percent (10–99.99)                          | `c(level)` |
| `seed(integer)`     | Random number generator seed                                    | `42`       |
| `maxiter(integer)`  | Maximum iterations per replication                              | `500`      |
| `tol(real)`         | Convergence tolerance per replication                           | `1e-6`     |
| `verbose`           | Display progress information                                    | off        |

## Stored Results

### Scalars

*Core estimates:*

| Scalar          | Description                                    |
| --------------- | ---------------------------------------------- |
| `e(att)`        | ATT point estimate                             |
| `e(se)`         | Bootstrap standard error                       |
| `e(t)`          | t statistic (att/se)                           |
| `e(pvalue)`     | Two-sided p-value for the primary interval     |
| `e(ci_lower)`   | Primary CI lower bound (one of the three candidates below, selected by `cimethod()`) |
| `e(ci_upper)`   | Primary CI upper bound                         |
| `e(df_r)`       | `max(1, N_1 - 1)` where `N_1 = e(N_treated_units)`; missing when `N_1 < 2` (normal fallback) |
| `e(mu)`         | Global intercept (joint only; missing for twostep) |

*Confidence interval candidates:*

All three candidate pairs are written to `e()` whenever bootstrap is
enabled, so downstream code can switch `cimethod()` without re-estimating.

| Scalar                        | Description                                              |
| ----------------------------- | -------------------------------------------------------- |
| `e(ci_lower_t)` / `e(ci_upper_t)` | t-wrap CI using `e(se)` and a t(`e(df_r)`) reference |
| `e(pvalue_t)`                 | Two-sided p-value from the t-wrap                        |
| `e(ci_lower_normal)` / `e(ci_upper_normal)` | Gaussian-wrap CI using `e(se)` and N(0,1) |
| `e(pvalue_normal)`            | Two-sided p-value from the normal-wrap                   |
| `e(ci_lower_percentile)` / `e(ci_upper_percentile)` | Percentile CI from the bootstrap empirical CDF (Algorithm 3 step 6) |

*Tuning parameters:*

| Scalar           | Description                                   |
| ---------------- | --------------------------------------------- |
| `e(lambda_time)` | Selected lambda_time                          |
| `e(lambda_unit)` | Selected lambda_unit                          |
| `e(lambda_nn)`   | Selected lambda_nn                            |
| `e(loocv_score)` | Optimal LOOCV score Q(lambda_hat)             |

*Sample information:*

| Scalar                | Description                              |
| --------------------- | ---------------------------------------- |
| `e(N_units)`          | Number of panel units (N)                |
| `e(N_periods)`        | Number of time periods (T)               |
| `e(N_obs)`            | Total observations                       |
| `e(N_treat)`          | Treated unit-period **cells** (W=1); legacy alias of `e(N_treated_obs)` |
| `e(N_treated)`        | Length of `e(tau)` = treated-cell count; equals `e(N_treated_obs)` |
| `e(N_treated_obs)`    | Treated unit-period **cells** (W=1 count, same quantity as `e(N_treat)` and `e(N_treated)`) |
| `e(N_treated_units)`  | Ever-treated **units** (N_1); cluster count for Algorithm 3 bootstrap |
| `e(N_control)`        | Number of control observations           |
| `e(N_control_units)`  | Never-treated units (N_0)                |
| `e(T_treat_periods)`  | Number of treatment periods              |
| `e(bootstrap_reps)`   | Number of bootstrap replications         |

*Convergence:*

| Scalar             | Description                                |
| ------------------ | ------------------------------------------ |
| `e(n_iterations)`  | Number of iterations                       |
| `e(converged)`     | Convergence indicator (1/0)                |
| `e(n_obs_estimated)` | Successfully estimated observations (twostep only) |
| `e(n_obs_failed)`  | Failed observations (twostep, if > 0)      |

*LOOCV diagnostics:*

| Scalar                     | Description                             |
| -------------------------- | --------------------------------------- |
| `e(loocv_n_valid)`         | Number of valid LOOCV evaluations       |
| `e(loocv_n_attempted)`     | Number of attempted LOOCV evaluations (= every D=0 cell, paper Eq. 5) |
| `e(loocv_fail_rate)`       | LOOCV failure rate                      |
| `e(loocv_used)`            | Whether LOOCV was performed (1/0)       |
| `e(seed)`                  | RNG seed used                           |

*Grid information:*

| Scalar                   | Description                                        |
| ------------------------ | -------------------------------------------------- |
| `e(n_lambda_time)`       | Number of lambda_time grid values                  |
| `e(n_lambda_unit)`       | Number of lambda_unit grid values                  |
| `e(n_lambda_nn)`         | Number of lambda_nn grid values                    |
| `e(n_grid_combinations)` | Total Cartesian grid combinations                  |
| `e(n_grid_per_cycle)`    | Grid evaluations per coordinate-descent cycle      |

*Other:*

| Scalar                    | Description                             |
| ------------------------- | --------------------------------------- |
| `e(balanced)`             | Balanced panel indicator (1/0)          |
| `e(miss_rate)`            | Missing data rate                       |
| `e(alpha_level)`          | Significance level for CI               |
| `e(effective_rank)`       | Effective rank of factor matrix         |
| `e(n_bootstrap_valid)`    | Number of valid bootstrap replications  |
| `e(data_validated)`       | Data validation indicator (1/0)         |
| `e(loocv_rmse)`           | LOOCV RMSE = sqrt(Q(lambda_hat) / n_valid) |
| `e(condition_number)`     | WLS design matrix condition number         |
| `e(bootstrap_fail_rate)`  | Bootstrap failure rate (0 to 1)            |
| `e(n_covariates)`         | Number of covariates (0 if none)           |
| `e(deff_weights)`         | Kish design effect of pweights             |


### Macros

| Macro                  | Description                              |
| ---------------------- | ---------------------------------------- |
| `e(cmd)`               | `"trop"`                                 |
| `e(cmdline)`           | Full command line as issued              |
| `e(method)`            | `"twostep"` or `"joint"`                |
| `e(grid_style)`        | `"default"`, `"fine"`, `"extended"`, or `"custom"` |
| `e(depvar)`            | Dependent variable name                  |
| `e(treatvar)`          | Treatment variable name                  |
| `e(panelvar)`          | Panel variable name                      |
| `e(timevar)`           | Time variable name                       |
| `e(vcetype)`           | `"Bootstrap"` or `""`                    |
| `e(bsvariance)`        | Bootstrap variance denominator actually used: `sample` or `paper` |
| `e(cimethod)`          | Primary CI method: `percentile`, `t`, or `normal`; `"percentile->t"` on downgrade |
| `e(estat_cmd)`         | `"trop_estat"`                           |
| `e(treatment_pattern)` | Treatment assignment pattern description |
| `e(twostep_loocv)`     | Twostep LOOCV strategy: `cycling` or `exhaustive` |
| `e(joint_loocv)`       | Joint LOOCV strategy: `cycling` or `exhaustive`   |
| `e(covariates)`        | Space-separated covariate variable names  |
| `e(spec_string)`       | Specification string for reproducibility  |
| `e(strata_var)`        | Stratification variable (survey only)     |
| `e(psu_var)`           | PSU variable (survey only)                |
| `e(fpc_var)`           | FPC variable (survey only)                |
| `e(bootstrap_type)`    | Bootstrap type: `standard` or `rao_wu`    |


### Matrices

| Matrix                   | Description                                          |
| ------------------------ | ---------------------------------------------------- |
| `e(b)`                   | Coefficient vector (1×1, ATT)                        |
| `e(V)`                   | Variance-covariance matrix (1×1; requires bootstrap) |
| `e(alpha)`               | Unit fixed effects (N×1); row names are the sorted unique values of `e(panelvar)` on the estimation sample (sanitised to valid Stata matrix identifiers) |
| `e(beta)`                | Time fixed effects (T×1); row names are the sorted unique values of `e(timevar)` on the estimation sample (sanitised to valid Stata matrix identifiers) |
| `e(factor_matrix)`       | Low-rank factor matrix L (T×N)                       |
| `e(tau)`                 | Per-cell treatment effects (N_treated×1); populated for both methods. For `joint` the vector carries the scalar `tau` replicated, so `mean(e(tau)) == e(att)` holds to machine precision for either method. |
| `e(tau_matrix)`          | Treatment effects arranged as a T×N panel-shaped matrix with `.` in untreated cells (when panel metadata is available) |
| `e(converged_by_obs)`    | Convergence flag per treated cell (`1` converged, `0` hit `maxiter()`, `-1` solver error); twostep only |
| `e(n_iters_by_obs)`      | Iteration count per treated cell; twostep only |
| `e(bootstrap_estimates)` | Bootstrap distribution (B×1; requires bootstrap)     |
| `e(theta)`               | Time weights (twostep only)                          |
| `e(omega)`               | Unit weights (twostep only)                          |
| `e(delta_time)`          | Time weights (joint only)                            |
| `e(delta_unit)`          | Unit weights (joint only)                            |
| `e(lambda_time_grid)`    | Lambda time grid values                              |
| `e(lambda_unit_grid)`    | Lambda unit grid values                              |
| `e(lambda_nn_grid)`      | Lambda nuclear norm grid values                      |
| `e(gamma)`               | Covariate coefficients (1×p; only with `covariates()`) |
| `e(lambda_grid)`         | Cartesian product of lambda grids (K×3)               |
| `e(cv_curve)`            | LOOCV scores at grid points (K×4)                    |

## Post-Estimation

### estat Subcommands

| Subcommand          | Abbreviation | Description                               |
| ------------------- | ------------ | ----------------------------------------- |
| `estat summarize`   | `sum`        | Sample structure and treatment allocation  |
| `estat vce`         |              | Variance-covariance matrix display         |
| `estat sensitivity` | `sens`       | Hyperparameter sensitivity analysis        |
| `estat weights`     | `weight`     | Unit and time weight diagnostics           |
| `estat bootstrap`   | `boot`       | Bootstrap distribution diagnostics         |
| `estat loocv`       |              | LOOCV hyperparameter selection diagnostics |
| `estat factors`     |              | Factor matrix (L) SVD analysis             |
| `estat triplerob`   | `trip`       | Theorem 5.1 triple-robustness bias bound decomposition (`\|Δᵘ\|₂ · \|Δᵗ\|₂ · \|B\|_*`) |
| `estat distance`    | `dist`       | Unit distance distribution diagnostics     |
| `estat mht`         |              | Multiple hypothesis testing correction     |
| `estat eventstudy`  | `es`         | Event-study dynamic treatment effects      |
| `estat pretrend`    | `pretest`    | Pre-trend test (all pre-treatment effects = 0) |
| `estat table`       |              | Export results as formatted table (LaTeX/Markdown/CSV) |

### predict Types

After `trop`, use `predict newvar, type` to generate predictions:

| Type             | Description                                           |
| ---------------- | ----------------------------------------------------- |
| `y0`             | Counterfactual outcome Y(0) **[default]**             |
| `y1`             | Counterfactual outcome Y(1)                           |
| `te`             | Treatment effect (treated obs only)                   |
| `residuals`      | Residuals Y - Y(0) - τ·W                              |
| `fitted`         | Fitted values Ŷ = Y(0) + τ·W                          |
| `alpha`          | Unit fixed effects                                    |
| `beta`           | Time fixed effects                                    |
| `mu`             | Global intercept (joint only)                         |
| `xb`             | Linear prediction (equivalent to `y0`)                |
| `att`            | Treatment effect (alias for `te`)                     |
| `counterfactual` | Counterfactual Y(0) (alias for `y0`)                  |

## Methodology

### The TROP Estimator

The TROP estimator models the potential control outcome as $Y_{it}(0) = \alpha_i + \beta_t + L_{it} + \epsilon_{it}$, where $\alpha_i$ are unit fixed effects, $\beta_t$ are time fixed effects, $L_{it}$ is a low-rank factor component, and $\epsilon_{it}$ is idiosyncratic noise.

For each treated unit–time pair $(i,t)$, the estimator predicts the counterfactual outcome by solving a weighted nuclear-norm penalized regression (Eq. 2 of the paper):

$$(\hat{\alpha}, \hat{\beta}, \hat{\mathbf{L}}) = \arg\min_{\alpha, \beta, \mathbf{L}} \sum_{j=1}^{N} \sum_{s=1}^{T} \theta_s^{i,t} \omega_j^{i,t} (1-W_{js})(Y_{js} - \alpha_j - \beta_s - L_{js})^2 + \lambda_{nn} \|\mathbf{L}\|_*$$

where the weights exhibit exponential decay (Eq. 3):

$$\theta_s^{i,t} = \exp(-\lambda_{\text{time}} \cdot |t - s|), \qquad \omega_j^{i,t} = \exp(-\lambda_{\text{unit}} \cdot \text{dist}_{-t}^{\text{unit}}(j, i))$$

The unit distance measures RMSE of outcome differences over shared control periods:

$$\text{dist}_{-t}^{\text{unit}}(j,i) = \left(\frac{\sum_{u} \mathbf{1}\{u \neq t\}(1-W_{iu})(1-W_{ju})(Y_{iu}-Y_{ju})^2}{\sum_{u} \mathbf{1}\{u \neq t\}(1-W_{iu})(1-W_{ju})}\right)^{1/2}$$

The treatment effect is then $\hat{\tau}_{it} = Y_{it} - \hat{\alpha}_i - \hat{\beta}_t - \hat{L}_{it}$.

This formulation encompasses DID, SC, MC, and SDID all as special cases. For $\lambda_{nn} = \infty$ and $\omega_j = \theta_s = 1$, we recover the DID/TWFE estimator. For $\omega_j = \theta_s = 1$ and $\lambda_{nn} < \infty$, we recover the MC estimator. For $\lambda_{nn} = \infty$ with specific unit and time weights, we recover SC and SDID.

### Triple Robustness Property

For the underlying concepts, see [Key Concepts](#key-concepts) above.

The bias satisfies a multiplicative bound (Theorem 5.1):

$$\left|\mathbb{E}[\hat{\tau} - \tau \mid \mathbf{L}]\right| \leq \|\Delta^{\mathbf{u}}(\omega, \Gamma)\|_2 \times \|\Delta^{\mathbf{t}}(\theta, \Lambda)\|_2 \times \|B\|_*$$

where $\Delta^{\mathbf{u}}$ is unit imbalance, $\Delta^{\mathbf{t}}$ is time imbalance, and $B$ captures regression adjustment misspecification. The estimator is consistent if **any one** of the three terms is negligible (Corollary 1):

1. Balance over unit factor loadings ($\|\Delta^{\mathbf{u}}\|_2 \approx 0$)
2. Balance over time factor loadings ($\|\Delta^{\mathbf{t}}\|_2 \approx 0$)
3. Correct regression adjustment specification ($\|B\|_* \approx 0$)

### Tuning Parameter Selection

The triplet $(\lambda_{\text{time}}, \lambda_{\text{unit}}, \lambda_{nn})$ is selected via leave-one-out cross-validation minimizing (Eq. 5):

$$Q(\lambda) = \sum_{i=1}^{N} \sum_{t=1}^{T} (1 - W_{it})(\hat{\tau}_{it}(\lambda))^2$$

This is equivalent to choosing the tuning parameters with the smallest out-of-sample squared error for predicting the potential outcome under control on the control observations. The grid search uses coordinate descent (Algorithm 1): each parameter is optimized in turn while holding the other two at their most recently selected values, then the cycle repeats until convergence.

### Estimation Methods

- **Twostep** (Algorithm 2): Per-observation estimation allowing heterogeneous treatment effects. For each treated pair $(i,t)$, the model is fitted as if $(i,t)$ were the only treated observation. The ATT is $\hat{\tau} = \frac{1}{\sum_{i,t} W_{it}} \sum_{i,t} W_{it} \hat{\tau}_{it}$.
- **Joint** (Remark 6.1): Weighted least squares with a single scalar treatment effect $\tau$, assuming homogeneous effects across all treated unit–time pairs. Uses global weights shared across all treated observations. Computationally more efficient when the homogeneity assumption holds.

### Bootstrap Inference

Variance estimation follows Algorithm 3: stratified unit block bootstrap
that separately resamples $N_0$ control units and $N_1$ treated units
with replacement.  For each replication $b = 1, \ldots, B$, the full
estimation procedure (including LOOCV if applicable) is repeated to
obtain $\hat{\tau}^{(b)}$.  The bootstrap variance is

$$\hat{V}_{\tau} = \frac{1}{B - 1} \sum_{b=1}^{B} (\hat{\tau}^{(b)} - \bar{\hat{\tau}})^2$$

by default (Bessel-corrected sample variance).  The paper's original
population-variance denominator $1/B$ is selectable via
`bsvariance(paper)`; the two choices differ by at most 0.5% at $B = 200$.

**Reference distribution.** Because Algorithm 3 resamples
*units*, the cluster count that governs the small-sample df is
$N_1 = $`e(N_treated_units)`.  Stata's `trop` therefore uses
$t(N_1 - 1)$ whenever $N_1 \geq 2$ and falls back to $\mathcal{N}(0,1)$
otherwise.  `e(df_r)` is `max(1, N_1 - 1)` or missing (normal
fallback).  Using a `df` derived from the number of treated *cells*
would inflate significance as $T_\text{post}$ grows.

**Three confidence intervals, one primary.** Every bootstrap run
produces three CI candidates:

1. **Percentile CI** — the $\alpha/2$ and $1-\alpha/2$ quantiles of the
   bootstrap empirical CDF (Algorithm 3 step 6).
2. **t-wrap CI** — `att ± invttail(df_r, α/2) · se`.
3. **Normal-wrap CI** — `att ± invnormal(1 − α/2) · se`.

The `cimethod(percentile | t | normal)` option selects which pair is
promoted to the primary `e(ci_lower)` / `e(ci_upper)`.  The default is
`percentile` whenever `bootstrap > 0` (the paper's recommended
distribution-free interval); when `bootstrap(0)` is combined with
`cimethod(percentile)` the parser downgrades to `t` and records the
trace in `e(cimethod)` as `"percentile->t"`.  All three candidate pairs
are always persisted on `e()` so a downstream analyst can switch
`cimethod()` without re-estimating.

## Architecture

The package uses a four-layer design for performance and numerical accuracy:

```
┌─────────────────────────────────────────────────┐
│  Stata User Interface  (trop.ado)               │
│  - Syntax parsing, option handling              │
├─────────────────────────────────────────────────┤
│  Mata Interface Layer                           │
│  - Input validation and data conversion         │
│  - e() result storage                           │
├─────────────────────────────────────────────────┤
│  C Bridge Plugin                                │
│  - Pointer conversion, error code mapping       │
├─────────────────────────────────────────────────┤
│  Rust Core                                      │
│  - Distance matrices, weight computation        │
│  - LOOCV grid search, SVD estimation            │
│  - Stratified unit block bootstrap              │
└─────────────────────────────────────────────────┘
```

All numerical computation (LOOCV, SVD, bootstrap) is performed in Rust for speed and precision. The Mata layer handles data validation and result storage. Users do not need the Rust toolchain — the pre-compiled plugin is included.

## Numerical robustness choices

Nine implementation choices in `trop` are documented below.  Each one
is pinned by a regression test so future refactors cannot silently
regress, and each is motivated by a first-principles concern rather
than personal preference:

1. **FISTA adaptive restart disabled** (`rust/src/estimation.rs`).  The
   nuclear-norm proximal solver does **not** use the monotone
   gradient-restart scheme of O'Donoghue & Candès (2015).  Although
   the restart criterion
   `⟨y_k − x_k, x_k − x_{k−1}⟩ > 0` can eliminate momentum
   oscillations in theory, it fires too aggressively on small panels
   and prevents convergence.  The reference Python implementation
   (`diff-diff` v3.1.1) does not use restart either, so we disable it
   to maintain numerical consistency.  The test
   `tests/test_fista_restart_stability.do` verifies that the FISTA
   solver remains stable across a range of `lambda_nn` values without
   restart.

2. **LAPACK `dgelsd` for the weighted least-squares step**
   (`rust/src/estimation.rs`).  The SVD-based minimum-norm solver
   returns a Moore–Penrose pseudoinverse solution on rank-deficient
   designs that arise when the weight vector zeroes out entire
   rows/columns of the design matrix.  The SVD truncation tolerance
   `rcond` is `max(ε · max(m, n), 1e-12)` — the floor stabilises
   $\hat\alpha / \hat\beta$ on the smallest benchmark panels
   (Basque N = 17, West Germany N = 16) without perturbing $\hat\tau$.
   Pinned by `tests/test_dgelsd_rank_deficient_wls.do`.

3. **`UnitDistanceCache`** (`rust/src/distance.rs`).  Pairwise
   $\sum_u (Y_{iu} − Y_{ju})^2$ sums are precomputed once; each
   leave-$t$-out distance $\text{dist}_{-t}(j, i)$ is then an O(1)
   subtraction instead of an O(T) rescan.  Cache equivalence is
   verified to < 10⁻¹⁰ by
   `tests/test_unit_distance_cache_equivalence.do`.

4. **Deterministic LOOCV tie-breaker** (`rust/src/loocv.rs`,
   `better_candidate`).  When two `(lambda_time, lambda_unit,
   lambda_nn)` triples score within `TIE_TOL = 1 × 10⁻¹⁰` of each
   other, `trop` prefers the larger `lambda_nn`, then the smaller
   `lambda_time`, then the smaller `lambda_unit`.  ULP-level BLAS
   differences can otherwise flip `argmin Q(λ)` across platforms.
   Pinned by `tests/test_loocv_tiebreak_determinism.do`.

5. **Inference reference distribution** (`ado/trop.ado`,
   `mata/trop_ereturn_store.mata`).  The bootstrap resamples units in
   stratified fashion (Algorithm 3 step 3), so the cluster count that
   governs the small-sample reference df is $N_1$, the number of
   ever-treated units — not the number of treated cells.  `trop`
   therefore uses $t(N_1 - 1)$ whenever $N_1 \geq 2$ and falls back to
   the standard normal otherwise.  The primary CI defaults to the
   paper-specified percentile interval whenever bootstrap is enabled;
   `cimethod()` re-selects the primary pair from the three candidates
   (percentile, t, normal).  Pinned by
   `tests/test_inference_df_is_treated_units.do` and
   `tests/test_cimethod_option.do`.

6. **Simultaneous-adoption guard for the joint method**
   (`rust/src/loocv.rs::check_simultaneous_adoption`).  The joint
   estimator's global weight matrix $\delta$ depends on a shared
   `treated_periods` count; it is only well-defined when every
   treated unit enters treatment at the same period $T_1$ and stays
   treated through the end of the panel (paper Remark 6.1).  The
   Stata front-end already refuses staggered $D$ for
   `method(joint)`; as defence-in-depth, every joint C-ABI entry
   (`stata_estimate_joint`, `stata_bootstrap_trop_variance_joint`,
   `stata_loocv_grid_search_joint`, `stata_loocv_cycling_search_joint`,
   and the `_weighted` variants) now short-circuits with
   `TropError::InvalidDimension` on staggered/non-absorbing $D$ rather
   than silently mis-computing $T_1$.  Pinned by five unit tests in
   `rust/src/loocv.rs`.

7. **λ_nn = 0 closed form** (`rust/src/estimation.rs`).  At
   $\lambda_{nn}=0$, paper Eq. 2 reduces to $\hat{L}_{t,i} =
   Y_{t,i} - \hat\alpha_i - \hat\beta_t$ on the weighted support
   ($W>0$) while $\hat{L}$ is unidentified off the support ($W=0$).
   The implementation therefore sets $\hat{L}$ to the closed-form
   residual on valid cells and preserves the previous iterate on
   invalid cells; a debug-build post-condition verifies the latter
   invariant.  Pinned by
   `estimation::tests::test_lambda_nn_zero_closed_form_preserves_invalid_cells`.

8. **Unified 5% failure-rate thresholds for LOOCV and bootstrap**
   (`mata/trop_ereturn_store.mata`, `mata/trop_rust_interface.mata`).
   Both `_trop_display_bootstrap_warnings` and the Mata-side
   `check_loocv_fail_rate()` now issue an advisory at **5%** and abort
   with `rc ∈ {498, 504}` at **50%**.  The 5% threshold is tight
   enough that, on a panel with ~1,000 `D=0` cells, roughly 50 failed
   leave-one-out fits surface instead of passing silently — a
   failure rate of that magnitude can perturb the selected
   `(lambda_time, lambda_unit, lambda_nn)` off of `Q(λ)`'s true argmin
   (paper Eq. 5).  The same threshold at the bootstrap stage means
   11 failures out of 200 replicates are no longer hidden behind the
   historical 10% gate.  Both failure rates are exposed on
   `e(loocv_fail_rate)` / `e(bootstrap_fail_rate)` for downstream
   diagnostics.  Pinned by
   `tests/test_bootstrap_fail_rate_threshold.do`,
   `tests/test_ereturn_fail_rate_coverage.do`, and
   `tests/test_loocv_fail_rate_threshold.do`.

9. **Original-ID row names on `e(alpha)` and `e(beta)`**
   (`ado/_trop_attach_idnames.ado`).  After estimation completes,
   `e(alpha)` (`N × 1`) and `e(beta)` (`T × 1`) are rewritten with
   matrix row names drawn from the sorted unique values of the
   user-supplied `panelvar` / `timevar` on the estimation sample.
   Because both `egen ... = group()` and `levelsof` return sorted
   unique values, row `i` of `e(alpha)` corresponds to the `i`-th
   unique panel identifier; the plugin indices 1..N remain available
   through scalar access (`e(alpha)[i, 1]`).  Identifiers are
   sanitised to valid Stata matrix names (letters / digits /
   underscore, ≤ 32 chars, non-leading-digit), so numeric IDs like
   `1989, 1990, ...` render as `_1989, _1990, ...` in
   `matrix list e(beta)`.  Pinned by
   `tests/test_alpha_beta_rownames.do`.

**Out of scope (current release).** Time-varying covariates
$X_{it}\beta$ (the current implementation supports time-invariant
covariates $X_i'\gamma$ per paper Section 6.2 Eq. 14, but not
full panel-varying regressors); and switching-treatment patterns
under `method(joint)`.  These lie outside the current scope and are
planned for a future release.

## Known Limitations

### Functional Constraints

| Limitation | Description | Status |
|-----------|-------------|--------|
| Covariate dimension | Requires p < min(N, T) where p is the number of covariates | By design |
| Time-varying covariates | X_{it}β form not supported | Planned |
| Staggered adoption (Joint) | `method(joint)` requires simultaneous treatment adoption | Paper constraint |
| Staggered adoption (Twostep) | `method(twostep)` implicitly allows staggering without full theoretical guarantee | Use with caution |

### Performance & Memory

| Constraint | Description | Mitigation |
|-----------|-------------|------------|
| LOOCV complexity | Large panels (N>200, T>50) may be slow | Use `fixedlambda()` or `grid_style(default)` |
| Bootstrap memory | B iterations require O(B·N·T) memory | Reduce `bootstrap()` count |
| Distance matrix | O(N²) unit distance matrix storage | Keep panel size N<500 |

## Troubleshooting

### Common Errors

| Error Code | Meaning | Solution |
|-----------|---------|----------|
| 3 | Invalid treatment assignment | Verify treatment variable is binary (0/1) |
| 4 | No valid control observations | Ensure pre-treatment control observations exist |
| 5 | Estimation did not converge | Try increasing `maxiter()` or relaxing `tol()` |
| 8 | Invalid panel structure | Check panel/time variable uniqueness |
| 12 | SVD decomposition failed | Check for perfectly collinear covariates |
| 13 | Singleton PSU | Use `singleunit(centered)` option |

### Numerical Stability Issues

**Symptoms**: High LOOCV failure rate (>10%) or non-convergence

**Diagnosis**:
1. Check `e(loocv_fail_rate)` — if >0.10, grid search quality is compromised
2. Check `e(condition_number)` — if >1e10, design matrix is ill-conditioned
3. Check `e(effective_rank)` — low rank may indicate overfitting

**Solutions**:
- Reduce covariates or check for multicollinearity
- Use `grid_style(default)` instead of `grid_style(extended)`
- Consider winsorizing extreme values

### Performance Tips

| Scenario | Recommendation |
|----------|---------------|
| Large panels (N>100, T>30) | Use `fixedlambda()` to skip LOOCV |
| Slow bootstrap | Start with `bootstrap(200)`, increase after confirmation |
| Memory issues | Reduce panel size or estimate on subsamples |

## Third-Party Command Integration

### estout / esttab Integration

`trop` stores its results in `e()` following Stata conventions. To export results via `estout` / `esttab`, construct a compatible coefficient vector:

```stata
* Run TROP estimation
trop y d, panelvar(id) timevar(t) fixedlambda(0.1 0 0.9) bootstrap(200) seed(42)

* Approach 1: Post e(b) and e(V) directly (already 1x1 matrices)
eststo trop_model

* Approach 2: Add custom scalars for richer tables
estadd scalar att = e(att)
estadd scalar se = e(se)
estadd scalar pvalue = e(pvalue)
estadd scalar ci_lo = e(ci_lower)
estadd scalar ci_hi = e(ci_upper)
estadd local method = e(method)

* Export to LaTeX
esttab trop_model, stats(att se pvalue ci_lo ci_hi method) ///
    title("TROP Estimation Results")

* Export to CSV
esttab trop_model using results.csv, stats(att se pvalue) csv replace
```

### coefplot Integration

`trop` results can be visualised with `coefplot` after estimation:

```stata
* Standard coefficient plot (plots e(b) with e(V))
trop y d, panelvar(id) timevar(t) fixedlambda(0.1 0 0.9) bootstrap(200) seed(42)
coefplot, title("TROP ATT Estimate")
```

For event study plots, use the built-in `estat eventstudy` command:

```stata
* Event study with internal plotting
estat eventstudy, graph

* Or extract data for custom coefplot formatting
estat eventstudy, nograph
matrix es = r(event_effects)
```

### Multiple Model Comparison

```stata
* Compare twostep vs joint
trop y d, panelvar(id) timevar(t) fixedlambda(0.1 0 0.9) bootstrap(200) seed(42)
eststo twostep_model

trop y d, panelvar(id) timevar(t) method(joint) fixedlambda(0.1 0 0.9) bootstrap(200) seed(42)
eststo joint_model

esttab twostep_model joint_model, ///
    mtitles("Twostep" "Joint") ///
    stats(att se pvalue N_obs)
```

## Performance Tuning

### Bootstrap Parallel Computation

TROP's bootstrap variance estimation uses the Rayon parallel library in the Rust backend, utilising all available CPU cores by default.

**Thread control:** Set the `RAYON_NUM_THREADS` environment variable before calling `trop`:

```stata
* Limit to 4 threads
shell export RAYON_NUM_THREADS=4

* Or set in your shell startup script
* ~/.bashrc: export RAYON_NUM_THREADS=4
```

**Performance guidelines:**

| Panel size | Bootstrap(B) | Approximate time |
|-----------|--------------|------------------|
| Small (N<50, T<20) | 200 | < 30 seconds |
| Medium (N=50–200, T=20–50) | 500 | 2–10 minutes |
| Large (N>200, T>50) | 200 | 10–60 minutes |

**Memory estimate:** Approximately `8 × N × T × B` bytes. For example, N=100, T=50, B=500 requires ~200 MB.

**Tips for large panels:**
- Use `fixedlambda()` to skip LOOCV (the most time-intensive step)
- Start with `bootstrap(200)` for exploratory analysis
- Increase to `bootstrap(500)` or `bootstrap(1000)` for final results
- Monitor `e(loocv_fail_rate)` — high failure rates indicate numerical difficulty

## References

Athey, S., Imbens, G., Qu, Z., & Viviano, D. (2025). Triply robust panel estimators. arXiv preprint arXiv:2508.21536.

## Authors

**Stata Implementation:**

- **Xuanyu Cai**, City University of Macau
  Email: [xuanyuCAI@outlook.com](mailto:xuanyuCAI@outlook.com)
- **Wenli Xu**, City University of Macau
  Email: [wlxu@cityu.edu.mo](mailto:wlxu@cityu.edu.mo)

**Methodology:**

- **Susan Athey**, Stanford University
- **Guido Imbens**, Stanford University
- **Zhaonan Qu**, Columbia University
- **Davide Viviano**, Harvard University

## License

AGPL-3.0. See [LICENSE](LICENSE) for details.

## Citation

If you use this package in your research, please cite both the methodology paper and the Stata implementation:

**APA Format:**

> Cai, X., & Xu, W. (2025). *trop: Stata module for Triply Robust Panel estimation* [Computer software]. GitHub. https://github.com/gorgeousfish/TROP
>
> Athey, S., Imbens, G., Qu, Z., & Viviano, D. (2025). Triply robust panel estimators. arXiv preprint arXiv:2508.21536.

**BibTeX:**

```bibtex
@software{trop2025stata,
  title={trop: Stata module for Triply Robust Panel estimation},
  author={Xuanyu Cai and Wenli Xu},
  year={2025},
  version={1.2.0},
  url={https://github.com/gorgeousfish/TROP}
}

@article{athey2025triply,
  title={Triply robust panel estimators},
  author={Athey, Susan and Imbens, Guido and Qu, Zhaonan and Viviano, Davide},
  journal={arXiv preprint arXiv:2508.21536},
  year={2025}
}
```

## See Also

- Original paper by Athey, Imbens, Qu & Viviano: https://arxiv.org/abs/2508.21536
- Related Stata packages: [`sdid`](https://github.com/Daniel-Pailanir/sdid) (Synthetic DID), [`diddesign`](https://github.com/gorgeousfish/diddesign) (Double DID)

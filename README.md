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
- **Covariate Adjustment** — Covariates X_{j,s}'γ (paper Section 6.2 Eq. 14) with automatic WLS projection
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
Triply Robust Panel Estimator                   Number of obs      =    2,000
Method: twostep                                 Number of units    =       50
                                                Time periods       =       40
                                                Treated obs        =        8
                                                Bootstrap reps     =      200
------------------------------------------------------------------------------
             |      ATT       Std. err.    t     P>|t|    [95% conf. interval]
-------------+----------------------------------------------------------------
           d |   0.031406     0.014098      2.23  0.061    0.004771    0.056458
------------------------------------------------------------------------------
Lambda: time = 0.100, unit = 0.000, nn = 0.900 (fixed)
Convergence: Yes (5 iterations)
------------------------------------------------------------------------------
```

### Example 2: LOOCV-Selected Hyperparameters

```stata
trop_data cps_logwage
trop y d, panelvar(id) timevar(t)
```

Output (LOOCV selects optimal hyperparameters via coordinate-descent grid search):

```
------------------------------------------------------------------------------
Triply Robust Panel Estimator                   Number of obs      =    2,000
Method: twostep                                 Number of units    =       50
                                                Time periods       =       40
                                                Treated obs        =        8
                                                Bootstrap reps     =      200
------------------------------------------------------------------------------
             |      ATT       Std. err.    t     P>|t|    [95% conf. interval]
-------------+----------------------------------------------------------------
           d |   0.034188     ...           ...   ...     ...         ...
------------------------------------------------------------------------------
Lambda: time = 0.500, unit = 5.000, nn = 1.000 (LOOCV, Q = 3.4717)
Convergence: Yes (... iterations)
------------------------------------------------------------------------------
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
------------------------------------------------------------------------------
Triply Robust Panel Estimator                   Number of obs      =    2,000
Method: twostep                                 Number of units    =       50
                                                Time periods       =       40
                                                Treated obs        =        8
                                                Bootstrap reps     =      200
------------------------------------------------------------------------------
             |      ATT       Std. err.    t     P>|t|    [95% conf. interval]
-------------+----------------------------------------------------------------
           d |   0.031406     0.014098      2.23  0.061    0.004771    0.056458
------------------------------------------------------------------------------
Lambda: time = 0.100, unit = 0.000, nn = 0.900 (fixed)
Convergence: Yes (5 iterations)
------------------------------------------------------------------------------
```

### Example 4: Joint Method

```stata
trop_data cps_logwage
trop y d, panelvar(id) timevar(t) ///
    method(joint) fixedlambda(0.1 0 0.9)
```

Output:

```
------------------------------------------------------------------------------
Triply Robust Panel Estimator                   Number of obs      =    2,000
Method: joint                                   Number of units    =       50
                                                Time periods       =       40
                                                Treated obs        =        8
                                                Bootstrap reps     =      200
------------------------------------------------------------------------------
             |      ATT       Std. err.    t     P>|t|    [95% conf. interval]
-------------+----------------------------------------------------------------
           d |   0.031406     0.014097      2.23  0.061    0.004771    0.056458
------------------------------------------------------------------------------
Lambda: time = 0.100, unit = 0.000, nn = 0.900 (fixed)
Convergence: Yes (2 iterations)
Global intercept (mu):   5.154320
------------------------------------------------------------------------------
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
------------------------------------------------------------------------------
Estimation sample summary
------------------------------------------------------------------------------
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

Outcome variable (y):
  Mean:          5.925
  Std. Dev:      0.444
  Min:           4.798
  Max:           6.806
  p25:           5.584
  p75:           6.298

Estimation details:
  Method:        twostep (Algorithm 2 default)
  Outcome var:   y
  Treatment var: d
  Panel var:     id
  Time var:      t
------------------------------------------------------------------------------
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
trop y d, panelvar(id) timevar(t) fixedlambda(0.1 0 0.9) bootstrap(0)

* Then add bootstrap inference separately
trop_bootstrap, nreps(200) seed(42)
```

Output:

```
------------------------------------------------------------
TROP Bootstrap Inference Results
------------------------------------------------------------
ATT estimate:       0.031406
Bootstrap SE:       0.014098
95% CI:       [   -0.001929,     0.064742]
p-value:              0.0612

Bootstrap reps:       200
Valid reps:           200
------------------------------------------------------------
```

### Example 7: PWT Log-GDP Panel (111 Countries × 48 Years)

For a large panel, use the Penn World Tables democracy dataset:

```stata
trop_data pwt_loggdp

* Paper's hyperparameters for PWT. Large panels with very small lambda_nn
* may require many iterations; use maxiter(1000) for tighter convergence.
trop y d, panelvar(id) timevar(t) fixedlambda(0.4 0.3 0.006) maxiter(1000) bootstrap(0)
```

Output:

```
------------------------------------------------------------------------------
Triply Robust Panel Estimator                   Number of obs      =    5,328
Method: twostep                                 Number of units    =      111
                                                Time periods       =       48
                                                Treated obs        =       29
------------------------------------------------------------------------------
             |      ATT
-------------+----------------------------------------------------------------
           d |  -0.016024
------------------------------------------------------------------------------
Lambda: time = 0.400, unit = 0.300, nn = 0.006 (fixed)
Convergence: No (1000 iterations)
Note: SE/CI require bootstrap(); re-run with bootstrap(200).
------------------------------------------------------------------------------
```

> **Convergence note.** With a very small `lambda_nn` (e.g. 0.006), the low-rank
> factor matrix L can have many nonzero singular values, and the alternating
> minimization progresses slowly on large panels. The strict `tol(1e-6)` default
> may not be reached within `maxiter(1000)`; the point estimate is nevertheless
> stable to the third decimal place. For faster and fully-converged estimates,
> either (i) increase `lambda_nn` (e.g. `fixedlambda(0.4 0.3 0.1)` converges in
> about 110 iterations to τ = -0.006672) or (ii) relax the tolerance via
> `tol(1e-4)`. Use `bootstrap(0)` on large panels for rapid exploratory analysis,
> then add `trop_bootstrap` or re-run with `bootstrap(200)` for inference.

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

| Option | Description | Default |
|:--|:--|:--|
| `covariates(varlist)`        | Covariates for X'γ adjustment (paper Section 6.2 Eq. 14) | —          |
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
- **Large-panel performance:** `lambda_nn` values in the open interval `(0, 0.1)`
  — especially the `0.01` point of the `default` grid — are by far the costliest
  to evaluate, because `0 < lambda_nn < 0.1` triggers a FISTA solve (inner cap 50,
  one full T×N SVD per inner step) whereas `lambda_nn = 0` and `lambda_nn ≥ 0.1`
  use cheap closed-form paths. At ~8,300 control cells one interior candidate was
  measured tens of thousands of times slower per cell than `lambda_nn = 0` (≈12,600
  ms/cell vs ≈0.2 ms/cell), i.e. tens of hours for that single candidate. On large
  panels prefer a custom grid that avoids the open interval, e.g.
  `lambda_nn_grid(0 1 10)`.

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
| `e(stage1_lambda_nn)` | Stage-1 nuclear norm penalty (joint cycling) |
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
| `e(loocv_first_failed_t)`  | Time index of first LOOCV failure       |
| `e(loocv_first_failed_i)`  | Unit index of first LOOCV failure       |
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
| `estat triplerob`   | `trip`       | Theorem 5.1 triple-robustness bias bound decomposition (`‖Δᵘ‖₂ · ‖Δᵗ‖₂ · ‖B‖_*`) |
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

## Known Limitations

| Constraint | Description | Mitigation |
|-----------|-------------|------------|
| `method(joint)` requires simultaneous adoption | All treated units must enter treatment at the same period (paper Remark 6.1) | Use `method(twostep)` for staggered designs |
| LOOCV can be slow on large panels | O(grid × N×T) evaluations | Use `fixedlambda()` for quick results; `grid_style(default)` for moderate speed |
| Time-varying coefficient models β(t) | Not supported; current `covariates()` estimates a single shared γ | — |

For advanced topics (methodology, architecture, troubleshooting, performance tuning), see [Advanced Guide](docs/advanced.md).

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


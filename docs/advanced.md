# trop — Advanced Guide

This guide collects advanced and developer-oriented material for the `trop`
package: recommended workflow, the interactive tutorial, methodology details,
package architecture, troubleshooting, third-party command integration, and
performance tuning. For installation, quick-start examples, options, stored
results, and post-estimation reference, see the main [README](../README.md).

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

For the underlying concepts, see [Key Concepts](../README.md#key-concepts).

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

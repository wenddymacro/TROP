# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2026-06-20

### Added (Covariate adjustment, paper Section 6.2 Equation 14)
- **`trop_stata/ado/trop.ado`** — new `covariates(varlist)` option allows
  users to include covariate adjustment in estimation.  Covariates enter
  the alternating minimisation as a third WLS step that updates the
  γ coefficient vector.  Syntax:
  `trop y d, panelvar(id) timevar(t) covariates(x1 x2 x3)`.
  Returns `e(gamma)` (1×p matrix of covariate coefficients),
  `e(n_covariates)` (scalar), and `e(covariates)` (macro listing the
  covariate names).  Seven layers modified end-to-end:
- **`trop_stata/rust/src/estimation.rs`** — `estimate_model()` gains
  `x: Option<&ArrayView2<f64>>` and `gamma_init` parameters; alternating
  minimisation extended from three steps (α → β → L) to four steps
  (α → β → γ(WLS) → L(SVT)).  WLS solved via Cholesky decomposition;
  rank-deficient cases fall back to SVD least-squares (`dgelsd`).
- **`trop_stata/rust/src/loocv.rs`** — all LOOCV scoring functions
  propagate the X matrix; pseudo treatment effect becomes
  τ = Y − α − β − L − X′γ.
- **`trop_stata/rust/src/bootstrap.rs`** — bootstrap resampling
  reorganises X by unit index; τ computation includes X′γ.
- **`trop_stata/rust/src/lib.rs`** — six new `_with_covariates` C-ABI
  export functions (`stata_loocv_grid_search_with_covariates`,
  `stata_loocv_cycling_search_joint_with_covariates`,
  `stata_estimate_twostep_with_covariates`,
  `stata_estimate_joint_with_covariates`,
  `stata_bootstrap_trop_variance_with_covariates`,
  `stata_bootstrap_trop_variance_joint_with_covariates`).
- **`trop_stata/plugin/stata_bridge.c`** — six handler functions gain
  covariate branches: read `__trop_n_covariates` scalar, load data from
  `__trop_covariates` matrix, dispatch to `_with_covariates` Rust
  functions, write back `__trop_gamma`.
- **`trop_stata/mata/trop_data_transfer.mata`** — new
  `trop_prepare_covariates()` function pivots user-supplied covariates
  into a (T×N) × p matrix compatible with the Rust core's column-major
  layout.
- Numerical validation: Joint method ATT difference < 1e-8, Twostep
  method ATT difference < 5e-8 (against Python reference implementation).

### Added (Multi-platform CI/CD)
- **`.github/workflows/build-plugins.yml`** — new 4-platform matrix build
  workflow (macOS ARM64, macOS Intel, Linux x64, Windows x64); triggered
  on tag push, automatically compiles platform-specific plugins and
  creates a GitHub Release with all four binaries attached.
- **`trop.pkg`** — plugin distribution paths changed to `ado/`-sibling
  directory, supporting 4-platform precompiled binary distribution.
- **`ado/_trop_load_plugin.ado`** — plugin search gains a new
  "same directory as this ado file" priority path, adapting to
  `net install` deployment where the plugin lives alongside the ADO
  files rather than in PLUS/PERSONAL.

### Changed (README_CN.md sync)
- **`README_CN.md`** — complete rewrite synchronised with the English
  `README.md` (388 lines), covering installation, quickstart, options,
  stored results, post-estimation, methodology, architecture, and
  citation sections.

### Changed (estat triplerob: basis-arbitrariness warning in the L = 0 corner)
- **`trop_stata/mata/trop_estat_helpers.mata`** — when the estimated
  factor matrix `L` is numerically zero (`||L||_* < 1e-30`; typical
  causes are `lambda_nn = +Inf` or an alpha+beta fit that absorbs every
  signal), `estat triplerob` still prints the three Theorem-5.1
  components in its usual layout, but now appends a concluding "Note"
  block explaining that the SVD loadings `Gamma = Vt'` and `Lambda = U`
  are mathematically arbitrary in this case — any orthogonal basis
  satisfies `L = U * 0 * V'` — and that the reported `|Delta^u|_2` /
  `|Delta^t|_2` magnitudes therefore depend on the LAPACK-chosen basis
  and may differ across BLAS backends (Accelerate / OpenBLAS / MKL).
  The *product* bound is always exactly 0 (platform-invariant) because
  `||B||_*` times anything equals 0, so the scientific conclusion is
  unchanged.  Rationale: surface the subtlety without hiding the
  raw numbers.
- **`trop_stata/tests/test_estat_triplerob_corner.do`** — new regression
  asserting that `estat triplerob` runs under `fixedlambda(0 0 .)` /
  `fixedlambda(0.5 0.5 .)` (the DID/TWFE closed form), that
  `r(residual) == 0` and `r(bias_bound) == 0` exactly, and that
  `r(delta_unit)` / `r(delta_time)` are finite.  Parity three-pack
  (`test_joint_outer_convergence_parity.do`,
  `test_joint_exhaustive_parity.do`,
  `test_twostep_exhaustive_parity.do`) re-verified — this change is
  display-only and does not shift any numerical output.

### Added (Stage-1 LOOCV diagnostics, paper Footnote 2)
- **`trop_stata/rust/src/loocv.rs`** — new public API
  `loocv_grid_search_with_stage1()` / `loocv_cycling_search_joint_with_stage1()`
  returning the `(lambda_time_init, lambda_unit_init, lambda_nn_init)`
  triple that Stage-1 univariate sweeps select as the seed for Stage-2
  coordinate descent.  The thin wrappers `loocv_grid_search` /
  `loocv_cycling_search_joint` preserve the historical signature by
  discarding the Stage-1 triple, so existing Rust tests continue to
  compile unchanged.
- **`trop_stata/rust/src/lib.rs`** / **`trop_stata/plugin/stata_bridge.{h,c}`**
  — the C-ABI entries `stata_loocv_grid_search` and
  `stata_loocv_cycling_search_joint` gain three new **NULL-safe** out
  pointers (`stage1_lambda_time_out`, `stage1_lambda_unit_out`,
  `stage1_lambda_nn_out`); callers passing `NULL` see unchanged
  behaviour.  The plugin writes the triple to the scratch scalars
  `__trop_stage1_lambda_{time,unit,nn}`.
- **`trop_stata/mata/trop_ereturn_store.mata`** — surfaces the Stage-1
  triple as `e(stage1_lambda_time)`, `e(stage1_lambda_unit)`, and
  `e(stage1_lambda_nn)`.  The exhaustive LOOCV paths do not use a
  Stage-1 initialisation and leave these scalars missing, which is the
  correct semantic marker for "not applicable".  `e(stage1_lambda_nn)`
  round-trips the `+Inf -> .` sentinel so `if e(stage1_lambda_nn) >= .`
  parses as expected.
- **`trop_stata/ado/trop_estat_loocv.ado`** — `estat loocv` prints a
  `Stage-1 univariate init (Footnote 2; cycling only)` block above the
  performance summary when the triple is populated.  A `*` marker next
  to a Stage-1 value signals that Stage-2 cycling polished away from
  the seed, i.e. the Q(lambda) surface was non-convex enough that
  coordinate descent was non-trivial.  The block is silently omitted
  on the exhaustive path.
- **`trop_stata/ado/trop.sthlp`** — new stored-result rows documenting
  `e(stage1_lambda_{time,unit,nn})` and their cycling-only semantics.
- **`trop_stata/tests/test_loocv_stage1_exposure.do`** — new regression
  locking the contract end-to-end: finite Stage-1 values within their
  respective grids on cycling (twostep and joint), all three missing on
  the exhaustive paths.  Parity regressions
  (`test_joint_outer_convergence_parity.do`,
  `test_joint_exhaustive_parity.do`,
  `test_twostep_exhaustive_parity.do`) re-verified at the documented
  tolerances — Stage-1 exposure is additive and does not shift any
  numerical output.

## [1.1.0] - 2026-04-20

### Highlights

- **Triple-robustness bias diagnostic** — a new post-estimation
  subcommand `estat triplerob` reports the paper Theorem 5.1
  decomposition `|E[tauhat - tau | L]| <= ||Delta^u||_2 *
  ||Delta^t||_2 * ||B||_*` so users can see which of the three
  ingredients (unit balance, time balance, regression adjustment) is
  driving any residual bias on their panel.
- **Original-ID row names on `e(alpha)` / `e(beta)`** — fixed effects
  are now keyed by the user's original `panelvar` / `timevar`
  identifiers (sanitised to valid Stata matrix names), so
  `matrix list e(alpha)` prints labelled rows.
- **Unified 5% failure-rate threshold** for LOOCV and bootstrap
  diagnostics — both surfaces raise the advisory at the same
  sensitivity, matching the tightness of the bootstrap warning that
  was already in place.
- **Numerical fidelity baseline bumped** to v3.2.0 in the
  numerical-consistency CI job; all parity tests hold to the
  previously documented tolerances (e.g. `|Δτ| < 4e-7` on CPS / PWT).
- **Documentation sweep** — `trop.sthlp` gains rowname and
  `estat triplerob` coverage, the README "Numerical robustness
  choices" section is now nine items (covers the unified threshold
  and the rowname helper), and `trop.pkg` no longer references files
  that were renamed or removed.

### Changed (`README.md`)
- **`README.md`** — Features bullet now advertises 8 `estat`
  subcommands (the new `estat triplerob` is listed in the
  post-estimation table with its decomposition formula).  The Stored-
  results / Matrices table for `e(alpha)` and `e(beta)` notes the
  attached row names.  The "Numerical robustness choices" section
  grew from 8 to 9 items: item 8 now describes the unified 5% LOOCV /
  bootstrap failure-rate threshold (previously only the bootstrap
  side was documented), and item 9 is new, covering the original-ID
  row-name injection.  README_CN.md is intentionally left untouched
  in this cycle — it will be re-aligned in a dedicated CN translation
  pass.

### Changed (`trop.sthlp` completeness audit)
- **`ado/trop.sthlp`** — the Stored-results entries for `e(alpha)` /
  `e(beta)` now document the original-ID row names attached by
  `_trop_attach_idnames`.  New Examples subsections demonstrate
  `matrix list e(alpha)` with the sanitised labels and walk through
  every `estat` subcommand (summarize / loocv / weights / factors /
  bootstrap / triplerob).  Sthlp syntax passes `do tests/check_ado_syntax.do`
  (29/29 ado files clean).

### Added (original-ID row names on `e(alpha)` and `e(beta)`)
- **`ado/_trop_attach_idnames.ado`**, **`ado/trop.ado`** — new private
  helper `_trop_attach_idnames` runs after estimation and rewrites
  `e(alpha)` (N x 1) and `e(beta)` (T x 1) with matrix row names drawn
  from the sorted unique values of the user's `panelvar` / `timevar` on
  the estimation sample.  The panelvar / timevar indexing used inside
  the plugin (`egen ... = group()`) is order-consistent with `levelsof`,
  so row `i` of `e(alpha)` corresponds to the `i`-th sorted unique
  panel identifier.  Users can now write `matrix list e(alpha)` and see
  the fixed effects keyed by their original identifiers; scalar access
  via `e(alpha)[i, 1]` continues to work.  Names are sanitised to valid
  Stata matrix identifiers (letters/digits/underscore, ≤ 32 chars,
  non-leading-digit), so numeric IDs `1, 2, …` render as `_1, _2, …`
  and identifiers containing `-`, `.`, space, `#`, etc. are rewritten
  with `_`.  On any mismatch (for example if `panelvar` has been
  dropped by a user script before the helper runs) the helper fails
  silently to keep the estimation pipeline non-fragile.  Pinned by
  `tests/test_alpha_beta_rownames.do`.

### Added (`estat triplerob` subcommand)
- **`ado/trop_estat_triplerob.ado`**,
  **`ado/trop_estat.ado`**,
  **`ado/trop_estat.sthlp`**,
  **`mata/trop_estat_helpers.mata`** — new post-estimation subcommand
  `estat triplerob` that reports a diagnostic decomposition of the paper
  Theorem 5.1 bias bound:

      |E[tauhat - tau | L]| <= ||Delta^u||_2 * ||Delta^t||_2 * ||B||_*

  The three factors are computed from the rank-k SVD of `e(factor_matrix)`,
  the method-specific weight vectors (`e(delta_time)`/`e(delta_unit)` for
  joint; `e(theta)`/`e(omega)` for twostep), and the discarded nuclear
  mass.  Supports `rank(#)` and `topk(#)` options.  Stores
  `r(delta_unit)`, `r(delta_time)`, `r(residual)`, `r(bias_bound)`,
  `r(rank)`, `r(method)` for programmatic use.  Help entry and a
  Methods-and-formulas block document the exact identities implemented.
  Pinned by `tests/test_estat_triplerob.do`.

### Fixed (`tests/check_ado_syntax.do` drift)
- **`tests/check_ado_syntax.do`** — stale Mata and ADO file lists
  replaced with the current set of 11 Mata modules and 28 ADO files.
  The previous list referenced six Mata files and several ADO files
  that had been renamed/removed, causing the syntax check to abort at
  `r(601)` before any file was inspected.  The new list is grouped by
  role (core command, pre-estimation helpers, estimation helpers, estat
  + predict, validators, private utilities) to make future sync easy.

### Changed (LOOCV failure-rate signalling)
- **`mata/trop_rust_interface.mata`** — `check_loocv_fail_rate()` now
  emits its advisory when the LOOCV failure rate exceeds **5 %** (was
  10 %), matching `_trop_display_bootstrap_warnings()` so LOOCV and
  bootstrap surface failures at a single, unified sensitivity.  On a
  panel with ~1,000 D=0 cells a 5 % LOOCV failure rate means ~50
  leave-one-out fits did not converge, which can bias the selected
  `(lambda_time, lambda_unit, lambda_nn)` triple off of Q(λ)'s true
  argmin (paper Eq. 5).  The 50 % abort threshold is unchanged
  (`rc=498`).  Banner format is also aligned with the bootstrap warning:
  both now include a `(n_valid/n_total successful)` denominator for
  ≤ 50 % failures and `n_failed of n_total` for > 50 % failures.
  Pinned by `tests/test_loocv_fail_rate_threshold.do`.

### Changed (numerical fidelity baseline)
- **`scripts/regenerate_py311_parity.py`**,
  **`.github/workflows/numerical-consistency.yml`**,
  **`trop_stata/tests/test_joint_outer_convergence_parity.do`**,
  **`trop_stata/tests/test_joint_exhaustive_parity.do`**,
  **`trop_stata/tests/test_joint_exhaustive_parity.py`**,
  **`trop_stata/tests/test_paper_table5_corners.do`**,
  **`trop_stata/README.md`** — numerical-parity baseline upgraded
  from v3.1.1 to v3.2.0.  Release 3.2.0 adds only
  diagnostic `UserWarning` emissions for the TROP solvers (PR #317
  non-convergence signalling; PR #324 bootstrap failure-rate guards) and
  does **not** change any numerical values; the regenerated
  `trop_stata/tests/reference/py311_parity.json` / `.csv` are bit-identical
  to the prior 3.1.1 snapshot.  Three parity regressions continue to pass
  at the documented tolerance on the refreshed baseline:
  `test_joint_outer_convergence_parity.do` (|Δτ| ≤ 3.76e-07 across
  CPS/PWT/simulated panels), `test_joint_exhaustive_parity.do`
  (|Δτ| = 5.42e-09, exact λ match), and `test_twostep_exhaustive_parity.do`
  (bit-equal across runs).  User-facing README text reworded to refer to
  "the released numerical baseline" without naming any external package.

### Fixed (bootstrap failure-rate signalling)
- **`mata/trop_ereturn_store.mata`** —
  `_trop_display_bootstrap_warnings()` now emits its advisory when the
  bootstrap failure rate exceeds **5 %** (was 10 %).  Eleven failures
  out of 200 replicates now surface instead of passing silently.  The
  50 % abort threshold is unchanged (`rc=504`).  Also fixes a latent
  Mata `printf` bug where the failure-rate warning used `%5.1f` without
  multiplying by 100, producing messages like `"0.1 %"` instead of
  `"10.0 %"`.  Pinned by `tests/test_bootstrap_fail_rate_threshold.do`.

### Fixed (joint LOOCV input validation)
- **`rust/src/loocv.rs`**, **`rust/src/lib.rs`** — every joint C-ABI
  entry (`stata_estimate_joint`, `stata_estimate_joint_weighted`,
  `stata_bootstrap_trop_variance_joint`,
  `stata_bootstrap_trop_variance_joint_weighted`,
  `stata_loocv_grid_search_joint`,
  `stata_loocv_cycling_search_joint`) now calls the shared
  `check_simultaneous_adoption(&d)` helper before any further work and
  short-circuits with `TropError::InvalidDimension` on staggered or
  non-absorbing `D`.  This is defence-in-depth: the Stata front-end
  already gates `method(joint)` on simultaneous adoption, but the
  Rust-side guard prevents mis-computed `treated_periods` if a future
  caller bypasses the ADO layer.  Pinned by five unit tests in
  `rust/src/loocv.rs` covering no-treatment, valid simultaneous,
  staggered start, treatment switch-off, and pre-`T_1` pulses.

### Changed (λ_nn = 0 closed-form documentation + invariant)
- **`rust/src/estimation.rs`** — the λ_nn = 0 branch of
  `estimate_model()` is now annotated with the paper Eq. 2 reduction
  and a `debug_assert` post-condition verifies that cells with
  `W == 0` retain their previous iterate (i.e. the closed form only
  writes to the weighted support).  This locks in the invariant that
  zero-weight cells must not pick up fabricated signal, even if the
  initial `L` is non-zero.  Pinned by
  `estimation::tests::test_lambda_nn_zero_closed_form_preserves_invalid_cells`.

### Added (bootstrap diagnostic field)
- **`mata/trop_ereturn_store.mata`**, **`ado/trop_bootstrap.ado`**,
  **`ado/trop_estat_bootstrap.ado`**, **`ado/trop.sthlp`** —
  `e(bootstrap_fail_rate)` is now persisted as a convenience scalar
  mirroring `e(loocv_fail_rate)`.  `estat bootstrap` surfaces the rate
  explicitly and prints an advisory when it exceeds 5 %.  Pinned by
  `tests/test_ereturn_fail_rate_coverage.do`.

### Fixed (inference correctness)
- **`mata/trop_ereturn_store.mata`**, **`mata/trop_main.mata`**,
  **`ado/trop_bootstrap.ado`** — the t-based reference distribution used
  for the primary p-value and confidence interval now reads its
  degrees of freedom from `N_1` (number of ever-treated units) rather
  than from the treated-cell count.  Algorithm 3 resamples units in
  stratified fashion, so `N_1` is the effective cluster count.  The
  previous `df = N_treated_cells - 1` was a latent bug: on a panel with
  one treated unit and `T_post = 10` treated cells it declared `df = 9`,
  inflating significance and narrowing the CI.  `e(df_r)` is now
  `max(1, N_1 - 1)` when `N_1 >= 2` and missing otherwise (normal
  fallback); `e(pvalue)` and the three stored CI pairs use the same
  reference distribution.  **Breaking**: re-running prior estimations
  on panels with few treated units will produce wider CIs and larger
  p-values.  Pinned by `tests/test_inference_df_is_treated_units.do`.

### Added (inference UX: primary CI selection)
- **`ado/trop.ado`**, **`ado/trop_bootstrap.ado`**,
  **`mata/trop_ereturn_store.mata`** — new option
  `cimethod(percentile | t | normal)` promotes one of three bootstrap
  CI candidates to the primary `e(ci_lower)`/`e(ci_upper)` pair.  The
  default is `percentile` whenever `bootstrap > 0` (Algorithm 3 step 6)
  and `t` otherwise.  When `bootstrap(0)` is paired with
  `cimethod(percentile)` the parser downgrades to `t` and emits a note;
  the downgrade trace is preserved as `e(cimethod) = "percentile->t"`.
- **`e()` additions** — `e(ci_lower_t)/e(ci_upper_t)/e(pvalue_t)`,
  `e(ci_lower_normal)/e(ci_upper_normal)/e(pvalue_normal)`,
  `e(ci_lower_percentile)/e(ci_upper_percentile)` are always persisted
  when defined so downstream code can switch primary CI without
  re-estimating.  `e(cimethod)` records the selected primary (with a
  `request->actual` trace on downgrade).  Pinned by
  `tests/test_cimethod_option.do`.

### Changed (display)
- **`ado/trop.ado`** — the post-estimation results table now labels the
  primary CI with a `[<cimethod>]` tag (e.g. `95% CI [percentile]`) and
  echoes the two non-primary CIs beneath it.  Previously only t-based
  and percentile CIs were shown, with t-based flagged as primary.

### Changed (documentation hygiene — 5th Implementation note)
- **`ado/trop.sthlp`** — the "Implementation notes" section now lists
  five Stata-specific robustness choices, adding a new bullet for the
  inference reference distribution (N_1 df, percentile-primary CI,
  cimethod option).  The `e(df_r)` entry in "Stored results" documents
  the N_1 basis explicitly and the new `e(ci_*)` scalars are listed
  alongside `e(cimethod)`.  `e(N_treat)/e(N_treated)/e(N_treated_obs)`
  now have cell-count-vs-unit-count semantics spelled out inline.

### Added (tests)
- **`tests/test_inference_df_is_treated_units.do`** — locks the N_1
  df derivation on a 2-treated-unit / 5-post-period panel
  (`e(df_r) == 1`, not 9).
- **`tests/test_cimethod_option.do`** — locks the four
  `cimethod()` modes (default, `t`, `normal`, percentile+bootstrap(0)
  downgrade) and invalid-value rejection.
- **`tests/test_estored_naming.do`** — locks the
  `e(N_treated)/e(N_treated_obs)/e(N_treated_units)` semantics across
  both `method(twostep)` and `method(joint)`.

### Changed (breaking: default lambda grid realignment)
- **`ado/_trop_set_grid.ado`**, **`mata/trop_main.mata`**, **`mata/trop_lambda_grid.mata`** —
  the `default` LOOCV lambda_nn grid is now the five-point log-decade
  ladder `(0, 0.01, 0.1, 1, 10)`, matching the released numerical baseline's
  actual default.  The old grid `(0, 0.01, 0.1, 1, 10, .)` additionally
  enumerated the DID/TWFE corner (`.` = +∞, L ≡ 0); that corner is now
  reserved for `grid_style(extended)` so the default preset yields
  reproducible LOOCV selections across implementations.
  Users who relied on the DID/TWFE corner being in the default grid
  should either switch to `grid_style(extended)` or supply it explicitly
  via `lambda_nn_grid(0 0.01 0.1 1 10 .)`.
  The `fine` preset is bumped in parallel to `(0, 0.01, 0.0316, 0.1,
  0.316, 1, 10)` (7 points).  New combination counts:
  `default 6x6x5 = 180`, `fine 7x7x7 = 343`, `extended 14x16x19 = 4,256`
  (extended is unchanged).
- **`ado/_trop_validate_params.ado`** — error message for `grid_style()`
  now prints the new combination counts.
- **`ado/trop.sthlp`** — Lambda-grid comparison table and `grid_style()`
  prose refreshed to match the new preset sizes.

### Added (tests)
- **`tests/test_default_grid_parity.do`** — pins the `grid_style(default)`
  layout (values, cardinality, coordinate-descent cycle size) and runs
  an end-to-end LOOCV estimation on a deterministic synthetic panel,
  asserting the selected lambda triple falls inside the default grid.

### Changed (CI)
- **`.github/workflows/numerical-consistency.yml`** — bumps the pinned
  numerical fidelity baseline from v2.1.9 to v3.1.1, which is
  the current ground truth consulted by the numerical-consistency checks.

### Changed (documentation hygiene)
- **`ado/*.ado`, `ado/*.sthlp`, `mata/*.mata`** — scrubbed user-facing
  package source of references to external tooling (upstream package
  names, external numerical libraries, etc.).
  Comments now describe behaviour in paper-algorithm terms (Eq. 2 /
  Algorithm 1-3) or Stata-internal terms.  Cross-implementation notes
  live in this `CHANGELOG.md` and in `scripts/` only.
- **`ado/trop.sthlp`** — the "Differences from the external baseline"
  viewer section is renamed to "Implementation notes" and rewritten to
  document the four Stata-specific robustness choices (FISTA restart
  disabled, `dgelsd` WLS, unit-distance cache, LOOCV tie-breaker)

### Fixed (external-baseline alignment, first-principles audit)
- **`rust/src/loocv.rs`** — `loocv_cycling_search_joint` no longer seeds
  the `lambda_nn` Stage-1 univariate search with `lambda_time = ∞`.  Fixing
  `lambda_time = ∞` collapsed all time weight onto the target period
  precisely when the search was trying to pick `lambda_nn`, biasing the
  Stage-1 seed that Stage-2 coordinate descent subsequently polishes.  The
  new seed uses `(lambda_time, lambda_unit) = (0, 0)`, matching both the
  twostep path (`loocv_grid_search`) and the released numerical baseline
  (v3.1.1, internal `_fit_local` path).  Previously-selected
  lambda triples on panels where the joint-cycling path actually engaged
  (e.g. small, non-convex `Q(λ)` surfaces like Basque / West Germany) may
  shift slightly; the exhaustive joint path is unchanged.
- **`mata/trop_lambda_grid.mata`** and **`mata/trop_main.mata`** — default
  `lambda_nn` grid now appends Stata missing (`.` = `+∞`) in both the
  `default` and `extended` presets so every Mata/ADO entry point exposes
  the same Cartesian product as `_trop_set_grid.ado`.  Previously
  `trop_get_lambda_grid("default", "nn")` returned 5 finite values while
  `_trop_set_grid.ado` returned 6 values including the DID/TWFE corner,
  so consumers that by-passed the ADO layer saw a silently smaller grid.
  `trop_validate_table2_coverage()` continues to accept the augmented
  grid without change.

### Added (external-baseline differences locked in by tests)
- **`tests/diagnostics/phase0_python_vs_stata_diagnostic.{py,do}`** —
  shared-panel diagnostic that runs the v3.1.1 numerical baseline
  (local + global) and the Stata `trop` command in three modes (twostep
  cycling, joint exhaustive, joint cycling), writing a diff table
  `diagnostics/diff_table.csv` (and the matching `*_baseline.csv`
  snapshot) for regression reviews.  Documents the four numerical
  departures from the external baseline that Stata preserves by design.
- **`tests/test_fista_restart_stability.do`** — tests the FISTA solver
  stability (O'Donoghue & Candès 2015 restart is disabled; the test
  verifies the solver produces stable results without it).
  Sweeps `lambda_nn ∈ {0, 0.01, 0.1, 1, 10}` and asserts that every
  λ returns a finite, bounded ATT and that the two corners `0` and `10`
  achieve hard `e(converged) == 1`.
- **`tests/test_dgelsd_rank_deficient_wls.do`** — pins the SVD-based
  minimum-norm solver (`dgelsd`) for weighted least squares against a
  deliberately collinear panel (units 5 and 6 are perfect clones).
  Verifies finite ATT and a strictly-positive bootstrap SE, which an
  external `lstsq` + `pinv` fallback can fail to deliver.
- **`tests/test_unit_distance_cache_equivalence.do`** — pins the
  `UnitDistanceCache` used inside LOOCV: (i) runs `trop` twice on the same
  panel and requires bit-equal `e(att)`, `e(loocv_score)`, and
  `e(lambda_nn)`; (ii) writes a Mata-computed reference distance matrix
  to `diagnostics/mata_unit_distance.csv` for future regression diffing.
- **`tests/test_loocv_tiebreak_determinism.do`** — pins the deterministic
  LOOCV tie-breaker (`better_candidate`, `TIE_TOL = 1e-10`): three
  independent runs on the Phase-0 panel produce bit-identical
  `e(att)/e(lambda_*)` and the `lambda_nn` tie is resolved to the larger
  value, as designed (Plan §2.1).

### Docs
- **`README.md`** — new "Differences from the external baseline" section
  enumerates the four Stata-only numerical choices (FISTA restart
  disabled, LAPACK `dgelsd`, `UnitDistanceCache`, deterministic LOOCV
  tie-breaker), pointing each to its locking test, and clarifies that
  survey-design features from the upstream side (pweight / strata / PSU /
  Rao-Wu rescaled bootstrap) are intentionally out of scope.

### Added (LOOCV determinism & grid diagnostics, Phase 1-A)
- **`twostep_loocv(cycling|exhaustive)` option** on `trop` mirrors the
  existing `joint_loocv()` knob.  `cycling` (default) preserves the
  historical coordinate-descent behaviour; `exhaustive` performs the full
  Cartesian search over the user/preset grid and is guaranteed to return
  the global grid minimum.  The exhaustive path is the recommended choice
  for small panels (e.g. `basque`, `germany`) where the LOOCV objective
  `Q(lambda)` is non-convex and coordinate descent can stall at local
  minima that differ across BLAS backends.
- **Deterministic LOOCV tie-breaker** (Rust core, both methods, both search
  strategies) now resolves near-ties via lexicographic preference
  `(largest lambda_nn, smallest lambda_time, smallest lambda_unit)`.  This
  eliminates the cross-BLAS lambda drift reported on the Basque / West
  Germany replications without biasing the estimator's statistical
  properties: the tie-breaker activates only when Q-values are within
  `1e-12` of the current best.
- **New `grid_style(fine)` preset** (7 x 7 x 8 = 392 combinations) sits
  between `default` (216) and `extended` (4,256).  It inserts half-decade
  steps (`0.0316`, `0.316`) into the critical 0.01-1 band of the default
  `lambda_nn` grid and a `0.3` point into the `lambda_time` / `lambda_unit`
  grids, substantially reducing the search gap that caused LOOCV to round
  to different local minima across platforms.
- **`estat loocv, stability` subcommand option** surfaces the LOOCV search
  strategy in use (cycling vs exhaustive), reports the size and range of
  each lambda grid, and warns when the selected lambda coincides with the
  lowest or highest finite grid point, suggesting the grid should be
  widened.  The `lambda_nn = .` (DID/TWFE) corner is recognised as a
  legitimate corner solution and is never flagged.
- **`e(twostep_loocv)` estimation macro** stores the chosen twostep LOOCV
  strategy alongside the existing `e(joint_loocv)`.

### Testing (LOOCV determinism & grid diagnostics)
- `trop_stata/tests/test_twostep_exhaustive_parity.do` — verifies on a
  20-unit, 15-period synthetic panel that (i) both strategies complete,
  (ii) exhaustive never scores worse than cycling, and (iii) exhaustive
  is deterministic across repeated runs.
- `trop_stata/tests/test_estat_loocv_stability.do` — verifies that
  `grid_style(fine)` produces the advertised dimensions, that the
  stability block correctly reports LOOCV strategy and grid sizes, and
  that a narrow user-supplied grid triggers the boundary-hit warning.

### Testing & audit (first-principles review, Phase 1)
- Line-by-line bit-exact audit of `trop_stata/rust` against the
  v3.1.1 numerical baseline rust core.  No numerical bugs identified; the five
  residual differences (FISTA restart disabled, LAPACK `dgelsd` for
  `solve_joint_no_lowrank`, stricter inner/outer convergence tests,
  `n_pre==0` edge case, missing survey-weight hook) are intentional
  optimisations or out-of-scope gaps, all documented in source comments.
- New regression test pair anchored to the v3.1.1 numerical baseline
  ground truth:
  - `tests/phase1_regen_baseline.py` (twostep / local method) and
    `tests/phase2_regen_joint_baseline.py` (joint / global method)
    regenerate the JSON baselines under the current numerical baseline,
    replacing the stale 2.1.9-era snapshot.
  - `tests/phase1_regression_small.do` and
    `tests/phase2_regression_joint.do` compare the Stata ATT and
    per-cell tau against the refreshed baselines.  All five numerical
    anchors now pass at `|Delta| < 4e-7`, matching the project's
    advertised tolerance.
- `weights.rs::compute_joint_weight_vectors`: the `n_pre == 0` branch
  (previously silent uniform-weight fallback) now carries an explicit
  source comment explaining that the upstream baseline raises
  `ValueError`, while the Stata pipeline rejects such panels upstream
  via `_trop_chk_common_ctrl_periods`.  New unit test
  `test_joint_weights_zero_pre_periods_returns_finite` pins the
  fallback behaviour so any future refactor must change it
  deliberately.
- `loocv.rs`: doc comment for `loocv_grid_search_joint_exhaustive`
  corrected — the v2.1.9-era fallback phrasing was replaced with
  the current 3.1.1 contract.

### Added (first-principles alignment with the paper & numerical baseline v3.1.1)
- Default `lambda_nn_grid` now appends `.` (Stata missing = `+inf`) so that
  LOOCV explores the "no factor structure" corner of the paper's
  regularisation hypercube, mirroring the external baseline's default
  `lambda_nn_grid = [inf] + log-spaced ladder`.  At `lambda_nn = inf` the
  nuclear-norm proximal step returns the zero matrix, giving the classical
  DID/SC estimator as a LOOCV-selectable special case.
- `lambda_nn_grid(.)` and `fixedlambda(... .)` continue to accept `.` as
  the infinity marker.  Internally this is mapped to the Rust `f64::INFINITY`
  sentinel; the full-diagnostic LOOCV path and the bootstrap pipelines
  handle it without special-casing in user code.
- Rust core: `UnitDistanceCache` caches the unrestricted pairwise
  `Σ_u 1{D=0}(Y_{ui}-Y_{uj})^2` and `Σ_u 1{D=0}` sums, then answers each
  Eq. (3) query `dist_{−t}(j, i)` in O(1) by subtracting the single
  period `t`'s contribution.  Replaces the per-call O(T) pass inside
  `compute_unit_distance_for_obs` used by every weight-matrix evaluation
  during LOOCV and per-observation twostep estimation.
- Rust core: `loocv_score_for_params_full_diagnostic` mirrors the existing
  short-circuit `loocv_score_for_params` but iterates through every
  control observation, recording the full set of failing (period, unit)
  pairs.  The public `loocv_grid_search` now uses this path for its
  **final evaluation** only — so `e(loocv_n_valid)` always reflects every
  successful fit rather than "all successes up to the first failure".

### Changed (first-principles alignment)
- `lambda_time_grid()` and `lambda_unit_grid()` now reject Stata missing
  values at parse time with a clear numlist error (`invalid numlist has
  missing values`).  Unit / time weights are defined as
  `exp(-lambda_time * |t_1 - t_2|)` and
  `exp(-lambda_unit * dist^unit(j, i))` in the paper (Section 4.1), which
  degenerate to `1 * {t=t_0}` / `1 * {j=i}` when `lambda -> inf` — neither
  the released numerical baseline nor the paper contemplate those
  corners for the time / unit grids, so allowing them silently produced
  estimator output the paper would classify as undefined.  `.` remains
  permitted for `lambda_nn_grid()` and the third `fixedlambda` slot.
- FISTA restart safeguard is now **disabled** for both
  `method(twostep)` and `method(joint)` to match the Python reference
  implementation (`diff-diff` v3.1.1) which does not use restart.
  The O'Donoghue & Candès (2015) monotone-descent restart was
  previously enabled by default but fired too aggressively on small
  panels, preventing convergence.  The `__trop_fista_restart` global
  is no longer consulted.
- `check_loocv_fail_rate()` now prints the first failing LOO (period, unit)
  alongside the failure-rate percentage, so users hitting the
  `>10%` warning or `>50%` abort can immediately cross-reference
  `e(panelvar)`, `e(timevar)`, and the estimation sample instead of
  chasing the indices through `e(loocv_first_failed_t)` / `..._i`
  manually.  Indices are displayed 1-based to match Stata conventions.

### Added
- `method()` now accepts the paper-consistent aliases `method(local)` (= `twostep`)
  and `method(global)` (= `joint`).  Typos and unknown methods are rejected at
  parse time with a helpful error message (r(198)).
- Per-observation convergence diagnostics for `method(twostep)`:
  - Returned matrices `e(converged_by_obs)` (N_treated x 1; 1 = converged,
    0 = maxiter hit, -1 = solver failure) and `e(n_iters_by_obs)` (iterations
    used per treated cell).
  - When any cell fails to converge, the results table lists the first five
    offending (tau, index) pairs so users can target `maxiter()` or
    `lambda_nn` adjustments without inspecting the matrices manually.
- FISTA monotonicity safeguard inside the nuclear-norm proximal solver:
  the solver relies on standard FISTA momentum without adaptive restart.
  Plain FISTA without restart matches the reference Python
  implementation.  The solver remains stable across the full
  `lambda_nn` range tested in `test_fista_restart_stability.do`.
- `fixedlambda(lambda_time lambda_unit .)` now accepts Stata missing (`.`)
  in the third slot as a request for an effectively infinite `lambda_nn`
  (10^10), mirroring the external baseline's infinity-marker path.  The
  first two slots continue to require finite non-negative numbers; a
  missing value there is rejected with a pointer back to Eq. 3 of the paper.

### Fixed
- `trop.ado` set the internal macro `__trop_joint_loocv_mode` via Stata's
  `global` command, which rejects identifiers starting with an underscore
  (r(198)) and caused every estimation path to fail with `invalid syntax`.
  The macro is now installed via Mata's `st_global()`, which has no such
  restriction.  Regression covered by the updated `_smoke.do` run.
- `trop_bootstrap` passed seven arguments to `trop_prepare_options`, which
  expects six, so every post-estimation bootstrap call aborted with r(3001).
  The stray leading zero has been removed.
- `e(level)` from both `trop` and `trop_bootstrap` is now reported in the
  Stata percent convention (e.g. 95) rather than the probability form
  (e.g. 0.95) previously written by the plugin.
- `_trop_build_tau_matrix` no longer crashes when the touse tempvar has
  been consumed by `ereturn post … esample(var)`; `trop.ado` clones the
  tempvar for posting so that the original remains accessible to Mata.
  A secondary guard uses `_st_varindex()` to bail out silently if any of
  the referenced variables has been dropped.
- `_trop_build_tau_matrix` contained a stray adjacent string-literal pair
  (Mata does not concatenate such literals) that prevented the function
  from compiling; it was silently omitted from `ltrop.mlib` and the
  downstream call reported "_trop_build_tau_matrix() not found".  The two
  fragments have been merged into one format string.

### Added (pre-existing pipeline improvements)
- New `joint_loocv(cycling|exhaustive)` option on `trop` selects the LOOCV
  search strategy when `method(joint)` is combined with grid-based LOOCV.
  - `cycling` (default) keeps the existing two-stage coordinate-descent
    behaviour adapted from Footnote 2 of Athey, Imbens, Qu & Viviano (2025);
    cost O(|grid| * cycles).
  - `exhaustive` evaluates the full Cartesian product of
    `(lambda_time, lambda_unit, lambda_nn)` in parallel, matching the
    v3.1.1 numerical baseline (global path) bit-for-bit and guaranteeing
    the global LOOCV minimum over the given grid; cost O(|grid|^3).
  - Returned via `e(joint_loocv)`.  Rust core exposes
    `stata_loocv_grid_search_joint_exhaustive` alongside the existing cycling
    entry point.
  - End-to-end parity against numerical baseline v3.1.1 verified: identical (lambda_time,
    lambda_unit, lambda_nn) triple selected and |Delta ATT| < 1e-8 on a
    shared synthetic panel (`tests/test_joint_exhaustive_parity.{py,do}`).
- Bootstrap percentile CI from paper Algorithm 3 now surfaced through the
  ADO interface and the results display:
  - Return scalars `e(ci_lower_percentile)` / `e(ci_upper_percentile)`
    populated whenever bootstrap inference ran successfully.
  - Results table shows both the t-based CI (primary; unchanged backward
    compatibility) and the percentile CI side by side, labelled
    `95% CI (t-based)` and `95% CI (percentile)` respectively.
  - Rust bootstrap pipelines already computed the percentile bounds; this
    release propagates them through the C bridge, Mata `ereturn_store`
    layer, and `trop.ado`.

### Changed
- Default `bootstrap(#)` raised from 0 to 200 so that a bare call to
  `trop y d, panelvar(id) timevar(t)` now returns a bootstrap SE and CI
  out of the box, matching paper Algorithm 3 and the upstream default.
  Users who explicitly request `bootstrap(0)` still get the point
  estimate with inference skipped.
- `e(tau)` is now always stored as an N_treated x 1 vector of per-cell
  treatment effects, both for `method(twostep)` (unchanged) and for
  `method(joint)` (previously a 1x1 scalar).  In the joint case the
  vector is the single scalar tau replicated, so `mean(e(tau))` equals
  `e(att)` to machine precision for both methods.  The T x N matrix view
  is exposed via `e(tau_matrix)` when the panel metadata is available.

### Removed (BREAKING)
- `max_loocv_samples()` option dropped from `trop`, along with associated
  `e(max_loocv_samples)`, `e(loocv_subsampled)`, and `e(loocv_n_control_total)`
  return scalars.
  - LOOCV now always sums over every D=0 cell, in strict accordance with paper
    Eq. 5.  Subsampling could not provide a 100% guarantee that the selected
    lambda matches the full LOOCV minimum (in heavy-tailed settings the
    argmin can flip), and the option had no counterpart in the upstream
    numerical baseline.
  - Scripts that explicitly pass `max_loocv_samples()` will fail to parse;
    remove the option to restore correct behaviour.  The default behaviour
    (no subsampling) is preserved.

### Docs
- `trop.sthlp` updated:
  - Syntax table and dialog section document the new `joint_loocv()` option.
  - New Bootstrap Inference subsection explains the two CIs side by side.
  - Corrected the `bootstrap(#)` default from "0 (disabled)" to the actual
    value "200" (paper Algorithm 3 default).
  - Added an example invoking `joint_loocv(exhaustive)`.
  - Stored-results list now contains `e(joint_loocv)` and clarifies that
    `e(ci_lower)` / `e(ci_upper)` are the primary t-based interval while
    `e(ci_lower_percentile)` / `e(ci_upper_percentile)` hold the paper
    Alg 3 percentile interval.

## [1.0.0] - 2026-04-12

### Added
- Initial release of TROP Stata package
- Twostep estimation method with triply robust properties
- Joint estimation method for simultaneous optimization
- Precompiled plugin for macOS ARM64; other platforms buildable from Rust source
- Bootstrap inference with parallel computation
- LOOCV cross-validation for regularization selection
- Comprehensive predict functionality (y0, y1, te, residuals, mu, alpha, beta)
- estat subcommands (summarize, weights, loocv, factors, bootstrap, sensitivity, vce)
- Full help documentation

### Fixed
- Numerical consistency verified against the numerical fidelity baseline

### Notes
- Requires Stata 17.0 or higher
- Rust-based computational backend for performance

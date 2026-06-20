"""Generate simulated panel data with survey design structure for Rao-Wu bootstrap testing."""
import numpy as np
import pandas as pd


def generate_survey_panel():
    """
    Generate panel data with complex survey design.

    Structure:
    - N=50 units, T=20 periods
    - 5 strata (10 units per stratum)
    - Each unit is its own PSU (implicit PSU)
    - 10 treated units (2 per stratum, last 5 periods treated)
    - Unequal probability weights: w_i ~ Gamma(2, 1)
    - FPC: N_h = 20 per stratum (sampling rate 50%)

    Outcome: Y_{it} = alpha_i + beta_t + L_{it} + tau*D_{it} + eps
    where tau = 0.5 (true ATT)
    """
    np.random.seed(42)

    N = 50
    T = 20
    n_strata = 5
    units_per_stratum = N // n_strata
    n_treated_per_stratum = 2
    treatment_start = 15  # periods 15-19 are post-treatment
    true_att = 0.5

    # Unit effects
    alpha = np.random.normal(0, 1, N)
    # Time effects
    beta = np.cumsum(np.random.normal(0, 0.1, T))
    # Low-rank factor (rank 2)
    gamma = np.random.normal(0, 0.5, (N, 2))
    lam = np.random.normal(0, 0.5, (T, 2))
    L = lam @ gamma.T  # T x N

    # Treatment assignment: 2 units per stratum
    treated_units = []
    for s in range(n_strata):
        start_idx = s * units_per_stratum
        # Select first 2 units in each stratum as treated
        treated_units.extend([start_idx, start_idx + 1])

    # Build panel
    records = []
    for i in range(N):
        stratum = i // units_per_stratum
        for t in range(T):
            d = 1 if (i in treated_units and t >= treatment_start) else 0
            y = alpha[i] + beta[t] + L[t, i] + true_att * d + np.random.normal(0, 0.3)
            records.append({
                'unit': i,
                'time': t,
                'y': y,
                'd': d,
                'stratum': stratum,
                'psu': i,  # each unit is its own PSU
                'weight': np.random.gamma(2, 1),  # will be set per-unit below
                'fpc': units_per_stratum * 2,  # N_h = 20, sampling 10/20 = 50%
            })

    df = pd.DataFrame(records)

    # Make weights constant within unit
    unit_weights = np.random.default_rng(123).gamma(2, 1, N)
    for i in range(N):
        df.loc[df['unit'] == i, 'weight'] = unit_weights[i]

    return df


if __name__ == '__main__':
    df = generate_survey_panel()
    # Save as CSV for both Python and Stata
    df.to_csv('/Users/cxy/Desktop/2026project/trop/trop_stata/tests/diagnostics/survey_test_panel.csv', index=False)
    # Save as Stata format
    df.to_stata('/Users/cxy/Desktop/2026project/trop/trop_stata/data/survey_test_panel.dta', write_index=False)
    print(f"Generated survey test panel: {len(df)} obs, {df['unit'].nunique()} units, {df['time'].nunique()} periods")
    print(f"Strata: {df['stratum'].nunique()}, Treated units: {df[df['d']==1]['unit'].nunique()}")

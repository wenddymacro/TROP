"""
Generate Python Rao-Wu Bootstrap reference values for Stata parity testing.

Runs Python TROP with survey_design to get Rao-Wu SE, saves reference values.
"""
import json
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/Users/cxy/Desktop/2026project/trop/diff-diff-main')

from generate_survey_test_data import generate_survey_panel


def run_trop_with_survey():
    """Run TROP with various survey design configurations and collect reference values."""
    from diff_diff import TROP
    from diff_diff.survey import SurveyDesign

    df = generate_survey_panel()
    results = {}

    # Test 1: pweight only (should match simple weighted bootstrap)
    print("Test 1: pweight only...")
    sd1 = SurveyDesign(weights='weight', weight_type='pweight')
    trop1 = TROP(method='local', n_bootstrap=200, seed=42,
                 lambda_time_grid=[0.0, 1.0], lambda_unit_grid=[0.0, 1.0],
                 lambda_nn_grid=[0.1, 1.0])
    r1 = trop1.fit(df, outcome='y', treatment='d', unit='unit', time='time',
                   survey_design=sd1)
    results['pweight_only'] = {
        'att': float(r1.att),
        'se': float(r1.se),
        'lambda_time': float(r1.lambda_time),
        'lambda_unit': float(r1.lambda_unit),
        'lambda_nn': float(r1.lambda_nn),
    }
    print(f"  ATT={r1.att:.6f}, SE={r1.se:.6f}")

    # Test 2: pweight + strata
    print("Test 2: pweight + strata...")
    sd2 = SurveyDesign(weights='weight', strata='stratum', weight_type='pweight')
    trop2 = TROP(method='local', n_bootstrap=200, seed=42,
                 lambda_time_grid=[0.0, 1.0], lambda_unit_grid=[0.0, 1.0],
                 lambda_nn_grid=[0.1, 1.0])
    r2 = trop2.fit(df, outcome='y', treatment='d', unit='unit', time='time',
                   survey_design=sd2)
    results['pweight_strata'] = {
        'att': float(r2.att),
        'se': float(r2.se),
        'lambda_time': float(r2.lambda_time),
        'lambda_unit': float(r2.lambda_unit),
        'lambda_nn': float(r2.lambda_nn),
    }
    print(f"  ATT={r2.att:.6f}, SE={r2.se:.6f}")

    # Test 3: pweight + strata + PSU + FPC
    print("Test 3: pweight + strata + PSU + FPC...")
    sd3 = SurveyDesign(weights='weight', strata='stratum', psu='psu',
                       fpc='fpc', weight_type='pweight')
    trop3 = TROP(method='local', n_bootstrap=200, seed=42,
                 lambda_time_grid=[0.0, 1.0], lambda_unit_grid=[0.0, 1.0],
                 lambda_nn_grid=[0.1, 1.0])
    r3 = trop3.fit(df, outcome='y', treatment='d', unit='unit', time='time',
                   survey_design=sd3)
    results['pweight_strata_psu_fpc'] = {
        'att': float(r3.att),
        'se': float(r3.se),
        'lambda_time': float(r3.lambda_time),
        'lambda_unit': float(r3.lambda_unit),
        'lambda_nn': float(r3.lambda_nn),
    }
    print(f"  ATT={r3.att:.6f}, SE={r3.se:.6f}")

    # Test 4: Joint method with strata
    print("Test 4: Joint method + strata...")
    sd4 = SurveyDesign(weights='weight', strata='stratum', weight_type='pweight')
    trop4 = TROP(method='global', n_bootstrap=200, seed=42,
                 lambda_time_grid=[0.0, 1.0], lambda_unit_grid=[0.0, 1.0],
                 lambda_nn_grid=[0.1, 1.0])
    r4 = trop4.fit(df, outcome='y', treatment='d', unit='unit', time='time',
                   survey_design=sd4)
    results['joint_strata'] = {
        'att': float(r4.att),
        'se': float(r4.se),
        'lambda_time': float(r4.lambda_time),
        'lambda_unit': float(r4.lambda_unit),
        'lambda_nn': float(r4.lambda_nn),
    }
    print(f"  ATT={r4.att:.6f}, SE={r4.se:.6f}")

    # Save reference values
    output_path = '/Users/cxy/Desktop/2026project/trop/trop_stata/tests/reference/rao_wu_parity.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nReference values saved to: {output_path}")

    return results


if __name__ == '__main__':
    run_trop_with_survey()

"""
Leave-one-N-out cross-validation for the phase-separated cost models.
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')


MASTER_CSV     = 'master_tidy.csv'
FG_DEDUP_KEY   = ['N', 'capacity', 'error_rate', 'flip_probability']
HELD_OUT_SIZES = [2000, 5000, 10000, 20000]

# Same formulas as fit_models.py — single source of truth would be
# better, but for verification purposes restating them here is clearer.
FORMULAS = {
    'fg': ('log_fg_energy ~ log_N + log_bf_bits + log_bf_hashes'),
    'nn': ('log_nn_energy ~ log_N + log_epochs + log_num_hidden_layers '
           '+ C(neurons_cat, Treatment("4"))'),
}

RESPONSE_COLS = {
    'fg': 'log_fg_energy',
    'nn': 'log_nn_energy',
}

PHASE_LABELS = {
    'fg': 'Feature-generation',
    'nn': 'NN training + evaluation',
}



def load_master_and_derive_predictors():

    df = pd.read_csv(MASTER_CSV)

    df['bf_bits']   = np.ceil(-df['capacity'] * np.log(df['error_rate']) / np.log(2)**2)
    df['bf_hashes'] = np.ceil(-np.log(df['error_rate']) / np.log(2))

    for col in ['N', 'epochs', 'num_hidden_layers',
                'num_of_neurons_in_hidden_layer', 'bf_bits', 'bf_hashes']:
        df[f'log_{col}'] = np.log(df[col])

    df['log_fg_energy'] = np.log(df['fg_energy_mean'])
    df['log_nn_energy'] = np.log(df['nn_energy_mean'])

    df['neurons_cat'] = df['num_of_neurons_in_hidden_layer'].astype(str)
    df['layers_cat']  = df['num_hidden_layers'].astype(str)

    print(f"[load] master loaded: {df.shape}")
    return df



def prepare_fg_data(df):

    fg = df.drop_duplicates(subset=FG_DEDUP_KEY).copy()
    assert len(fg) == 48
    print(f"[prepare_fg] {len(fg)} unique fg measurements")
    return fg


def prepare_nn_data(df):

    print(f"[prepare_nn] {len(df)} NN measurements")
    return df



def lonocv_one_fold(data, formula, response_col, held_out_N):

    train = data[data['N'] != held_out_N]
    test  = data[data['N'] == held_out_N]

    model = smf.ols(formula, data=train).fit()
    pred_log = model.predict(test)
    actual_log = test[response_col]

    log_residuals = pred_log - actual_log
    log_rmse = float(np.sqrt((log_residuals ** 2).mean()))

    pct_errs = (np.exp(log_residuals) - 1).abs() * 100

    return {
        'held_out_N':    int(held_out_N),
        'train_n':       int(len(train)),
        'test_n':        int(len(test)),
        'test_type':     classify_test(held_out_N),
        'log_rmse':      log_rmse,
        'pct_err_median':  float(pct_errs.median()),
        'pct_err_p90':     float(pct_errs.quantile(0.9)),
    }


def classify_test(held_out_N):

    lo, hi = min(HELD_OUT_SIZES), max(HELD_OUT_SIZES)
    if held_out_N == lo:
        return 'extrapolate down'
    if held_out_N == hi:
        return 'extrapolate up'
    return 'interpolate'



def lonocv_for_phase(data, phase):

    formula      = FORMULAS[phase]
    response_col = RESPONSE_COLS[phase]
    rows = []
    for held_out_N in HELD_OUT_SIZES:
        result = lonocv_one_fold(data, formula, response_col, held_out_N)
        result['phase'] = PHASE_LABELS[phase]
        rows.append(result)
    return pd.DataFrame(rows)



def print_combined_table(fg_table, nn_table):

    print(f"\n{'='*92}")
    print("Leave-one-N-out cross-validation, phase-separated cost models")
    print(f"{'='*92}")
    header = (f"{'Phase':28s}  {'Held-out N':>10s}  {'Train n':>7s}  "
              f"{'Test n':>6s}  {'Test type':<18s}  "
              f"{'log-RMSE':>9s}  {'Median |%err|':>13s}  {'90th |%err|':>12s}")
    print(header)
    print('-' * len(header))

    combined = pd.concat([fg_table, nn_table], ignore_index=True)
    for _, r in combined.iterrows():
        print(f"{r['phase']:28s}  {r['held_out_N']:>10d}  {r['train_n']:>7d}  "
              f"{r['test_n']:>6d}  {r['test_type']:<18s}  "
              f"{r['log_rmse']:>9.3f}  {r['pct_err_median']:>12.1f}%  "
              f"{r['pct_err_p90']:>11.1f}%")
    return combined



def main():
    df      = load_master_and_derive_predictors()
    fg_data = prepare_fg_data(df)
    nn_data = prepare_nn_data(df)

    fg_table = lonocv_for_phase(fg_data, 'fg')
    nn_table = lonocv_for_phase(nn_data, 'nn')

    combined = print_combined_table(fg_table, nn_table)
    combined.to_csv('lonocv_results.csv', index=False)
    print(f"\nResults saved to lonocv_results.csv")
    return combined


if __name__ == '__main__':
    main()

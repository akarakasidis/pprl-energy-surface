"""
Consolidate the five raw experimental files into a single tidy dataframe.
"""
import pandas as pd
import numpy as np
import re


EXPECTED_N_FG_CONFIGS = 48          # 4 dataset sizes x 2 cap x 2 err x 3 flip
EXPECTED_N_NN_CONFIGS = 1296        # 48 x 3 layers x 3 neurons x 3 epochs
EXPECTED_REPLICATES_PER_CONFIG = 3  # for the precision/recall file


CONFIG_KEY = [
    'N', 'faults_per_record', 'capacity', 'error_rate', 'flip_probability',
    'num_hidden_layers', 'num_of_neurons_in_hidden_layer', 'epochs',
]


FG_KEY = ['N', 'faults_per_record', 'capacity', 'error_rate', 'flip_probability']


def step1_load_precision_recall(path):
    df = pd.read_csv(path, sep='\t')
    print(f"[step1] loaded {len(df)} replicate rows")
    return df



def step2_compute_quality_metrics(pr):

    pr = pr.copy()

    # precision = TP / (TP + FP), with 0/0 -> 0 by convention
    denom_p = pr['TP'] + pr['FP']
    pr['precision'] = np.where(denom_p > 0, pr['TP'] / denom_p, 0.0)

    # recall = TP / (TP + FN), with 0/0 -> 0 by convention
    denom_r = pr['TP'] + pr['FN']
    pr['recall'] = np.where(denom_r > 0, pr['TP'] / denom_r, 0.0)

    # F1 = 2PR / (P+R), with 0/0 -> 0
    denom_f = pr['precision'] + pr['recall']
    pr['f1'] = np.where(denom_f > 0,
                        2 * pr['precision'] * pr['recall'] / denom_f,
                        0.0)

    print(f"[step2] added precision/recall/f1 columns")
    return pr



def _extract_N_from_filename(s):
    """Extract dataset size from a filename like 'POW_A_10000.csv-POW_B_1_10000.csv'."""
    m = re.search(r'POW_A_(\d+)', s)
    if m is None:
        raise ValueError(f"Cannot parse dataset size from: {s!r}")
    return int(m.group(1))


def step3_extract_N_from_dataset_name(pr):
    """Add an N column by parsing the dataset filename."""
    pr = pr.copy()
    pr['N'] = pr['dataset'].apply(_extract_N_from_filename)
    print(f"[step3] dataset sizes found: {sorted(pr['N'].unique())}")
    return pr



def step4_aggregate_replicates_to_configs(pr):

    aggregated = pr.groupby(CONFIG_KEY).agg(
        n_replicates   = ('precision', 'size'),
        precision_mean = ('precision', 'mean'),
        precision_std  = ('precision', 'std'),
        recall_mean    = ('recall',    'mean'),
        recall_std     = ('recall',    'std'),
        f1_mean        = ('f1',        'mean'),
        f1_std         = ('f1',        'std'),
        TP_mean        = ('TP',        'mean'),
        FP_mean        = ('FP',        'mean'),
        TN_mean        = ('TN',        'mean'),
        FN_mean        = ('FN',        'mean'),
    ).reset_index()

    print(f"[step4] aggregated to {len(aggregated)} configurations")
    print(f"        replicate counts present: {sorted(aggregated['n_replicates'].unique())}")
    return aggregated


PERUN_METRICS_REQUIRED = {
    'runtime_mean':  'metrics.runtime.mean',
    'runtime_std':   'metrics.runtime.std',
    'energy_mean':   'metrics.energy.mean',
    'energy_std':    'metrics.energy.std',
    'power_mean':    'metrics.power.mean',
    'dram_mem_mean': 'metrics.dram_mem.mean',
}

PERUN_METRICS_OPTIONAL = {
    'cpu_energy_mean': 'metrics.cpu_energy.mean',
    'cpu_power_mean':  'metrics.cpu_power.mean',
    'co2_mean':        'metrics.co2.mean',
    'money_mean':      'metrics.money.mean',
}


def _select_perun_columns(df, phase_prefix):
 
    rename_map = {}
    for short_name, full_name in PERUN_METRICS_REQUIRED.items():
        if full_name not in df.columns:
            raise KeyError(f"Required column {full_name!r} missing from input")
        rename_map[full_name] = f'{phase_prefix}_{short_name}'
    for short_name, full_name in PERUN_METRICS_OPTIONAL.items():
        if full_name in df.columns:
            rename_map[full_name] = f'{phase_prefix}_{short_name}'
    return df.rename(columns=rename_map)


def _parse_fg26_experiment_string(s):
    """
        Format: '{faults}_{cap}_{err}_{flip}_{batch}_POW_A_{N}.csv_POW_B_{faults}_{N}.csv'
    """
    pattern = (r'(\d+)_(\d+)_([\d\.]+)_([\d\.]+)_(\d+)'
               r'_POW_A_(\d+)\.csv_POW_B_\d+_\d+\.csv')
    m = re.match(pattern, s)
    if m is None:
        raise ValueError(f"Cannot parse fg experiment string: {s!r}")
    return pd.Series({
        'faults_per_record': int(m.group(1)),
        'capacity':          int(m.group(2)),
        'error_rate':        float(m.group(3)),
        'flip_probability':  float(m.group(4)),
        'batch_size':        int(m.group(5)),
        'N':                 int(m.group(6)),
    })


def step5_load_and_merge_feature_gen_files(path_2025, path_2026):
    fg25 = pd.read_csv(path_2025).assign(N=10000)

    fg26 = pd.read_csv(path_2026)
    parsed = fg26['experiment'].apply(_parse_fg26_experiment_string)
    fg26 = pd.concat([fg26, parsed], axis=1)

    fg25 = _select_perun_columns(fg25, phase_prefix='fg')
    fg26 = _select_perun_columns(fg26, phase_prefix='fg')

    keep_cols = FG_KEY + [c for c in fg25.columns if c.startswith('fg_')]
    fg25 = fg25[keep_cols]
    keep_cols = FG_KEY + [c for c in fg26.columns if c.startswith('fg_')]
    fg26 = fg26[keep_cols]

    fg = pd.concat([fg25, fg26], ignore_index=True)

    for col in ['N', 'faults_per_record', 'capacity']:
        fg[col] = fg[col].astype(int)

    print(f"[step5] fg loaded: 2025={len(fg25)}, 2026={len(fg26)}, total={len(fg)}")
    return fg

def _parse_nn26_experiment_string(s):

    pattern = (r'(\d+)_(\d+)_([\d\.]+)_([\d\.]+)_(\d+)_(\d+)_(\d+)_(\d+)'
               r'_POW_A_(\d+)\.csv-POW_B_\d+_\d+\.csv')
    m = re.match(pattern, s)
    if m is None:
        raise ValueError(f"Cannot parse nn experiment string: {s!r}")
    return pd.Series({
        'faults_per_record':              int(m.group(1)),
        'capacity':                       int(m.group(2)),
        'error_rate':                     float(m.group(3)),
        'flip_probability':               float(m.group(4)),
        'num_hidden_layers':              int(m.group(5)),
        'num_of_neurons_in_hidden_layer': int(m.group(6)),
        'epochs':                         int(m.group(7)),
        'batch_size':                     int(m.group(8)),
        'N':                              int(m.group(9)),
    })


def step6_load_and_merge_nn_files(path_2025, path_2026):

    nn25 = pd.read_csv(path_2025).copy().assign(N=10000)

    nn26 = pd.read_csv(path_2026)
    parsed = nn26['experiment'].apply(_parse_nn26_experiment_string)
    nn26 = pd.concat([nn26, parsed], axis=1)

    nn25 = _select_perun_columns(nn25, phase_prefix='nn')
    nn26 = _select_perun_columns(nn26, phase_prefix='nn')

    keep_cols = CONFIG_KEY + [c for c in nn25.columns if c.startswith('nn_')]
    nn25 = nn25[keep_cols]
    keep_cols = CONFIG_KEY + [c for c in nn26.columns if c.startswith('nn_')]
    nn26 = nn26[keep_cols]

    nn = pd.concat([nn25, nn26], ignore_index=True)

    for col in ['N', 'faults_per_record', 'capacity', 'num_hidden_layers',
                'num_of_neurons_in_hidden_layer', 'epochs']:
        nn[col] = nn[col].astype(int)

    print(f"[step6] nn loaded: 2025={len(nn25)}, 2026={len(nn26)}, total={len(nn)}")
    return nn



def step7_join_master_table(pr_aggregated, nn, fg):
     master = pr_aggregated.merge(nn, on=CONFIG_KEY, how='outer',
                                 validate='one_to_one')
    master = master.merge(fg, on=FG_KEY, how='left',
                          validate='many_to_one')
    print(f"[step7] master table: {len(master)} rows, {len(master.columns)} cols")
    return master


def step8_add_derived_totals(master):
    master = master.copy()
    master['total_energy_mean']  = master['fg_energy_mean']  + master['nn_energy_mean']
    master['total_runtime_mean'] = master['fg_runtime_mean'] + master['nn_runtime_mean']

    for metric in ['co2', 'money']:
        fg_col = f'fg_{metric}_mean'
        nn_col = f'nn_{metric}_mean'
        if fg_col in master.columns and nn_col in master.columns:
            master[f'total_{metric}_mean'] = master[fg_col] + master[nn_col]

    print(f"[step8] derived columns added")
    return master



def step9_sanity_checks(master):
    n_rows = len(master)
    assert n_rows == EXPECTED_N_NN_CONFIGS, (
        f"Expected {EXPECTED_N_NN_CONFIGS} rows, got {n_rows}"
    )

    # Each dataset size should contribute exactly 324 rows
    rows_per_N = master.groupby('N').size().to_dict()
    for N, count in rows_per_N.items():
        assert count == 324, f"N={N} has {count} rows, expected 324"

    # Core output columns should have no missing values
    for col in ['precision_mean', 'recall_mean', 'f1_mean',
                'fg_energy_mean', 'nn_energy_mean', 'total_energy_mean']:
        n_missing = master[col].isna().sum()
        assert n_missing == 0, f"{col} has {n_missing} missing values"

    # All configurations should have 3 replicates
    rep_counts = master['n_replicates'].value_counts().to_dict()
    assert rep_counts == {EXPECTED_REPLICATES_PER_CONFIG: n_rows}, (
        f"Unexpected replicate counts: {rep_counts}"
    )

    print(f"[step9] all checks passed: "
          f"{n_rows} rows, 324 per dataset size, no missing core values")



def main(
    path_precision_recall,
    path_fg_2025, path_fg_2026,
    path_nn_2025, path_nn_2026,
    path_output,
):
    pr  = step1_load_precision_recall(path_precision_recall)
    pr  = step2_compute_quality_metrics(pr)
    pr  = step3_extract_N_from_dataset_name(pr)
    pra = step4_aggregate_replicates_to_configs(pr)
    fg  = step5_load_and_merge_feature_gen_files(path_fg_2025, path_fg_2026)
    nn  = step6_load_and_merge_nn_files(path_nn_2025, path_nn_2026)
    m   = step7_join_master_table(pra, nn, fg)
    m   = step8_add_derived_totals(m)
    step9_sanity_checks(m)
    m.to_csv(path_output, index=False)
    print(f"\nSaved master table to {path_output}")
    return m


if __name__ == '__main__':
    main(
        path_precision_recall = 'data/source/all_precision_recall.csv',
        path_fg_2025          = 'data/source/adbis_2025_ncvoters_10000_records_1_error_processed_power_measurements_encoding_data.csv',
        path_fg_2026          = 'data/source/adbis_26_encoding_data_ncvoters.csv',
        path_nn_2025          = 'data/source/adbis_2025_ncvoters_10000_records_1_error_processed_power_measurements_NN.csv',
        path_nn_2026          = 'data/source/adbis_26_ncvoters_train_test_results.csv',
        path_output           = 'output/master_tidy.csv',
    )

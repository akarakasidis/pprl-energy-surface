"""
Variance decomposition (ANOVA) for the three response variables:
  - log(fg_energy_mean)
  - log(nn_energy_mean)
  - f1_mean

"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')


MASTER_CSV   = 'master_tidy.csv'
FG_DEDUP_KEY = ['N', 'capacity', 'error_rate', 'flip_probability']

FG_FACTORS = ['N', 'capacity', 'error_rate', 'flip_probability']
NN_FACTORS = ['N', 'capacity', 'error_rate', 'flip_probability',
              'num_hidden_layers', 'num_of_neurons_in_hidden_layer', 'epochs']


def load_master():

    df = pd.read_csv(MASTER_CSV)
    df['log_fg_energy'] = np.log(df['fg_energy_mean'])
    df['log_nn_energy'] = np.log(df['nn_energy_mean'])
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


def anova_decomposition(data, response, factors):

    rhs = ' + '.join(f'C({f})' for f in factors)
    formula = f'{response} ~ {rhs}'

    model = smf.ols(formula, data=data).fit()
    aov   = sm.stats.anova_lm(model, typ=2)

    total_ss = aov['sum_sq'].sum()
    aov['eta_sq']     = aov['sum_sq'] / total_ss
    aov['eta_sq_pct'] = aov['eta_sq'] * 100

    # Sort by effect size descending, keep Residual at the bottom for readability
    main_effects = aov.drop(index='Residual').sort_values('eta_sq', ascending=False)
    residual     = aov.loc[['Residual']]
    return pd.concat([main_effects, residual])


def print_decomposition(aov_table, response_label):

    print(f"\n{'='*72}")
    print(f"Variance decomposition: {response_label}")
    print(f"{'='*72}")
    print(f"{'Factor':45s}  {'sum_sq':>10s}  {'df':>5s}  {'eta^2':>8s}  {'%var':>6s}")
    print('-' * 84)
    for factor, row in aov_table.iterrows():
        clean_name = factor.replace('C(', '').replace(')', '')
        print(f"{clean_name:45s}  {row['sum_sq']:10.3f}  "
              f"{row['df']:>5.0f}  {row['eta_sq']:8.4f}  "
              f"{row['eta_sq_pct']:>5.1f}%")

    # Headline: top factor name and its share
    top = aov_table.drop(index='Residual').iloc[0]
    top_name = aov_table.drop(index='Residual').index[0]
    clean_top = top_name.replace('C(', '').replace(')', '')
    print(f"\nTop factor: {clean_top} ({top['eta_sq_pct']:.1f}% of variance)")


def main():
    df      = load_master()
    fg_data = prepare_fg_data(df)
    nn_data = prepare_nn_data(df)

    fg_aov   = anova_decomposition(fg_data, 'log_fg_energy', FG_FACTORS)
    nn_aov_e = anova_decomposition(nn_data, 'log_nn_energy', NN_FACTORS)
    nn_aov_f = anova_decomposition(nn_data, 'f1_mean',        NN_FACTORS)

    print_decomposition(fg_aov,   'log_fg_energy   (n=48 unique fg measurements)')
    print_decomposition(nn_aov_e, 'log_nn_energy   (n=1296 NN measurements)')
    print_decomposition(nn_aov_f, 'F1              (n=1296 NN measurements)')

    fg_aov.to_csv('verify_anova_fg_energy.csv')
    nn_aov_e.to_csv('verify_anova_nn_energy.csv')
    nn_aov_f.to_csv('verify_anova_f1.csv')
    print("\nSaved CSVs: verify_anova_fg_energy, verify_anova_nn_energy, verify_anova_f1")


if __name__ == '__main__':
    main()

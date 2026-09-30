"""
Fit the phase-separated energy cost models.

Two regressions are produced:
  - Feature-generation (fg) energy, fit on the 48 unique fg measurements
  - NN-phase energy, fit on the 1,296 per-configuration measurements
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')



MASTER_CSV = 'master_tidy.csv'


FG_DEDUP_KEY = ['N', 'capacity', 'error_rate', 'flip_probability']


FORMULA_FG = 'log_fg_energy ~ log_N + log_bf_bits + log_bf_hashes'

FORMULA_NN = (
    'log_nn_energy ~ log_N + log_epochs + log_num_hidden_layers '
    '+ C(neurons_cat, Treatment("4"))'
)


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
    expected = 4 * 2 * 2 * 3   # 4 dataset sizes x 2 cap x 2 err x 3 flip
    assert len(fg) == expected, f"Expected {expected} fg rows, got {len(fg)}"
    print(f"[prepare_fg] {len(fg)} unique fg measurements")
    return fg


def prepare_nn_data(df):

    print(f"[prepare_nn] {len(df)} NN measurements")
    return df



def fit_fg_model(fg_data):

    model = smf.ols(FORMULA_FG, data=fg_data).fit()
    print(f"[fit_fg] R^2 = {model.rsquared:.4f}, n = {int(model.nobs)}")
    return model


def fit_nn_model(nn_data):

    model = smf.ols(FORMULA_NN, data=nn_data).fit()
    print(f"[fit_nn] R^2 = {model.rsquared:.4f}, n = {int(model.nobs)}")
    return model



def summarize(model, label):

    print(f"\n{'='*72}")
    print(f"Model: {label}")
    print(f"  R^2      = {model.rsquared:.4f}")
    print(f"  Adj. R^2 = {model.rsquared_adj:.4f}")
    print(f"  n        = {int(model.nobs)}")
    print(f"  AIC      = {model.aic:.1f}")
    print(f"  BIC      = {model.bic:.1f}")
    print(f"{'='*72}")

    print(f"\n{'Term':40s}  {'Coef':>10s}  {'SE':>8s}  {'95% CI':>22s}  {'p':>8s}")
    print('-' * 96)
    ci = model.conf_int()
    for term in model.params.index:
        coef = model.params[term]
        se   = model.bse[term]
        lo, hi = ci.loc[term]
        p    = model.pvalues[term]
        ci_str = f"[{lo:+.4f}, {hi:+.4f}]"
        print(f"{term:40s}  {coef:+10.4f}  {se:8.4f}  {ci_str:>22s}  {p:8.4f}")




def compare_nn_specifications(nn_data):

    print(f"\n{'='*72}")
    print("NN-phase specification comparison (one-off, for justification only)")
    print(f"{'='*72}")

    specs = {
        'M1: all continuous': (
            'log_nn_energy ~ log_N + log_epochs + log_num_hidden_layers '
            '+ log_num_of_neurons_in_hidden_layer'
        ),
        'M2: neurons categorical': (
            'log_nn_energy ~ log_N + log_epochs + log_num_hidden_layers '
            '+ C(neurons_cat, Treatment("4"))'
        ),
        'M3: neurons + layers categorical': (
            'log_nn_energy ~ log_N + log_epochs '
            '+ C(layers_cat, Treatment("1")) + C(neurons_cat, Treatment("4"))'
        ),
    }

    fits = {}
    print(f"\n{'Specification':40s}  {'R^2':>8s}  {'AIC':>10s}  {'dAIC vs M1':>12s}")
    print('-' * 78)
    baseline_aic = None
    for name, formula in specs.items():
        model = smf.ols(formula, data=nn_data).fit()
        fits[name] = model
        if baseline_aic is None:
            baseline_aic = model.aic
            delta = 0.0
        else:
            delta = model.aic - baseline_aic
        print(f"{name:40s}  {model.rsquared:8.4f}  {model.aic:10.1f}  {delta:+12.1f}")

    # Show the residual-by-neurons pattern for M1 to make the bias visible
    print("\nMean residual by neuron count under M1 (continuous neurons):")
    print("(if M1 were adequate these should all be ~0)")
    m1 = fits['M1: all continuous']
    nn_data = nn_data.copy()
    nn_data['resid_m1'] = m1.resid
    by_neu = nn_data.groupby('num_of_neurons_in_hidden_layer')['resid_m1'].agg(
        ['mean', 'std', 'count']
    )
    print(by_neu.round(4))

    return fits


def main(run_comparison=True):
    df      = load_master_and_derive_predictors()
    fg_data = prepare_fg_data(df)
    nn_data = prepare_nn_data(df)

    fg_model = fit_fg_model(fg_data)
    nn_model = fit_nn_model(nn_data)

    summarize(fg_model, label='Feature-generation energy')
    summarize(nn_model, label='NN-phase energy')

    if run_comparison:
        compare_nn_specifications(nn_data)

    return fg_model, nn_model


if __name__ == '__main__':
    main()

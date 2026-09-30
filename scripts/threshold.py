"""
Threshold accuracy model.

Fits a logistic regression for P(F1 >= 0.9) as a function of configuration
parameters. 
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.metrics import brier_score_loss, accuracy_score
import warnings
warnings.filterwarnings('ignore')


MASTER_CSV   = 'master_tidy.csv'
THRESHOLD    = 0.9
FORMULA_THRESHOLD = (
    'meets_threshold ~ log_N + log_epochs + log_num_hidden_layers '
    '+ C(neurons_cat, Treatment("4")) + C(flip_cat, Treatment("0.0")) '
    '+ log_bf_bits + log_bf_hashes'
)


def load_master_and_derive_predictors():

    df = pd.read_csv(MASTER_CSV)

    df['bf_bits']   = np.ceil(-df['capacity'] * np.log(df['error_rate']) / np.log(2)**2)
    df['bf_hashes'] = np.ceil(-np.log(df['error_rate']) / np.log(2))

    for col in ['N', 'epochs', 'num_hidden_layers', 'bf_bits', 'bf_hashes']:
        df[f'log_{col}'] = np.log(df[col])

    df['neurons_cat'] = df['num_of_neurons_in_hidden_layer'].astype(str)
    df['flip_cat']    = df['flip_probability'].astype(str)

    print(f"[load] master loaded: {df.shape}")
    return df


def add_threshold_outcome(df, threshold=THRESHOLD):

    df = df.copy()
    df['meets_threshold'] = (df['f1_mean'] >= threshold).astype(int)
    rate = df['meets_threshold'].mean()
    print(f"[threshold] threshold={threshold}, "
          f"{df['meets_threshold'].sum()} / {len(df)} configs meet it "
          f"({100*rate:.1f}%)")
    return df

def fit_threshold_model(df):

    model = smf.logit(FORMULA_THRESHOLD, data=df).fit(disp=0)
    print(f"[fit_threshold] pseudo-R^2 (McFadden) = {model.prsquared:.4f}, "
          f"n = {int(model.nobs)}")
    return model


def summarize(model):

    print(f"\n{'='*88}")
    print("Threshold model: P(F1 >= 0.9) ~ config")
    print(f"  Pseudo-R^2 (McFadden) = {model.prsquared:.4f}")
    print(f"  n                     = {int(model.nobs)}")
    print(f"  Log-likelihood        = {model.llf:.1f}")
    print(f"{'='*88}")

    ci = model.conf_int()
    print(f"\n{'Term':45s}  {'Coef':>9s}  {'OR':>8s}  {'OR 95% CI':>22s}  {'p':>8s}")
    print('-' * 100)
    for term in model.params.index:
        coef = model.params[term]
        p    = model.pvalues[term]
        lo, hi = ci.loc[term]
        if term == 'Intercept':
            print(f"{term:45s}  {coef:+9.4f}  {'—':>8s}  {'—':>22s}  {p:8.4f}")
        else:
            or_val = np.exp(coef)
            or_lo, or_hi = np.exp(lo), np.exp(hi)
            or_str = f"[{or_lo:5.3f}, {or_hi:6.3f}]"
            print(f"{term:45s}  {coef:+9.4f}  {or_val:8.3f}  {or_str:>22s}  {p:8.4f}")


def report_fit_quality(model, df):

    pred_proba = model.predict(df)
    pred_class = (pred_proba >= 0.5).astype(int)

    brier  = brier_score_loss(df['meets_threshold'], pred_proba)
    acc    = accuracy_score(df['meets_threshold'], pred_class)
    base   = max(df['meets_threshold'].mean(), 1 - df['meets_threshold'].mean())

    print(f"\nGoodness-of-fit:")
    print(f"  Brier score:                 {brier:.4f}")
    print(f"  Classification accuracy:     {acc:.4f}  (cutoff = 0.5)")
    print(f"  Majority-class base rate:    {base:.4f}")
    print(f"  Improvement over base rate:  {(acc-base)*100:+.1f} pct points")
    return pred_proba


def print_threshold_by_N_and_flip(df):

    pivot = df.pivot_table(index='N', columns='flip_probability',
                          values='meets_threshold', aggfunc='mean')
    print("\nProportion of configs meeting F1 >= 0.9 threshold,")
    print("by dataset size N (rows) and flip probability (columns):")
    print(pivot.round(3).to_string())


def save_predictions(df, pred_proba, path='verify_threshold_predictions.csv'):

    out = df[['N', 'capacity', 'error_rate', 'flip_probability',
              'num_hidden_layers', 'num_of_neurons_in_hidden_layer', 'epochs',
              'f1_mean', 'meets_threshold']].copy()
    out['pred_proba'] = pred_proba
    out.to_csv(path, index=False)
    print(f"\n[save] per-config predictions saved to {path}")



def main():
    df    = load_master_and_derive_predictors()
    df    = add_threshold_outcome(df, threshold=THRESHOLD)
    model = fit_threshold_model(df)

    summarize(model)
    pred_proba = report_fit_quality(model, df)
    print_threshold_by_N_and_flip(df)
    save_predictions(df, pred_proba)

    return model


if __name__ == '__main__':
    main()

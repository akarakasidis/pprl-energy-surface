"""
Pareto frontier per dataset size, plus quality-threshold analysis.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.3})


MASTER_CSV = 'master_tidy.csv'
OUT_DIR    = '.'
DATASET_SIZES = [2000, 5000, 10000, 20000]


def load_master():
    df = pd.read_csv(MASTER_CSV)
    print(f"[load] master loaded: {df.shape}")
    return df

def pareto_frontier_at_N(df_at_N):
    sorted_df = df_at_N.sort_values('total_energy_mean')
    keep_indices = []
    best_f1_so_far = -np.inf
    for idx, row in sorted_df.iterrows():
        if row['f1_mean'] > best_f1_so_far:
            keep_indices.append(idx)
            best_f1_so_far = row['f1_mean']
    return df_at_N.loc[keep_indices]


def compute_pareto_frontiers(df):

    fronts = {}
    print(f"\n{'='*80}")
    print("Pareto-optimal configurations per dataset size")
    print(f"{'='*80}")
    for N in DATASET_SIZES:
        g  = df[df['N'] == N]
        pf = pareto_frontier_at_N(g)
        fronts[N] = pf
        print(f"  N = {N:>5}: {len(pf):>3} of {len(g)} configurations "
              f"are Pareto-optimal")
    return fronts


def plateau_at_threshold(df_at_N, threshold):
    q = df_at_N[df_at_N['f1_mean'] >= threshold]
    if len(q) == 0:
        return {'count': 0}

    cheapest = q.loc[q['total_energy_mean'].idxmin()]
    priciest = q.loc[q['total_energy_mean'].idxmax()]
    ratio    = priciest['total_energy_mean'] / cheapest['total_energy_mean']

    return {
        'count':    len(q),
        'cheapest': cheapest,
        'priciest': priciest,
        'ratio':    ratio,
    }


def summarize_plateau(df, thresholds=(0.9, 0.95)):
    print(f"\n{'='*80}")
    print("High-quality plateau: configurations clearing each F1 threshold")
    print(f"{'='*80}")

    rows = []
    for threshold in thresholds:
        print(f"\nF1 threshold = {threshold}")
        print('-' * 80)
        for N in DATASET_SIZES:
            g = df[df['N'] == N]
            s = plateau_at_threshold(g, threshold)
            if s['count'] == 0:
                print(f"  N = {N:>5}: no configs meet F1 >= {threshold}")
                continue
            ch = s['cheapest']
            pr = s['priciest']
            print(f"  N = {N:>5}: {s['count']:>3} configs qualify  "
                  f"|  energy {ch['total_energy_mean']/1000:>6.2f}--"
                  f"{pr['total_energy_mean']/1000:.2f} kJ ({s['ratio']:.2f}x)")
            print(f"         cheapest: F1={ch['f1_mean']:.3f}, "
                  f"cap={int(ch['capacity'])}/err={ch['error_rate']}, "
                  f"L={int(ch['num_hidden_layers'])}, "
                  f"neu={int(ch['num_of_neurons_in_hidden_layer'])}, "
                  f"ep={int(ch['epochs'])}, "
                  f"flip={ch['flip_probability']}")
            rows.append({
                'N':         N,
                'threshold': threshold,
                'n_configs': s['count'],
                'min_kJ':    ch['total_energy_mean'] / 1000,
                'max_kJ':    pr['total_energy_mean'] / 1000,
                'ratio':     s['ratio'],
            })

    return pd.DataFrame(rows)


def make_pareto_figure(df, fronts):

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    for ax, N in zip(axes, DATASET_SIZES):
        g  = df[df['N'] == N]
        pf = fronts[N]
        ax.scatter(g['total_energy_mean'] / 1000, g['f1_mean'],
                   alpha=0.3, s=12, color='steelblue', label='all configs')
        ax.scatter(pf['total_energy_mean'] / 1000, pf['f1_mean'],
                   s=24, color='crimson', label='Pareto-optimal', zorder=5)
        ax.axhline(0.9, color='black', linestyle=':', alpha=0.5, linewidth=0.8)
        ax.set_xscale('log')
        ax.set_xlabel('total energy (kJ)')
        if ax is axes[0]:
            ax.set_ylabel('F1')
        ax.set_title(f'N = {N}')
        ax.legend(loc='lower right', fontsize=7)
    plt.tight_layout()
    return fig


def save_figure(fig, name):

    for ext in ('pdf', 'png'):
        path = f'{OUT_DIR}/{name}.{ext}'
        fig.savefig(path, dpi=120, bbox_inches='tight')
    print(f"\n[save] figure saved as {name}.{{pdf,png}}")


def save_pareto_sets(fronts, path='verify_pareto_frontiers.csv'):

    frames = [pf.assign(N_for_frontier=N) for N, pf in fronts.items()]
    out    = pd.concat(frames, ignore_index=True)
    out.to_csv(path, index=False)
    print(f"\n[save] Pareto frontiers saved to {path} ({len(out)} configs total)")


def main():
    df      = load_master()
    fronts  = compute_pareto_frontiers(df)
    plateau = summarize_plateau(df, thresholds=(0.9, 0.95))
    fig     = make_pareto_figure(df, fronts)
    save_figure(fig, 'verify_pareto')
    plt.close(fig)
    save_pareto_sets(fronts)
    return fronts, plateau


if __name__ == '__main__':
    main()

"""Figure 2 in paper (file fig4_neurons): NN energy ratio vs 8-neuron baseline.
"""
import pandas as pd, numpy as np, matplotlib.pyplot as plt, matplotlib as mpl
mpl.rcParams.update({'font.size': 11, 'axes.grid': True, 'grid.alpha': 0.3})

df = pd.read_csv('master_tidy.csv')


KEY = ['N','num_hidden_layers','epochs','capacity','error_rate','flip_probability']
piv = df.pivot_table(index=KEY, columns='num_of_neurons_in_hidden_layer',
                     values='nn_energy_mean').dropna()
widths = [4,8,16]
fig, ax = plt.subplots(figsize=(5,3.6))
colors = [
    "#1F77B4",   # blue
    "#D62728",   # red
    "#2CA02C",   # green
    "#9467BD",   # purple
]
for c,N in zip(colors,[2000,5000,10000,20000]):
    sub = piv.reset_index(); sub = sub[sub.N==N]
    ratios = {w:(sub[w]/sub[8]) for w in widths}
    means = [ratios[w].mean() for w in widths]
    stds  = [ratios[w].std()  for w in widths]
    ax.errorbar(widths, means, yerr=stds, fmt='o-', capsize=3, color=c, label=f'$N$ = {N:,}')
ax.axhline(1.0, color='gray', ls=':', lw=1.2)
ax.set_xticks(widths); ax.set_xlabel('Neurons per hidden layer')
ax.set_ylabel('NN energy relative to 8 neurons')
ax.legend(fontsize=10)
plt.tight_layout()
for ext in ('pdf','png'): fig.savefig(f'fig4_neurons.{ext}', bbox_inches='tight', dpi=150)
print('saved fig4_neurons')
for N in [2000,5000,10000,20000]:
    sub=piv.reset_index(); sub=sub[sub.N==N]
    print(f'  N={N}: 4/8={ (sub[4]/sub[8]).mean():.3f}  16/8={(sub[16]/sub[8]).mean():.3f}')

"""Figure 3 in paper (file fig2_pareto): F1 vs total energy, 4 panels, Pareto points red."""
import pandas as pd, numpy as np, matplotlib.pyplot as plt, matplotlib as mpl
mpl.rcParams.update({'font.size': 11, 'axes.grid': True, 'grid.alpha': 0.3})

df = pd.read_csv('master_tidy.csv')
Ns=[2000,5000,10000,20000]
def pareto(g):
    s=g.sort_values('total_energy_mean'); keep=[]; best=-np.inf
    for i,r in s.iterrows():
        if r.f1_mean>best: keep.append(i); best=r.f1_mean
    return g.loc[keep]
fig,axes=plt.subplots(2,2,figsize=(8.4,7.2))
for ax,N in zip(axes.ravel(),Ns):
    g=df[df.N==N]; pf=pareto(g)
    ax.scatter(g['total_energy_mean']/1000,g['f1_mean'],alpha=0.3,s=14,color='steelblue',label='All configurations')
    ax.scatter(pf['total_energy_mean']/1000,pf['f1_mean'],s=36,color='crimson',zorder=5,label='Pareto-optimal',edgecolors='k',linewidths=0.5)
    ax.axhline(0.9,color='k',ls=':',lw=1.2)
    ax.set_xscale('log'); ax.set_title(f'$N$ = {N:,}')
    ax.set_xlabel('Total energy (kJ)'); ax.set_ylabel('$F_1$'); ax.set_ylim(0,1.05)
handles, labels = axes.ravel()[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc='lower center',
    ncol=2,
    frameon=False,
    fontsize=10,
    bbox_to_anchor=(0.5, 0.04)
)

plt.tight_layout(rect=[0,0.08,1,1], pad=1.2, w_pad=1.0, h_pad=1.4)

for ext in ('pdf','png'):
    fig.savefig(f'fig2_pareto.{ext}',bbox_inches='tight',dpi=150)
print('saved fig2_pareto')
for N in Ns:
    g=df[df.N==N]; print(f'  N={N}: {len(pareto(g))} Pareto-optimal of {len(g)}')

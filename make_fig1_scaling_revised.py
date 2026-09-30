"""Figure 1: energy vs dataset size N, log-log, both phases.
Revised plotting script.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

mpl.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

df = pd.read_csv("master_tidy.csv")

FG_KEY = ["N","capacity","error_rate","flip_probability"]
fg = df.drop_duplicates(subset=FG_KEY).copy()

Ns = sorted(df["N"].unique())
x = np.array([min(Ns), max(Ns)])

fig, axes = plt.subplots(1,2,figsize=(8,3.4))

ax = axes[0]

for _, g in fg.groupby(["capacity","error_rate","flip_probability"]):
    g = g.sort_values("N")
    ax.loglog(
        g["N"], g["fg_energy_mean"],
        "o-",
        color="steelblue",
        alpha=0.5,
        ms=3,
        zorder=1,
    )

ref0 = 2.0e3

ax.loglog(
    x,
    ref0 * (x/min(Ns))**2,
    "k--",
    lw=1.8,
    label="slope = 2",
    zorder=1000,
)

ax.set_xlabel("Dataset size $N$")
ax.set_ylabel("Feature-gen energy (J)")
ax.set_title("(a) Feature generation")

ax.set_xscale("log")
ax.xaxis.set_major_locator(FixedLocator(Ns))
ax.xaxis.set_major_formatter(
    FuncFormatter(lambda v, p: f"{int(v)}" if int(round(v)) in Ns else "")
)
ax.xaxis.set_minor_locator(NullLocator())
ax.set_xlim(1800,22000)

ax.legend(loc="lower right", fontsize=10, frameon=False)


ax = axes[1]

for _, g in df.groupby([
    "num_hidden_layers",
    "num_of_neurons_in_hidden_layer",
    "epochs",
]):
    g = g.groupby("N")["nn_energy_mean"].mean().reset_index()

    ax.loglog(
        g["N"],
        g["nn_energy_mean"],
        "o-",
        color="steelblue",
        alpha=0.25,
        lw=0.8,
        ms=3,
        zorder=1,
    )

ref0 = 2.0e3

ax.loglog(
    x,
    ref0 * (x/min(Ns))**2,
    "k--",
    lw=1.8,
    label="slope = 2",
    zorder=1000,
)

ax.set_xlabel("Dataset size $N$")
ax.set_ylabel("NN-phase energy (J)")
ax.set_title("(b) NN training + evaluation")

ax.set_xscale("log")
ax.xaxis.set_major_locator(FixedLocator(Ns))
ax.xaxis.set_major_formatter(
    FuncFormatter(lambda v, p: f"{int(v)}" if int(round(v)) in Ns else "")
)
ax.xaxis.set_minor_locator(NullLocator())
ax.set_xlim(1800,22000)

ax.legend(loc="lower right", fontsize=10, frameon=False)

plt.tight_layout()

for ext in ("pdf","png"):
    fig.savefig(f"fig1_scaling.{ext}", bbox_inches="tight", dpi=150)

print("saved fig1_scaling")

for lab,d,col in [("fg",fg,"fg_energy_mean"),("nn",df,"nn_energy_mean")]:
    gg = d.groupby("N")[col].mean()
    slope = np.polyfit(np.log(gg.index), np.log(gg.values),1)[0]
    print(f"{lab}: slope = {slope:.3f}")

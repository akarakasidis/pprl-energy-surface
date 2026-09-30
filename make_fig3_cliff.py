"""Figure 4 in paper (fig3_cliff):
Mean F1 vs flip probability.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


df = pd.read_csv("master_tidy.csv")

Ns = [2000, 5000, 10000, 20000]
flips = sorted(df["flip_probability"].unique())

fig, ax = plt.subplots(figsize=(5.6, 3.8))


x = np.arange(len(flips))


palette = [
    "#1F77B4",   # blue
    "#D62728",   # red
    "#2CA02C",   # green
    "#9467BD",   # purple
]

markers = [
    "o",
    "s",
    "^",
    "D",
]


for colour, marker, N in zip(palette, markers, Ns):

    g = (
        df[df.N == N]
        .groupby("flip_probability")["f1_mean"]
        .agg(["mean", "std"])
        .reindex(flips)
        .reset_index()
    )

    ax.errorbar(
        x,
        g["mean"],
        yerr=g["std"],
        fmt=marker + "-",
        color=colour,
        lw=1.8,
        ms=5,
        elinewidth=1.3,
        capsize=3,
        label=f"$N$ = {N:,}",
    )


ax.set_xticks(x)
ax.set_xticklabels([str(fp) for fp in flips])

ax.set_xlabel(r"Flip probability $p_{\mathrm{flip}}$")
ax.set_ylabel(r"$F_1$ (mean across configurations)")

ax.set_ylim(0.30, 1.05)


ax.legend(
    loc="lower left",
    ncol=2,
    fontsize=10,
    frameon=False,
)

plt.tight_layout()


for ext in ("pdf", "png"):
    fig.savefig(
        f"fig3_cliff.{ext}",
        bbox_inches="tight",
        dpi=150,
    )

print("saved fig3_cliff")


for fp in flips:

    means = [
        df[
            (df.N == N)
            & (df.flip_probability == fp)
        ]["f1_mean"].mean()
        for N in Ns
    ]

    print(
        f"flip={fp}: "
        f"F1 range {min(means):.3f}-{max(means):.3f}"
    )

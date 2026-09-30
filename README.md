# Energy Plateaus and Privacy Cliffs in Deep Learning Privacy Preserving Record Linkage

Reproduction materials for the paper by Alexandros Karakasidis and Chairi Kiourt. The package reproduces the energy models, variance decomposition, quality-threshold analysis, Pareto analysis and figures from recorded experimental measurements.

The experiments cover 1,296 configurations across four dataset sizes (2,000, 5,000, 10,000 and 20,000 records), with three replicates per configuration. Feature-generation measurements are shared across NN settings, giving 48 unique feature-generation configurations.

## Scope

The workflow starts from recorded matching outcomes and processed energy summaries. It does not rerun neural-network training or collect new energy measurements. Original experiment code, datasets and measurement logs can be distributed separately as supplementary materials.

The energy models are evaluated using leave-one-dataset-size-out validation. The quality-threshold model's reported classification accuracy is calculated on its fitting data; it is not a held-out accuracy estimate.

## Repository layout

Arrange the selected release files as follows. Paths in the commands below are relative to this layout.

| Directory | Contents |
|---|---|
| `data/source/` | Raw source CSV files |
| `data/reference/` | One reference copy of `master_tidy.csv` |
| `scripts/` | Consolidation, analysis and final plotting scripts |
 |


## Paper

Alexandros Karakasidis and Chairi Kiourt. *Energy Plateaus and Privacy Cliffs in Deep-Learning Privacy-Preserving Record Linkage*. ADBIS 2026.

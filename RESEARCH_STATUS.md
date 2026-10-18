# Research status and reproduction audit

## Current conference-paper campaign

The current paper evaluation focuses strictly on the **ISOMORPH** and **SupplyGraph** supply chain benchmark datasets. Existing Drive checkpoints were evaluated with forward passes only. No training epochs, hyperparameter searches, or Job C scripts were run for this extraction.

Verified published artifacts are in `paper_outputs/`:

- unified validation/test metrics and mean/sample-standard-deviation summaries (`paper_main_metrics.csv`, `paper_summary_stats.csv`);
- Job B real-vs-permuted paired tests across 3 seeds (`paper_job_b_paired.csv`);
- multi-horizon, node-role/sparsity, learned-gate, and compute-cost tables;
- window-level significance and rigorous time-aware block bootstrap statistics (`paper_window_significance.csv`, `paper_rigorous_stats.csv`);
- publication-ready 300-DPI PNG and vector PDF paper figures.

The six Job B permuted STGT runs are complete on Drive across seeds `42`, `123`, and `2026`. The separate `job_b_pipeline_real_supplygraph_v2` seed-42 reproduction gate remains interrupted with no final result row, so it is explicitly recorded as **not passed** (`paper_job_b_gate_status.json`) rather than silently inferred from another run.

Drive tokens, OAuth client files, checkpoints, raw data, and downloaded private artifacts are not repository contents. The repository includes only portable code, reviewed aggregate outputs, documentation, and tests.

## Evidence available

- Benchmark datasets: ISOMORPH (13 supply chain nodes) and SupplyGraph (39 supply chain nodes).
- Architectures: STGT reference baseline, Adaptive STGT (proposed dynamic gating), GAT-LSTM comparison, and ablation variants (Temporal-only, Spatial-only, Fixed Fusion).
- Multi-seed evaluations across three fixed random seeds: `42`, `123`, and `2026`.
- Metric suite: MAE, RMSE, WAPE, sMAPE, $R^2$, and Masked MAPE, evaluated across rolling multi-horizon forecasting windows ($h = 1, \dots, 7$).
- Static/synthetic test suite: unit and integration verification for graph structure, feature scalers, metric calculations, and checkpoint loaders.

## Limitations and reproduction notes

- Multi-step horizon projections and graph-topology adaptive fusion are adaptations evaluated under the specified 70% train / 15% validation / 15% test splits.
- Predictions are inverse-scaled and non-negative clipped ($y \ge 0$) following standard physical demand constraints.
- Parameter counts, inference latencies, and epoch selections are reported transparently.
- GPU determinism and cloud storage locking across separate distributed sessions depend on runtime environments.

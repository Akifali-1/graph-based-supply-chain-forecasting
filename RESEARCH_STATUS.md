# Research status and reproduction audit

## Current conference-paper campaign

The paper evaluation benchmark covers **ISOMORPH** (13 nodes, $T=52{,}560$) and **SupplyGraph** (39 nodes, $T=221$). The full 5-seed benchmark across seeds `{42, 123, 2026, 7, 99}` is completely trained, verified, and audited.

Verified published artifacts are in `paper_outputs/` and `results/expansion_5seeds/`:

- Unified validation/test metrics and 5-seed mean/sample-standard-deviation summaries (`paper_main_metrics.csv`, `paper_summary_stats.csv`);
- Real-vs-permuted graph paired hypothesis tests across all 5 seeds (`paper_job_b_paired.csv`, $p = 0.0066$ on ISOMORPH);
- Multi-horizon, node-role/sparsity, learned-gate, and compute-cost tables;
- Window-level significance and rigorous time-aware block bootstrap statistics (`paper_window_significance.csv`, `paper_rigorous_stats.csv`);
- Publication-ready 300-DPI PNG and vector PDF paper figures.

Drive tokens, OAuth credentials, raw multi-gigabyte data files, and raw model weights remain safely in private cloud storage and are never committed to Git. The repository includes strictly portable code, reproducible configurations, reviewed aggregate outputs, documentation, tests, and execution notebooks.

## Evidence available

- Benchmark datasets: ISOMORPH (13 supply chain nodes, $T=52{,}560$) and SupplyGraph (39 supply chain nodes, $T=221$).
- Architectures: STGT reference baseline, Adaptive STGT (proposed dynamic gating), and ablation variants (Temporal-only, Spatial-only, Fixed Fusion, Permuted-Graph STGT).
- Multi-seed evaluations across five fixed random seeds: `42`, `123`, `2026`, `7`, and `99`.
- Metric suite: MAE, RMSE, WAPE, sMAPE, $R^2$, and Masked MAPE, evaluated across rolling multi-horizon forecasting windows ($h = 1, \dots, 7$).
- Static/synthetic test suite: unit and integration verification for graph structure, feature scalers, metric calculations, and checkpoint loaders.

## Limitations and reproduction notes

- Multi-step horizon projections and graph-topology adaptive fusion are adaptations evaluated under the specified 70% train / 15% validation / 15% test splits.
- Predictions are inverse-scaled and non-negative clipped ($y \ge 0$) following standard physical demand constraints.
- Parameter counts, inference latencies, and epoch selections are reported transparently.
- GPU determinism and cloud storage locking across separate distributed sessions depend on runtime environments.

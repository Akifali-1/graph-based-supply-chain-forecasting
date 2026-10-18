# Adaptive STGT vs Adaptive Fusion GAT-LSTM for Interpretable Supply Chain Demand Forecasting

Semi-automated, reproducible benchmark comparison for graph-based multi-echelon supply chain demand forecasting.
Code is versioned in GitHub; model checkpoints and large data artifacts live in private Drive storage; large processing and model campaigns run in an authorized execution runtime.

**Status:** The experimental campaigns and reproducible paper evaluation are complete for the benchmark datasets: **ISOMORPH** and **SupplyGraph**. Final evaluation tables and publication figures are stored in `paper_outputs/`; model checkpoints and raw inputs remain in private Drive storage and are never committed. The SupplyGraph real-graph Job B reproduction gate is recorded as incomplete rather than claimed as passed. See `RESEARCH_STATUS.md` for the artifact audit and limitations.

## Repository Structure

```text
graph-based-supply-chain-forecasting/
├── configs/
│   ├── experiment.yaml              # shared seed, protocol, and model parameters
│   ├── graph_ablation.yaml          # real vs permuted graph ablation config
│   ├── isomorph_campaign_12.yaml    # ISOMORPH multi-seed campaign configuration
│   └── search_space.yaml            # search constraints and budget
├── data/
│   ├── raw/                         # source data directory (ignored by Git)
│   └── processed/                   # processed cache directory (ignored by Git)
├── notebooks/
│   └── colab_experiment.ipynb      # interactive runtime harness
├── paper_outputs/                   # reviewed aggregate paper tables and figures
│   ├── fig1_architecture.{png,pdf}
│   ├── fig2_horizon_and_roles.{png,pdf}
│   ├── fig3_gate_topology.{png,pdf}
│   ├── paper_compute_cost.csv
│   ├── paper_gate_summary.csv
│   ├── paper_gate_weights.csv
│   ├── paper_horizon_metrics.csv
│   ├── paper_job_b_gate_status.json
│   ├── paper_job_b_paired.csv
│   ├── paper_main_metrics.csv
│   ├── paper_node_regime_metrics.csv
│   ├── paper_rigorous_stats.csv
│   ├── paper_summary_stats.csv
│   └── paper_window_significance.csv
├── src/
│   ├── checkpointing.py             # state saving and atomic checkpoints
│   ├── dataset.py                   # rolling-window multi-horizon PyTorch dataset
│   ├── day2_ablation.py             # temporal-only, spatial-only, and fixed-fusion models
│   ├── evaluate.py                  # multi-metric forward-pass evaluation
│   ├── evaluation_recovery.py       # safe checkpoint resume logic
│   ├── experiment_registry.py       # persistent experiment registry
│   ├── gat_lstm.py                  # GAT-LSTM baseline architecture
│   ├── metrics.py                   # MAE, RMSE, WAPE, sMAPE, R2, and Masked MAPE
│   ├── preprocessing.py             # per-node normalization and graph construction
│   ├── stgt.py                      # STGT and proposed Adaptive STGT architectures
│   ├── train.py                     # training loop with early stopping
│   └── utils.py                     # graph indexing and batch helpers
├── tests/                           # comprehensive test suite (unit and synthetic)
├── Dockerfile                       # isolated reproducible execution container
├── Makefile                         # standard test and evaluation targets
├── requirements.txt                 # package dependencies
├── requirements.lock                # pinned reproducible dependency lockfile
├── download_drive_artifacts.py      # token-path parameterized Drive downloader
├── paper_evaluation.py              # zero-training forward pass paper evaluation
├── run_statistical_pipeline.py      # block bootstrap and window significance pipeline
├── generate_figures.py              # 300-DPI PNG and vector PDF publication figures
├── verify_deliverables.py           # deliverable schema and gate assertion verifier
├── RESEARCH_STATUS.md               # research status and reproduction audit
├── SECURITY.md                      # credential and privacy protection policy
└── README.md
```

## Benchmark Datasets

The study evaluates multi-echelon forecasting across two benchmark supply chain graphs:

1. **ISOMORPH Benchmark**:
   - 13 interconnected supply chain nodes.
   - Distinct echelons including Factories, Central Warehouses, Regional Distribution Centers, and Retail Outlets.
   - Multi-step demand forecasting ($H = 7$) with a 14-day observation lookback window ($L = 14$).

2. **SupplyGraph Benchmark**:
   - 39 interconnected supply chain entities.
   - Complex topological relationships across multiple product categories and tier suppliers.
   - Evaluated under identical multi-horizon ($H = 7$) and lookback ($L = 14$) configurations.

Chronological split protocol: 70% Train, 15% Validation, and 15% Test. All metrics are computed in original inverse-scaled physical units with demand clipped at 0.0 ($y \ge 0$).

## Model Architectures

- **Reference STGT**: Temporal Transformer encoder, learnable positional encoding, node-type embeddings, dual `TransformerConv` spatial layers, and direct multi-horizon forecasting head.
- **Adaptive STGT (Proposed)**: Features a dynamic per-node gating mechanism ($\alpha_i$) that adaptively fuses temporal sequence features ($h_i^{\text{temp}}$) and spatial relational graph features ($h_i^{\text{spat}}$) based on local topology and degree regimes:
  $$\hat{h}_i = \alpha_i h_i^{\text{temp}} + (1 - \alpha_i) h_i^{\text{spat}}$$
- **GAT-LSTM**: Bidirectional LSTM temporal encoder combined with multi-head Graph Attention Networks (GAT).
- **Ablation Controls**:
  - *Temporal-Only STGT*: Temporal attention without spatial message passing.
  - *Spatial-Only STGT*: Graph message passing driven by instantaneous inputs without temporal attention.
  - *Fixed Fusion STGT*: Uniform static equal-weight fusion ($\alpha_i = 0.5$).
  - *Job B Graph Ablation (Permuted)*: STGT trained on edge-permuted topologies to measure structural informativeness.

## Reproducing Paper Deliverables (Zero Training)

All paper results and figures can be verified directly without training any model:

```bash
# 1. Download verified artifacts (using external Drive credentials)
python download_drive_artifacts.py \
  --token-file "$DRIVE_TOKEN_FILE" \
  --drive-root-folder-id "$DRIVE_ROOT_FOLDER_ID" \
  --output drive_artifacts

# 2. Extract unified metrics and gate tables
python paper_evaluation.py --artifacts drive_artifacts --output paper_outputs

# 3. Compute time-aware block bootstrap and window significance
python run_statistical_pipeline.py --artifacts drive_artifacts --output paper_outputs

# 4. Generate 300-DPI publication figures (PNG and vector PDF)
python generate_figures.py --output paper_outputs

# 5. Programmatically verify deliverables and gate conditions
python verify_deliverables.py --output paper_outputs
```

## Local Testing and Environment

Using the pinned dependencies in `requirements.lock`:

```bash
# Run the complete test suite
make test

# Or via pytest directly
python -m pytest -q
```

Or using Docker:

```bash
docker build -t supply-chain-research .
docker run --rm supply-chain-research
```

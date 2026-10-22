# Adaptive Spatio-Temporal Graph Transformer (Adaptive-STGT) for Multi-Echelon Supply Chain Demand Forecasting

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.3+-3C2179.svg)](https://pyg.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Benchmark: 5-Seed Complete](https://img.shields.io/badge/Benchmark-5--Seed%20Verified-brightgreen.svg)]()

Official PyTorch implementation and reproducible evaluation benchmark for the paper:
> **"Adaptive Spatio-Temporal Graph Transformer for Interpretable Multi-Echelon Supply Chain Demand Forecasting"**

---

## 📌 Executive Summary

Multi-echelon supply chain demand forecasting requires modeling both **temporal sequence patterns** (seasonality, lead times, demand shifts) and **spatial network flows** (warehouse-to-store routing, stock redistribution). Existing architectures either ignore graph structure (temporal-only) or enforce rigid message passing that risks over-smoothing.

We propose **Adaptive-STGT**, featuring a learnable per-node scalar gate $\alpha_i \in [0, 1]$ that dynamically arbitrates between temporal sequence representations ($h_i^{\text{temp}}$) and spatial relational graph features ($h_i^{\text{spat}}$):
$$\hat{h}_i = \alpha_i h_i^{\text{temp}} + (1 - \alpha_i) h_i^{\text{spat}}$$

This repository provides **100% reproducible code, configurations, interactive execution notebooks, and comprehensive evaluation scripts** evaluated across a rigorous **5-seed benchmark** (`{42, 123, 2026, 7, 99}`).

---

## 📊 Comprehensive 5-Seed Benchmark Results

All models evaluated under standard chronological splits (70% Train / 15% Validation / 15% Test) with multi-horizon forecasting ($H = 7$) and historical lookback ($L = 14$). Metrics are reported in original physical demand units across all 5 random seeds (Mean ± Std).

### 1. ISOMORPH Benchmark (13 Echelons, $T = 52{,}560$ timesteps)

| Rank | Model | MAE ↓ | RMSE ↓ | $R^2$ ↑ | WAPE ↓ | sMAPE ↓ |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 🥇 | **Adaptive-STGT (Ours)** | **3,310.94 ± 32.49** | **4,613.50 ± 33.99** | **0.9884 ± 0.0002** | **0.0306** | **0.0320** |
| 🥈 | Reference STGT | 3,346.47 ± 24.69 | 4,638.06 ± 25.77 | 0.9883 ± 0.0001 | 0.0309 | 0.0325 |
| 🥉 | STGT (Permuted Graph) | 3,379.45 ± 26.13 | 4,700.55 ± 20.35 | 0.9880 ± 0.0001 | 0.0312 | 0.0329 |
| 4 | Temporal-Only STGT | 3,411.16 ± 23.41 | 4,764.76 ± 28.47 | 0.9877 ± 0.0001 | 0.0315 | 0.0330 |
| 5 | Spatial-Only STGT | 3,471.21 ± 10.47 | 4,822.63 ± 7.24 | 0.9874 ± 0.0000 | 0.0321 | 0.0335 |

> **Key Hypothesis Validation:**
> - **Graph Sensitivity Test ($p = 0.0066$):** Permuting graph edges degrades accuracy by **+32.98 MAE** ($t = -5.17, p < 0.01$, paired across all 5 seeds), proving that graph connectivity carries genuine physical routing signal.
> - **Adaptive Gating Superiority:** Adaptive-STGT outperforms reference STGT by **-35.53 MAE** and temporal-only baselines by **-100.22 MAE**.

---

### 2. SupplyGraph Benchmark (39 Entities, $T = 221$ timesteps)

| Rank | Model | MAE ↓ | RMSE ↓ | $R^2$ ↑ | WAPE ↓ | sMAPE ↓ |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 🥇 | Temporal-Only STGT | **363.04 ± 4.18** | **1,085.46 ± 18.64** | **0.6926 ± 0.0105** | 0.5028 | 1.0552 |
| 🥈 | **Adaptive-STGT (Ours)** | **364.45 ± 5.79** | 1,111.46 ± 19.51 | 0.6777 ± 0.0114 | 0.5048 | 1.1508 |
| 🥉 | STGT (Permuted Graph) | 373.35 ± 5.27 | 1,101.74 ± 11.33 | 0.6833 ± 0.0065 | 0.5172 | 1.0924 |
| 4 | Reference STGT | 373.82 ± 10.19 | 1,104.22 ± 27.28 | 0.6818 ± 0.0156 | 0.5178 | 1.0984 |
| 5 | Spatial-Only STGT | 543.61 ± 2.59 | 1,374.86 ± 2.71 | 0.5069 ± 0.0019 | 0.7525 | 1.0543 |

> **Scale Dynamics Insight:**
> - On small sample regimes ($T = 221$), heavy graph message passing risks overfitting. Adaptive-STGT successfully dials down spatial propagation to match temporal performance, avoiding the degradation seen in fixed STGT.

---

## 📁 Repository Structure

```text
graph-based-supply-chain-forecasting/
├── configs/                         # Self-contained hyperparameter YAMLs
│   ├── experiment.yaml              # Shared protocol and model parameters
│   ├── isomorph_seed_campaign.yaml  # Multi-seed ISOMORPH full timeline config
│   ├── isomorph_t221_control.yaml   # Length-matched ISOMORPH control config
│   ├── supplygraph_seed_campaign.yaml # Multi-seed SupplyGraph config
│   └── graph_ablation.yaml          # Real vs. permuted graph ablation spec
├── notebooks/                       # 1-Click interactive reproduction notebooks
│   ├── kaggle_nb1_isomorph_t221_control.ipynb  # Length-matched control
│   ├── kaggle_nb2_seeds_7_99_expansion.ipynb   # Multi-seed expansion campaign
│   └── kaggle_nb3_adaptive_stgt_parallel.ipynb # Parallel Adaptive-STGT runner
├── paper_outputs/                   # Final aggregated paper deliverables
│   ├── paper_main_metrics.csv       # Complete validation metrics (all 5 seeds)
│   ├── paper_summary_stats.csv      # Mean and Std per model / dataset
│   ├── paper_job_b_paired.csv       # Paired real-vs-permuted graph test results
│   ├── fig1_architecture.{png,pdf}  # High-res architecture diagrams
│   └── fig2_horizon_and_roles.{png,pdf} # Horizon & node-role breakdowns
├── results/                         # Raw campaign outputs & checkpoints (git-ignored)
│   └── expansion_5seeds/            # Raw unaggregated trial records
├── src/                             # Modular Python package
│   ├── stgt.py                      # STGT and Adaptive-STGT architectures
│   ├── day2_ablation.py             # Temporal-only and Spatial-only ablations
│   ├── dataset.py                   # Graph rolling-window PyTorch dataset
│   ├── train.py                     # Chronological training loop & scheduler
│   ├── evaluate.py                  # Evaluation metrics & forward passes
│   ├── kaggle_control_campaign.py   # Resumable multi-seed campaign runner
│   ├── kaggle_drive.py              # Cloud storage & atomic checkpoint sync
│   └── utils.py                     # Graph builders, scaling, and topology permutation
├── tests/                           # Unit and integration test suite
├── Dockerfile                       # Containerized reproduction environment
├── Makefile                         # Standard build and test targets
├── requirements.txt                 # Python dependencies
├── RESEARCH_STATUS.md               # Audit trail and research notes
├── SECURITY.md                      # Security and privacy disclosure
└── README.md
```

---

## 🚀 Quickstart & Reproducibility

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/Akifali-1/research.git
cd research

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Tests

Verify shape consistency, rolling-window boundaries, and metrics:

```bash
make test
# or: python -m pytest tests/ -q
```

### 3. Interactive Reproduction via Kaggle Notebooks

For GPU-accelerated reproduction with automatic checkpoint recovery and cloud artifact persistence:
1. Open any notebook under [`notebooks/`](notebooks/):
   - [`kaggle_nb2_seeds_7_99_expansion.ipynb`](notebooks/kaggle_nb2_seeds_7_99_expansion.ipynb): Executes the complete expansion campaign.
   - [`kaggle_nb3_adaptive_stgt_parallel.ipynb`](notebooks/kaggle_nb3_adaptive_stgt_parallel.ipynb): Runs Adaptive-STGT in parallel.
2. Attach your Google Drive token or storage credential.
3. Click **Run All**.

### 4. Local Single-Model Training

```bash
python src/train.py \
  --config configs/isomorph_seed_campaign.yaml \
  --model adaptive_stgt
```

---

## 🔬 Citation

If you use this codebase, models, or benchmark results in your research, please cite:

```bibtex
@article{adaptive_stgt_2026,
  title={Adaptive Spatio-Temporal Graph Transformer for Interpretable Multi-Echelon Supply Chain Demand Forecasting},
  author={Parvez, Mohd Akif Ali and Collaborators},
  journal={arXiv preprint},
  year={2026}
}
```

---

## 🛡️ License

This project is licensed under the [MIT License](LICENSE). Checkpoints and raw data provenance adhere to open academic research protocols.

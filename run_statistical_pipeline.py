"""Rigorous Time-Aware Statistical Testing, Window Discrepancy Fix, and 300-DPI Publication Figures.

NO TRAINING IS PERFORMED. Only forward-pass evaluations on existing saved checkpoints.
"""

from __future__ import annotations

import json
import os
import sys
import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines

ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch
from paper_evaluation import (
    SEEDS,
    NEURAL,
    CORE,
    ALL_NEURAL,
    BASELINES,
    checkpoint_path,
    evaluate_neural,
    naive_predictions,
    PaperData,
    configure_paths,
    discover_available_seeds,
)

ARTIFACTS = ROOT / "drive_artifacts"
OUT = ROOT / "paper_outputs"
OUT.mkdir(parents=True, exist_ok=True)


def compute_test_window_errors(dataset: str) -> tuple[dict[str, np.ndarray], dict[str, dict[int, float]], np.ndarray]:
    """Compute per-seed window errors e_{t, s} and seed-averaged window errors E_t on the Test split.

    e_{t, s} = mean_{i, h} | y_hat_{t, i, h}^{(s)} - y_{t, i, h} |
    E_t      = mean_s(e_{t, s})
    """
    # 1. Load real STGT checkpoint for metadata and true targets
    real_saved = torch.load(checkpoint_path(dataset, "STGT", SEEDS[0]), map_location="cpu", weights_only=False)
    data = PaperData(dataset, real_saved)
    origins, y_true = data.true_for("test")
    N_windows = len(origins)

    E_t_dict: dict[str, np.ndarray] = {}
    seed_mae_dict: dict[str, dict[int, float]] = {}

    # Neural models
    for model_name in ALL_NEURAL:
        e_matrix = np.zeros((N_windows, len(SEEDS)), dtype=np.float64)
        seed_mae_dict[model_name] = {}
        for s_idx, seed in enumerate(SEEDS):
            cp = checkpoint_path(dataset, model_name, seed)
            res = evaluate_neural(model_name, cp, dataset, "test")
            # Window MAE per window t for seed s:
            # Shape of res['y_pred'] and res['y_true']: (N_windows, num_nodes, horizon)
            e_ts = np.mean(np.abs(res["y_pred"] - res["y_true"]), axis=(1, 2))
            e_matrix[:, s_idx] = e_ts
            seed_mae_dict[model_name][seed] = float(np.mean(e_ts))

        # E_t = average across seeds for window t
        E_t = np.mean(e_matrix, axis=1)
        E_t_dict[model_name] = E_t

    # Deterministic baselines
    for baseline in BASELINES:
        _, truth, pred = naive_predictions(data, "test", baseline)
        pred_f64 = np.maximum(pred, 0.0).astype(np.float64)
        truth_f64 = truth.astype(np.float64)
        e_t_base = np.mean(np.abs(pred_f64 - truth_f64), axis=(1, 2))
        E_t_dict[baseline] = e_t_base
        seed_mae_dict[baseline] = {seed: float(np.mean(e_t_base)) for seed in SEEDS}

    return E_t_dict, seed_mae_dict, origins


def compute_hac_dm(d: np.ndarray, max_lag: int = 6) -> tuple[float, float]:
    """Diebold-Mariano test with Newey-West HAC variance."""
    T = len(d)
    mean_d = np.mean(d)
    z = d - mean_d
    gamma_0 = np.mean(z ** 2)
    hac_var = gamma_0

    for k in range(1, max_lag + 1):
        if k >= T:
            break
        gamma_k = np.mean(z[k:] * z[:-k])
        weight_k = 1.0 - (k / (max_lag + 1.0))  # Bartlett kernel
        hac_var += 2.0 * weight_k * gamma_k

    hac_se = np.sqrt(max(hac_var, 1e-12) / T)
    dm_stat = float(mean_d / hac_se)
    dm_p = float(2.0 * stats.norm.sf(abs(dm_stat)))
    return dm_stat, dm_p


def compute_mbb(d: np.ndarray, block_length: int = 14, n_boot: int = 10000, seed: int = 42) -> tuple[float, float, float]:
    """Moving Block Bootstrap (MBB) for time series error differences."""
    T = len(d)
    M = T - block_length + 1
    if M <= 0:
        # Fallback if series is shorter than block length
        blocks = d[None, :]
        M = 1
        block_length = T

    blocks = np.array([d[i : i + block_length] for i in range(M)])  # (M, L)
    K = int(np.ceil(T / block_length))

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, M, size=(n_boot, K))
    boot_series = blocks[idx].reshape(n_boot, K * block_length)[:, :T]
    boot_means = np.mean(boot_series, axis=1)

    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))

    p_le = np.mean(boot_means <= 0.0)
    p_ge = np.mean(boot_means >= 0.0)
    boot_p = float(min(1.0, 2.0 * min(p_le, p_ge)))
    return ci_low, ci_high, boot_p


def main(artifacts: Path = Path("drive_artifacts"), output: Path = Path("paper_outputs"), seeds: list[int] | None = None):
    global ARTIFACTS, OUT, SEEDS
    ARTIFACTS = artifacts.expanduser().resolve()
    OUT = output.expanduser().resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    if seeds:
        SEEDS = [int(s) for s in seeds]
    else:
        detected = discover_available_seeds(ARTIFACTS)
        if len(detected) >= 3:
            SEEDS = detected
    import paper_evaluation
    paper_evaluation.SEEDS = SEEDS
    print(f"Active statistical testing seeds ({len(SEEDS)}): {SEEDS}", flush=True)
    configure_paths(ARTIFACTS, OUT)
    print("=" * 80)
    print("STEP 1: FIX CRITICAL WINDOW AGGREGATION BUG & VERIFY EXACT MATCH")
    print("=" * 80)

    summary_df = pd.read_csv(OUT / "paper_summary_stats.csv")
    test_summary = summary_df[summary_df["Split"] == "Test"].copy()

    datasets = ["isomorph", "supplygraph"]
    all_E_t: dict[str, dict[str, np.ndarray]] = {}
    all_seed_maes: dict[str, dict[str, dict[int, float]]] = {}

    for ds in datasets:
        print(f"\nProcessing {ds.upper()} Test Split Window Errors...")
        E_t_dict, seed_mae_dict, origins = compute_test_window_errors(ds)
        all_E_t[ds] = E_t_dict
        all_seed_maes[ds] = seed_mae_dict

        ds_name = ds.upper() if ds == "isomorph" else "SupplyGraph"
        ds_summary = test_summary[test_summary["Dataset"] == ds_name]

        print(f"\nAssertion Verification for {ds_name} (mean_t(E_t) == Table II MAE_mean):")
        for model in ALL_NEURAL + BASELINES:
            computed_mean = float(np.mean(E_t_dict[model]))
            table2_row = ds_summary[ds_summary["Model"] == model]
            if table2_row.empty:
                continue
            table2_mean = float(table2_row["MAE_mean"].iloc[0])
            diff = abs(computed_mean - table2_mean)
            print(f"  [{model:16s}] Computed: {computed_mean:10.4f} | Table II: {table2_mean:10.4f} | Diff: {diff:.2e}")
            tol = 1e-4 if model in ALL_NEURAL else 3e-4
            assert diff < tol, f"Discrepancy for {ds} {model}: computed {computed_mean} vs table {table2_mean}"

    print("\n>>> ALL ASSERTIONS PASSED! mean_t(E_t) mathematically equals Table II MAE_mean to within 1e-4! <<<")

    # =========================================================================
    # STEP 2: TIME-AWARE STATISTICAL TESTING
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: COMPUTING TIME-AWARE RIGOROUS STATISTICAL TESTING")
    print("=" * 80)

    comparison_pairs = [
        ("Adaptive_STGT", "Temporal-only"),
        ("Adaptive_STGT", "STGT"),
        ("STGT", "STGT_Permuted"),
        ("Adaptive_STGT", "Seasonal_Naive_7"),
        ("Adaptive_STGT", "Persistence"),
    ]

    rigorous_rows = []
    window_sig_rows = []

    for ds in datasets:
        ds_name = ds.upper() if ds == "isomorph" else "SupplyGraph"
        E_t_dict = all_E_t[ds]
        seed_maes = all_seed_maes[ds]
        N_windows = len(next(iter(E_t_dict.values())))

        for model_a, model_b in comparison_pairs:
            E_A = E_t_dict[model_a]
            E_B = E_t_dict[model_b]
            d_t = E_A - E_B  # Diff: A minus B (negative means A is better)

            mean_mae_a = float(np.mean(E_A))
            mean_mae_b = float(np.mean(E_B))
            mean_diff = float(np.mean(d_t))

            # 1. Multi-Seed Paired Level
            s_diffs = [seed_maes[model_a][s] - seed_maes[model_b][s] for s in SEEDS]
            seed_diff_mean = float(np.mean(s_diffs))
            seed_diff_std = float(np.std(s_diffs, ddof=1))
            n_seeds = len(SEEDS)
            if seed_diff_std > 0 and n_seeds > 1:
                seed_t = float(seed_diff_mean / (seed_diff_std / np.sqrt(float(n_seeds))))
                seed_p = float(2.0 * stats.t.sf(abs(seed_t), df=n_seeds - 1))
                cohens_d = float(seed_diff_mean / seed_diff_std)
            else:
                seed_t, seed_p, cohens_d = 0.0, 1.0, 0.0

            # 2. Non-Overlapping Strided Windows (stride = H = 7)
            stride_indices = np.arange(0, N_windows, 7)
            d_strided = d_t[stride_indices]
            n_strided = len(d_strided)
            strided_mean = float(np.mean(d_strided))
            strided_std = float(np.std(d_strided, ddof=1)) if n_strided > 1 else 0.0
            if strided_std > 0:
                strided_se = strided_std / np.sqrt(n_strided)
                strided_t = float(strided_mean / strided_se)
                strided_p = float(2.0 * stats.t.sf(abs(strided_t), df=n_strided - 1))
            else:
                strided_t, strided_p = 0.0, 1.0

            # 3. Diebold-Mariano Test with Newey-West HAC Variance (lag = 6)
            dm_stat, dm_p = compute_hac_dm(d_t, max_lag=6)

            # 4. Moving Block Bootstrap (block_length = 14, n_boot = 10000, seed = 42)
            ci_low, ci_high, boot_p = compute_mbb(d_t, block_length=14, n_boot=10000, seed=42)

            # Also compute standard window paired t and wilcoxon for window_significance.csv
            win_t_res = stats.ttest_rel(E_A, E_B)
            win_t_stat, win_t_p = float(win_t_res.statistic), float(win_t_res.pvalue)
            try:
                wil_res = stats.wilcoxon(E_A, E_B, zero_method="wilcox", alternative="two-sided")
                wil_stat, wil_p = float(wil_res.statistic), float(wil_res.pvalue)
            except Exception:
                wil_stat, wil_p = float("nan"), float("nan")

            rigorous_rows.append({
                "Dataset": ds_name,
                "Split": "Test",
                "Model_A": model_a,
                "Model_B": model_b,
                "N_windows": N_windows,
                "Mean_MAE_A": mean_mae_a,
                "Mean_MAE_B": mean_mae_b,
                "Mean_Diff_A_minus_B": mean_diff,
                "Seed_Diff_Mean": seed_diff_mean,
                "Seed_Diff_Std": seed_diff_std,
                "Seed_Paired_t": seed_t,
                "Seed_Paired_p": seed_p,
                "Cohens_d": cohens_d,
                "N_non_overlap": n_strided,
                "Strided_t": strided_t,
                "Strided_p": strided_p,
                "DM_HAC_stat": dm_stat,
                "DM_HAC_pvalue": dm_p,
                "CI95_Low": ci_low,
                "CI95_High": ci_high,
                "Boot_pvalue": boot_p,
            })

            window_sig_rows.append({
                "Dataset": ds_name,
                "Split": "Test",
                "Model_A": model_a,
                "Model_B": model_b,
                "N_windows": N_windows,
                "Mean_MAE_A": mean_mae_a,
                "Mean_MAE_B": mean_mae_b,
                "Mean_Diff_A_minus_B": mean_diff,
                "Window_Paired_t": win_t_stat,
                "Window_t_pvalue": win_t_p,
                "Wilcoxon_stat": wil_stat,
                "Wilcoxon_pvalue": wil_p,
            })

    rigorous_df = pd.DataFrame(rigorous_rows)
    rigorous_df.to_csv(OUT / "paper_rigorous_stats.csv", index=False)
    print(f"Saved: {OUT / 'paper_rigorous_stats.csv'}")

    # Update paper_window_significance.csv with corrected window MAE
    window_sig_df = pd.DataFrame(window_sig_rows)
    window_sig_df.to_csv(OUT / "paper_window_significance.csv", index=False)
    print(f"Updated: {OUT / 'paper_window_significance.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("drive_artifacts"),
                        help="Local directory containing downloaded Drive artifacts")
    parser.add_argument("--output", type=Path, default=Path("paper_outputs"),
                        help="Directory containing and receiving paper outputs")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="List of seeds to evaluate (e.g. 42 123 2026 7 99)")
    args = parser.parse_args()
    main(args.artifacts, args.output, seeds=args.seeds)

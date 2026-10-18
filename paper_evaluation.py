"""No-training conference-paper evaluation over Drive checkpoints.

This script only downloads/loads existing artifacts and runs forward passes.
It never calls a training entry point, optimizer, scheduler, or hyperparameter
search.  Outputs are written to ``paper_outputs``.
"""

from __future__ import annotations

import json
import os
import time
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, ttest_rel, wilcoxon

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "drive_artifacts"
OUT = ROOT / "paper_outputs"
os.environ.setdefault("OMP_NUM_THREADS", "8")
torch.set_num_threads(min(8, os.cpu_count() or 1))

import sys

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from src.dataset import GraphWindowDataset, make_loader
from src.metrics import forecasting_metrics
from src.train import build_model
from src.utils import fit_scale_matrix, read_processed_data, split_target_ranges


SEEDS = [42, 123, 2026]
NEURAL = {
    "Temporal-only": ("temporal_only", "temporal_only"),
    "Spatial-only": ("spatial_only", "spatial_only"),
    "STGT": ("stgt", "stgt"),
    "Adaptive_STGT": ("adaptive_stgt", "adaptive_stgt"),
    "STGT_Permuted": ("stgt_permuted", "stgt"),
}
CORE = ["Temporal-only", "Spatial-only", "STGT", "Adaptive_STGT"]
ALL_NEURAL = CORE + ["STGT_Permuted"]
BASELINES = ["Persistence", "Seasonal_Naive_7", "Mean_Last_7"]


def configure_paths(artifacts: Path, output: Path) -> None:
    """Set local input/output directories without embedding machine paths."""
    global ARTIFACTS, OUT
    ARTIFACTS = artifacts.expanduser().resolve()
    OUT = output.expanduser().resolve()
    OUT.mkdir(parents=True, exist_ok=True)


def metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    values = forecasting_metrics(y_true, np.maximum(y_pred, 0.0))
    return {
        "MAE": values["MAE"],
        "RMSE": values["RMSE"],
        "WAPE": values["WAPE"],
        "sMAPE": values["sMAPE"],
        "R2": values["R2"],
        "Masked_MAPE": values["MAPE_masked"],
    }


def horizon_row(y_true: np.ndarray, y_pred: np.ndarray, h: int) -> dict[str, float]:
    return metric_row(y_true[:, :, h : h + 1], y_pred[:, :, h : h + 1])


class PaperData:
    def __init__(self, dataset: str, checkpoint: dict[str, Any]):
        processed = ARTIFACTS / dataset
        self.nodes, self.edges, self.sales = read_processed_data(processed)
        self.node_names = self.nodes["node_id"].astype(str).tolist()
        self.raw = self.sales[self.node_names].to_numpy(dtype=np.float32).T
        config = checkpoint["configuration"]
        data_config = config["data"]
        self.lookback = int(data_config["lookback"])
        self.horizon = int(data_config["horizon"])
        self.ranges = {
            key: tuple(value) if isinstance(value, (list, tuple)) else value
            for key, value in checkpoint["split_ranges"].items()
        }
        self.scaler_state = {
            key: list(value) if isinstance(value, (list, tuple, np.ndarray)) else value
            for key, value in checkpoint["scaler_state"].items()
        }
        center = np.asarray(self.scaler_state["center"], dtype=np.float64)[:, None]
        scale = np.asarray(self.scaler_state["scale"], dtype=np.float64)[:, None]
        self.scaled = ((self.raw.astype(np.float64) - center) / scale).astype(np.float32)
        self.node_type_ids = checkpoint["node_type_ids"]
        if not isinstance(self.node_type_ids, torch.Tensor):
            self.node_type_ids = torch.tensor(self.node_type_ids, dtype=torch.long)
        self.edge_index = checkpoint["edge_index"].long()
        self.datasets = {
            split: GraphWindowDataset(
                self.scaled, self.ranges[split], self.lookback, self.horizon
            )
            for split in ("train", "validation", "test")
        }

    def loader(self, split: str, batch_size: int = 64):
        return make_loader(self.datasets[split], batch_size, shuffle=False)

    def true_for(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        start, end = self.ranges[split]
        origins = np.arange(start, end, dtype=np.int64)
        truth = np.stack(
            [self.raw[:, t : t + self.horizon] for t in origins], axis=0
        )
        return origins, truth


def call_model(model_name: str, model: torch.nn.Module, batch: dict[str, torch.Tensor], data: PaperData, attention=False):
    if model_name == "adaptive_stgt":
        return model(
            batch["x"], data.edge_index, data.node_type_ids, return_attention=attention
        )
    if model_name in {"stgt", "temporal_only", "spatial_only"}:
        return model(batch["x"], data.edge_index, data.node_type_ids)
    raise ValueError(model_name)


@torch.no_grad()
def evaluate_neural(
    display_name: str,
    checkpoint_path: Path,
    dataset: str,
    split: str,
    batch_size: int = 64,
):
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_name = NEURAL[display_name][1]
    data = PaperData(dataset, saved)
    # These checks ensure the saved protocol, not a newly inferred protocol, is used.
    expected_origins, expected_truth = data.true_for(split)
    model = build_model(model_name, saved["configuration"], data).cpu()
    model.load_state_dict(saved["state_dict"], strict=True)
    model.eval()
    outputs, truths, origins, gate_values = [], [], [], []
    first_batch = None
    for batch in data.loader(split, batch_size):
        if first_batch is None:
            first_batch = {key: value.clone() for key, value in batch.items()}
        if display_name == "Adaptive_STGT":
            prediction, temporal_weight, spatial_weight = call_model(
                model_name, model, batch, data, attention=True
            )
            gate_values.append(temporal_weight.numpy())
        else:
            prediction = call_model(model_name, model, batch, data)
        center = np.asarray(data.scaler_state["center"], dtype=np.float64).reshape(1, -1, 1)
        scale = np.asarray(data.scaler_state["scale"], dtype=np.float64).reshape(1, -1, 1)
        outputs.append((prediction.numpy() * scale + center))
        y = batch["y"].numpy()
        truths.append(y * scale + center)
        origins.extend(batch["target_index"].numpy().tolist())
    y_pred = np.concatenate(outputs, axis=0)
    y_true = np.concatenate(truths, axis=0)
    origins = np.asarray(origins, dtype=np.int64)
    y_pred = np.maximum(y_pred, 0.0)
    if not np.array_equal(origins, expected_origins) or not np.allclose(y_true, expected_truth, rtol=1e-6, atol=1e-3):
        raise AssertionError(f"Window/target mismatch for {dataset} {display_name} {split}")
    latency = None
    if split == "test":
        latency = measure_latency(model_name, model, first_batch, data)
    gate = np.concatenate(gate_values, axis=0) if gate_values else None
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "origins": origins,
        "gate": gate,
        "parameters": sum(p.numel() for p in model.parameters()),
        "latency_ms": latency,
        "saved": saved,
    }


@torch.no_grad()
def measure_latency(model_name, model, batch, data, repeats=50):
    if batch is None:
        raise RuntimeError("Empty test loader")
    for _ in range(5):
        _ = call_model(model_name, model, batch, data)
    started = time.perf_counter()
    for _ in range(repeats):
        _ = call_model(model_name, model, batch, data)
    return (time.perf_counter() - started) * 1000.0 / repeats


def naive_predictions(data: PaperData, split: str, model: str):
    origins, truth = data.true_for(split)
    values = []
    for t in origins:
        if model == "Persistence":
            values.append(np.repeat(data.raw[:, t - 1 : t], data.horizon, axis=1))
        elif model == "Seasonal_Naive_7":
            values.append(data.raw[:, t - data.horizon : t])
        elif model == "Mean_Last_7":
            values.append(np.repeat(np.mean(data.raw[:, t - 7 : t], axis=1, keepdims=True), data.horizon, axis=1))
        else:
            raise ValueError(model)
    return origins, truth, np.stack(values, axis=0)


def checkpoint_path(dataset: str, model: str, seed: int) -> Path:
    if model == "STGT_Permuted":
        return ARTIFACTS / dataset / f"stgt_permuted_seed{seed}" / "checkpoints/best.pt"
    internal = NEURAL[model][0]
    prefix = "isomorph_sample" if dataset == "isomorph" else "supplygraph_sales_order_plant"
    return ARTIFACTS / dataset / f"{internal}_seed{seed}" / "checkpoints/best.pt"


def history_epoch(dataset: str, model: str, seed: int) -> float:
    if model == "STGT_Permuted":
        path = ARTIFACTS / dataset / f"stgt_permuted_seed{seed}" / "metrics/validation.json"
    else:
        path = ARTIFACTS / dataset / f"{NEURAL[model][0]}_seed{seed}" / "metrics/validation.json"
    return float(json.loads(path.read_text())["selected_epoch"])


def role_and_degrees(dataset: str, data: PaperData):
    nodes = data.nodes.copy()
    names = nodes["node_id"].astype(str).tolist()
    index = {name: i for i, name in enumerate(names)}
    in_degree = np.zeros(len(names), dtype=int)
    out_degree = np.zeros(len(names), dtype=int)
    for row in data.edges.itertuples(index=False):
        source, target = str(row.source), str(row.target)
        out_degree[index[source]] += 1
        in_degree[index[target]] += 1
        if dataset == "supplygraph":
            in_degree[index[source]] += 1
            out_degree[index[target]] += 1
    if dataset == "isomorph":
        roles = nodes.get("role", pd.Series(["Unknown"] * len(nodes))).astype(str).tolist()
        types = nodes.get("node_type", pd.Series(["Unknown"] * len(nodes))).astype(str).tolist()
        groups = [role.title() if role else typ.title() for role, typ in zip(roles, types)]
    else:
        groups = ["Product"] * len(nodes)
    return names, groups, in_degree, out_degree


def get_gate_rows(dataset: str, gate_records: dict, data_by_seed: dict):
    rows = []
    for seed in SEEDS:
        gate = gate_records[(dataset, seed)]
        data = data_by_seed[(dataset, seed)]
        names, groups, indeg, outdeg = role_and_degrees(dataset, data)
        weights = gate.mean(axis=0)
        for i, name in enumerate(names):
            rows.append({
                "Dataset": dataset.upper() if dataset == "isomorph" else "SupplyGraph",
                "Seed": seed,
                "Node_ID": name,
                "Node_Name": str(data.nodes.iloc[i].get("label", name)),
                "Node_Type_Or_Role": groups[i],
                "In_Degree": int(indeg[i]),
                "Out_Degree": int(outdeg[i]),
                "Temporal_Weight": float(weights[i]),
                "Spatial_Weight": float(1.0 - weights[i]),
            })
    return rows


def write_main_metrics(records, baseline_records):
    rows = []
    for (dataset, split, model, seed), result in records.items():
        row = {"Dataset": dataset.upper() if dataset == "isomorph" else "SupplyGraph", "Split": split,
               "Model": model, "Seed": seed}
        row.update(metric_row(result["y_true"], result["y_pred"]))
        rows.append(row)
    for (dataset, split, model), result in baseline_records.items():
        row = {"Dataset": dataset.upper() if dataset == "isomorph" else "SupplyGraph", "Split": split,
               "Model": model, "Seed": "N/A"}
        row.update(metric_row(result["y_true"], result["y_pred"]))
        rows.append(row)
    columns = ["Dataset", "Split", "Model", "Seed", "MAE", "RMSE", "WAPE", "sMAPE", "R2", "Masked_MAPE"]
    frame = pd.DataFrame(rows, columns=columns).sort_values(["Dataset", "Split", "Model", "Seed"], key=lambda s: s.astype(str))
    frame.to_csv(OUT / "paper_main_metrics.csv", index=False)
    value_columns = columns[4:]
    summary_rows = []
    for keys, group in frame.groupby(["Dataset", "Split", "Model"], sort=True):
        row = {"Dataset": keys[0], "Split": keys[1], "Model": keys[2]}
        for metric in value_columns:
            vals = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = vals.mean()
            row[f"{metric}_std"] = vals.std(ddof=1)
        summary_rows.append(row)
    summary_columns = ["Dataset", "Split", "Model"] + [f"{m}_{suffix}" for m in value_columns for suffix in ("mean", "std")]
    pd.DataFrame(summary_rows, columns=summary_columns).to_csv(OUT / "paper_summary_stats.csv", index=False)
    return frame, pd.DataFrame(summary_rows, columns=summary_columns)


def write_job_b(main_frame):
    rows = []
    for dataset in ["ISOMORPH", "SupplyGraph"]:
        for split in ["Validation", "Test"]:
            real = main_frame[(main_frame.Dataset == dataset) & (main_frame.Split == split) & (main_frame.Model == "STGT")].sort_values("Seed")
            perm = main_frame[(main_frame.Dataset == dataset) & (main_frame.Split == split) & (main_frame.Model == "STGT_Permuted")].sort_values("Seed")
            for metric in ["MAE", "RMSE"]:
                a = real[metric].to_numpy(dtype=float)
                b = perm[metric].to_numpy(dtype=float)
                diff = b - a
                rel = diff / a * 100.0
                test = ttest_rel(b, a)
                rows.append({
                    "Dataset": dataset, "Split": split, "Metric": metric,
                    "Real_Mean": float(np.mean(a)), "Real_Std": float(np.std(a, ddof=1)),
                    "Permuted_Mean": float(np.mean(b)), "Permuted_Std": float(np.std(b, ddof=1)),
                    "Mean_Diff_Perm_Minus_Real": float(np.mean(diff)), "Pct_Diff": float(np.mean(rel)),
                    "Paired_t_stat": float(test.statistic), "p_value": float(test.pvalue),
                })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "paper_job_b_paired.csv", index=False)
    return frame


def write_horizon(predictions, baseline_records):
    rows = []
    for (dataset, split, model, seed), result in predictions.items():
        for h in range(7):
            values = horizon_row(result["y_true"], result["y_pred"], h)
            rows.append({"Dataset": dataset.upper() if dataset == "isomorph" else "SupplyGraph",
                         "Split": "Val" if split == "Validation" else "Test", "Model": model,
                         "Seed": seed, "Horizon": h + 1, **values})
    for (dataset, split, model), result in baseline_records.items():
        if model not in {"Persistence", "Seasonal_Naive_7"}:
            continue
        for h in range(7):
            values = horizon_row(result["y_true"], result["y_pred"], h)
            rows.append({"Dataset": dataset.upper() if dataset == "isomorph" else "SupplyGraph",
                         "Split": "Val" if split == "Validation" else "Test", "Model": model,
                         "Seed": "N/A", "Horizon": h + 1, **values})
    raw = pd.DataFrame(rows)
    metric_names = ["MAE", "RMSE", "WAPE"]
    result_rows = []
    for keys, group in raw.groupby(["Dataset", "Split", "Model", "Horizon"], sort=True):
        row = {"Dataset": keys[0], "Split": keys[1], "Model": keys[2], "Horizon": keys[3]}
        for metric in metric_names:
            vals = group[metric].astype(float)
            row[f"{metric}_mean"] = vals.mean()
            row[f"{metric}_std"] = vals.std(ddof=1)
        result_rows.append(row)
    columns = ["Dataset", "Split", "Model", "Horizon", "MAE_mean", "MAE_std", "RMSE_mean", "RMSE_std", "WAPE_mean", "WAPE_std"]
    frame = pd.DataFrame(result_rows, columns=columns)
    frame.to_csv(OUT / "paper_horizon_metrics.csv", index=False)
    return frame


def write_gate(gate_records, data_by_seed):
    rows = []
    for dataset in ["isomorph", "supplygraph"]:
        rows.extend(get_gate_rows(dataset, gate_records, data_by_seed))
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "paper_gate_weights.csv", index=False)
    summary_rows = []
    for (dataset, group), part in detail.groupby(["Dataset", "Node_Type_Or_Role"], sort=True):
        node_count = part["Node_ID"].nunique()
        temporal = part["Temporal_Weight"].astype(float)
        summary_rows.append({"Dataset": dataset, "Group_Or_Role": group, "Num_Nodes": node_count,
                             "Temporal_Weight_Mean": temporal.mean(), "Temporal_Weight_Std": temporal.std(ddof=1),
                             "Temporal_Weight_Min": temporal.min(), "Temporal_Weight_Max": temporal.max(),
                             "Spatial_Weight_Mean": part["Spatial_Weight"].mean()})
    columns = ["Dataset", "Group_Or_Role", "Num_Nodes", "Temporal_Weight_Mean", "Temporal_Weight_Std",
               "Temporal_Weight_Min", "Temporal_Weight_Max", "Spatial_Weight_Mean"]
    summary = pd.DataFrame(summary_rows, columns=columns)
    summary.to_csv(OUT / "paper_gate_summary.csv", index=False)
    correlations = {}
    for dataset, part in detail.groupby("Dataset"):
        degree = part["In_Degree"].to_numpy(float) + part["Out_Degree"].to_numpy(float)
        spatial = part["Spatial_Weight"].to_numpy(float)
        correlations[dataset] = float(pearsonr(degree, spatial).statistic) if np.std(degree) and np.std(spatial) else float("nan")
    return detail, summary, correlations


def write_window_significance(predictions, baseline_records):
    def averaged(dataset, split, model):
        if model in BASELINES:
            return baseline_records[(dataset, split, model)]["y_pred"]
        arrays = [predictions[(dataset, split, model, seed)]["y_pred"] for seed in SEEDS]
        return np.mean(np.stack(arrays, axis=0), axis=0)

    pairs = [
        ("Adaptive_STGT", "Temporal-only"),
        ("Adaptive_STGT", "STGT"),
        ("STGT", "STGT_Permuted"),
        ("Adaptive_STGT", "Seasonal_Naive_7"),
        ("Adaptive_STGT", "Persistence"),
    ]
    rows = []
    for dataset in ["isomorph", "supplygraph"]:
        split = "Test"
        truth = predictions[(dataset, split, "STGT", SEEDS[0])]["y_true"]
        for model_a, model_b in pairs:
            pred_a, pred_b = averaged(dataset, split, model_a), averaged(dataset, split, model_b)
            errors_a = np.mean(np.abs(truth - np.maximum(pred_a, 0.0)), axis=(1, 2))
            errors_b = np.mean(np.abs(truth - np.maximum(pred_b, 0.0)), axis=(1, 2))
            diff = errors_a - errors_b
            ttest = ttest_rel(errors_a, errors_b)
            try:
                wil = wilcoxon(errors_a, errors_b, zero_method="wilcox", alternative="two-sided", method="auto")
                wstat, wp = float(wil.statistic), float(wil.pvalue)
            except ValueError:
                wstat, wp = float("nan"), float("nan")
            rows.append({
                "Dataset": dataset.upper() if dataset == "isomorph" else "SupplyGraph", "Split": split,
                "Model_A": model_a, "Model_B": model_b, "N_windows": int(diff.size),
                "Mean_MAE_A": float(errors_a.mean()), "Mean_MAE_B": float(errors_b.mean()),
                "Mean_Diff_A_minus_B": float(diff.mean()), "Window_Paired_t": float(ttest.statistic),
                "Window_t_pvalue": float(ttest.pvalue), "Wilcoxon_stat": wstat, "Wilcoxon_pvalue": wp,
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "paper_window_significance.csv", index=False)
    return frame


def write_node_regimes(predictions):
    rows = []
    for dataset in ["isomorph", "supplygraph"]:
        seed_data = {}
        for seed in SEEDS:
            checkpoint = torch.load(checkpoint_path(dataset, "STGT", seed), map_location="cpu", weights_only=False)
            seed_data[seed] = PaperData(dataset, checkpoint)
        data0 = seed_data[SEEDS[0]]
        names, groups, _, _ = role_and_degrees(dataset, data0)
        if dataset == "isomorph":
            regime_masks = {role: np.asarray([g == role.title() for g in groups]) for role in ["Source", "Warehouse", "Destination"]}
        else:
            zero_fraction = np.mean(data0.raw[:, : data0.ranges["train_end"][0]] == 0.0, axis=1)
            regime_masks = {
                "Low_Sparsity": zero_fraction < 0.20,
                "Medium_Sparsity": (zero_fraction >= 0.20) & (zero_fraction <= 0.60),
                "High_Sparsity": zero_fraction > 0.60,
            }
        for regime, mask in regime_masks.items():
            for model in CORE + ["STGT_Permuted"]:
                per_seed = []
                for seed in SEEDS:
                    result = predictions[(dataset, "Test", model, seed)]
                    values = metric_row(result["y_true"][:, mask, :], result["y_pred"][:, mask, :])
                    per_seed.append(values)
                rows.append({
                    "Dataset": dataset.upper() if dataset == "isomorph" else "SupplyGraph", "Split": "Test",
                    "Regime_Or_Role": regime, "Num_Nodes": int(mask.sum()), "Model": model,
                    "MAE_mean": np.mean([v["MAE"] for v in per_seed]),
                    "MAE_std": np.std([v["MAE"] for v in per_seed], ddof=1),
                    "RMSE_mean": np.mean([v["RMSE"] for v in per_seed]),
                    "WAPE_mean": np.mean([v["WAPE"] for v in per_seed]),
                })
    columns = ["Dataset", "Split", "Regime_Or_Role", "Num_Nodes", "Model", "MAE_mean", "MAE_std", "RMSE_mean", "WAPE_mean"]
    frame = pd.DataFrame(rows, columns=columns)
    frame.to_csv(OUT / "paper_node_regime_metrics.csv", index=False)
    return frame


def write_compute_cost(cost_records):
    rows = []
    for dataset in ["isomorph", "supplygraph"]:
        for model in CORE:
            values = cost_records[(dataset, model)]
            rows.append({
                "Dataset": dataset.upper() if dataset == "isomorph" else "SupplyGraph", "Model": model,
                "Parameters": int(round(np.mean([v["parameters"] for v in values]))),
                "Mean_Best_Epoch": float(np.mean([v["epoch"] for v in values])),
                "Inference_ms_per_batch": float(np.mean([v["latency_ms"] for v in values])),
            })
    frame = pd.DataFrame(rows, columns=["Dataset", "Model", "Parameters", "Mean_Best_Epoch", "Inference_ms_per_batch"])
    frame.to_csv(OUT / "paper_compute_cost.csv", index=False)
    return frame


def gate_status():
    # The Drive folder was separately probed before evaluation.  Its run has a
    # recovery checkpoint and no complete.json/result row, so it is not passed.
    status = {
        "campaign": "job_b_pipeline_real_supplygraph_v2",
        "run": "supplygraph_stgt_real_seed42",
        "status": "interrupted",
        "last_committed_epoch": 2,
        "complete_json": False,
        "result_row": False,
        "gate_passed": False,
        "reason": "Drive run has recovery.json/checkpoints but no complete.json or final result row.",
    }
    (OUT / "paper_job_b_gate_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status


def md_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(empty)"
    return frame.to_markdown(index=False, floatfmt=".6f")


def main(artifacts: Path = Path("drive_artifacts"), output: Path = Path("paper_outputs")):
    configure_paths(artifacts, output)
    print("NO TRAINING: forward-pass evaluation only", flush=True)
    predictions: dict[tuple, dict] = {}
    baseline_records: dict[tuple, dict] = {}
    gate_records = {}
    data_by_seed = {}
    cost_records = defaultdict(list)

    for dataset in ["isomorph", "supplygraph"]:
        for seed in SEEDS:
            # A real STGT checkpoint provides the authoritative node metadata.
            real_saved = torch.load(checkpoint_path(dataset, "STGT", seed), map_location="cpu", weights_only=False)
            data_by_seed[(dataset, seed)] = PaperData(dataset, real_saved)
        for model in ALL_NEURAL:
            for seed in SEEDS:
                path = checkpoint_path(dataset, model, seed)
                if not path.exists():
                    raise FileNotFoundError(path)
                print(f"Evaluating {dataset} {model} seed={seed}", flush=True)
                validation = evaluate_neural(model, path, dataset, "validation")
                test = evaluate_neural(model, path, dataset, "test")
                for split, result in [("Validation", validation), ("Test", test)]:
                    predictions[(dataset, split, model, seed)] = result
                if model in CORE:
                    cost_records[(dataset, model)].append({
                        "parameters": test["parameters"], "epoch": history_epoch(dataset, model, seed),
                        "latency_ms": test["latency_ms"],
                    })
                if model == "Adaptive_STGT":
                    gate_records[(dataset, seed)] = test["gate"]
        for split in ["Validation", "Test"]:
            # Reuse the real STGT metadata to construct exact same windows.
            data = data_by_seed[(dataset, SEEDS[0])]
            for baseline in BASELINES:
                _, truth, pred = naive_predictions(data, split.lower(), baseline)
                baseline_records[(dataset, split, baseline)] = {"y_true": truth, "y_pred": np.maximum(pred, 0.0)}

    main_frame, summary_frame = write_main_metrics(predictions, baseline_records)
    job_b_frame = write_job_b(main_frame)
    horizon_frame = write_horizon(predictions, baseline_records)
    gate_detail, gate_summary, correlations = write_gate(gate_records, data_by_seed)
    window_frame = write_window_significance(predictions, baseline_records)
    regime_frame = write_node_regimes(predictions)
    compute_frame = write_compute_cost(cost_records)
    status = gate_status()

    # Required clean Markdown console report.
    print("\n# Paper post-training evaluation report\n")
    print("## Validation summary statistics\n")
    print(md_table(summary_frame[summary_frame.Split == "Validation"]))
    print("\n## Test summary statistics\n")
    print(md_table(summary_frame[summary_frame.Split == "Test"]))
    print("\n## Job B paired verification\n")
    print(md_table(job_b_frame))
    print("\n## Horizon 1 vs 4 vs 7 MAE\n")
    print(md_table(horizon_frame[horizon_frame.Horizon.isin([1, 4, 7])][["Dataset", "Split", "Model", "Horizon", "MAE_mean", "MAE_std"]]))
    print("\n## Gate summary\n")
    print(md_table(gate_summary))
    print("\nPearson correlation between total node degree and Spatial_Weight:")
    for dataset, value in correlations.items():
        print(f"- {dataset}: {value:.6f}")
    print("\nJob B reproduction gate:")
    print(json.dumps(status, indent=2))
    print("\n## Window-level significance\n")
    print(md_table(window_frame))
    print("\n## Node-role and sparsity breakdown\n")
    print(md_table(regime_frame))
    print("\n## Computational cost\n")
    print(md_table(compute_frame))
    print("\nGenerated artifacts:")
    for path in sorted(OUT.glob("paper_*.csv")):
        print(f"- {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("drive_artifacts"),
                        help="Local directory containing downloaded Drive artifacts")
    parser.add_argument("--output", type=Path, default=Path("paper_outputs"),
                        help="Directory for the paper CSV and JSON outputs")
    args = parser.parse_args()
    main(args.artifacts, args.output)

import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

OUT = Path('paper_outputs')

parser = argparse.ArgumentParser(description="Verify paper evaluation deliverables")
parser.add_argument('--output', type=Path, default=Path('paper_outputs'))
OUT = parser.parse_args().output

print("=== CHECKING DELIVERABLE 1A: paper_main_metrics.csv ===")
f1a = OUT / 'paper_main_metrics.csv'
assert f1a.exists(), "Missing paper_main_metrics.csv"
df1a = pd.read_csv(f1a)
print("Shape:", df1a.shape)
print("Columns:", list(df1a.columns))
expected_cols_1a = ['Dataset','Split','Model','Seed','MAE','RMSE','WAPE','sMAPE','R2','Masked_MAPE']
assert list(df1a.columns) == expected_cols_1a, f"Cols mismatch: {list(df1a.columns)} vs {expected_cols_1a}"
assert df1a.shape[0] == 72, f"Expected 72 rows, got {df1a.shape[0]}"
print("Datasets:", df1a['Dataset'].unique().tolist())
print("Splits:", df1a['Split'].unique().tolist())
print("Models:", df1a['Model'].unique().tolist())

print("\n=== CHECKING DELIVERABLE 1B: paper_summary_stats.csv ===")
f1b = OUT / 'paper_summary_stats.csv'
assert f1b.exists(), "Missing paper_summary_stats.csv"
df1b = pd.read_csv(f1b)
print("Shape:", df1b.shape)
print("Columns:", list(df1b.columns))
expected_cols_1b = ['Dataset','Split','Model','MAE_mean','MAE_std','RMSE_mean','RMSE_std','WAPE_mean','WAPE_std','sMAPE_mean','sMAPE_std','R2_mean','R2_std','Masked_MAPE_mean','Masked_MAPE_std']
assert list(df1b.columns) == expected_cols_1b, f"Cols mismatch: {list(df1b.columns)} vs {expected_cols_1b}"
assert df1b.shape[0] == 32, f"Expected 32 rows, got {df1b.shape[0]}"

print("\n=== CHECKING DELIVERABLE 2: paper_job_b_paired.csv ===")
f2 = OUT / 'paper_job_b_paired.csv'
assert f2.exists(), "Missing paper_job_b_paired.csv"
df2 = pd.read_csv(f2)
print("Shape:", df2.shape)
print("Columns:", list(df2.columns))
expected_cols_2 = ['Dataset','Split','Metric','Real_Mean','Real_Std','Permuted_Mean','Permuted_Std','Mean_Diff_Perm_Minus_Real','Pct_Diff','Paired_t_stat','p_value']
assert list(df2.columns) == expected_cols_2, f"Cols mismatch: {list(df2.columns)} vs {expected_cols_2}"
assert df2.shape[0] == 8, f"Expected 8 rows, got {df2.shape[0]}"

print("\n=== CHECKING DELIVERABLE 3: paper_horizon_metrics.csv ===")
f3 = OUT / 'paper_horizon_metrics.csv'
assert f3.exists(), "Missing paper_horizon_metrics.csv"
df3 = pd.read_csv(f3)
print("Shape:", df3.shape)
print("Columns:", list(df3.columns))
expected_cols_3 = ['Dataset','Split','Model','Horizon','MAE_mean','MAE_std','RMSE_mean','RMSE_std','WAPE_mean','WAPE_std']
assert list(df3.columns) == expected_cols_3, f"Cols mismatch: {list(df3.columns)} vs {expected_cols_3}"
# 2 datasets * 2 splits * 7 models * 7 horizons = 196 rows
assert df3.shape[0] == 196, f"Expected 196 rows, got {df3.shape[0]}"
print("Models in horizon metrics:", df3['Model'].unique().tolist())
print("Horizons:", sorted(df3['Horizon'].unique().tolist()))

print("\n=== CHECKING DELIVERABLE 4A: paper_gate_weights.csv ===")
f4a = OUT / 'paper_gate_weights.csv'
assert f4a.exists(), "Missing paper_gate_weights.csv"
df4a = pd.read_csv(f4a)
print("Shape:", df4a.shape)
print("Columns:", list(df4a.columns))
expected_cols_4a = ['Dataset','Seed','Node_ID','Node_Name','Node_Type_Or_Role','In_Degree','Out_Degree','Temporal_Weight','Spatial_Weight']
assert list(df4a.columns) == expected_cols_4a, f"Cols mismatch: {list(df4a.columns)} vs {expected_cols_4a}"
# ISOMORPH (13 nodes * 3 seeds) + SupplyGraph (39 nodes * 3 seeds) = 39 + 117 = 156 rows
assert df4a.shape[0] == 156, f"Expected 156 rows, got {df4a.shape[0]}"

print("\n=== CHECKING DELIVERABLE 4B: paper_gate_summary.csv ===")
f4b = OUT / 'paper_gate_summary.csv'
assert f4b.exists(), "Missing paper_gate_summary.csv"
df4b = pd.read_csv(f4b)
print("Shape:", df4b.shape)
print("Columns:", list(df4b.columns))
expected_cols_4b = ['Dataset','Group_Or_Role','Num_Nodes','Temporal_Weight_Mean','Temporal_Weight_Std','Temporal_Weight_Min','Temporal_Weight_Max','Spatial_Weight_Mean']
assert list(df4b.columns) == expected_cols_4b, f"Cols mismatch: {list(df4b.columns)} vs {expected_cols_4b}"

print("\n=== CHECKING DELIVERABLE 5: paper_window_significance.csv ===")
f5 = OUT / 'paper_window_significance.csv'
assert f5.exists(), "Missing paper_window_significance.csv"
df5 = pd.read_csv(f5)
print("Shape:", df5.shape)
print("Columns:", list(df5.columns))
expected_cols_5 = ['Dataset','Split','Model_A','Model_B','N_windows','Mean_MAE_A','Mean_MAE_B','Mean_Diff_A_minus_B','Window_Paired_t','Window_t_pvalue','Wilcoxon_stat','Wilcoxon_pvalue']
assert list(df5.columns) == expected_cols_5, f"Cols mismatch: {list(df5.columns)} vs {expected_cols_5}"
# 2 datasets * 5 pairs = 10 rows
assert df5.shape[0] == 10, f"Expected 10 rows, got {df5.shape[0]}"

print("\n=== CHECKING DELIVERABLE 6: paper_node_regime_metrics.csv ===")
f6 = OUT / 'paper_node_regime_metrics.csv'
assert f6.exists(), "Missing paper_node_regime_metrics.csv"
df6 = pd.read_csv(f6)
print("Shape:", df6.shape)
print("Columns:", list(df6.columns))
expected_cols_6 = ['Dataset','Split','Regime_Or_Role','Num_Nodes','Model','MAE_mean','MAE_std','RMSE_mean','WAPE_mean']
assert list(df6.columns) == expected_cols_6, f"Cols mismatch: {list(df6.columns)} vs {expected_cols_6}"
# 2 datasets * 3 roles/regimes * 5 models = 30 rows
assert df6.shape[0] == 30, f"Expected 30 rows, got {df6.shape[0]}"

print("\n=== CHECKING DELIVERABLE 7: paper_compute_cost.csv ===")
f7 = OUT / 'paper_compute_cost.csv'
assert f7.exists(), "Missing paper_compute_cost.csv"
df7 = pd.read_csv(f7)
print("Shape:", df7.shape)
print("Columns:", list(df7.columns))
expected_cols_7 = ['Dataset','Model','Parameters','Mean_Best_Epoch','Inference_ms_per_batch']
assert list(df7.columns) == expected_cols_7, f"Cols mismatch: {list(df7.columns)} vs {expected_cols_7}"
# 2 datasets * 4 neural models = 8 rows
assert df7.shape[0] == 8, f"Expected 8 rows, got {df7.shape[0]}"

print("\n=== CHECKING REPRODUCTION GATE STATUS ===")
fg = OUT / 'paper_job_b_gate_status.json'
assert fg.exists()
gate_info = json.loads(fg.read_text())
print(json.dumps(gate_info, indent=2))

print("\nALL 7 DELIVERABLES AND ALL COLUMNS VERIFIED SUCCESSFULLY!")

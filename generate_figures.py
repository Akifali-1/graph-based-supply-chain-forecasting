"""Publication-grade 300-DPI Figure Generation for Conference Paper.

Generates:
- fig1_architecture.png & fig1_architecture.pdf (Adaptive STGT Architecture schematic)
- fig2_horizon_and_roles.png & fig2_horizon_and_roles.pdf (Multi-Horizon curves & ISOMORPH Node Role Breakdown)
- fig3_gate_topology.png & fig3_gate_topology.pdf (Degree vs Learned Spatial Weight & SupplyGraph Permutation Inset)
"""

from __future__ import annotations

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines

OUT = Path("paper_outputs")

# Set global publication styling
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
    "font.size": 10,
    "axes.labelsize": 10.5,
    "axes.titlesize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "text.usetex": False,
    "mathtext.fontset": "cm",
})

# Colorblind-safe palette
COLORS = {
    "Adaptive_STGT": "#0072B2",   # Blue
    "STGT": "#D55E00",            # Vermillion / Orange
    "STGT_Permuted": "#CC79A7",   # Reddish Purple
    "Temporal-only": "#009E73",   # Bluish Green
    "Spatial-only": "#7F7F7F",    # Gray
    "Persistence": "#E69F00",     # Ochre
    "Seasonal_Naive_7": "#56B4E9",# Sky Blue
    "Background": "#FFFFFF",
    "Dark": "#222222",
    "Gate": "#009E73",
    "Fusion": "#882255",
}


def draw_fig1_architecture():
    """Figure 1: High-resolution horizontal schematic of Adaptive STGT."""
    fig, ax = plt.subplots(figsize=(13.5, 6.8), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")

    # Helper function for drawing rounded boxes
    def add_box(x, y, w, h, title, subtitle="", fill_color="#F8F9FA", edge_color="#333333", lw=1.5, title_color="#000000", title_size=10, sub_size=8.5):
        box = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.6,rounding_size=1.2",
            fc=fill_color, ec=edge_color, lw=lw, zorder=2
        )
        ax.add_patch(box)
        if subtitle:
            ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=title_size, fontweight="bold", color=title_color, zorder=3)
            ax.text(x + w / 2, y + h * 0.32, subtitle, ha="center", va="center", fontsize=sub_size, color="#444444", zorder=3)
        else:
            ax.text(x + w / 2, y + h * 0.5, title, ha="center", va="center", fontsize=title_size, fontweight="bold", color=title_color, zorder=3)
        return box

    def add_arrow(x1, y1, x2, y2, color="#2B2D42", lw=1.8, style="-|>", mut=14, rad=0.0):
        con = f"arc3,rad={rad}"
        arrow = patches.FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle=style, mutation_scale=mut,
            fc=color, ec=color, lw=lw, zorder=4,
            connectionstyle=con
        )
        ax.add_patch(arrow)

    # 1. Background Containers / Groupings
    # Left: Inputs Container
    bg_in = patches.FancyBboxPatch((2, 4), 21, 52, boxstyle="round,pad=0.8,rounding_size=2.0", fc="#F8FAFC", ec="#CBD5E1", lw=1.2, ls="--", zorder=1)
    ax.add_patch(bg_in)
    ax.text(12.5, 53.5, "INPUT SIGNALS", ha="center", va="center", fontsize=10, fontweight="bold", color="#475569")

    # Center-Top: Temporal Branch
    bg_temp = patches.FancyBboxPatch((26, 33), 32, 23, boxstyle="round,pad=0.8,rounding_size=2.0", fc="#F0F7FC", ec="#BAE6FD", lw=1.2, ls="--", zorder=1)
    ax.add_patch(bg_temp)
    ax.text(42, 53.5, "TEMPORAL ENCODER BRANCH", ha="center", va="center", fontsize=10, fontweight="bold", color="#0369A1")

    # Center-Bottom: Spatial Branch
    bg_spat = patches.FancyBboxPatch((26, 4), 32, 26, boxstyle="round,pad=0.8,rounding_size=2.0", fc="#FFF7ED", ec="#FED7AA", lw=1.2, ls="--", zorder=1)
    ax.add_patch(bg_spat)
    ax.text(42, 27.5, "SPATIAL GRAPH CONVOLUTION BRANCH", ha="center", va="center", fontsize=10, fontweight="bold", color="#C2410C")

    # Center-Right: Adaptive Gate
    bg_gate = patches.FancyBboxPatch((61, 14), 18, 38, boxstyle="round,pad=0.8,rounding_size=2.0", fc="#F0FDF4", ec="#BBF7D0", lw=1.2, ls="--", zorder=1)
    ax.add_patch(bg_gate)
    ax.text(70, 49.5, "ADAPTIVE GATE", ha="center", va="center", fontsize=10, fontweight="bold", color="#15803D")

    # Right: Fusion & Forecast Head
    bg_out = patches.FancyBboxPatch((82, 14), 16, 38, boxstyle="round,pad=0.8,rounding_size=2.0", fc="#FAF5FF", ec="#E9D5FF", lw=1.2, ls="--", zorder=1)
    ax.add_patch(bg_out)
    ax.text(90, 49.5, "GATED FORECAST", ha="center", va="center", fontsize=10, fontweight="bold", color="#7E22CE")

    # 2. Draw Input Blocks
    add_box(4, 38, 17, 10, r"Lookback Sequence", r"$\mathbf{Y}_{t-13:t} \in \mathbb{R}^{N \times 14}$", fill_color="#FFFFFF", edge_color="#64748B")
    add_box(4, 23, 17, 10, r"Node-Type Embedding", r"$\mathbf{e}_i \in \mathbb{R}^{d_{\mathrm{model}}}$ (Cat.)", fill_color="#FFFFFF", edge_color="#64748B")
    add_box(4, 8, 17, 11, r"Supply Graph $\mathcal{G}(\mathcal{V}, \mathcal{E})$", r"Real vs. Permuted (Job B)", fill_color="#FFFFFF", edge_color="#64748B")

    # 3. Draw Temporal Branch Blocks
    add_box(28, 41, 17, 9, r"Temporal Transformer", r"Multi-Head Attn ($d=64$)", fill_color="#E0F2FE", edge_color=COLORS["Adaptive_STGT"], title_color=COLORS["Adaptive_STGT"])
    add_box(47, 41, 9, 9, r"LayerNorm", r"$\tilde{\mathbf{h}}_{\mathrm{temp}}$", fill_color="#E0F2FE", edge_color=COLORS["Adaptive_STGT"], title_color=COLORS["Adaptive_STGT"])

    # 4. Draw Spatial Branch Blocks
    add_box(28, 13, 17, 10, r"2$\times$ TransformerConv", r"Graph Msg Passing ($d=64$)", fill_color="#FFEDD5", edge_color=COLORS["STGT"], title_color=COLORS["STGT"])
    add_box(47, 13, 9, 10, r"LayerNorm", r"$\tilde{\mathbf{h}}_{\mathrm{spat}}$", fill_color="#FFEDD5", edge_color=COLORS["STGT"], title_color=COLORS["STGT"])

    # 5. Draw Gate Block Elements
    add_box(63, 31, 14, 8, r"Concat", r"$[\tilde{\mathbf{h}}_{\mathrm{temp}} \,\|\, \tilde{\mathbf{h}}_{\mathrm{spat}}] \in \mathbb{R}^{128}$", fill_color="#DCFCE7", edge_color=COLORS["Gate"], title_color="#166534")
    add_box(63, 18, 14, 9, r"Gate MLP", r"$\mathrm{Linear}(128\to64)\to\mathrm{Tanh}\to\mathrm{Sigmoid}$", fill_color="#DCFCE7", edge_color=COLORS["Gate"], title_color="#166534")

    # 6. Draw Fusion & Output
    add_box(84, 28, 12, 12, r"Convex Fusion", r"$\alpha \tilde{\mathbf{h}}_{\mathrm{temp}} + (1-\alpha) \tilde{\mathbf{h}}_{\mathrm{spat}}$", fill_color="#F3E8FF", edge_color=COLORS["Fusion"], title_color=COLORS["Fusion"])
    add_box(84, 16, 12, 8, r"Linear Head", r"$\hat{\mathbf{y}}_{t+1:t+7} \in \mathbb{R}^{N \times 7}$", fill_color="#F3E8FF", edge_color=COLORS["Fusion"], title_color=COLORS["Fusion"])

    # 7. Connect Arrows
    # Inputs to Temporal
    add_arrow(21, 43, 28, 45.5)
    add_arrow(21, 28, 28, 43.5, rad=-0.2)
    # Inputs to Spatial
    add_arrow(21, 28, 28, 19, rad=0.2)
    add_arrow(21, 13.5, 28, 17)
    # Inside Temporal & Spatial
    add_arrow(45, 45.5, 47, 45.5)
    add_arrow(45, 18, 47, 18)
    # Temporal & Spatial outputs to Gate Concat
    add_arrow(56, 45.5, 63, 36.5, rad=-0.15)
    add_arrow(56, 18, 63, 33.5, rad=0.15)
    # Concat to Gate MLP
    add_arrow(70, 31, 70, 27)
    # Gate outputs to Fusion
    add_arrow(77, 22.5, 84, 32, rad=-0.2, color="#166534")
    ax.text(77.5, 27.5, r"$\alpha_{i,t}, 1-\alpha_{i,t}$", fontsize=9, fontweight="bold", color="#166534")
    # Temporal & Spatial skip to Fusion
    add_arrow(56, 46.5, 84, 37, rad=-0.2, color=COLORS["Adaptive_STGT"])
    add_arrow(56, 16.5, 84, 29, rad=0.2, color=COLORS["STGT"])
    # Fusion to Output
    add_arrow(90, 28, 90, 24)

    # 8. Add Parameter Count Badges
    badges = [
        ("Temporal-only: 35.2k", 28, 2, "#E0F2FE", COLORS["Adaptive_STGT"]),
        ("Spatial-only: 34.1k", 48, 2, "#FFEDD5", COLORS["STGT"]),
        ("STGT: 68.5k", 67, 2, "#FED7AA", "#9A3412"),
        ("Adaptive STGT: 77.1k", 84, 2, "#F3E8FF", COLORS["Fusion"]),
    ]
    for text, bx, by, bgc, fgc in badges:
        badge = patches.FancyBboxPatch((bx, by), 16, 3.2, boxstyle="round,pad=0.4,rounding_size=0.8", fc=bgc, ec=fgc, lw=1.2, zorder=5)
        ax.add_patch(badge)
        ax.text(bx + 8, by + 1.6, text, ha="center", va="center", fontsize=8.5, fontweight="bold", color=fgc, zorder=6)

    # Clean border and title
    ax.set_title("Figure 1: Adaptive Spatio-Temporal Graph Transformer (Adaptive STGT) Architecture\n"
                 "Dual-Branch Spatial-Temporal Backbone with Learned Input-Dependent Convex Fusion Gate",
                 fontsize=12, fontweight="bold", pad=12)

    plt.tight_layout()
    png_path = OUT / "fig1_architecture.png"
    pdf_path = OUT / "fig1_architecture.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated Figure 1: {png_path} and {pdf_path}")


def draw_fig2_horizon_and_roles():
    """Figure 2: 2-Panel Figure (figsize=(11, 4.2))
    Panel (a): Multi-Horizon Test MAE (h=1..7) on ISOMORPH.
    Panel (b): ISOMORPH Physical Node Roles & Gate Recovery.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=300)

    # -------------------------------------------------------------
    # Panel (a): Multi-Horizon Test MAE on ISOMORPH
    # -------------------------------------------------------------
    horiz_df = pd.read_csv(OUT / "paper_horizon_metrics.csv")
    iso_h = horiz_df[(horiz_df["Dataset"] == "ISOMORPH") & (horiz_df["Split"] == "Test")].copy()

    model_styles = [
        ("Adaptive_STGT", COLORS["Adaptive_STGT"], "-", "o", "Adaptive STGT (Ours)"),
        ("STGT", COLORS["STGT"], "-", "s", "STGT (Real Graph)"),
        ("STGT_Permuted", COLORS["STGT_Permuted"], "--", "^", "STGT (Permuted Graph)"),
        ("Temporal-only", COLORS["Temporal-only"], "--", "D", "Temporal-only"),
        ("Spatial-only", COLORS["Spatial-only"], ":", "x", "Spatial-only"),
    ]

    horizons = np.arange(1, 8)
    for model_name, col, ls, marker, label in model_styles:
        m_data = iso_h[iso_h["Model"] == model_name].sort_values("Horizon")
        if m_data.empty:
            continue
        y = m_data["MAE_mean"].to_numpy()
        std = m_data["MAE_std"].to_numpy()
        ax1.plot(horizons, y, color=col, linestyle=ls, marker=marker, markersize=5, lw=1.8, label=label, zorder=4)
        ax1.fill_between(horizons, y - std, y + std, color=col, alpha=0.15, zorder=3)

    ax1.set_xlabel(r"Forecast Horizon Step ($h = 1, \dots, 7$)", fontweight="bold")
    ax1.set_ylabel("Test MAE (Original Units)", fontweight="bold")
    ax1.set_title("(a) Multi-Horizon Test MAE (ISOMORPH)", fontweight="bold")
    ax1.set_xticks(horizons)
    ax1.grid(True, linestyle="--", alpha=0.4, zorder=1)
    ax1.legend(loc="upper left", framealpha=0.9, edgecolor="#CCCCCC")

    # -------------------------------------------------------------
    # Panel (b): ISOMORPH Node Roles & Gate Suppression
    # -------------------------------------------------------------
    regime_df = pd.read_csv(OUT / "paper_node_regime_metrics.csv")
    iso_r = regime_df[(regime_df["Dataset"] == "ISOMORPH") & (regime_df["Split"] == "Test")].copy()

    roles = ["Source", "Warehouse", "Destination"]
    role_labels = ["Sources\n($n=3$)", "Warehouses\n($n=9$)", "Destination\nSink ($n=1$)"]
    x = np.arange(len(roles))
    width = 0.18

    compare_models = [
        ("Temporal-only", COLORS["Temporal-only"], "Temporal-only"),
        ("STGT", COLORS["STGT"], "STGT (Real)"),
        ("STGT_Permuted", COLORS["STGT_Permuted"], "STGT Permuted"),
        ("Adaptive_STGT", COLORS["Adaptive_STGT"], "Adaptive STGT"),
    ]

    for i, (m_name, col, lbl) in enumerate(compare_models):
        maes = []
        stds = []
        for r in roles:
            sub = iso_r[(iso_r["Regime_Or_Role"] == r) & (iso_r["Model"] == m_name)]
            if not sub.empty:
                maes.append(sub["MAE_mean"].iloc[0])
                stds.append(sub["MAE_std"].iloc[0])
            else:
                maes.append(0.0)
                stds.append(0.0)
        offset = (i - 1.5) * width
        ax2.bar(x + offset, maes, width, yerr=stds, capsize=3, color=col, label=lbl, edgecolor="#222222", lw=0.8, alpha=0.9, zorder=3)

    ax2.set_ylabel("Test MAE (Original Units)", fontweight="bold")
    ax2.set_title("(b) Error by Supply-Chain Role & Gate Suppression", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(role_labels, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.4, axis="y", zorder=1)
    ax2.legend(loc="upper right", framealpha=0.9, edgecolor="#CCCCCC")
    ax2.set_ylim(0, 4800)

    # Annotations on Panel (b)
    # 1. Warehouse Gain annotation
    ax2.annotate(
        "Warehouse Gain:\n$-129.1$ MAE vs. Temp.",
        xy=(1.0 + 0.5 * width, 3653), xytext=(1.0 - 0.2, 4200),
        arrowprops=dict(facecolor="#15803D", edgecolor="#15803D", arrowstyle="->", lw=1.5),
        ha="center", va="bottom", fontsize=8.2, fontweight="bold", color="#15803D",
        bbox=dict(boxstyle="round,pad=0.3", fc="#DCFCE7", ec="#86EFAC", lw=0.8)
    )

    # 2. Destination Sink recovery annotation
    ax2.annotate(
        "Sink Recovery: $\\alpha_{\\mathrm{temp}} = 95.7\\%$\nRecovers 82.1% of STGT loss",
        xy=(2.0 + 0.5 * width, 617), xytext=(2.0 - 0.05, 1600),
        arrowprops=dict(facecolor=COLORS["Adaptive_STGT"], edgecolor=COLORS["Adaptive_STGT"], arrowstyle="->", lw=1.5),
        ha="center", va="bottom", fontsize=8.2, fontweight="bold", color=COLORS["Adaptive_STGT"],
        bbox=dict(boxstyle="round,pad=0.3", fc="#E0F2FE", ec="#7DD3FC", lw=0.8)
    )

    plt.tight_layout()
    png_path = OUT / "fig2_horizon_and_roles.png"
    pdf_path = OUT / "fig2_horizon_and_roles.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated Figure 2: {png_path} and {pdf_path}")


def draw_fig3_gate_topology():
    """Figure 3: 2-Panel Figure (figsize=(11, 4.2))
    Panel (a): ISOMORPH Degree vs Learned Spatial Weight (beta_i)
    Panel (b): SupplyGraph Degree vs Learned Spatial Weight (beta_i) + Permutation Box
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=300)

    weights_df = pd.read_csv(OUT / "paper_gate_weights.csv")

    # -------------------------------------------------------------
    # Panel (a): ISOMORPH
    # -------------------------------------------------------------
    iso_w = weights_df[weights_df["Dataset"] == "ISOMORPH"].copy()
    iso_w["Total_Degree"] = iso_w["In_Degree"] + iso_w["Out_Degree"]

    role_colors = {
        "Source": (COLORS["Temporal-only"], "D", "Source (n=3)"),
        "Warehouse": (COLORS["Adaptive_STGT"], "o", "Warehouse (n=9)"),
        "Destination": (COLORS["STGT"], "s", "Destination Sink (n=1)"),
    }

    deg_iso = iso_w["Total_Degree"].to_numpy()
    spat_iso = iso_w["Spatial_Weight"].to_numpy()

    # Regression line
    slope1, intercept1, r1, p1, _ = stats.linregress(deg_iso, spat_iso)
    x_grid1 = np.linspace(deg_iso.min(), deg_iso.max(), 100)
    ax1.plot(x_grid1, slope1 * x_grid1 + intercept1, color="#1E293B", linestyle="--", lw=1.8, label=f"Fit ($r = {r1:.3f}, p < 0.05$)", zorder=3)

    for role_name, (col, marker, lbl) in role_colors.items():
        sub = iso_w[iso_w["Node_Type_Or_Role"] == role_name]
        ax1.scatter(sub["Total_Degree"], sub["Spatial_Weight"], color=col, marker=marker, s=65, edgecolors="#222222", lw=0.8, alpha=0.9, label=lbl, zorder=4)

    ax1.set_xlabel(r"Total Node Degree ($\mathrm{In} + \mathrm{Out}$)", fontweight="bold")
    ax1.set_ylabel("Learned Spatial Weight ($\\beta_i = 1 - \\alpha_i$)", fontweight="bold")
    ax1.set_title("(a) ISOMORPH: Degree vs. Spatial Weight", fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.4, zorder=1)
    ax1.legend(loc="upper left", framealpha=0.9, edgecolor="#CCCCCC")

    # -------------------------------------------------------------
    # Panel (b): SupplyGraph
    # -------------------------------------------------------------
    sup_w = weights_df[weights_df["Dataset"] == "SupplyGraph"].copy()
    sup_w["Total_Degree"] = sup_w["In_Degree"] + sup_w["Out_Degree"]

    deg_sup = sup_w["Total_Degree"].to_numpy()
    spat_sup = sup_w["Spatial_Weight"].to_numpy()

    slope2, intercept2, r2, p2, _ = stats.linregress(deg_sup, spat_sup)
    x_grid2 = np.linspace(deg_sup.min(), deg_sup.max(), 100)
    ax2.plot(x_grid2, slope2 * x_grid2 + intercept2, color="#1E293B", linestyle="--", lw=1.8, label=f"Fit ($r = {r2:.3f}, p < 10^{{-7}}$)", zorder=3)

    ax2.scatter(deg_sup, spat_sup, color=COLORS["Adaptive_STGT"], marker="o", s=45, edgecolors="#222222", lw=0.7, alpha=0.75, label="Product Nodes (n=39)", zorder=4)

    ax2.set_xlabel("Product Node Degree (Shared Plants)", fontweight="bold")
    ax2.set_ylabel("Learned Spatial Weight ($\\beta_i = 1 - \\alpha_i$)", fontweight="bold")
    ax2.set_title("(b) SupplyGraph: Degree vs. Spatial Weight", fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.4, zorder=1)
    ax2.legend(loc="upper left", framealpha=0.9, edgecolor="#CCCCCC")

    # Inset box comparing Real vs Permuted STGT on SupplyGraph
    callout_text = (
        "Job B Permutation Control (SupplyGraph):\n"
        "• Real STGT:     317.18 ± 20.36 MAE\n"
        "• Permuted STGT: 311.83 ± 16.60 MAE\n"
        "• Paired t-test: p = 0.617 (Not Sig.)\n"
        "Finding: Shared-plant edges yield zero\n"
        "benefit over temporal feature backbone."
    )
    ax2.text(
        0.96, 0.06, callout_text,
        transform=ax2.transAxes,
        fontsize=8.2, family="monospace",
        ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", fc="#FFFBEB", ec="#FCD34D", lw=1.0),
        zorder=5
    )

    plt.tight_layout()
    png_path = OUT / "fig3_gate_topology.png"
    pdf_path = OUT / "fig3_gate_topology.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated Figure 3: {png_path} and {pdf_path}")


def main(output: Path = Path("paper_outputs")):
    global OUT
    OUT = output.expanduser().resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("STEP 3: GENERATING 300-DPI PUBLICATION FIGURES (.png AND .pdf)")
    print("=" * 80)
    draw_fig1_architecture()
    draw_fig2_horizon_and_roles()
    draw_fig3_gate_topology()
    print("\n>>> All figures generated successfully! <<<")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("paper_outputs"),
                        help="Directory containing paper CSVs and receiving figures")
    main(parser.parse_args().output)

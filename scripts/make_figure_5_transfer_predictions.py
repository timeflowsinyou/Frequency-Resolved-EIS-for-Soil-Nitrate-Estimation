"""Create Figure 5: predicted versus nominal nitrate-N for historical transfer soils."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/reference/historical_transfer_predictions.csv"
SOILS = ["Z05_03", "Z10_02", "Z15_04"]
COLOR = "#18713C"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "figures/figure_5_transfer_predictions.png")
    args = parser.parse_args()
    data = pd.read_csv(DATA)
    identity = ["network_seed", "specimen_id", "concentration_stage", "technical_repeat"]
    if len(data) != 3600 or data.duplicated(identity).any():
        raise RuntimeError("Transfer prediction coverage is incomplete or non-unique")
    if set(data["specimen_id"]) != set(SOILS) or data["network_seed"].nunique() != 10:
        raise RuntimeError("Unexpected transfer population or seed coverage")

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.5), sharex=True, sharey=True, layout="constrained")
    limits = (-0.35, 14.5)
    for panel, (ax, soil) in enumerate(zip(axes, SOILS)):
        subset = data.loc[data["specimen_id"] == soil].copy()
        jitter_code = subset["technical_repeat"] + 23 * subset["network_seed"]
        jitter = ((jitter_code % 19) - 9) / 9 * 0.075
        ax.scatter(
            subset["nominal_nitrate_n_mg_l"] + jitter,
            subset["predicted_nitrate_n_mg_l"],
            s=12,
            color=COLOR,
            alpha=0.075,
            edgecolors="none",
            rasterized=True,
        )

        seed_stage = (
            subset.groupby(["network_seed", "nominal_nitrate_n_mg_l"], as_index=False)
            ["predicted_nitrate_n_mg_l"].mean()
        )
        summary = seed_stage.groupby("nominal_nitrate_n_mg_l")["predicted_nitrate_n_mg_l"].agg(
            mean="mean", standard_deviation="std"
        )
        ax.errorbar(
            summary.index,
            summary["mean"],
            yerr=summary["standard_deviation"],
            fmt="o",
            color=COLOR,
            markeredgecolor="white",
            markeredgewidth=0.7,
            markersize=6.5,
            capsize=3,
            linewidth=1.25,
            zorder=3,
        )
        ax.plot(limits, limits, linestyle="--", color="#555555", linewidth=1.1, zorder=0)

        error = subset["signed_error_mg_l"].to_numpy(float)
        rmse = float(np.sqrt(np.mean(error ** 2)))
        bias = float(np.mean(error))
        ax.text(
            0.04,
            0.96,
            f"RMSE = {rmse:.2f} mg/L\nBias = {bias:+.2f} mg/L",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.88, "pad": 4},
        )
        ax.set_title(f"{chr(65 + panel)}  {soil}", loc="left", fontweight="bold")
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Nominal nitrate-N (mg/L)")
    axes[0].set_ylabel("Predicted nitrate-N (mg/L)")
    fig.text(
        0.5,
        -0.02,
        "Points: technical-measurement predictions across ten network seeds; "
        "markers: stage mean ± between-seed sample SD.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()

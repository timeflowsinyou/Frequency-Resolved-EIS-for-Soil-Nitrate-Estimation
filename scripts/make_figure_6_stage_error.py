"""Create Figure 6: concentration-stage signed error and endpoint bias."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
POPULATIONS = [
    (
        ROOT / "results/reference/grouped_cross_validation_predictions.csv",
        "A  19-soil grouped cross-validation",
        22800,
    ),
    (
        ROOT / "results/reference/historical_transfer_predictions.csv",
        "B  Historical three-soil transfer",
        3600,
    ),
]
COLOR = "#18713C"


def summarize(path: Path, expected_rows: int) -> pd.DataFrame:
    data = pd.read_csv(path)
    identity = ["network_seed", "specimen_id", "concentration_stage", "technical_repeat"]
    if len(data) != expected_rows or data.duplicated(identity).any():
        raise RuntimeError(f"Incomplete or non-unique prediction coverage in {path}")
    if data["network_seed"].nunique() != 10 or set(data["concentration_stage"]) != set(range(1, 7)):
        raise RuntimeError(f"Unexpected seed or concentration-stage coverage in {path}")
    seed_stage = (
        data.groupby(["network_seed", "concentration_stage"], as_index=False)["signed_error_mg_l"]
        .mean()
    )
    return seed_stage.groupby("concentration_stage")["signed_error_mg_l"].agg(
        mean="mean", standard_deviation="std"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "figures/figure_6_stage_error.png")
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.5), sharex=True, sharey=True, layout="constrained")
    for ax, (path, title, expected_rows) in zip(axes, POPULATIONS):
        summary = summarize(path, expected_rows)
        ax.axhline(0, color="#555555", linestyle="--", linewidth=1.1)
        ax.errorbar(
            summary.index,
            summary["mean"],
            yerr=summary["standard_deviation"],
            color=COLOR,
            marker="o",
            markerfacecolor=COLOR,
            markeredgecolor="white",
            markeredgewidth=0.8,
            markersize=7,
            linewidth=1.8,
            capsize=5,
        )
        ax.fill_between(summary.index, 0, summary["mean"], color=COLOR, alpha=0.10)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xticks(range(1, 7))
        ax.set_xlim(0.7, 6.3)
        ax.set_ylim(-2.5, 2.5)
        ax.set_xlabel("Concentration stage")
        ax.grid(axis="y", alpha=0.22)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(0.98, 0.95, "Overprediction", transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color="#555555")
        ax.text(0.98, 0.06, "Underprediction", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9, color="#555555")
    axes[0].set_ylabel("Signed error (mg/L nitrate-N)\npredicted − nominal")
    fig.text(
        0.5,
        -0.02,
        "Markers show the ten-seed mean; error bars show the between-seed sample SD. "
        "Historical nominal concentrations vary with preparation protocol.",
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

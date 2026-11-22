"""Create Figure 4: multi-seed stability in grouped CV and historical transfer."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/reference/multiseed_stability.csv"
MODELS = ["Admittance CNN", "Spectral level/shape CNN", "Fixed equal-weight ensemble"]
COLORS = {"Admittance CNN": "#4C78A8", "Spectral level/shape CNN": "#59A14F",
          "Fixed equal-weight ensemble": "#18713C"}
PANELS = [
    ("19-soil grouped cross-validation", "A  19-soil grouped cross-validation"),
    ("Historical three-soil transfer", "B  Historical three-soil transfer"),
]


def plot_population(
    ax: plt.Axes,
    data: pd.DataFrame,
    population: str,
    title: str,
    upper_limit: float,
) -> None:
    subset = data.loc[data["evaluation_population"] == population]
    pivot = subset.pivot(index="network_seed", columns="model_name", values="rmse_mg_l")
    pivot = pivot.loc[:, MODELS].sort_index()
    means = pivot.mean(axis=0).to_numpy(float)
    standard_deviations = pivot.std(axis=0, ddof=1).to_numpy(float)
    x = range(len(MODELS))
    bars = ax.bar(
        x,
        means,
        yerr=standard_deviations,
        color=[COLORS[model] for model in MODELS],
        edgecolor="#333333",
        linewidth=0.8,
        capsize=6,
        error_kw={"elinewidth": 1.5, "capthick": 1.5},
        width=0.66,
    )
    for bar, mean, standard_deviation in zip(bars, means, standard_deviations):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + standard_deviation + upper_limit * 0.035,
            f"{mean:.2f} ± {standard_deviation:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xticks(x, ["Admittance", "Spectral\nlevel/shape", "Fixed\nensemble"])
    ax.set_ylabel("RMSE (mg/L nitrate-N)")
    ax.set_ylim(0, upper_limit)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "figures/figure_4_multiseed_stability.png")
    args = parser.parse_args()
    data = pd.read_csv(DATA)
    expected = set(MODELS)
    for population, _ in PANELS:
        subset = data.loc[data["evaluation_population"] == population]
        if set(subset["model_name"]) != expected or subset["network_seed"].nunique() != 10:
            raise RuntimeError(f"Incomplete ten-seed results for {population}")
        if subset.duplicated(["network_seed", "model_name"]).any():
            raise RuntimeError(f"Duplicate seed/model rows for {population}")
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.0), layout="constrained")
    upper_limits = [2.15, 2.20]
    for ax, (population, title), upper_limit in zip(axes, PANELS, upper_limits):
        plot_population(ax, data, population, title, upper_limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()

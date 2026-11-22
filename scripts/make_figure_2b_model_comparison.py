"""Create Figure 2B: reference grouped-CV model comparison."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=ROOT/"figures/figure_2b_model_comparison.png")
    args=parser.parse_args()
    results=pd.read_csv(ROOT/"results/reference/reference_seed_model_comparison.csv")
    labels=["Direct five-channel CNN","Direct EIS–physical fusion CNN","Admittance CNN","Spectral level/shape CNN","Fixed representation ensemble","Matched replay","Physics-guided augmentation"]
    order=["Direct five-channel CNN","Direct EIS–physical fusion CNN","Admittance-based CNN","Spectral level/shape CNN","Fixed representation ensemble","Admittance-based CNN with matched replay","Admittance-based CNN with physics-guided augmentation"]
    data=results.set_index("model_description").loc[order].reset_index()
    colors=["#9ecae1","#6baed6","#4c78a8","#59a14f","#176d3a","#f2a65a","#e17c5b"]

    fig,ax=plt.subplots(figsize=(8.6,6.6),layout="constrained")
    bars=ax.barh(range(len(labels)),data.rmse_mg_l,color=colors)
    ax.set_yticks(range(len(labels)),labels); ax.invert_yaxis()
    ax.set_xlabel("RMSE (mg/L; lower is better)")
    ax.set_title("Reference grouped-CV model comparison",loc="left",fontweight="bold")
    ax.grid(axis="x",alpha=.25)
    for bar,r2,rmse in zip(bars,data.coefficient_of_determination,data.rmse_mg_l):
        ax.text(rmse+.025,bar.get_y()+bar.get_height()/2,f"RMSE {rmse:.3f}   $R^2$ {r2:.3f}",va="center",fontsize=9)
    ax.set_xlim(0,max(data.rmse_mg_l)+.48)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=300,bbox_inches="tight")
    plt.close(fig)
    print(args.output)


if __name__=="__main__":
    main()

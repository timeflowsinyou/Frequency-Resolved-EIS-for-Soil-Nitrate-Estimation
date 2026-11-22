"""Create Figure 3B: shared late-fusion CNN architecture."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT=Path(__file__).resolve().parents[1]


def box(ax,xy,width,height,text,color,fontsize=8.7,edge="#333333"):
    patch=FancyBboxPatch(xy,width,height,boxstyle="round,pad=0.025",facecolor=color,edgecolor=edge,linewidth=1.15)
    ax.add_patch(patch); ax.text(xy[0]+width/2,xy[1]+height/2,text,ha="center",va="center",fontsize=fontsize)


def arrow(ax,start,end):
    ax.annotate("",end,start,arrowprops={"arrowstyle":"->","linewidth":1.2,"color":"#444444"})


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,default=ROOT/"figures/figure_3b_cnn_architecture.png"); args=parser.parse_args()
    fig,ax=plt.subplots(figsize=(9.2,6.2),layout="constrained"); ax.set_xlim(0,12); ax.set_ylim(0,10); ax.axis("off"); ax.set_title("Shared late-fusion CNN architecture",loc="left",fontweight="bold")
    ax.text(6,9.55,"Identical convolutional and fusion layers for both retained representations",ha="center",fontsize=8.5,color="#444444")
    box(ax,(.2,7.35),2.3,1.15,"Representation-specific EIS\n$C \\times 201$ frequencies","#d9eaf7")
    box(ax,(3.05,6.95),3.65,1.95,"Conv1D 16, $k=5$ + BN + ReLU + pool\nConv1D 32, $k=5$ + BN + ReLU + pool\nConv1D 64, $k=3$ + BN + ReLU","#b8d8ee",8.0)
    box(ax,(7.35,7.35),2.85,1.15,"Flatten 3,200 → dense 32\nEIS embedding","#b8d8ee")
    box(ax,(.2,4.35),2.3,1.15,"Scalar conditions\n$P$ physical features","#f5dfb8")
    box(ax,(3.05,4.35),3.65,1.15,"Dense 8 + ReLU\nphysical embedding","#f5dfb8")
    box(ax,(8.15,4.05),3.2,1.7,"Concatenate 40\nDropout 0.2\nLinear regression output","#cde8cf")
    arrow(ax,(2.5,7.92),(3.05,7.92)); arrow(ax,(6.7,7.92),(7.35,7.92)); arrow(ax,(2.5,4.92),(3.05,4.92)); arrow(ax,(8.78,7.35),(9.42,5.75)); arrow(ax,(6.7,4.92),(8.15,4.92))
    box(ax,(.55,1.15),3.25,1.15,"Admittance model\n$G(f)$, $B(f)$ + $\\rho_c$, $w_c$","#b8d8ee",8.2)
    ax.text(4.08,1.72,"+",ha="center",va="center",fontsize=14,fontweight="bold")
    box(ax,(4.35,1.15),4.25,1.15,"Level/shape model\n$S(f)$, $\\sin\\phi$, $\\cos\\phi$ + $L$, $\\rho_c$, $w_c$","#cde8cf",8.0)
    box(ax,(9.15,1.15),2.6,1.15,"Fixed arithmetic mean\n$(\\hat y_A+\\hat y_S)/2$","#b7dfc2",8.2,edge="#176d3a")
    arrow(ax,(8.6,1.72),(9.15,1.72))
    ax.text(6,.35,"The validation prediction is the predefined equal-weight mean of two independently trained models.",ha="center",fontsize=8.3,color="#444444")
    args.output.parent.mkdir(parents=True,exist_ok=True); fig.savefig(args.output,dpi=300,bbox_inches="tight"); plt.close(fig); print(args.output)


if __name__=="__main__": main()

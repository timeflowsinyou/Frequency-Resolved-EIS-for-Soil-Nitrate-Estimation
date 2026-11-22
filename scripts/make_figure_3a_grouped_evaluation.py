"""Create Figure 3A: specimen-grouped five-fold evaluation design."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT=Path(__file__).resolve().parents[1]


def box(ax,xy,width,height,text,color,fontsize=9):
    patch=FancyBboxPatch(xy,width,height,boxstyle="round,pad=0.025",facecolor=color,edgecolor="#333333",linewidth=1.15)
    ax.add_patch(patch); ax.text(xy[0]+width/2,xy[1]+height/2,text,ha="center",va="center",fontsize=fontsize)


def arrow(ax,start,end):
    ax.annotate("",end,start,arrowprops={"arrowstyle":"->","linewidth":1.15,"color":"#444444"})


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,default=ROOT/"figures/figure_3a_grouped_evaluation.png"); args=parser.parse_args()
    fig,ax=plt.subplots(figsize=(9.2,6.2),layout="constrained"); ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis("off"); ax.set_title("Specimen-grouped five-fold evaluation",loc="left",fontweight="bold")
    box(ax,(2.65,8.55),4.7,.85,"19 soil specimens  |  2,280 measured observations","#d9eaf7",9.5)
    box(ax,(1.45,7.05),7.1,.72,"Grouping rule: all 6 concentration stages and all 20 repeats from one soil remain together","#eeeeee",8.2)
    arrow(ax,(5,8.55),(5,7.77))
    counts=[(5,14,600),(4,15,480),(4,15,480),(2,17,240),(4,15,480)]; centers=[]
    for i,(valid,train,rows) in enumerate(counts):
        x=.2+i*1.96; centers.append(x+.8); box(ax,(x,4.5),1.6,1.75,f"Fold {i+1}\n{valid} validation soils\n{train} training soils\n{rows} OOF rows","#cde8cf",7.6); arrow(ax,(5,7.05),(x+.8,6.25))
    box(ax,(2.55,2.45),4.9,1.0,"Complete and unique out-of-fold predictions\n19 soils × 6 stages × 20 technical repeats","#f5dfb8",8.7)
    for center in centers: arrow(ax,(center,4.5),(5,3.45))
    box(ax,(.45,.35),4.25,1.0,"Training-fold preprocessing only\nnormalization never uses validation soils","#eeeeee",8.0)
    box(ax,(5.3,.35),4.25,1.0,"Validation uses measured observations only\nno generated or replayed validation samples","#eeeeee",8.0)
    args.output.parent.mkdir(parents=True,exist_ok=True); fig.savefig(args.output,dpi=300,bbox_inches="tight"); plt.close(fig); print(args.output)


if __name__=="__main__": main()

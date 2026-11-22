"""Create Figure 2A: EIS representations and physical-condition fusion."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT=Path(__file__).resolve().parents[1]


def box(ax,xy,width,height,text,color,fontsize=10):
    patch=FancyBboxPatch(xy,width,height,boxstyle="round,pad=0.025",facecolor=color,edgecolor="#333333",linewidth=1.2)
    ax.add_patch(patch)
    ax.text(xy[0]+width/2,xy[1]+height/2,text,ha="center",va="center",fontsize=fontsize)


def arrow(ax,start,end):
    ax.annotate("",end,start,arrowprops={"arrowstyle":"->","linewidth":1.4,"color":"#333333"})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=ROOT/"figures/figure_2a_representation_fusion.png")
    args=parser.parse_args()

    fig,ax=plt.subplots(figsize=(8.2,5.6),layout="constrained")
    ax.set_xlim(0,12); ax.set_ylim(0,10); ax.axis("off")
    ax.set_title("EIS representations and physical-condition fusion",loc="left",fontweight="bold")

    box(ax,(.3,4.1),2.25,1.8,"Measured EIS\n$|Z(f)|$, $\\phi(f)$\n201 frequencies","#d9eaf7")
    box(ax,(3.25,6.65),3.25,1.75,"Admittance representation\n$Y(f)=1/Z(f)$\n$G(f)=\\mathrm{Re}[Y(f)]$,  $B(f)=\\mathrm{Im}[Y(f)]$","#b8d8ee",9.2)
    box(ax,(3.25,1.55),3.25,2.0,"Spectral level/shape\n$L=\\langle\\log|Z(f)|\\rangle_f$\n$S(f)=\\log|Z(f)|-L$\n$\\sin\\phi(f)$,  $\\cos\\phi(f)$","#cde8cf",8.8)
    box(ax,(4.0,4.25),2.35,1.45,"Physical conditions\n$\\rho_{c}$,  $w_{c}$","#f5dfb8")
    box(ax,(8.0,6.65),3.35,1.75,"Admittance CNN\nEIS branch + physical branch\nfeature fusion","#b8d8ee",9.5)
    box(ax,(8.0,1.55),3.35,2.0,"Level/shape CNN\nspectral branch: $S$, $\\sin\\phi$, $\\cos\\phi$\nscalar branch: $L$, $\\rho_c$, $w_c$","#cde8cf",8.8)

    arrow(ax,(2.55,5.0),(3.25,7.52)); arrow(ax,(2.55,5.0),(3.25,2.55))
    arrow(ax,(6.5,7.52),(8.0,7.52)); arrow(ax,(6.5,2.55),(8.0,2.55))
    arrow(ax,(6.35,4.98),(8.0,6.65)); arrow(ax,(6.35,4.98),(8.0,3.55))

    ax.text(5.18,.48,"$\\rho_c$: corrected density     $w_c$: corrected water content",ha="center",fontsize=8.5,color="#444444")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=300,bbox_inches="tight")
    plt.close(fig)
    print(args.output)


if __name__=="__main__":
    main()

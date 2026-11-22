"""Create Figure 1: full-spectrum response and controlled input-range evidence."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from frequency_eis.data import build_measured_dataset

def main():
    p=argparse.ArgumentParser(); p.add_argument("--specimen",default="M10_01"); p.add_argument("--output",type=Path,default=ROOT/"figures/figure_1_frequency_information.png"); args=p.parse_args(); data,_=build_measured_dataset(ROOT/"data/raw"); rows=data.metadata.specimen_id.eq(args.specimen).to_numpy(); meta=data.metadata[rows].reset_index(drop=True); magnitude=data.magnitude_ohm[rows]/1000.0; phase=data.phase_deg[rows]; evidence=pd.read_csv(ROOT/"results/reference/frequency_input_design.csv")
    fig=plt.figure(figsize=(13,5.4),layout="constrained"); outer=fig.add_gridspec(1,2,width_ratios=[1.35,1]); left=outer[0].subgridspec(2,1,hspace=.08); ax_mag=fig.add_subplot(left[0]); ax_phase=fig.add_subplot(left[1],sharex=ax_mag); colors=plt.cm.viridis(np.linspace(.08,.92,6))
    for stage,color in zip(range(1,7),colors):
        idx=meta.concentration_stage.eq(stage).to_numpy(); mean=magnitude[idx].mean(axis=0); sd=magnitude[idx].std(axis=0,ddof=1); ax_mag.plot(data.frequency_hz,mean,color=color,label=f"Stage {stage}"); ax_mag.fill_between(data.frequency_hz,mean-sd,mean+sd,color=color,alpha=.12); pm=phase[idx].mean(axis=0); ps=phase[idx].std(axis=0,ddof=1); ax_phase.plot(data.frequency_hz,pm,color=color); ax_phase.fill_between(data.frequency_hz,pm-ps,pm+ps,color=color,alpha=.12)
    ax_mag.set_xscale("log"); ax_mag.set_ylabel("Magnitude (kΩ)"); ax_mag.legend(ncol=3,fontsize=8); ax_mag.tick_params(labelbottom=False); ax_phase.set_xscale("log"); ax_phase.set_xlabel("Frequency (Hz)"); ax_phase.set_ylabel("Phase (degrees)"); ax_mag.set_title(f"A  Full-spectrum measured response: {args.specimen}",loc="left",fontweight="bold")
    ax=fig.add_subplot(outer[1]); designs=evidence.input_design.drop_duplicates().tolist(); representations=evidence.representation.drop_duplicates().tolist(); x=np.arange(len(designs)); width=.36
    for j,representation in enumerate(representations):
        values=[evidence[(evidence.representation==representation)&(evidence.input_design==design)].coefficient_of_determination.iloc[0] for design in designs]; ax.bar(x+(j-.5)*width,values,width,label=representation)
    ax.set_xticks(x,["201-point\ncomplete spectrum","98-point\ncomplete-range grid","98-point\nlow-frequency range"]); ax.set_ylabel("Coefficient of determination"); ax.set_ylim(0,1); ax.set_title("B  Representation-dependent input-range effect",loc="left",fontweight="bold"); ax.legend(loc="upper left",frameon=False); ax.grid(axis="y",alpha=.25)
    for container in ax.containers:
        ax.bar_label(container,fmt="%.3f",padding=3,fontsize=8)
    args.output.parent.mkdir(parents=True,exist_ok=True); fig.savefig(args.output,dpi=300,bbox_inches="tight"); plt.close(fig); print(args.output)

if __name__=="__main__": main()


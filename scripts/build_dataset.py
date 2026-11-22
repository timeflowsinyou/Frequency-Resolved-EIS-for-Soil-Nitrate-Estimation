"""Build the deterministic processed dataset from the public MATLAB files."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from frequency_eis.data import build_measured_dataset,save_processed_dataset

def main():
    p=argparse.ArgumentParser(); p.add_argument("--raw-dir",type=Path,default=ROOT/"data/raw"); p.add_argument("--output",type=Path,default=ROOT/"data/processed/measured_eis_dataset.npz"); args=p.parse_args(); dataset,corrections=build_measured_dataset(args.raw_dir); save_processed_dataset(dataset,corrections,args.output)
    summary={"output":str(args.output),"observations":len(dataset.metadata),"specimens":int(dataset.metadata.specimen_id.nunique()),"frequency_points":len(dataset.frequency_hz),"stages":int(dataset.metadata.concentration_stage.nunique()),"technical_repeats_per_stage":20,"confirmed_measurement_corrections":corrections}; print(json.dumps(summary,indent=2))

if __name__=="__main__": main()


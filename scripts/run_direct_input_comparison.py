"""Compare direct five-channel convolution with late physical fusion."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

from frequency_eis.data import build_measured_dataset,load_processed_dataset,save_processed_dataset
from frequency_eis.evaluation import build_metrics
from frequency_eis.representations import fit_direct_representation
from frequency_eis.training import select_device,train_measured_model

MODELS={
    "direct_five_channel_cnn":False,
    "direct_eis_physical_fusion_cnn":True,
}


def signature(payload):
    return hashlib.sha256(json.dumps(payload,sort_keys=True,default=str).encode()).hexdigest().upper()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,default=ROOT/"configs/reference_experiment.yaml")
    parser.add_argument("--device",choices=["auto","cpu","cuda"],default="auto")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"results/reproduced/direct_input_seed_42")
    args=parser.parse_args()
    config=yaml.safe_load(args.config.read_text())
    if int(config["cross_validation"]["split_seed"])!=42: raise RuntimeError("This fixed comparison requires split seed 42")
    seeds=list(config["training"]["network_seeds"])
    if 42 not in seeds: raise RuntimeError("Reference configuration does not include network seed 42")

    processed=ROOT/config["dataset"]["processed_path"]
    if processed.exists(): dataset=load_processed_dataset(processed)
    else:
        dataset,corrections=build_measured_dataset(ROOT/config["dataset"]["raw_directory"])
        save_processed_dataset(dataset,corrections,processed)
    folds=pd.read_csv(ROOT/config["cross_validation"]["fold_assignments"])
    args.output_dir.mkdir(parents=True,exist_ok=True)
    device=select_device(args.device)
    frames=[]; training_audit=[]

    for fold,assignment in folds.groupby("fold"):
        train_ids=assignment[assignment.role=="training"].specimen_id
        validation_ids=assignment[assignment.role=="validation"].specimen_id
        training_indices=np.flatnonzero(dataset.metadata.specimen_id.isin(train_ids))
        validation_indices=np.flatnonzero(dataset.metadata.specimen_id.isin(validation_ids))
        if set(dataset.metadata.iloc[training_indices].specimen_id)&set(dataset.metadata.iloc[validation_indices].specimen_id): raise RuntimeError("Specimen leakage")
        raw=np.concatenate([dataset.features[training_indices],dataset.features[validation_indices]],axis=0)
        train_count=len(training_indices)
        for model_name,late_fusion in MODELS.items():
            transformed=fit_direct_representation(raw,np.arange(train_count),late_fusion)
            run_signature=signature({"model":model_name,"fold":int(fold),"network_seed":42,"configuration":config})
            checkpoint=args.output_dir/f"checkpoints/{model_name}_fold_{int(fold)}.pt"
            result=train_measured_model(model_name,transformed.public_name,transformed.spectral[:train_count],transformed.physical[:train_count],dataset.labels_mg_l[training_indices],transformed.spectral[train_count:],transformed.physical[train_count:],42,int(fold),config["training"],checkpoint,run_signature,args.resume,device)
            frame=dataset.metadata.iloc[validation_indices].copy()
            frame["evaluation_scope"]="specimen_grouped_cross_validation"
            frame["network_seed"]=42
            frame["model_name"]=model_name
            frame["representation"]=transformed.public_name
            frame["fold"]=int(fold)
            frame["predicted_concentration_mg_l"]=result["predictions"]
            frame["residual_mg_l"]=frame.predicted_concentration_mg_l-frame.nominal_concentration_mg_l
            frames.append(frame)
            training_audit.append({"model_name":model_name,"fold":int(fold),"training_observations":len(training_indices),"validation_observations":len(validation_indices),"parameter_count":result["parameter_count"],"optimizer_updates":result["actual_optimizer_updates"],"training_seconds":result["training_seconds"],"checkpoint":str(checkpoint.relative_to(args.output_dir)),"normalization":transformed.parameters,"diagnostics":transformed.diagnostics})

    predictions=pd.concat(frames,ignore_index=True,sort=False)
    identity=["evaluation_scope","network_seed","model_name","representation","fold","specimen_id","source_filename","cohort","observation_index","concentration_stage","repeat_index","nominal_concentration_mg_l","predicted_concentration_mg_l","residual_mg_l"]
    predictions[identity].to_csv(args.output_dir/"predictions.csv",index=False)
    metrics=build_metrics(predictions)
    metrics.to_csv(args.output_dir/"metrics_summary.csv",index=False)
    audit={"status":"COMPLETE","scientific_question":"Effect of direct five-channel convolution versus late physical-condition fusion","population":"19 original-cohort soil specimens","excluded_specimens":config["dataset"]["excluded_specimens"],"network_seed":42,"models":list(MODELS),"prediction_rows":len(predictions),"device":str(device),"fold_training":training_audit}
    (args.output_dir/"run_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    pooled=metrics[(metrics.record_type=="seed_metric")&(metrics.scope_type=="pooled")]
    print(pooled[["model_name","coefficient_of_determination","rmse_mg_l","mae_mg_l","bias_mg_l"]].to_string(index=False))
    print(f"Completed: {args.output_dir}")


if __name__=="__main__":
    main()

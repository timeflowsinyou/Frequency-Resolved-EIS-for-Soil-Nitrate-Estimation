"""Reproduce the reference grouped-CV and historical-transfer evaluations."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from frequency_eis.data import build_measured_dataset,load_processed_dataset,save_processed_dataset
from frequency_eis.evaluation import build_metrics
from frequency_eis.representations import fit_admittance_representation,fit_level_shape_representation
from frequency_eis.training import select_device,train_measured_model

PUBLIC_MODELS={"admittance_cnn":"admittance","spectral_level_shape_cnn":"spectral_level_shape"}
ENSEMBLE_NAME="fixed_equal_weight_representation_ensemble"

def _hash(payload): return hashlib.sha256(json.dumps(payload,sort_keys=True,default=str).encode()).hexdigest().upper()
def _seeds(text,default): return [int(x) for x in text.split(",")] if text else list(default)
def _representation(features,training_indices,name): return fit_admittance_representation(features,training_indices) if name=="admittance" else fit_level_shape_representation(features,training_indices)

def _ensemble(frame):
    parts=[frame[frame.model_name==name].sort_values("observation_index") for name in PUBLIC_MODELS]
    if not np.array_equal(parts[0].observation_index,parts[1].observation_index): raise RuntimeError("Ensemble observation identities differ")
    x=parts[0].copy(); x["model_name"]=ENSEMBLE_NAME; x["representation"]="fixed_equal_weight_ensemble"; x["predicted_concentration_mg_l"]=.5*(parts[0].predicted_concentration_mg_l.to_numpy(float)+parts[1].predicted_concentration_mg_l.to_numpy(float)); x["residual_mg_l"]=x.predicted_concentration_mg_l-x.nominal_concentration_mg_l; return pd.concat([frame,x],ignore_index=True,sort=False)

def _train_scope(dataset,folds,seed,scope,config,output,resume,device):
    frames=[]; metadata=dataset.metadata
    if scope=="specimen_grouped_cross_validation": identities=[(fold,np.flatnonzero(metadata.specimen_id.isin(group[group.role=="training"].specimen_id)),np.flatnonzero(metadata.specimen_id.isin(group[group.role=="validation"].specimen_id))) for fold,group in folds.groupby("fold")]
    else:
        train=np.flatnonzero((metadata.cohort=="original")&~metadata.specimen_id.isin(config["dataset"]["excluded_specimens"])); validation=np.flatnonzero(metadata.specimen_id.isin(config["dataset"]["historical_transfer_specimens"])); identities=[(0,train,validation)]
    for fold,training_indices,validation_indices in identities:
        if set(metadata.iloc[training_indices].specimen_id)&set(metadata.iloc[validation_indices].specimen_id): raise RuntimeError("Specimen leakage")
        for model_name,representation_name in PUBLIC_MODELS.items():
            raw=np.concatenate([dataset.features[training_indices],dataset.features[validation_indices]],axis=0); transformed=_representation(raw,np.arange(len(training_indices)),representation_name); train_count=len(training_indices); signature=_hash({"scope":scope,"seed":seed,"fold":int(fold),"model":model_name,"config":config}); checkpoint=output/f"checkpoints/{scope}/seed_{seed}/{model_name}_fold_{fold}.pt"
            result=train_measured_model(model_name,representation_name,transformed.spectral[:train_count],transformed.physical[:train_count],dataset.labels_mg_l[training_indices],transformed.spectral[train_count:],transformed.physical[train_count:],seed,int(fold),config["training"],checkpoint,signature,resume,device)
            x=metadata.iloc[validation_indices].copy(); x["evaluation_scope"]=scope; x["network_seed"]=seed; x["model_name"]=model_name; x["representation"]=representation_name; x["fold"]=fold if scope=="specimen_grouped_cross_validation" else np.nan; x["predicted_concentration_mg_l"]=result["predictions"]; x["residual_mg_l"]=x.predicted_concentration_mg_l-x.nominal_concentration_mg_l; frames.append(x)
    return _ensemble(pd.concat(frames,ignore_index=True,sort=False))

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,default=ROOT/"configs/reference_experiment.yaml"); p.add_argument("--scope",choices=["grouped-cross-validation","historical-transfer","both"],default="both"); p.add_argument("--seeds"); p.add_argument("--device",choices=["auto","cpu","cuda"],default="auto"); p.add_argument("--resume",action="store_true"); p.add_argument("--output-dir",type=Path,default=ROOT/"results/reproduced"); args=p.parse_args(); config=yaml.safe_load(args.config.read_text()); processed=ROOT/config["dataset"]["processed_path"]
    if processed.exists(): dataset=load_processed_dataset(processed)
    else: dataset,corrections=build_measured_dataset(ROOT/config["dataset"]["raw_directory"]); save_processed_dataset(dataset,corrections,processed)
    folds=pd.read_csv(ROOT/config["cross_validation"]["fold_assignments"]); seeds=_seeds(args.seeds,config["training"]["network_seeds"]); device=select_device(args.device); args.output_dir.mkdir(parents=True,exist_ok=True); public_scopes={"grouped-cross-validation":"specimen_grouped_cross_validation","historical-transfer":"historical_transfer"}; scopes=["specimen_grouped_cross_validation","historical_transfer"] if args.scope=="both" else [public_scopes[args.scope]]; frames=[]
    for seed in seeds:
        for scope in scopes: frames.append(_train_scope(dataset,folds,seed,scope,config,args.output_dir,args.resume,device))
    predictions=pd.concat(frames,ignore_index=True,sort=False); identity=["evaluation_scope","network_seed","model_name","representation","fold","specimen_id","source_filename","cohort","observation_index","concentration_stage","repeat_index","nominal_concentration_mg_l","predicted_concentration_mg_l","residual_mg_l"]; predictions[identity].to_csv(args.output_dir/"predictions.csv",index=False); metrics=build_metrics(predictions); metrics.to_csv(args.output_dir/"metrics_summary.csv",index=False); audit={"status":"COMPLETE","scopes":scopes,"network_seeds":seeds,"models":list(PUBLIC_MODELS)+[ENSEMBLE_NAME],"prediction_rows":len(predictions),"device":str(device),"configuration":config}; (args.output_dir/"run_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8"); print(json.dumps(audit,indent=2))

if __name__=="__main__": main()


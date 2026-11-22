"""Load public MATLAB measurements and build the deterministic 201-point dataset."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
from .metadata import ALL_SPECIMENS,ORIGINAL_SPECIMENS,ORIGINAL_STAGE_MULTIPLIERS,TRANSFER_STAGE_MULTIPLIERS,N_FREQUENCIES,N_REPEATS,N_STAGES,corrected_conditions

SENTINEL=9.9e37
SENTINEL_FILE="V18%_01.mat"
SENTINEL_INDEX=(195,8,5)

@dataclass
class MeasuredEISDataset:
    features: np.ndarray
    magnitude_ohm: np.ndarray
    phase_deg: np.ndarray
    labels_mg_l: np.ndarray
    metadata: pd.DataFrame
    frequency_hz: np.ndarray

def _sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest().upper()

def load_frequency_grid(raw_dir:Path)->np.ndarray:
    values=np.asarray(loadmat(raw_dir/"f.mat")["f"]).reshape(-1)
    if values.size!=4020: raise ValueError(f"Expected 4020 frequency entries, found {values.size}")
    sweeps=values.reshape(20,201)
    if not np.array_equal(sweeps,np.repeat(sweeps[:1],20,axis=0)): raise ValueError("Frequency sweeps differ")
    frequency=sweeps[0].astype(np.float64)
    if not np.all(np.diff(frequency)>0): raise ValueError("Frequency grid is not increasing")
    return frequency

def _load_specimen(path:Path)->tuple[np.ndarray,np.ndarray]:
    data=loadmat(path,struct_as_record=False,squeeze_me=False)
    if "Imp" not in data: raise ValueError(f"Missing Imp structure: {path}")
    imp=data["Imp"][0,0]; magnitude=np.asarray(imp.Z,dtype=np.float64); phase=np.asarray(imp.phi,dtype=np.float64)
    if magnitude.shape!=(201,20,6) or phase.shape!=(201,20,6): raise ValueError(f"Unexpected measurement shape: {path}")
    return magnitude,phase

def _apply_confirmed_sentinel_correction(frequency,magnitude,phase,filename):
    if filename!=SENTINEL_FILE: return None
    fi,ri,si=SENTINEL_INDEX
    if magnitude[fi,ri,si]!=SENTINEL or phase[fi,ri,si]!=SENTINEL: raise RuntimeError("Confirmed sentinel identity changed")
    left,right=fi-1,fi+1; t=(np.log10(frequency[fi])-np.log10(frequency[left]))/(np.log10(frequency[right])-np.log10(frequency[left]))
    z=1000*10**((1-t)*np.log10(magnitude[left,ri,si]/1000)+t*np.log10(magnitude[right,ri,si]/1000)); p=np.rad2deg(np.unwrap(np.deg2rad([phase[left,ri,si],phase[right,ri,si]]))); p=((1-t)*p[0]+t*p[1]+180)%360-180
    record={"source_filename":filename,"frequency_index_zero_based":fi,"repeat_index_zero_based":ri,"stage_index_zero_based":si,"original_value":SENTINEL,"corrected_magnitude_ohm":float(z),"corrected_phase_deg":float(p),"interpolation_weight":float(t)}
    magnitude[fi,ri,si]=z; phase[fi,ri,si]=p
    return record

def build_measured_dataset(raw_dir:Path)->tuple[MeasuredEISDataset,list[dict]]:
    raw_dir=Path(raw_dir); frequency=load_frequency_grid(raw_dir); n=len(ALL_SPECIMENS)*N_STAGES*N_REPEATS; magnitude=np.empty((n,201)); phase=np.empty_like(magnitude); features=np.empty((n,5,201)); labels=np.empty(n); records=[]; corrections=[]; row=0
    for specimen_index,specimen in enumerate(ALL_SPECIMENS):
        path=raw_dir/specimen.filename; z,p=_load_specimen(path); correction=_apply_confirmed_sentinel_correction(frequency,z,p,specimen.filename)
        if correction: correction["source_sha256"]=_sha256(path); corrections.append(correction)
        original=specimen_index<len(ORIGINAL_SPECIMENS); stages=ORIGINAL_STAGE_MULTIPLIERS if original else TRANSFER_STAGE_MULTIPLIERS
        for stage_index in range(6):
            density,water=corrected_conditions(specimen_index,stage_index) if original else (specimen.density,specimen.water_content)
            for repeat_index in range(20):
                zr=z[:,repeat_index,stage_index]; pr=p[:,repeat_index,stage_index]; reduced=np.remainder(pr,360.0); magnitude[row]=zr; phase[row]=pr; features[row,0]=np.log10(zr/1000.0); features[row,1]=np.sin(np.deg2rad(reduced)); features[row,2]=np.cos(np.deg2rad(reduced)); features[row,3]=density; features[row,4]=water; labels[row]=specimen.concentration_factor*stages[stage_index]
                records.append({"observation_index":row+1,"specimen_id":specimen.specimen_id,"source_filename":specimen.filename,"cohort":specimen.cohort,"concentration_stage":stage_index+1,"repeat_index":repeat_index+1,"corrected_density":density,"corrected_water_content":water,"nominal_concentration_mg_l":labels[row]}); row+=1
    if not np.all(np.isfinite(features)) or not np.all(np.isfinite(labels)): raise RuntimeError("Non-finite processed data")
    return MeasuredEISDataset(features,magnitude,phase,labels,pd.DataFrame(records),frequency),corrections

def save_processed_dataset(dataset:MeasuredEISDataset,corrections:list[dict],path:Path)->None:
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(path,features=dataset.features,magnitude_ohm=dataset.magnitude_ohm,phase_deg=dataset.phase_deg,labels_mg_l=dataset.labels_mg_l,frequency_hz=dataset.frequency_hz,metadata_json=np.array(json.dumps(dataset.metadata.to_dict("records"))),corrections_json=np.array(json.dumps(corrections)))

def load_processed_dataset(path:Path)->MeasuredEISDataset:
    with np.load(path,allow_pickle=False) as z:
        metadata=pd.DataFrame(json.loads(str(z["metadata_json"]))); return MeasuredEISDataset(np.asarray(z["features"],float),np.asarray(z["magnitude_ohm"],float),np.asarray(z["phase_deg"],float),np.asarray(z["labels_mg_l"],float),metadata,np.asarray(z["frequency_hz"],float))


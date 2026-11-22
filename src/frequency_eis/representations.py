"""Training-only normalization for the two frozen physical EIS representations."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class RepresentationData:
    public_name: str
    spectral: np.ndarray
    physical: np.ndarray
    parameters: dict
    diagnostics: dict

def _mean_sd(values):
    mean=np.mean(values,dtype=np.float64); sd=np.std(values,dtype=np.float64,ddof=1)
    if not np.isfinite(sd) or sd<1e-12: sd=1.0
    return float(mean),float(sd)

def fit_admittance_representation(features,training_indices,scale_floor=1e-12,fallback=1.0):
    magnitude_kohm=10.0**features[:,0]; conductance=features[:,2]/magnitude_kohm; susceptance=-features[:,1]/magnitude_kohm; channels=[]; parameters=[]
    for channel in (conductance,susceptance):
        scale=float(np.median(np.abs(channel[training_indices]))); scale=scale if np.isfinite(scale) and scale>=scale_floor else fallback; transformed=np.arcsinh(channel/scale); mean,sd=_mean_sd(transformed[training_indices].reshape(-1)); channels.append((transformed-mean)/sd); parameters.append({"robust_scale":scale,"mean":mean,"sd":sd})
    physical=np.stack([features[:,3,0],features[:,4,0]],axis=1); physical_parameters=[]
    for j in range(2): mean,sd=_mean_sd(physical[training_indices,j]); physical[:,j]=(physical[:,j]-mean)/sd; physical_parameters.append({"mean":mean,"sd":sd})
    reconstructed=1/np.sqrt(conductance**2+susceptance**2); diagnostics={"max_magnitude_reconstruction_error_kohm":float(np.max(np.abs(reconstructed-magnitude_kohm))),"all_finite":bool(np.all(np.isfinite(channels)) and np.all(np.isfinite(physical)))}
    return RepresentationData("admittance",np.stack(channels,axis=1),physical,{"spectral":parameters,"physical":physical_parameters},diagnostics)

def fit_level_shape_representation(features,training_indices):
    log_magnitude=features[:,0]; level=log_magnitude.mean(axis=1); shape=log_magnitude-level[:,None]; mean,sd=_mean_sd(shape[training_indices].reshape(-1)); spectral=np.stack([(shape-mean)/sd,features[:,1],features[:,2]],axis=1); physical=np.stack([level,features[:,3,0],features[:,4,0]],axis=1); parameters=[]
    for j in range(3): m,s=_mean_sd(physical[training_indices,j]); physical[:,j]=(physical[:,j]-m)/s; parameters.append({"mean":m,"sd":s})
    diagnostics={"max_log_magnitude_reconstruction_error":float(np.max(np.abs(shape+level[:,None]-log_magnitude))),"max_centered_shape_mean":float(np.max(np.abs(shape.mean(axis=1)))),"all_finite":bool(np.all(np.isfinite(spectral)) and np.all(np.isfinite(physical)))}
    return RepresentationData("spectral_level_shape",spectral,physical,{"shape_mean":mean,"shape_sd":sd,"physical":parameters},diagnostics)

def fit_direct_representation(features,training_indices,late_fusion):
    """Normalize the original five quantities using real training observations only."""
    normalized=np.empty_like(features,dtype=np.float64); parameters=[]
    for channel in range(5):
        mean,sd=_mean_sd(features[training_indices,channel].reshape(-1)); normalized[:,channel]=(features[:,channel]-mean)/sd; parameters.append({"mean":mean,"sd":sd})
    if late_fusion:
        spectral=normalized[:,:3]
        physical=np.stack([normalized[:,3,0],normalized[:,4,0]],axis=1)
        name="direct_eis_physical_fusion"
    else:
        spectral=normalized
        physical=np.empty((len(features),0),dtype=np.float64)
        name="direct_five_channel"
    diagnostics={"all_finite":bool(np.all(np.isfinite(spectral)) and np.all(np.isfinite(physical))),"physical_frequency_constancy":bool(np.all(features[:,3:5]==features[:,3:5,:1]))}
    return RepresentationData(name,spectral,physical,{"channels":parameters,"late_fusion":bool(late_fusion)},diagnostics)


from pathlib import Path
import numpy as np
import torch
from frequency_eis.data import build_measured_dataset
from frequency_eis.model import SpectralPhysicalFusionCNN,count_trainable_parameters
from frequency_eis.representations import fit_admittance_representation,fit_level_shape_representation

ROOT=Path(__file__).resolve().parents[1]

def test_public_dataset_contract():
    data,corrections=build_measured_dataset(ROOT/"data/raw")
    assert data.features.shape==(2880,5,201)
    assert data.metadata.specimen_id.nunique()==24
    assert data.metadata.groupby(["specimen_id","concentration_stage"]).repeat_index.nunique().eq(20).all()
    assert len(corrections)==1 and np.all(np.diff(data.frequency_hz)>0)

def test_training_only_representations_and_reconstruction():
    data,_=build_measured_dataset(ROOT/"data/raw"); keep=~data.metadata.specimen_id.isin(["M15_02","M15_03","Z05_03","Z10_02","Z15_04"]).to_numpy(); indices=np.flatnonzero(keep); admittance=fit_admittance_representation(data.features,indices); level_shape=fit_level_shape_representation(data.features,indices)
    assert admittance.spectral.shape==(2880,2,201) and admittance.physical.shape==(2880,2)
    assert level_shape.spectral.shape==(2880,3,201) and level_shape.physical.shape==(2880,3)
    assert admittance.diagnostics["all_finite"] and level_shape.diagnostics["all_finite"]
    assert admittance.diagnostics["max_magnitude_reconstruction_error_kohm"]<1e-10
    assert level_shape.diagnostics["max_log_magnitude_reconstruction_error"]<1e-12

def test_frozen_model_shapes_and_parameter_counts():
    admittance=SpectralPhysicalFusionCNN(2,2); level_shape=SpectralPhysicalFusionCNN(3,3)
    assert count_trainable_parameters(admittance)==111697
    assert count_trainable_parameters(level_shape)==111785
    assert admittance(torch.zeros(4,2,201),torch.zeros(4,2)).shape==(4,)
    assert level_shape(torch.zeros(4,3,201),torch.zeros(4,3)).shape==(4,)


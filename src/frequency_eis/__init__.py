"""Frequency-resolved soil EIS data and frozen CNN representations."""

from .data import MeasuredEISDataset, build_measured_dataset, load_processed_dataset
from .model import SpectralPhysicalFusionCNN
from .representations import fit_admittance_representation, fit_level_shape_representation

__all__ = [
    "MeasuredEISDataset",
    "SpectralPhysicalFusionCNN",
    "build_measured_dataset",
    "fit_admittance_representation",
    "fit_level_shape_representation",
    "load_processed_dataset",
]


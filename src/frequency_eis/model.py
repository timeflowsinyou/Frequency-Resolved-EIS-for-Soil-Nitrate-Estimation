"""Frozen late-fusion CNN for frequency-resolved EIS and physical conditions."""
from __future__ import annotations
import torch
from torch import nn

class SpectralPhysicalFusionCNN(nn.Module):
    def __init__(self,spectral_channels:int,physical_features:int,dropout:float=0.2):
        super().__init__(); self.spectral_channels=spectral_channels; self.physical_features=physical_features
        self.spectral_encoder=nn.Sequential(nn.Conv1d(spectral_channels,16,5,padding="same"),nn.BatchNorm1d(16),nn.ReLU(),nn.MaxPool1d(2),nn.Conv1d(16,32,5,padding="same"),nn.BatchNorm1d(32),nn.ReLU(),nn.MaxPool1d(2),nn.Conv1d(32,64,3,padding="same"),nn.BatchNorm1d(64),nn.ReLU())
        self.spectral_projection=nn.Sequential(nn.Flatten(),nn.Linear(3200,32),nn.ReLU()); self.physical_projection=nn.Sequential(nn.Linear(physical_features,8),nn.ReLU()); self.regressor=nn.Sequential(nn.Dropout(dropout),nn.Linear(40,1))
    def forward(self,spectral,physical):
        if spectral.ndim!=3 or spectral.shape[1:]!=(self.spectral_channels,201): raise ValueError(f"Expected [N,{self.spectral_channels},201]")
        if physical.ndim!=2 or physical.shape[1]!=self.physical_features: raise ValueError(f"Expected [N,{self.physical_features}] physical inputs")
        encoded=self.spectral_projection(self.spectral_encoder(spectral)); conditions=self.physical_projection(physical); return self.regressor(torch.cat([encoded,conditions],dim=1)).squeeze(-1)

class DirectFiveChannelCNN(nn.Module):
    """Apply the convolutional encoder directly to all five repeated channels."""
    def __init__(self,dropout:float=0.2):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv1d(5,16,5,padding="same"),nn.BatchNorm1d(16),nn.ReLU(),nn.MaxPool1d(2),nn.Conv1d(16,32,5,padding="same"),nn.BatchNorm1d(32),nn.ReLU(),nn.MaxPool1d(2),nn.Conv1d(32,64,3,padding="same"),nn.BatchNorm1d(64),nn.ReLU())
        self.regressor=nn.Sequential(nn.Flatten(),nn.Linear(3200,32),nn.ReLU(),nn.Dropout(dropout),nn.Linear(32,1))

    def forward(self,spectral,physical=None):
        if spectral.ndim!=3 or spectral.shape[1:]!=(5,201): raise ValueError("Expected [N,5,201]")
        return self.regressor(self.encoder(spectral)).squeeze(-1)

def count_trainable_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)


"""Deterministic continuous-minibatch training for the frozen measured-data CNNs."""
from __future__ import annotations
import hashlib,os,random,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .model import DirectFiveChannelCNN,SpectralPhysicalFusionCNN,count_trainable_parameters

REPRESENTATION_DIMENSIONS={"admittance":(2,2),"spectral_level_shape":(3,3),"direct_eis_physical_fusion":(3,2),"direct_five_channel":(5,0)}

def set_reproducibility(seed:int,deterministic:bool=True):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8"); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=deterministic; torch.use_deterministic_algorithms(deterministic,warn_only=True)

def select_device(requested:str):
    if requested=="auto": return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested=="cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA was requested but is unavailable")
    if requested not in {"cpu","cuda"}: raise ValueError(requested)
    return torch.device(requested)

def continuous_minibatch_stream(population_size:int,nominal_passes:int,seed:int,fold_identity:int):
    parts=[]
    for pass_index in range(nominal_passes):
        generator=torch.Generator(device="cpu"); generator.manual_seed(seed+fold_identity*100_000+pass_index); parts.append(torch.randperm(population_size,generator=generator).numpy())
    stream=np.concatenate(parts).astype(np.int64)
    if len(stream)%16: raise RuntimeError("Sample presentations must be divisible by batch size 16")
    return stream

def _optimizer(model,learning_rate,l2):
    decay_ids={id(module.weight) for module in model.modules() if isinstance(module,(nn.Conv1d,nn.Linear))}; decayed=[]; not_decayed=[]
    for parameter in model.parameters(): (decayed if id(parameter) in decay_ids else not_decayed).append(parameter)
    return torch.optim.Adam([{"params":decayed,"weight_decay":l2},{"params":not_decayed,"weight_decay":0.0}],lr=learning_rate)

def _atomic_save(payload,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); temporary=path.with_suffix(path.suffix+".tmp"); torch.save(payload,temporary); os.replace(temporary,path)

def train_measured_model(model_name,representation_name,spectral,physical,labels,validation_spectral,validation_physical,seed,fold_identity,config,checkpoint_path,signature,resume,device):
    dimensions=REPRESENTATION_DIMENSIONS[representation_name]; stream=continuous_minibatch_stream(len(labels),int(config["nominal_passes"]),seed,fold_identity); expected_updates=len(stream)//int(config["batch_size"]); set_reproducibility(seed,bool(config["deterministic_algorithms"])); model=(DirectFiveChannelCNN(float(config["dropout"])) if representation_name=="direct_five_channel" else SpectralPhysicalFusionCNN(*dimensions,float(config["dropout"]))).to(device); optimizer=_optimizer(model,float(config["learning_rate"]),float(config["l2_regularization"])); criterion=nn.MSELoss(); next_update=0; previous_seconds=0.0; history=[]
    checkpoint_path=Path(checkpoint_path)
    if checkpoint_path.exists():
        if not resume: raise FileExistsError(f"Checkpoint exists; pass --resume: {checkpoint_path}")
        checkpoint=torch.load(checkpoint_path,map_location=device,weights_only=False)
        if checkpoint.get("signature")!=signature or checkpoint.get("model_name")!=model_name or checkpoint.get("network_seed")!=seed: raise RuntimeError(f"Incompatible checkpoint: {checkpoint_path}")
        if checkpoint.get("completed"): return checkpoint
        model.load_state_dict(checkpoint["model_state"]); optimizer.load_state_dict(checkpoint["optimizer_state"]); next_update=int(checkpoint["next_update"]); previous_seconds=float(checkpoint["training_seconds"]); history=checkpoint["history"]; random.setstate(checkpoint["python_rng_state"]); np.random.set_state(checkpoint["numpy_rng_state"]); torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if device.type=="cuda" and checkpoint.get("cuda_rng_state") is not None: torch.cuda.set_rng_state_all([x.cpu() for x in checkpoint["cuda_rng_state"]])
    xs=torch.from_numpy(spectral.astype(np.float32)).to(device); xp=torch.from_numpy(physical.astype(np.float32)).to(device); y=torch.from_numpy(labels.astype(np.float32)).to(device); started=time.perf_counter(); loss_sum=0.0; loss_count=0; logged_pass=len(history)
    for update in range(next_update,expected_updates):
        begin=update*int(config["batch_size"]); batch=torch.from_numpy(stream[begin:begin+int(config["batch_size"])]).to(device); model.train(); optimizer.zero_grad(set_to_none=True); prediction=model(xs[batch],xp[batch]); loss=criterion(prediction,y[batch]); loss.backward(); gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),float(config["gradient_clip_norm"]),error_if_nonfinite=True); optimizer.step(); loss_sum+=float(loss.detach().cpu())*len(batch); loss_count+=len(batch); presentations=(update+1)*int(config["batch_size"]); completed_passes=min(int(config["nominal_passes"]),presentations//len(labels)); elapsed=previous_seconds+time.perf_counter()-started
        if completed_passes>logged_pass or update+1==expected_updates:
            logged_pass=completed_passes; history.append({"nominal_passes_completed":completed_passes,"sample_presentations":presentations,"optimizer_updates":update+1,"training_loss":loss_sum/loss_count,"elapsed_seconds":elapsed,"last_gradient_norm_before_clipping":float(gradient.detach().cpu())}); loss_sum=0.0; loss_count=0
            checkpoint={"format_version":"public-frequency-eis-v1","signature":signature,"model_name":model_name,"representation":representation_name,"network_seed":seed,"fold_identity":fold_identity,"completed":False,"next_update":update+1,"model_state":model.state_dict(),"optimizer_state":optimizer.state_dict(),"history":history,"optimizer_updates":update+1,"sample_presentations":presentations,"training_seconds":elapsed,"stream_sha256":hashlib.sha256(stream.tobytes()).hexdigest().upper(),"python_rng_state":random.getstate(),"numpy_rng_state":np.random.get_state(),"torch_rng_state":torch.get_rng_state(),"cuda_rng_state":torch.cuda.get_rng_state_all() if device.type=="cuda" else None,"parameter_count":count_trainable_parameters(model)}; _atomic_save(checkpoint,checkpoint_path)
    model.eval(); vs=torch.from_numpy(validation_spectral.astype(np.float32)).to(device); vp=torch.from_numpy(validation_physical.astype(np.float32)).to(device)
    with torch.no_grad(): predictions=model(vs,vp).cpu().numpy().astype(np.float64)
    checkpoint.update(completed=True,predictions=predictions,actual_optimizer_updates=expected_updates,completed_at_unix=time.time()); _atomic_save(checkpoint,checkpoint_path); return checkpoint


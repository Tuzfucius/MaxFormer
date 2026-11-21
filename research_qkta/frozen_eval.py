"""Frozen QKTA evaluation, deterministic loaders, and GPU budget accounting."""
from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import importlib
from itertools import chain
import json
import os
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from research_qkta.adapter import QKTAExperimentContext, set_lif_backend
from research_qkta.checkpoint_audit import audit_and_load_checkpoint, verify_imagenette_wnid_mapping
from research_qkta.gates import QKTAConfig
from research_qkta.telemetry import QKTATelemetry


class BudgetExceededError(RuntimeError):
    """Raised before a run would exceed the configured GPU-hours budget."""


class IndexedSubset(Subset):
    """Subset that returns the source dataset index for fixed-example artifacts."""
    def __getitem__(self, position: int):
        index=self.indices[position]; item=self.dataset[index]
        return (*item,index)


class GpuBudgetTracker:
    def __init__(self, log_path: str | Path = "outputs/qkta/gpu_budget_log.json",
                 soft_cap_hours: float = 7.0, hard_cap_hours: float = 8.0) -> None:
        self.log_path=Path(log_path); self.soft_cap_hours=float(soft_cap_hours); self.hard_cap_hours=float(hard_cap_hours)
        self.log_path.parent.mkdir(parents=True,exist_ok=True)
        try:
            raw=json.loads(self.log_path.read_text(encoding="utf-8"))
            self.runs=raw.get("runs",[]) if isinstance(raw,dict) else raw
            existing_total=raw.get("total_gpu_hours",0.0) if isinstance(raw,dict) else 0.0
        except FileNotFoundError:
            self.runs=[]; existing_total=0.0
        self.total_gpu_hours=float(existing_total or (self.runs[-1].get("total_gpu_hours",0.0) if self.runs else 0.0))

    def check_can_start(self, estimated_hours: float = 0.0, stage: str = "eval") -> None:
        projected=self.total_gpu_hours+max(0.0,float(estimated_hours))
        if self.total_gpu_hours >= self.hard_cap_hours or projected >= self.hard_cap_hours:
            raise BudgetExceededError(f"GPU hard cap {self.hard_cap_hours:.2f}h reached/projected (current {self.total_gpu_hours:.3f}h).")
        if stage in {"train", "train_cifar", "sweep", "cal", "adapt", "adapt_cifar"} and self.total_gpu_hours >= self.soft_cap_hours:
            raise BudgetExceededError(f"GPU soft cap {self.soft_cap_hours:.2f}h reached for {stage} (current {self.total_gpu_hours:.3f}h).")

    def record_run(self, stage: str, track: str, start_utc: str, end_utc: str,
                   duration_seconds: float, peak_vram_mb: float, run_id: str | None = None) -> dict[str, Any]:
        self.total_gpu_hours += float(duration_seconds)/3600.0
        entry={"run_id":run_id or str(uuid.uuid4()),"stage":stage,"track":track,"start_utc":start_utc,
               "end_utc":end_utc,"duration_seconds":float(duration_seconds),"peak_vram_mb":float(peak_vram_mb),
               "total_gpu_hours":self.total_gpu_hours}
        self.runs.append(entry)
        ledger={"budget_soft_hours":self.soft_cap_hours,"budget_hard_hours":self.hard_cap_hours,
                "total_gpu_seconds":self.total_gpu_hours*3600.0,"total_gpu_hours":self.total_gpu_hours,"runs":self.runs}
        self.log_path.write_text(json.dumps(ledger,indent=2),encoding="utf-8")
        return entry


def _import_track_model(folder: str, module_name: str, symbol: str, **kwargs: Any) -> torch.nn.Module:
    """Import one duplicate-name upstream model without leaving its hubs in global import state."""
    root=str((Path(__file__).resolve().parents[1]/folder).resolve())
    old_path=list(sys.path); saved={n:sys.modules.pop(n) for n in ("mixer_hub","embedding_hub","ms_qkformer") if n in sys.modules}
    try:
        sys.path.insert(0,root); mod=importlib.import_module(module_name)
        factory=getattr(mod,symbol)
        return factory(**kwargs)
    finally:
        sys.path[:]=old_path
        for n in ("mixer_hub","embedding_hub","ms_qkformer"):
            sys.modules.pop(n,None)
        sys.modules.update(saved)


def _select_lif_backend(model: torch.nn.Module, device: str | torch.device, lif_backend: str) -> str:
    device=torch.device(device)
    if device.type!="cuda":
        set_lif_backend(model,"torch")
        return "torch"
    if lif_backend not in {"auto","cupy","torch"}:
        raise ValueError("lif_backend must be one of 'auto', 'cupy', or 'torch'")
    if lif_backend=="torch":
        set_lif_backend(model,"torch")
        return "torch"
    backend="torch"
    if torch.cuda.is_available():
        try:
            from spikingjelly.clock_driven.neuron import MultiStepLIFNode
            probe=MultiStepLIFNode(tau=2.0,backend="cupy").to(device)
            _reset(probe)
            with torch.no_grad(): probe(torch.zeros((2,1),device=device))
            torch.cuda.synchronize(device)
            _reset(probe)
            backend="cupy"
        except Exception:
            backend="torch"
    set_lif_backend(model,backend)
    return backend


def load_imagenet_model(checkpoint: str | Path="imagenet/checkpoints/ms_qk_10-384_T4.pth.tar", T: int=4,
                        device: str | torch.device="cpu", lif_backend: str="auto") -> tuple[torch.nn.Module, Any]:
    model=_import_track_model("imagenet","ms_qkformer","ms_qkformer_10_384",T=T)
    report=audit_and_load_checkpoint(model,checkpoint,strict=True); model.to(device); _select_lif_backend(model,device,lif_backend); model.eval()
    return model,report


def load_cifar10_model(T: int=4, embed_dims: int=384, device: str | torch.device="cpu", lif_backend: str="auto") -> torch.nn.Module:
    model=_import_track_model("cifar10-100","ms_qkformer","MS_QKformer",in_channels=3,num_classes=10,embed_dims=embed_dims,T=T)
    model.to(device); _select_lif_backend(model,device,lif_backend); model.eval(); return model


def load_cifar10_checkpoint(model: torch.nn.Module, checkpoint: str | Path) -> Any:
    """Strictly audit and load the registered CIFAR-10 baseline checkpoint."""
    return audit_and_load_checkpoint(model,checkpoint,strict=True)


def imagenet_dataset(val_dir: str | Path="data/imagenette2-320/val") -> Dataset:
    from torchvision import datasets
    mapping=verify_imagenette_wnid_mapping(val_dir)
    root=str((Path(__file__).resolve().parents[1]/"imagenet").resolve())
    old_path=list(sys.path); saved={n:sys.modules.pop(n) for n in ("misc","utils") if n in sys.modules}
    try:
        sys.path.insert(0,root); utils=importlib.import_module("utils")
        from types import SimpleNamespace
        transform=utils.build_transform(is_train=False,args=SimpleNamespace(input_size=224))
    finally:
        sys.path[:]=old_path
        for n in ("misc","utils"): sys.modules.pop(n,None)
        sys.modules.update(saved)
    dataset=datasets.ImageFolder(str(val_dir),transform=transform)
    remapped=[mapping["wnid_to_imagenet1k_idx"][dataset.classes[y]] for _,y in dataset.samples]
    dataset.targets=remapped; dataset.samples=[(path,remapped[i]) for i,(path,_) in enumerate(dataset.samples)]
    return dataset


def _stratified_indices(targets: Sequence[int], seed: int, sizes: tuple[int,int]) -> tuple[list[int],list[int]]:
    rng=np.random.default_rng(seed); labels=np.asarray(targets); cal=[]; ev=[]
    classes=np.unique(labels)
    for cls in classes:
        ids=np.flatnonzero(labels==cls); rng.shuffle(ids)
        ncal=round(len(ids)*sizes[0]/sum(sizes)); cal.extend(ids[:ncal].tolist()); ev.extend(ids[ncal:].tolist())
    rng.shuffle(cal); rng.shuffle(ev)
    # Correct small rounding drift while retaining class-balanced random ordering.
    if len(cal)>sizes[0]: ev=cal[sizes[0]:]+ev; cal=cal[:sizes[0]]
    elif len(cal)<sizes[0]:
        need=sizes[0]-len(cal); cal+=ev[:need]; ev=ev[need:]
    return cal[:sizes[0]],ev[:sizes[1]]


def save_split_indices(dataset: Dataset, output_dir: str | Path, cal_size: int, eval_size: int,
                       seed: int=42) -> tuple[list[int],list[int]]:
    targets=getattr(dataset,"targets",None)
    if targets is None: targets=[dataset[i][1] for i in range(len(dataset))]
    if cal_size+eval_size>len(targets): raise ValueError("Requested split exceeds dataset size")
    cal,ev=_stratified_indices(targets,seed,(cal_size,eval_size)); out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    (out/"cal_indices.json").write_text(json.dumps(cal),encoding="utf-8"); (out/"eval_indices.json").write_text(json.dumps(ev),encoding="utf-8")
    return cal,ev


def cifar10_datasets(data_root: str | Path="data/cifar10") -> tuple[Dataset,Dataset]:
    from torchvision import datasets
    mean=(0.4914,0.4822,0.4465); std=(0.2470,0.2435,0.2616)
    root=str((Path(__file__).resolve().parents[1]/"cifar10-100").resolve())
    old_path=list(sys.path); saved={n:sys.modules.pop(n) for n in ("transforms_factory","aa_snn") if n in sys.modules}
    try:
        sys.path.insert(0,root); factory=importlib.import_module("transforms_factory").create_transform
        train_tf=factory(32,is_training=True,scale=(1.,1.),ratio=(1.,1.),hflip=.5,vflip=0.,color_jitter=0.,
                         auto_augment="rand-m9-n1-mstd0.4-inc1",interpolation="bicubic",use_prefetcher=False,
                         mean=mean,std=std,re_prob=.25,re_mode="const",re_count=1)
        eval_tf=factory(32,is_training=False,interpolation="bicubic",use_prefetcher=False,mean=mean,std=std,crop_pct=1.)
    finally:
        sys.path[:]=old_path
        for n in ("transforms_factory","aa_snn"): sys.modules.pop(n,None)
        sys.modules.update(saved)
    return datasets.CIFAR10(str(data_root),train=True,download=False,transform=train_tf),datasets.CIFAR10(str(data_root),train=False,download=False,transform=eval_tf)


def _reset(model: torch.nn.Module) -> None:
    from spikingjelly.clock_driven import functional
    functional.reset_net(model)


def verify_determinism_and_reset(model: torch.nn.Module, batch: torch.Tensor, device: str | torch.device="cpu",
                                 atol: float=1e-5, rtol: float=1e-4) -> dict[str,float]:
    model.eval(); x=batch.to(device)
    with torch.no_grad():
        _reset(model); a=model(x); _reset(model); b=model(x)
        _reset(model); rev=model(x.flip(0)).flip(0); _reset(model)
    for module in model.modules():
        if type(module).__name__!="MultiStepLIFNode" or not hasattr(module,"v"): continue
        state=module.v
        if isinstance(state,torch.Tensor) and torch.count_nonzero(state).item()!=0:
            raise AssertionError("functional.reset_net(model) left nonzero MultiStepLIFNode membrane state")
        if not isinstance(state,torch.Tensor) and isinstance(state,(float,int)) and float(state)!=0.0:
            raise AssertionError("functional.reset_net(model) did not restore scalar LIF state to zero")
    repeat=float((a-b).abs().max().item()); reverse=float((a-rev).abs().max().item())
    if not torch.allclose(a,b,atol=atol,rtol=rtol) or not torch.allclose(a,rev,atol=atol,rtol=rtol):
        raise AssertionError(f"Determinism/reset failed: repeat max diff={repeat}, reversed max diff={reverse}")
    return {"repeat_max_abs_diff":repeat,"reverse_order_max_abs_diff":reverse}


class _SnapshotTelemetry(QKTATelemetry):
    def __init__(self, enabled: bool=True) -> None:
        super().__init__(enabled); self.snapshots: dict[str,dict[str,np.ndarray]]={}
    def record(self, layer_name: str, q_sum: torch.Tensor, attn: torch.Tensor, k: torch.Tensor, post_k: torch.Tensor) -> None:
        super().record(layer_name,q_sum,attn,k,post_k)
        if self.enabled and layer_name not in self.snapshots:
            self.snapshots[layer_name]={"q_sum":q_sum.detach().cpu().numpy(),"A":attn.detach().cpu().numpy()}


def evaluate_model_config(model: torch.nn.Module, loader: Iterable, config: QKTAConfig,
                          target_layers: str | Sequence[str], device: str | torch.device="cpu",
                          telemetry_enabled: bool=True, sample_indices: Sequence[int] | None=None,
                          tracker: GpuBudgetTracker | None=None, track: str="unknown", stage: str="eval") -> dict[str,Any]:
    """Evaluate a loader in separate telemetry and clean timed passes with resets around every batch."""
    model.eval(); dev=torch.device(device); start=datetime_now(); t0=time.perf_counter()
    if tracker: tracker.check_can_start(0.0,stage)
    telemetry=_SnapshotTelemetry(enabled=telemetry_enabled); correct=total=0; snapshots={}
    is_cuda=dev.type=="cuda"
    if is_cuda: torch.cuda.reset_peak_memory_stats(dev)
    with QKTAExperimentContext(model,config,target_layers,telemetry):
        with torch.no_grad():
            snapshot_written=False
            for batch in loader:
                images,labels=batch[:2]; images=images.to(dev); labels=labels.to(dev); _reset(model)
                logits=model(images); correct+=int((logits.argmax(1)==labels).sum().item()); total+=labels.numel(); _reset(model)
                if telemetry_enabled and not snapshot_written:
                    indices=batch[2].tolist() if len(batch)>2 else list(range(images.shape[0]))
                    first_layer=next(iter(telemetry.snapshots),None)
                    for name,arrs in telemetry.snapshots.items():
                        if name!=first_layer: continue
                        for j in range(images.shape[0]):
                            sample_key=f"sample_{int(indices[j]):05d}"
                            snapshots.setdefault(f"{sample_key}__q_sum",arrs["q_sum"][:,j])
                            snapshots[f"{sample_key}__{name}__A"]=arrs["A"][:,j]
                            snapshots[f"{sample_key}__image"]=images[j].detach().cpu().numpy()
                            snapshots[f"{sample_key}__target"]=labels[j].detach().cpu().numpy()
                            snapshots[f"{sample_key}__pred"]=logits[j].argmax().detach().cpu().numpy()
                    telemetry.snapshots.clear()
                    snapshot_written=True
    summary=telemetry.summary(); elapsed=time.perf_counter()-t0
    # Hook-free pass: no telemetry recorder is enabled, and CUDA timing is synchronized.
    clean=QKTATelemetry(enabled=False); durations=[]; clean_correct=clean_total=0
    if is_cuda: torch.cuda.reset_peak_memory_stats(dev)
    with QKTAExperimentContext(model,config,target_layers,clean):
        with torch.no_grad():
            iterator=iter(loader)
            warm=next(iterator,None)
            if warm is not None:
                wi,wl=warm[:2]; _reset(model)
                with torch.no_grad(): model(wi.to(dev))
                _reset(model)
                if is_cuda: torch.cuda.synchronize(dev)
                iterator=chain((warm,),iterator)
            for batch in iterator:
                images,labels=batch[:2]; images=images.to(dev); labels=labels.to(dev); _reset(model)
                if is_cuda: torch.cuda.synchronize(dev)
                t=time.perf_counter(); logits=model(images)
                if is_cuda: torch.cuda.synchronize(dev)
                durations.append((time.perf_counter()-t)/max(1,images.shape[0])); clean_correct+=int((logits.argmax(1)==labels).sum().item()); clean_total+=labels.numel(); _reset(model)
    peak=float(torch.cuda.max_memory_allocated(dev)/1024**2) if is_cuda else 0.0
    duration=time.perf_counter()-t0; end=datetime_now()
    rec={"run_id":str(uuid.uuid4()),"stage":stage,"track":track,"start_utc":start,"end_utc":end,
         "duration_seconds":duration,"peak_vram_mb":peak,"top1_acc":correct/max(total,1),"clean_top1_acc":clean_correct/max(clean_total,1),
         "latency_ms_per_img":float(np.mean(durations)*1000) if durations else 0.0,"telemetry":summary,"snapshots":snapshots,
         "mode":config.mode,"params":config.__dict__.copy(),"target_layers":target_layers}
    if tracker: tracker.record_run(stage,track,start,end,duration,peak,rec["run_id"])
    return rec


def datetime_now() -> str: return dt.datetime.now(dt.timezone.utc).isoformat()


def train_cifar10_baseline(data_root: str | Path="data/cifar10", output_path: str | Path="outputs/qkta/track2_cifar10/cifar10_ms_qk_t4_baseline.pth",
                           epochs: int=100,batch_size: int=128,lr: float=1.5e-3,weight_decay: float=6e-2,
                           T: int=4,embed_dims: int=384,device: str="cuda",max_train_hours: float=2.2,
                           tracker: GpuBudgetTracker | None=None,seed: int=42) -> dict[str,Any]:
    """Train the untouched CIFAR10 baseline under an absolute wall clock and ledger cap."""
    if device!="cuda" or not torch.cuda.is_available(): raise RuntimeError("Baseline training requires the prepared CUDA environment.")
    if tracker is None: tracker=GpuBudgetTracker()
    tracker.check_can_start(max_train_hours,"train_cifar")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    train_ds,val_ds=cifar10_datasets(data_root); train_loader=DataLoader(train_ds,batch_size=batch_size,shuffle=True,num_workers=4,pin_memory=True)
    val_loader=DataLoader(val_ds,batch_size=batch_size,shuffle=False,num_workers=4,pin_memory=True)
    model=load_cifar10_model(T,embed_dims,device); opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=weight_decay)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs,eta_min=1e-5); scaler=torch.amp.GradScaler("cuda")
    start=datetime_now(); t0=time.monotonic(); best=-1.0; out=Path(output_path); out.parent.mkdir(parents=True,exist_ok=True)
    from spikingjelly.clock_driven import functional
    from timm.data.mixup import Mixup
    mixup=Mixup(mixup_alpha=.5,cutmix_alpha=.5,prob=1.,switch_prob=.5,mode="batch",label_smoothing=.1,num_classes=10)
    for epoch in range(epochs):
        model.train()
        for images,labels in train_loader:
            if time.monotonic()-t0>=max_train_hours*3600: break
            images=images.to(device,non_blocking=True); labels=labels.to(device,non_blocking=True); functional.reset_net(model); opt.zero_grad(set_to_none=True)
            images,soft_labels=mixup(images,labels)
            with torch.amp.autocast("cuda"):
                logits=model(images); loss=torch.nn.functional.cross_entropy(logits,soft_labels)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); functional.reset_net(model)
        model.eval(); good=n=0
        with torch.no_grad():
            for images,labels in val_loader:
                if time.monotonic()-t0>=max_train_hours*3600: break
                functional.reset_net(model); logits=model(images.to(device)); good+=int((logits.argmax(1)==labels.to(device)).sum()); n+=len(labels); functional.reset_net(model)
        acc=good/max(n,1)
        if acc>best:
            best=acc; torch.save({"state_dict":model.state_dict(),"epoch":epoch+1,"val_acc":best,
                                 "args":{"T":T,"embed_dims":embed_dims,"batch_size":batch_size,"lr":lr,"seed":seed}},out)
        scheduler.step()
        if time.monotonic()-t0>=max_train_hours*3600: break
    duration=time.monotonic()-t0; end=datetime_now(); peak=float(torch.cuda.max_memory_allocated()/1024**2)
    tracker.record_run("train_cifar","cifar10",start,end,duration,peak)
    return {"checkpoint":str(out),"best_val_acc":best,"duration_seconds":duration}


def run_cifar10_lightweight_adapt(model: torch.nn.Module, train_loader: Iterable, *, max_epochs: int=10,
                                  enabled: bool=False, device: str="cuda", lr: float=1e-5,
                                  max_train_hours: float=0.25, tracker: GpuBudgetTracker | None=None,
                                  track: str="cifar10", **kwargs: Any) -> dict[str,Any]:
    """Optional isolated adaptation hook; disabled unless explicitly enabled by caller."""
    if not enabled: return {"enabled":False,"epochs":0}
    if not 1<=max_epochs<=10: raise ValueError("Lightweight adaptation requires 1..10 epochs")
    if tracker is None: tracker=GpuBudgetTracker()
    tracker.check_can_start(max_train_hours,"adapt_cifar")
    model.to(device); model.train(); optimizer=torch.optim.AdamW(model.parameters(),lr=lr)
    scaler=torch.amp.GradScaler("cuda",enabled=str(device).startswith("cuda")); start=datetime_now(); t0=time.monotonic()
    from spikingjelly.clock_driven import functional
    completed=0
    for epoch in range(max_epochs):
        for images,labels in train_loader:
            if time.monotonic()-t0>=max_train_hours*3600: break
            images=images.to(device); labels=labels.to(device); functional.reset_net(model); optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda",enabled=str(device).startswith("cuda")):
                loss=torch.nn.functional.cross_entropy(model(images),labels)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); functional.reset_net(model)
        completed=epoch+1
        if time.monotonic()-t0>=max_train_hours*3600: break
    duration=time.monotonic()-t0; end=datetime_now(); peak=float(torch.cuda.max_memory_allocated()/1024**2) if str(device).startswith("cuda") else 0.
    tracker.record_run("adapt_cifar",track,start,end,duration,peak)
    return {"enabled":True,"duration_seconds":duration,"epochs":completed}

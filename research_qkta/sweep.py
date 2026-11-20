"""Command line runner for registered QKTA sanity, calibration, and frozen evaluation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from research_qkta.frozen_eval import (GpuBudgetTracker, IndexedSubset, cifar10_datasets, evaluate_model_config,
    imagenet_dataset, load_cifar10_checkpoint, load_cifar10_model, load_imagenet_model, save_split_indices,
    train_cifar10_baseline, verify_determinism_and_reset)
from research_qkta.adapter import QKTAExperimentContext
from research_qkta.gates import QKTAConfig
from research_qkta.plots import generate_frozen_plots

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/qkta"


def _configs() -> list[tuple[str,QKTAConfig]]:
    result=[("B0_original",QKTAConfig(mode="B0_original")),("B0_custom",QKTAConfig(mode="B0_custom"))]
    result += [(f"B1_lif_theta{x}",QKTAConfig(mode="B1_lif",theta0=x)) for x in (.5,1.,1.5,2.)]
    result += [(f"B1_count_k{x}",QKTAConfig(mode="B1_count",k_thresh=x)) for x in (1,2,3,4)]
    result += [(f"H_b{x}",QKTAConfig(mode="H",eta=0.,rho=.5,b=x)) for x in (.5,1.,2.,3.,4.)]
    result += [(f"C_b{x}",QKTAConfig(mode="C",eta=.5,kappa=1.,rho=.5,b=x)) for x in (.5,1.,2.,3.,4.)]
    return result


def _load_track(track: str, device: str, args: argparse.Namespace):
    if track=="imagenet":
        model,report=load_imagenet_model(args.checkpoint or ROOT/"imagenet/checkpoints/ms_qk_10-384_T4.pth.tar",4,device)
        ds=imagenet_dataset(args.data_root or ROOT/"data/imagenette2-320/val")
        return model,ds,report
    train_ds,test_ds=cifar10_datasets(args.data_root or ROOT/"data/cifar10")
    model=load_cifar10_model(4,384,device)
    cifar_ckpt=args.checkpoint or ROOT/"outputs/qkta/track2_cifar10/cifar10_ms_qk_t4_baseline.pth"
    report=load_cifar10_checkpoint(model,cifar_ckpt)
    return model,test_ds,report


def _git_head() -> str:
    return subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()


def _sha(path: str | Path | None) -> str | None:
    if not path or not Path(path).is_file(): return None
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()


def _metric_record(rec: dict[str,Any], mode_label: str, baseline: float | None=None) -> dict[str,Any]:
    tel=rec.get("telemetry",{}); layers=list(tel.values())
    kpre=sum(x["K_pre_spikes"] for x in layers); kpost=sum(x["K_post_spikes"] for x in layers)
    openrate=float(np.mean([x["A_open_rate"] for x in layers])) if layers else 0.
    per_step_open=np.mean([x["per_step_open_rate"] for x in layers],axis=0).tolist() if layers else []
    per_step_k=np.sum([x["per_step_post_k"] for x in layers],axis=0).tolist() if layers else []
    acc=rec["top1_acc"]
    return {"run_id":rec["run_id"],"stage":rec["stage"],"track":rec["track"],"mode":mode_label,
        "git_head":_git_head(),"checkpoint_sha256":rec.get("checkpoint_sha256"),"dataset":rec.get("dataset"),"split":rec.get("split"),
        "sample_indices_file":rec.get("sample_indices_file"),"T":4,"device":rec.get("device"),"seed":42,
        "target_layers":rec["target_layers"],"params":rec["params"],"top1_acc":acc,
        "delta_top1_pp":(acc-baseline)*100 if baseline is not None else 0.0,"K_pre_spikes":kpre,"K_post_spikes":kpost,
        "K_post_vs_B0":rec.get("K_post_vs_B0",1.0),"K_retention_vs_own_input":kpost/max(kpre,1),"A_open_rate":openrate,
        "per_step_open_rate":per_step_open,"per_step_post_k":per_step_k,"latency_ms_per_img":rec["latency_ms_per_img"],
        "peak_vram_mb":rec["peak_vram_mb"]}


def _run_one(model,loader,config,layers,device,tracker,track,stage,baseline=None):
    rec=evaluate_model_config(model,loader,config,layers,device,True,tracker=tracker,track=track,stage=stage)
    rec["K_post_vs_B0"]=1.0
    if baseline is not None:
        b0=baseline.get("telemetry",{}); current=rec["telemetry"]
        rec["K_post_vs_B0"]=sum(x["K_post_spikes"] for x in current.values())/max(sum(x["K_post_spikes"] for x in b0.values()),1)
    return rec


def _capture_mask_snapshots(rec: dict[str,Any], mode: str, payload: dict[str,np.ndarray]) -> None:
    """Store common fixed samples and first QKTA-layer masks from the multi-layer pass."""
    if rec.get("target_layers")!="all_qkta": return
    if mode not in {"B0_original","H","C","P"}: return
    if mode=="P" and int(rec.get("params",{}).get("perm_seed",-1))!=101: return
    prefix="B0" if mode=="B0_original" else mode
    for key,value in rec.get("snapshots",{}).items():
        sample_key,_,field=key.partition("__")
        if field in {"image","pred","target"}:
            payload.setdefault(key,np.asarray(value))
        elif field=="q_sum":
            payload.setdefault(key,np.asarray(value))
        elif field.endswith("__A"):
            layer=field[:-3]
            # frozen_eval keeps only the first selected layer, with its module path explicit.
            payload.setdefault(f"{sample_key}__{layer}__A",np.asarray(value))
            payload.setdefault(f"{sample_key}__{prefix}_A",np.asarray(value))
            q_key=f"{sample_key}__q_sum"
            if q_key in payload:
                payload.setdefault(f"{sample_key}__{prefix}_suppressed",payload[q_key]*(1-np.asarray(value)))


def _write_rows(rows: list[dict[str,Any]], csv_path: Path, json_path: Path) -> None:
    csv_path.parent.mkdir(parents=True,exist_ok=True); json_path.parent.mkdir(parents=True,exist_ok=True)
    scalar=["run_id","stage","track","mode","git_head","checkpoint_sha256","dataset","split","sample_indices_file","T","device","seed","target_layers","params","top1_acc","delta_top1_pp","K_pre_spikes","K_post_spikes","K_post_vs_B0","K_retention_vs_own_input","A_open_rate","per_step_open_rate","per_step_post_k","latency_ms_per_img","peak_vram_mb"]
    with csv_path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=scalar); writer.writeheader()
        for row in rows: writer.writerow({k:json.dumps(row.get(k)) if isinstance(row.get(k),(dict,list)) else row.get(k) for k in scalar})
    p_groups={}
    for row in rows:
        if row.get("mode")!="P": continue
        key=(row.get("target_layers"),row.get("params",{}).get("b"))
        p_groups.setdefault(key,[]).append(row)
    p_summary=[]
    for (layer,b),group in p_groups.items():
        p_summary.append({"target_layers":layer,"b":b,"num_seeds":len(group),
            "top1_acc_mean":float(np.mean([x["top1_acc"] for x in group])),"top1_acc_min":float(min(x["top1_acc"] for x in group)),"top1_acc_max":float(max(x["top1_acc"] for x in group)),
            "K_post_mean":float(np.mean([x["K_post_spikes"] for x in group])),"K_post_min":int(min(x["K_post_spikes"] for x in group)),"K_post_max":int(max(x["K_post_spikes"] for x in group))})
    json_path.write_text(json.dumps({"results":rows,"P_seed_summary":p_summary},indent=2),encoding="utf-8")


def _phase(args: argparse.Namespace) -> None:
    track=args.track; device=args.device
    if args.phase=="train_cifar":
        train_cifar10_baseline(args.data_root or ROOT/"data/cifar10",device=device,max_train_hours=2.2,tracker=GpuBudgetTracker())
        return
    if args.phase=="adapt_cifar":
        if not args.enable_adapt: print("CIFAR-10 lightweight adaptation is disabled (pass --enable-adapt to activate).")
        else:
            from research_qkta.frozen_eval import run_cifar10_lightweight_adapt
            train_ds,_=cifar10_datasets(args.data_root or ROOT/"data/cifar10")
            train_loader=DataLoader(train_ds,batch_size=128,shuffle=True,num_workers=args.workers)
            adapt_model=load_cifar10_model(device=device)
            load_cifar10_checkpoint(adapt_model,args.checkpoint or ROOT/"outputs/qkta/track2_cifar10/cifar10_ms_qk_t4_baseline.pth")
            run_cifar10_lightweight_adapt(adapt_model,train_loader,enabled=True,device=device,
                                           max_epochs=args.adapt_epochs,tracker=GpuBudgetTracker())
        return
    model,dataset,audit=_load_track(track,device,args)
    checkpoint_path=args.checkpoint or (ROOT/"imagenet/checkpoints/ms_qk_10-384_T4.pth.tar" if track=="imagenet" else ROOT/"outputs/qkta/track2_cifar10/cifar10_ms_qk_t4_baseline.pth")
    checkpoint_hash=audit.sha256 if hasattr(audit,"sha256") else _sha(checkpoint_path)
    run_dir=OUT/("track1_imagenet" if track=="imagenet" else "track2_cifar10")
    cal_size,eval_size=((1000,2925) if track=="imagenet" else (2000,8000))
    cal,ev=save_split_indices(dataset,run_dir,cal_size,eval_size,42)
    chosen=cal if args.phase in ("cal","sanity") else ev
    split_name="Calibration" if args.phase in ("cal","sanity") else "Independent Eval"
    indices_file=run_dir/("cal_indices.json" if split_name=="Calibration" else "eval_indices.json")
    loader=DataLoader(IndexedSubset(dataset,chosen),batch_size=args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=device.startswith("cuda"))
    tracker=GpuBudgetTracker()
    if args.phase=="sanity":
        first=next(iter(loader))[0][:min(4,args.batch_size)]
        determinism=verify_determinism_and_reset(model,first,device)
        b0=_run_one(model,loader,QKTAConfig(mode="B0_original"),"stage1",device,tracker,track,"sanity")
        with torch.no_grad():
            from spikingjelly.clock_driven import functional
            functional.reset_net(model)
            with QKTAExperimentContext(model,QKTAConfig(mode="B0_original"),"stage1"):
                base_logits=model(first.to(device))
            functional.reset_net(model)
            with QKTAExperimentContext(model,QKTAConfig(mode="B0_custom"),"stage1"):
                custom_logits=model(first.to(device))
            functional.reset_net(model)
        logit_diff=float((base_logits-custom_logits).abs().max().item())
        b0_clean={k:v for k,v in b0.items() if k!="snapshots"}
        dest=run_dir/"sanity.json"; dest.write_text(json.dumps({"audit":audit.to_dict() if audit else None,"determinism":determinism,"B0_original_full_split":b0_clean,"B0_original_vs_custom_logits_max_abs_diff":logit_diff},indent=2),encoding="utf-8")
        return
    if args.phase=="cal":
        rows=[]
        for layer in ("stage1","all_qkta"):
            for name,cfg in _configs():
                rec=_run_one(model,loader,cfg,layer,device,tracker,track,"cal")
                rec.update(dataset=track,split=split_name,sample_indices_file=str(indices_file),device=device,checkpoint_sha256=checkpoint_hash)
                rows.append(_metric_record(rec,name))
        _write_rows(rows,run_dir/"cal_results.csv",run_dir/"cal_results.json"); return
    if not args.candidates_json: raise ValueError("--candidates-json is required for locked independent evaluation")
    candidates=json.loads(Path(args.candidates_json).read_text(encoding="utf-8"))
    candidates=candidates.get("candidates",candidates.get("results",candidates))
    if isinstance(candidates,dict): candidates=[candidates]
    locked=[]; locked_lif=[]; locked_count=[]
    for c in candidates:
        params=c.get("params",c)
        mode=params.get("mode",c.get("mode","C")); target=c.get("target_layers",c.get("target_layer","all_qkta"))
        if mode=="C": locked.append(("C",QKTAConfig(**params),target))
        elif mode=="B1_lif": locked_lif.append((mode,QKTAConfig(**params),target))
        elif mode=="B1_count": locked_count.append((mode,QKTAConfig(**params),target))
        else: raise ValueError(f"Unsupported locked eval candidate mode: {mode}")
    rows=[]; snap_payload={}; baselines={}
    for layer in ("stage1","all_qkta"):
        b0=_run_one(model,loader,QKTAConfig(mode="B0_original"),layer,device,tracker,track,"eval")
        b0.update(dataset=track,split=split_name,sample_indices_file=str(indices_file),device=device,checkpoint_sha256=checkpoint_hash)
        baselines[layer]=b0; rows.append(_metric_record(b0,"B0_original"))
        _capture_mask_snapshots(b0,"B0_original",snap_payload)
        custom=_run_one(model,loader,QKTAConfig(mode="B0_custom"),layer,device,tracker,track,"eval",b0)
        custom.update(dataset=track,split=split_name,sample_indices_file=str(indices_file),device=device,checkpoint_sha256=checkpoint_hash)
        rows.append(_metric_record(custom,"B0_custom",b0["top1_acc"]))
        lif_for_layer=[item for item in locked_lif if item[2]==layer]
        count_for_layer=[item for item in locked_count if item[2]==layer]
        control_cfgs=(lif_for_layer if lif_for_layer else [("B1_lif",QKTAConfig(mode="B1_lif",theta0=t),layer) for t in (.5,1.,1.5,2.)])
        control_cfgs += (count_for_layer if count_for_layer else [("B1_count",QKTAConfig(mode="B1_count",k_thresh=k),layer) for k in (1,2,3,4)])
        for _,c_cfg,target in locked:
            if target==layer:
                control_cfgs.append(("H",QKTAConfig(mode="H",rho=.5,b=c_cfg.b,eta=0.),layer))
        for control_name,control_cfg,_target in control_cfgs:
            control=_run_one(model,loader,control_cfg,layer,device,tracker,track,"eval",b0)
            control.update(dataset=track,split=split_name,sample_indices_file=str(indices_file),device=device,checkpoint_sha256=checkpoint_hash)
            rows.append(_metric_record(control,control_name,b0["top1_acc"]))
            if layer=="all_qkta" and control_name=="H":
                _capture_mask_snapshots(control,"H",snap_payload)
        for _,cfg,target in locked:
            if target!=layer: continue
            candidates_cfg=[("C",cfg)] + [("P",QKTAConfig(**{**asdict(cfg),"mode":"P","perm_seed":s})) for s in (101,202,303)]
            for name,gate in candidates_cfg:
                rec=_run_one(model,loader,gate,layer,device,tracker,track,"eval",b0)
                rec.update(dataset=track,split=split_name,sample_indices_file=str(indices_file),device=device,checkpoint_sha256=checkpoint_hash)
                rows.append(_metric_record(rec,name,b0["top1_acc"]))
                _capture_mask_snapshots(rec,name,snap_payload)
    result_json=run_dir/"frozen_results.json"; result_csv=run_dir/"frozen_results.csv"; _write_rows(rows,result_csv,result_json)
    snapshot_file=run_dir/"mask_snapshots.npz"; np.savez_compressed(snapshot_file,**snap_payload)
    metadata={"dataset":track,"dataset_disclaimer":"10-synset ImageNet-1K subset (1000-way logits, NOT full 1000-class ImageNet Top-1)" if track=="imagenet" else "CIFAR-10 test set",
        "sample_count":len(ev),"split":split_name,"cal_size":cal_size,"eval_size":eval_size,
        "model_variant":"ms_qkformer_10_384" if track=="imagenet" else "cifar10_ms_qkformer","T":4,"target_layers":"stage1 and all_qkta",
        "checkpoint_filename":str(checkpoint_path),"checkpoint_sha256":checkpoint_hash,"git_head":_git_head(),
        "mask_sample_indices":ev[:min(args.batch_size,len(ev))]}
    if checkpoint_hash: metadata["checkpoint_sha256_short"]=checkpoint_hash[:12]
    (run_dir/"frozen_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    generate_frozen_plots(result_json,snapshot_file,run_dir/"plots",metadata)


def build_parser() -> argparse.ArgumentParser:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--track",choices=("imagenet","cifar10"),required=True)
    parser.add_argument("--phase",choices=("sanity","cal","eval","train_cifar","adapt_cifar"),required=True)
    parser.add_argument("--candidates-json"); parser.add_argument("--checkpoint"); parser.add_argument("--data-root")
    parser.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu"); parser.add_argument("--batch-size",type=int,default=64)
    parser.add_argument("--workers",type=int,default=4); parser.add_argument("--sanity-samples",type=int,default=32)
    parser.add_argument("--enable-adapt",action="store_true"); parser.add_argument("--adapt-epochs",type=int,default=5)
    return parser


def main(argv: list[str] | None=None) -> None:
    args=build_parser().parse_args(argv)
    if args.track=="imagenet" and args.phase in ("train_cifar","adapt_cifar"): raise ValueError(f"{args.phase} is only supported on CIFAR10")
    _phase(args)


if __name__=="__main__": main()

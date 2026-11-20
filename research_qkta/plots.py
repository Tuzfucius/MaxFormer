"""Headless, protocol-labeled figures for QKTA synthetic and frozen evaluations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FAMILIES = ("B0_custom", "B1_count", "B1_lif", "H", "C", "P")
COLORS = dict(zip(("B0_original", *FAMILIES), plt.cm.tab10.colors))


def _json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def _npz(path: str | Path) -> Any:
    return np.load(path, allow_pickle=False)


def _save(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _arr(npz: Any, key: str, default: Any = None) -> Any:
    if key not in npz:
        return default
    a = np.asarray(npz[key])
    # Preserve leading time and image axes; only peel batch/head/channel singleton axes.
    while a.ndim > 3:
        axis = next((i for i in range(1, a.ndim) if a.shape[i] == 1), None)
        if axis is None: break
        a = np.squeeze(a, axis=axis)
    return a


def generate_synthetic_plots(results_json_path: str | Path, traces_npz_path: str | Path,
                             output_dir: str | Path) -> list[Path]:
    """Create per-pattern temporal spatial maps and a registered-gate comparison."""
    payload, traces = _json(results_json_path), _npz(traces_npz_path)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    records = payload.get("results", [])
    dims = payload.get("dimensions", {})
    T, H, W, Dh = (dims.get(k, d) for k, d in (("T", 10), ("H", 16), ("W", 16), ("D_head", 8)))
    saved: list[Path] = []
    chosen = ("B0_custom", "B1_count_k1", "B1_lif_theta0.5", "H_b2.0", "C_b2.0", "P_b2.0_seed101")
    patterns = payload.get("patterns", {})
    for pattern, info in patterns.items():
        q, gt = _arr(traces, f"{pattern}__q_sum"), _arr(traces, f"{pattern}__gt_mask")
        if q is None or gt is None:
            continue
        q = q.reshape(-1, H, W); gt = gt.reshape(-1, H, W)
        fig, axes = plt.subplots(len(chosen), T * 7, figsize=(2.0 * T * 7, 2.3 * len(chosen)))
        axes = np.asarray(axes).reshape(len(chosen), -1)
        for ri, cid in enumerate(chosen):
            aa = _arr(traces, f"{pattern}__{cid}__A")
            rr = _arr(traces, f"{pattern}__{cid}__R")
            th = _arr(traces, f"{pattern}__{cid}__theta")
            cc = _arr(traces, f"{pattern}__{cid}__c")
            aa = np.zeros_like(q) if aa is None else aa.reshape(-1, H, W)
            rr = np.zeros_like(q) if rr is None else rr.reshape(-1, H, W)
            th = np.zeros_like(q) if th is None else th.reshape(-1, H, W)
            cc = np.zeros_like(q) if cc is None else cc.reshape(-1, H, W)
            for t in range(T):
                vals = (q[t], gt[t], aa[t], q[t] * (1 - aa[t]), rr[t], th[t], cc[t])
                labels = ("q_sum", "GT", "A", "suppressed", "R trace", "dynamic θ", "contrast c")
                for k, (val, label) in enumerate(zip(vals, labels)):
                    ax = axes[ri, t * 7 + k]; ax.imshow(val, cmap="magma" if k == 0 else "viridis")
                    ax.set_title(f"t={t} {label}", fontsize=8); ax.set_xticks([]); ax.set_yticks([])
            # Per-step traces and contrast masks are summarized in a compact strip in figure title/footer.
            axes[ri, 0].set_ylabel(f"{cid}\nR̄={np.mean(rr):.2f}, θ̄={np.mean(th):.2f}, c̄={np.mean(cc):.2f}", fontsize=8)
        alert = "COUNTEREXAMPLE" if pattern in ("high_contrast_flickering_noise", "low_contrast_true_target") else ""
        fig.suptitle(f"{pattern} {alert} | seed={info.get('seed', payload.get('base_seed', 42))} | {H}×{W}, T={T}, D_head={Dh}\n"
                     "Definitions: TPR_gate=TP/(GT positives); FPR_gate=FP/(GT negatives); A_open_rate=mean(A); R is pre-update history; θ is per-step threshold; c is contrast mask")
        p = out / f"synthetic_{pattern}.png"; _save(fig, p); saved.append(p)
    # Aggregate seed records. P controls retain a min/max envelope across the declared permutation seeds.
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for pattern in patterns:
        group = [r for r in records if r.get("pattern") == pattern]
        for mode in ("B0_custom", "B1_lif", "B1_count", "H", "C", "P"):
            rs = [r for r in group if r.get("mode") == mode]
            if not rs: continue
            # One representative per family; report range across parameter grid, and permutation seeds for P.
            vals = [r for r in rs if (mode != "P" or float(r.get("b", 0)) == 2.0)]
            x = [r.get("FPR_gate") for r in vals if r.get("FPR_gate") is not None]
            y = [r.get("TPR_gate") for r in vals if r.get("TPR_gate") is not None]
            op = [r.get("A_open_rate") for r in vals if r.get("A_open_rate") is not None]
            if x and y:
                axes[0].scatter(np.mean(x), np.mean(y), label=f"{pattern[:10]}:{mode}", s=24,
                                color=COLORS.get(mode, "black"), alpha=.7)
                axes[0].errorbar(np.mean(x), np.mean(y),
                    xerr=[[np.mean(x)-min(x)],[max(x)-np.mean(x)]],
                    yerr=[[np.mean(y)-min(y)],[max(y)-np.mean(y)]],
                    fmt="none", color=COLORS.get(mode,"black"), alpha=.7, capsize=2)
                axes[1].errorbar(mode, np.mean(op), yerr=[[np.mean(op)-min(op)], [max(op)-np.mean(op)]], fmt="o",
                                 color=COLORS.get(mode, "black"), capsize=3)
    axes[0].set(xlabel="FPR_gate = FP / GT-negative positions", ylabel="TPR_gate = TP / GT-positive positions", title="Gate detection comparison")
    axes[0].legend(fontsize=6, ncol=2); axes[1].set(ylabel="A_open_rate = open gate positions / all positions", title="Gate open rate"); axes[1].tick_params(axis="x", rotation=30)
    fig.suptitle(f"Synthetic summary | seed={payload.get('base_seed', 42)} | {H}×{W}, T={T}, D_head={Dh}")
    p = out / "synthetic_summary.png"; _save(fig, p); saved.append(p)
    return saved


def _metadata_text(metadata: Mapping[str, Any]) -> str:
    fields = ("dataset", "dataset_disclaimer", "sample_count", "split", "cal_size", "eval_size", "model_variant",
              "T", "target_layers", "checkpoint_filename", "checkpoint_sha256", "git_head")
    values={k:metadata.get(k, 'n/a') for k in fields}
    values["checkpoint_sha256"]=metadata.get("checkpoint_sha256_short",str(values["checkpoint_sha256"])[:12])
    return " | ".join(f"{k}={values[k]}" for k in fields)


def _imshow_snapshot(ax: Any, value: Any, field: str) -> None:
    a=np.asarray(value)
    if field=="image":
        if a.ndim==3 and a.shape[0] in (1,3): a=np.moveaxis(a,0,-1)
        if a.ndim==3 and a.shape[-1] in (3,4):
            a=np.nan_to_num(a.astype(float),copy=False)
            lo,hi=float(a.min()),float(a.max())
            a=(a-lo)/(hi-lo) if hi>lo else np.zeros_like(a)
        ax.imshow(a.squeeze(),cmap="gray" if a.squeeze().ndim==2 else None)
        return
    a=np.squeeze(a)
    if a.ndim>1:
        # Reduce timestep/head axes to a deterministic spatial example view.
        while a.ndim>2: a=a.mean(axis=0)
        if a.ndim==2 and a.shape[0]!=a.shape[1] and int(np.sqrt(a.size))**2==a.size:
            a=a.reshape(int(np.sqrt(a.size)),-1)
    if a.ndim==1 and int(np.sqrt(a.size))**2==a.size: a=a.reshape(int(np.sqrt(a.size)),-1)
    ax.imshow(a,cmap="magma")


def _records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list): return payload
    return payload.get("results", payload.get("runs", [])) if isinstance(payload, dict) else []


def generate_frozen_plots(results_json_path: str | Path, mask_snapshots_npz_path: str | Path,
                          output_dir: str | Path, metadata: Mapping[str, Any]) -> list[Path]:
    """Generate the four required frozen evaluation figures with protocol metadata."""
    payload, snapshots = _json(results_json_path), _npz(mask_snapshots_npz_path)
    rows, out = _records(payload), Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    names = ("B0_original", "B0_custom", "B1_lif", "B1_count", "H", "C", "P")
    footer = _metadata_text(metadata)
    # Keep every registered threshold/b parameter visible; only P is seed-aggregated.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, layer in zip(axes, ("stage1", "all_qkta")):
        layer_rows=[r for r in rows if r.get("target_layers")==layer]
        b0=next((100*float(r.get("top1_acc",0)) for r in layer_rows if r.get("mode",r.get("gate"))=="B0_original"),None)
        for name in names:
            group=[r for r in layer_rows if r.get("mode",r.get("gate",""))==name]
            if not group: continue
            if name=="P":
                buckets={}
                for row in group: buckets.setdefault((layer,row.get("params",{}).get("b")),[]).append(row)
                for (_,b),seed_rows in buckets.items():
                    xs=[float(r.get("K_post_vs_B0",1)) for r in seed_rows]
                    ys=[100*float(r.get("top1_acc",0)) for r in seed_rows]
                    mx,my=float(np.mean(xs)),float(np.mean(ys))
                    ax.errorbar(mx,my,xerr=[[mx-min(xs)],[max(xs)-mx]],yerr=[[my-min(ys)],[max(ys)-my]],
                                fmt="o",color=COLORS[name],capsize=3,label=f"P b={b} (seed mean/range)")
                    if b0 is not None: ax.annotate(f"{my-b0:+.2f} pp",(mx,my),fontsize=7)
            else:
                for ri,row in enumerate(group):
                    x=float(row.get("K_post_vs_B0",1)); y=100*float(row.get("top1_acc",0))
                    params=row.get("params",{})
                    parameter=(f"θ₀={params['theta0']}" if name=="B1_lif" and "theta0" in params else
                               f"k={params['k_thresh']}" if name=="B1_count" and "k_thresh" in params else
                               f"b={params['b']}" if name in ("H","C") and "b" in params else "")
                    ax.scatter(x,y,color=COLORS.get(name,"black"),marker="o",label=name if ri==0 else None)
                    if b0 is not None: ax.annotate(f"{parameter} {y-b0:+.2f} pp".strip(),(x,y),fontsize=6)
        ax.set_title("Single-layer" if layer == "stage1" else "Multi-layer")
        ax.set_xlabel("post-gate K spikes / B0-original post-gate K spikes"); ax.set_ylabel("Top-1 Accuracy (%)"); ax.legend(fontsize=8)
    fig.suptitle("Accuracy and K-spike ratio\n" + footer, fontsize=8); fig.text(.01, .005, footer, fontsize=6, wrap=True)
    p=out/"accuracy_vs_postK_ratio.png"; _save(fig,p); saved.append(p)
    # Per timestep gate and post-K rates, grouped by gate and QKTA layer.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for row in rows:
        mode = row.get("mode", row.get("gate", ""))
        if mode not in ("B0_original", "B0_custom", "H", "C", "P"): continue
        steps = row.get("per_step_open_rate", []); post = row.get("per_step_post_k", [])
        layer = row.get("target_layers", "unknown")
        label = f"{mode} {layer}"
        if steps: axes[0].plot(range(len(steps)), steps, marker=".", label=label)
        if post: axes[1].plot(range(len(post)), post, marker=".", label=label)
    axes[0].set(xlabel="Timestep t", ylabel="A_open_rate", title="Per-step gate open rate")
    axes[1].set(xlabel="Timestep t", ylabel="K_post spikes", title="Per-step K_post")
    for ax in axes: ax.legend(fontsize=6, ncol=2)
    fig.suptitle("Per-step QKTA telemetry\n"+footer, fontsize=8); fig.text(.01,.005,footer,fontsize=6,wrap=True)
    p=out/"per_step_gate_rate.png"; _save(fig,p); saved.append(p)
    # Pre-fixed indices are retained in the snapshot artifact and never selected by performance.
    fig, axes = plt.subplots(4, 10, figsize=(22, 10)); fields=("image", "q_sum", "B0_A", "B0_suppressed", "H_A", "H_suppressed", "C_A", "C_suppressed", "P_A", "P_suppressed")
    keys=list(snapshots.files)
    sample_ids=[f"sample_{int(i):05d}" for i in metadata.get("mask_sample_indices",[])[:4]]
    if not sample_ids: sample_ids=sorted({k.split("__")[0] for k in keys})[:4]
    for ri, sid in enumerate(sample_ids):
        for ci, field in enumerate(fields):
            key=f"{sid}__{field}"; ax=axes[ri,ci]
            if key in snapshots:
                _imshow_snapshot(ax,snapshots[key],"image" if field=="image" else field)
            elif field.endswith("_suppressed"):
                prefix=field[:-len("_suppressed")]
                q_key=f"{sid}__q_sum"; a_key=f"{sid}__{prefix}_A"
                if q_key in snapshots and a_key in snapshots:
                    _imshow_snapshot(ax,np.asarray(snapshots[q_key])*(1-np.asarray(snapshots[a_key])),field)
            ax.set_title(f"{sid}: {field}",fontsize=8); ax.axis("off")
    for ri in range(len(sample_ids),4):
        for ax in axes[ri]: ax.axis("off")
    fig.suptitle("Fixed-index mask examples: B0 vs H vs C vs P; suppressed region = q_sum × (1−A)\n"+footer,fontsize=8); fig.text(.01,.005,footer,fontsize=6,wrap=True)
    p=out/"mask_examples.png"; _save(fig,p); saved.append(p)
    # Controls and potential failure cases are shown from all saved fixed samples.
    fig, (ax, failure_ax)=plt.subplots(1,2,figsize=(14,6),gridspec_kw={"width_ratios":[1.5,1]})
    for name in ("B1_lif","B1_count","H","C","P"):
        group=[r for r in rows if r.get("mode",r.get("gate"))==name]
        if not group: continue
        if name=="P":
            buckets={}
            for row in group: buckets.setdefault((row.get("target_layers"),row.get("params",{}).get("b")),[]).append(row)
            for (layer,b),seed_rows in buckets.items():
                xs=[float(r.get("K_post_vs_B0",1)) for r in seed_rows]; ys=[float(r.get("delta_top1_pp",0)) for r in seed_rows]
                mx,my=float(np.mean(xs)),float(np.mean(ys))
                ax.errorbar(mx,my,xerr=[[mx-min(xs)],[max(xs)-mx]],yerr=[[my-min(ys)],[max(ys)-my]],marker="o",label=f"P b={b} {layer}",capsize=3)
        else:
            for i,row in enumerate(group):
                x=float(row.get("K_post_vs_B0",1)); y=float(row.get("delta_top1_pp",0))
                params=row.get("params",{})
                parameter=(f" θ₀={params['theta0']}" if name=="B1_lif" and "theta0" in params else
                           f" k={params['k_thresh']}" if name=="B1_count" and "k_thresh" in params else
                           f" b={params['b']}" if name in ("H","C") and "b" in params else "")
                ax.scatter(x,y,color=COLORS.get(name,"black"),label=name if i==0 else None)
                ax.annotate(f"{name}{parameter}",(x,y),fontsize=6)
    ax.axhline(0,color="black",linewidth=.7); ax.set(xlabel="K_post / K_post_B0",ylabel="Accuracy delta vs B0 (pp)",title="Controls and failure cases")
    ax.legend();
    failures=[]
    for sid in sample_ids:
        pred_key=f"{sid}__pred"; target_key=f"{sid}__target"
        if pred_key in snapshots and target_key in snapshots and int(np.asarray(snapshots[pred_key]).item())!=int(np.asarray(snapshots[target_key]).item()): failures.append(sid)
    failure_ax.axis("off")
    if failures:
        shown=failures[:4]
        for i,sid in enumerate(shown):
            row=i//2; col=i%2
            thumb=failure_ax.inset_axes([col*.5,row*.5,.47,.43])
            _imshow_snapshot(thumb,snapshots[f"{sid}__image"],"image")
            pred=int(np.asarray(snapshots[sid+"__pred"]).item()); target=int(np.asarray(snapshots[sid+"__target"]).item())
            thumb.set_title(f"{sid}: pred {pred}, target {target}",fontsize=7); thumb.axis("off")
        failure_ax.set_title(f"Misclassified fixed samples ({len(failures)} total; showing {len(shown)})",fontsize=9)
    else:
        failure_ax.text(.5,.5,"No misclassified fixed samples among displayed indices",ha="center",va="center",wrap=True)
    fig.suptitle("B1 controls, H/C and 3-seed P permutation control\n"+footer,fontsize=8); fig.text(.01,.005,footer,fontsize=6,wrap=True)
    p=out/"controls_and_failure_cases.png"; _save(fig,p); saved.append(p)
    return saved


def main(argv: list[str] | None = None) -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="kind",required=True)
    p=sub.add_parser("synthetic"); p.add_argument("results_json"); p.add_argument("traces_npz"); p.add_argument("output_dir")
    p=sub.add_parser("frozen"); p.add_argument("results_json"); p.add_argument("snapshots_npz"); p.add_argument("output_dir"); p.add_argument("--metadata-json",required=True)
    a=parser.parse_args(argv)
    if a.kind=="synthetic": generate_synthetic_plots(a.results_json,a.traces_npz,a.output_dir)
    else: generate_frozen_plots(a.results_json,a.snapshots_npz,a.output_dir,_json(a.metadata_json))

if __name__=="__main__": main()

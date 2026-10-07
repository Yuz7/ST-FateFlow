from __future__ import annotations

import gc
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import anndata as ad
import numpy as np
import torch
from torch.utils.data import DataLoader

from common import ResourceProfiler, seed_everything


def _compute_fgwot(adata: ad.AnnData, cache_path: Path) -> None:
    from moscot.problems.spatiotemporal import SpatioTemporalProblem

    pair = (9.5, 11.5)
    counts = adata.obs["time"].value_counts()
    rank = min(200, int(counts.loc[pair[0]]), int(counts.loc[pair[1]]))
    problem = SpatioTemporalProblem(adata)
    problem = problem.prepare(
        time_key="time",
        spatial_key="spatial",
        policy="explicit",
        subset=[pair],
        joint_attr={"attr": "X"},
        cost={"xy": "sq_euclidean", "x": "sq_euclidean", "y": "sq_euclidean"},
    )
    problem = problem.solve(
        alpha=0.5,
        epsilon=0,
        rank=rank,
        tau_a=0.97,
        tau_b=0.93,
    )
    pi = np.asarray(problem.solutions[pair].transport_matrix, dtype=np.float64)
    pi = np.clip(pi, a_min=0.0, a_max=None)
    if float(pi.sum()) <= 0:
        raise ValueError("FUGWOT returned a transport plan with zero mass")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **{"transition_9.5_11.5": pi})


def run_stfateflow(
    adata: ad.AnnData,
    run_dir: Path,
    prediction_path: Path,
    profiler: ResourceProfiler,
    device: torch.device,
    seed: int,
    train_steps: int = 1000,
    use_early_stopping: bool = True,
) -> dict[str, Any]:
    if device.type == "cpu":
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from core.datasets.model_dataset import STFateFlowDataset, stfateflow_collate
    from core.models.STFateFlow import STFateFlow_Module
    from core.models.backbones.pc_transformer import PointCloudTransformer
    from core.training import EarlyStoppingConfig, train_stfateflow

    seed_everything(seed)
    torch.set_float32_matmul_precision("high")
    run_dir.mkdir(parents=True, exist_ok=True)
    adata.obsm["spatial"] = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    adata.obs["annotation"] = adata.obs["annotation"].astype("category")

    fgw_cache = run_dir / "fgwot_9p5_11p5.npz"
    with profiler.phase("fugwot_spatial_alignment_preprocessing"):
        _compute_fgwot(adata, fgw_cache)

    size_per_slice = 4096
    dataset = STFateFlowDataset(
        adata=adata,
        timepoint_column="time",
        cell_type_column="annotation",
        timepoints_ordered=[9.5, 11.5],
        seed=seed,
        size_per_slice=size_per_slice,
        ot_plan_cache_path=fgw_cache,
        ot_plan_normalize=True,
        test_timepoint=-1.0,
    )
    loader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=0,
        pin_memory=True,
        collate_fn=stfateflow_collate,
    )

    backbone = PointCloudTransformer(gene_dim=adata.n_vars, coord_dim=2)
    model = STFateFlow_Module(
        lambda_features=0.1,
        lambda_pos=1.0,
        lambda_cdm=10.0,
        backbone=backbone,
        num_steps=20,
        spatial_loss_type="mse",
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16
    with profiler.phase("training"):
        training_result = train_stfateflow(
            model,
            loader,
            optimizer,
            device=device,
            max_steps=train_steps,
            early_stopping=EarlyStoppingConfig() if use_early_stopping else None,
            amp_dtype=amp_dtype,
            use_amp=use_amp,
        )
        checkpoint = {
            "flow_state_dict": model.state_dict(),
            "n_genes": int(adata.n_vars),
            "timepoints_ordered": [9.5, 11.5],
            "config": {
                "size_per_slice": size_per_slice,
                "train_steps": int(train_steps),
                "completed_steps": int(training_result.completed_steps),
                "early_stopping": training_result.early_stopping,
                "lr": 3e-4,
                "lambda_features": 0.1,
                "lambda_pos": 1.0,
                "lambda_cdm": 10.0,
                "num_steps_ode": 20,
            },
        }
        checkpoint_path = run_dir / "stfateflow.pt"
        torch.save(checkpoint, checkpoint_path)

    with profiler.phase("inference"):
        model.eval()
        source = dataset.timepoint_pc[9.5]
        with torch.no_grad():
            output = model.sample(
                X_t0=source.x.unsqueeze(0).to(device),
                pos_t0=source.pos.unsqueeze(0).to(device),
                t_start=0.0,
                t_end=1.0,
                keep_trajectory=False,
                trajectory_device="cpu",
                inference_chunk_size=4096,
            )
        pred_x = output["x_traj"].reshape(-1, adata.n_vars).cpu().numpy()
        pred_pos = output["pos_traj"].reshape(-1, 2).cpu().numpy()
        prediction = ad.AnnData(X=np.asarray(pred_x, dtype=np.float32))
        prediction.obsm["pred_spatial"] = np.asarray(pred_pos, dtype=np.float32)
        prediction.obs["time"] = 10.5
        prediction.uns["benchmark_method"] = "stfateflow"
        prediction.uns["spatial_frame"] = "MOSTA input frame"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        prediction.write_h5ad(prediction_path)

    del model, optimizer, loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "training_steps": int(training_result.completed_steps),
        "training_steps_requested": int(train_steps),
        "training_steps_completed": int(training_result.completed_steps),
        "stopped_early": bool(training_result.stopped_early),
        "best_step": int(training_result.best_step),
        "best_smoothed_loss": float(training_result.best_smoothed_loss),
        "size_per_slice": size_per_slice,
        "prediction_path": str(prediction_path),
        "checkpoint_path": str(checkpoint_path),
        "fgwot_cache_path": str(fgw_cache),
        "n_predicted_cells": int(len(pred_x)),
    }

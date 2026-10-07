from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import torch

from common import ResourceProfiler, dense_float32, seed_everything


def _import_stvcr(stvcr_root: Path):
    source = stvcr_root / "src"
    if not source.exists():
        raise FileNotFoundError(f"stVCR source directory not found: {source}")
    sys.path.insert(0, str(source))
    from stvcr.downstream.utils import evolution_forward_sim_rgb_data
    from stvcr.preprocessing.autoencoder import ae_dim_reduction
    from stvcr.preprocessing.pp import pp_init
    from stvcr.training.train import train_stvcr

    return ae_dim_reduction, pp_init, train_stvcr, evolution_forward_sim_rgb_data


def _run_inference(
    adata: ad.AnnData,
    model_path: Path,
    ae_model_path: Path,
    interpolation_time: float,
    spatial_dim: int,
    device: torch.device,
    evolution_forward,
) -> tuple[np.ndarray, np.ndarray]:
    time_adjust = float(np.min(adata.obs["time"]))
    interpolation_time_input = interpolation_time - time_adjust
    initial_time = float(
        np.unique(adata.obs.loc[adata.obs["time_input"] < interpolation_time_input, "time_input"])[-1]
    )
    initial_mask = np.isclose(np.asarray(adata.obs["time_input"], dtype=float), initial_time)
    initial = torch.cat(
        (
            torch.from_numpy(np.asarray(adata.obsm["X_spatial_aligned"][initial_mask])),
            torch.from_numpy(np.asarray(adata.obsm["X_gene_input"][initial_mask])),
        ),
        dim=1,
    ).float()

    # The official stVCR downstream integrator is CPU based.
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    spatial_series, expression_series, _, _, _ = evolution_forward(
        initial,
        model,
        initial_time,
        interpolation_time_input,
        spatial_dim=spatial_dim,
        delta_t=0.1,
    )
    latent = torch.as_tensor(expression_series[-1], dtype=torch.float32)
    autoencoder = torch.load(ae_model_path, map_location="cpu", weights_only=False)
    with torch.no_grad():
        expression = autoencoder.decode_mlp1(latent).cpu().numpy()

    aligned_spatial = np.asarray(spatial_series[-1], dtype=np.float32)
    scale = float(adata.uns.get("spatial_scale_factor", 1.0))
    source_time = float(initial_time + time_adjust)
    source_mask = np.isclose(np.asarray(adata.obs["time"], dtype=float), source_time)
    source_center = np.asarray(
        adata.uns.get("benchmark_source_center", np.zeros(spatial_dim)), dtype=np.float32
    )
    if not source_mask.any():
        raise ValueError(f"No source cells found at time {source_time}")

    # pp_init centers all slices and applies one global scalar. Its predicted
    # trajectory is in the source orientation, so invert the scalar and center.
    spatial = aligned_spatial * scale + source_center
    del model, autoencoder, latent
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.asarray(expression, dtype=np.float32), np.asarray(spatial, dtype=np.float32)


def run_stvcr(
    adata: ad.AnnData,
    mode: str,
    run_dir: Path,
    prediction_path: Path,
    profiler: ResourceProfiler,
    device: torch.device,
    stvcr_root: Path,
    seed: int,
    epochs: int,
    ae_epochs: int,
    alignment_iterations: int,
) -> dict[str, Any]:
    if mode not in {"default", "all-cells"}:
        raise ValueError(f"Unknown stVCR mode: {mode}")
    ae_dim_reduction, pp_init, train_stvcr, evolution_forward = _import_stvcr(stvcr_root)
    seed_everything(seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Existing latent representations may have been fitted on the held-out
    # time point. Refit the stVCR autoencoder on training cells only.
    for key in ("X_ae", "X_gene_input", "X_spatial_input", "X_spatial_aligned"):
        if key in adata.obsm:
            del adata.obsm[key]
    source_mask = np.isclose(np.asarray(adata.obs["time"], dtype=float), 9.5)
    adata.uns["benchmark_source_center"] = np.asarray(
        adata.obsm["spatial"][source_mask], dtype=np.float32
    ).mean(axis=0)

    ae_path = run_dir / "autoencoder.pt"
    model_path = run_dir / "stvcr_model.pt"
    rigid_path = run_dir / "rigid_model.pt"

    with profiler.phase("autoencoder_preprocessing"):
        ae_dim_reduction(
            adata,
            ae_model_save_path=str(ae_path),
            gene_expression_key=None,
            z_dims=10,
            device=device,
            learning_rate=1e-3,
            n_epochs=ae_epochs,
            batch_size=1000,
            early_stop=30,
            valid_ratio=0.1,
            seed=seed,
        )

    with profiler.phase("spatial_alignment_preprocessing"):
        pp_init(
            adata,
            spatial_key="spatial",
            gene_redunction_key="X_ae",
            time_key="time",
            use_initial_alignment=True,
            alpha=0.002,
            down_sampling_number=5000,
            iter_num=alignment_iterations,
            normlize_spatial_coordinate=True,
        )

    counts = [
        int(np.isclose(np.asarray(adata.obs["time"], dtype=float), t).sum())
        for t in sorted(np.unique(np.asarray(adata.obs["time"], dtype=float)))
    ]
    num_samples: int | list[int] = 1000 if mode == "default" else counts
    config = {
        "learning_rate": 1e-3,
        "learning_rate_rigid": 1e-4,
        "n_epochs": int(epochs),
        "num_samples": num_samples,
        "lambda_match": 4e5,
        "alpha_exp": 0.01,
        "alpha_gro": 0.0002,
        "kappa_exp": 0.02,
        "kappa_gro": 0.1,
    }

    with profiler.phase("training"):
        train_stvcr(
            adata=adata,
            model_path=str(model_path),
            rigid_transformation_path=str(rigid_path),
            config=config,
            use_gene=True,
            use_spatial=True,
            use_growth=True,
            use_alignment=True,
            device=device,
        )

    with profiler.phase("inference"):
        pred_x, pred_pos = _run_inference(
            adata=adata,
            model_path=model_path,
            ae_model_path=ae_path,
            interpolation_time=10.5,
            spatial_dim=int(adata.obsm["spatial"].shape[1]),
            device=device,
            evolution_forward=evolution_forward,
        )
        prediction = ad.AnnData(X=pred_x)
        prediction.obsm["pred_spatial"] = pred_pos
        prediction.obs["time"] = 10.5
        prediction.uns["benchmark_method"] = f"stvcr-{mode}"
        prediction.uns["spatial_frame"] = "MOSTA input frame"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        prediction.write_h5ad(prediction_path)

    effective_samples = [min(1000, count) for count in counts] if mode == "default" else counts
    return {
        "training_epochs": int(epochs),
        "autoencoder_epochs_max": int(ae_epochs),
        "alignment_iterations": int(alignment_iterations),
        "requested_num_samples": num_samples,
        "effective_num_samples": effective_samples,
        "prediction_path": str(prediction_path),
        "model_path": str(model_path),
        "n_predicted_cells": int(len(pred_x)),
        "inference_device": "cpu (official stVCR downstream integrator)",
    }

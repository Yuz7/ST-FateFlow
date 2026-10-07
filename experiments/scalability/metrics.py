from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import ot
import torch
from scipy.spatial.distance import cdist

from common import TEST_TIME, dense_float32


def _sample(x: np.ndarray, n: int, seed: int) -> np.ndarray:
    if len(x) <= n:
        return x
    rng = np.random.default_rng(seed)
    return x[rng.choice(len(x), size=n, replace=False)]


def _median_gt_sigma(gt: np.ndarray, max_points: int, seed: int) -> float:
    sampled = _sample(gt, max_points, seed)
    distances = cdist(sampled, sampled, metric="euclidean")
    nonzero = distances[distances > 0]
    if not len(nonzero):
        return 1.0
    return max(float(np.median(nonzero)), 1e-8)


def _rbf_mmd(x: np.ndarray, y: np.ndarray, sigma: float) -> float:
    gamma = 1.0 / (2.0 * sigma**2)
    k_xx = np.exp(-gamma * cdist(x, x, metric="sqeuclidean")).mean()
    k_yy = np.exp(-gamma * cdist(y, y, metric="sqeuclidean")).mean()
    k_xy = np.exp(-gamma * cdist(x, y, metric="sqeuclidean")).mean()
    return float(k_xx + k_yy - 2.0 * k_xy)


def _transport_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    scale: float,
    sinkhorn_reg: float,
) -> tuple[float, float]:
    cost = cdist(pred, gt, metric="euclidean") / max(scale, 1e-8)
    a = ot.unif(len(pred))
    b = ot.unif(len(gt))
    w1 = float(ot.emd2(a, b, cost, numItermax=1_000_000))
    sinkhorn = float(ot.sinkhorn2(a, b, cost, reg=sinkhorn_reg))
    return w1, sinkhorn


def _chunked_min_distances(x: np.ndarray, y: np.ndarray, chunk_size: int = 2048) -> np.ndarray:
    y_t = torch.as_tensor(y, dtype=torch.float32)
    chunks = []
    for start in range(0, len(x), chunk_size):
        x_t = torch.as_tensor(x[start : start + chunk_size], dtype=torch.float32)
        chunks.append(torch.cdist(x_t, y_t).min(dim=1).values.numpy())
    return np.concatenate(chunks)


def evaluate_prediction(
    prediction_path: Path,
    ground_truth: ad.AnnData,
    method: str,
    fraction: float,
    seed: int,
    expression_n: int = 1000,
    spatial_n: int = 3000,
    kernel_n: int = 1000,
    sinkhorn_reg: float = 0.05,
) -> dict[str, Any]:
    pred = ad.read_h5ad(prediction_path)
    target_mask = np.isclose(np.asarray(ground_truth.obs["time"], dtype=float), TEST_TIME)
    gt = ground_truth[target_mask]

    pred_x_all = dense_float32(pred.X)
    gt_x_all = dense_float32(gt.X)
    pred_x = _sample(pred_x_all, expression_n, seed + 11)
    gt_x = _sample(gt_x_all, expression_n, seed + 13)

    gt_scale_sample = _sample(gt_x_all, kernel_n, seed + 17)
    gt_pairwise = cdist(gt_scale_sample, gt_scale_sample, metric="euclidean")
    gt_scale = max(float(gt_pairwise.mean()), 1e-8)
    sigma = _median_gt_sigma(gt_x_all, kernel_n, seed + 19)
    rna_w1, rna_sinkhorn = _transport_metrics(pred_x, gt_x, gt_scale, sinkhorn_reg)

    pred_pos_all = np.asarray(pred.obsm["pred_spatial"], dtype=np.float32)
    gt_pos_all = np.asarray(gt.obsm["spatial"], dtype=np.float32)
    pred_pos = _sample(pred_pos_all, spatial_n, seed + 23)
    gt_pos = _sample(gt_pos_all, spatial_n, seed + 29)
    gt_pos_scale_sample = _sample(gt_pos_all, spatial_n, seed + 31)
    spatial_scale = max(
        float(cdist(gt_pos_scale_sample, gt_pos_scale_sample).mean()), 1e-8
    )
    spatial_w1, _ = _transport_metrics(pred_pos, gt_pos, spatial_scale, sinkhorn_reg)
    pred_to_gt = _chunked_min_distances(pred_pos, gt_pos)
    gt_to_pred = _chunked_min_distances(gt_pos, pred_pos)

    return {
        "run_id": prediction_path.stem,
        "method": method,
        "fraction": float(fraction),
        "seed": int(seed),
        "n_pred": int(pred.n_obs),
        "n_ground_truth": int(gt.n_obs),
        "rna_pearson_r": float(
            np.corrcoef(pred_x_all.mean(axis=0), gt_x_all.mean(axis=0))[0, 1]
        ),
        "rna_mmd_rbf": _rbf_mmd(pred_x, gt_x, sigma),
        "rna_wasserstein_1": rna_w1,
        "rna_sinkhorn": rna_sinkhorn,
        "rna_gt_scale": gt_scale,
        "rna_rbf_sigma": sigma,
        "spatial_chamfer_l2": float(pred_to_gt.mean() + gt_to_pred.mean()),
        "spatial_wasserstein_1": spatial_w1,
        "spatial_hausdorff95": float(
            max(np.quantile(pred_to_gt, 0.95), np.quantile(gt_to_pred, 0.95))
        ),
        "expression_eval_n": int(expression_n),
        "spatial_eval_n": int(spatial_n),
    }


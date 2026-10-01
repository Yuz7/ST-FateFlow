from __future__ import annotations

import math
from functools import partial
from typing import Any

import numpy as np
import torch


def as_float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def canonical_time_order(values) -> list[Any]:
    vals = list(values)
    as_float = [as_float_or_none(v) for v in vals]
    if all(v is not None for v in as_float):
        return [x for _, x in sorted(zip(as_float, vals), key=lambda z: z[0])]
    return sorted(vals, key=lambda x: str(x))


def find_time_key(keys, query):
    query_f = as_float_or_none(query)
    if query in keys:
        return query
    for key in keys:
        key_f = as_float_or_none(key)
        if query_f is not None and key_f is not None and abs(key_f - query_f) < 1e-9:
            return key
    query_str = str(query)
    for key in keys:
        if str(key) == query_str:
            return key
    raise KeyError(f"Timepoint {query} not found in {keys}")


def nn_of_x_in_y(x: torch.Tensor, y: torch.Tensor, chunk_size: int = 5000):
    idxs = []
    dists = []

    n_chunks = math.ceil(x.size(0) / chunk_size)
    for i in range(n_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        d = torch.cdist(x[start:end], y)
        min_dist, min_idx = torch.min(d, dim=-1)
        idxs.append(min_idx)
        dists.append(min_dist)

    return torch.cat(idxs), torch.cat(dists)


def bidirectional_nn_metrics(x: torch.Tensor, y: torch.Tensor, chunk_size: int = 5000):
    idx_x_to_y, dist_x_to_y = nn_of_x_in_y(x, y, chunk_size=chunk_size)
    idx_y_to_x, dist_y_to_x = nn_of_x_in_y(y, x, chunk_size=chunk_size)
    cd_l2 = dist_x_to_y.mean() + dist_y_to_x.mean()
    cd_rmse = torch.sqrt((dist_x_to_y**2).mean()) + torch.sqrt((dist_y_to_x**2).mean())
    cd_sq = (dist_x_to_y**2).mean() + (dist_y_to_x**2).mean()
    hd95 = torch.maximum(torch.quantile(dist_x_to_y, 0.95), torch.quantile(dist_y_to_x, 0.95))
    hd = torch.maximum(dist_x_to_y.max(), dist_y_to_x.max())
    return {
        "idx_x_to_y": idx_x_to_y,
        "idx_y_to_x": idx_y_to_x,
        "dist_x_to_y": dist_x_to_y,
        "dist_y_to_x": dist_y_to_x,
        "cd_l2": cd_l2,
        "cd_rmse": cd_rmse,
        "cd_sq": cd_sq,
        "hd95": hd95,
        "hd": hd,
    }


def wasserstein_distance(
    x0: torch.Tensor,
    x1: torch.Tensor,
    method: str | None = None,
    reg: float = 0.05,
    power: int = 1,
    **kwargs,
) -> float:
    if power not in {1, 2}:
        raise ValueError(f"`power` must be 1 or 2, got {power}.")

    try:
        import ot as pot
    except ImportError as exc:
        raise ImportError("`wasserstein_distance` requires POT. Install it with `pip install POT`.") from exc

    if method == "exact" or method is None:
        ot_fn = pot.emd2
    elif method == "sinkhorn":
        ot_fn = partial(pot.sinkhorn2, reg=reg)
    else:
        raise ValueError(f"Unknown method: {method}")

    a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
    if x0.dim() > 2:
        x0 = x0.reshape(x0.shape[0], -1)
    if x1.dim() > 2:
        x1 = x1.reshape(x1.shape[0], -1)
    cost = torch.cdist(x0, x1)
    if power == 2:
        cost = cost**2
    ret = ot_fn(a, b, cost.detach().cpu().numpy(), numItermax=1e7, **kwargs)
    if power == 2:
        ret = math.sqrt(ret)
    return float(ret)

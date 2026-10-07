"""Rigid time-slice registration initialized by FUGWOT correspondences.

This module adapts stVCR's coupling-weighted generalized Procrustes
preprocessing to STFateFlow.  The cross-slice coupling is estimated with
Moscot's fused unbalanced Gromov-Wasserstein OT (FUGWOT), rather than balanced
OT over a coordinate-dependent cross-slice cost.

The registration coupling is intentionally separate from the full-resolution
FUGWOT plans used later by flow matching.  Registration may use a deterministic
subsample because its only output is one rigid transform per time point.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import anndata as ad
import numpy as np


PlanSolver = Callable[
    [ad.AnnData, ad.AnnData, str, str | None, "FUGWOTConfig"], np.ndarray
]


@dataclass(frozen=True)
class FUGWOTConfig:
    """Moscot parameters used only to infer rigid-registration correspondences."""

    alpha: float = 0.5
    epsilon: float = 0.0
    rank: int = 200
    tau_a: float = 0.97
    tau_b: float = 0.93
    device: str = "gpu"
    seed: int = 42


@dataclass
class RigidFUGWOTResult:
    """Registered data plus auditable downsampled registration couplings."""

    adata: ad.AnnData
    plans: dict[tuple[Any, Any], np.ndarray] = field(default_factory=dict)
    source_indices: dict[tuple[Any, Any], np.ndarray] = field(default_factory=dict)
    target_indices: dict[tuple[Any, Any], np.ndarray] = field(default_factory=dict)


def weighted_procrustes(
    source: np.ndarray,
    target: np.ndarray,
    coupling: np.ndarray,
    *,
    allow_reflection: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit ``target @ rotation + translation`` to ``source``.

    The objective is weighted by a non-negative source-by-target transport
    plan.  This is the row-vector equivalent of the generalized Procrustes
    update used by stVCR/PASTE.
    """

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    coupling = np.asarray(coupling, dtype=np.float64)
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("source and target coordinates must be 2D arrays")
    if source.shape[1] != target.shape[1]:
        raise ValueError("source and target must have the same spatial dimension")
    if coupling.shape != (source.shape[0], target.shape[0]):
        raise ValueError(
            f"coupling shape {coupling.shape} does not match "
            f"({source.shape[0]}, {target.shape[0]})"
        )
    if not np.isfinite(coupling).all() or np.any(coupling < 0):
        raise ValueError("coupling must contain finite, non-negative values")
    mass = float(coupling.sum())
    if mass <= 0:
        raise ValueError("coupling must have positive total mass")

    coupling = coupling / mass
    source_center = coupling.sum(axis=1) @ source
    target_center = coupling.sum(axis=0) @ target
    source_centered = source - source_center
    target_centered = target - target_center
    cross_covariance = target_centered.T @ coupling.T @ source_centered
    left, _, right_t = np.linalg.svd(cross_covariance, full_matrices=False)
    rotation = left @ right_t
    if not allow_reflection and np.linalg.det(rotation) < 0:
        left[:, -1] *= -1.0
        rotation = left @ right_t
    translation = source_center - target_center @ rotation
    aligned = target @ rotation + translation
    return aligned, rotation, translation


def _deterministic_subsample_indices(
    n_source: int,
    n_target: int,
    total_budget: int | None,
    spatial_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    if n_source == 0 or n_target == 0:
        raise ValueError("cannot register an empty time point")
    if total_budget is None or n_source + n_target <= total_budget:
        return np.arange(n_source), np.arange(n_target)
    minimum = spatial_dim + 1
    if total_budget < 2 * minimum:
        raise ValueError(
            f"downsample_total must be at least {2 * minimum} for {spatial_dim}D registration"
        )
    source_budget = int(round(total_budget * n_source / (n_source + n_target)))
    source_budget = min(n_source, max(minimum, source_budget))
    target_budget = min(n_target, max(minimum, total_budget - source_budget))
    if source_budget + target_budget > total_budget:
        source_budget = max(minimum, total_budget - target_budget)
    source_idx = np.linspace(0, n_source - 1, source_budget, dtype=np.int64)
    target_idx = np.linspace(0, n_target - 1, target_budget, dtype=np.int64)
    return np.unique(source_idx), np.unique(target_idx)


def _solve_fugwot_pair(
    source: ad.AnnData,
    target: ad.AnnData,
    spatial_key: str,
    joint_attr_key: str | None,
    config: FUGWOTConfig,
) -> np.ndarray:
    """Compute one source-by-target registration plan with Moscot."""

    try:
        import jax
        from moscot.problems.spatiotemporal import SpatioTemporalProblem
    except ImportError as exc:  # pragma: no cover - depends on optional stack
        raise ImportError(
            "Rigid FUGWOT registration requires moscot, ott-jax, and jax."
        ) from exc

    source = source.copy()
    target = target.copy()
    source.obs["_stfateflow_registration_time"] = 0.0
    target.obs["_stfateflow_registration_time"] = 1.0
    pair = ad.concat(
        [source, target],
        axis=0,
        join="inner",
        merge="same",
        index_unique="-stfateflow-rigid",
    )
    if joint_attr_key is None or joint_attr_key == "X":
        joint_attr: dict[str, str] = {"attr": "X"}
    else:
        if joint_attr_key not in pair.obsm:
            raise KeyError(f"joint feature key {joint_attr_key!r} is absent from adata.obsm")
        joint_attr = {"attr": "obsm", "key": joint_attr_key}

    problem = SpatioTemporalProblem(pair).prepare(
        time_key="_stfateflow_registration_time",
        spatial_key=spatial_key,
        policy="sequential",
        joint_attr=joint_attr,
        cost={"xy": "sq_euclidean", "x": "sq_euclidean", "y": "sq_euclidean"},
    )
    problem = problem.solve(
        alpha=float(config.alpha),
        epsilon=float(config.epsilon),
        rank=int(config.rank),
        tau_a=float(config.tau_a),
        tau_b=float(config.tau_b),
        device=config.device,
        initializer_kwargs={"rng": jax.random.PRNGKey(int(config.seed))},
    )
    plan = np.asarray(problem.solutions[(0.0, 1.0)].transport_matrix, dtype=np.float64)
    return np.clip(plan, 0.0, None)


def rigid_register_time_series(
    adata: ad.AnnData,
    *,
    time_key: str = "time",
    spatial_key: str = "spatial",
    output_spatial_key: str = "X_spatial_input",
    joint_attr_key: str | None = None,
    timepoints: Sequence[Any] | None = None,
    downsample_total: int | None = 5000,
    normalize_coordinates: bool = True,
    allow_reflection: bool = False,
    config: FUGWOTConfig | None = None,
    copy: bool = True,
    plan_solver: PlanSolver | None = None,
) -> RigidFUGWOTResult:
    """Sequentially rigid-register time slices using FUGWOT correspondences.

    The earliest time point is centered and used as the fixed reference.  Each
    later slice is aligned to the preceding already-aligned slice.  Only the
    resulting global rotation/translation is applied to all cells; FUGWOT does
    not warp individual coordinates.

    Notes
    -----
    The returned plans are downsampled *registration* plans.  Recompute the
    full-resolution FUGWOT plans on ``output_spatial_key`` before STFateFlow
    training.
    """

    if time_key not in adata.obs:
        raise KeyError(f"time key {time_key!r} is absent from adata.obs")
    if spatial_key not in adata.obsm:
        raise KeyError(f"spatial key {spatial_key!r} is absent from adata.obsm")
    result_adata = adata.copy() if copy else adata
    raw_coordinates = np.asarray(result_adata.obsm[spatial_key], dtype=np.float64)
    if raw_coordinates.ndim != 2 or raw_coordinates.shape[0] != result_adata.n_obs:
        raise ValueError("spatial coordinates must have shape (n_obs, spatial_dim)")
    if not np.isfinite(raw_coordinates).all():
        raise ValueError("spatial coordinates contain non-finite values")

    observed = list(result_adata.obs[time_key].unique())
    ordered = list(timepoints) if timepoints is not None else sorted(observed)
    if len(ordered) < 2:
        raise ValueError("at least two time points are required for registration")
    missing = [value for value in ordered if value not in observed]
    if missing:
        raise ValueError(f"requested time points are absent from the data: {missing}")

    cfg = config or FUGWOTConfig()
    solve_plan = plan_solver or _solve_fugwot_pair
    aligned_coordinates = raw_coordinates.copy()
    first_mask = np.asarray(result_adata.obs[time_key] == ordered[0])
    first_center = aligned_coordinates[first_mask].mean(axis=0)
    aligned_coordinates[first_mask] -= first_center

    plans: dict[tuple[Any, Any], np.ndarray] = {}
    source_indices: dict[tuple[Any, Any], np.ndarray] = {}
    target_indices: dict[tuple[Any, Any], np.ndarray] = {}
    transforms: dict[str, dict[str, Any]] = {
        "step_0": {
            "timepoint": str(ordered[0]),
            "reference_timepoint": "",
            "rotation": np.eye(raw_coordinates.shape[1]),
            "translation": -first_center,
            "determinant": 1.0,
        }
    }

    for source_time, target_time in zip(ordered[:-1], ordered[1:], strict=True):
        source_mask = np.asarray(result_adata.obs[time_key] == source_time)
        target_mask = np.asarray(result_adata.obs[time_key] == target_time)
        source_all = aligned_coordinates[source_mask]
        target_all = raw_coordinates[target_mask]
        src_idx, tgt_idx = _deterministic_subsample_indices(
            len(source_all), len(target_all), downsample_total, raw_coordinates.shape[1]
        )

        source_slice = result_adata[source_mask].copy()[src_idx].copy()
        target_slice = result_adata[target_mask].copy()[tgt_idx].copy()
        source_slice.obsm["_stfateflow_rigid_working"] = source_all[src_idx]
        target_slice.obsm["_stfateflow_rigid_working"] = target_all[tgt_idx]
        plan = solve_plan(
            source_slice,
            target_slice,
            "_stfateflow_rigid_working",
            joint_attr_key,
            cfg,
        )
        _, rotation, translation = weighted_procrustes(
            source_all[src_idx],
            target_all[tgt_idx],
            plan,
            allow_reflection=allow_reflection,
        )
        aligned_coordinates[target_mask] = target_all @ rotation + translation
        transition = (source_time, target_time)
        plans[transition] = plan
        source_indices[transition] = src_idx
        target_indices[transition] = tgt_idx
        transforms[f"step_{len(transforms)}"] = {
            "timepoint": str(target_time),
            "reference_timepoint": str(source_time),
            "rotation": rotation,
            "translation": translation,
            "determinant": float(np.linalg.det(rotation)),
            "registration_plan_shape": np.asarray(plan.shape, dtype=np.int64),
            "registration_plan_mass": float(plan.sum()),
        }

    scale_factor = float(np.max(np.abs(aligned_coordinates)))
    if normalize_coordinates:
        if scale_factor <= 0:
            raise ValueError("cannot normalize zero-valued spatial coordinates")
        aligned_coordinates = aligned_coordinates / scale_factor
    result_adata.obsm[output_spatial_key] = aligned_coordinates.astype(np.float32)
    result_adata.uns["stfateflow_rigid_fugwot"] = {
        "method": "sequential FUGWOT-weighted rigid Procrustes",
        "time_key": time_key,
        "input_spatial_key": spatial_key,
        "output_spatial_key": output_spatial_key,
        "timepoints": [str(value) for value in ordered],
        "joint_attr_key": "X" if joint_attr_key is None else joint_attr_key,
        "downsample_total": -1 if downsample_total is None else downsample_total,
        "normalize_coordinates": normalize_coordinates,
        "spatial_scale_factor": scale_factor,
        "allow_reflection": allow_reflection,
        "fugwot": {
            "alpha": cfg.alpha,
            "epsilon": cfg.epsilon,
            "rank": cfg.rank,
            "tau_a": cfg.tau_a,
            "tau_b": cfg.tau_b,
            "device": cfg.device,
            "seed": cfg.seed,
        },
        "transforms": transforms,
        "note": (
            "The stored couplings are downsampled registration plans. Recompute "
            "full-resolution FUGWOT plans on the aligned coordinates for flow training."
        ),
    }
    return RigidFUGWOTResult(
        adata=result_adata,
        plans=plans,
        source_indices=source_indices,
        target_indices=target_indices,
    )


__all__ = [
    "FUGWOTConfig",
    "RigidFUGWOTResult",
    "rigid_register_time_series",
    "weighted_procrustes",
]

#!/usr/bin/env python3
"""Run FUGWOT-weighted rigid registration on a time-resolved H5AD file."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/stfateflow-matplotlib")

import anndata as ad
import numpy as np


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.preprocessing.rigid_fugwot import FUGWOTConfig, rigid_register_time_series


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sequentially rigid-register time slices using Moscot FUGWOT "
            "couplings and coupling-weighted Procrustes transforms."
        )
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--time-key", default="time")
    parser.add_argument("--spatial-key", default="spatial")
    parser.add_argument("--output-spatial-key", default="X_spatial_input")
    parser.add_argument(
        "--joint-attr-key",
        default=None,
        help="AnnData.obsm expression feature key; omit to use adata.X.",
    )
    parser.add_argument("--timepoints", nargs="+", default=None)
    parser.add_argument("--downsample-total", type=int, default=5000)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--rank", type=int, default=200)
    parser.add_argument("--tau-a", type=float, default=0.97)
    parser.add_argument("--tau-b", type=float, default=0.93)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-reflection", action="store_true")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--registration-plans", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=None)
    return parser.parse_args()


def _coerce_timepoints(values: list[str] | None, observed: np.ndarray) -> list[object] | None:
    if values is None:
        return None
    if np.issubdtype(observed.dtype, np.number):
        return [float(value) for value in values]
    return values


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    args = parse_args()
    if args.device == "cpu":
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
    adata = ad.read_h5ad(args.input)
    timepoints = _coerce_timepoints(
        args.timepoints, np.asarray(adata.obs[args.time_key])
    )
    config = FUGWOTConfig(
        alpha=args.alpha,
        epsilon=args.epsilon,
        rank=args.rank,
        tau_a=args.tau_a,
        tau_b=args.tau_b,
        device=args.device,
        seed=args.seed,
    )
    result = rigid_register_time_series(
        adata,
        time_key=args.time_key,
        spatial_key=args.spatial_key,
        output_spatial_key=args.output_spatial_key,
        joint_attr_key=args.joint_attr_key,
        timepoints=timepoints,
        downsample_total=args.downsample_total,
        normalize_coordinates=not args.no_normalize,
        allow_reflection=args.allow_reflection,
        config=config,
        copy=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.adata.write_h5ad(args.output, compression="gzip")

    plan_path = args.registration_plans or args.output.with_suffix(".rigid_plans.npz")
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for index, transition in enumerate(result.plans):
        payload[f"plan_{index}"] = result.plans[transition]
        payload[f"source_indices_{index}"] = result.source_indices[transition]
        payload[f"target_indices_{index}"] = result.target_indices[transition]
    np.savez_compressed(plan_path, **payload)

    metadata_path = args.metadata or args.output.with_suffix(".rigid_metadata.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = _jsonable(dict(result.adata.uns["stfateflow_rigid_fugwot"]))
    metadata["input"] = str(args.input)
    metadata["output"] = str(args.output)
    metadata["registration_plans"] = str(plan_path)
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()

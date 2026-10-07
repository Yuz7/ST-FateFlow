from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import anndata as ad
import pandas as pd
import torch

from common import (
    DEFAULT_DATA,
    DEFAULT_STVCR_ROOT,
    RESULTS_ROOT,
    ResourceProfiler,
    base_profile,
    load_train_subset,
    run_id,
    write_json,
)
from metrics import evaluate_prediction


METHODS = ("stfateflow", "stvcr-default", "stvcr-all-cells")


def _device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {value}")
    return device


def run(args: argparse.Namespace) -> None:
    data_path = Path(args.data)
    rid = run_id(args.method, args.fraction, args.seed)
    profile_path = RESULTS_ROOT / "profiles" / f"{rid}.json"
    prediction_path = RESULTS_ROOT / "predictions" / f"{rid}.h5ad"
    model_dir = RESULTS_ROOT / "models" / rid

    train, subset_metadata = load_train_subset(
        data_path=data_path,
        fraction=args.fraction,
        seed=args.seed,
    )
    profile = base_profile(
        method=args.method,
        fraction=args.fraction,
        seed=args.seed,
        data_path=data_path,
        subset_metadata=subset_metadata,
    )
    write_json(profile_path, profile)
    device = _device(args.device)
    profiler = ResourceProfiler(device)

    try:
        if args.method == "stfateflow":
            from run_stfateflow import run_stfateflow

            details = run_stfateflow(
                adata=train,
                run_dir=model_dir,
                prediction_path=prediction_path,
                profiler=profiler,
                device=device,
                seed=args.seed,
                train_steps=args.stfateflow_steps,
                use_early_stopping=args.stfateflow_early_stopping,
            )
        else:
            from run_stvcr import run_stvcr

            mode = args.method.removeprefix("stvcr-")
            details = run_stvcr(
                adata=train,
                mode=mode,
                run_dir=model_dir,
                prediction_path=prediction_path,
                profiler=profiler,
                device=device,
                stvcr_root=Path(args.stvcr_root),
                seed=args.seed,
                epochs=args.stvcr_epochs,
                ae_epochs=args.stvcr_ae_epochs,
                alignment_iterations=args.stvcr_alignment_iterations,
            )
        profile.update(
            {
                "status": "complete",
                "phases": profiler.phases,
                "details": details,
            }
        )
    except BaseException as exc:
        profile.update(
            {
                "status": "failed",
                "phases": profiler.phases,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        write_json(profile_path, profile)
        raise
    write_json(profile_path, profile)
    print(f"Completed {rid}")
    print(f"Profile: {profile_path}")
    print(f"Prediction: {prediction_path}")


def _resource_rows(profile_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(profile_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        base = {
            "run_id": payload.get("run_id", path.stem),
            "method": payload.get("method"),
            "fraction": payload.get("fraction"),
            "seed": payload.get("seed"),
            "status": payload.get("status"),
            "error_type": payload.get("error_type"),
            "error": payload.get("error"),
            "n_train_total": payload.get("subset", {}).get("n_train_total"),
            "n_train_9p5": payload.get("subset", {}).get("counts", {}).get("9.5"),
            "n_train_11p5": payload.get("subset", {}).get("counts", {}).get("11.5"),
        }
        phases = payload.get("phases", {})
        base["total_profiled_seconds"] = sum(
            float(stats.get("wall_seconds", 0.0)) for stats in phases.values()
        )
        for phase, stats in phases.items():
            for key, value in stats.items():
                base[f"{phase}__{key}"] = value
        rows.append(base)
    return rows


def evaluate(args: argparse.Namespace) -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    ground_truth = ad.read_h5ad(args.data)
    resource_rows = _resource_rows(RESULTS_ROOT / "profiles")
    resources = pd.DataFrame(resource_rows)
    resources.to_csv(RESULTS_ROOT / "benchmark_resources.csv", index=False)

    metric_rows = []
    for row in resource_rows:
        if row["status"] != "complete":
            continue
        prediction_path = RESULTS_ROOT / "predictions" / f"{row['run_id']}.h5ad"
        if not prediction_path.exists():
            continue
        metric_rows.append(
            evaluate_prediction(
                prediction_path=prediction_path,
                ground_truth=ground_truth,
                method=row["method"],
                fraction=float(row["fraction"]),
                seed=int(row["seed"]),
                expression_n=args.expression_eval_n,
                spatial_n=args.spatial_eval_n,
                kernel_n=args.kernel_n,
            )
        )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(RESULTS_ROOT / "benchmark_metrics.csv", index=False)

    if not resources.empty and not metrics.empty:
        summary = resources.merge(metrics, on=["run_id", "method", "fraction", "seed"], how="left")
    else:
        summary = resources.copy()
    summary.to_csv(RESULTS_ROOT / "benchmark_summary.csv", index=False)
    print(f"Wrote summaries to {RESULTS_ROOT}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MOSTA 9.5/11.5 scalability benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run one method/fraction/seed")
    run_parser.add_argument("--method", choices=METHODS, required=True)
    run_parser.add_argument("--fraction", type=float, required=True)
    run_parser.add_argument("--seed", type=int, default=2026)
    run_parser.add_argument("--device", default="cuda:0")
    run_parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    run_parser.add_argument("--stvcr-root", type=Path, default=DEFAULT_STVCR_ROOT)
    run_parser.add_argument("--stvcr-epochs", type=int, default=2001)
    run_parser.add_argument("--stvcr-ae-epochs", type=int, default=1000)
    run_parser.add_argument("--stvcr-alignment-iterations", type=int, default=5)
    run_parser.add_argument("--stfateflow-steps", type=int, default=1000)
    run_parser.add_argument(
        "--stfateflow-early-stopping",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use smoothed-loss early stopping (use --no-stfateflow-early-stopping for a fixed step budget).",
    )
    run_parser.set_defaults(func=run)

    eval_parser = subparsers.add_parser("evaluate", help="Aggregate profiles and evaluate predictions")
    eval_parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    eval_parser.add_argument("--expression-eval-n", type=int, default=1000)
    eval_parser.add_argument("--spatial-eval-n", type=int, default=3000)
    eval_parser.add_argument("--kernel-n", type=int, default=1000)
    eval_parser.set_defaults(func=evaluate)
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    parsed.func(parsed)

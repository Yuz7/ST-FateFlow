from __future__ import annotations

import argparse
import csv
import subprocess
import time
from pathlib import Path

from common import RESULTS_ROOT


DEFAULT_METHODS = ("stfateflow", "stvcr-default", "stvcr-all-cells")
DEFAULT_FRACTIONS = (0.125, 0.25, 0.5, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch isolated MOSTA benchmark runs")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--stfateflow-python",
        default="/data/yuz/envs/nicheflow/bin/python",
    )
    parser.add_argument(
        "--stvcr-python",
        default="/data/yuz/envs/stfm/bin/python",
    )
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--fractions", nargs="+", type=float, default=DEFAULT_FRACTIONS)
    parser.add_argument("--seeds", nargs="+", type=int, default=(2026, 2027, 2028))
    parser.add_argument("--stvcr-epochs", type=int, default=2001)
    parser.add_argument("--stvcr-ae-epochs", type=int, default=1000)
    parser.add_argument("--stvcr-alignment-iterations", type=int, default=5)
    parser.add_argument("--stfateflow-steps", type=int, default=1000)
    parser.add_argument(
        "--stfateflow-early-stopping",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    script = Path(__file__).with_name("run_benchmark.py")
    commands = []
    for seed in args.seeds:
        for fraction in args.fractions:
            for method in args.methods:
                python_executable = (
                    args.stfateflow_python if method == "stfateflow" else args.stvcr_python
                )
                commands.append(
                    [
                        python_executable,
                        str(script),
                        "run",
                        "--method",
                        method,
                        "--fraction",
                        str(fraction),
                        "--seed",
                        str(seed),
                        "--device",
                        args.device,
                        "--stvcr-epochs",
                        str(args.stvcr_epochs),
                        "--stvcr-ae-epochs",
                        str(args.stvcr_ae_epochs),
                        "--stvcr-alignment-iterations",
                        str(args.stvcr_alignment_iterations),
                        "--stfateflow-steps",
                        str(args.stfateflow_steps),
                    ]
                )
                if method == "stfateflow":
                    commands[-1].append(
                        "--stfateflow-early-stopping"
                        if args.stfateflow_early_stopping
                        else "--no-stfateflow-early-stopping"
                    )

    for command in commands:
        print(" ".join(command))
    if not args.execute:
        return

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    status_path = RESULTS_ROOT / "grid_status.csv"
    with status_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "fraction", "seed", "return_code", "wall_seconds", "command"],
        )
        writer.writeheader()
        for command in commands:
            started = time.perf_counter()
            completed = subprocess.run(command, check=False)
            writer.writerow(
                {
                    "method": command[command.index("--method") + 1],
                    "fraction": command[command.index("--fraction") + 1],
                    "seed": command[command.index("--seed") + 1],
                    "return_code": completed.returncode,
                    "wall_seconds": time.perf_counter() - started,
                    "command": " ".join(command),
                }
            )
            handle.flush()


if __name__ == "__main__":
    main()

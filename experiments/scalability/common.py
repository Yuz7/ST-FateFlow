from __future__ import annotations

import json
import os
import random
import resource
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import anndata as ad
import numpy as np
import torch
from scipy import sparse


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = Path(
    "/data/yuz/spatialtranscriptomics/spatiotemporal/spatiotemporal_data/"
    "MOSTA/mosta_affine_subset_9_11.h5ad"
)
DEFAULT_STVCR_ROOT = REPO_ROOT.parent / "stVCR"
RESULTS_ROOT = Path(
    os.environ.get(
        "STFATEFLOW_BENCHMARK_RESULTS",
        str(Path(__file__).resolve().parent / "results"),
    )
)
TRAIN_TIMES = (9.5, 11.5)
TEST_TIME = 10.5
os.environ.setdefault("NUMBA_CACHE_DIR", str(RESULTS_ROOT / "numba_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(RESULTS_ROOT / "matplotlib_cache"))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dense_float32(x: Any) -> np.ndarray:
    if sparse.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def fraction_token(fraction: float) -> str:
    return f"f{fraction:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def run_id(method: str, fraction: float, seed: int) -> str:
    return f"mosta_{method}_{fraction_token(fraction)}_seed{seed}"


def _stratified_priority_indices(
    obs,
    fraction: float,
    seed: int,
    strata_key: str,
) -> np.ndarray:
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if fraction == 1:
        return np.arange(len(obs), dtype=np.int64)

    labels = obs[strata_key].astype(str).to_numpy()
    selected: list[np.ndarray] = []
    for stratum_idx, label in enumerate(sorted(np.unique(labels))):
        idx = np.flatnonzero(labels == label)
        rng = np.random.default_rng(seed + 104729 * (stratum_idx + 1))
        ordered = idx[rng.permutation(len(idx))]
        n_keep = max(1, int(round(len(idx) * fraction)))
        selected.append(ordered[: min(n_keep, len(idx))])
    return np.sort(np.concatenate(selected)).astype(np.int64, copy=False)


def load_train_subset(
    data_path: Path,
    fraction: float,
    seed: int,
    time_key: str = "time",
    strata_key: str = "annotation",
) -> tuple[ad.AnnData, dict[str, Any]]:
    full = ad.read_h5ad(data_path)
    pieces = []
    counts: dict[str, int] = {}
    original_counts: dict[str, int] = {}

    for timepoint in TRAIN_TIMES:
        current = full[np.isclose(np.asarray(full.obs[time_key], dtype=float), timepoint)].copy()
        original_counts[str(timepoint)] = int(current.n_obs)
        local_idx = _stratified_priority_indices(current.obs, fraction, seed, strata_key)
        current = current[local_idx].copy()
        current.obs["benchmark_original_obs_name"] = current.obs_names.astype(str)
        pieces.append(current)
        counts[str(timepoint)] = int(current.n_obs)

    train = ad.concat(pieces, axis=0, join="inner", merge="same", uns_merge="same")
    train.obs_names_make_unique()
    train.obs[time_key] = train.obs[time_key].astype(float)
    if not hasattr(train.obs[strata_key].dtype, "categories"):
        train.obs[strata_key] = train.obs[strata_key].astype("category")

    metadata = {
        "fraction": float(fraction),
        "seed": int(seed),
        "train_times": list(TRAIN_TIMES),
        "test_time": TEST_TIME,
        "counts": counts,
        "original_counts": original_counts,
        "n_train_total": int(train.n_obs),
        "n_genes": int(train.n_vars),
    }
    return train, metadata


class _ProcessMemoryPoller:
    def __init__(self, interval_seconds: float = 0.1) -> None:
        self.interval_seconds = interval_seconds
        self.peak_gpu_mib: float | None = None
        self.peak_rss_gib = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvml = None
        self._handle = None
        self._device_index = 0

    def _init_nvml(self, device_index: int) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        except Exception:
            self._nvml = None
            self._handle = None

    def start(self, device_index: int) -> None:
        self._device_index = device_index
        self._init_nvml(device_index)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        pid = os.getpid()
        try:
            import psutil

            process = psutil.Process(pid)
        except Exception:
            process = None

        while not self._stop.wait(self.interval_seconds):
            if process is not None:
                try:
                    self.peak_rss_gib = max(
                        self.peak_rss_gib, process.memory_info().rss / 1024**3
                    )
                except Exception:
                    pass
            if self._nvml is not None and self._handle is not None:
                try:
                    processes = self._nvml.nvmlDeviceGetComputeRunningProcesses(self._handle)
                    used = sum(
                        p.usedGpuMemory for p in processes if p.pid == pid and p.usedGpuMemory
                    )
                    used_mib = used / 1024**2
                    self.peak_gpu_mib = max(self.peak_gpu_mib or 0.0, used_mib)
                except Exception:
                    pass
            else:
                try:
                    query = subprocess.run(
                        [
                            "nvidia-smi",
                            "-i",
                            str(self._device_index),
                            "--query-compute-apps=pid,used_gpu_memory",
                            "--format=csv,noheader,nounits",
                        ],
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=2,
                    )
                    used_mib = 0.0
                    for line in query.stdout.splitlines():
                        fields = [field.strip() for field in line.split(",")]
                        if len(fields) == 2 and int(fields[0]) == pid:
                            used_mib += float(fields[1])
                    self.peak_gpu_mib = max(self.peak_gpu_mib or 0.0, used_mib)
                except Exception:
                    pass

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass


@dataclass
class PhaseStats:
    wall_seconds: float
    peak_process_gpu_mib: float | None
    peak_torch_allocated_mib: float | None
    peak_torch_reserved_mib: float | None
    peak_process_rss_gib: float
    max_rss_gib: float


class ResourceProfiler:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.phases: dict[str, dict[str, Any]] = {}

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        use_cuda = self.device.type == "cuda" and torch.cuda.is_available()
        device_index = self.device.index or 0
        if use_cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)

        poller = _ProcessMemoryPoller()
        poller.start(device_index)
        started = time.perf_counter()
        try:
            yield
        finally:
            if use_cuda:
                torch.cuda.synchronize(self.device)
            elapsed = time.perf_counter() - started
            poller.stop()
            max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
            stats = PhaseStats(
                wall_seconds=elapsed,
                peak_process_gpu_mib=poller.peak_gpu_mib,
                peak_torch_allocated_mib=(
                    torch.cuda.max_memory_allocated(self.device) / 1024**2 if use_cuda else None
                ),
                peak_torch_reserved_mib=(
                    torch.cuda.max_memory_reserved(self.device) / 1024**2 if use_cuda else None
                ),
                peak_process_rss_gib=poller.peak_rss_gib,
                max_rss_gib=max_rss,
            )
            self.phases[name] = asdict(stats)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def base_profile(
    method: str,
    fraction: float,
    seed: int,
    data_path: Path,
    subset_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": run_id(method, fraction, seed),
        "method": method,
        "fraction": float(fraction),
        "seed": int(seed),
        "data_path": str(data_path),
        "subset": subset_metadata,
        "status": "running",
        "phases": {},
        "hardware": {
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
            "gpu_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        },
    }

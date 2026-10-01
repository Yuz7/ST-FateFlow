from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import torch
import torch.nn.functional as F
from scanpy import AnnData
from scipy import sparse
from torch.utils.data import IterableDataset
from torch_geometric.data import Data

from core.utils.datasets import init_worker_rng
from core.utils.fgwot import FGWOTPlanSampler
from core.utils.log import RankedLogger

_logger = RankedLogger(__name__, rank_zero_only=True)


class TrainItem(TypedDict):
    X_t0: torch.Tensor
    pos_t0: torch.Tensor
    ct_t0_ohe: torch.Tensor
    t0_value: torch.Tensor
    t0_raw: torch.Tensor
    X_t1: torch.Tensor
    pos_t1: torch.Tensor
    pos_t1_dist: torch.Tensor
    ct_t1_ohe: torch.Tensor
    t1_value: torch.Tensor
    t1_raw: torch.Tensor


class TrainBatch(TypedDict):
    X_t0: torch.Tensor
    pos_t0: torch.Tensor
    ct_t0_ohe: torch.Tensor
    t0_value: torch.Tensor
    t0_raw: torch.Tensor
    X_t1: torch.Tensor
    pos_t1: torch.Tensor
    pos_t1_dist: torch.Tensor
    ct_t1_ohe: torch.Tensor
    t1_value: torch.Tensor
    t1_raw: torch.Tensor
    X_t: torch.Tensor
    pos_t: torch.Tensor
    t: torch.Tensor
    tau: torch.Tensor
    delta_t: torch.Tensor
    vf_x: torch.Tensor
    vf_pos: torch.Tensor


class STFateFlowDataset(IterableDataset):
    def __init__(
        self,
        adata: AnnData,
        timepoint_column: str,
        cell_type_column: str,
        timepoints_ordered: list[Any],
        gene_key: str = "X",
        size_per_slice: int = 1024,
        seed: int = 2025,
        ot_plan_sampler=None,
        ot_plan_cache_path: str | Path | None = None,
        ot_plan_normalize: bool = True,
        ot_lambda: float = 0.1,
        test_timepoint: float = 0.0,
    ) -> None:
        super().__init__()
        self._preprocess_adata(
            adata=adata,
            gene_key=gene_key,
            timepoint_column=timepoint_column,
            cell_type_column=cell_type_column,
            timepoints_ordered=timepoints_ordered
        )
        self.ot_lambda = torch.tensor(ot_lambda)
        self.test_timepoint = test_timepoint
        self.timepoint_pc: dict[str, Data] = {}
        self._compute_timepoint_pc()

        self.train_timepoints_ordered = [
            timepoint for timepoint in self.timepoints_ordered if float(timepoint) != test_timepoint
        ]
        self.consecutive_pairs: list[tuple[str, str]] = list(
            zip(self.train_timepoints_ordered[:-1], self.train_timepoints_ordered[1:], strict=False)
        )
        self.num_pairs = len(self.consecutive_pairs)
        self.ot_plan_sampler = ot_plan_sampler or self.load_ot_plan_sampler(
            ot_plan_cache_path=ot_plan_cache_path,
            normalize_pi=ot_plan_normalize,
            consecutive_pairs=self.consecutive_pairs,
        )
        self.size_per_slice = size_per_slice
        self.seed = seed
        self.rng: np.random.Generator = np.random.default_rng(self.seed)

    def _preprocess_adata(
        self,
        adata: AnnData,
        timepoint_column: str,
        cell_type_column: str,
        timepoints_ordered: list[Any],
        gene_key: str = "X"
    ) -> None:
        coords = np.asarray(adata.obsm["spatial"], dtype=np.float32).copy()
        timepoint_indices = {
            timepoint: np.where(adata.obs[timepoint_column] == timepoint)[0] for timepoint in timepoints_ordered
        }

        ct_obs = adata.obs[cell_type_column]
        if hasattr(ct_obs, "cat"):
            ct_categories = sorted(ct_obs.cat.categories)
        else:
            ct_categories = sorted(np.asarray(ct_obs.dropna().unique()).tolist())
        ct_to_int = {annotation: i for i, annotation in enumerate(ct_categories)}

        if gene_key == "X":
            X_gene = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)  # noqa: N806
            X_gene = X_gene.astype(np.float32, copy=False)
            self.X_gene = X_gene
        else:
            X_gene = np.asarray(adata.obsm[gene_key], dtype=np.float32).copy()
            self.X_gene = X_gene
            
        self.X_gene = X_gene
        self.coords = coords
        self.ct = np.array(ct_obs)
        self.timepoints_ordered = timepoints_ordered
        self.time_origin = min(float(timepoint) for timepoint in timepoints_ordered)
        self.timepoint_to_value = {
            timepoint: float(timepoint) - self.time_origin for timepoint in timepoints_ordered
        }
        self.timepoint_indices = timepoint_indices
        self.ct_to_int = ct_to_int

    @staticmethod
    def load_ot_plan_sampler(
        ot_plan_cache_path: str | Path | None,
        normalize_pi: bool,
        consecutive_pairs: list[tuple[str, str]],
    ) -> FGWOTPlanSampler:
        if ot_plan_cache_path is None:
            raise ValueError("Either `ot_plan_sampler` or `ot_plan_cache_path` must be provided.")

        fp = Path(ot_plan_cache_path)
        if not fp.exists():
            raise FileNotFoundError(f"FGWOT cache not found: {fp}")

        cached = np.load(fp, allow_pickle=False)
        pi_by_transition: dict[tuple[float, float], np.ndarray] = {}
        loaded_keys: list[tuple[float, float]] = []
        for key in cached.files:
            if not key.startswith("transition_"):
                continue
            parts = key.split("_")[1:]
            if len(parts) >= 2:
                transition_key = (float(parts[0]), float(parts[1]))
            elif len(parts) == 1:
                pair_idx = int(parts[0])
                if pair_idx >= len(consecutive_pairs):
                    raise IndexError(
                        f"Transition index {pair_idx} in {fp} exceeds consecutive pairs: {consecutive_pairs}"
                    )
                t1, t2 = consecutive_pairs[pair_idx]
                transition_key = (float(t1), float(t2))
            else:
                continue

            pi = np.asarray(cached[key], dtype=np.float64)
            pi = np.clip(pi, a_min=0.0, a_max=None)
            mass = float(pi.sum())
            if mass <= 0:
                raise ValueError(f"Invalid cached pi mass for transition {transition_key}")
            pi_by_transition[transition_key] = pi
            loaded_keys.append(transition_key)

        if not pi_by_transition:
            raise ValueError(f"No transition_* arrays found in cache: {fp}")
        _logger.info(f"Loaded FGWOT transitions from {fp}: {sorted(loaded_keys)}")
        return FGWOTPlanSampler(pi_matrix=pi_by_transition, normalize_pi=normalize_pi)

    def _compute_timepoint_pc(self) -> None:
        _logger.info("Creating per timepoint PyTorch Geometric Data objects")
        ct_get_vec = np.vectorize(self.ct_to_int.get)

        for timepoint in self.timepoints_ordered:
            indices = self.timepoint_indices[timepoint]
            self.timepoint_pc[timepoint] = Data(
                x=torch.Tensor(self.X_gene[indices]),
                pos=torch.Tensor(self.coords[indices]),
                ct=torch.as_tensor(ct_get_vec(self.ct[indices]), dtype=torch.long),
                t_value=torch.tensor(self.timepoint_to_value[timepoint], dtype=torch.float32),
                t_raw=torch.tensor(float(timepoint), dtype=torch.float32),
            )
            self.timepoint_pc[timepoint].ct_ohe = F.one_hot(
                self.timepoint_pc[timepoint].ct,
                num_classes=len(self.ct_to_int),
            ).float()

    @staticmethod
    def _safe_float(value: str | float | int) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _sample_global_indices(self, n_cells: int) -> list[int]:
        replace = n_cells < self.size_per_slice
        idx = self.rng.choice(n_cells, size=self.size_per_slice, replace=replace)
        return idx.tolist()

    def _compute_ot(
        self,
        transition_idx: tuple[float, float],
        source_indices: list[int],
        target_indices: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        if isinstance(self.ot_plan_sampler, FGWOTPlanSampler):
            pi = self.ot_plan_sampler.get_map(
                transition_idx=transition_idx,
                source_indices=torch.as_tensor(source_indices, dtype=torch.long),
                target_indices=torch.as_tensor(target_indices, dtype=torch.long),
            )
        else:
            raise ValueError("OT plan sampler must be an instance of FGWOTPlanSampler to use transition-aware OT.")

        src_idx, tgt_idx = self.ot_plan_sampler.sample_map(
            pi, batch_size=self.size_per_slice, replace=False
        )
        return (
            torch.as_tensor(src_idx, dtype=torch.long),
            torch.as_tensor(tgt_idx, dtype=torch.long),
            pi,
        )

    def __iter__(self):
        self.rng = init_worker_rng(seed=self.seed)
        while True:
            transition_idx = int(self.rng.integers(self.num_pairs))
            sample = self._sample_aligned_pair(transition_idx=transition_idx)
            yield {
                "X_t0": sample["X_t1"],
                "pos_t0": sample["pos_t1"],
                "ct_t0_ohe": sample["ct_t1_ohe"],
                "t0_value": sample["t1_value"],
                "t0_raw": sample["t1_raw"],
                "X_t1": sample["X_t2"],
                "pos_t1": sample["pos_t2"],
                "pos_t1_dist": sample["pos_t2_dist"],
                "ct_t1_ohe": sample["ct_t2_ohe"],
                "t1_value": sample["t2_value"],
                "t1_raw": sample["t2_raw"],
            }

    def _sample_aligned_pair(self, transition_idx: int) -> dict[str, torch.Tensor]:
        t1, t2 = self.consecutive_pairs[transition_idx]

        pc_t1 = self.timepoint_pc[t1]
        pc_t2 = self.timepoint_pc[t2]

        idx_t1 = self._sample_global_indices(pc_t1.x.size(0))
        idx_t2 = self._sample_global_indices(pc_t2.x.size(0))

        src_local, tgt_local, pi = self._compute_ot(
            transition_idx=(float(t1), float(t2)),
            source_indices=idx_t1,
            target_indices=idx_t2,
        )

        # src_local/tgt_local are indices in the OT sub-plan built on idx_t1/idx_t2.
        # They must be mapped back to global point-cloud indices before gathering.
        if int(src_local.max()) >= len(idx_t1) or int(tgt_local.max()) >= len(idx_t2):
            raise IndexError("OT local indices exceed sampled subset size; index mapping is invalid.")

        idx_t1_t = torch.as_tensor(idx_t1, dtype=torch.long)
        idx_t2_t = torch.as_tensor(idx_t2, dtype=torch.long)
        src_global = idx_t1_t.index_select(0, src_local)
        tgt_global = idx_t2_t.index_select(0, tgt_local)

        x_t1 = pc_t1.x.index_select(0, src_global)
        pos_t1 = pc_t1.pos.index_select(0, src_global)
        x_t2 = pc_t2.x.index_select(0, tgt_global)
        pos_t2 = pc_t2.pos.index_select(0, tgt_global)
        pos_t2_dist = pc_t2.pos.index_select(0, idx_t2_t)

        return {
            "X_t1": x_t1,
            "pos_t1": pos_t1,
            "ct_t1_ohe": pc_t1.ct_ohe.index_select(0, src_global),
            "t1_value": pc_t1.t_value,
            "t1_raw": pc_t1.t_raw,
            "X_t2": x_t2,
            "pos_t2": pos_t2,
            "pos_t2_dist": pos_t2_dist,
            "ct_t2_ohe": pc_t2.ct_ohe.index_select(0, tgt_global),
            "t2_value": pc_t2.t_value,
            "t2_raw": pc_t2.t_raw,
        }


def _stack_scalar(batch: list[TrainItem], key: str, dtype: torch.dtype) -> torch.Tensor:
    return torch.stack([sample[key].reshape(()) for sample in batch], dim=0).to(dtype=dtype)


def stfateflow_collate(batch: list[TrainItem]) -> TrainBatch:
    X_t0 = torch.stack([sample["X_t0"] for sample in batch], dim=0)
    pos_t0 = torch.stack([sample["pos_t0"] for sample in batch], dim=0)
    X_t1 = torch.stack([sample["X_t1"] for sample in batch], dim=0)
    pos_t1 = torch.stack([sample["pos_t1"] for sample in batch], dim=0)
    pos_t1_dist = torch.stack([sample["pos_t1_dist"] for sample in batch], dim=0)

    bs = X_t0.shape[0]
    ct_t0_ohe = torch.stack([sample["ct_t0_ohe"] for sample in batch], dim=0)
    ct_t1_ohe = torch.stack([sample["ct_t1_ohe"] for sample in batch], dim=0)
    t0_value = _stack_scalar(batch, "t0_value", X_t0.dtype)
    t1_value = _stack_scalar(batch, "t1_value", X_t0.dtype)
    t0_raw = _stack_scalar(batch, "t0_raw", X_t0.dtype)
    t1_raw = _stack_scalar(batch, "t1_raw", X_t0.dtype)

    tau = torch.rand(bs, dtype=X_t0.dtype)
    tau_unsq = tau[:, None, None]
    delta_t = t1_value - t0_value
    if torch.any(delta_t <= 0):
        raise ValueError(f"Expected increasing timepoints, got t0={t0_value}, t1={t1_value}.")
    t = t0_value + tau * delta_t

    X_t = (1 - tau_unsq) * X_t0 + tau_unsq * X_t1
    pos_t = (1 - tau_unsq) * pos_t0 + tau_unsq * pos_t1

    vf_x = (X_t1 - X_t0) / delta_t[:, None, None]
    vf_pos = (pos_t1 - pos_t0) / delta_t[:, None, None]

    return {
        "X_t0": X_t0,
        "pos_t0": pos_t0,
        "ct_t0_ohe": ct_t0_ohe,
        "t0_value": t0_value,
        "t0_raw": t0_raw,
        "X_t1": X_t1,
        "pos_t1": pos_t1,
        "pos_t1_dist": pos_t1_dist,
        "ct_t1_ohe": ct_t1_ohe,
        "t1_value": t1_value,
        "t1_raw": t1_raw,
        "X_t": X_t,
        "pos_t": pos_t,
        "t": t,
        "tau": tau,
        "delta_t": delta_t,
        "vf_x": vf_x,
        "vf_pos": vf_pos,
    }

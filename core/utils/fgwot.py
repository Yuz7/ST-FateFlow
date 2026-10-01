from typing import Optional

import numpy as np
import torch


class FGWOTPlanSampler:
    """Sampler that consumes a precomputed FGW transport plan and never recomputes it."""

    def __init__(
        self,
        pi_matrix: Optional[np.ndarray | dict[tuple[float, float], np.ndarray]] = None,
        normalize_pi: bool = True,
    ) -> None:
        self.pi_matrix: np.ndarray | None = None
        self.pi_by_transition: dict[tuple[float, float], np.ndarray] = {}
        self.normalize_pi = bool(normalize_pi)
        if pi_matrix is not None:
            self._assign_pi(pi_matrix)

    @staticmethod
    def _normalize_pi(pi: np.ndarray) -> np.ndarray:
        pi = np.asarray(pi, dtype=np.float64)
        if pi.ndim != 2:
            raise ValueError(f"pi must be 2D, got shape={pi.shape}")
        pi = np.maximum(pi, 0.0)
        mass = float(pi.sum())
        if mass <= 1e-12:
            raise ValueError("pi mass must be positive.")
        return pi / mass

    def _assign_pi(self, pi_matrix: np.ndarray | dict[tuple[float, float], np.ndarray]) -> None:
        if isinstance(pi_matrix, dict):
            prepared: dict[tuple[float, float], np.ndarray] = {}
            for k, v in pi_matrix.items():
                arr = np.asarray(v, dtype=np.float64)
                if arr.ndim != 2:
                    raise ValueError(f"pi must be 2D, got shape={arr.shape}")
                arr = np.maximum(arr, 0.0)
                mass = float(arr.sum())
                if mass <= 1e-12:
                    raise ValueError("pi mass must be positive.")
                prepared[k] = self._normalize_pi(arr) if self.normalize_pi else arr

            if not prepared:
                raise ValueError("pi_matrix dict is empty.")
            self.pi_by_transition = prepared
            first_key = sorted(prepared.keys())[0]
            self.pi_matrix = prepared[first_key]
        else:
            arr = np.asarray(pi_matrix, dtype=np.float64)
            if arr.ndim != 2:
                raise ValueError(f"pi must be 2D, got shape={arr.shape}")
            arr = np.maximum(arr, 0.0)
            mass = float(arr.sum())
            if mass <= 1e-12:
                raise ValueError("pi mass must be positive.")
            self.pi_matrix = self._normalize_pi(arr) if self.normalize_pi else arr
            self.pi_by_transition = {}

    def set_precomputed_plans(self, plans: dict[tuple[float, float], np.ndarray]) -> None:
        self._assign_pi(plans)

    def get_map(
        self,
        x0: Optional[torch.Tensor] = None,
        x1: Optional[torch.Tensor] = None,
        f0: Optional[torch.Tensor] = None,
        f1: Optional[torch.Tensor] = None,
        transition_idx: tuple[float, float] = (0.0, 0.0),
        source_indices: Optional[torch.Tensor] = None,
        target_indices: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        del x0, x1, f0, f1

        if self.pi_matrix is None:
            raise ValueError("FGWOTPlanSampler requires precomputed pi_matrix at initialization.")

        if self.pi_by_transition:
            base_pi = self.pi_by_transition[transition_idx]
        else:
            base_pi = self.pi_matrix

        # Allow full-plan access when sub-indices are not provided.
        if source_indices is None or target_indices is None:
            return base_pi

        src_idx = source_indices.detach().cpu().numpy().astype(np.int64)
        tgt_idx = target_indices.detach().cpu().numpy().astype(np.int64)
        sub_pi = base_pi[np.ix_(src_idx, tgt_idx)]
        return self._normalize_pi(sub_pi) if self.normalize_pi else sub_pi

    @staticmethod
    def sample_map(pi: np.ndarray, batch_size: int, replace: bool = True) -> tuple[np.ndarray, np.ndarray]:
        row_sums = np.asarray(pi, dtype=np.float64).sum(axis=1)
        if row_sums.sum() <= 0:
            raise ValueError("Invalid transport plan: row sums are all zero.")

        valid_rows = row_sums > 0
        row_probs = np.zeros_like(row_sums, dtype=np.float64)
        row_probs[valid_rows] = row_sums[valid_rows]
        row_probs = row_probs / row_probs.sum()

        n_valid = int(valid_rows.sum())
        if not replace and batch_size > n_valid:
            replace = True

        i_samples = np.random.choice(
            pi.shape[0],
            p=row_probs,
            size=batch_size,
            replace=replace,
        ).astype(np.int64, copy=False)

        selected_rows = pi[i_samples]
        denom = row_sums[i_samples][:, None]
        row_p = np.divide(
            selected_rows,
            denom,
            out=np.zeros_like(selected_rows, dtype=np.float64),
            where=denom > 0,
        )

        cdf = np.cumsum(row_p, axis=1)
        cdf[:, -1] = 1.0
        u = np.random.random((batch_size, 1))
        j_samples = (cdf < u).sum(axis=1).astype(np.int64, copy=False)

        return i_samples, j_samples

    def sample_plan(
        self,
        c0: torch.Tensor,
        c1: torch.Tensor,
        f0: Optional[torch.Tensor] = None,
        f1: Optional[torch.Tensor] = None,
        transition_idx: tuple[float, float] = (0.0, 0.0),
        source_indices: Optional[torch.Tensor] = None,
        target_indices: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
        replace: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        f0_t = f0 if f0 is not None else c0
        f1_t = f1 if f1 is not None else c1
        if c0.shape[0] == 0 or c1.shape[0] == 0:
            raise ValueError("Cannot sample from empty batch.")
        if batch_size is None:
            raise ValueError("batch_size must be provided to sample_plan.")

        pi = self.get_map(
            transition_idx=transition_idx,
            source_indices=source_indices,
            target_indices=target_indices,
        )
        i_idx, j_idx = self.sample_map(pi, int(batch_size), replace=replace)

        i_t = torch.as_tensor(i_idx, device=c0.device, dtype=torch.long)
        j_t = torch.as_tensor(j_idx, device=c1.device, dtype=torch.long)
        return (
            c0.index_select(0, i_t),
            c1.index_select(0, j_t),
            f0_t.index_select(0, i_t),
            f1_t.index_select(0, j_t),
        )

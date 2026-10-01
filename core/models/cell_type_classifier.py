from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy import sparse


class FourierTimeEmbedding(nn.Module):
    def __init__(self, embed_dim: int = 64, max_period: int = 10000) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.max_period = max_period

    def forward(self, t: Tensor) -> Tensor:
        half = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t.float().reshape(-1, 1) * freqs.reshape(1, -1)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.embed_dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb


class CellTypeClassifier(nn.Module):
    """Per-cell classifier using gene expression, spatial coordinate, and time."""

    def __init__(
        self,
        gene_dim: int,
        coord_dim: int,
        num_cell_types: int,
        hidden_dim: int = 256,
        depth: int = 4,
        time_embed_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.gene_dim = gene_dim
        self.coord_dim = coord_dim
        self.num_cell_types = num_cell_types
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.time_embed_dim = time_embed_dim
        self.dropout = dropout

        self.time_embed = FourierTimeEmbedding(embed_dim=time_embed_dim)
        input_dim = gene_dim + coord_dim + time_embed_dim

        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(depth):
            layers.extend(
                [
                    nn.Linear(dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                ]
            )
            dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, num_cell_types))
        self.net = nn.Sequential(*layers)

    @property
    def config(self) -> dict[str, Any]:
        return {
            "gene_dim": self.gene_dim,
            "coord_dim": self.coord_dim,
            "num_cell_types": self.num_cell_types,
            "hidden_dim": self.hidden_dim,
            "depth": self.depth,
            "time_embed_dim": self.time_embed_dim,
            "dropout": self.dropout,
        }

    def forward(self, gene: Tensor, coord: Tensor, time: Tensor) -> Tensor:
        leading_shape = gene.shape[:-1]
        gene_flat = gene.reshape(-1, self.gene_dim)
        coord_flat = coord.reshape(-1, self.coord_dim)
        if time.ndim == 1 and len(leading_shape) == 2 and time.shape[0] == leading_shape[0]:
            time = time[:, None].expand(leading_shape)
        else:
            time = time.expand(leading_shape)
        time_flat = time.reshape(-1)
        time_emb = self.time_embed(time_flat)
        logits = self.net(torch.cat([gene_flat, coord_flat, time_emb], dim=-1))
        return logits.reshape(*leading_shape, self.num_cell_types)

    def loss(
        self,
        gene: Tensor,
        coord: Tensor,
        time: Tensor,
        label: Tensor,
        weight: Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> Tensor:
        logits = self.forward(gene, coord, time)
        return F.cross_entropy(
            logits.reshape(-1, self.num_cell_types),
            label.reshape(-1),
            weight=weight,
            label_smoothing=label_smoothing,
        )

    @torch.no_grad()
    def predict_proba(self, gene: Tensor, coord: Tensor, time: Tensor) -> Tensor:
        return self.forward(gene, coord, time).softmax(dim=-1)

    @torch.no_grad()
    def predict(self, gene: Tensor, coord: Tensor, time: Tensor) -> Tensor:
        return self.forward(gene, coord, time).argmax(dim=-1)

    def save(
        self,
        path: str | Path,
        *,
        label_to_cell_type: dict[int, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": self.config,
                "label_to_cell_type": label_to_cell_type,
                "metadata": metadata,
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> tuple["CellTypeClassifier", dict[str, Any]]:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model, ckpt


class CellTypeClassifierInterface:
    def __init__(
        self,
        adata=None,
        gene_dim: int | None = None,
        coord_dim: int | None = None,
        num_cell_types: int | None = None,
        *,
        gene_key: str = "X",
        coord_key: str = "spatial",
        label_key: str = "Annotation",
        hidden_dim: int = 256,
        depth: int = 4,
        time_embed_dim: int = 64,
        dropout: float = 0.1,
        ckpt_path: str | Path | None = None,
        device: str | torch.device = "cuda",
        label_to_cell_type: dict[int, str] | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.label_to_cell_type = label_to_cell_type
        self.gene_key = gene_key
        self.coord_key = coord_key
        self.label_key = label_key
        self.input_stats = None

        if ckpt_path is None and adata is not None:
            gene_dim = self._adata_gene(adata, gene_key).shape[1]
            coord_dim = np.asarray(adata.obsm[coord_key]).shape[1]
            categories = list(adata.obs[label_key].cat.categories)
            num_cell_types = len(categories)
            self.label_to_cell_type = {i: str(cell_type) for i, cell_type in enumerate(categories)}

        if ckpt_path is None:
            self.model = CellTypeClassifier(
                gene_dim=gene_dim,
                coord_dim=coord_dim,
                num_cell_types=num_cell_types,
                hidden_dim=hidden_dim,
                depth=depth,
                time_embed_dim=time_embed_dim,
                dropout=dropout,
            ).to(self.device)
        else:
            self.model, ckpt = CellTypeClassifier.load(ckpt_path, map_location=self.device)
            self.model = self.model.to(self.device)
            self.label_to_cell_type = ckpt.get("label_to_cell_type")
            metadata = ckpt.get("metadata") or {}
            self.gene_key = metadata.get("gene_key", self.gene_key)
            self.coord_key = metadata.get("coord_key", self.coord_key)
            self.label_key = metadata.get("label_key", self.label_key)
            self.input_stats = metadata.get("input_stats")

    @staticmethod
    def _as_tensor(x, dtype: torch.dtype) -> Tensor:
        if torch.is_tensor(x):
            return x.to(dtype=dtype)
        return torch.tensor(x, dtype=dtype)

    @staticmethod
    def _adata_gene(adata, gene_key: str):
        x = adata.X if gene_key == "X" else adata.obsm[gene_key]
        if sparse.issparse(x):
            x = x.toarray()
        return np.asarray(x, dtype=np.float32)

    def _adata_inputs(
        self,
        adata,
        *,
        gene_key: str,
        coord_key: str,
        time_key: str,
        label_key: str | None = None,
    ):
        gene = self._adata_gene(adata, gene_key)
        coord = np.asarray(adata.obsm[coord_key], dtype=np.float32)
        time = np.asarray(adata.obs[time_key], dtype=np.float32)
        if label_key is None:
            return gene, coord, time
        label = adata.obs[label_key].cat.codes.to_numpy()
        return gene, coord, time, label

    @staticmethod
    def _fit_input_stats(gene, coord, time):
        return {
            "gene_mean": np.asarray(gene, dtype=np.float32).mean(axis=0),
            "gene_std": np.asarray(gene, dtype=np.float32).std(axis=0) + 1e-6,
            "coord_mean": np.asarray(coord, dtype=np.float32).mean(axis=0),
            "coord_std": np.asarray(coord, dtype=np.float32).std(axis=0) + 1e-6,
            "time_mean": np.asarray(time, dtype=np.float32).mean(),
            "time_std": np.asarray(time, dtype=np.float32).std() + 1e-6,
        }

    def _apply_input_stats(self, gene, coord, time):
        stats = self.input_stats
        gene = (np.asarray(gene, dtype=np.float32) - stats["gene_mean"]) / stats["gene_std"]
        coord = (np.asarray(coord, dtype=np.float32) - stats["coord_mean"]) / stats["coord_std"]
        time = (np.asarray(time, dtype=np.float32) - stats["time_mean"]) / stats["time_std"]
        return gene, coord, time

    def _flatten_inputs(self, gene, coord, time, label=None):
        gene_t = self._as_tensor(gene, torch.float32)
        coord_t = self._as_tensor(coord, torch.float32)
        time_t = self._as_tensor(time, torch.float32)
        leading_shape = gene_t.shape[:-1]

        if time_t.ndim == 1 and len(leading_shape) == 2 and time_t.shape[0] == leading_shape[0]:
            time_t = time_t[:, None].expand(leading_shape)
        else:
            time_t = time_t.expand(leading_shape)

        gene_t = gene_t.reshape(-1, self.model.gene_dim)
        coord_t = coord_t.reshape(-1, self.model.coord_dim)
        time_t = time_t.reshape(-1)

        if label is None:
            return gene_t, coord_t, time_t, leading_shape

        label_t = self._as_tensor(label, torch.long).reshape(-1)
        return gene_t, coord_t, time_t, label_t, leading_shape

    def train(
        self,
        adata=None,
        *,
        gene=None,
        coord=None,
        time=None,
        label=None,
        gene_key: str | None = None,
        coord_key: str | None = None,
        time_key: str = "time",
        label_key: str | None = None,
        epochs: int = 100,
        batch_size: int = 1024,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        class_weight: bool = True,
        label_smoothing: float = 0.05,
        save_path: str | Path | None = None,
    ) -> list[dict[str, float]]:
        if adata is not None:
            gene_key = gene_key or self.gene_key
            coord_key = coord_key or self.coord_key
            label_key = label_key or self.label_key
            categories = list(adata.obs[label_key].cat.categories)
            self.label_to_cell_type = {i: str(cell_type) for i, cell_type in enumerate(categories)}
            gene, coord, time, label = self._adata_inputs(
                adata,
                gene_key=gene_key,
                coord_key=coord_key,
                time_key=time_key,
                label_key=label_key,
            )

        self.input_stats = self._fit_input_stats(gene, coord, time)
        gene, coord, time = self._apply_input_stats(gene, coord, time)
        gene_t, coord_t, time_t, label_t, _ = self._flatten_inputs(gene, coord, time, label)
        dataset = TensorDataset(gene_t, coord_t, time_t, label_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_weight = None
        if class_weight:
            counts = torch.bincount(label_t, minlength=self.model.num_cell_types).float()
            loss_weight = counts.sum() / (counts.clamp_min(1.0) * self.model.num_cell_types)
            loss_weight = (loss_weight / loss_weight.mean()).to(self.device)

        history = []
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            total_correct = 0
            total_count = 0

            for gene_b, coord_b, time_b, label_b in loader:
                gene_b = gene_b.to(self.device)
                coord_b = coord_b.to(self.device)
                time_b = time_b.to(self.device)
                label_b = label_b.to(self.device)

                optimizer.zero_grad(set_to_none=True)
                loss = self.model.loss(
                    gene_b,
                    coord_b,
                    time_b,
                    label_b,
                    weight=loss_weight,
                    label_smoothing=label_smoothing,
                )
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    pred = self.model.predict(gene_b, coord_b, time_b)
                    total_loss += float(loss.detach().cpu()) * label_b.numel()
                    total_correct += int((pred == label_b).sum().detach().cpu())
                    total_count += int(label_b.numel())

            history.append(
                {
                    "epoch": float(epoch + 1),
                    "loss": total_loss / total_count,
                    "accuracy": total_correct / total_count,
                }
            )

        if save_path is not None:
            self.model.save(
                save_path,
                label_to_cell_type=self.label_to_cell_type,
                metadata={
                    "gene_key": gene_key or self.gene_key,
                    "coord_key": coord_key or self.coord_key,
                    "label_key": label_key or self.label_key,
                    "time_key": time_key,
                    "input_stats": self.input_stats,
                },
            )

        return history

    @torch.no_grad()
    def predict(
        self,
        adata=None,
        *,
        gene=None,
        coord=None,
        time=None,
        gene_key: str | None = None,
        coord_key: str | None = None,
        time_key: str = "time",
        batch_size: int = 4096,
        return_proba: bool = False,
        return_cell_type: bool = False,
        smooth_k: int = 0,
        smooth_chunk_size: int = 2048,
    ):
        if adata is not None:
            gene_key = gene_key or self.gene_key
            coord_key = coord_key or self.coord_key
            gene, coord, time = self._adata_inputs(
                adata,
                gene_key=gene_key,
                coord_key=coord_key,
                time_key=time_key,
            )

        gene, coord, time = self._apply_input_stats(gene, coord, time)
        gene_t, coord_t, time_t, leading_shape = self._flatten_inputs(gene, coord, time)
        dataset = TensorDataset(gene_t, coord_t, time_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        pred_chunks = []
        proba_chunks = []
        self.model.eval()
        for gene_b, coord_b, time_b in loader:
            gene_b = gene_b.to(self.device)
            coord_b = coord_b.to(self.device)
            time_b = time_b.to(self.device)
            logits = self.model(gene_b, coord_b, time_b)
            proba = logits.softmax(dim=-1)
            proba_chunks.append(proba.detach().cpu())
            pred_chunks.append(proba.argmax(dim=-1).detach().cpu())

        proba = torch.cat(proba_chunks, dim=0).reshape(*leading_shape, self.model.num_cell_types)
        if smooth_k > 0:
            proba = self._smooth_proba_by_coord(
                proba.reshape(-1, self.model.num_cell_types),
                coord_t,
                k=smooth_k,
                chunk_size=smooth_chunk_size,
            ).reshape(*leading_shape, self.model.num_cell_types)
        pred = proba.argmax(dim=-1)

        output = pred
        if return_cell_type:
            output = np.array([self.label_to_cell_type[int(i)] for i in pred.reshape(-1)]).reshape(leading_shape)
        if return_proba:
            return output, proba
        return output

    @torch.no_grad()
    def _smooth_proba_by_coord(
        self,
        proba: Tensor,
        coord: Tensor,
        *,
        k: int,
        chunk_size: int,
    ) -> Tensor:
        coord_device = coord.to(self.device)
        proba_device = proba.to(self.device)
        chunks = []
        for start in range(0, coord_device.shape[0], chunk_size):
            end = min(start + chunk_size, coord_device.shape[0])
            d = torch.cdist(coord_device[start:end], coord_device)
            nn_idx = d.topk(k=k, largest=False).indices
            chunks.append(proba_device[nn_idx].mean(dim=1).detach().cpu())
        return torch.cat(chunks, dim=0)

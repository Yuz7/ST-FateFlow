from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import torch

from core.models.STFateFlow import STFateFlow_Module
from core.models.backbones.pc_transformer import PointCloudTransformer


def _checkpoint_config(ckpt: dict[str, Any]) -> dict[str, Any]:
    config = ckpt.get("config", {})
    return config if isinstance(config, dict) else {}


def load_stfateflow_checkpoint(
    ckpt_path: str | Path,
    *,
    device: str | torch.device | None = None,
    gene_dim: int | None = None,
    coord_dim: int = 2,
    patch_size: int = 16,
    hidden_size: int | None = None,
    depth: int = 6,
    mlp_ratio: float = 4.0,
    embed_dim: int = 128,
    num_heads: int = 4,
    dropout: float = 0.0,
    use_potential: bool = False,
    gradient_checkpointing: bool = False,
    lambda_features: float | None = None,
    lambda_pos: float | None = None,
    lambda_cdm: float | None = None,
    spatial_loss_type: str | None = None,
    num_steps: int | None = None,
    strict: bool = True,
    map_location: str | torch.device = "cpu",
) -> tuple[STFateFlow_Module, dict[str, Any]]:
    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    if not isinstance(ckpt, dict):
        raise TypeError(f"Expected checkpoint dict from {ckpt_path}, got {type(ckpt)!r}.")

    config = _checkpoint_config(ckpt)
    state_dict = ckpt.get("flow_state_dict", ckpt)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Checkpoint {ckpt_path} does not contain a valid state dict.")

    gene_dim = int(gene_dim if gene_dim is not None else ckpt["n_genes"])
    lambda_features = float(lambda_features if lambda_features is not None else config.get("lambda_features", 0.1))
    lambda_pos = float(lambda_pos if lambda_pos is not None else config.get("lambda_pos", 1.0))
    lambda_cdm = float(lambda_cdm if lambda_cdm is not None else config.get("lambda_cdm", 1.0))
    spatial_loss_type = str(
        spatial_loss_type
        if spatial_loss_type is not None
        else config.get("spatial_loss_type", config.get("spatial_loss", "mse"))
    )
    num_steps = int(num_steps if num_steps is not None else config.get("num_steps_ode", 20))

    backbone = PointCloudTransformer(
        gene_dim=gene_dim,
        coord_dim=coord_dim,
        patch_size=patch_size,
        hidden_size=hidden_size,
        depth=depth,
        mlp_ratio=mlp_ratio,
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=dropout,
        use_potential=use_potential,
        gradient_checkpointing=gradient_checkpointing,
    )
    model = STFateFlow_Module(
        lambda_features=lambda_features,
        lambda_pos=lambda_pos,
        lambda_cdm=lambda_cdm,
        backbone=backbone,
        num_steps=num_steps,
        spatial_loss_type=spatial_loss_type,
        use_potential=use_potential,
    )

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    model.eval()
    if device is not None:
        model = model.to(device)

    load_info = {
        "path": ckpt_path,
        "checkpoint": ckpt,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "config": config,
    }
    return model, load_info


def release_training_memory(*names: str, namespace: dict[str, Any] | None = None) -> None:
    namespace = globals() if namespace is None else namespace
    for name in names:
        obj = namespace.pop(name, None)
        if hasattr(obj, "zero_grad"):
            obj.zero_grad(set_to_none=True)
        del obj

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

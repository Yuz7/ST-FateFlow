import math

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: Tensor, dim: int, max_period: int = 10000) -> Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: Tensor) -> Tensor:
        return self.mlp(self.timestep_embedding(t.view(-1), self.frequency_embedding_size))


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be even, got {embed_dim}.")
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def get_1d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    return get_1d_sincos_pos_embed_from_grid(embed_dim, np.arange(grid_size, dtype=np.float32))


class PatchEmbedder(nn.Module):
    def __init__(self, input_size: int, patch_size: int, hidden_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = math.ceil(input_size / patch_size)
        self.mlp = nn.Sequential(
            nn.Linear(patch_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        _, length = x.shape
        pad_size = (self.patch_size - (length % self.patch_size)) % self.patch_size
        if pad_size > 0:
            x = F.pad(x, (0, pad_size), "constant", 0)
        x = x.reshape(x.shape[0], -1, self.patch_size)
        return self.mlp(x)


class GiTBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size),
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        attn_in = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out = self.attn(attn_in, attn_in, attn_in, need_weights=False)[0]
        x = x + gate_msa.unsqueeze(1) * attn_out
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class PointCloudTransformer(nn.Module):
    """GiT-style backbone adapted to the project's point-cloud vector-field API."""

    def __init__(
        self,
        gene_dim: int | None = None,
        coord_dim: int | None = None,
        patch_size: int = 16,
        hidden_size: int | None = None,
        depth: int = 6,
        mlp_ratio: float = 4.0,
        embed_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.0,
        use_potential: bool = False,
        gradient_checkpointing: bool = False,
        **_: object,
    ) -> None:
        super().__init__()
        if gene_dim is None:
            raise ValueError("`gene_dim` must be provided.")
        if coord_dim is None:
            raise ValueError("`coord_dim` must be provided.")
        hidden_size = embed_dim if hidden_size is None else hidden_size
        if hidden_size % num_heads != 0:
            hidden_size = num_heads * (hidden_size // num_heads)
        if hidden_size <= 0:
            raise ValueError("hidden_size/embed_dim must be at least num_heads.")

        self.gene_dim = gene_dim
        self.coord_dim = coord_dim
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.use_potential = use_potential
        self.gradient_checkpointing = gradient_checkpointing

        self.num_patches_pos = 1
        self.num_patches_gene = math.ceil(gene_dim / patch_size)
        self.num_patches = self.num_patches_pos + self.num_patches_gene

        self.x_embedder = nn.Linear(coord_dim, hidden_size)
        self.g_embedder = PatchEmbedder(gene_dim, patch_size, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList(
            [GiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(depth)]
        )

        self.x_head = FinalLayer(hidden_size, coord_dim, 1)
        self.g_head = FinalLayer(hidden_size, patch_size, 1)
        self.potential_x_head = FinalLayer(hidden_size, 1, 1)
        self.potential_pos_head = FinalLayer(hidden_size, 1, 1)
        self.potential_head = FinalLayer(hidden_size, 1, 1)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        def basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(basic_init)
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.num_patches)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        for head in (self.x_head, self.g_head, self.potential_head, self.potential_x_head, self.potential_pos_head):
        # for head in (self.x_head, self.g_head, self.potential_head):
            nn.init.constant_(head.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(head.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(head.linear.weight, 0)
            nn.init.constant_(head.linear.bias, 0)

    @staticmethod
    def _expand_time(t: Tensor, batch_size: int, n_cells: int) -> Tensor:
        if t.ndim == 0:
            return t.reshape(1).expand(batch_size * n_cells)
        if t.ndim == 1:
            if t.numel() == batch_size:
                return t[:, None].expand(batch_size, n_cells).reshape(-1)
            if t.numel() == batch_size * n_cells:
                return t.reshape(-1)
        if t.ndim >= 2:
            return t.reshape(batch_size, -1)[:, :n_cells].reshape(-1)
        raise ValueError(f"Unable to expand time tensor with shape {tuple(t.shape)}.")

    def forward(
        self,
        pcs_mid: Tensor,
        pos_mid: Tensor,
        t: Tensor,
        is_joint_potential: bool = True,
        return_potential: bool | None = None,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        batch_size, n_cells, gene_dim = pcs_mid.shape
        if gene_dim != self.gene_dim:
            raise ValueError(f"Expected pcs_mid last dim {self.gene_dim}, found {gene_dim}.")

        gt = pcs_mid.reshape(batch_size * n_cells, gene_dim)
        xt = pos_mid.reshape(batch_size * n_cells, self.coord_dim)
        t_flat = self._expand_time(t, batch_size, n_cells).to(device=gt.device, dtype=gt.dtype)

        x_feat = self.x_embedder(xt).unsqueeze(1)
        g_feat = self.g_embedder(gt)
        h = torch.cat([x_feat, g_feat], dim=1) + self.pos_embed.to(device=gt.device, dtype=gt.dtype)

        cond = self.t_embedder(t_flat)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                h = checkpoint(block, h, cond, use_reentrant=False)
            else:
                h = block(h, cond)

        use_potential = self.use_potential if return_potential is None else return_potential
        if use_potential:
            if is_joint_potential:
                potential = self.potential_head(h.mean(dim=1, keepdim=True), cond).squeeze(-1).squeeze(-1)
                aux = {"potential": potential.reshape(batch_size, n_cells)}
            # potential_x = self.potential_x_head(h.mean(dim=1, keepdim=True), cond).squeeze(-1).squeeze(-1)
            # potential_pos = self.potential_pos_head(h.mean(dim=1, keepdim=True), cond).squeeze(-1).squeeze(-1)
            # aux = {"potential_x": potential_x.reshape(batch_size, n_cells), "potential_pos": potential_pos.reshape(batch_size, n_cells)}
                return gt.new_empty(0), xt.new_empty(0), aux
            else:
                potential_x = self.potential_x_head(h.mean(dim=1, keepdim=True), cond).squeeze(-1).squeeze(-1)
                potential_pos = self.potential_pos_head(h.mean(dim=1, keepdim=True), cond).squeeze(-1).squeeze(-1)
                aux = {"potential_x": potential_x.reshape(batch_size, n_cells), "potential_pos": potential_pos.reshape(batch_size, n_cells)}
                return gt.new_empty(0), xt.new_empty(0), aux
        else:
            vpos = self.x_head(h[:, :1, :], cond).squeeze(1)
            vx = self.g_head(h[:, 1:, :], cond).reshape(batch_size * n_cells, -1)[:, :gene_dim]

            return (
                vx.reshape(batch_size, n_cells, gene_dim),
                vpos.reshape(batch_size, n_cells, self.coord_dim),
                {},
            )

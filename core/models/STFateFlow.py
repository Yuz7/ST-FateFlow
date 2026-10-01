import torch
from torch import Tensor, nn
import torch.nn.functional as F

from core.datasets.model_dataset import TrainBatch
from core.models.backbones.pc_transformer import PointCloudTransformer


class STFateFlow_Module(nn.Module):
    def __init__(
        self,
        lambda_features: float,
        lambda_pos: float,
        lambda_cdm: float,
        backbone: PointCloudTransformer,
        num_steps: int = 20,
        use_potential: bool = False,
        potential_mode: str = "joint",
        spatial_loss_type: str = "mse"
    ) -> None:
        super().__init__()
        self.lambda_features = lambda_features
        self.lambda_pos = lambda_pos
        self.lambda_cdm = lambda_cdm
        self.backbone = backbone
        self.num_steps = num_steps
        self.use_potential = use_potential
        self.potential_mode = potential_mode
        self.spatial_loss_type = spatial_loss_type
        if self.use_potential:
            self.backbone.use_potential = True
            self.potential_mode = potential_mode

    def _predict_velocity(
        self,
        batch: TrainBatch,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        pcs_mid = batch["X_t"]
        pos_mid = batch["pos_t"]
        if self.use_potential:
            pcs_mid = pcs_mid.detach().requires_grad_(True)
            pos_mid = pos_mid.detach().requires_grad_(True)

        backbone_kwargs = {
            "pcs_mid": pcs_mid,
            "pos_mid": pos_mid,
            "t": batch["t"],
            "is_joint_potential": self.potential_mode == "joint",
            "return_potential": self.use_potential,
        }

        pred_vx, pred_vpos, aux = self.backbone(**backbone_kwargs)
        if self.use_potential:
            pred_vx, pred_vpos = self._velocity_from_potential(
                aux=aux,
                pcs_mid=pcs_mid,
                pos_mid=pos_mid,
                create_graph=self.training,
            )

        return pred_vx, pred_vpos, aux

    def _velocity_from_potential(
        self,
        aux: dict[str, Tensor],
        pcs_mid: Tensor,
        pos_mid: Tensor,
        create_graph: bool,
    ) -> tuple[Tensor, Tensor]:
        if self.potential_mode == "joint":
            grad_x, grad_pos = torch.autograd.grad(
                aux["potential"].sum(),
                (pcs_mid, pos_mid),
                create_graph=create_graph,
                retain_graph=create_graph,
            )
            return -grad_x, -grad_pos
        else:
            grad_x = torch.autograd.grad(
                aux["potential_x"].sum(),
                pcs_mid,
                create_graph=create_graph,
                retain_graph=True,
            )[0]
            
            grad_pos = torch.autograd.grad(
                aux["potential_pos"].sum(),
                pos_mid,
                create_graph=create_graph,
                retain_graph=create_graph,
            )[0]
            return -grad_x, -grad_pos

    def _predict_velocity_step(
        self,
        X_t: Tensor,
        pos_t: Tensor,
        t: Tensor,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        if self.use_potential:
            with torch.enable_grad():
                X_t = X_t.detach().requires_grad_(True)
                pos_t = pos_t.detach().requires_grad_(True)
                backbone_kwargs = {
                    "pcs_mid": X_t,
                    "pos_mid": pos_t,
                    "t": t,
                    "return_potential": True,
                    "is_joint_potential": self.potential_mode == "joint",
                }
                _, _, aux = self.backbone(**backbone_kwargs)
                vf_x, vf_pos = self._velocity_from_potential(
                    aux=aux,
                    pcs_mid=X_t,
                    pos_mid=pos_t,
                    create_graph=False,
                )
            return vf_x, vf_pos, aux

        backbone_kwargs = {
            "pcs_mid": X_t,
            "pos_mid": pos_t,
            "t": t,
            "return_potential": False,
        }

        vf_x, vf_pos, aux = self.backbone(**backbone_kwargs)
            
        return vf_x, vf_pos, aux

    def predict_potential(
        self,
        X_t: Tensor,
        pos_t: Tensor,
        t: Tensor | float,
        chunk_size: int = 256,
        output_device: str | torch.device = "cpu",
        potential_key: str | None = None,
    ) -> Tensor:
        compute_device = next(self.parameters()).device
        output_device = torch.device(output_device)
        batch_size, n_cells, _ = X_t.shape
        potential_key = potential_key or ("potential" if self.potential_mode == "joint" else "potential_x")

        was_training = self.training
        self.eval()
        potential_chunks = []
        with torch.no_grad():
            for start in range(0, n_cells, chunk_size):
                end = min(start + chunk_size, n_cells)
                x_chunk = X_t[:, start:end].to(compute_device)
                pos_chunk = pos_t[:, start:end].to(compute_device)

                if isinstance(t, Tensor):
                    if t.ndim >= 2:
                        t_chunk = t[:, start:end].to(compute_device)
                    else:
                        t_chunk = t.to(compute_device)
                else:
                    t_chunk = torch.full((batch_size,), float(t), device=compute_device, dtype=x_chunk.dtype)

                _, _, aux = self.backbone(
                    pcs_mid=x_chunk,
                    pos_mid=pos_chunk,
                    t=t_chunk,
                    return_potential=True,
                    is_joint_potential=self.potential_mode == "joint"
                )
                potential_chunks.append(aux[potential_key].detach().to(output_device))
                del x_chunk, pos_chunk, t_chunk, aux

        self.train(was_training)
        return torch.cat(potential_chunks, dim=1)

    @staticmethod
    def _predict_pos_t1(batch: TrainBatch, pred_vpos: Tensor) -> Tensor:
        remaining_t = (batch["t1_value"] - batch["t"])[:, None, None]
        return batch["pos_t"] + remaining_t * pred_vpos

    @staticmethod
    def _pairwise_min_dists(pred_pos: Tensor, target_pos: Tensor) -> tuple[Tensor, Tensor]:
        dmat = torch.cdist(pred_pos, target_pos, p=2)
        d_pred_to_target = dmat.min(dim=-1).values
        d_target_to_pred = dmat.min(dim=-2).values
        return d_pred_to_target, d_target_to_pred

    @staticmethod
    def _chamfer_from_min_dists(d_pred_to_target, d_target_to_pred):
        return d_pred_to_target.mean() + d_target_to_pred.mean()
    
    def loss(self, batch: TrainBatch) -> dict[str, Tensor]:
        pred_vx, pred_vpos, aux = self._predict_velocity(
            batch,
        )

        loss_x = F.mse_loss(pred_vx, batch["vf_x"])
        if self.spatial_loss_type == "mse":
            loss_pos = F.mse_loss(pred_vpos, batch["vf_pos"])
        else:
            loss_pos = F.smooth_l1_loss(pred_vpos, batch["vf_pos"], beta=0.1)
        del aux


        weighted_loss_x = self.lambda_features * loss_x
        weighted_loss_pos = self.lambda_pos * loss_pos

        if self.lambda_cdm != 0:
            pred_pos_t1 = self._predict_pos_t1(batch, pred_vpos)
            endpoint_target_pos = batch.get("pos_t1_dist", batch["pos_t1"])
            d_pred_to_target, d_target_to_pred = self._pairwise_min_dists(pred_pos_t1, endpoint_target_pos)
            loss_cdm = self._chamfer_from_min_dists(d_pred_to_target, d_target_to_pred)
            loss = (
                self.lambda_features * loss_x
                + self.lambda_pos * loss_pos
                + self.lambda_cdm * loss_cdm
            )
            weighted_loss_cdm = self.lambda_cdm * loss_cdm

            return {
                "loss": loss,
                "loss_x": loss_x,
                "loss_pos": loss_pos,
                "loss_cdm": loss_cdm,
                "weighted_loss_x": weighted_loss_x,
                "weighted_loss_pos": weighted_loss_pos,
                "weighted_loss_cdm": weighted_loss_cdm,
                "pred_vx_norm": pred_vx.detach().norm(dim=-1).mean(),
                "pred_vpos_norm": pred_vpos.detach().norm(dim=-1).mean(),
                "target_vx_norm": batch["vf_x"].detach().norm(dim=-1).mean(),
                "target_vpos_norm": batch["vf_pos"].detach().norm(dim=-1).mean(),
            }
        
        loss = (
                self.lambda_features * loss_x
                + self.lambda_pos * loss_pos
            )
        return {
                "loss": loss,
                "loss_x": loss_x,
                "loss_pos": loss_pos,
                # "loss_cdm": loss_cdm,
                "weighted_loss_x": weighted_loss_x,
                "weighted_loss_pos": weighted_loss_pos,
                # "weighted_loss_cdm": weighted_loss_cdm,
                "pred_vx_norm": pred_vx.detach().norm(dim=-1).mean(),
                "pred_vpos_norm": pred_vpos.detach().norm(dim=-1).mean(),
                "target_vx_norm": batch["vf_x"].detach().norm(dim=-1).mean(),
                "target_vpos_norm": batch["vf_pos"].detach().norm(dim=-1).mean(),
            }

    def sample(
        self,
        X_t0: Tensor,
        pos_t0: Tensor,
        t_start: float | Tensor = 0.0,
        t_end: float | Tensor = 1.0,
        keep_trajectory: bool = True,
        trajectory_device: str | torch.device | None = None,
        inference_chunk_size: int | None = None,
    ) -> dict[str, object]:
        compute_device = next(self.parameters()).device
        x_curr = X_t0.to(compute_device).clone()
        pos_curr = pos_t0.to(compute_device).clone()

        traj_device = torch.device(trajectory_device) if trajectory_device is not None else x_curr.device
        x_traj = [x_curr.detach().to(traj_device)] if keep_trajectory else []
        pos_traj = [pos_curr.detach().to(traj_device)] if keep_trajectory else []
        potential_traj = []
        
        t_start_t = torch.as_tensor(t_start, device=x_curr.device, dtype=x_curr.dtype)
        t_end_t = torch.as_tensor(t_end, device=x_curr.device, dtype=x_curr.dtype)
        t_span = torch.linspace(
            float(t_start_t.detach().cpu()),
            float(t_end_t.detach().cpu()),
            self.num_steps + 1,
            device=x_curr.device,
            dtype=x_curr.dtype,
        )

        for i in range(self.num_steps):
            t = t_span[i].repeat(x_curr.size(0))
            dt = t_span[i + 1] - t_span[i]

            if inference_chunk_size is None or inference_chunk_size >= x_curr.size(1):
                with torch.no_grad():
                    vf_x, vf_pos, aux = self._predict_velocity_step(
                        X_t=x_curr,
                        pos_t=pos_curr,
                        t=t,
                    )

                x_curr = (x_curr + dt * vf_x).detach()
                pos_curr = (pos_curr + dt * vf_pos).detach()
                del vf_x, vf_pos
                step_potential = aux.get("potential")
                del aux
            else:
                potential_chunks = []

                for start in range(0, x_curr.size(1), inference_chunk_size):
                    end = min(start + inference_chunk_size, x_curr.size(1))
                    x_chunk = x_curr[:, start:end]
                    pos_chunk = pos_curr[:, start:end]

                    with torch.no_grad():
                        vf_x, vf_pos, aux = self._predict_velocity_step(
                            X_t=x_chunk,
                            pos_t=pos_chunk,
                            t=t,
                        )

                    x_curr[:, start:end] = (x_chunk + dt * vf_x).detach()
                    pos_curr[:, start:end] = (pos_chunk + dt * vf_pos).detach()
                    del x_chunk, pos_chunk, vf_x, vf_pos

                    if "potential" in aux:
                        potential_chunks.append(aux["potential"].detach().to(traj_device))
                    elif "potential_x" in aux and "potential_pos" in aux:
                        potential_chunks.append((aux["potential_x"].detach().to(traj_device), aux["potential_pos"].detach().to(traj_device)))
                    del aux

                x_curr = x_curr.detach()
                pos_curr = pos_curr.detach()
                step_potential = torch.stack(potential_chunks, dim=1) if potential_chunks else None
                del potential_chunks

            if keep_trajectory:
                x_traj.append(x_curr.detach().to(traj_device))
                pos_traj.append(pos_curr.detach().to(traj_device))
            if step_potential is not None:
                potential_traj.append(step_potential.detach().to(traj_device))
            del step_potential
            
        out: dict[str, object] = {
            "x_traj": x_traj if keep_trajectory else x_curr.detach().to(traj_device),
            "pos_traj": pos_traj if keep_trajectory else pos_curr.detach().to(traj_device),
            "potential_traj": potential_traj
        }

        return out

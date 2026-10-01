from abc import ABC, abstractmethod
from typing import TypedDict

import torch
import torch.nn.functional as torchfunc
from torch import Tensor, nn


class FlowLosses(TypedDict):
    loss: Tensor
    loss_x: Tensor
    loss_pos: Tensor


class FlowLoss(nn.Module, ABC):
    def __init__(self, lambda_features: float, lambda_pos: float) -> None:
        super().__init__()
        self.lambda_features = lambda_features
        self.lambda_pos = lambda_pos

    @abstractmethod
    def forward(self, gt_x: Tensor, gt_pos: Tensor, pred_x: Tensor, pred_pos: Tensor) -> FlowLosses:
        raise NotImplementedError("The `forward` method of the loss has to be implemented!")


class BaseMSEFlowLoss(FlowLoss):
    """
    Base class for MSE-based flow losses over gene expressions (X) and positions (pos).

    Used by both:
    - Conditional Flow Matching (CFM)
    - Gaussian Variational Flow Matching (VFM)

    This class assumes the targets have already been computed correctly outside.
    """

    def forward(self, gt_x: Tensor, gt_pos: Tensor, pred_x: Tensor, pred_pos: Tensor) -> FlowLosses:
        loss_x = torchfunc.mse_loss(pred_x, gt_x) * self.lambda_features
        loss_pos = torchfunc.mse_loss(pred_pos, gt_pos) * self.lambda_pos
        loss = loss_x + loss_pos

        return {"loss": loss, "loss_x": loss_x, "loss_pos": loss_pos}


class CFMLoss(BaseMSEFlowLoss):
    """
    Loss for Conditional Flow Matching (CFM).

    Regresses the conditional vector field (the derivative of the interpolation).
    """

    pass


class GVFMLoss(BaseMSEFlowLoss):
    """
    Loss for Gaussian Variational Flow Matching (VFM).

    Regresses the final states directly.
    """

    pass

def chunked_chamfer_distance(x, y, chunk_size=1024):
    """
    大批次友好的纯 PyTorch Chamfer Distance 实现 (使用双向分块防 OOM)
    x: (B, N, D)
    y: (B, M, D)
    chunk_size: 每次计算的点数，根据您的 GPU 显存调节 (默认1024)
    """
    squeeze_output = False
    if x.ndim == 2:
        x = x.unsqueeze(0)  # (N, D) -> (1, N, D)
        y = y.unsqueeze(0)  # (M, D) -> (1, M, D)
        squeeze_output = True
    else:
        squeeze_output = False
        
    B, N, D = x.shape
    _, M, _ = y.shape
    
    # 初始化最小距离矩阵为无穷大
    min_dist_x_to_y = torch.full((B, N), float('inf'), device=x.device, dtype=x.dtype)
    min_dist_y_to_x = torch.full((B, M), float('inf'), device=x.device, dtype=x.dtype)

    # 外层循环：遍历 x 的分块
    for i in range(0, N, chunk_size):
        x_chunk = x[:, i:i+chunk_size, :]  # (B, C_x, D)
        
        # 内层循环：遍历 y 的分块
        for j in range(0, M, chunk_size):
            y_chunk = y[:, j:j+chunk_size, :]  # (B, C_y, D)
            
            # 计算局部距离矩阵 (B, C_x, C_y)
            # 使用 squared L2 distance (CD 通常使用平方欧氏距离)
            # 这里利用广播机制 (x-y)^2 快速计算，比 cdist 稍快但精度稍低，若需高精度可用 torch.cdist
            diff = x_chunk.unsqueeze(2) - y_chunk.unsqueeze(1)
            dist_sq = torch.sum(diff ** 2, dim=-1) # (B, C_x, C_y)
            
            # 更新 x 到 y 的最小距离
            min_dx, _ = torch.min(dist_sq, dim=2)
            min_dist_x_to_y[:, i:i+chunk_size] = torch.min(
                min_dist_x_to_y[:, i:i+chunk_size], 
                min_dx
            )
            
            # 更新 y 到 x 的最小距离
            min_dy, _ = torch.min(dist_sq, dim=1)
            min_dist_y_to_x[:, j:j+chunk_size] = torch.min(
                min_dist_y_to_x[:, j:j+chunk_size], 
                min_dy
            )

    # 最终 Loss 是两部分的平均值，单位是坐标单位的平方。
    loss = min_dist_x_to_y.mean() + min_dist_y_to_x.mean()
    if squeeze_output:
        loss = loss.squeeze(0)
    return loss

class GLVFMLoss(FlowLoss):
    def forward(self, gt_x: Tensor, gt_pos: Tensor, pred_x: Tensor, pred_pos: Tensor) -> FlowLosses:
        loss_x = torchfunc.mse_loss(input=pred_x, target=gt_x) * self.lambda_features
        loss_pos = torch.mean(torch.abs(pred_pos - gt_pos)) * self.lambda_pos
        loss = loss_x + loss_pos

        return {"loss": loss, "loss_x": loss_x, "loss_pos": loss_pos}


__all__ = ["CFMLoss", "FlowLoss", "FlowLosses", "GLVFMLoss", "GVFMLoss"]

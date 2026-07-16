import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def replace_bn_with_gn(root_module: nn.Module, features_per_group: int = 16) -> nn.Module:
    """Replace every BatchNorm2d submodule under ``root_module`` with GroupNorm."""
    replace_submodules(
        root_module=root_module,
        predicate=lambda module: isinstance(module, nn.BatchNorm2d),
        func=lambda module: nn.GroupNorm(
            num_groups=module.num_features // features_per_group,
            num_channels=module.num_features,
        ),
    )
    return root_module


def replace_submodules(
    root_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
    func: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    """Replace submodules matching ``predicate`` with the result of ``func``."""
    if predicate(root_module):
        return func(root_module)

    target_paths = [
        key.split(".")
        for key, module in root_module.named_modules(remove_duplicate=True)
        if predicate(module)
    ]
    for *parent, key in target_paths:
        parent_module = root_module.get_submodule(".".join(parent)) if parent else root_module
        source_module = parent_module[int(key)] if isinstance(parent_module, nn.Sequential) else getattr(parent_module, key)
        target_module = func(source_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(key)] = target_module
        else:
            setattr(parent_module, key, target_module)

    assert not any(predicate(module) for _, module in root_module.named_modules(remove_duplicate=True))
    return root_module


class PositionalEncoding(nn.Module):
    """Standard sine-cosine positional encoding for history tokens."""

    def __init__(self, d_model, max_seq_len=6):
        super().__init__()
        pos_enc = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("pos_enc", pos_enc.unsqueeze(0))

    def forward(self, x):
        return x + self.pos_enc[:, :x.size(1), :]


def clip_preprocess_tensor(rgb_tensor, n_px=224):
    """Resize and normalize a tensor batch the same way CLIP preprocess does."""
    if rgb_tensor.dtype != torch.float32:
        rgb_tensor = rgb_tensor.float()
    if rgb_tensor.max() > 1.0:
        rgb_tensor = rgb_tensor / 255.0

    rgb_resized = F.interpolate(rgb_tensor, size=(n_px, n_px), mode="bicubic", align_corners=False)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=rgb_resized.device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=rgb_resized.device).view(1, 3, 1, 1)
    return (rgb_resized - mean) / std


def action_reduce(unreduced_loss: torch.Tensor):
    """Average a loss tensor over all non-batch dimensions."""
    while unreduced_loss.dim() > 1:
        unreduced_loss = unreduced_loss.mean(dim=-1)
    return unreduced_loss.mean()


def traj_to_grid(traj, config):
    """Convert local metric trajectories from meters into normalized grid_sample coordinates."""
    grid_col = (-traj[:, :, 1] / config.grid_size) + (config.occupancy_width / 2)
    grid_row = traj[:, :, 0] / config.grid_size
    grid_traj = torch.stack([grid_col, grid_row], dim=-1)
    grid_traj[:, :, 0] = (grid_traj[:, :, 0] / (config.occupancy_width - 1)) * 2 - 1
    grid_traj[:, :, 1] = (grid_traj[:, :, 1] / (config.occupancy_height - 1)) * 2 - 1
    return grid_traj.unsqueeze(2)

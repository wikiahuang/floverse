import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from distmap import euclidean_signed_transform

from utils.data_utils import ACTION_STATS, get_action, get_delta, normalize_data
from utils.model_utils import traj_to_grid


class Refiner(nn.Module):
    """Refine planner trajectories with a local occupancy map built from current depth."""

    def __init__(self, config, accelerate=None):
        super().__init__()
        if accelerate is not None:
            self.accelerate = accelerate
            self.device = accelerate.device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.intrics = config.camera_intrinsics
        self.condition_dim = config.refiner_condition_dim
        self.num_diffusion_steps = config.num_diffusion_steps

        self.map_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.noise_pred_net = ConditionalUnet1D(
            input_dim=2,
            local_cond_dim=37,
            global_cond_dim=0,
            down_dims=config.down_dims,
            cond_predict_scale=config.cond_predict_scale,
        ).to(self.device)

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_steps,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

    def depth_to_occupancy(self, depth):
        """Convert a depth image into a local free-space occupancy map."""
        batch_size, _, height, width = depth.shape
        device = depth.device
        grid_size = self.config.grid_size
        max_range = self.config.get("occupancy_max_range", 4.0)
        floor_percentile = self.config.get("floor_percentile", 0.90)
        floor_tol = self.config.get("floor_tolerance", 0.1)
        extra_fov_deg = self.config.get("near_blindspot_extra_fov_deg", 15.0)
        width_occ = self.config.occupancy_width
        height_occ = self.config.occupancy_height
        fx, fy, cx, cy = (
            self.intrics.camera_fx,
            self.intrics.camera_fy,
            self.intrics.camera_cx,
            self.intrics.camera_cy,
        )

        y, x = torch.meshgrid(
            torch.arange(0, height, device=device),
            torch.arange(0, width, device=device),
            indexing="ij",
        )
        x, y = x.float(), y.float()
        raw = depth.squeeze(1)
        r = (raw / 255.0) * 10.0

        dx = (x - cx) / fx
        dy = (y - cy) / fy
        dz = torch.ones_like(dx)
        norm = torch.sqrt(dx**2 + dy**2 + dz**2)
        x_world = r * dx / norm
        y_world = r * dy / norm
        z_world = r * dz / norm

        valid = (z_world >= 0) & (z_world <= max_range) & (raw > 0)
        y_for_pct = torch.where(valid, y_world, torch.full_like(y_world, -1e6))
        floor_y = torch.quantile(y_for_pct.reshape(batch_size, -1), floor_percentile, dim=1)
        is_floor = valid & (torch.abs(y_world - floor_y.view(batch_size, 1, 1)) < floor_tol)

        occupancy_maps = torch.zeros((batch_size, height_occ, width_occ), device=device)
        batch_idx = torch.arange(batch_size, device=device).view(batch_size, 1, 1).expand(-1, height, width)
        gx = (x_world / grid_size).floor().long() + (width_occ // 2)
        gz = (z_world / grid_size).floor().long()
        in_bounds = is_floor & (gx >= 0) & (gx < width_occ) & (gz >= 0) & (gz < height_occ)
        occupancy_maps[batch_idx[in_bounds], gz[in_bounds], gx[in_bounds]] = 1.0

        z_floor_masked = torch.where(is_floor, z_world, torch.full_like(z_world, float("inf")))
        nearest_floor_z = z_floor_masked.min(dim=1).values

        margin_px = fx * np.tan(np.deg2rad(extra_fov_deg))
        n_fill_cols = width + 2 * int(margin_px) + 1
        fill_cols = torch.linspace(-margin_px, (width - 1) + margin_px, n_fill_cols, device=device)
        clamped_cols = fill_cols.clamp(0, width - 1).round().long()
        nz_lookup = nearest_floor_z[:, clamped_cols]

        bx = (fill_cols - cx) / fx
        bz = torch.ones_like(bx)
        bnorm = torch.sqrt(bx**2 + bz**2)
        bx, bz = bx / bnorm, bz / bnorm

        n_steps = int(2 * max_range / grid_size) + 1
        steps = torch.linspace(0, max_range, n_steps, device=device)

        fill_mask = (steps.view(1, 1, -1) < nz_lookup.unsqueeze(-1)) & torch.isfinite(nz_lookup).unsqueeze(-1)
        xe = (steps.view(1, 1, -1) * bx.view(1, -1, 1)).expand(batch_size, -1, -1)
        ze = (steps.view(1, 1, -1) * bz.view(1, -1, 1)).expand(batch_size, -1, -1)
        gxe = (xe / grid_size).floor().long() + (width_occ // 2)
        gze = (ze / grid_size).floor().long()
        batch_idx_e = torch.arange(batch_size, device=device).view(batch_size, 1, 1).expand(-1, n_fill_cols, n_steps)
        in_bounds_e = fill_mask & (gxe >= 0) & (gxe < width_occ) & (gze >= 0) & (gze < height_occ)
        occupancy_maps[batch_idx_e[in_bounds_e], gze[in_bounds_e], gxe[in_bounds_e]] = 1.0
        return occupancy_maps

    def _get_local_condition(self, traj_to_refine, distance_field_map):
        """Sample local map features and SDF values along a trajectory."""
        grad_y, grad_x = torch.gradient(distance_field_map, dim=(-2, -1))
        sdf_features_map = torch.stack([distance_field_map, grad_x, grad_y], dim=1)
        spatial_features = self.map_encoder(distance_field_map.unsqueeze(1))
        grid_traj_to_refine = traj_to_grid(traj_to_refine, self.config)
        sampled_features = F.grid_sample(
            spatial_features,
            grid_traj_to_refine,
            mode="bilinear",
            align_corners=True,
        )
        sampled_features = sampled_features.squeeze(-1).transpose(1, 2)
        sampled_sdf = F.grid_sample(
            sdf_features_map,
            grid_traj_to_refine,
            mode="bilinear",
            align_corners=True,
        )
        sampled_sdf = sampled_sdf.squeeze(-1).transpose(1, 2)
        return torch.cat([traj_to_refine, sampled_features, sampled_sdf], dim=-1)

    def forward(self, traj_to_refine, depth, traj_gt, idx, accelerate, mode):
        """Run refiner training or inference on a batch of planner trajectories."""
        del idx, accelerate
        batch_size = traj_to_refine.shape[0]
        occupancy_map = self.depth_to_occupancy(depth)

        distance_field_map = euclidean_signed_transform(occupancy_map)
        distance_field_map = torch.clamp(distance_field_map, min=-5.0, max=5.0) / 5.0
        local_cond_tensor = self._get_local_condition(traj_to_refine, distance_field_map)
        ndeltas = normalize_data(get_delta(traj_gt), ACTION_STATS).to(dtype=torch.float32)

        if mode == "train":
            noise = torch.randn(ndeltas.shape, device=self.device)
            time_steps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (batch_size,),
                device=self.device,
            ).long()
            noisy_deltas = self.noise_scheduler.add_noise(ndeltas, noise, time_steps).to(self.device)

            noise_pred = self.noise_pred_net(
                sample=noisy_deltas,
                timestep=time_steps,
                global_cond=None,
                local_cond=local_cond_tensor,
            )

            alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)
            alpha_prod_t = alphas_cumprod[time_steps].view(-1, 1, 1)
            diffusion_output = (noisy_deltas - (1 - alpha_prod_t).sqrt() * noise_pred) / alpha_prod_t.sqrt()
            diffusion_output = diffusion_output.clamp(-1.0, 1.0)
            actions = get_action(diffusion_output, ACTION_STATS)

            grid_traj = traj_to_grid(actions, self.config)
            sampled_distance_values = F.grid_sample(
                distance_field_map.unsqueeze(1),
                grid_traj,
                mode="bilinear",
                align_corners=True,
            ).squeeze(1).squeeze(2)

            _, traj_len = grid_traj.shape[:2]
            half = traj_len // 2
            time_weights = torch.cat(
                [
                    torch.full((half,), 1.0, device=grid_traj.device),
                    torch.full((traj_len - half,), 0.5, device=grid_traj.device),
                ]
            ).unsqueeze(0).expand(batch_size, traj_len)
            weighted_sampled_distance_values = sampled_distance_values * time_weights
            return noise_pred, noise, occupancy_map, weighted_sampled_distance_values, actions

        noise_eval = torch.randn_like(ndeltas).to(self.device)
        diffusion_output = noise_eval
        for timestep in self.noise_scheduler.timesteps:
            noise_pred_eval = self.noise_pred_net(
                sample=diffusion_output,
                timestep=timestep.repeat(batch_size).to(self.device),
                global_cond=None,
                local_cond=local_cond_tensor,
            )
            diffusion_output = self.noise_scheduler.step(
                model_output=noise_pred_eval,
                timestep=int(timestep),
                sample=diffusion_output,
            ).prev_sample
        actions = get_action(diffusion_output, ACTION_STATS)
        return actions, occupancy_map

    def inference(self, traj_to_refine, depth):
        """Refine a batch of planner trajectories without ground-truth supervision."""
        batch_size = traj_to_refine.shape[0]
        occupancy_map = self.depth_to_occupancy(depth)
        distance_field_map = euclidean_signed_transform(occupancy_map)
        distance_field_map = torch.clamp(distance_field_map, min=-5.0, max=5.0) / 5.0
        local_cond_tensor = self._get_local_condition(traj_to_refine, distance_field_map)

        diffusion_output = torch.randn(batch_size, traj_to_refine.shape[1], 2, device=self.device)
        for timestep in self.noise_scheduler.timesteps:
            noise_pred_eval = self.noise_pred_net(
                sample=diffusion_output,
                timestep=timestep.repeat(batch_size).to(self.device),
                global_cond=None,
                local_cond=local_cond_tensor,
            )
            diffusion_output = self.noise_scheduler.step(
                model_output=noise_pred_eval,
                timestep=int(timestep),
                sample=diffusion_output,
            ).prev_sample
        return get_action(diffusion_output, ACTION_STATS)

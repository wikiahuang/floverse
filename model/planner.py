import clip
import torch
import torch.nn as nn
from accelerate import Accelerator
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from efficientnet_pytorch import EfficientNet

from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from utils.data_utils import ACTION_STATS, get_action, get_delta, normalize_data
from utils.model_utils import PositionalEncoding, clip_preprocess_tensor, replace_bn_with_gn


class Planner(nn.Module):
    """Predict future local trajectories from RGB-D history, floorplan, and a goal condition."""

    def __init__(self, config, accelerate: Accelerator = None):
        super().__init__()
        self.accelerate = accelerate
        self.device = accelerate.device if accelerate is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.config = config

        self.point_feature_dim = config.point_feature_dim
        self.context_dim = config.context_dim
        self.goal_dim = config.goal_dim
        self.clip_feature_dim = config.clip_feature_dim
        self.condition_dim = config.condition_dim
        self.num_diffusion_steps = config.num_diffusion_steps

        self.rgb_encoder = replace_bn_with_gn(EfficientNet.from_name("efficientnet-b0", in_channels=3).to(self.device))
        self.depth_encoder = replace_bn_with_gn(EfficientNet.from_name("efficientnet-b0", in_channels=1).to(self.device))
        self.floorplan_encoder = replace_bn_with_gn(EfficientNet.from_name("efficientnet-b0", in_channels=1).to(self.device))
        self.img_goal_encoder = replace_bn_with_gn(EfficientNet.from_name("efficientnet-b0", in_channels=3).to(self.device))

        self.pose_encoder = nn.Linear(2, self.point_feature_dim).to(self.device)
        self.point_goal_encoder = nn.Linear(2, self.point_feature_dim).to(self.device)

        self.clip_encoder, _ = clip.load("ViT-B/32")
        for param in self.clip_encoder.parameters():
            param.requires_grad = False
        self.clip_encoder = self.clip_encoder.to(self.device)
        self.clip_encoder.eval()

        self.compression_layer = nn.Linear(
            self.rgb_encoder._fc.in_features + self.depth_encoder._fc.in_features + self.point_feature_dim,
            self.context_dim,
        ).to(self.device)
        self.positional_encoding = PositionalEncoding(self.context_dim, max_seq_len=config.history_length).to(self.device)
        self.sa_layer = nn.TransformerEncoderLayer(
            d_model=self.context_dim,
            nhead=config.mha_num_attention_heads,
            dim_feedforward=config.mha_ff_dim_factor * self.context_dim,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        ).to(self.device)
        self.sa_encoder = nn.TransformerEncoder(self.sa_layer, num_layers=config.mha_num_attention_layers).to(self.device)

        self.projecter_image_goal = nn.Linear(self.rgb_encoder._fc.in_features * 2, self.goal_dim).to(self.device)
        self.projecter_point_goal = nn.Linear(self.point_feature_dim * 2, self.goal_dim).to(self.device)
        self.projecter_object_goal = nn.Linear(self.clip_feature_dim * 2, self.goal_dim).to(self.device)
        self.condition_compressor = nn.Linear(
            self.context_dim + self.goal_dim + self.floorplan_encoder._fc.in_features,
            self.condition_dim,
        ).to(self.device)

        self.noise_pred_net = ConditionalUnet1D(
            input_dim=2,
            local_cond_dim=None,
            global_cond_dim=self.condition_dim,
            down_dims=config.down_dims,
            cond_predict_scale=config.cond_predict_scale,
        ).to(self.device)
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_steps,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

    def _encode_backbone_sequence(self, tensor, encoder):
        batch_size, seq_len = tensor.shape[:2]
        flat_tensor = tensor.reshape(batch_size * seq_len, *tensor.shape[2:])
        embedding = encoder.extract_features(flat_tensor)
        embedding = encoder._avg_pooling(embedding)
        if encoder._global_params.include_top:
            embedding = embedding.flatten(start_dim=1)
            embedding = encoder._dropout(embedding)
        return embedding.reshape(batch_size, seq_len, -1)

    def _encode_single_backbone(self, tensor, encoder):
        embedding = encoder.extract_features(tensor)
        embedding = encoder._avg_pooling(embedding)
        if encoder._global_params.include_top:
            embedding = embedding.flatten(start_dim=1)
            embedding = encoder._dropout(embedding)
        return embedding

    def _encode_history(self, rgb, depth, history_pose):
        rgb_embedding = self._encode_backbone_sequence(rgb, self.rgb_encoder)
        depth_embedding = self._encode_backbone_sequence(depth, self.depth_encoder)
        pose_embedding = self.pose_encoder(history_pose)

        current_obs_feature = torch.cat([rgb_embedding, depth_embedding, pose_embedding], dim=-1)
        current_obs_feature = self.compression_layer(current_obs_feature)
        context_feature = self.positional_encoding(current_obs_feature)
        context_feature = self.sa_encoder(context_feature)
        context_feature = torch.mean(context_feature, dim=1)
        return rgb_embedding, pose_embedding, context_feature

    def _encode_floor_plan(self, floor_plan):
        return self._encode_single_backbone(floor_plan, self.floorplan_encoder)

    def _encode_image_goal(self, last_rgb_embedding, img_goal):
        img_embedding = self._encode_single_backbone(img_goal, self.img_goal_encoder)
        return self.projecter_image_goal(torch.cat([last_rgb_embedding, img_embedding], dim=-1))

    def _encode_point_goal(self, last_pose_embedding, point_goal):
        point_embedding = self.point_goal_encoder(point_goal)
        return self.projecter_point_goal(torch.cat([last_pose_embedding, point_embedding.squeeze(1)], dim=-1))

    def _encode_object_goal(self, last_rgb, object_goal):
        object_rgb_preprocessed = clip_preprocess_tensor(last_rgb)
        object_rgb_embedding = self.clip_encoder.encode_image(object_rgb_preprocessed)
        object_text = clip.tokenize(list(object_goal)).to(self.device)
        object_text_embedding = self.clip_encoder.encode_text(object_text)
        object_goal_embedding = torch.cat([object_rgb_embedding, object_text_embedding], dim=-1).float()
        return self.projecter_object_goal(object_goal_embedding)

    def _sample_training_goal_embedding(self, rgb, rgb_embedding, pose_embedding, img_goal, point_goal, object_goal):
        batch_size = rgb.shape[0]
        img_prob = float(getattr(self.config, "img_goal_train_prob", 1.0 / 3.0))
        point_prob = float(getattr(self.config, "point_goal_train_prob", 1.0 / 3.0))
        random_nums = torch.rand(batch_size, device=self.device)
        has_obj = torch.tensor([obj != "null" for obj in object_goal], device=self.device)
        no_obj_denom = max(img_prob + point_prob, 1e-6)
        img_threshold = torch.where(
            has_obj,
            torch.full((batch_size,), img_prob, device=self.device),
            torch.full((batch_size,), img_prob / no_obj_denom, device=self.device),
        )
        point_threshold = torch.where(
            has_obj,
            torch.full((batch_size,), img_prob + point_prob, device=self.device),
            torch.ones(batch_size, device=self.device),
        )

        goal_embedding = torch.zeros((batch_size, self.goal_dim), device=self.device)
        img_indices = torch.where(random_nums < img_threshold)[0]
        point_indices = torch.where((random_nums >= img_threshold) & (random_nums < point_threshold))[0]
        object_indices = torch.where(has_obj & (random_nums >= point_threshold))[0]

        if len(img_indices) > 0:
            goal_embedding[img_indices] = self._encode_image_goal(rgb_embedding[img_indices][:, -1, :], img_goal[img_indices])
        if len(point_indices) > 0:
            goal_embedding[point_indices] = self._encode_point_goal(pose_embedding[point_indices][:, -1, :], point_goal[point_indices])
        if len(object_indices) > 0:
            object_batch = [object_goal[i] for i in object_indices.tolist()]
            goal_embedding[object_indices] = self._encode_object_goal(rgb[object_indices][:, -1], object_batch)
        return goal_embedding

    def _build_condition(self, context_feature, floor_plan_embedding, goal_embedding):
        condition_feature = torch.cat([context_feature, goal_embedding, floor_plan_embedding], dim=-1)
        return self.condition_compressor(condition_feature)

    def _sample_trajectory(self, condition_feature, batch_size, traj_len):
        diffusion_output = torch.randn((batch_size, traj_len, 2), device=self.device)
        for timestep in self.noise_scheduler.timesteps:
            noise_pred = self.noise_pred_net(
                sample=diffusion_output,
                global_cond=condition_feature,
                timestep=timestep.repeat(batch_size).to(self.device),
            )
            diffusion_output = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=int(timestep),
                sample=diffusion_output,
            ).prev_sample
        return get_action(diffusion_output, ACTION_STATS)

    def inference(self, rgb, depth, history_pose, floor_plan, img_goal, point_goal, object_goal):
        """Run one-goal inference for image, point, or object navigation."""
        rgb = rgb.to(self.device)
        depth = depth.to(self.device)
        history_pose = history_pose.to(self.device)
        floor_plan = floor_plan.to(self.device)
        img_goal = img_goal.to(self.device) if img_goal is not None else None
        point_goal = point_goal.to(self.device) if point_goal is not None else None

        rgb_embedding, pose_embedding, context_feature = self._encode_history(rgb, depth, history_pose)
        floor_plan_embedding = self._encode_floor_plan(floor_plan)

        if img_goal is not None:
            goal_embedding = self._encode_image_goal(rgb_embedding[:, -1, :], img_goal)
        elif point_goal is not None:
            goal_embedding = self._encode_point_goal(pose_embedding[:, -1, :], point_goal)
        elif object_goal is not None:
            goal_embedding = self._encode_object_goal(rgb[:, -1], object_goal)
        else:
            raise ValueError("One goal condition must be provided for inference.")

        condition_feature = self._build_condition(context_feature, floor_plan_embedding, goal_embedding)
        return self._sample_trajectory(condition_feature, rgb.shape[0], self.config.len_traj_pred)

    def forward(self, rgb, depth, history_pose, floor_plan, img_goal, point_goal, object_goal, gt_pose, mode):
        """Run planner training or rollout mode on a batch."""
        rgb = rgb.to(self.device)
        depth = depth.to(self.device)
        history_pose = history_pose.to(self.device)
        floor_plan = floor_plan.to(self.device)
        img_goal = img_goal.to(self.device)
        point_goal = point_goal.to(self.device)
        gt_pose = gt_pose.to(self.device)
        object_goal = list(object_goal)

        rgb_embedding, pose_embedding, context_feature = self._encode_history(rgb, depth, history_pose)
        floor_plan_embedding = self._encode_floor_plan(floor_plan)
        goal_embedding = self._sample_training_goal_embedding(
            rgb,
            rgb_embedding,
            pose_embedding,
            img_goal,
            point_goal,
            object_goal,
        )
        condition_feature = self._build_condition(context_feature, floor_plan_embedding, goal_embedding)

        deltas = get_delta(gt_pose)
        ndeltas = normalize_data(deltas, ACTION_STATS).to(dtype=torch.float32)
        if mode == "train":
            noise = torch.randn_like(ndeltas)
            time_steps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (rgb.shape[0],), device=self.device).long()
            noisy_deltas = self.noise_scheduler.add_noise(ndeltas, noise, time_steps)
            noise_pred = self.noise_pred_net(
                sample=noisy_deltas,
                timestep=time_steps,
                global_cond=condition_feature,
                local_cond=None,
            )
            return noise_pred, noise

        if mode == "eval_1":
            actions = self._sample_trajectory(condition_feature, rgb.shape[0], ndeltas.shape[1])
            return actions, gt_pose

        raise ValueError(f"Unsupported planner mode: {mode}")

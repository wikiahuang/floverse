import hashlib
import json
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

DEFAULT_OVERLAP_SCENE_PATH = Path(__file__).resolve().parent / "train_eval_overlap_scenes.txt"
LOGGER = logging.getLogger(__name__)


class NavDataset(Dataset):
    """Load navigation training samples from per-scene trajectory folders."""

    def __init__(
        self,
        data_folder,
        len_traj_pred,
        traj_pace,
        history_length,
        need_currdep,
        overlap_scene_path=DEFAULT_OVERLAP_SCENE_PATH,
    ):
        self.data_folder = data_folder
        self.len_traj_pred = len_traj_pred
        self.traj_pace = traj_pace
        self.history_length = history_length
        self.need_currdep = need_currdep
        self.overlap_scene_path = Path(overlap_scene_path) if overlap_scene_path else None
        self.transform = transforms.Compose([transforms.Resize((96, 96)), transforms.ToTensor()])
        self.samples_index = self.build_index()
        LOGGER.info("Dataset length: %d", len(self.samples_index))

    def read_traj_txt(self, path):
        """Parse a trajectory text file into a list of 4D poses."""
        num_pos = 0
        all_pose = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                num_pos += 1
                step_pose = list(map(float, line.strip().split()))
                if len(step_pose) != 4:
                    LOGGER.warning("Skipping malformed pose line %d in %s: expected 4 values, got %d", num_pos, path, len(step_pose))
                    num_pos -= 1
                    continue
                all_pose.append(step_pose)
        return num_pos, all_pose

    def read_object_json(self, path):
        """Read a trajectory object-goal annotation and return its category string."""
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            data = data[0] if data else {}
        category = data.get("object_category", "null")
        if category == "null":
            LOGGER.warning("Object annotation missing category in %s", path)
        return category

    def _data_folders(self):
        return [self.data_folder] if isinstance(self.data_folder, str) else list(self.data_folder)

    def _overlap_scenes(self):
        if not self.overlap_scene_path or not self.overlap_scene_path.exists():
            return set()
        with open(self.overlap_scene_path, "r", encoding="utf-8") as handle:
            return {line.strip() for line in handle if line.strip()}

    def build_index(self):
        """Build or load the cached sample index for the current dataset configuration."""
        folder_str = self.data_folder if isinstance(self.data_folder, str) else "_".join(sorted(self.data_folder))
        config_str = f"{folder_str}_{self.len_traj_pred}_{self.traj_pace}_{self.history_length}_{self.need_currdep}"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()
        cache_dir = self.data_folder if isinstance(self.data_folder, str) else list(self.data_folder)[0]
        cache_path = os.path.join(cache_dir, f"dataset_index_cache_{config_hash}.pkl")

        if os.path.exists(cache_path):
            LOGGER.info("Loading dataset index from cache: %s", cache_path)
            with open(cache_path, "rb") as handle:
                return pickle.load(handle)

        distributed = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if distributed else 0
        world_size = dist.get_world_size() if distributed else 1
        overlap_scenes = self._overlap_scenes()

        LOGGER.info("Building dataset index [rank %d/%d]", rank, world_size)
        samples_index_shard = []
        for data_fd in self._data_folders():
            scenes = sorted(os.listdir(data_fd))
            for scene in tqdm(scenes[rank::world_size], desc=f"Scanning {data_fd} [rank {rank}/{world_size}]"):
                scene_info = os.path.join(data_fd, scene)
                if not os.path.isdir(scene_info):
                    continue
                if scene.rsplit("_", 1)[0] in overlap_scenes:
                    continue

                for traj_name in sorted(os.listdir(scene_info)):
                    if traj_name == "floorplan.png":
                        continue
                    trajdir_path = os.path.join(scene_info, traj_name)
                    trajinfo_path = os.path.join(trajdir_path, f"{traj_name}.txt")
                    traj_len, _ = self.read_traj_txt(trajinfo_path)
                    for start_idx in range(0, traj_len, self.traj_pace):
                        if start_idx + self.len_traj_pred > traj_len:
                            break
                        samples_index_shard.append(f"{scene}::{traj_name}::{start_idx}|{data_fd}")

        if distributed:
            gathered_shards = [None] * world_size
            dist.all_gather_object(gathered_shards, samples_index_shard)
            samples_index = [item for shard in gathered_shards for item in shard]
        else:
            samples_index = samples_index_shard

        if rank == 0:
            try:
                with open(cache_path, "wb") as handle:
                    pickle.dump(samples_index, handle)
                LOGGER.info("Saved dataset index cache to: %s", cache_path)
            except Exception as exc:
                LOGGER.warning("Failed to save dataset index cache to %s: %s", cache_path, exc)
        return samples_index

    def yaw_rotmat(self, yaw):
        return np.array(
            [
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw), np.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ],
        )

    def get_yaw(self, x, y, x1, y1):
        dx = x1 - x
        dy = y1 - y
        if dx == 0 and dy == 0:
            raise ValueError("The current point coincides with the target point, so the orientation cannot be computed.")
        return np.arctan2(dy, dx)

    def to_local_coords(self, positions, curr_pos, curr_yaw):
        """Convert global positions into the agent-local coordinate frame."""
        positions = np.array(positions)
        curr_pos = np.array(curr_pos)
        rotmat = self.yaw_rotmat(float(curr_yaw))
        if positions.shape[-1] == 2:
            rotmat = rotmat[:2, :2]
        elif positions.shape[-1] != 3:
            raise ValueError("positions must be 2D or 3D.")
        return (positions - curr_pos).dot(rotmat)

    def _compute_actions(self, traj_data):
        traj_data = np.array(traj_data)
        positions = traj_data[:, :2].copy()
        cur_pos = positions[0]
        yaw = self.get_yaw(traj_data[0][0], traj_data[0][1], traj_data[0][2], traj_data[0][3])
        cur_ori = cur_pos + np.array([np.cos(yaw), np.sin(yaw)])
        waypoints = self.to_local_coords(positions, positions[0], yaw)
        return waypoints, cur_pos, cur_ori

    def _load_sorted_frames(self, frame_dir, convert_rgb=False):
        frame_names = sorted(
            [name for name in os.listdir(frame_dir) if name.endswith(".png")],
            key=lambda name: int(name.split("_")[1].split(".")[0]),
        )
        frames = []
        for name in frame_names:
            image = Image.open(os.path.join(frame_dir, name))
            if convert_rgb:
                image = image.convert("RGB")
            frames.append(self.transform(image))
        return frame_names, frames

    def _history_window(self, frames, start_idx):
        if start_idx - self.history_length < 0:
            prefix = [frames[0]] * (self.history_length - start_idx)
            return prefix + frames[:start_idx]
        return frames[start_idx - self.history_length:start_idx]

    def _pose_history_window(self, all_pose, start_idx):
        if start_idx - self.history_length < 0:
            first_pose = all_pose[0]
            return [first_pose for _ in range(self.history_length - start_idx)] + all_pose[: start_idx + 1]
        return all_pose[start_idx - self.history_length : start_idx + 1]

    def _parse_index_item(self, idx):
        item_str = self.samples_index[idx]
        item_info, data_fd = item_str.split("|") if "|" in item_str else (item_str, self._data_folders()[0])
        scene, traj_name, start_idx = item_info.split("::")
        return item_info, data_fd, scene, traj_name, int(start_idx)

    def _history_pose_tensor(self, all_pose, start_idx):
        history_pose = self._pose_history_window(all_pose, start_idx)
        history_pose.reverse()
        history_pose_numpy, _, _ = self._compute_actions(history_pose)
        return torch.tensor(history_pose_numpy[1:].tolist()[::-1], dtype=torch.float32)[:, :2]

    def _future_pose_tensor(self, all_pose, start_idx):
        future_pose, _, _ = self._compute_actions(all_pose[start_idx : start_idx + self.len_traj_pred])
        return torch.tensor(future_pose, dtype=torch.float32)[:, :2]

    def _point_goal_tensor(self, all_pose, start_idx):
        point_goal = all_pose[-1]
        curr_pos = all_pose[start_idx]
        curr_yaw = self.get_yaw(curr_pos[0], curr_pos[1], curr_pos[2], curr_pos[3])
        point_goal = self.to_local_coords(point_goal[:2], curr_pos[:2], curr_yaw)
        return torch.tensor(point_goal, dtype=torch.float32).unsqueeze(0)[:, :2]

    def __len__(self):
        return len(self.samples_index)

    def __getitem__(self, idx):
        """Load one training sample with history observations, goals, and target trajectory."""
        item_info, data_fd, scene, traj_name, start_idx = self._parse_index_item(idx)
        scene_dir = os.path.join(data_fd, scene)

        floorplan = self.transform(Image.open(os.path.join(scene_dir, "floorplan.png")))
        traj_dir = os.path.join(scene_dir, traj_name)

        object_goal = "null"
        rgb_dir = os.path.join(traj_dir, "rgb")
        depth_dir = os.path.join(traj_dir, "depth")
        traj_txt_path = os.path.join(traj_dir, f"{traj_name}.txt")
        object_json_path = os.path.join(traj_dir, "object", "object.json")

        if os.path.exists(object_json_path):
            object_goal = self.read_object_json(object_json_path)

        _, rgb_frames = self._load_sorted_frames(rgb_dir, convert_rgb=True)
        _, depth_frames = self._load_sorted_frames(depth_dir, convert_rgb=False)
        _, all_pose = self.read_traj_txt(traj_txt_path)

        image_goal = rgb_frames[-1]
        stack_rgb = torch.stack(self._history_window(rgb_frames, start_idx), dim=0)
        stack_depth = torch.stack(self._history_window(depth_frames, start_idx), dim=0)

        current_depth = None
        if self.need_currdep:
            current_depth = np.array(Image.open(os.path.join(depth_dir, f"depth_{start_idx}.png")))

        point_goal = self._point_goal_tensor(all_pose, start_idx)
        history_pose = self._history_pose_tensor(all_pose, start_idx)
        future_pose = self._future_pose_tensor(all_pose, start_idx)

        if self.need_currdep:
            return (
                torch.as_tensor(current_depth, dtype=torch.float32),
                stack_rgb.to(dtype=torch.float32),
                stack_depth.to(dtype=torch.float32),
                history_pose,
                future_pose,
                floorplan.to(dtype=torch.float32),
                image_goal.to(dtype=torch.float32),
                point_goal,
                object_goal,
                item_info,
            )

        return (
            stack_rgb.to(dtype=torch.float32),
            stack_depth.to(dtype=torch.float32),
            history_pose,
            future_pose,
            floorplan.to(dtype=torch.float32),
            image_goal.to(dtype=torch.float32),
            point_goal,
            object_goal,
            item_info,
        )

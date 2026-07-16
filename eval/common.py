import argparse
import json
import logging
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pybullet as p
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import yaml
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont

import igibson
from igibson.envs.igibson_env import iGibsonEnv
from igibson.robots.manipulation_robot import ManipulationRobot
from igibson.simulator import Simulator

from distmap import euclidean_signed_transform
from model import Planner, Refiner
from utils import traj_to_grid


LOGGER = logging.getLogger("floverse.eval")
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
])
COMMON_EXCLUDED_SCENES = {
    "00094-WT4QWwXrMzs_0",
    "00150-LcAd9dhvVwh_0",
    "00189-KHhgcNqsc9h_0",
    "00209-C5RbHBQ76DE_0",
    "00525-iKFn6fzyRqs_0",
    "00810-CrMo8WxCyVb_0",
    "00832-qyAac8rV8Zk_0",
}
IMAGE_ONLY_EXCLUDED_SCENES = {"00007-UQuchpekHRJ_0"}
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"
DEFAULT_MODEL_CONFIG_PATH = REPO_ROOT / "config" / "floverse.yaml"
DEFAULT_PLANNER_CKPT_PATH = REPO_ROOT / "ckpt" / "planner.pth"
DEFAULT_REFINER_CKPT_PATH = REPO_ROOT / "ckpt" / "refiner.pth"
DEFAULT_ENV_CONFIG_PATH = REPO_ROOT / "config" / "locobot_floverse_nav.yaml"
DEFAULT_GIBSON_DATASET_ROOT = REPO_ROOT / "iGibson" / "igibson" / "data" / "g_dataset"
_DEFAULT_FALLBACK_PATH_INDEX = 0
PANEL_SIZE = 320
PANEL_GUTTER = 44
PANEL_OUTER = 16
PANEL_TOP_MARGIN = 44
PANEL_BOTTOM_MARGIN = 6


def setup_logging():
    """Initialize a compact process-safe logger for evaluation scripts."""
    if LOGGER.handlers:
        return LOGGER
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", "%H:%M:%S"))
    LOGGER.setLevel(logging.INFO)
    LOGGER.addHandler(handler)
    LOGGER.propagate = False
    return LOGGER


def apply_eval_cli_overrides(eval_config, default_scene_ids=None):
    """Override eval config from CLI while preserving script defaults when flags are omitted."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes_dir", type=str)
    parser.add_argument("--scene_ids", nargs="*")
    parser.add_argument("--model_config", type=str)
    parser.add_argument("--planner_ckpt_path", type=str)
    parser.add_argument("--refiner_ckpt_path", type=str)
    parser.add_argument("--traj_save_dir", type=str)
    parser.add_argument("--execute_steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--collision_limit", type=int)
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--max_trajs_per_scene", type=int)
    parser.add_argument("--use_refiner", dest="use_refiner", action="store_true")
    parser.add_argument("--no_refiner", dest="use_refiner", action="store_false")
    parser.add_argument("--save_images", dest="save_images", action="store_true")
    parser.add_argument("--no_save_images", dest="save_images", action="store_false")
    parser.set_defaults(use_refiner=None, save_images=None)
    args = parser.parse_args()

    overrides = {
        "scenes_dir": args.scenes_dir,
        "scene_ids": args.scene_ids,
        "model_config": args.model_config,
        "planner_ckpt_path": args.planner_ckpt_path,
        "refiner_ckpt_path": args.refiner_ckpt_path,
        "traj_save_dir": args.traj_save_dir,
        "execute_steps": args.execute_steps,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "collision_limit": args.collision_limit,
        "max_steps": args.max_steps,
        "max_trajs_per_scene": args.max_trajs_per_scene,
        "use_refiner": args.use_refiner,
        "save_images": args.save_images,
    }
    for key, value in overrides.items():
        if value is not None:
            eval_config[key] = value

    if "scene_ids" not in eval_config:
        eval_config["scene_ids"] = default_scene_ids
    return eval_config


def _load_font(size):
    """Load a readable TrueType font when available, otherwise fall back to PIL default."""
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def load_floorplan_transform(scene_name, floor_id, dataset_root=DEFAULT_GIBSON_DATASET_ROOT):
    """Load the world-coordinate to floorplan-pixel transform for one scene floor."""
    scene_path = os.path.join(dataset_root, scene_name)
    scale_file = os.path.join(scene_path, "scale.txt")
    offset_file = os.path.join(scene_path, "offset.txt")

    scale = 1.0
    if os.path.exists(scale_file):
        with open(scale_file, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        if floor_id < len(lines):
            scale = float(lines[floor_id])

    offset_x, offset_y = 0.0, 0.0
    if os.path.exists(offset_file):
        with open(offset_file, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        if floor_id < len(lines):
            parts = lines[floor_id].split()
            if len(parts) >= 2:
                offset_x = float(parts[0])
                offset_y = float(parts[1])

    return scale, offset_x, offset_y


def load_floor_height(scene_name, floor_id, dataset_root=DEFAULT_GIBSON_DATASET_ROOT):
    """Load the absolute world z height for one scene floor."""
    floors_file = os.path.join(dataset_root, scene_name, "floors.txt")
    if not os.path.exists(floors_file):
        return 0.0
    with open(floors_file, "r") as f:
        heights = [float(line.strip()) for line in f.readlines() if line.strip()]
    if floor_id < len(heights):
        return heights[floor_id]
    return 0.0


def world_to_floorplan(x, y, scale, offset_x, offset_y):
    """Convert iGibson world XY coordinates to floorplan pixel coordinates."""
    return int(x * scale + offset_x), int(y * scale + offset_y)


def choose_fallback_path_index(path_length, fallback_path_index=_DEFAULT_FALLBACK_PATH_INDEX):
    """Select a fallback waypoint along the shortest path."""
    return min(fallback_path_index, max(path_length - 1, 0))


def _draw_floorplan_marker(draw, xy, radius, fill, outline=None):
    outline = outline or fill
    x, y = xy
    draw.ellipse(
        [x - radius, y - radius, x + radius, y + radius],
        fill=fill,
        outline=outline,
        width=max(1, radius // 3),
    )


def _to_rgb_image(image):
    """Convert a PIL image, numpy array, or CHW/BCHW torch tensor to a PIL RGB image."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if torch.is_tensor(image):
        image = image.detach().cpu()
        while image.ndim > 3:
            image = image[0]
        if image.ndim == 3 and image.shape[0] in (1, 3):
            image = image.permute(1, 2, 0)
        image = image.numpy()
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.shape[-1] > 3:
        image = image[:, :, :3]
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return Image.fromarray(image).convert("RGB")


def _draw_local_traj_panel(
    history_local_positions=None,
    predicted_local_positions=None,
    size=320,
):
    """Draw a local-coordinate trajectory panel with ticks and axes."""
    panel = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(panel)
    body_font = _load_font(18)
    margin_left = 42
    margin_right = 18
    margin_top = 18
    margin_bottom = 42

    points = [np.array([[0.0, 0.0]], dtype=np.float32)]
    if history_local_positions is not None and len(history_local_positions) > 0:
        points.append(np.asarray(history_local_positions)[:, :2])
    if predicted_local_positions is not None and len(predicted_local_positions) > 0:
        points.append(np.asarray(predicted_local_positions)[:, :2])
    all_points = np.concatenate(points, axis=0)

    xmin, ymin = all_points.min(axis=0)
    xmax, ymax = all_points.max(axis=0)
    span_x = max(xmax - xmin, 4.0)
    span_y = max(ymax - ymin, 4.0)
    pad_x = span_x * 0.15
    pad_y = span_y * 0.15
    xmin, xmax = xmin - pad_x, xmax + pad_x
    ymin, ymax = ymin - pad_y, ymax + pad_y
    x_span = max(xmax - xmin, 1e-6)
    y_span = max(ymax - ymin, 1e-6)
    plot_w = size - margin_left - margin_right
    plot_h = size - margin_top - margin_bottom

    def to_px(position):
        x, y = float(position[0]), float(position[1])
        px = margin_left + (x - xmin) / x_span * plot_w
        py = size - margin_bottom - (y - ymin) / y_span * plot_h
        return px, py

    draw.rectangle(
        [margin_left, margin_top, size - margin_right, size - margin_bottom],
        outline=(70, 70, 70),
        width=1,
    )

    def tick_values(vmin, vmax):
        step = max((vmax - vmin) / 4.0, 0.5)
        step = np.ceil(step * 2.0) / 2.0
        start = np.ceil(vmin / step) * step
        values = []
        value = start
        while value <= vmax + 1e-6:
            values.append(round(float(value), 2))
            value += step
        return values

    for x_tick in tick_values(xmin, xmax):
        tx, _ = to_px((x_tick, 0.0))
        draw.line([(tx, size - margin_bottom), (tx, size - margin_bottom + 6)], fill=(70, 70, 70), width=1)
        draw.text((tx - 10, size - margin_bottom + 10), f"{x_tick:g}", fill=(60, 60, 60))

    for y_tick in tick_values(ymin, ymax):
        _, ty = to_px((0.0, y_tick))
        draw.line([(margin_left - 6, ty), (margin_left, ty)], fill=(70, 70, 70), width=1)
        draw.text((6, ty - 6), f"{y_tick:g}", fill=(60, 60, 60))

    ox, oy = to_px((0.0, 0.0))
    if ymin <= 0.0 <= ymax:
        draw.line([(margin_left, oy), (size - margin_right, oy)], fill=(200, 200, 200), width=2)
    if xmin <= 0.0 <= xmax:
        draw.line([(ox, margin_top), (ox, size - margin_bottom)], fill=(200, 200, 200), width=2)

    if history_local_positions is not None and len(history_local_positions) > 0:
        history_points = [to_px(p) for p in np.asarray(history_local_positions)[:, :2]]
        if len(history_points) >= 2:
            draw.line(history_points, fill=(32, 96, 255), width=5)
        for point in history_points:
            _draw_floorplan_marker(draw, point, 4, (32, 96, 255))

    if predicted_local_positions is not None and len(predicted_local_positions) > 0:
        pred_points = [to_px(p) for p in np.asarray(predicted_local_positions)[:, :2]]
        if len(pred_points) >= 2:
            for p0, p1 in zip(pred_points[:-1], pred_points[1:]):
                draw.line([p0, p1], fill=(255, 64, 64), width=5)
        for point in pred_points:
            _draw_floorplan_marker(draw, point, 4, (255, 64, 64))

    _draw_floorplan_marker(draw, (ox, oy), 7, (255, 220, 64), (0, 0, 0))
    draw.text((size // 2 - 8, size - 24), "x", fill=(60, 60, 60), font=body_font)
    draw.text((16, size // 2 - 8), "y", fill=(60, 60, 60), font=body_font)
    return panel


def save_eval_step_image(
    rgb,
    floorplan_path,
    scene_name,
    floor_id,
    current_position,
    history_positions,
    start_position,
    goal_position,
    save_path,
    image_goal=None,
    predicted_positions=None,
    history_local_positions=None,
    predicted_local_positions=None,
    goal_text=None,
):
    """Save the current RGB observation next to floorplan and local-trajectory visualizations."""
    rgb_img = _to_rgb_image(rgb).resize((320, 320), Image.LANCZOS)

    floorplan = Image.open(floorplan_path).convert("RGB")
    draw = ImageDraw.Draw(floorplan)
    scale, offset_x, offset_y = load_floorplan_transform(scene_name, floor_id)

    def to_px(position):
        return world_to_floorplan(float(position[0]), float(position[1]), scale, offset_x, offset_y)

    if history_positions is not None and len(history_positions) > 0:
        points = [to_px(position) for position in np.asarray(history_positions)[:, :2]]
        if len(points) >= 2:
            draw.line(points, fill=(255, 64, 64), width=6)
        for point in points[::max(1, len(points) // 40)]:
            _draw_floorplan_marker(draw, point, 5, (255, 64, 64))

    if predicted_positions is not None and len(predicted_positions) > 0:
        pred_positions = np.asarray(predicted_positions)[:, :2]
        if current_position is not None:
            pred_positions = np.vstack([np.asarray(current_position)[:2], pred_positions])
        pred_points = [to_px(position) for position in pred_positions]
        if len(pred_points) >= 2:
            for p0, p1 in zip(pred_points[:-1], pred_points[1:]):
                draw.line([p0, p1], fill=(0, 0, 0), width=24)
                draw.line([p0, p1], fill=(32, 210, 255), width=16)
        for point in pred_points:
            _draw_floorplan_marker(draw, point, 10, (32, 210, 255), (0, 0, 0))

    if start_position is not None:
        _draw_floorplan_marker(draw, to_px(start_position), 12, (64, 128, 255))
    if goal_position is not None:
        _draw_floorplan_marker(draw, to_px(goal_position), 12, (64, 220, 96))
    if current_position is not None:
        _draw_floorplan_marker(draw, to_px(current_position), 14, (255, 220, 64), (0, 0, 0))

    floorplan = floorplan.resize((320, 320), Image.LANCZOS)
    if goal_text is not None:
        floorplan_draw = ImageDraw.Draw(floorplan)
        title_font = _load_font(40)
        floorplan_draw.rounded_rectangle(
            [8, 8, 312, 64],
            radius=10,
            fill=(255, 255, 255),
            outline=(120, 120, 120),
            width=3,
        )
        floorplan_draw.text((18, 18), str(goal_text), fill=(20, 20, 20), font=title_font)

    panels = [rgb_img]
    if image_goal is not None:
        panels.append(_to_rgb_image(image_goal).resize((PANEL_SIZE, PANEL_SIZE), Image.LANCZOS))
    if history_local_positions is not None or predicted_local_positions is not None:
        panels.append(
            _draw_local_traj_panel(
                history_local_positions=history_local_positions,
                predicted_local_positions=predicted_local_positions,
                size=PANEL_SIZE,
            )
        )
    panels.append(floorplan)

    panel_size = PANEL_SIZE
    gutter = PANEL_GUTTER
    outer = PANEL_OUTER
    top_margin = PANEL_TOP_MARGIN
    bottom_margin = PANEL_BOTTOM_MARGIN
    if len(panels) == 4:
        canvas_w = outer * 2 + panel_size * 2 + gutter
        canvas_h = top_margin + bottom_margin + panel_size * 2 + gutter
        canvas = Image.new("RGB", (canvas_w, canvas_h), (236, 236, 236))
        positions = [
            (outer, top_margin),
            (outer + panel_size + gutter, top_margin),
            (outer, top_margin + panel_size + gutter),
            (outer + panel_size + gutter, top_margin + panel_size + gutter),
        ]
    else:
        canvas_w = outer * 2 + panel_size * len(panels) + gutter * (len(panels) - 1)
        canvas_h = top_margin + bottom_margin + panel_size
        canvas = Image.new("RGB", (canvas_w, canvas_h), (236, 236, 236))
        positions = [(outer + idx * (panel_size + gutter), top_margin) for idx in range(len(panels))]

    for panel, (x, y) in zip(panels, positions):
        framed = Image.new("RGB", (panel_size + 2, panel_size + 2), (210, 210, 210))
        framed.paste(panel, (1, 1))
        canvas.paste(framed, (x - 1, y - 1))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tmp_path = f"{save_path}.tmp.png"
    canvas.save(tmp_path, format="PNG")
    os.replace(tmp_path, save_path)


def annotate_eval_image(
    save_path,
    panel_titles,
    goal_text=None,
    goal_text_panel_index=-1,
    goal_text_anchor="top_left",
    goal_font_size=26,
):
    """Overlay panel titles and an optional compact goal label onto a saved eval visualization."""
    image = Image.open(save_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    title_font = _load_font(18)
    goal_font = _load_font(goal_font_size)
    panel_size = PANEL_SIZE
    gutter = PANEL_GUTTER
    outer = PANEL_OUTER
    top_margin = PANEL_TOP_MARGIN

    if len(panel_titles) == 4:
        positions = [
            (outer, top_margin),
            (outer + panel_size + gutter, top_margin),
            (outer, top_margin + panel_size + gutter),
            (outer + panel_size + gutter, top_margin + panel_size + gutter),
        ]
    else:
        positions = [(outer + idx * (panel_size + gutter), top_margin) for idx in range(len(panel_titles))]

    for title, (x, y) in zip(panel_titles, positions):
        text_bbox = draw.textbbox((0, 0), title, font=title_font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        padding_x = 12
        padding_y = 6
        box_w = text_w + padding_x * 2
        box_h = text_h + padding_y * 2
        center_x = x + panel_size / 2
        box = [
            center_x - box_w / 2,
            y - box_h - 8,
            center_x + box_w / 2,
            y - 8,
        ]
        draw.rounded_rectangle(box, radius=8, fill=(255, 255, 255), outline=(160, 160, 160), width=2)
        center_y = (box[1] + box[3]) / 2
        draw.text(
            (center_x, center_y),
            title,
            fill=(30, 30, 30),
            font=title_font,
            anchor="mm",
        )

    if goal_text and positions:
        panel_idx = goal_text_panel_index % len(positions)
        x, y = positions[panel_idx]
        if goal_text_anchor == "top_right_text":
            text_bbox = draw.textbbox((0, 0), goal_text, font=goal_font)
            text_w = text_bbox[2] - text_bbox[0]
            padding_x = 12
            padding_y = 8
            text_x = x + panel_size - 8 - text_w - padding_x
            text_y = y + 8 + padding_y
            draw.text((text_x, text_y), goal_text, fill=(30, 30, 30), font=goal_font)
        else:
            text_bbox = draw.textbbox((0, 0), goal_text, font=goal_font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            padding_x = 12
            padding_y = 8
            box = [
                x + 8,
                y + 50,
                x + 8 + text_w + padding_x * 2,
                y + 50 + text_h + padding_y * 2,
            ]
            draw.rounded_rectangle(box, radius=8, fill=(255, 255, 255), outline=(160, 160, 160), width=2)
            draw.text((box[0] + padding_x, box[1] + padding_y), goal_text, fill=(30, 30, 30), font=goal_font)

    image.save(save_path)


def yaw_rotmat(yaw):
    """Return the 3x3 rotation matrix for a yaw angle."""
    c = np.cos(yaw)
    s = np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def to_global_coords(positions, curr_pos, curr_yaw):
    """Convert positions from the local frame into global coordinates."""
    positions = np.array(positions)
    curr_pos = np.array(curr_pos)
    curr_yaw = float(curr_yaw)
    rotmat = yaw_rotmat(curr_yaw).T
    if positions.shape[-1] == 2:
        rotmat = rotmat[:2, :2]
    elif positions.shape[-1] != 3:
        raise ValueError("Expected 2D or 3D positions.")
    return positions.dot(rotmat) + curr_pos


def get_pose_from_position(predicted_position):
    """Convert a sequence of global XY positions into poses and per-step yaw values."""
    delta = predicted_position[1:] - predicted_position[:-1]
    yaws = np.arctan2(delta[:, 1], delta[:, 0])
    quaternions = np.array([p.getQuaternionFromEuler([0, 0, yaw]) for yaw in yaws])
    poses = np.concatenate([predicted_position[1:], quaternions], axis=1)
    return poses, yaws


def render_history_observation(s: Simulator, device="cuda:0"):
    """Render the current robot RGB-D frame and convert it to the planner's 96x96 tensor format."""
    cur_frames = s.renderer.render_robot_cameras(modes=("rgb", "3d"), cache=False)
    cur_rgb = Image.fromarray((255 * cur_frames[0][:, :, :3]).astype(np.uint8))
    cur_rgb = cur_rgb.resize((96, 96), Image.LANCZOS)
    cur_rgb = transforms.ToTensor()(cur_rgb).unsqueeze(0).to(device)

    cur_depth = np.linalg.norm(cur_frames[1][:, :, :3], axis=2)
    cur_depth = np.clip(cur_depth, None, 10) * 25.5
    cur_depth = Image.fromarray(cur_depth.astype(np.uint8))
    cur_depth = cur_depth.resize((96, 96), Image.NEAREST)
    cur_depth = transforms.ToTensor()(cur_depth).unsqueeze(0).to(device)
    return cur_rgb, cur_depth


def updata_state(
    s: Simulator,
    robot: ManipulationRobot,
    history_rgb,
    history_depth,
    history_global_position,
    current_rgb,
    current_depth,
    current_global_position,
    device="cuda:0",
):
    """Roll the planner history forward after one waypoint execution."""
    if history_rgb.shape[1] == 8:
        history_rgb = torch.cat([history_rgb[:, 1:, :, :, :], current_rgb.unsqueeze(1)], dim=1)
        history_depth = torch.cat([history_depth[:, 1:, :, :, :], current_depth.unsqueeze(1)], dim=1)
    else:
        history_rgb = torch.cat([history_rgb, current_rgb.unsqueeze(1)], dim=1)
        history_depth = torch.cat([history_depth, current_depth.unsqueeze(1)], dim=1)

    history_global_position = np.concatenate(
        [history_global_position[1:, :], np.asarray(current_global_position).reshape(1, 2)],
        axis=0,
    )
    new_current_global_position = robot.get_position()[:2]
    history_pose = to_local_coords(
        history_global_position.tolist(),
        new_current_global_position.tolist(),
        robot.get_rpy()[2],
    )
    history_pose = torch.from_numpy(history_pose).unsqueeze(0).to(device).float()
    new_current_rgb, new_current_depth = render_history_observation(s, device=device)
    return (
        history_rgb,
        history_depth,
        history_pose,
        history_global_position,
        new_current_rgb,
        new_current_depth,
        new_current_global_position,
    )


def reset_history_to_current(s: Simulator, robot: ManipulationRobot, history_length=8, device="cuda:0"):
    """Reinitialize planner history from the robot's current observation after a fallback move."""
    current_rgb, current_depth = render_history_observation(s, device=device)
    current_global_position = robot.get_position()[:2]
    history_rgb = current_rgb.unsqueeze(1).repeat(1, history_length, 1, 1, 1)
    history_depth = current_depth.unsqueeze(1).repeat(1, history_length, 1, 1, 1)
    history_pose = torch.zeros((1, history_length, 2), device=device, dtype=torch.float32)
    history_global_position = np.repeat(
        np.asarray(current_global_position, dtype=np.float32).reshape(1, 2),
        history_length,
        axis=0,
    )
    return (
        history_rgb,
        history_depth,
        history_pose,
        history_global_position,
        current_rgb,
        current_depth,
        current_global_position,
    )


def to_local_coords(positions, curr_pos, curr_yaw):
    """Convert global positions to the current local coordinate frame."""
    positions = np.array(positions)
    curr_pos = np.array(curr_pos)
    curr_yaw = float(curr_yaw)
    translated = positions - curr_pos
    rotmat = yaw_rotmat(curr_yaw)
    if translated.shape[-1] == 2:
        rotmat = rotmat[:2, :2]
    elif translated.shape[-1] != 3:
        raise ValueError("Expected 2D or 3D positions.")
    return translated.dot(rotmat)


def set_seed(seed):
    """Seed NumPy and PyTorch RNGs used during evaluation."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_string_seed(value):
    """Return a deterministic integer offset derived from a string."""
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(value))


def resolve_gibson_device_id(device_id):
    """Map a logical CUDA index onto the EGL minor id used by iGibson."""
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices:
        entries = [entry.strip() for entry in visible_devices.split(",") if entry.strip()]
        if device_id < len(entries):
            physical_device = entries[device_id]
            try:
                output = subprocess.check_output(["nvidia-smi", "-q", "-i", physical_device], text=True)
                for line in output.splitlines():
                    if "Minor Number" in line:
                        return line.split(":")[-1].strip()
            except Exception:
                return physical_device
    return str(device_id)


def _load_ckpt(model, ckpt_path, device):
    """Load a checkpoint saved with or without DDP's `module.` prefix."""
    ckpt = torch.load(ckpt_path, map_location=device)
    for key in list(ckpt.keys()):
        if "module." in key:
            ckpt[key.replace("module.", "")] = ckpt.pop(key)
    model.load_state_dict(ckpt)
    model.eval()


def load_model(eval_config, device):
    """Instantiate planner and optional refiner from the configured checkpoints."""
    config = OmegaConf.load(eval_config["model_config"])
    policy = Planner(config).to(device)
    policy.device = device
    _load_ckpt(policy, eval_config["planner_ckpt_path"], device)
    if not eval_config["use_refiner"]:
        return policy, None

    refiner_model = Refiner(config).to(device)
    refiner_model.device = device
    _load_ckpt(refiner_model, eval_config["refiner_ckpt_path"], device)
    return policy, refiner_model


def capture_native_depth(env, device):
    """Render the current depth frame at native 256x256 resolution for refiner scoring."""
    depth = env.simulator.renderer.render_robot_cameras(modes=("3d",), cache=False)[0]
    depth = np.linalg.norm(depth[:, :, :3], axis=2)
    depth = np.clip(depth, None, 10) * 25.5
    depth = depth.astype(np.uint8)
    return torch.from_numpy(depth).float().unsqueeze(0).unsqueeze(0).to(device)


def choose_safer_actions(refiner_model, planner_actions, refined_actions, depth, num_steps):
    """Choose planner or refiner output using obstacle loss over the executed prefix."""
    occupancy_map = refiner_model.depth_to_occupancy(depth)
    distance_field_map = euclidean_signed_transform(occupancy_map)
    distance_field_map = torch.clamp(distance_field_map, min=-5.0, max=5.0) / 5.0

    def obstacle_loss(actions):
        grid_traj = traj_to_grid(actions, refiner_model.config)
        sampled_distance_values = F.grid_sample(
            distance_field_map.unsqueeze(1),
            grid_traj,
            mode="bilinear",
            align_corners=True,
        ).squeeze(1).squeeze(2)
        batch_size, horizon = sampled_distance_values.shape
        half = horizon // 2
        time_weights = torch.cat([
            torch.full((half,), 1.0, device=sampled_distance_values.device),
            torch.full((horizon - half,), 0.5, device=sampled_distance_values.device),
        ]).unsqueeze(0).expand(batch_size, horizon)
        safe_distance = torch.clamp(sampled_distance_values * time_weights, min=-5.0, max=5.0)
        return torch.mean(torch.exp(-refiner_model.config.obstacle_loss_weight * safe_distance))

    planner_loss = obstacle_loss(planner_actions[:, :num_steps])
    refiner_loss = obstacle_loss(refined_actions[:, :num_steps])
    return refined_actions if refiner_loss <= planner_loss else planner_actions


def build_env(scene_name, floor, device_id):
    """Create one headless iGibson env pinned to the requested EGL device."""
    os.environ["GIBSON_DEVICE_ID"] = resolve_gibson_device_id(device_id)
    os.environ.pop("DISPLAY", None)
    config_data = yaml.load(open(DEFAULT_ENV_CONFIG_PATH, "r"), Loader=yaml.FullLoader)
    config_data["enable_shadow"] = False
    config_data["enable_pbr"] = False
    config_data["scene_id"] = scene_name
    env = iGibsonEnv(config_file=config_data, mode="headless")
    floor_height = load_floor_height(scene_name, floor)
    robot_z = floor_height + config_data.get("initial_pos_z_offset", 0.1)
    return env, robot_z


def load_floorplan_tensor(scenes_dir, scene_id, device):
    """Load the floorplan image and convert it to the planner tensor format."""
    floorplan_path = os.path.join(scenes_dir, scene_id, "floorplan.png")
    floorplan = Image.open(floorplan_path)
    return floorplan_path, IMAGE_TRANSFORM(floorplan).unsqueeze(0).to(device)


def list_traj_ids(scenes_dir, scene_id):
    """List trajectory directories in numeric order."""
    trajs = [
        traj_id
        for traj_id in os.listdir(os.path.join(scenes_dir, scene_id))
        if os.path.isdir(os.path.join(scenes_dir, scene_id, traj_id))
    ]
    return sorted(trajs, key=lambda name: int(re.search(r"\d+", name).group()))


def list_scene_ids(scenes_dir, explicit_scene_ids=None, excluded_scene_ids=None):
    """Return valid scene ids from the evaluation directory, with optional filtering."""
    excluded_scene_ids = set(excluded_scene_ids or [])
    if explicit_scene_ids:
        scene_ids = explicit_scene_ids
    else:
        scene_ids = os.listdir(scenes_dir)
    return sorted(
        scene_id
        for scene_id in scene_ids
        if os.path.isdir(os.path.join(scenes_dir, scene_id)) and scene_id not in excluded_scene_ids
    )


def load_traj_data(scenes_dir, scene_id, traj_id):
    """Load one trajectory txt file."""
    return np.loadtxt(os.path.join(scenes_dir, scene_id, traj_id, f"{traj_id}.txt"))


def _build_history_state(
    env,
    history_global_position,
    history_orientation_yaw,
    current_position,
    current_yaw,
    robot_z,
    device,
):
    current_quaternion = p.getQuaternionFromEuler([0, 0, current_yaw])
    history_quaternion = np.array([p.getQuaternionFromEuler([0, 0, yaw]) for yaw in history_orientation_yaw])
    history_position = to_local_coords(
        history_global_position.tolist(),
        current_position.tolist(),
        current_yaw,
    )

    history_rgb_list = []
    history_depth_list = []
    for pose, orientation in zip(history_global_position, history_quaternion):
        env.robots[0].set_position([pose[0], pose[1], robot_z])
        env.robots[0].set_orientation(orientation)
        env.simulator_step()
        rgb, depth = env.simulator.renderer.render_robot_cameras(modes=("rgb", "3d"), cache=False)
        rgb = Image.fromarray((255 * rgb[:, :, :3]).astype(np.uint8))
        history_rgb_list.append(IMAGE_TRANSFORM(rgb).unsqueeze(0))
        depth = np.linalg.norm(depth[:, :, :3], axis=2)
        depth = np.clip(depth, None, 10) * 25.5
        depth = Image.fromarray(depth.astype(np.uint8))
        history_depth_list.append(IMAGE_TRANSFORM(depth).unsqueeze(0))

    env.robots[0].set_position([current_position[0], current_position[1], robot_z])
    env.robots[0].set_orientation(current_quaternion)
    env.simulator_step()
    current_rgb, current_depth = render_history_observation(env.simulator, device=device)
    history_rgb = torch.cat(history_rgb_list, dim=0).unsqueeze(0).to(device)
    history_depth = torch.cat(history_depth_list, dim=0).unsqueeze(0).to(device)
    history_pose = torch.from_numpy(history_position).unsqueeze(0).to(device).float()
    return (
        history_rgb,
        history_depth,
        history_pose,
        history_global_position,
        current_rgb,
        current_depth,
        current_position,
    )


def capture_image_goal(env, goal_position, goal_direction, robot_z, device):
    """Render the image-goal observation at the target pose for both model input and visualization."""
    goal_yaw = np.arctan2(goal_direction[1], goal_direction[0])
    goal_quat = p.getQuaternionFromEuler([0, 0, goal_yaw])
    env.robots[0].set_position([goal_position[0], goal_position[1], robot_z])
    env.robots[0].set_orientation(goal_quat)
    env.simulator_step()
    image_goal = env.simulator.renderer.render_robot_cameras(modes=("rgb",), cache=False)[0]
    image_goal_vis = Image.fromarray((255 * image_goal[:, :, :3]).astype(np.uint8))
    image_goal_tensor = IMAGE_TRANSFORM(image_goal_vis).unsqueeze(0).to(device)
    return image_goal_tensor, np.asarray(image_goal_vis)


def read_object_goal(scenes_dir, scene_id, traj_id):
    """Read the object category for one object-goal trajectory."""
    obj_json_path = os.path.join(scenes_dir, scene_id, traj_id, "object", "object.json")
    with open(obj_json_path, "r") as f:
        obj_data = json.load(f)
    if isinstance(obj_data, list):
        obj_data = obj_data[0] if obj_data else {}
    return [obj_data.get("object_category", "null")]


def relative_point_goal_tensor(point_goal, current_position, current_yaw, device):
    """Convert a global point goal into the current local frame."""
    relative_point_goal = to_local_coords(point_goal.tolist(), current_position.tolist(), current_yaw)
    return torch.tensor(relative_point_goal).unsqueeze(0).to(device).float()


def init_point_goal_state(env, scenes_dir, scene_id, traj_id, robot_z, device):
    """Initialize planner history, current observation, and point-goal condition."""
    traj_data = load_traj_data(scenes_dir, scene_id, traj_id)
    point_goal = traj_data[-1, :2]
    current_position = traj_data[8, :2]
    current_orientation_point = traj_data[8, 2:4]
    current_yaw = np.arctan2(
        current_orientation_point[1] - current_position[1],
        current_orientation_point[0] - current_position[0],
    )
    history_global_position = traj_data[:8, :2]
    history_global_orientation_point = traj_data[:8, 2:4]
    history_orientation_yaw = np.arctan2(
        history_global_orientation_point[:, 1] - history_global_position[:, 1],
        history_global_orientation_point[:, 0] - history_global_position[:, 0],
    )
    (
        history_rgb,
        history_depth,
        history_pose,
        history_global_position,
        current_rgb,
        current_depth,
        current_position,
    ) = _build_history_state(
        env,
        history_global_position,
        history_orientation_yaw,
        current_position,
        current_yaw,
        robot_z,
        device,
    )
    return {
        "history_rgb": history_rgb,
        "history_depth": history_depth,
        "history_pose": history_pose,
        "history_global_position": history_global_position,
        "current_rgb": current_rgb,
        "current_depth": current_depth,
        "current_position": current_position,
        "point_goal": point_goal,
        "relative_point_goal": relative_point_goal_tensor(point_goal, current_position, current_yaw, device),
    }


def init_image_goal_state(env, scenes_dir, scene_id, traj_id, robot_z, device):
    """Initialize planner history, current observation, image goal, and GT point goal."""
    traj_data = load_traj_data(scenes_dir, scene_id, traj_id)
    point_goal = traj_data[-1, :2]
    goal_orientation = traj_data[-1, 2:4] - traj_data[-1, :2]
    image_goal, image_goal_vis = capture_image_goal(env, point_goal, goal_orientation, robot_z, device)
    current_position = traj_data[8, :2]
    current_orientation_point = traj_data[8, 2:4]
    current_yaw = np.arctan2(
        current_orientation_point[1] - current_position[1],
        current_orientation_point[0] - current_position[0],
    )
    history_global_position = traj_data[:8, :2]
    history_global_orientation_point = traj_data[:8, 2:4]
    history_orientation_yaw = np.arctan2(
        history_global_orientation_point[:, 1] - history_global_position[:, 1],
        history_global_orientation_point[:, 0] - history_global_position[:, 0],
    )
    (
        history_rgb,
        history_depth,
        history_pose,
        history_global_position,
        current_rgb,
        current_depth,
        current_position,
    ) = _build_history_state(
        env,
        history_global_position,
        history_orientation_yaw,
        current_position,
        current_yaw,
        robot_z,
        device,
    )
    return {
        "history_rgb": history_rgb,
        "history_depth": history_depth,
        "history_pose": history_pose,
        "history_global_position": history_global_position,
        "current_rgb": current_rgb,
        "current_depth": current_depth,
        "current_position": current_position,
        "point_goal": point_goal,
        "image_goal": image_goal,
        "image_goal_vis": image_goal_vis,
        "relative_point_goal": relative_point_goal_tensor(point_goal, current_position, current_yaw, device),
    }


def init_object_goal_state(env, scenes_dir, scene_id, traj_id, robot_z, device):
    """Initialize planner history, current observation, object goal, and GT point goal."""
    traj_data = load_traj_data(scenes_dir, scene_id, traj_id)
    point_goal = traj_data[-1, :2]
    object_goal = read_object_goal(scenes_dir, scene_id, traj_id)
    current_position = traj_data[8, :2]
    current_orientation_point = traj_data[8, 2:4]
    current_yaw = np.arctan2(
        current_orientation_point[1] - current_position[1],
        current_orientation_point[0] - current_position[0],
    )
    history_global_position = traj_data[:8, :2]
    history_global_orientation_point = traj_data[:8, 2:4]
    history_orientation_yaw = np.arctan2(
        history_global_orientation_point[:, 1] - history_global_position[:, 1],
        history_global_orientation_point[:, 0] - history_global_position[:, 0],
    )
    (
        history_rgb,
        history_depth,
        history_pose,
        history_global_position,
        current_rgb,
        current_depth,
        current_position,
    ) = _build_history_state(
        env,
        history_global_position,
        history_orientation_yaw,
        current_position,
        current_yaw,
        robot_z,
        device,
    )
    return {
        "history_rgb": history_rgb,
        "history_depth": history_depth,
        "history_pose": history_pose,
        "history_global_position": history_global_position,
        "current_rgb": current_rgb,
        "current_depth": current_depth,
        "current_position": current_position,
        "point_goal": point_goal,
        "object_goal": object_goal,
        "relative_point_goal": relative_point_goal_tensor(point_goal, current_position, current_yaw, device),
    }


def infer_actions(
    policy,
    refiner_model,
    history_rgb,
    history_depth,
    history_pose,
    floorplan,
    current_depth,
    execute_steps,
    image_goal=None,
    relative_point_goal=None,
    object_goal=None,
):
    """Run one planner sample and optional planner/refiner safety selection."""
    planner_actions = policy.inference(
        history_rgb,
        history_depth,
        history_pose,
        floorplan,
        image_goal,
        relative_point_goal,
        object_goal,
    )
    if refiner_model is None:
        return planner_actions

    refined_actions = refiner_model.inference(planner_actions, current_depth)
    return choose_safer_actions(
        refiner_model,
        planner_actions,
        refined_actions,
        current_depth,
        execute_steps,
    )


def move_to_fallback(env, floor, point_goal, robot_z, current_yaw):
    """Move to the configured shortest-path fallback waypoint and return its pose."""
    shortest_path_output = env.simulator.scene.get_shortest_path(
        floor,
        env.robots[0].get_position()[:2],
        point_goal,
        entire_path=True,
    )
    if len(shortest_path_output) == 3:
        safe_path, _, _ = shortest_path_output
    elif len(shortest_path_output) == 2:
        safe_path, _ = shortest_path_output
    else:
        raise ValueError(
            f"Unexpected get_shortest_path return format: {len(shortest_path_output)} values"
        )
    if len(safe_path) < 2:
        return None, None

    fallback_idx = choose_fallback_path_index(len(safe_path))
    next_position = safe_path[fallback_idx]
    ori_from_idx = max(fallback_idx - 1, 0)
    next_orientation = safe_path[fallback_idx] - safe_path[ori_from_idx]
    if np.linalg.norm(next_orientation) < 1e-6 and fallback_idx + 1 < len(safe_path):
        next_orientation = safe_path[fallback_idx + 1] - safe_path[fallback_idx]
    if np.linalg.norm(next_orientation) < 1e-6:
        next_orientation = np.array([np.cos(current_yaw), np.sin(current_yaw)])

    next_yaw = np.arctan2(next_orientation[1], next_orientation[0])
    next_quat = p.getQuaternionFromEuler([0, 0, next_yaw])
    env.robots[0].set_position([next_position[0], next_position[1], robot_z])
    env.robots[0].set_orientation(next_quat)
    env.simulator.step()
    return next_position, next_yaw

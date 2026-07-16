import numpy as np
import torch
import matplotlib.pyplot as plt 

ACTION_STATS = {}
ACTION_STATS["min"] = np.array([-2.5, -4])
ACTION_STATS["max"] = np.array([5, 4])    

def get_delta(actions):       # (0,0)->first action point, first action point->second action point, ...
    # append zeros to first action, preserving PyTorch tensors if passed
    if isinstance(actions, torch.Tensor):
        zeros = torch.zeros((actions.shape[0], 1, actions.shape[-1]), device=actions.device, dtype=actions.dtype)
        ex_actions = torch.cat([zeros, actions], dim=1)
    else:
        ex_actions = np.concatenate([np.zeros((actions.shape[0], 1, actions.shape[-1])), actions], axis=1)
    
    delta = ex_actions[:,1:] - ex_actions[:,:-1]
    return delta

def normalize_data(data, stats):
    if isinstance(data, torch.Tensor):
        min_val = torch.tensor(stats['min'], device=data.device, dtype=data.dtype)
        max_val = torch.tensor(stats['max'], device=data.device, dtype=data.dtype)
        ndata = (data - min_val) / (max_val - min_val)
        ndata = ndata * 2 - 1
        return ndata
    else:
        # nomalize to [0,1]
        ndata = (data - stats['min']) / (stats['max'] - stats['min'])
        # normalize to [-1, 1]
        ndata = ndata * 2 - 1
        return ndata

def unnormalize_data(ndata, stats):
    device = ndata.device
    stats = {'min': stats['min'].to(device), 'max': stats['max'].to(device)}
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

def get_action(diffusion_output, action_stats=ACTION_STATS):
    # diffusion_output: (B, 2*T+1, 1)
    # return: (B, T-1)
    ndeltas = diffusion_output
    ndeltas = ndeltas.reshape(ndeltas.shape[0], -1, 2)
    min_val = torch.tensor(action_stats['min'], dtype=ndeltas.dtype)
    max_val = torch.tensor(action_stats['max'], dtype=ndeltas.dtype)
    stats = {'min': min_val, 'max': max_val}
    ndeltas = unnormalize_data(ndeltas, stats)
    actions = torch.cumsum(ndeltas, dim=1)
    return actions


def _to_hwc_uint8(image):
    if image is None:
        return None
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def draw_multi_goal(
    history_pose: np.ndarray,
    predicted_img_pose: np.ndarray,
    predicted_point_pose: np.ndarray,
    predicted_obj_pose: np.ndarray,
    gt_pose: np.ndarray,
    save_path: str,
    rgb_image: np.ndarray = None,
    goal_image: np.ndarray = None,
    object_goal: str = None,
):
    """Save one combined planner visualization for img/point/obj goals on the same sample."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax_rgb, ax_goal, ax_traj = axes

    if rgb_image is not None:
        ax_rgb.imshow(_to_hwc_uint8(rgb_image))
    ax_rgb.axis("off")
    ax_rgb.set_title("Current RGB Observation")

    if goal_image is not None:
        ax_goal.imshow(_to_hwc_uint8(goal_image))
    ax_goal.axis("off")
    ax_goal.set_title(str(object_goal) if object_goal is not None else "object")

    if history_pose is not None:
        ax_traj.plot(history_pose[:, 0], history_pose[:, 1], "bo-", label="History Pose")
    if gt_pose is not None:
        ax_traj.plot(gt_pose[:, 0], gt_pose[:, 1], "go--", label="Ground Truth Pose")
    if predicted_img_pose is not None:
        ax_traj.plot(predicted_img_pose[:, 0], predicted_img_pose[:, 1], "ro--", label="Pred Img")
    if predicted_point_pose is not None:
        ax_traj.plot(predicted_point_pose[:, 0], predicted_point_pose[:, 1], color="orange", linestyle="--", marker="o", label="Pred Point")
    if predicted_obj_pose is not None:
        ax_traj.plot(predicted_obj_pose[:, 0], predicted_obj_pose[:, 1], color="purple", linestyle="--", marker="o", label="Pred Obj")

    ax_traj.legend()
    ax_traj.axis("equal")
    try:
        filename = save_path.split("/")[-1]
        if "sample_" in filename:
            sample_id = filename.split("sample_")[1].replace("_train.png", "").replace("_eval.png", "").replace(".png", "")
            title = f"{sample_id}::multi_goal"
        else:
            title = "multi_goal"
    except Exception:
        title = "multi_goal"
    ax_traj.set_title(title)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def draw_occupancy_map(config, occupancy_map:np.ndarray, orin_traj:np.ndarray, refined_traj:np.ndarray, gt_traj:np.ndarray, save_path:str, goal_type:str=None, rgb_image:np.ndarray=None):
    """
    occupancy_map: (H_o, W_o)
    orin_traj: (T, 2)
    refined_traj: (T, 2)
    save_path: str
    goal_type: str
    rgb_image: (3, H, W) or (H, W, 3) image
    """
    if rgb_image is not None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        ax_rgb = axes[0]
        ax_map = axes[1]

        if rgb_image.shape[0] == 3:  # (C, H, W)
            rgb_image = np.transpose(rgb_image, (1, 2, 0))
        if rgb_image.max() <= 1.0:
            rgb_image = (rgb_image * 255).astype(np.uint8)

        ax_rgb.imshow(rgb_image)
        ax_rgb.axis('off')
        ax_rgb.set_title("Current RGB Observation")
    else:
        fig, ax_map = plt.subplots(figsize=(5, 5))

    ax_map.imshow(occupancy_map, cmap='gray', origin='lower')
    H_o = occupancy_map.shape[0]
    W_o = occupancy_map.shape[1]
    grid_size = config.grid_size
    if orin_traj is not None:
        ax_map.plot( - orin_traj[:,1] / grid_size + W_o / 2, orin_traj[:,0] / grid_size, 'bo--', label='Original Traj')
    if refined_traj is not None:
        ax_map.plot( - refined_traj[:,1] / grid_size + W_o / 2, refined_traj[:,0] / grid_size, 'ro--', label='Refined Traj')
    if gt_traj is not None:
        ax_map.plot( - gt_traj[:,1] / grid_size + W_o / 2, gt_traj[:,0] / grid_size, 'go--', label='Ground Truth Traj')
    # Lock the view to the occupancy map's own extent — otherwise an off-the-rails trajectory
    # prediction (common early in training) makes matplotlib autoscale the axes to fit it,
    # shrinking the map itself in the saved image instead of just clipping the bad point.
    ax_map.set_xlim(0, W_o)
    ax_map.set_ylim(0, H_o)
    ax_map.legend()
    try:
        filename = save_path.split('/')[-1]
        if 'sample_' in filename:
            sample_id = filename.split('sample_')[1].replace('_train.png', '').replace('_eval.png', '').replace('.png', '')
            title = f"{sample_id}::{goal_type}"
        else:
            title = str(goal_type)
    except Exception:
        title = str(goal_type)
    ax_map.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

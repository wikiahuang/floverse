import os
import time
from multiprocessing import Pool, set_start_method

import numpy as np
import torch

from eval.common import (
    COMMON_EXCLUDED_SCENES,
    DEFAULT_RESULTS_ROOT,
    DEFAULT_MODEL_CONFIG_PATH,
    DEFAULT_PLANNER_CKPT_PATH,
    DEFAULT_REFINER_CKPT_PATH,
    IMAGE_ONLY_EXCLUDED_SCENES,
    apply_eval_cli_overrides,
    annotate_eval_image,
    build_env,
    capture_native_depth,
    get_pose_from_position,
    infer_actions,
    init_image_goal_state,
    list_scene_ids,
    list_traj_ids,
    load_floorplan_tensor,
    load_model,
    move_to_fallback,
    reset_history_to_current,
    save_eval_step_image,
    set_seed,
    setup_logging,
    stable_string_seed,
    to_global_coords,
    updata_state,
)


LOGGER = setup_logging()


def save_step_visualization(
    save_images,
    env,
    floorplan_path,
    scene_name,
    floor,
    current_position,
    step_history,
    start_position,
    point_goal,
    save_path,
    image_goal_vis,
    predicted_positions,
    history_local_positions,
    predicted_local_positions,
):
    """Save one RGB + image-goal + local-traj + floorplan visualization frame."""
    if not save_images:
        return

    current_rgb_vis = env.simulator.renderer.render_robot_cameras(modes=("rgb",), cache=False)[0]
    save_eval_step_image(
        current_rgb_vis,
        floorplan_path,
        scene_name,
        floor,
        current_position,
        step_history,
        start_position,
        point_goal,
        save_path,
        image_goal=image_goal_vis,
        predicted_positions=predicted_positions,
        history_local_positions=history_local_positions,
        predicted_local_positions=predicted_local_positions,
    )
    annotate_eval_image(
        save_path,
        [
            "Current Observation",
            "Goal Image",
            "Local Trajectory Prediction",
            "Global Trajectory Overview",
        ],
    )


def eval_scene(scene_id, eval_config, worker_id):
    """Evaluate image-goal trajectories while visualizing an auxiliary point-goal rollout."""
    device_id = worker_id % torch.cuda.device_count()
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device)
    set_seed(eval_config["seed"] + stable_string_seed(scene_id))
    LOGGER.info("[PID %s] Evaluating %s on %s", os.getpid(), scene_id, device)

    policy, refiner_model = load_model(eval_config, device)
    scenes_dir = eval_config["scenes_dir"]
    execute_steps = eval_config["execute_steps"]
    save_images = eval_config["save_images"]
    max_steps = eval_config["max_steps"]
    max_trajs_per_scene = eval_config["max_trajs_per_scene"]
    collision_limit = eval_config.get("collision_limit", 15)
    traj_save_scene_dir = os.path.join(eval_config["traj_save_dir"], scene_id)
    os.makedirs(traj_save_scene_dir, exist_ok=True)

    scene_name = scene_id.split("_")[0]
    floor = int(scene_id.split("_")[1])
    env, robot_z = build_env(scene_name, floor, device_id)
    floorplan_path, floorplan = load_floorplan_tensor(scenes_dir, scene_id, device)

    for traj_id in list_traj_ids(scenes_dir, scene_id)[:max_trajs_per_scene]:
        start_time = time.time()
        LOGGER.info("Evaluating %s/%s", scene_id, traj_id)
        traj_save_id_dir = os.path.join(traj_save_scene_dir, traj_id)
        os.makedirs(traj_save_id_dir, exist_ok=True)

        traj_npy = []
        collisions = 0
        steps = 0
        image_steps = 0
        state = init_image_goal_state(env, scenes_dir, scene_id, traj_id, robot_z, device)
        start_position = state["current_position"].copy()
        point_goal = state["point_goal"]

        while (
            np.linalg.norm(state["current_position"] - point_goal) > 1
            and steps < max_steps
            and collisions < collision_limit
        ):
            history_local_for_vis = state["history_pose"].squeeze(0).detach().cpu().numpy()
            with torch.no_grad():
                native_depth = capture_native_depth(env, device)
                image_actions = infer_actions(
                    policy,
                    refiner_model,
                    state["history_rgb"],
                    state["history_depth"],
                    state["history_pose"],
                    floorplan,
                    native_depth,
                    execute_steps,
                    image_goal=state["image_goal"],
                ).squeeze(0).cpu().numpy()

            predicted_local_for_vis = image_actions[:execute_steps].copy()
            current_position = env.robots[0].get_position()[:2]
            current_yaw = env.robots[0].get_rpy()[2]
            predicted_positions = to_global_coords(
                image_actions[:execute_steps].tolist(),
                current_position.tolist(),
                current_yaw,
            )
            predicted_pose, yaws = get_pose_from_position(predicted_positions)

            for waypoint_idx in range(len(predicted_pose)):
                position_3d = np.array([predicted_pose[waypoint_idx, 0], predicted_pose[waypoint_idx, 1], robot_z])
                euler = np.array([0, 0, yaws[waypoint_idx]])
                valid = env.test_valid_position(
                    env.robots[0],
                    position_3d,
                    euler,
                    ignore_self_collision=True,
                )
                if not valid:
                    next_position, next_yaw = move_to_fallback(env, floor, point_goal, robot_z, current_yaw)
                    if next_position is None:
                        break

                    (
                        state["history_rgb"],
                        state["history_depth"],
                        state["history_pose"],
                        state["history_global_position"],
                        state["current_rgb"],
                        state["current_depth"],
                        state["current_position"],
                    ) = reset_history_to_current(
                        env.simulator,
                        env.robots[0],
                        history_length=state["history_rgb"].shape[1],
                        device=device,
                    )
                    collisions += 1
                    traj_npy.append(np.array([next_position[0], next_position[1], next_yaw, 1]))
                    save_step_visualization(
                        save_images,
                        env,
                        floorplan_path,
                        scene_name,
                        floor,
                        next_position,
                        np.array([row[:2] for row in traj_npy], dtype=np.float32),
                        start_position,
                        point_goal,
                        os.path.join(
                            traj_save_id_dir,
                            f"step_{image_steps:06d}_chunk_{steps:04d}_fallback.png",
                        ),
                        state["image_goal_vis"],
                        predicted_positions,
                        history_local_for_vis,
                        predicted_local_for_vis,
                    )
                    image_steps += int(save_images)
                    break

                next_position = predicted_pose[waypoint_idx, :2]
                next_quat = predicted_pose[waypoint_idx, 2:]
                env.robots[0].set_position([next_position[0], next_position[1], robot_z])
                env.robots[0].set_orientation(next_quat)
                env.simulator.step()
                (
                    state["history_rgb"],
                    state["history_depth"],
                    state["history_pose"],
                    state["history_global_position"],
                    state["current_rgb"],
                    state["current_depth"],
                    state["current_position"],
                ) = updata_state(
                    env.simulator,
                    env.robots[0],
                    state["history_rgb"],
                    state["history_depth"],
                    state["history_global_position"],
                    state["current_rgb"],
                    state["current_depth"],
                    state["current_position"],
                    device,
                )
                traj_npy.append(np.array([next_position[0], next_position[1], yaws[waypoint_idx], 0]))
                save_step_visualization(
                    save_images,
                    env,
                    floorplan_path,
                    scene_name,
                    floor,
                    next_position,
                    np.array([row[:2] for row in traj_npy], dtype=np.float32),
                    start_position,
                    point_goal,
                    os.path.join(
                        traj_save_id_dir,
                        f"step_{image_steps:06d}_chunk_{steps:04d}_wp_{waypoint_idx:02d}.png",
                    ),
                    state["image_goal_vis"],
                    predicted_positions,
                    history_local_for_vis,
                    predicted_local_for_vis,
                )
                image_steps += int(save_images)

            steps += 1

        np.savetxt(os.path.join(traj_save_id_dir, f"{traj_id}.txt"), np.array(traj_npy))
        LOGGER.info(
            "Finished %s/%s, steps=%s, collisions=%s, time=%.2fs",
            scene_id,
            traj_id,
            steps,
            collisions,
            time.time() - start_time,
        )

    env.close()


def main(eval_config):
    """Run image-goal evaluation across the configured scene set."""
    excluded_scene_ids = COMMON_EXCLUDED_SCENES | IMAGE_ONLY_EXCLUDED_SCENES
    scene_ids = list_scene_ids(
        eval_config["scenes_dir"],
        eval_config.get("scene_ids"),
        excluded_scene_ids=excluded_scene_ids,
    )
    nproc = min(len(scene_ids), eval_config["num_workers"])
    LOGGER.info("Launching %s parallel processes", nproc)
    with Pool(nproc) as pool:
        pool.starmap(
            eval_scene,
            [(scene_id, eval_config, idx % nproc) for idx, scene_id in enumerate(scene_ids)],
            chunksize=1,
        )


if __name__ == "__main__":
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass

    scenes_dir = "/home/weiqi/data/floverse_data/eval/eval_with_obj"
    scene_ids = list_scene_ids(
        scenes_dir,
        excluded_scene_ids=COMMON_EXCLUDED_SCENES | IMAGE_ONLY_EXCLUDED_SCENES,
    )
    eval_config = {
        "scenes_dir": scenes_dir,
        "scene_ids": scene_ids,
        "model_config": str(DEFAULT_MODEL_CONFIG_PATH),
        "planner_ckpt_path": str(DEFAULT_PLANNER_CKPT_PATH),
        "refiner_ckpt_path": str(DEFAULT_REFINER_CKPT_PATH),
        "use_refiner": True,
        "execute_steps": 9,
        "seed": 0,
        "num_workers": 5,
        "save_images": True,
        "collision_limit": 15,
        "max_steps": 500,
        "max_trajs_per_scene": 30,
        "traj_save_dir": str(DEFAULT_RESULTS_ROOT / "image_goal"),
    }
    eval_config = apply_eval_cli_overrides(eval_config, default_scene_ids=scene_ids)
    main(eval_config=eval_config)

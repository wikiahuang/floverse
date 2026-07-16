import argparse
import os

import numpy as np
import tqdm


def path_length(points):
    """Compute 2D path length from a trajectory array."""
    points = np.atleast_2d(points)
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(points[1:, :2] - points[:-1, :2], axis=1)))


def load_traj_file(path):
    """Load a trajectory txt file and normalize it to shape [T, D]."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return np.empty((0, 4), dtype=np.float32)
    data = np.loadtxt(path)
    data = np.atleast_2d(data)
    if data.ndim != 2:
        return np.empty((0, 4), dtype=np.float32)
    return data


def judge_success(data, shortest_traj, collision_th=15, success_distance=1.0):
    """Evaluate one predicted trajectory against the shortest-path reference."""
    shortest_traj = np.atleast_2d(shortest_traj)
    shortest_dis = path_length(shortest_traj)
    goal = shortest_traj[-1, :2]
    start = shortest_traj[0, :2]
    d_0 = float(np.linalg.norm(start - goal))

    if data.size == 0 or data.shape[1] < 4:
        return {
            "arrive": False,
            "collision_num": 0,
            "shortest_dis": shortest_dis,
            "actual_dis": 0.0,
            "d_0": d_0,
            "d_t": d_0,
        }

    collision_num = 0
    actual_dis = 0.0
    d_t = float(np.linalg.norm(data[0, :2] - goal))
    arrive = False
    for idx, pose in enumerate(data):
        d_t = float(np.linalg.norm(pose[:2] - goal))
        if idx > 0:
            actual_dis += float(np.linalg.norm(pose[:2] - data[idx - 1, :2]))
        if collision_num >= collision_th:
            break
        if pose[3] == 1:
            collision_num += 1
        if d_t < success_distance:
            arrive = True
            break

    return {
        "arrive": arrive,
        "collision_num": collision_num,
        "shortest_dis": shortest_dis,
        "actual_dis": actual_dis,
        "d_0": d_0,
        "d_t": d_t,
    }


def evaluate_directory(traj_dir, shortest_traj_dir, collision_th=15, success_distance=1.0):
    """Evaluate all trajectories under one results directory."""
    metrics = {
        "arrives": [],
        "collision_nums": [],
        "shortest_distance": [],
        "actual_distance": [],
        "spl": [],
        "softspl": [],
    }

    scene_ids = sorted(os.listdir(traj_dir))
    for scene_id in tqdm.tqdm(scene_ids):
        scene_path = os.path.join(traj_dir, scene_id)
        if not os.path.isdir(scene_path):
            continue

        for traj_id in sorted(os.listdir(scene_path)):
            traj_path = os.path.join(scene_path, traj_id, f"{traj_id}.txt")
            shortest_traj_path = os.path.join(shortest_traj_dir, scene_id, traj_id, f"{traj_id}.txt")
            if not os.path.exists(traj_path) or not os.path.exists(shortest_traj_path):
                continue

            data = load_traj_file(traj_path)
            shortest_traj = load_traj_file(shortest_traj_path)
            if shortest_traj.size == 0 or shortest_traj.shape[1] < 2:
                continue

            sample = judge_success(
                data,
                shortest_traj,
                collision_th=collision_th,
                success_distance=success_distance,
            )
            shortest_dis = sample["shortest_dis"]
            actual_dis = sample["actual_dis"]
            denom = max(actual_dis, shortest_dis, 1e-8)
            d_0 = max(sample["d_0"], 1e-8)

            metrics["arrives"].append(sample["arrive"])
            metrics["collision_nums"].append(sample["collision_num"])
            metrics["shortest_distance"].append(shortest_dis)
            metrics["actual_distance"].append(actual_dis)
            metrics["spl"].append(float(sample["arrive"]) * shortest_dis / denom)
            metrics["softspl"].append((1.0 - sample["d_t"] / d_0) * shortest_dis / denom)

    return metrics


def main():
    """Parse CLI args, run evaluation, and print aggregate metrics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj_dir", type=str, required=True, help="Directory containing predicted trajectories")
    parser.add_argument(
        "--shortest_traj_dir",
        type=str,
        required=True,
        help="Directory containing shortest-path reference trajectories",
    )
    parser.add_argument("--collision_th", type=int, default=15, help="Maximum allowed collisions for success")
    parser.add_argument("--success_distance", type=float, default=1.0, help="Goal success radius in meters")
    args = parser.parse_args()

    metrics = evaluate_directory(
        args.traj_dir,
        args.shortest_traj_dir,
        collision_th=args.collision_th,
        success_distance=args.success_distance,
    )

    if not metrics["arrives"]:
        print("No valid trajectories found.")
        return

    print("Overall Results:")
    print("SR:", float(np.mean(metrics["arrives"])))
    print("SPL:", float(np.mean(metrics["spl"])))
    print("SoftSPL:", float(np.mean(metrics["softspl"])))


if __name__ == "__main__":
    main()

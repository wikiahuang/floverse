import argparse
import logging
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch
from tqdm import tqdm
from omegaconf import OmegaConf
import wandb
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
# diffusion_policy is a separate clone (see README) — override its location with
# the DIFFUSION_POLICY_ROOT env var if it isn't checked out at model/diffusion_policy.
sys.path.append(os.environ.get("DIFFUSION_POLICY_ROOT", str(PROJECT_ROOT / "model" / "diffusion_policy")))

from model.planner import Planner
from model.refiner import Refiner
from datasets.dataset import NavDataset
from utils.model_utils import action_reduce
from utils.data_utils import draw_multi_goal, draw_occupancy_map


def setup_logging(accelerate):
    """Configure a shared logger/formatter for training and dataset messages."""
    logger = logging.getLogger("floverse")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if accelerate.is_main_process:
        log_path = os.path.join(wandb.run.dir, "train_progress.log")
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info("Logging batch progress to %s", log_path)

    return logger


def load_ckpt(path, module, accelerate):
    """Load a checkpoint saved from either wrapped or unwrapped model state."""
    ckpt = torch.load(path, map_location="cpu")
    for key in list(ckpt.keys()):
        if 'module.' in key:
            ckpt[key.replace('module.', '')] = ckpt.pop(key)
    accelerate.unwrap_model(module).load_state_dict(ckpt)


def train(config, run_name="train", resume_planner_ckpt=None, resume_refiner_ckpt=None, refiner_start_epoch=None):
    """
    Trains the planner alone for the first `refiner_start_epoch` epochs, then jointly trains
    planner + refiner. The refiner's loss never updates the planner: the trajectory it
    consumes is produced by the planner under torch.no_grad(), and each model is optimized
    by its own optimizer.
    """
    accelerate = Accelerator(
        gradient_accumulation_steps=1,
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=True),
            InitProcessGroupKwargs(timeout=timedelta(hours=2)),
        ]
    )
    if accelerate.is_main_process:
        wandb.login()
        wandb.init(project="floverse", name=run_name, dir=str(PROJECT_ROOT))
    logger = setup_logging(accelerate)

    if refiner_start_epoch is None:
        refiner_start_epoch = config.get("refiner_start_epoch", 0)

    model = Planner(config, accelerate)
    refiner_model = Refiner(config, accelerate)

    dataset = NavDataset(
        data_folder=config.data_folder,
        len_traj_pred=config.len_traj_pred,
        traj_pace=config.traj_pace,
        history_length=config.history_length,
        need_currdep=True,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.get("num_workers", 6),
        pin_memory=True,
        persistent_workers=config.get("num_workers", 6) > 0,
    )

    optimizer_planner = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer_refiner = torch.optim.Adam(refiner_model.parameters(), lr=config.learning_rate)
    scheduler_planner = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_planner, T_max=config.epochs)
    scheduler_refiner = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_refiner, T_max=config.epochs)

    (
        model,
        refiner_model,
        optimizer_planner,
        optimizer_refiner,
        train_loader,
        scheduler_planner,
        scheduler_refiner,
    ) = accelerate.prepare(
        model,
        refiner_model,
        optimizer_planner,
        optimizer_refiner,
        train_loader,
        scheduler_planner,
        scheduler_refiner,
    )

    if resume_planner_ckpt:
        load_ckpt(resume_planner_ckpt, model, accelerate)
    if resume_refiner_ckpt:
        load_ckpt(resume_refiner_ckpt, refiner_model, accelerate)

    model.train()
    refiner_model.train()
    for epoch in range(config.epochs):
        train_refiner_this_epoch = epoch >= refiner_start_epoch
        prev_batch_end = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{config.epochs}]", disable=not accelerate.is_main_process)
        for batch_id, batch in enumerate(pbar):
            batch_start = time.time()
            data_wait_time = batch_start - prev_batch_end
            (
                current_depth,
                rgb,
                depth,
                history_pose,
                gt_pose,
                floor_plan,
                img_goal,
                point_goal,
                object_goal,
                idx,
            ) = batch
            planner_step_start = time.time()
            noise_pred, noise = model(
                rgb, depth, history_pose, floor_plan, img_goal, point_goal, object_goal, gt_pose, mode='train'
            )
            planner_loss = ((noise_pred - noise) ** 2).mean()
            optimizer_planner.zero_grad()
            planner_loss.backward()
            accelerate.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer_planner.step()
            planner_step_time = time.time() - planner_step_start

            if accelerate.is_main_process:
                wandb.log({"planner/train_loss": planner_loss.item()})

            need_rollout = train_refiner_this_epoch or (batch_id != 0 and batch_id % config.eval_per_batches == 0)
            rollout_time = 0.0
            if need_rollout:
                rollout_start = time.time()
                with torch.no_grad():
                    action, gt = model(
                        rgb, depth, history_pose, floor_plan, img_goal, point_goal, object_goal, gt_pose, mode='eval_1'
                    )
                rollout_time = time.time() - rollout_start

            if batch_id != 0 and batch_id % config.eval_per_batches == 0:
                with torch.no_grad():
                    action_img = accelerate.unwrap_model(model).inference(
                        rgb, depth, history_pose, floor_plan, img_goal, None, None
                    )
                    action_point = accelerate.unwrap_model(model).inference(
                        rgb, depth, history_pose, floor_plan, None, point_goal, None
                    )
                    action_obj = accelerate.unwrap_model(model).inference(
                        rgb, depth, history_pose, floor_plan, None, None, object_goal
                    )
                traj_cosine_sim = action_reduce(
                    (
                        F.cosine_similarity(torch.flatten(action_img, start_dim=1), torch.flatten(gt, start_dim=1), dim=1)
                        + F.cosine_similarity(torch.flatten(action_point, start_dim=1), torch.flatten(gt, start_dim=1), dim=1)
                        + F.cosine_similarity(torch.flatten(action_obj, start_dim=1), torch.flatten(gt, start_dim=1), dim=1)
                    ) / 3.0
                )
                if accelerate.is_main_process:
                    wandb.log({"planner/eval_traj_cosine_sim": traj_cosine_sim.mean().item()})
                    idx_list = list(idx)
                    for i in range(len(idx_list)):
                        draw_multi_goal(
                            history_pose=history_pose[i].detach().cpu().numpy(),
                            predicted_img_pose=action_img[i].detach().cpu().numpy(),
                            predicted_point_pose=action_point[i].detach().cpu().numpy(),
                            predicted_obj_pose=action_obj[i].detach().cpu().numpy(),
                            gt_pose=gt[i].detach().cpu().numpy(),
                            save_path=os.path.join(wandb.run.dir, f"planner_epoch_{epoch+1}_sample_{idx_list[i]}_train.png"),
                            rgb_image=rgb[i, -1].detach().cpu().numpy(),
                            goal_image=img_goal[i].detach().cpu().numpy(),
                            object_goal=object_goal[i],
                        )

            refiner_step_time = 0.0
            refiner_ran_this_batch = False
            if train_refiner_this_epoch:
                refiner_step_start = time.time()
                # Skip samples whose current depth frame is effectively blank.
                valid_frac = (current_depth > 0).float().mean(dim=(1, 2))
                valid_sample_mask = valid_frac >= 0.05
                n_dropped = (~valid_sample_mask).sum().item()
                if n_dropped > 0 and accelerate.is_main_process:
                    wandb.log({"refiner/dropped_corrupt_depth_samples": n_dropped})

                if valid_sample_mask.any():
                    refiner_ran_this_batch = True
                    action_f = action[valid_sample_mask]
                    current_depth_f = current_depth[valid_sample_mask]
                    gt_f = gt[valid_sample_mask]
                    idx_f = [idx[i] for i in range(len(idx)) if valid_sample_mask[i]]

                    noise_pred_r, noise_r, occupancy_map, sampled_distance_values, refined_actions = refiner_model(
                        action_f, current_depth_f.unsqueeze(1), gt_f, idx_f, accelerate, 'train'
                    )
                    safe_distance = torch.clamp(sampled_distance_values, min=-5.0, max=5.0)
                    obstacle_loss = torch.mean(torch.exp(-config.obstacle_loss_weight * safe_distance))
                    diffusion_loss = ((noise_pred_r - noise_r) ** 2).mean()
                    anchor_loss = ((refined_actions - action_f) ** 2).mean()
                    refiner_loss = (
                        config.similarity_loss_weight * diffusion_loss
                        + 0.05 * obstacle_loss
                        + config.anchor_loss_weight * anchor_loss
                    )

                    optimizer_refiner.zero_grad()
                    refiner_loss.backward()
                    accelerate.clip_grad_norm_(refiner_model.parameters(), max_norm=1.0)
                    optimizer_refiner.step()
                    refiner_step_time = time.time() - refiner_step_start

                    if accelerate.is_main_process:
                        wandb.log({
                            "refiner/train_loss": refiner_loss.item(),
                            "refiner/obstacle_loss": obstacle_loss.item(),
                            "refiner/diffusion_loss": diffusion_loss.item(),
                            "refiner/anchor_loss": anchor_loss.item(),
                        })

                if batch_id != 0 and batch_id % config.eval_per_batches == 0:
                    with torch.no_grad():
                        refined_img, occupancy_map = refiner_model(
                            action_img.clone().detach(), current_depth.unsqueeze(1).clone().detach(), gt.clone().detach(), idx, accelerate, 'eval'
                        )
                        refined_point, _ = refiner_model(
                            action_point.clone().detach(), current_depth.unsqueeze(1).clone().detach(), gt.clone().detach(), idx, accelerate, 'eval'
                        )
                        refined_obj, _ = refiner_model(
                            action_obj.clone().detach(), current_depth.unsqueeze(1).clone().detach(), gt.clone().detach(), idx, accelerate, 'eval'
                        )
                    if accelerate.is_main_process:
                        idx_list = list(idx)
                        for i in range(len(idx_list)):
                            draw_occupancy_map(
                                config,
                                occupancy_map=occupancy_map[i].detach().cpu().numpy(),
                                orin_traj=action_img[i].detach().cpu().numpy(),
                                refined_traj=refined_img[i].detach().cpu().numpy(),
                                gt_traj=gt[i].detach().cpu().numpy(),
                                save_path=os.path.join(wandb.run.dir, f"refiner_epoch_{epoch+1}_sample_{idx_list[i]}_img_train.png"),
                                goal_type="img_goal",
                                rgb_image=rgb[i, -1].detach().cpu().numpy()
                            )
                            draw_occupancy_map(
                                config,
                                occupancy_map=occupancy_map[i].detach().cpu().numpy(),
                                orin_traj=action_point[i].detach().cpu().numpy(),
                                refined_traj=refined_point[i].detach().cpu().numpy(),
                                gt_traj=gt[i].detach().cpu().numpy(),
                                save_path=os.path.join(wandb.run.dir, f"refiner_epoch_{epoch+1}_sample_{idx_list[i]}_point_train.png"),
                                goal_type="point_goal",
                                rgb_image=rgb[i, -1].detach().cpu().numpy()
                            )
                            draw_occupancy_map(
                                config,
                                occupancy_map=occupancy_map[i].detach().cpu().numpy(),
                                orin_traj=action_obj[i].detach().cpu().numpy(),
                                refined_traj=refined_obj[i].detach().cpu().numpy(),
                                gt_traj=gt[i].detach().cpu().numpy(),
                                save_path=os.path.join(wandb.run.dir, f"refiner_epoch_{epoch+1}_sample_{idx_list[i]}_obj_train.png"),
                                goal_type="obj",
                                rgb_image=rgb[i, -1].detach().cpu().numpy()
                            )

            batch_total_time = time.time() - batch_start
            prev_batch_end = time.time()
            if accelerate.is_main_process:
                wandb.log({
                    "time/data_wait": data_wait_time,
                    "time/planner_step": planner_step_time,
                    "time/rollout": rollout_time,
                    "time/refiner_step": refiner_step_time,
                    "time/batch_total": batch_total_time,
                })
                postfix = {"planner_loss": f"{planner_loss.item():.3f}", "s/it": f"{batch_total_time:.2f}"}
                if refiner_ran_this_batch:
                    postfix["refiner_loss"] = f"{refiner_loss.item():.3f}"
                pbar.set_postfix(postfix)

                log_msg = (
                    f"epoch={epoch+1}/{config.epochs} batch={batch_id} "
                    f"planner_loss={planner_loss.item():.4f}"
                )
                if refiner_ran_this_batch:
                    log_msg += (
                        f" refiner_loss={refiner_loss.item():.4f} "
                        f"obstacle_loss={obstacle_loss.item():.4f} diffusion_loss={diffusion_loss.item():.4f} "
                        f"anchor_loss={anchor_loss.item():.4f}"
                    )
                elif train_refiner_this_epoch:
                    log_msg += " refiner_loss=skipped(all samples had corrupt depth)"
                log_msg += (
                    f" | data_wait={data_wait_time:.3f}s planner={planner_step_time:.3f}s "
                    f"rollout={rollout_time:.3f}s refiner={refiner_step_time:.3f}s total={batch_total_time:.3f}s"
                )
                logger.info(log_msg)

            if batch_id != 0 and batch_id % config.save_pre_batches == 0:
                accelerate.wait_for_everyone()
                if accelerate.is_main_process:
                    torch.save(accelerate.unwrap_model(model).state_dict(), os.path.join(wandb.run.dir, f"planner_epoch_{epoch+1}_{batch_id}.pth"))
                    if train_refiner_this_epoch:
                        torch.save(accelerate.unwrap_model(refiner_model).state_dict(), os.path.join(wandb.run.dir, f"refiner_epoch_{epoch+1}_{batch_id}.pth"))

        scheduler_planner.step()
        if train_refiner_this_epoch:
            scheduler_refiner.step()


def parse_args():
    parser = argparse.ArgumentParser(description="Train the FloVerse navigation model (planner, then planner+refiner).")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "config" / "floverse.yaml"),
        help="Path to the training config YAML.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="train",
        help="wandb run name.",
    )
    parser.add_argument(
        "--resume_planner_ckpt",
        type=str,
        default=None,
        help="Path to a planner checkpoint to resume from. If not set, the planner trains from scratch.",
    )
    parser.add_argument(
        "--resume_refiner_ckpt",
        type=str,
        default=None,
        help="Path to a refiner checkpoint to resume from. If not set, the refiner trains from scratch.",
    )
    parser.add_argument(
        "--refiner_start_epoch",
        type=int,
        default=None,
        help="Epoch (0-indexed) at which refiner training turns on. Defaults to config.refiner_start_epoch, or 0.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = OmegaConf.load(args.config)
    train(
        config,
        run_name=args.run_name,
        resume_planner_ckpt=args.resume_planner_ckpt,
        resume_refiner_ckpt=args.resume_refiner_ckpt,
        refiner_start_epoch=args.refiner_start_epoch,
    )

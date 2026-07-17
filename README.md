# FloVerse: Floor Plan-Guided Multi-Modal Navigation

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://wikiahuang.github.io/floverse/)
[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2606.14267)
[![Dataset](https://img.shields.io/badge/Dataset-ModelScope-green)](https://modelscope.cn/datasets/weiqihuang/floverse-1.6k)

*Weiqi Huang, Shuangyi Dong, Jiaxin Li, Yifei Guo, Zan Wang, Wei Liang*

FloVerse is a floorplan-guided diffusion policy for point-goal, image-goal, and object-goal navigation.

![FloVerse pipeline](https://wikiahuang.github.io/floverse/assets/pipeline.png)

## Setup

### a. Create the environment

```bash
conda create -n floverse python=3.8 -y
conda activate floverse
cd floverse
```

```bash
pip install -r config/requirements.txt
pip install modelscope
```

### b. Install diffusion_policy

`diffusion_policy` is not published on PyPI and must be cloned into the project root:

```bash
git clone https://github.com/real-stanford/diffusion_policy.git
pip install -e diffusion_policy/
```

It must live at `floverse/diffusion_policy/` for the import path in [`utils/train.py`](utils/train.py) to resolve correctly.

### c. Install iGibson

FloVerse evaluation depends on a local iGibson checkout under the repository root:

```bash
git clone https://github.com/StanfordVL/iGibson.git --recursive
pip install -e iGibson/
```

### d. Prepare iGibson scene assets

Next, prepare the scene data used by iGibson evaluation:

Download the shared iGibson assets:

```bash
python -m igibson.utils.assets_utils --download_assets
```

Then download the cleaned FloVerse scene dataset from ModelScope and extract it into `iGibson/igibson/data/`:

```bash
cd floverse/iGibson/igibson/data
modelscope download --repo-type dataset --local-dir . weiqihuang/floverse-1.6k floverse_scene_dataset.tar
tar -xf floverse_scene_dataset.tar
```

After extraction, the directory layout should be:

```text
floverse/iGibson/igibson/data/
├── assets/
└── g_dataset/
```

### e. Prepare FloVerse data

The FloVerse training and evaluation data are hosted on [ModelScope](https://modelscope.cn/datasets/weiqihuang/floverse-1.6k/).

If you plan to train FloVerse, download the full dataset:

```bash
modelscope download --repo-type dataset --local-dir /path/to/floverse_data weiqihuang/floverse-1.6k
```

The full dataset contains:

- `train_without_obj` for point-goal training
- `train_with_obj` for image-goal and object-goal training
- `eval_without_obj` for point-goal evaluation
- `eval_with_obj` for image-goal and object-goal evaluation

After download, update `config/floverse.yaml` so that `data_folder` points to the training root:

```yaml
data_folder:
  - /path/to/floverse_data/train_without_obj
  - /path/to/floverse_data/train_with_obj
```

Each training scene directory should contain `floorplan.png` and `traj_*` subdirectories.

If you only need evaluation, download only the evaluation trajectories:

```bash
modelscope download --repo-type dataset --local-dir /path/to/floverse_eval --include "eval_without_obj/**" "eval_with_obj/**" weiqihuang/floverse-1.6k
```

This evaluation-only download is used together with `floverse_scene_dataset.tar` from step d.

### f. Repository layout

After setup, the project should look like:

```text
floverse/
├── ckpt/                       # released planner / refiner checkpoints
├── config/                     # training and evaluation configs
├── datasets/                   # dataloader
├── diffusion_policy/           # external dependency, cloned manually
├── eval/                       # point / image / object evaluation scripts
├── floorplan/                  # released floorplan assets
├── iGibson/                    # external dependency, cloned manually
├── model/                      # planner and refiner
├── results/                    # saved evaluation rollouts
└── utils/                      # training entrypoint and shared helpers
```

## Checkpoints

You can download our pretrained `planner` and `refiner` checkpoints from ModelScope and run evaluation directly:

```bash
modelscope download --repo-type dataset --local-dir ckpt --include "planner.pth" "refiner.pth" weiqihuang/floverse-1.6k
```

## Training

Launch multi-GPU training with `accelerate`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --multi_gpu --num_processes 8 utils/train.py --config config/floverse.yaml --run_name floverse_train
```

Notes:

- `config/floverse.yaml` controls dataset paths, batch size, goal sampling probabilities, and refiner start epoch.
- Planner visualizations and checkpoints are saved to the active `wandb` run directory.

## Evaluation

Evaluation code lives in [`eval/`](eval/).

### a. Point-goal evaluation

```bash
CUDA_VISIBLE_DEVICES=0,1 python -u -m eval.point_goal --scenes_dir /path/to/eval_without_obj/HM3D --traj_save_dir /path/to/results/point_goal
```

### b. Image-goal evaluation

```bash
CUDA_VISIBLE_DEVICES=0,1 python -u -m eval.image_goal --scenes_dir /path/to/eval_with_obj --traj_save_dir /path/to/results/image_goal
```

### c. Object-goal evaluation

```bash
CUDA_VISIBLE_DEVICES=0,1 python -u -m eval.object_goal --scenes_dir /path/to/eval_with_obj --traj_save_dir /path/to/results/object_goal
```

### d. Useful evaluation flags

```text
--scene_ids SCENE_A SCENE_B ...
--num_workers N
--execute_steps 9
--use_refiner / --no_refiner
--save_images / --no_save_images
--collision_limit 15
--max_steps 500
--max_trajs_per_scene 30
```

By default, results are saved under:

```text
results/
├── point_goal/
├── image_goal/
└── object_goal/
```

### e. Metric computation

Use [`eval/evaluate.py`](eval/evaluate.py) to compute `SR`, `SPL`, and `SoftSPL` from saved trajectories:

```bash
python -u -m eval.evaluate --traj_dir /path/to/results/image_goal --shortest_traj_dir /path/to/eval_with_obj
```

Optional metric flags:

```text
--collision_th 15
--success_distance 1.0
```

## Citation

If you use FloVerse in your research, please cite our paper:

```bibtex
@inproceedings{floverse2026,
  author    = {Weiqi Huang and Shuangyi Dong and Jiaxin Li and Yifei Guo and Zan Wang and Wei Liang},
  title     = {FloVerse: Floor Plan-Guided Multi-Modal Navigation},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

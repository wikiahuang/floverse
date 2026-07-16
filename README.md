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

It must live at `floverse/diffusion_policy/` for the import path in [`utils/train.py`](/media/data/weiqi_data/code/floverse/utils/train.py) to resolve correctly.

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

Then download the cleaned FloVerse `g_dataset` release from ModelScope and extract it into `iGibson/igibson/data/`:

```bash
cd floverse/iGibson/igibson/data
wget <MODELSCOPE_G_DATASET_URL> -O floverse_g_dataset_clean.tar
tar -xf floverse_g_dataset_clean.tar
```

After extraction, the directory layout should be:

```text
floverse/iGibson/igibson/data/
├── assets/
└── g_dataset/
```

### e. Prepare FloVerse data

The FloVerse training and evaluation data are hosted on [ModelScope](https://modelscope.cn/datasets/weiqihuang/floverse-1.6k/).

```bash
modelscope download --dataset weiqihuang/floverse-1.6k
```

The downloaded dataset contains:

- `train_dataset` for training
- `eval_without_obj/HM3D` for point-goal evaluation
- `eval_with_obj` for image-goal and object-goal evaluation

After download, update [`config/floverse.yaml`](/media/data/weiqi_data/code/floverse/config/floverse.yaml) so that `data_folder` points to the training root:

```yaml
data_folder:
  - /path/to/floverse/train_dataset
```

The training root should contain scene folders, and each scene folder should contain `floorplan.png` plus `traj_*` subdirectories.

### f. Repository layout

After setup, the project should look like:

```text
floverse/
├── ckpt/                       # released planner / refiner checkpoints
├── config/                     # training and evaluation configs
├── datasets/                   # dataset loader and overlap-scene list
├── diffusion_policy/           # external dependency, cloned manually
├── eval/                       # point / image / object evaluation scripts
├── floorplan/                  # released floorplan assets
├── iGibson/                    # external dependency, cloned manually
│   └── igibson/data/
│       ├── assets/
│       └── g_dataset/
├── model/                      # planner and refiner
├── results/                    # saved evaluation rollouts
└── utils/                      # training entrypoint and shared helpers
```

## Training

Single node, multi-GPU training is launched with `accelerate`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --multi_gpu --num_processes 8 \
  utils/train.py \
  --config config/floverse.yaml \
  --run_name floverse_train
```

Notes:

- `config/floverse.yaml` controls dataset paths, batch size, goal sampling probabilities, and refiner start epoch.
- Planner visualizations and checkpoints are saved to the active `wandb` run directory.

## Evaluation

Evaluation code lives in [`eval/`](/media/data/weiqi_data/code/floverse/eval).

All three navigation scripts support CLI overrides. If you do not pass any CLI flags, they run with the defaults defined inside each script.

### a. Point-goal evaluation

```bash
CUDA_VISIBLE_DEVICES=0,1 python -u -m eval.point_goal \
  --scenes_dir /path/to/eval_without_obj/HM3D \
  --traj_save_dir /path/to/results/point_goal
```

### b. Image-goal evaluation

```bash
CUDA_VISIBLE_DEVICES=0,1 python -u -m eval.image_goal \
  --scenes_dir /path/to/eval_with_obj \
  --traj_save_dir /path/to/results/image_goal
```

### c. Object-goal evaluation

```bash
CUDA_VISIBLE_DEVICES=0,1 python -u -m eval.object_goal \
  --scenes_dir /path/to/eval_with_obj \
  --traj_save_dir /path/to/results/object_goal
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

Use [`eval/evaluate.py`](/media/data/weiqi_data/code/floverse/eval/evaluate.py) to compute `SR`, `SPL`, and `SoftSPL` from saved trajectories:

```bash
python -u -m eval.evaluate \
  --traj_dir /path/to/results/image_goal \
  --shortest_traj_dir /path/to/eval_with_obj
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

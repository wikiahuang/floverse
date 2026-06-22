# FloVerse: Floor Plan-Guided Multi-Modal Navigation

[Project Page](https://wikiahuang.github.io/floverse/docs/) · [Paper (arXiv)](#) · [Dataset](https://modelscope.cn/datasets/weiqihuang/floverse-1.6k)

Weiqi Huang, Shuangyi Dong, Jiaxin Li, Yifei Guo, Zan Wang, Wei Liang
CVPR 2026

> **Status:** We are actively refactoring and optimizing the codebase ahead of release. A
> full, stable release is coming soon — thanks for your patience in the meantime.

## Code Structure

- **`floorplan`**: floor plan samples (Gibson / HM3D layouts).
- **`floorplan_reconstructor`**: scripts to reconstruct floor plans from mesh/point-cloud data.
- **`datasets`**: organizes the dataset into samples and batches for training.
- **`model`**: model implementation.
- **`utils`**: training/eval scripts and helper utilities.
- **`baseline`**: code for comparison baselines.
- **`config`**: training configs and dependency list.
- **`docs`**: project page source (served via GitHub Pages).

## Installation

TODO

## Dataset

The FloVerse dataset is available on ModelScope:

https://modelscope.cn/datasets/weiqihuang/floverse-1.6k

TODO: download/setup instructions.

## Training

TODO

## Testing / Deployment

TODO

## Baselines

TODO

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

## License

This project is released under the [MIT License](LICENSE).

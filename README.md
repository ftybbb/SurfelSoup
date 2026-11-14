# SurfelSoup

Official code release for:

**SurfelSoup: Learned Point Cloud Geometry Compression With a Probabilistic SurfelTree Representation**  
Tingyu Fan, Ran Gong, Yueyu Hu, Yao Wang

**ACM Multimedia 2026 (ACM MM '26) — Oral Presentation & Outstanding Paper**

## News

- **November 2026:** Selected for an **Oral presentation** and recognized as an **Outstanding Paper** at ACM Multimedia (ACM MM) 2026.
- **November 2026:** The paper is published in the ACM MM '26 proceedings (pp. 1773–1781).
- **Published paper:** https://doi.org/10.1145/3767308.3835549
- **arXiv preprint:** https://arxiv.org/abs/2602.00186

## Overview

SurfelSoup is an end-to-end learned framework for point cloud geometry compression.
It introduces:

- **pSurfel**: a probabilistic surfel representation for local occupancy modeling.
- **pSurfelTree**: an octree-like adaptive hierarchy for rate-distortion optimized granularity selection.

This folder (`SurfelSoup_v1`) contains the core model and evaluation/training scripts used in our experiments.

## Code Structure

- `SurfelSoup_v1.py`: main model definition (`get_model`).
- `models.py`: core network blocks, entropy modeling, coding logic.
- `dataset.py`: point cloud dataset loader and collate function.
- `test.py`: geometry compression evaluation script (includes reconstruction and PSNR computation).
- `trainer.py`: model training and validation script.
- `trainer.sh`: recommended SLURM/Singularity training launcher.

### Tested Environment

- Python 3.10.14
- CUDA Toolkit 11.8 (V11.8.89)
- PyTorch 2.4.0 (CUDA 11.8)
- MinkowskiEngine 0.5.4
- PyTorch3D 0.7.8
- Open3D, NumPy, h5py, and tqdm

### Suggested Setup (example)

```bash
conda create -n surfelsoup python=3.10 -y
conda activate surfelsoup

# Install PyTorch that matches your CUDA
# See: https://pytorch.org/get-started/locally/

pip install numpy open3d h5py tqdm

# Install MinkowskiEngine from official instructions:
# https://github.com/NVIDIA/MinkowskiEngine

# Install PyTorch3D from official instructions:
# https://github.com/facebookresearch/pytorch3d
```

## Data Preparation

`dataset.py` and `test.py` support point cloud files such as `.ply` (and partial hooks for `.npy` / `.h5`).

The input point cloud must be integer coordinates.

For evaluation, update `test_sequences` in `test.py`:

```python
test_sequences = [
    ['/path/to/sequence_or_single_ply', 1023],
]
```

- First item: a `.ply` file path or a directory containing `.ply` files.
- Second item: `peak_value` (geometry dynamic range, used by PSNR).

## Evaluation

1. Update model checkpoints in `models_and_ckpts` in `test.py`.
2. Update test data in `test_sequences` in `test.py`.
3. Run:

```bash
python test.py
```

Outputs:

- Reconstructed point cloud is written to `--out_dir` (default: `/scratch/tf2387/geo_result/rec.ply`).
- Console prints per-sequence RD metrics (`bpp`, D1-PSNR, D2-PSNR).

## Training

We recommend initializing the model from the provided pretrained checkpoint and
then fine-tuning it for the target rate-distortion setting. Training the entire
model from scratch may make the decision module unstable.

The provided `trainer.sh` follows this recommended setup and loads:

```text
./ckpts/pretrained.pth
```

### Recommended workflow

1. Edit the environment-specific paths in `trainer.sh`:
   - Singularity image and overlay paths.
   - Conda environment.
   - Training dataset path passed to `--dataset_dir`.
   - SLURM account, GPU constraint, output path, and other scheduler settings.
2. Select the desired training configuration, including `--lamb`,
   `--octree_end`, `--surfel_start`, and `--selection_mode`.
3. Keep `--pretrained='./ckpts/pretrained.pth'` enabled for stable fine-tuning.
4. Use a new `--exp_name` when starting a fine-tuning run.
5. Submit the job:

```bash
sbatch trainer.sh
```

The training command used by `trainer.sh` is equivalent to:

```bash
python -u trainer.py \
  --model=SurfelSoup_v1 \
  --save_every=5 \
  --epoch=40 \
  --batch_size=4 \
  --dataset_dir='/path/to/training/data' \
  --exp_name=s1 \
  --lamb=0.8 \
  --channels=32 \
  --hyper=1 \
  --learning_rate=0.0001 \
  --max_iter=800 \
  --val_num=0.006 \
  --layers=5 \
  --bar=0 \
  --octree_end=1 \
  --surfel_start=2 \
  --from_step=500 \
  --scheduler_step_size=15 \
  --selection_mode=gumbel \
  --pretrained='./ckpts/pretrained.pth'
```

`trainer.py` saves logs and checkpoints under:

```text
<root_dir>/<model>/<exp_name>/
├── log/
└── checkpoint/
    ├── model_last.pth
    ├── best_model.pth
    └── best_model_weights_only.pth
```

If `model_last.pth` already exists in the selected experiment directory,
`trainer.py` treats the run as a continuation. Therefore, use a fresh
`--exp_name` when beginning a new fine-tuning experiment from
`--pretrained`.

For training outside the original cluster environment, run the Python command
directly after installing the dependencies and replace all `/scratch/...`
paths with local paths.

## Citation

If you use SurfelSoup in your research, please cite the published ACM Multimedia 2026 paper:

```bibtex
@inproceedings{fan2026surfelsoup,
  author = {Fan, Tingyu and Gong, Ran and Hu, Yueyu and Wang, Yao},
  title = {SurfelSoup: Learned Point Cloud Geometry Compression With a Probabilistic SurfelTree Representation},
  booktitle = {Proceedings of the 34th ACM International Conference on Multimedia},
  year = {2026},
  pages = {1773--1781},
  publisher = {Association for Computing Machinery},
  doi = {10.1145/3767308.3835549},
  url = {https://doi.org/10.1145/3767308.3835549}
}
```

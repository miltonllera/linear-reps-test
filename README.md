# Linear Representations & Generalization

Tools and synthetic datasets for evaluating the latent representations of pre-trained vision models (e.g. DINOv2, DINOv3, and other HuggingFace vision backbones) on controlled factor-of-variation benchmarks (`dsprites`, `shapes3d`, and 2D/3D `pentominos`).

## Setup

This project uses [`uv`](https://docs.astral.sh/uv/) and requires Python `>=3.13` (see `pyproject.toml`).

```bash
uv sync
```

### Gated HuggingFace models

Models such as `facebook/dinov3-*` are gated on HuggingFace. After requesting access on the model card, authenticate locally before running:

```bash
uv run hf auth login
# or export HF_TOKEN=...
```

Ungated alternatives such as `facebook/dinov2-small` and `facebook/dinov2-with-registers-small` work without authentication.

## Project structure

- `src/dataset/` — PyTorch dataset loaders and synthetic sprite generators (`dsprites`, `shapes3d`, `pentominos`, `pentominos_3d`, `non_pentominos`, `colored_dsprites`, `multi_shape`).
- `src/analysis/` — Representation analysis and visualization utilities (PCA/UMAP reduction, trustworthiness/continuity metrics, Plotly visualizations, DCI, MDS, Hinton diagrams, traversals).
- `scripts/` — Entry-point scripts for dataset generation, embedding extraction, and latent analysis.
- `bin/` — Shell helpers and batch runners (`run_dino_tests.sh`, `gpu_queue.sh`, `generate_pentominos.sh`, `generate_non_pentominos.sh`).

## Datasets

Datasets registered in `scripts/common.py`:

| Name | Loader | Default path |
| --- | --- | --- |
| `dsprites` | `DSprites` | `data/datasets/dsprites/dsprites_train.hdf5` |
| `shapes3d` | `Shapes3D` | `data/datasets/shapes3d/3dshapes.h5` |
| `pentominos` | `Pentominos` | `data/datasets/pentominos` |
| `pentominos_colored` | `Pentominos` | `data/datasets/pentominos_colored` |
| `pentominos_colored_black_background` | `Pentominos` | `data/datasets/pentominos_colored_black_background` |
| `pentominos_3d` | `Pentominos3D` | `~/Repos/vision/blender stuff/data/temp/pentominos_3d` |

Any default path can be overridden at runtime with `--dataset_path`.

### Generating Pentomino datasets

Generate 2D Pentomino datasets with `scripts/generate_pentominos.py` (see `bin/generate_pentominos.sh` for preset configurations):

```bash
uv run python -m scripts.generate_pentominos \
  --height 60 --width 60 --pad 2 2 --aa 10 \
  --lim_angles 0 360 --num_angles 16 \
  --lim_scales 1.5 3.0 --num_scales 3 \
  --num_colors 10 --num_bg_colors 1 \
  --lim_xs -10 10 --num_xs 16 \
  --lim_ys -10 10 --num_ys 16 \
  --folder data/datasets/pentominos_colored_black_background
```

Other dataset utilities:
- `scripts/generate_3d_pentominos.py` — 3D Pentomino generation.
- `scripts/generate_non_pentominos.py` / `bin/generate_non_pentominos.sh` — non-pentomino shape generation.
- `scripts/visualize_dataset.py` — inspect dataset samples.

## Running latent analysis

`scripts/analyze_hf_latents.py` extracts and caches embeddings from a HuggingFace vision model (`HFVisionEmbedder` in `scripts/common.py`), runs PCA (and optional UMAP), logs per-factor statistics on PC1–PC3 ($\eta^2$ for categorical factors, Pearson $r$ for continuous factors), and exports interactive 3D Plotly HTML plots and CSV summaries.

```bash
# Quick PCA run on a 5,000-image subsample with DINOv2
uv run python -m scripts.analyze_hf_latents \
  --dataset_name pentominos_colored_black_background \
  --model_id facebook/dinov2-small \
  --batch_size 256 \
  --max_samples 5000

# Full dataset run with DINOv3 (default model) and UMAP
uv run python -m scripts.analyze_hf_latents \
  --dataset_name shapes3d \
  --model_id facebook/dinov3-vits16-pretrain-lvd1689m \
  --batch_size 256 \
  --run_umap
```

Common flags:
- `--dataset_name`: one of the dataset keys in `scripts/common.py` (required).
- `--model_id`: HuggingFace model ID (default: `facebook/dinov3-vits16-pretrain-lvd1689m`).
- `--dataset_path`: override the default dataset path.
- `--filter_expr`: optional factor filter expression to split the dataset into train/held-out subsets (omit to analyze the full dataset).
- `--max_samples`: cap the number of embedded images (uniformly subsampled).
- `--sample_size`: maximum number of points plotted in scatter plots (default: `10000`).
- `--run_umap`: also compute UMAP embeddings and trustworthiness/continuity metrics (`--umap_n_neighbors`, `--umap_min_dist`, `--umap_metric`, `--umap_subsample_ratio`).
- `--force_recompute`: ignore cached `embeddings.npz` and re-extract embeddings.
- `--output_dir`: base output directory (default: `plots/hf_latent_analysis`).

Outputs are written to `<output_dir>/<model_slug>/<dataset_tag>/`:
- `embeddings.npz` — cached embeddings for reuse across runs.
- `*_pca_<factor>.html` (and `*_umap_<factor>.html` if `--run_umap` is set) — interactive 3D scatter plots per generative factor.
- `*_pca_variance.*` — explained variance plot.
- `*_results.csv` — PCA explained variance (and UMAP metrics if enabled).

### Batch & multi-GPU helpers

- `bin/run_dino_tests.sh` — example end-to-end script generating `pentominos_colored_black_background` and running `analyze_hf_latents.py`.
- `bin/gpu_queue.sh` — reusable 1-job-per-GPU bash queue (`gpu_pool_init`, `gpu_pool_run`, `gpu_pool_wait`) for running multiple evaluations across GPUs.

## WIP analysis scripts

The following scripts from the original codebase are retained for upcoming experiments and still need to be migrated from the old model loader to `HFVisionEmbedder`:

- `scripts/latent_prediction.py` — predicting generative factors from latent representations.
- `scripts/mixed_effects.py` — mixed-effects statistical analysis of representations.

# CT-Sora

**CT-Sora** is a diffusion model for generating synthetic 3D CT scan video sequences. It is developed at the **IDEA Lab, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU), Germany**.

The project adapts the [Open-Sora](https://github.com/hpcaitech/Open-Sora) framework and replaces the original video generation pipeline with a medical imaging focus: training on CT volumetric data to enable controllable, high-quality CT video synthesis.

---

## Demo

<!-- Place demo images/videos in assets/ and update the paths below -->

---

## Architecture

- **Backbone**: MMDiT (Flux-style Multimodal Diffusion Transformer)
- **VAE**: HunyuanVideo causal 3D VAE for video encoding/decoding
- **Training framework**: [ColossalAI](https://github.com/hpcaitech/ColossalAI) (ZeRO, tensor parallel, sequence parallel)
- **Experiment tracking**: Weights & Biases (W&B)

### Training Pipeline

| Stage | Script | Config | Description |
|-------|--------|--------|-------------|
| Stage 1 | `scripts/diffusion/train_huny.py` | `configs/diffusion/train/stage1_new.py` | Text-conditioned CT video generation |
| Stage 2 | `scripts/diffusion/train_huny_denoiser.py` | *(denoiser config)* | Video-to-video denoising refinement |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/WongJiayi/CT-Sora.git
cd CT-Sora

# Install dependencies
pip install -e .
pip install -r requirements.txt
```

### Requirements

- Python >= 3.10
- PyTorch >= 2.4
- ColossalAI
- HuggingFace Transformers, Diffusers
- einops, decord, wandb

See `requirements.txt` for the full list.

---

## Model Weights

Download the required pretrained weights and place them in the `ckpt/` directory:

| Model | File | Source |
|-------|------|--------|
| HunyuanVideo VAE | `ckpt/hunyuan_vae.safetensors` | [Tencent HunyuanVideo](https://github.com/Tencent/HunyuanVideo) |

---

## Data Preparation

CT-Sora expects video data referenced by a CSV file, where the first column contains paths to `.mp4` video files:

```
/path/to/scan_001.mp4
/path/to/scan_002.mp4
...
```

To generate a CSV from a folder of videos, run:

```bash
python scripts/cnv/generate_csv.py --input_dir /path/to/videos --output /path/to/data.csv
```

---

## Training

### Stage 1 — CT Video Generation

Edit `configs/diffusion/train/stage1_new.py` to set your data path and output directory, then run:

```bash
torchrun --nproc_per_node 8 \
    scripts/diffusion/train_huny.py \
    configs/diffusion/train/stage1_new.py \
    --dataset.data-path /path/to/ct_videos.csv
```

### Stage 2 — Denoising Refinement

```bash
torchrun --nproc_per_node 8 \
    scripts/diffusion/train_huny_denoiser.py \
    configs/diffusion/train/stage1_new.py \
    --load /path/to/stage1/checkpoint
```

### Resuming from Checkpoint

Set the `load` field in your config:

```python
load = "/path/to/checkpoint/epoch-global_step"
```

---

## Inference

```bash
python scripts/diffusion/inference.py \
    configs/diffusion/inference/256px.py \
    --ckpt-path /path/to/checkpoint
```

---

## Project Structure

```
CT-Sora/
├── opensora/                       # Core library
│   ├── models/
│   │   ├── mmdit/                  # MMDiT transformer backbone
│   │   ├── hunyuan_vae/            # HunyuanVideo causal 3D VAE
│   │   ├── dc_ae/                  # DC-AE spatial autoencoder
│   │   └── vae/                    # VAE utilities
│   ├── datasets/                   # Data loading and sampling
│   ├── acceleration/               # Distributed training utilities
│   └── utils/                      # Config, logging, optimizer, etc.
├── scripts/
│   ├── diffusion/
│   │   ├── train_huny.py           # Stage 1 training
│   │   ├── train_huny_denoiser.py  # Stage 2 denoising training
│   │   └── inference.py            # Inference
│   ├── vae/                        # VAE training and evaluation
│   └── cnv/                        # Data conversion utilities
├── configs/
│   ├── diffusion/
│   │   ├── train/
│   │   │   ├── image_new.py        # Base training config
│   │   │   └── stage1_new.py       # Stage 1 config
│   │   └── inference/              # Inference configs
│   └── vae/                        # VAE configs
├── assets/                         # Demo images and example outputs
├── docs/                           # Documentation
├── ckpt/                           # Model weights (not tracked by git)
└── requirements.txt
```

---

## License

### This Project

Copyright 2025 IDEA Lab, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU), Germany.

This project is a **derivative work** based on [Open-Sora](https://github.com/hpcaitech/Open-Sora) by HPC-AI Technology Inc., and is licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) for the full text.

### Third-Party Components

This project incorporates the following third-party components, each subject to its own license:

| Component | License | Source |
|-----------|---------|--------|
| [Open-Sora](https://github.com/hpcaitech/Open-Sora) | Apache 2.0 — Copyright 2024 HPC-AI Technology Inc. | Base framework |
| [HunyuanVideo](https://github.com/Tencent/HunyuanVideo) | Tencent Hunyuan Community License | VAE weights |
| [FLUX](https://github.com/black-forest-labs/flux) | Apache 2.0 — Copyright 2024 Black Forest Labs | MMDiT architecture |
| [EfficientViT](https://github.com/mit-han-lab/efficientvit) | Apache 2.0 — Copyright 2023 Han Cai | DC-AE components |
| [T5](https://github.com/google-research/text-to-text-transfer-transformer) | Apache 2.0 — Copyright 2019 Google | Text encoder |
| [CLIP](https://github.com/openai/CLIP) | MIT License — Copyright 2021 OpenAI | Text encoder |

> **Important — HunyuanVideo VAE**: The VAE weights used in this project are derived from Tencent HunyuanVideo and are subject to the [Tencent Hunyuan Community License Agreement](https://github.com/Tencent/HunyuanVideo/blob/main/LICENSE). Usage of these weights must comply with that license, including geographic restrictions and acceptable use policies. See [LICENSE](LICENSE) for the full terms.

---

## Acknowledgements

This work builds upon [Open-Sora](https://github.com/hpcaitech/Open-Sora) by HPC-AI Technology Inc. We thank the Open-Sora team for their open-source contribution to the video generation community.

---

## Citation

If you use CT-Sora in your research, please cite:

```bibtex
@misc{ctsora2025,
  title        = {CT-Sora: Diffusion-based Synthetic 3D CT Video Generation},
  author       = {IDEA Lab, FAU Erlangen-Nürnberg},
  year         = {2025},
  institution  = {Friedrich-Alexander-Universität Erlangen-Nürnberg},
  howpublished = {\url{https://github.com/WongJiayi/CT-Sora}}
}
```
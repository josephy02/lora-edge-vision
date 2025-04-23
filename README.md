# LoRA Edge Vision

This repository demonstrates end-to-end Low-Rank Adaptation (LoRA) fine-tuning of Stable Diffusion for aerial imagery, optimized for on-device edge inference using ONNX.

## Repository Structure

```
.
├── .gitignore
├── configs/
│   └── train.yaml           # Training configuration
├── data/
│   ├── raw_images/          # Original images (ignored)
│   ├── processed_images/    # Processed & resized images
│   └── latents/             # Precomputed VAE latents
├── models/
│   ├── lora_adapters/       # Trained LoRA weights (ignored)
│   └── sd_lora_pipeline/    # Merged base+LoRA pipeline (ignored)
├── onnx/                    # ONNX export outputs (ignored)
├── outputs/                 # Sample inference outputs (ignored)
├── scripts/
│   ├── precompute_latents.py # Precompute VAE latents from images
│   ├── merge_pipeline.py     # Merge base SD + LoRA weights
│   └── test_unet_onnx.py     # Smoke-test ONNX UNet forward
├── src/
│   ├── dataset.py           # Image & latent dataset loaders
│   ├── train_lora.py        # LoRA fine-tuning script
│   ├── inference.py         # PyTorch inference with LoRA
│   ├── export_onnx.py       # ONNX export via manual torch.onnx
│   └── inference_onnx.py    # ONNX Runtime inference script
├── environment.yaml         # Conda environment spec
├── requirements.txt         # pip requirements
└── README.md
```

## Quickstart

### 1. Setup environment

```bash
conda env create -f environment.yaml
conda activate lora-sd
pip install -r requirements.txt
```

### 2. Precompute VAE latents

```bash
python scripts/precompute_latents.py
```

### 3. Train LoRA adapter

```bash
python src/train_lora.py --config configs/train.yaml
```

### 4. Inspect sample outputs

```bash
python src/inference.py \
  --base_model runwayml/stable-diffusion-v1-5 \
  --lora_adapter models/lora_adapters \
  --prompt "YOUR PROMPT HERE" \
  --output_dir outputs
```

### 5. Merge base + LoRA into full pipeline

```bash
python scripts/merge_pipeline.py
```

### 6. Export to ONNX

```bash
pip install optimum[exporters] onnx onnxruntime
optimum-cli export onnx \
  --model models/sd_lora_pipeline \
  onnx/ \
  --task text-to-image \
  --library diffusers \
  --framework pt \
  --opset 14 \
  --batch_size 1 \
  --height 512 \
  --width 512
```

### 7. Inference with ONNX Runtime

```bash
python src/inference_onnx.py \
  --onnx_dir onnx/ \
  --prompt "YOUR PROMPT HERE"
```

## Directory & File Notes

- `.gitignore`: excludes large model and data artifacts.
- `configs/train.yaml`: hyperparameters for LoRA training.
- `src/train_lora.py`: MPS/CPU‐friendly script with latent precompute support.
- `scripts/merge_pipeline.py`: merges base SD + LoRA into a single HF pipeline.
- `src/export_onnx.py`: manual ONNX export for the LoRA‐wrapped UNet.
- `src/inference_onnx.py`: runs the ONNX pipeline on CPU via ONNX Runtime.


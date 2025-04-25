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
│   ├── processed_images/    # Processed & resized images (ignored)
│   └── latents/             # Precomputed VAE latents (ignored)
├── models/
│   ├── lora_adapters/       # Trained LoRA weights (ignored)
│   └── sd_lora_pipeline/    # Merged base+LoRA pipeline (ignored)
├── onnx/                    # ONNX export outputs (ignored)
├── outputs/                 # Sample inference outputs
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

### 8. Run Complete Demo

To demonstrate the entire pipeline and benchmark performance check out `notebooks/exploratory.ipynb`

This will:
1. Generate images with both PyTorch and ONNX models
2. Benchmark performance differences
3. Create a visual comparison gallery
4. Save all results to `demo_outputs/`

#### Key Features

- **Memory-Efficient Training**: Optimized for devices with limited GPU memory, including Apple Silicon MPS support
- **Checkpointing**: Saves best model and regular checkpoints to prevent losing progress
- **ONNX Optimization**: Specialized export paths for different platforms, including macOS
- **Performance Benchmarking**: Tools to compare PyTorch vs ONNX inference speed and memory usage
- **Edge Deployment**: Optimized for on-device inference with reduced model size and memory footprint

#### Technical Notes

- **LoRA Configuration**: The default configuration uses rank=4, alpha=16 for a good balance of parameter efficiency and quality
- **Latent Precomputation**: Training uses precomputed latents to bypass VAE encoding during training, significantly speeding up the process
- **Attention Handling**: Special handling of attention mechanisms for ONNX compatibility
- **Quantization Options**: INT8 quantization available for further size reduction

#### Platform-Specific Notes

#### Apple Silicon (M1/M2/M3)

- Uses Metal Performance Shaders (MPS) backend during training when available
- Requires specific ONNX export handling for attention mechanisms
- May benefit from the macOS-specific export scripts

#### CUDA GPUs

- Faster training and inference compared to CPU/MPS
- Standard ONNX export works well without special handling
- Supports larger batch sizes during training

### Edge Devices

- INT8 quantization recommended for maximum efficiency
- Use only UNet in ONNX format if memory is very constrained
- Consider lower resolution (256×256) for faster inference

#### Customization

- Modify `configs/train.yaml` to adjust training parameters
- Edit prompt templates in dataset loaders for domain-specific fine-tuning
- Adjust ONNX export parameters for specific target devices
# LoRA Adapter for Edge Vision (Satellite/Drone Imagery)

**Project Goal:** Fine‑tune a Stable Diffusion model using LoRA adapters on satellite/drone imagery, then package it for efficient on‑device inference.

## 1. Repository Structure

```
lora-edge-vision/
├── data/
│   ├── raw_images/          # Original satellite/drone images
│   └── processed_images/    # Resized, normalized dataset
├── models/
│   ├── base_model/          # Stable Diffusion weights
│   └── lora_adapters/       # Trained LoRA adapter weights
├── notebooks/
│   └── exploratory.ipynb    # EDA & visualization
├── src/
│   ├── dataset.py           # Dataset loading & preprocessing
│   ├── train_lora.py        # LoRA fine‑tuning script
│   ├── inference.py         # Generate with base+adapter
│   ├── export_onnx.py       # ONNX export & quantization
│   └── utils.py             # Shared utilities
├── configs/
│   └── train.yaml           # Training hyperparameters
├── requirements.txt         # Python dependencies
├── README.md                # This overview
└── .gitignore               # Ignore files
```

## 2. Environment Setup

1. **Create & activate virtualenv**:

```bash
python3.10 -m venv venv
source venv/bin/activate
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

***Key deps:**** `torch`, `diffusers`, `transformers`, `accelerate`, `peft` (or `cloneofsimo/lora`), `datasets`, `Pillow`, `onnxruntime`.*

## 3. Data Preparation

* **Raw images**: Place all satellite/drone imagery in `data/raw_images/`.
* **Preprocessing**: Use `src/dataset.py` to:
   * Resize to model resolution (e.g. 512×512).
   * Normalize pixel values.
   * Optionally pair with text prompts (e.g. `"Aerial view of farmland"`).

## 4. Training LoRA Adapters

* **Run training**:

```bash
python src/train_lora.py --config configs/train.yaml
```

* **Config options** include:
   * Base model checkpoint path
   * Dataset directory
   * LoRA rank & alpha
   * Learning rate & batch size
   * Output adapter directory

## 5. Inference & Packaging

1. **Inference**:

```bash
python src/inference.py \
  --base_model models/base_model \
  --lora_adapter models/lora_adapters/adapter.pt \
  --prompt "Thermal map of industrial plant"
```

2. **ONNX Export & Quantization**:

```bash
python src/export_onnx.py \
  --model_dir models/base_model \
  --lora_dir models/lora_adapters \
  --output onnx/ \
  --quantize int8
```

## 6. Next Steps

* **Benchmark**: Evaluate latency & accuracy on CPU and ARM64 devices.
* **Merge Adapters**: Experiment with combining multiple task‑specific LoRA modules.
* **Edge Deployment**: Integrate into a mobile app, WebAssembly UI, or NVIDIA Jetson demo.
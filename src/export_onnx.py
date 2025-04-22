import os
import torch
from diffusers import StableDiffusionPipeline
from diffusers.onnx_utils import export_to_onnx


def export_model(base_model: str, lora_adapter: str, onnx_dir: str, quantize: bool = False):
  # Load pipeline
  pipe = StableDiffusionPipeline.from_pretrained(base_model, torch_dtype=torch.float32)
  pipe.unet.load_attn_procs(lora_adapter)

  # Export UNet to ONNX
  os.makedirs(onnx_dir, exist_ok=True)
  export_to_onnx(
    pipe.unet,
    os.path.join(onnx_dir, 'unet.onnx'),
    device='cpu',
    use_gpu=False,
  )

  # Optional quantization
  if quantize:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(
      os.path.join(onnx_dir, 'unet.onnx'),
      os.path.join(onnx_dir, 'unet.int8.onnx'),
      weight_type=QuantType.QUInt8
    )

  print(f"ONNX models exported to {onnx_dir}")


def main():
  import argparse
  parser = argparse.ArgumentParser(description="Export LoRA‑augmented UNet to ONNX.")
  parser.add_argument('--base_model', type=str, required=True)
  parser.add_argument('--lora_adapter', type=str, required=True)
  parser.add_argument('--output_dir', type=str, default='onnx')
  parser.add_argument('--quantize', action='store_true')
  args = parser.parse_args()

  export_model(args.base_model, args.lora_adapter, args.output_dir, args.quantize)

if __name__ == '__main__':
  main()
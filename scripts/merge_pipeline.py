import os
import argparse
import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel


def merge_lora_pipeline(base_model: str, lora_adapter: str, output_dir: str, dtype: torch.dtype = torch.float32):
  """
  Merge a LoRA adapter with a Stable Diffusion pipeline.
  This script loads a base Stable Diffusion model and a LoRA adapter,
  merges them, and saves the resulting model to a specified directory.

  Args:
    base_model (str): Path to the base model or Hugging Face model ID
    lora_adapter (str): Path to the LoRA adapter weights
    output_dir (str): Directory to save the merged pipeline
    dtype (torch.dtype): Data type for model weights
  """
  print(f"Loading base model from: {base_model}")
  base_pipe = StableDiffusionPipeline.from_pretrained(
    base_model,
    torch_dtype=dtype
  )

  # First save the base pipeline to the output directory
  os.makedirs(output_dir, exist_ok=True)
  base_pipe.save_pretrained(output_dir)
  print(f"Saved base pipeline to: {output_dir}")

  # Then load it back (this avoids keeping two copies in memory)
  pipe = StableDiffusionPipeline.from_pretrained(
    output_dir,
    torch_dtype=dtype
  )

  print(f"Loading LoRA adapter from: {lora_adapter}")
  # Apply the LoRA adapter to the UNet
  pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_adapter)

  # If using "best_model" from our training script
  lora_adapter_name = os.path.basename(os.path.normpath(lora_adapter))
  print(f"Merging LoRA adapter: {lora_adapter_name}")

  # Save the merged UNet
  unet_dir = os.path.join(output_dir, "unet")
  os.makedirs(unet_dir, exist_ok=True)
  pipe.unet.save_pretrained(unet_dir)
  print(f"Saved merged UNet (base+adapter) to: {unet_dir}")

  return pipe


def main():
  parser = argparse.ArgumentParser(description="Merge LoRA adapter with base Stable Diffusion")
  parser.add_argument("--base_model", type=str, default="runwayml/stable-diffusion-v1-5",
                      help="Base model path or Hugging Face model ID")
  parser.add_argument("--lora_adapter", type=str, default="models/lora_adapters",
                      help="Path to the LoRA adapter weights")
  parser.add_argument("--output_dir", type=str, default="models/sd_lora_pipeline",
                      help="Directory to save the merged pipeline")
  parser.add_argument("--fp16", action="store_true",
                      help="Use float16 precision")
  args = parser.parse_args()

  # Select precision type
  dtype = torch.float16 if args.fp16 else torch.float32

  # Merge the pipeline
  merge_lora_pipeline(
    base_model=args.base_model,
    lora_adapter=args.lora_adapter,
    output_dir=args.output_dir,
    dtype=dtype
  )

  print(f"Pipeline successfully merged and saved to: {args.output_dir}")
  print("You can now use this pipeline for inference or export to ONNX.")


if __name__ == '__main__':
  main()

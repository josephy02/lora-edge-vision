import os
import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel


def main():
  """
  Merge a LoRA adapter with a Stable Diffusion pipeline.
  This script loads a base Stable Diffusion model and a LoRA adapter,
  merges them, and saves the resulting model to a specified directory.
  """
  base_model = "runwayml/stable-diffusion-v1-5"
  lora_adapter = "models/lora_adapters"
  output_dir = "models/sd_lora_pipeline"

  base_pipe = StableDiffusionPipeline.from_pretrained(
    base_model,
    torch_dtype=torch.float32
  )
  os.makedirs(output_dir, exist_ok=True)
  base_pipe.save_pretrained(output_dir)
  print(f"Saved base pipeline to: {output_dir}")

  pipe = StableDiffusionPipeline.from_pretrained(
    output_dir,
    torch_dtype=torch.float32
  )

  pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_adapter)

  unet_dir = os.path.join(output_dir, "unet")
  os.makedirs(unet_dir, exist_ok=True)
  pipe.unet.save_pretrained(unet_dir)
  print(f"Saved merged UNet (base+adapter) to: {unet_dir}")

if __name__ == '__main__':
  main()


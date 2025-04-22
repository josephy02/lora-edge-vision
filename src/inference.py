import os
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

def load_model(base_model: str, lora_adapter: str, device: str = 'cpu'):
  # Load base pipeline
  pipe = StableDiffusionPipeline.from_pretrained(base_model, torch_dtype=torch.float32)
  pipe = pipe.to(device)
  # Load LoRA adapters
  pipe.unet.load_attn_procs(lora_adapter)
  return pipe


def generate(pipe, prompt: str, num_inference_steps: int = 50, guidance_scale: float = 7.5):
  image = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
  return image


def main():
  import argparse
  parser = argparse.ArgumentParser(description="Generate an image with base+LoRA adapter.")
  parser.add_argument('--base_model', type=str, required=True)
  parser.add_argument('--lora_adapter', type=str, required=True)
  parser.add_argument('--prompt', type=str, required=True)
  parser.add_argument('--output_dir', type=str, default='outputs')
  parser.add_argument('--steps', type=int, default=50)
  parser.add_argument('--scale', type=float, default=7.5)
  args = parser.parse_args()

  os.makedirs(args.output_dir, exist_ok=True)
  device = 'mps' if torch.backends.mps.is_available() else 'cpu'

  pipe = load_model(args.base_model, args.lora_adapter, device)
  img = generate(pipe, args.prompt, num_inference_steps=args.steps, guidance_scale=args.scale)

  out_path = os.path.join(args.output_dir, 'generated.png')
  img.save(out_path)
  print(f"Saved generated image to {out_path}")

if __name__ == '__main__':
  main()
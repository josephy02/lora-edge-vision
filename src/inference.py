import os
import torch
from diffusers import StableDiffusionPipeline
# from diffusers import pipeline_utils
from peft import PeftModel
from PIL import Image


def load_model(base_model: str, lora_adapter: str, device: str = "cpu"):
  """
  Load the base Stable Diffusion model and wrap it with LoRA adapter weights.
  Args:
    base_model (str): Path to the base Stable Diffusion model.
    lora_adapter (str): Path to the LoRA adapter weights.
    device (str): Device to load the model on ("cpu" or "mps").
  Returns:
    StableDiffusionPipeline: The Stable Diffusion pipeline with LoRA adapter.
  """
  # Load base Stable Diffusion model
  pipe = StableDiffusionPipeline.from_pretrained(base_model, torch_dtype=torch.float32).to(device)
  pipe.enable_attention_slicing()
  # if device.type == "cuda":
  #   pipe.enable_model_cpu_offload()
  # Here we wrap UNet with your LoRA adapter weights,
  # assuming `lora_adapter` points at the folder that `save_pretrained` wrote.
  pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_adapter)
  return pipe


def generate(pipe, prompt: str, steps: int = 50, scale: float = 7.5):
  """
  Generate an image using the Stable Diffusion pipeline.
  Args:
    pipe (StableDiffusionPipeline): The Stable Diffusion pipeline with LoRA adapter.
    prompt (str): The text prompt to generate the image from.
    steps (int): Number of denoising steps.
    scale (float): Guidance scale for classifier-free guidance.
  Returns:
    PIL.Image: The generated image.
  """
  image = pipe(
    prompt,
    num_inference_steps=steps,
    guidance_scale=scale,
  ).images[0]
  return image


def main():
  import argparse
  parser = argparse.ArgumentParser(
      description="Generate images using base + LoRA adapter"
  )
  parser.add_argument("--base_model",   type=str, required=True)
  parser.add_argument("--lora_adapter", type=str, required=True,
                      help="Path to the folder where `save_pretrained` wrote your adapter")
  parser.add_argument("--prompt",       type=str, required=True)
  parser.add_argument("--output_dir",   type=str, default="outputs")
  parser.add_argument("--steps",        type=int, default=50)
  parser.add_argument("--scale",        type=float, default=7.5)
  args = parser.parse_args()

  os.makedirs(args.output_dir, exist_ok=True)
  device = "mps" if torch.backends.mps.is_available() else "cpu"

  pipe = load_model(args.base_model, args.lora_adapter, device)
  img  = generate(pipe, args.prompt, steps=args.steps, scale=args.scale)

  out_path = os.path.join(args.output_dir, "generated.png")
  img.save(out_path)
  print(f"Saved generated image to {out_path}")

if __name__ == "__main__":
  main()
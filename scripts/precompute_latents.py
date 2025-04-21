import os
import torch
from tqdm import tqdm
from diffusers import StableDiffusionPipeline
from PIL import Image

# Precompute VAE latents for all processed images

def main():
  model_id = "runwayml/stable-diffusion-v1-5"
  data_dir = "data/processed_images"
  out_dir = "data/latents"
  os.makedirs(out_dir, exist_ok=True)

  # Load pipeline (only VAE & feature extractor needed)
  pipe = StableDiffusionPipeline.from_pretrained(model_id)
  pipe = pipe.to("cpu")  # or "mps" if desired
  pipe.set_progress_bar_config(disable=True)

  for fname in tqdm(os.listdir(data_dir), desc="Precomputing latents"):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
      continue
    image_path = os.path.join(data_dir, fname)
    latent_path = os.path.join(out_dir, os.path.splitext(fname)[0] + ".pt")

    # Load the image with PIL
    image = Image.open(image_path).convert("RGB")

    # Extract pixel values via feature extractor
    inputs = pipe.feature_extractor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(pipe.device)

    # Encode to latents
    with torch.no_grad():
      latents = pipe.vae.encode(pixel_values).latent_dist.sample()
      latents = latents * pipe.vae.config.scaling_factor

    # Save latents to disk
    torch.save(latents.cpu(), latent_path)

  print(f"Saved latents for all images to {out_dir}")


if __name__ == '__main__':
  main()
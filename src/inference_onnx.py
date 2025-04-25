import os
import argparse
import numpy as np
import torch
import onnxruntime as ort
from diffusers import StableDiffusionPipeline, DDIMScheduler
from PIL import Image
from tqdm.auto import tqdm


def load_pipeline_components(onnx_dir):
  """
  Load required pipeline components for inference.

  Args:
    onnx_dir (str): Directory containing ONNX models

  Returns:
    tuple: (text_encoder_session, unet_session, vae_decoder, tokenizer, scheduler)
  """
  print(f"Loading pipeline components from {onnx_dir}...")

  # Set up ONNX Runtime session options
  sess_options = ort.SessionOptions()
  sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

  # Load ONNX UNet - the main exported model
  unet_path = os.path.join(onnx_dir, "unet.onnx")
  if not os.path.exists(unet_path):
    raise FileNotFoundError(f"UNet model not found at {unet_path}")

  unet_session = ort.InferenceSession(unet_path, sess_options)

  # Since we only exported the UNet to ONNX, we need to load other components from the original model
  # We'll use the base SD model for this
  base_model = "runwayml/stable-diffusion-v1-5"
  print(f"Loading other components from {base_model}...")

  sd_pipe = StableDiffusionPipeline.from_pretrained(
    base_model,
    torch_dtype=torch.float32
  )

  # Extract components we need
  tokenizer = sd_pipe.tokenizer
  text_encoder = sd_pipe.text_encoder
  vae_decoder = sd_pipe.vae
  scheduler = DDIMScheduler.from_pretrained(base_model, subfolder="scheduler")

  # Get text encoder session from the pipeline or use PyTorch
  # For simplicity in this example, we'll use the PyTorch text encoder

  return text_encoder, unet_session, vae_decoder, tokenizer, scheduler


def generate_image(
  prompt,
  text_encoder,
  unet_session,
  vae_decoder,
  tokenizer,
  scheduler,
  height=512,
  width=512,
  num_inference_steps=50,
  guidance_scale=7.5,
  negative_prompt="",
  seed=None
):
  """
  Generate an image using the ONNX UNet model.

  Args:
    prompt (str): Text prompt for image generation
    text_encoder: Text encoder model (PyTorch)
    unet_session: UNet ONNX session
    vae_decoder: VAE decoder model (PyTorch)
    tokenizer: Tokenizer for text processing
    scheduler: Diffusion scheduler
    height (int): Image height
    width (int): Image width
    num_inference_steps (int): Number of denoising steps
    guidance_scale (float): Guidance scale for classifier-free guidance
    negative_prompt (str): Negative prompt for classifier-free guidance
    seed (int, optional): Random seed for reproducibility

  Returns:
    PIL.Image: Generated image
  """
  # Set random seed for reproducibility
  if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)

  # Get UNet input and output names
  unet_inputs = unet_session.get_inputs()
  unet_output = unet_session.get_outputs()[0].name

  # Get input names for convenience
  input_names = {input.name: i for i, input in enumerate(unet_inputs)}

  # Check latent dimensions from UNet input
  latent_channels = unet_inputs[input_names.get("sample", 0)].shape[1]
  latent_height = unet_inputs[input_names.get("sample", 0)].shape[2]
  latent_width = unet_inputs[input_names.get("sample", 0)].shape[3]

  # Text encoding
  text_input = tokenizer(
    [prompt],
    padding="max_length",
    max_length=tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt",
  )

  with torch.no_grad():
    text_embeddings = text_encoder(text_input.input_ids)[0].numpy()

  # Negative prompt for classifier-free guidance
  if guidance_scale > 1.0:
    uncond_input = tokenizer(
      [negative_prompt],
      padding="max_length",
      max_length=tokenizer.model_max_length,
      truncation=True,
      return_tensors="pt",
    )

    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids)[0].numpy()

    # Concatenate conditional and unconditional embeddings
    text_embeddings = np.concatenate([uncond_embeddings, text_embeddings])

  # Set up scheduler
  scheduler.set_timesteps(num_inference_steps)

  # Create random latents
  latents = np.random.randn(1, latent_channels, latent_height, latent_width).astype(np.float32)

  # Scale the initial noise by the standard deviation required by the scheduler
  latents = latents * scheduler.init_noise_sigma

  # Denoising loop
  for i, t in enumerate(tqdm(scheduler.timesteps, desc="Denoising")):
    # Expand the latents for classifier-free guidance
    latent_model_input = np.repeat(latents, 2, axis=0) if guidance_scale > 1.0 else latents

    # Convert timestep to what the model expects
    timestep = np.array([t]).astype(np.int64)

    # Prepare inputs for the UNet model
    onnx_inputs = {}
    for input_info in unet_inputs:
        if "sample" in input_info.name:
            onnx_inputs[input_info.name] = latent_model_input
        elif "timestep" in input_info.name:
            onnx_inputs[input_info.name] = timestep
        elif "encoder_hidden_states" in input_info.name:
            onnx_inputs[input_info.name] = text_embeddings

    # Run UNet inference
    noise_pred = unet_session.run([unet_output], onnx_inputs)[0]

    # Perform guidance
    if guidance_scale > 1.0:
      noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2)
      noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    # Compute the previous noisy sample
    latents = scheduler.step(
      torch.from_numpy(noise_pred),
      t,
      torch.from_numpy(latents)
    ).prev_sample.numpy()

  # Scale and decode the latents with the VAE
  latents = 1 / 0.18215 * latents

  # Convert back to PyTorch tensor for VAE decoding
  with torch.no_grad():
    latents_torch = torch.from_numpy(latents)
    image = vae_decoder.decode(latents_torch).sample

  # Convert to image
  image = (image / 2 + 0.5).clamp(0, 1)
  image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
  image = (image * 255).round().astype("uint8")[0]

  return Image.fromarray(image)


def main():
  parser = argparse.ArgumentParser(description="Run inference with ONNX Stable Diffusion")
  parser.add_argument("--onnx_dir", type=str, required=True, help="Directory containing ONNX models")
  parser.add_argument("--prompt", type=str, required=True, help="Text prompt for generation")
  parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt")
  parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
  parser.add_argument("--steps", type=int, default=30, help="Number of inference steps")
  parser.add_argument("--scale", type=float, default=7.5, help="Guidance scale")
  parser.add_argument("--height", type=int, default=512, help="Image height")
  parser.add_argument("--width", type=int, default=512, help="Image width")
  parser.add_argument("--seed", type=int, default=None, help="Random seed")

  args = parser.parse_args()

  # Create output directory
  os.makedirs(args.output_dir, exist_ok=True)

  # Load pipeline components
  text_encoder, unet_session, vae_decoder, tokenizer, scheduler = load_pipeline_components(args.onnx_dir)

  # Generate image
  print(f"Generating image with prompt: {args.prompt}")
  image = generate_image(
    args.prompt,
    text_encoder,
    unet_session,
    vae_decoder,
    tokenizer,
    scheduler,
    height=args.height,
    width=args.width,
    num_inference_steps=args.steps,
    guidance_scale=args.scale,
    negative_prompt=args.negative_prompt,
    seed=args.seed
  )

  # Save image
  output_name = f"onnx_generated_{args.seed}.png" if args.seed else "onnx_generated.png"
  output_path = os.path.join(args.output_dir, output_name)
  image.save(output_path)
  print(f"Image saved to {output_path}")

if __name__ == "__main__":
  main()
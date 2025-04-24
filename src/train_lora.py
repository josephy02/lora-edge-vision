# import os
# import yaml
# import torch
# from torch.utils.data import Dataset, DataLoader
# from accelerate import Accelerator
# from peft import LoraConfig, get_peft_model
# from diffusers import StableDiffusionPipeline
# from tqdm import tqdm


# class LatentDataset(Dataset):
#   """Loads precomputed VAE latents (.pt files) and a prompt."""
#   def __init__(self, latents_dir, prompt="Aerial view photo"):
#     self.latent_paths = sorted(
#       f for f in os.listdir(latents_dir) if f.endswith(".pt")
#     )
#     self.latents_dir = latents_dir
#     self.prompt = prompt

#   def __len__(self):
#     return len(self.latent_paths)

#   def __getitem__(self, idx):
#     path = os.path.join(self.latents_dir, self.latent_paths[idx])
#     latents = torch.load(path)       # [1, C, H, W] is the shape of the latent
#     return {
#       "latents": latents.squeeze(0),  # [C, H, W] is the normalized latent
#       "prompt": self.prompt
#     }


# def load_cfg(path="configs/train.yaml"):
#   '''Loads training config from YAML file.'''
#   if not os.path.exists(path):
#     raise FileNotFoundError(f"Config file {path} not found.")
#   with open(path) as f:
#     return yaml.safe_load(f)


# def main():
#   '''Main function to train LoRA adapters for Stable Diffusion.'''
#   cfg = load_cfg()
#   # Initialize config & parameters
#   lr = float(cfg["learning_rate"])
#   bs = int(cfg["batch_size"])
#   epochs = int(cfg["num_epochs"])
#   rank = int(cfg["lora_rank"])
#   alpha = int(cfg["lora_alpha"])
#   accum = int(cfg.get("gradient_accumulation_steps", 1))

#   # Initialize accelerator
#   # Check if MPS is available and set device accordingly
#   device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
#   mp = cfg.get("mixed_precision") if device.type != "mps" else None
#   accel  = Accelerator(cpu=False, mixed_precision=mp)

#   # Load dataset & dataloader
#   dataset = LatentDataset(cfg["latents_dir"])
#   loader = DataLoader(dataset, batch_size=bs, shuffle=True,
#                         num_workers=4, pin_memory=True)

#   # Load LoRA‑wrapped model
#   dtype = torch.float16 if mp == "fp16" else torch.float32
#   pipe  = StableDiffusionPipeline.from_pretrained(cfg["base_model_path"], torch_dtype=dtype).to(device)
#   lora_cfg = LoraConfig(r=rank, lora_alpha=alpha, target_modules=["to_q","to_k","to_v"])
#   pipe.unet = get_peft_model(pipe.unet, lora_cfg)
#   pipe.unet.enable_gradient_checkpointing()
#   optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=lr)

#   # Prepare for accel
#   pipe.unet, optimizer, loader = accel.prepare(pipe.unet, optimizer, loader)

#   # This is the training loop in which we train the LoRA adapters
#   # We use gradient checkpointing to save memory
#   # and gradient accumulation to reduce the number of steps
#   for ep in range(epochs):
#     optimizer.zero_grad()
#     pbar = tqdm(loader, desc=f"Epoch {ep+1}/{epochs}")
#     for step, batch in enumerate(pbar):
#       latents = batch["latents"].to(device)
#       toks = pipe.tokenizer(batch["prompt"], padding="max_length",
#                             truncation=True, return_tensors="pt").to(device)
#       h = pipe.text_encoder(**toks).last_hidden_state

#       out = pipe.unet(latents, timestep=1000, encoder_hidden_states=h).sample
#       loss = torch.nn.functional.mse_loss(out, latents) / accum
#       accel.backward(loss)

#       if (step + 1) % accum == 0 or (step + 1) == len(loader):
#         optimizer.step()
#         optimizer.zero_grad()

#       pbar.set_postfix(loss=(loss.item() * accum))

#   # Here we save the LoRA adapters to the output directory
#   # and the model weights to the base model path
#   os.makedirs(cfg["output_dir"], exist_ok=True)
#   pipe.unet.save_pretrained(cfg["output_dir"])
#   print(f"Saved LoRA adapters to {cfg['output_dir']}")


# if __name__ == "__main__":
#   main()

import os
import yaml
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model
from tqdm import tqdm


class LatentDataset(Dataset):
  """Loads precomputed VAE latents (.pt files) and a prompt."""
  def __init__(self, latents_dir, prompt="Aerial view photo"):
    self.latent_paths = sorted(
      f for f in os.listdir(latents_dir) if f.endswith(".pt")
    )
    self.latents_dir = latents_dir
    self.prompt = prompt

  def __len__(self):
    return len(self.latent_paths)

  def __getitem__(self, idx):
    path = os.path.join(self.latents_dir, self.latent_paths[idx])
    latents = torch.load(path)       # [1, C, H, W] is the shape of the latent
    return {
      "latents": latents.squeeze(0),  # [C, H, W] is the normalized latent
      "prompt": self.prompt
    }


def load_cfg(path="configs/train.yaml"):
  '''Loads training config from YAML file.'''
  if not os.path.exists(path):
    raise FileNotFoundError(f"Config file {path} not found.")
  with open(path) as f:
    return yaml.safe_load(f)


def main():
  '''Main function to train LoRA adapters for Stable Diffusion.'''
  parser = argparse.ArgumentParser(description="Train LoRA adapters for Stable Diffusion")
  parser.add_argument("--config", type=str, default="configs/train.yaml",
                      help="Path to the configuration file")
  args = parser.parse_args()

  cfg = load_cfg(args.config)

  # Initialize config & parameters
  lr = float(cfg["learning_rate"])
  bs = int(cfg["batch_size"])
  epochs = int(cfg["num_epochs"])
  rank = int(cfg["lora_rank"])
  alpha = int(cfg["lora_alpha"])
  accum = int(cfg.get("gradient_accumulation_steps", 1))
  base_model_path = cfg["base_model_path"]

  # Initialize accelerator
  # Check if MPS is available and set device accordingly
  device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
  mp = cfg.get("mixed_precision") if device.type != "mps" else None
  accel = Accelerator(cpu=False, mixed_precision=mp)

  # Load dataset & dataloader
  dataset = LatentDataset(cfg["latents_dir"])
  loader = DataLoader(dataset, batch_size=bs, shuffle=True,
                      num_workers=4, pin_memory=True)

  # Load only necessary components (memory optimization)
  dtype = torch.float16 if mp == "fp16" else torch.float32

  # First load the full pipeline to get all components

  print(f"Loading base model: {base_model_path}")
  pipe = StableDiffusionPipeline.from_pretrained(
    base_model_path,
    torch_dtype=dtype
  )

  # Extract the components we need
  text_encoder = pipe.text_encoder
  text_encoder.to(device)
  text_encoder.requires_grad_(False)  # Freeze the text encoder

  # Get tokenizer
  tokenizer = pipe.tokenizer

  # Get UNet and apply LoRA
  unet = pipe.unet
  unet.to(device)

  # Free up memory by deleting the pipeline
  del pipe
  if torch.cuda.is_available():
    torch.cuda.empty_cache()

  print("Model components loaded successfully")

  # Configure LoRA
  lora_cfg = LoraConfig(
    r=rank,
    lora_alpha=alpha,
    target_modules=["to_q", "to_k", "to_v"],
    lora_dropout=0.1,  # Optional: Add dropout for regularization
  )
  unet = get_peft_model(unet, lora_cfg)
  unet.enable_gradient_checkpointing()

  # Set up optimizer
  optimizer = torch.optim.AdamW(unet.parameters(), lr=lr)

  # Set up optional learning rate scheduler
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=epochs * len(loader) // accum
  )

  # Prepare for accelerator
  unet, optimizer, loader = accel.prepare(unet, optimizer, loader)

  # Track best loss for model saving
  best_loss = float('inf')

  # Training loop
  for ep in range(epochs):
    epoch_losses = []
    optimizer.zero_grad()
    pbar = tqdm(loader, desc=f"Epoch {ep+1}/{epochs}")

    for step, batch in enumerate(pbar):
      latents = batch["latents"].to(device)

      # Encode text prompts
      text_input = tokenizer(
        batch["prompt"],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
      ).to(device)

      # Generate text embeddings
      with torch.no_grad():
        encoder_hidden_states = text_encoder(text_input.input_ids)[0]

      # Forward pass through UNet
      model_output = unet(
        latents,
        timestep=torch.tensor([1000], device=device),
        encoder_hidden_states=encoder_hidden_states
      ).sample

      # Calculate loss
      loss = torch.nn.functional.mse_loss(model_output, latents) / accum
      accel.backward(loss)
      epoch_losses.append(loss.item() * accum)

      # Update weights with gradient accumulation
      if (step + 1) % accum == 0 or (step + 1) == len(loader):
        optimizer.step()
        scheduler.step()  # Update learning rate
        optimizer.zero_grad()

      pbar.set_postfix(loss=loss.item() * accum, lr=scheduler.get_last_lr()[0])

    # Calculate average loss for the epoch
    avg_loss = sum(epoch_losses) / len(epoch_losses)
    print(f"Epoch {ep+1}/{epochs} - Average Loss: {avg_loss:.6f}")

    # Save checkpoint at end of epoch
    checkpoint_path = os.path.join(cfg["output_dir"], f"checkpoint-{ep+1}")
    os.makedirs(checkpoint_path, exist_ok=True)
    unet.save_pretrained(checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

    # Save best model based on loss
    if avg_loss < best_loss:
      best_loss = avg_loss
      best_path = os.path.join(cfg["output_dir"], "best_model")
      os.makedirs(best_path, exist_ok=True)
      unet.save_pretrained(best_path)
      print(f"New best model saved with loss: {best_loss:.6f}")

  # Save final model
  os.makedirs(cfg["output_dir"], exist_ok=True)
  unet.save_pretrained(cfg["output_dir"])
  print(f"Saved final LoRA adapters to {cfg['output_dir']}")


if __name__ == "__main__":
  main()


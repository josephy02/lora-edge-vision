import os
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from diffusers import StableDiffusionPipeline
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
  cfg = load_cfg()
  # Initialize config & parameters
  lr = float(cfg["learning_rate"])
  bs = int(cfg["batch_size"])
  epochs = int(cfg["num_epochs"])
  rank = int(cfg["lora_rank"])
  alpha = int(cfg["lora_alpha"])
  accum = int(cfg.get("gradient_accumulation_steps", 1))

  # Initialize accelerator
  # Check if MPS is available and set device accordingly
  device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
  mp = cfg.get("mixed_precision") if device.type != "mps" else None
  accel  = Accelerator(cpu=False, mixed_precision=mp)

  # Load dataset & dataloader
  dataset = LatentDataset(cfg["latents_dir"])
  loader = DataLoader(dataset, batch_size=bs, shuffle=True,
                        num_workers=4, pin_memory=True)

  # Load LoRA‑wrapped model
  dtype = torch.float16 if mp == "fp16" else torch.float32
  pipe  = StableDiffusionPipeline.from_pretrained(cfg["base_model_path"], torch_dtype=dtype).to(device)
  lora_cfg = LoraConfig(r=rank, lora_alpha=alpha, target_modules=["to_q","to_k","to_v"])
  pipe.unet = get_peft_model(pipe.unet, lora_cfg)
  pipe.unet.enable_gradient_checkpointing()
  optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=lr)

  # Prepare for accel
  pipe.unet, optimizer, loader = accel.prepare(pipe.unet, optimizer, loader)

  # This is the training loop in which we train the LoRA adapters
  # We use gradient checkpointing to save memory
  # and gradient accumulation to reduce the number of steps
  for ep in range(epochs):
    optimizer.zero_grad()
    pbar = tqdm(loader, desc=f"Epoch {ep+1}/{epochs}")
    for step, batch in enumerate(pbar):
      latents = batch["latents"].to(device)
      toks = pipe.tokenizer(batch["prompt"], padding="max_length",
                            truncation=True, return_tensors="pt").to(device)
      h = pipe.text_encoder(**toks).last_hidden_state

      out = pipe.unet(latents, timestep=1000, encoder_hidden_states=h).sample
      loss = torch.nn.functional.mse_loss(out, latents) / accum
      accel.backward(loss)

      if (step + 1) % accum == 0 or (step + 1) == len(loader):
        optimizer.step()
        optimizer.zero_grad()

      pbar.set_postfix(loss=(loss.item() * accum))

  # Here we save the LoRA adapters to the output directory
  # and the model weights to the base model path
  os.makedirs(cfg["output_dir"], exist_ok=True)
  pipe.unet.save_pretrained(cfg["output_dir"])
  print(f"Saved LoRA adapters to {cfg['output_dir']}")

if __name__ == "__main__":
  main()

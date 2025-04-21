import os
import yaml
import torch
import warnings
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from diffusers import StableDiffusionPipeline
from tqdm import tqdm

# Suppress repetitive torchvision JPEG warnings
warnings.filterwarnings(
  "ignore",
  ".*Failed to load image Python extension.*",
  category=UserWarning
)


class ImagePromptDataset(Dataset):
  """Loads images and a placeholder prompt."""
  def __init__(self, image_dir, resolution):
    valid_exts = ('.png', '.jpg', '.jpeg')
    self.image_paths = [os.path.join(image_dir, f)
                        for f in os.listdir(image_dir)
                        if f.lower().endswith(valid_exts)]
    self.transform = transforms.Compose([
      transforms.Resize((resolution, resolution)),
      transforms.ToTensor(),
      transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

  def __len__(self):
    return len(self.image_paths)

  def __getitem__(self, idx):
    img_path = self.image_paths[idx]
    image = Image.open(img_path).convert('RGB')
    pixel_values = self.transform(image)
    prompt = "Aerial view photo"
    return {"pixel_values": pixel_values, "prompt": prompt}


class LatentPromptDataset(Dataset):
  """Loads precomputed latents and a placeholder prompt."""
  def __init__(self, latents_dir):
    self.latent_paths = [os.path.join(latents_dir, f)
                          for f in os.listdir(latents_dir)
                          if f.endswith('.pt')]

  def __len__(self):
    return len(self.latent_paths)

  def __getitem__(self, idx):
    latents = torch.load(self.latent_paths[idx])  # [1, C, H, W]
    latents = latents.squeeze(0)                  # [C, H, W]
    prompt = "Aerial view photo"
    return {"latents": latents, "prompt": prompt}


def load_config(path):
  with open(path, 'r') as f:
    return yaml.safe_load(f)


def main():
  # Load config
  cfg = load_config('configs/train.yaml')

  # Cast and fetch parameters
  lr = float(cfg['learning_rate'])
  bs = int(cfg['batch_size'])
  epochs = int(cfg['num_epochs'])
  rank = int(cfg['lora_rank'])
  alpha = int(cfg['lora_alpha'])
  res = int(cfg.get('resolution', 256))
  accum = int(cfg.get('gradient_accumulation_steps', 1))

  # Device and precision
  device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
  mp = cfg.get('mixed_precision', None)
  if device.type == 'mps':
    mp = None
  accelerator = Accelerator(cpu=False, mixed_precision=mp)

  # Dataset selection
  use_latents = False
  if cfg.get('latents_dir') and os.path.isdir(cfg['latents_dir']):
    # Check for .pt files
    files = [f for f in os.listdir(cfg['latents_dir']) if f.endswith('.pt')]
    if files:
      dataset = LatentPromptDataset(cfg['latents_dir'])
      use_latents = True
      print(f"⚡️ Using {len(files)} precomputed latents from {cfg['latents_dir']}")
    else:
      raise ValueError(f"No .pt files found in latents_dir: {cfg['latents_dir']}")
  if not use_latents:
    if not os.path.isdir(cfg['dataset_dir']):
      raise ValueError(f"dataset_dir not found: {cfg['dataset_dir']}")
    dataset = ImagePromptDataset(cfg['dataset_dir'], resolution=res)
    print(f"🔄 Using {len(dataset)} images from {cfg['dataset_dir']}")

  loader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)

  # Load model
  dtype = torch.float16 if mp == 'fp16' and device.type != 'mps' else torch.float32
  pipe = StableDiffusionPipeline.from_pretrained(cfg['base_model_path'], torch_dtype=dtype)
  pipe = pipe.to(device)

  # Apply LoRA
  lora_cfg = LoraConfig(r=rank, lora_alpha=alpha, target_modules=["to_q","to_k","to_v"])
  unet = get_peft_model(pipe.unet, lora_cfg)
  unet.enable_gradient_checkpointing()
  pipe.unet = unet
  optimizer = torch.optim.AdamW(unet.parameters(), lr=lr)

  # Prepare with accelerator
  pipe.unet, optimizer, loader = accelerator.prepare(pipe.unet, optimizer, loader)

  # Training loop
  for ep in range(epochs):
    optimizer.zero_grad()
    pbar = tqdm(loader, desc=f"Epoch {ep+1}/{epochs}")
    for step, batch in enumerate(pbar):
      # Load latents or encode images
      if use_latents:
        latents = batch['latents'].to(device)
      else:
        px = batch['pixel_values'].to(device)
        latents = pipe.vae.encode(px).latent_dist.sample() * pipe.vae.config.scaling_factor

      # Text encode
      toks = pipe.tokenizer(batch['prompt'], padding='max_length',
                            truncation=True, return_tensors='pt').to(device)
      h = pipe.text_encoder(**toks).last_hidden_state

      # UNet forward and loss
      out = pipe.unet(latents, timestep=1000, encoder_hidden_states=h).sample
      loss = torch.nn.functional.mse_loss(out, latents) / accum
      accelerator.backward(loss)

      # Step and zero_grad
      if (step + 1) % accum == 0 or (step + 1) == len(loader):
        optimizer.step()
        optimizer.zero_grad()

      pbar.set_postfix({'loss': loss.item() * accum})

  # Save adapters
  os.makedirs(cfg['output_dir'], exist_ok=True)
  unet.save_pretrained(cfg['output_dir'])
  print(f"LoRA adapters saved to {cfg['output_dir']}")


if __name__ == '__main__':
  main()
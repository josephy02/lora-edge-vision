import os
import torch
import argparse
from diffusers import StableDiffusionPipeline
from peft import PeftModel

def export_onnx_for_mac(base_model, lora_adapter, onnx_dir, opset_version=13):
  """
  Export the UNet model to ONNX format with settings optimized for macOS compatibility.

  Args:
    base_model (str): Path to the base model
    lora_adapter (str): Path to LoRA adapter
    onnx_dir (str): Directory to save ONNX files
    opset_version (int): ONNX opset version (13 works better on macOS)
  """
  os.makedirs(onnx_dir, exist_ok=True)

  print(f"Loading pipeline from {base_model}...")
  pipe = StableDiffusionPipeline.from_pretrained(base_model, torch_dtype=torch.float32)

  print(f"Applying LoRA adapter from {lora_adapter}...")
  pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_adapter)

  # Move to CPU and set to eval mode
  unet = pipe.unet.to("cpu").eval()

  # Set up dummy inputs - simplified for better compatibility
  # Standard dimensions for stable diffusion
  batch_size = 1
  channels = 4
  height = width = 64  # standard latent size for 512x512 images

  # Create sample inputs
  sample = torch.randn(batch_size, channels, height, width, dtype=torch.float32)
  timestep = torch.tensor([1000], dtype=torch.int64)
  encoder_hidden_states = torch.randn(batch_size, 77, 768, dtype=torch.float32)

  # Path for the exported model
  onnx_model_path = os.path.join(onnx_dir, "unet.onnx")

  print(f"Exporting UNet to ONNX (opset version {opset_version})...")
  # Export with carefully selected settings for macOS compatibility
  torch.onnx.export(
    unet,
    args=(sample, timestep, encoder_hidden_states),
    f=onnx_model_path,
    input_names=["sample", "timestep", "encoder_hidden_states"],
    output_names=["output"],
    dynamic_axes={
        "sample": {0: "batch_size"},
        "encoder_hidden_states": {0: "batch_size"},
    },
    opset_version=opset_version,
    do_constant_folding=True,
    # Lower level settings for better compatibility
    operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
    export_params=True,
    keep_initializers_as_inputs=False,
    verbose=False
  )

  print(f"UNet successfully exported to {onnx_model_path}")
  return onnx_model_path

def main():
  parser = argparse.ArgumentParser(description="Export LoRA-adapted UNet to ONNX (macOS optimized)")
  parser.add_argument("--base_model", required=True, help="Path to base model")
  parser.add_argument("--lora_adapter", required=True, help="Path to LoRA adapter")
  parser.add_argument("--output_dir", default="onnx_mac", help="Directory to save ONNX files")
  parser.add_argument("--opset", type=int, default=13, help="ONNX opset version (13 recommended for macOS)")

  args = parser.parse_args()

  export_onnx_for_mac(
    args.base_model,
    args.lora_adapter,
    args.output_dir,
    args.opset
  )

if __name__ == "__main__":
  main()
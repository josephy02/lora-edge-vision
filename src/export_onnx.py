import os
import torch
import yaml
import argparse
from diffusers import StableDiffusionPipeline
from onnxruntime.quantization import quantize_dynamic, QuantType
from peft import PeftModel


def load_cfg(path="configs/train.yaml"):
  """Loads training config from YAML file."""
  with open(path, "r") as f:
    return yaml.safe_load(f)


def export_onnx(base_model: str, lora_adapter: str, onnx_dir: str, quantize: bool = False):
  """
  Export the LoRA-augmented Stable Diffusion model to ONNX format.
  Args:
    base_model (str): Path to the base Stable Diffusion model.
    lora_adapter (str): Path to the LoRA adapter weights.
    onnx_dir (str): Directory to save the exported ONNX model.
    quantize (bool): Whether to quantize the model to INT8 format.
  """
  # once again we load the config
  cfg = load_cfg()
  os.makedirs(onnx_dir, exist_ok=True)

  # load pipeline and wrap UNet with LoRA adapter
  pipe = StableDiffusionPipeline.from_pretrained(base_model, torch_dtype=torch.float32)
  pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_adapter)
  unet = pipe.unet.to("cpu").eval()

  # Here we set up dummy inputs for the export, this is the same as in the training script
  C = unet.config.in_channels
  H = W = unet.config.sample_size  # typically 64 for 512×512 default SD
  dummy_latent = torch.randn(1, C, H, W, dtype=torch.float32)
  dummy_timestep = torch.tensor(1, dtype=torch.int64)
  seq_len = pipe.text_encoder.config.max_position_embeddings
  hidden_size = pipe.text_encoder.config.hidden_size
  dummy_hs = torch.randn(1, seq_len, hidden_size, dtype=torch.float32)

  # we then export the model to ONNX
  # Note: The input names and output names are important for ONNX export
  # and should match the expected input/output names in your inference code
  onnx_path = os.path.join(onnx_dir, "unet.onnx")
  torch.onnx.export(
    unet,
    (dummy_latent, dummy_timestep, dummy_hs),
    onnx_path,
    input_names=["sample", "timestep", "encoder_hidden_states"],
    output_names=["out_sample"],
    opset_version=14,
    dynamic_axes={
      "sample": {0: "batch"},
      "encoder_hidden_states": {0: "batch"},
    },
  )
  print(f"UNet exported to {onnx_path}")

  # this step is optional, but it can help reduce the model size and improve inference speed
  # by quantizing the model to INT8 format. Note: This step requires the onnxruntime and onnxruntime-tools packages
  if quantize:
    quant_path = os.path.join(onnx_dir, "unet.int8.onnx")
    quantize_dynamic(onnx_path, quant_path, weight_type=QuantType.QInt8)
    print(f"Quantized model saved to {quant_path}")


def main():
  '''Main function to export LoRA-augmented UNet to ONNX format.'''
  parser = argparse.ArgumentParser("Export LoRA-augmented UNet to ONNX")
  parser.add_argument("--base_model",   type=str, required=True,
                      help="Hugging Face model ID or local base folder")
  parser.add_argument("--lora_adapter", type=str, required=True,
                      help="Folder containing your saved LoRA adapter")
  parser.add_argument("--output_dir",   type=str, default="onnx")
  parser.add_argument("--quantize",     action="store_true",
                      help="Also produce an INT8-quantized ONNX")
  args = parser.parse_args()

  export_onnx(args.base_model, args.lora_adapter, args.output_dir, args.quantize)

if __name__ == "__main__":
  main()

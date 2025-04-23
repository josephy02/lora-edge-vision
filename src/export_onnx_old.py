# IGNORED


# import os
# import torch
# import argparse
# from diffusers import StableDiffusionPipeline
# from diffusers.onnx_utils import export_to_onnx - THIS WAS DEPRECATED


# def export_model(base_model: str, lora_adapter: str, onnx_dir: str, quantize: bool = False):
#   """
#   Export the LoRA-augmented UNet to ONNX format.
#   Args:
#     base_model (str): Path to the base Stable Diffusion model.
#     lora_adapter (str): Path to the LoRA adapter weights.
#     onnx_dir (str): Directory to save the exported ONNX model.
#     quantize (bool): Whether to quantize the model to int8.
#   """
#   # Load the pipeline and wrap UNet with LoRA adapter weights
#   pipe = StableDiffusionPipeline.from_pretrained(base_model, torch_dtype=torch.float32)
#   pipe.unet.load_attn_procs(lora_adapter)

#   # Here we export UNet to ONNX
#   os.makedirs(onnx_dir, exist_ok=True)
#   export_to_onnx(
#     pipe.unet,
#     os.path.join(onnx_dir, 'unet.onnx'),
#     device='cpu',
#     use_gpu=False,
#   )

#   # This step is optional, but it can help reduce the model size
#   if quantize:
#     from onnxruntime.quantization import quantize_dynamic, QuantType
#     quantize_dynamic(
#       os.path.join(onnx_dir, 'unet.onnx'),
#       os.path.join(onnx_dir, 'unet.int8.onnx'),
#       weight_type=QuantType.QUInt8
#     )

#   print(f"ONNX models exported to {onnx_dir}")


# def main():
#   """
#   Main function to export LoRA-augmented UNet to ONNX format.
#   """
#   parser = argparse.ArgumentParser(description="Export LoRA‑augmented UNet to ONNX.")
#   parser.add_argument('--base_model', type=str, required=True)
#   parser.add_argument('--lora_adapter', type=str, required=True)
#   parser.add_argument('--output_dir', type=str, default='onnx')
#   parser.add_argument('--quantize', action='store_true')
#   args = parser.parse_args()

#   export_model(args.base_model, args.lora_adapter, args.output_dir, args.quantize)

# if __name__ == '__main__':
#   main()


# import os
# import torch
# import argparse
# import yaml
# from optimum.onnxruntime import ORTStableDiffusionPipeline -- DOESNT WORK OFFLINE


# def load_cfg(path="configs/train.yaml"):
#   """
#   Loads training config from YAML file.
#   Args:
#     path (str): Path to the YAML config file.
#   Returns:
#     dict: Loaded configuration as a dictionary.
#   Raises:
#     FileNotFoundError: If the config file does not exist.
#   """
#   with open(path, "r") as f:
#     return yaml.safe_load(f)


# def export_with_optimum(model_dir, onnx_dir):
#   """
#   Export the LoRA-augmented Stable Diffusion model to ONNX format using Optimum.
#   Args:
#     model_dir (str): Path to the base Stable Diffusion model with LoRA adapter.
#     onnx_dir (str): Directory to save the exported ONNX model.
#   """
#   # Load the pipeline and wrap UNet with LoRA adapter weights
#   pipe = ORTStableDiffusionPipeline.from_pretrained(
#     model_dir,  # can be HF repo id or local path containing base + LoRA
#     export=True,              # triggers ONNX export
#     library="diffusers",
#     trust_remote_code=True    # if you have custom modules
#   )
#   os.makedirs(onnx_dir, exist_ok=True)
#   pipe.save_pretrained(onnx_dir)
#   print(f"Exported ONNX pipeline to {onnx_dir}")


# def main():
#   """
#   Main function to export LoRA-augmented Stable Diffusion model to ONNX format.
#   """
#   parser = argparse.ArgumentParser("Export SD+LoRA to ONNX via Optimum")
#   parser.add_argument("--model_dir",    required=True,
#                       help="Path or repo ID for base+adapter")
#   parser.add_argument("--output_dir",   default="onnx",
#                       help="Where to write the ONNX pipeline")
#   args = parser.parse_args()


#   export_with_optimum(args.model_dir, args.output_dir)


# if __name__ == "__main__":
#   main()
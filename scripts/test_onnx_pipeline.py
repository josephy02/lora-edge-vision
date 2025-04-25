import numpy as np
import onnxruntime as ort
from PIL import Image
import os


# Set up ONNX Runtime session options for better performance
session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session_options.intra_op_num_threads = 1

# Load the text encoder ONNX model
text_encoder_path = "onnx/text_encoder/model.onnx"
if os.path.exists(text_encoder_path):
    text_encoder = ort.InferenceSession(text_encoder_path, session_options)
    print("Loaded text encoder")
else:
    print(f"Warning: Text encoder not found at {text_encoder_path}")
    text_encoder = None

# Load the UNet ONNX model - try different paths since Optimum export might use different naming
unet_paths = ["onnx/unet/model.onnx", "onnx/unet.onnx"]
unet_model_path = None
for path in unet_paths:
    if os.path.exists(path):
        unet_model_path = path
        break

if unet_model_path:
    print(f"Using UNet model from: {unet_model_path}")
    unet = ort.InferenceSession(unet_model_path, session_options)
else:
    raise FileNotFoundError("UNet model not found in onnx directory")

# Try a basic forward pass to see if the model works
print("Running basic UNet forward pass...")

# Create dummy inputs for UNet
dummy_latent = np.random.randn(1, 4, 64, 64).astype(np.float32)  # Latent input
dummy_t = np.array([1], dtype=np.int64)  # Timestep
dummy_hs = np.random.randn(1, 77, 768).astype(np.float32)  # Text embeddings

# Get input names
unet_inputs = unet.get_inputs()
input_names = [input.name for input in unet_inputs]
print(f"UNet input names: {input_names}")

# Get output names
unet_outputs = unet.get_outputs()
output_names = [output.name for output in unet_outputs]
print(f"UNet output names: {output_names}")

# Create input feed based on actual input names
input_feed = {}
for i, name in enumerate(input_names):
    if "sample" in name.lower():
        input_feed[name] = dummy_latent
    elif "timestep" in name.lower():
        input_feed[name] = dummy_t
    elif "encoder_hidden_states" in name.lower() or "text_embeds" in name.lower():
        input_feed[name] = dummy_hs
    else:
        print(f"Warning: Unknown input name {name}, skipping")

# Run the model
try:
    outputs = unet.run(output_names, input_feed)
    print(f"UNet forward pass successful! Output shape: {outputs[0].shape}")
except Exception as e:
    print(f"Error running UNet forward pass: {str(e)}")
    import traceback
    traceback.print_exc()
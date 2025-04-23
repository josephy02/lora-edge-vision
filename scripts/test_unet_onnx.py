import numpy as np
import torch
import onnxruntime as ort

'''
This script loads a LoRA-augmented Stable Diffusion model in ONNX format,
runs a forward pass with dummy inputs, and prints the output shape.
'''

sess = ort.InferenceSession("onnx/unet.onnx")
# build the same dummy inputs you used to export
#    – batch size 1, C channels, H×W spatial dims
#    (unet.config.sample_size was 64, and in_channels was 4)
dummy_latent = np.random.randn(1, 4, 64, 64).astype(np.float32)
dummy_t     = np.array([1], dtype=np.int64)
# text_encoder.config.max_position_embeddings=77, hidden_size=768 for SD v1.5
dummy_hs    = np.random.randn(1, 77, 768).astype(np.float32)
# Map inputs by name
inp0 = sess.get_inputs()[0].name
inp1 = sess.get_inputs()[1].name
inp2 = sess.get_inputs()[2].name
out_name = sess.get_outputs()[0].name

# run & inspect shape
outs = sess.run([out_name], {
    inp0: dummy_latent,
    inp1: dummy_t,
    inp2: dummy_hs,
})
print("ONNX UNet forward OK, output shape:", outs[0].shape)

import os
import numpy as np
import onnxruntime as ort


def test_onnx_model(model_path):
    """Test if an ONNX model can be loaded and run with a simple input."""
    print(f"Testing ONNX model: {model_path}")

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return False

    try:
        # Load the model
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        session = ort.InferenceSession(model_path, session_options)
        print("Successfully loaded the model!")

        # Get input details
        inputs = session.get_inputs()
        print(f"Model has {len(inputs)} inputs:")
        for i, input_info in enumerate(inputs):
            print(f"Input #{i}:")
            print(f"Name: {input_info.name}")
            print(f"Shape: {input_info.shape}")
            print(f"Type: {input_info.type}")

        # Get output details
        outputs = session.get_outputs()
        print(f"Model has {len(outputs)} outputs:")
        for i, output_info in enumerate(outputs):
            print(f"Output #{i}:")
            print(f"Name: {output_info.name}")
            print(f"Shape: {output_info.shape}")
            print(f"Type: {output_info.type}")

        return True

    except Exception as e:
        print(f"Error testing the model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    # Check all possible model locations
    possible_paths = [
        "onnx/unet.onnx",
        "onnx/unet/model.onnx",
        "onnx_mac/unet.onnx"
    ]

    success = False
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found model at {path}")
            if test_onnx_model(path):
                success = True
                print(f"Successfully tested model at {path}")
            else:
                print(f"Failed to test model at {path}")
        else:
            print(f"Model not found at {path}")

    if not success:
        print("No models were successfully tested.")
        print("Try exporting with a different method or opset version.")

if __name__ == "__main__":
    main()
import torch
import os
from torch.utils.mobile_optimizer import optimize_for_mobile

# Define paths
MODELS_DIR = 'models' # Assuming 'models' directory is relative to this script
input_model_path = os.path.join(MODELS_DIR, 'ai_detector_model_pytorch_script.ts')
output_model_path = os.path.join(MODELS_DIR, 'ai_detector_model_pytorch_script.ptl')

def optimize_model_for_mobile(input_path, output_path):
    print(f"Loading TorchScript model from: {input_path}")
    # Load the TorchScript model
    # Ensure it's loaded onto CPU if not explicitly traced for mobile CPU
    model = torch.jit.load(input_path, map_location='cpu')
    model.eval() # Set to evaluation mode

    print("Optimizing model for mobile...")
    # Apply mobile optimization
    # The optimize_for_mobile function expects a ScriptModule
    optimized_model = optimize_for_mobile(model)

    print(f"Saving optimized model to: {output_path}")
    # Save the optimized model in PyTorch Lite format
    optimized_model._save_for_lite_interpreter(output_path)
    print("Optimization complete.")

if __name__ == '__main__':
    if os.path.exists(input_model_path):
        # Create MODELS_DIR if it doesn't exist for the output model
        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR)
        optimize_model_for_mobile(input_model_path, output_model_path)
    else:
        print(f"Error: Input model not found at {input_model_path}. Please ensure the trained TorchScript model exists.")
        print("You might need to run src/train_pytorch.py first to generate the model.")
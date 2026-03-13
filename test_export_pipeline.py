"""
test_export_pipeline.py — Verification script for ONNX and TorchScript exports.

This script:
  1. Exports the model using export_model.py
  2. Loads the ONNX model and runs a test inference.
  3. Compares ONNX results with the original PyTorch results.
"""

import torch
import numpy as np
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from model_prnu import DeepFusionNet
from export_model import export_onnx, export_torchscript, ExportWrapper

IMG_SIZE = 512
PRNU_DIM = 16
MODELS_DIR = 'models'

def verify_onnx(torch_model, onnx_path):
    import onnxruntime as ort
    print("\n── Verifying ONNX ──")
    
    wrapper = ExportWrapper(torch_model)
    wrapper.eval()
    
    # Create dummy input: (1, 19, 512, 512)
    dummy_input = torch.randn(1, 3 + PRNU_DIM, IMG_SIZE, IMG_SIZE)
    
    # Get PyTorch output
    with torch.no_grad():
        torch_out = wrapper(dummy_input).numpy()
    
    # Get ONNX output
    ort_session = ort.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    onnx_out = ort_outs[0]
    
    # Compare
    diff = np.abs(torch_out - onnx_out).max()
    print(f"  Max difference: {diff:.2e}")
    if diff < 1e-4:
        print("  ✅ ONNX verification PASSED")
    else:
        print("  ❌ ONNX verification FAILED (large difference)")

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    print("Initializing model...")
    model = DeepFusionNet()
    model.eval()
    
    print("Exporting to ONNX...")
    onnx_path = export_onnx(model, MODELS_DIR)
    
    if onnx_path and os.path.exists(onnx_path):
        verify_onnx(model, onnx_path)
    else:
        print("ONNX export failed, skipping verification.")

    print("\nExporting to TorchScript...")
    ts_path = export_torchscript(model, MODELS_DIR)
    if ts_path:
        print(f"  ✅ TorchScript exported to {ts_path}")

if __name__ == "__main__":
    main()

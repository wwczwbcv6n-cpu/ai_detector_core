"""
export_model.py — Multi-target model export pipeline

Exports the trained DeepFusionNet to:

  1. TorchScript (.ts)     — current mobile format (KMP/Android)
  2. ONNX (.onnx)          — cross-platform: C++ server, Windows, Linux
  3. Core ML (.mlpackage)  — iOS/macOS Apple Neural Engine (Swift)

Usage:
    python export_model.py                          # exports all formats
    python export_model.py --format coreml          # only Core ML
    python export_model.py --format onnx            # only ONNX
    python export_model.py --model models/my.pth    # custom weights

Why each format:
  TorchScript  — runs via LibTorch in C++ or Android (TorchScript Java API)
  ONNX         — C++ inference server with onnxruntime (2-5× faster than Python)
                 Also runs on Windows, Linux, ARM, WASM
  Core ML      — Runs on Apple Neural Engine (~10-50× faster than Python API)
                 Swift code calls the model entirely on-device, no network needed

Language stack comparison:
  Python/Flask   — easy, flexible, ~50ms/image
  C++/ONNX RT    — ~5-15ms/image, low memory, no Python dependency
  Swift/CoreML   — ~1-3ms/image on A-series chip ANE, fully on-device
  Rust/candle    — future: zero-copy, memory-safe, ~10ms/image
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from model_prnu import DeepFusionNet

IMG_SIZE     = 512
PRNU_DIM     = 16
MODELS_DIR   = os.path.join(os.path.dirname(__file__), 'models')
WEIGHTS_PATH = os.path.join(MODELS_DIR, 'ai_detector_prnu_fusion.pth')


# ---------------------------------------------------------------------------
#  Wrapper: single-input model for export
#  (ONNX and CoreML work best with one input tensor)
# ---------------------------------------------------------------------------

class ExportWrapper(nn.Module):
    """
    Wraps DeepFusionNet so it accepts a single concatenated input:
        x: (B, 3 + 16, H, W) — image (3ch) + PRNU vector tiled to (16, H, W)

    For ONNX and CoreML, multiple inputs are harder to handle in Swift/C++.
    Tiling the PRNU vector spatially lets us keep a single tensor input.
    The model strips the PRNU channels back out internally.
    """
    def __init__(self, model: DeepFusionNet):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 19, H, W)
            channels 0-2   → RGB image
            channels 3-18  → PRNU features tiled spatially (16 values)
        Returns: (B, 1) logit  →  apply sigmoid for probability
        """
        img       = x[:, :3, :, :]                    # (B, 3, H, W)
        prnu_tile = x[:, 3:, :, :]                    # (B, 16, H, W)
        prnu      = prnu_tile.mean(dim=[2, 3])         # (B, 16) — spatial avg
        return self.model(img, prnu)


def load_model(weights_path: str) -> DeepFusionNet:
    model = DeepFusionNet()
    if os.path.exists(weights_path):
        state = torch.load(weights_path, map_location='cpu')
        try:
            model.load_state_dict(state)
            print(f"✓ Loaded weights from {weights_path}")
        except RuntimeError as e:
            print(f"⚠  Weight mismatch ({e}) — exporting with random weights.")
    else:
        print(f"⚠  No weights at {weights_path} — exporting with random weights.")
    model.eval()
    return model


# ---------------------------------------------------------------------------
#  1. TorchScript export (existing format, updated for v4)
# ---------------------------------------------------------------------------

def export_torchscript(model: DeepFusionNet, out_dir: str):
    print("\n── TorchScript export ──")
    dummy_img  = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    dummy_prnu = torch.randn(1, PRNU_DIM)

    try:
        scripted = torch.jit.trace(model, (dummy_img, dummy_prnu), check_trace=False, strict=True)
        out_path = os.path.join(out_dir, 'ai_detector_prnu_fusion_v4.ts')
        scripted.save(out_path)
        print(f"  Saved → {out_path}")

        # Quantised (INT8) version — smaller file for mobile
        print("  Generating quantized version...")
        model_cpu = model.cpu()
        quantised = torch.quantization.quantize_dynamic(
            model_cpu, {nn.Linear}, dtype=torch.qint8
        )
        q_scripted = torch.jit.trace(quantised, (dummy_img.cpu(), dummy_prnu.cpu()), check_trace=False, strict=True)
        q_path = os.path.join(out_dir, 'ai_detector_quantized_v4_int8.pt')
        q_scripted.save(q_path)
        print(f"  Quantised INT8 → {q_path}")
        return out_path
    except Exception as e:
        print(f"  ✗ TorchScript failed: {e}")
        return None


# ---------------------------------------------------------------------------
#  2. ONNX export
# ---------------------------------------------------------------------------

def export_onnx(model: DeepFusionNet, out_dir: str):
    print("\n── ONNX export ──")
    try:
        import onnx
    except ImportError:
        print("  ✗ onnx not installed. Run: pip install onnx onnxruntime")
        return None

    wrapper = ExportWrapper(model)
    wrapper.eval()

    # Single combined input: image + tiled PRNU
    dummy = torch.randn(1, 3 + PRNU_DIM, IMG_SIZE, IMG_SIZE)
    out_path = os.path.join(out_dir, 'ai_detector_v4.onnx')

    try:
        torch.onnx.export(
            wrapper, dummy, out_path,
            input_names=['image_and_prnu'],
            output_names=['logit'],
            dynamic_axes={
                'image_and_prnu': {0: 'batch'},
                'logit':          {0: 'batch'},
            },
            opset_version=17,
            do_constant_folding=True,
        )
        
        # Add ONNX Metadata for comprehensivity
        model_onnx = onnx.load(out_path)
        meta = model_onnx.metadata_props.add()
        meta.key, meta.value = "description", "AI Image Detector v4 — 5-branch Fusion with PRNU features"
        meta = model_onnx.metadata_props.add()
        meta.key, meta.value = "input_format", "NCHW: [B, 19, 512, 512]. Ch 0-2: RGB, Ch 3-18: Tiled PRNU"
        onnx.save(model_onnx, out_path)

        size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f"  Saved → {out_path}  ({size_mb:.1f} MB)")
        print("""
  C++ usage with onnxruntime:
    #include <onnxruntime/core/session/onnxruntime_cxx_api.h>
    Ort::Session session(env, L"ai_detector_v4.onnx", opts);
    // ~5-15ms per image vs ~50ms in Python
        """)
        return out_path
    except Exception as e:
        print(f"  ✗ ONNX export failed: {e}")
        return None


# ---------------------------------------------------------------------------
#  3. Core ML export (iOS / macOS / Apple Neural Engine)
# ---------------------------------------------------------------------------

def export_coreml(model: DeepFusionNet, out_dir: str):
    print("\n── Core ML export (iOS / macOS / Apple Neural Engine) ──")
    try:
        import coremltools as ct
    except ImportError:
        print("  ✗ coremltools not installed. Run: pip install coremltools")
        print("  Note: requires macOS or Linux with coremltools >= 7.0")
        return None

    wrapper = ExportWrapper(model)
    wrapper.eval()

    dummy = torch.randn(1, 3 + PRNU_DIM, IMG_SIZE, IMG_SIZE)

    try:
        # Trace via TorchScript first
        traced = torch.jit.trace(wrapper, dummy, check_trace=False)

        # Convert to Core ML
        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(
                    name="image_and_prnu",
                    shape=(1, 3 + PRNU_DIM, IMG_SIZE, IMG_SIZE),
                )
            ],
            outputs=[ct.TensorType(name="logit")],
            minimum_deployment_target=ct.target.iOS16,
            compute_precision=ct.precision.FLOAT16,   # ANE requires FP16
            compute_units=ct.ComputeUnit.ALL,          # CPU + GPU + ANE
        )

        # Add metadata for comprehensivity
        mlmodel.short_description = "AI Image Detector v4 — DeepFusionNet"
        mlmodel.input_description["image_and_prnu"] = "512x512 Image (RGB) + 16-dim PRNU features tiled spatially"
        mlmodel.output_description["logit"] = "Raw model logit. Apply sigmoid(x) for probability (0=REAL, 1=AI)."
        mlmodel.author = "AI Detector Core"
        mlmodel.license = "Private"
        mlmodel.version = "4.0"

        out_path = os.path.join(out_dir, 'AIDetector_v4.mlpackage')
        mlmodel.save(out_path)
        size_mb = sum(
            os.path.getsize(os.path.join(r, f))
            for r, _, fs in os.walk(out_path) for f in fs
        ) / 1024 / 1024
        print(f"  Saved → {out_path}  ({size_mb:.1f} MB)")

        print("""
  Swift usage (add AIDetector_v4.mlpackage to Xcode project):

    import CoreML
    import Vision

    // Load model (once, at startup)
    let model = try AIDetector_v4(configuration: MLModelConfiguration())

    // Predict on a UIImage
    func detectAI(image: UIImage) -> Double {
        guard let cgImage = image.cgImage else { return 0 }
        let handler = VNImageRequestHandler(cgImage: cgImage)

        // Prepare input: (1, 19, 512, 512)
        let pixelBuffer = image.toMLMultiArray(size: 512, prnuFeats: prnuFeatures)

        let input = AIDetector_v4Input(image_and_prnu: pixelBuffer)
        let output = try! model.prediction(input: input)
        let prob = sigmoid(output.logit[0].doubleValue)
        return prob   // 0=REAL, 1=AI
    }
    // Runs on Apple Neural Engine — ~1-3ms per image on A15+
        """)
        return out_path
    except Exception as e:
        print(f"  ✗ Core ML export failed: {e}")
        import traceback; traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
#  4. Rust/candle guide (cannot auto-export, but show the path)
# ---------------------------------------------------------------------------

def show_rust_guide():
    print("""
── Rust / candle (future — highest performance) ──

  Hugging Face's `candle` is a pure-Rust ML framework:
    • Zero Python dependency
    • Memory-safe (no segfaults)
    • Compiles to a single static binary
    • CUDA support via cuDNN
    • ~10ms/image on GPU, ~30ms on CPU

  Steps to use candle for inference:
    1. Export model weights as ONNX (done above)
    2. In Rust (Cargo.toml):
         [dependencies]
         candle-core = "0.8"
         candle-nn   = "0.8"
         candle-onnx = "0.8"   # load ONNX models directly

    3. Rust inference server (Axum):
         use candle_onnx::OnnxModel;
         use axum::{routing::post, Router};

         let model = OnnxModel::load("ai_detector_v4.onnx")?;
         let app = Router::new()
             .route("/analyze", post(analyze_handler));
         axum::serve(listener, app).await?;

  Benefits over Python:
    • 10× lower memory usage
    • No GIL — true parallel inference
    • Single binary deployment (no Python runtime)
    • Can be compiled to WebAssembly for browser inference

  See: https://github.com/huggingface/candle
    """)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Export DeepFusionNet v4")
    parser.add_argument('--model',  default=WEIGHTS_PATH, help='Path to .pth weights')
    parser.add_argument('--out',    default=MODELS_DIR,   help='Output directory')
    parser.add_argument('--format', default='all',
                        choices=['all', 'torchscript', 'onnx', 'coreml'])
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  DeepFusionNet v4 — Multi-target Export")
    print(f"  Weights : {args.model}")
    print(f"  Output  : {args.out}")
    print(f"{'='*60}")

    model = load_model(args.model)

    if args.format in ('all', 'torchscript'):
        export_torchscript(model, args.out)
    if args.format in ('all', 'onnx'):
        export_onnx(model, args.out)
    if args.format in ('all', 'coreml'):
        export_coreml(model, args.out)
    if args.format == 'all':
        show_rust_guide()

    print(f"\n{'='*60}")
    print("  Export complete.")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

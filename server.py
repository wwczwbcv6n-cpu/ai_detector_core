import sys
import os
import io
import base64
from flask import Flask, request, jsonify

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from detect import Detector

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20 MB max upload

# ── Model version metadata ──────────────────────────────────────────────────
MODEL_VERSION = "2.0-PRNUFusion"

# ── Globals ─────────────────────────────────────────────────────────────────
try:
    detector = Detector()
    _model_type = getattr(detector, 'use_fusion', False)
    print(f"✓ Detector ready — using {'PRNUFusionNet' if _model_type else 'PyTorchCNN (legacy)'}")
except FileNotFoundError as e:
    print(f"CRITICAL ERROR: Could not initialize detector.\n  {e}")
    print("  Run training first: python src/train_pytorch.py")
    detector = None


# ── Health endpoint ──────────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    if detector is None:
        return jsonify({"status": "error", "message": "Detector not initialized"}), 503
    return jsonify({
        "status":        "ok",
        "model":         "PRNUFusionNet" if detector.use_fusion else "PyTorchCNN",
        "model_version": MODEL_VERSION,
        "prnu_enabled":  True,
        "gradcam":       True,
    })


# ── Main analysis endpoint ───────────────────────────────────────────────────
@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Analyze an image for AI-generated content.

    Request (multipart/form-data):
        image       — image file (required)

    Query params:
        heatmap     — "true" (default) | "false"  skip Grad-CAM for speed

    Response JSON:
        {
          "ai_probability":  float,            // 0-1
          "conclusion":      "AI-Generated" | "REAL",
          "model_type":      str,
          "prnu_analysis":   { ... },          // PRNU forensic metrics
          "heatmap_base64":  str | null,       // base64 PNG overlay
          "heatmap_width":   int,
          "heatmap_height":  int,
        }
    """
    if detector is None:
        return jsonify({"error": "Detector not initialized. Check server logs."}), 500

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided in 'image' field."}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file."}), 400

    # ?heatmap=false disables Grad-CAM for faster pre-screening
    want_heatmap = request.args.get('heatmap', 'true').lower() != 'false'

    try:
        image_data = file.read()
        result = detector.predict(image_data, compute_heatmap=want_heatmap)

        if result:
            return jsonify(result)
        else:
            return jsonify({"error": "Failed to process image."}), 500

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Internal error: {str(e)}"}), 500


# ── Video frame endpoint (stub for mobile app) ───────────────────────────────
@app.route('/analyze/frame', methods=['POST'])
def analyze_frame():
    """
    Lightweight single-frame analysis without Grad-CAM.
    Used by the mobile app for fast video frame streaming.
    Returns only ai_probability and conclusion (no heatmap).
    """
    if detector is None:
        return jsonify({"error": "Detector not initialized."}), 500

    if 'frame' not in request.files:
        return jsonify({"error": "No frame file provided in 'frame' field."}), 400

    file = request.files['frame']
    try:
        image_data = file.read()
        result = detector.predict(image_data, compute_heatmap=False)
        # Return slim response for video frame streaming
        return jsonify({
            "ai_probability": result.get("ai_probability"),
            "conclusion":     result.get("conclusion"),
            "model_type":     result.get("model_type"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Entry point ──────────────────────────────────────────────────────────────
def main():
    if detector is None:
        print("Exiting: Detector could not be initialized.")
        return

    print(f"Starting AI Detector Server v{MODEL_VERSION} ...")
    print("  POST /analyze          — full analysis with Grad-CAM heatmap")
    print("  POST /analyze/frame    — fast single-frame (no heatmap)")
    print("  GET  /health           — model and health status")
    print("  Listening on http://0.0.0.0:8080")
    app.run(host='0.0.0.0', port=8080, debug=False)


if __name__ == '__main__':
    main()

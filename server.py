"""
AI Detector Server v3.0
Flask REST API with full Apple format support (HEIC, HEVC, ProRAW).

Endpoints:
  POST /analyze          — full image analysis (JPEG, PNG, HEIC, WEBP…)
  POST /analyze/frame    — fast single-frame (no Grad-CAM), for video streaming
  POST /analyze/heic     — dedicated HEIC/ProRAW endpoint (Swift optimised)
  POST /analyze/batch    — analyse multiple images in one request
  GET  /health           — server and model status
  GET  /capabilities     — list supported formats and model info
"""

import sys
import os
import io
import base64
import time
import traceback
from flask import Flask, request, jsonify

# ── Apple HEIC/HEIF support via pillow-heif ──────────────────────────────────
try:
    import pillow_heif
    pillow_heif.register_heif_opener()   # makes PIL.Image.open() handle HEIC
    _HEIC_SUPPORT = True
except ImportError:
    _HEIC_SUPPORT = False
    print("⚠  pillow-heif not installed — HEIC support disabled.")
    print("   Install: pip install pillow-heif")

from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from detect import Detector

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024   # 50 MB (HEIC files are large)

# ── Model version ─────────────────────────────────────────────────────────────
MODEL_VERSION = "3.0-PRNUFusion"

# Formats natively supported by PIL + pillow-heif
SUPPORTED_FORMATS = [
    "image/jpeg", "image/png", "image/webp", "image/gif",
    "image/heic", "image/heif",              # Apple HEIC/HEIF
    "image/avif",                            # AVIF (pillow-heif)
    "image/tiff", "image/bmp",
]

# ── Globals ───────────────────────────────────────────────────────────────────
try:
    detector = Detector()
    _model_type = "PRNUFusionNet" if detector.use_fusion else "PyTorchCNN (legacy)"
    print(f"✓ Detector ready — {_model_type}")
except FileNotFoundError as e:
    print(f"CRITICAL: Could not initialise detector.\n  {e}")
    detector = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _decode_image(file_storage) -> bytes:
    """
    Read uploaded file.  For HEIC/HEIF, convert to JPEG bytes so the
    detector pipeline doesn't need to know about the container format.
    """
    raw = file_storage.read()
    fname = (file_storage.filename or "").lower()
    ctype = (file_storage.content_type or "").lower()

    is_heic = (
        fname.endswith(('.heic', '.heif', '.avif'))
        or 'heic' in ctype or 'heif' in ctype
    )

    if is_heic:
        if not _HEIC_SUPPORT:
            raise ValueError("HEIC support not available. Install pillow-heif.")
        # Decode HEIC → RGB PIL image → JPEG bytes
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        return buf.getvalue()

    return raw


def _run_analysis(image_data: bytes, want_heatmap: bool,
                  platform: str = None) -> dict:
    """Run detector and add server-level metadata."""
    t0 = time.perf_counter()
    result = detector.predict(image_data, compute_heatmap=want_heatmap,
                              platform=platform)
    result["inference_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    result["server_version"] = MODEL_VERSION
    return result


# ── Health ────────────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    if detector is None:
        return jsonify({"status": "error", "message": "Detector not initialised"}), 503
    return jsonify({
        "status":         "ok",
        "model":          _model_type,
        "model_version":  MODEL_VERSION,
        "heic_support":   _HEIC_SUPPORT,
        "prnu_enabled":   True,
        "gradcam":        True,
    })


@app.route('/capabilities', methods=['GET'])
def capabilities():
    """Return supported formats and model capabilities — used by Swift client."""
    return jsonify({
        "supported_formats":   SUPPORTED_FORMATS,
        "heic_support":        _HEIC_SUPPORT,
        "max_upload_bytes":    app.config['MAX_CONTENT_LENGTH'],
        "endpoints": {
            "analyze":       "POST /analyze        — full analysis",
            "analyze_frame": "POST /analyze/frame  — fast frame (no heatmap)",
            "analyze_heic":  "POST /analyze/heic   — HEIC/ProRAW optimised",
            "analyze_batch": "POST /analyze/batch  — up to 8 images",
        },
        "platform_hint": {
            "param":       "?platform=<name>  (optional query parameter)",
            "description": "Tells the detector which platform the content came from. "
                           "Adds compression tier, codec, and PRNU reliability info to the response.",
            "supported": [
                "youtube", "vimeo", "tiktok", "instagram",
                "facebook", "twitter", "snapchat",
                "telegram",           # MTProto / Telegram photo send
                "heic",               # Apple HEIC/HEIF (alias: heif, apple)
                "h264",               # generic H.264/AVC (alias: avc)
                "h265",               # generic H.265/HEVC (alias: hevc, h264/h265)
            ],
            "aliases": {
                "x": "twitter", "twitter/x": "twitter",
                "heif": "heic", "apple": "heic",
                "mtproto": "telegram",
                "avc": "h264", "hevc": "h265", "h264/h265": "h265",
            },
        },
        "model_version": MODEL_VERSION,
    })


# ── Main analysis ─────────────────────────────────────────────────────────────

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Analyse an image for AI-generated content.

    Request (multipart/form-data):
        image       — image file (JPEG, PNG, HEIC, WEBP…)

    Query params:
        heatmap     — "true" (default) | "false"

    Response JSON:
        ai_probability   float [0-1]
        conclusion       "AI-Generated" | "REAL"
        model_type       str
        prnu_analysis    { noise_strength, noise_uniformity, … }
        heatmap_base64   str | null
        inference_ms     float
    """
    if detector is None:
        return jsonify({"error": "Detector not initialised"}), 500

    if 'image' not in request.files:
        return jsonify({"error": "No image file in 'image' field."}), 400

    want_heatmap = request.args.get('heatmap', 'true').lower() != 'false'
    platform     = request.args.get('platform', None)

    try:
        image_data = _decode_image(request.files['image'])
        return jsonify(_run_analysis(image_data, want_heatmap, platform=platform))
    except ValueError as e:
        return jsonify({"error": str(e)}), 415
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Internal error: {e}"}), 500


# ── HEIC / ProRAW dedicated endpoint (Swift-optimised) ────────────────────────

@app.route('/analyze/heic', methods=['POST'])
def analyze_heic():
    """
    Dedicated endpoint for HEIC, HEIF and Apple ProRAW images.

    Accepts:
        multipart/form-data  with field 'image'   (file upload)
        application/octet-stream                  (raw bytes in body)

    Swift client example:
        var req = URLRequest(url: URL(string: "http://host/analyze/heic")!)
        req.httpMethod = "POST"
        req.setValue("application/octet-stream", forHTTPHeaderField: "Content-Type")
        req.httpBody = heicData          // raw Data from PHAsset / UIImage
        URLSession.shared.dataTask(with: req) { ... }

    Response JSON: same as /analyze
    """
    if detector is None:
        return jsonify({"error": "Detector not initialised"}), 500

    if not _HEIC_SUPPORT:
        return jsonify({"error": "HEIC support not available on this server."}), 415

    try:
        # Accept raw body OR multipart
        if request.content_type and 'octet-stream' in request.content_type:
            raw = request.get_data()
        elif 'image' in request.files:
            raw = request.files['image'].read()
        else:
            return jsonify({"error": "Send HEIC bytes as body or 'image' field."}), 400

        # Decode HEIC → JPEG bytes
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        image_data = buf.getvalue()

        want_heatmap = request.args.get('heatmap', 'false').lower() != 'false'
        platform     = request.args.get('platform', None)
        return jsonify(_run_analysis(image_data, want_heatmap, platform=platform))

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"HEIC decode failed: {e}"}), 500


# ── Fast frame endpoint (video streaming from Swift) ──────────────────────────

@app.route('/analyze/frame', methods=['POST'])
def analyze_frame():
    """
    Lightweight single-frame analysis — no Grad-CAM.
    Used by the mobile app for real-time video frame streaming.

    Accepts JPEG bytes (raw body OR 'frame' field).

    Swift client example — send CMSampleBuffer as JPEG:
        let imageData = frame.jpegData(compressionQuality: 0.7)!
        var req = URLRequest(url: frameURL)
        req.httpMethod = "POST"
        req.setValue("application/octet-stream", forHTTPHeaderField: "Content-Type")
        req.httpBody = imageData
    """
    if detector is None:
        return jsonify({"error": "Detector not initialised"}), 500

    try:
        if request.content_type and 'octet-stream' in request.content_type:
            image_data = request.get_data()
        elif 'frame' in request.files:
            image_data = _decode_image(request.files['frame'])
        else:
            return jsonify({"error": "Send frame as body or 'frame' field."}), 400

        platform = request.args.get('platform', None)
        result   = _run_analysis(image_data, want_heatmap=False, platform=platform)
        return jsonify({
            "ai_probability":      result.get("ai_probability"),
            "conclusion":          result.get("conclusion"),
            "model_type":          result.get("model_type"),
            "inference_ms":        result.get("inference_ms"),
            "platform_compression": result.get("platform_compression"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Batch endpoint ────────────────────────────────────────────────────────────

@app.route('/analyze/batch', methods=['POST'])
def analyze_batch():
    """
    Analyse up to 8 images in a single request.
    All images in multipart/form-data under any field name.

    Response JSON:
        { "results": [ { filename, ai_probability, conclusion, inference_ms }, … ] }
    """
    if detector is None:
        return jsonify({"error": "Detector not initialised"}), 500

    files = list(request.files.values())
    if not files:
        return jsonify({"error": "No files provided."}), 400
    if len(files) > 8:
        return jsonify({"error": "Maximum 8 images per batch."}), 400

    platform = request.args.get('platform', None)
    results  = []
    for f in files:
        try:
            image_data = _decode_image(f)
            r = _run_analysis(image_data, want_heatmap=False, platform=platform)
            results.append({
                "filename":            f.filename,
                "ai_probability":      r.get("ai_probability"),
                "conclusion":          r.get("conclusion"),
                "inference_ms":        r.get("inference_ms"),
                "prnu_analysis":       r.get("prnu_analysis"),
                "platform_compression": r.get("platform_compression"),
            })
        except Exception as e:
            results.append({"filename": f.filename, "error": str(e)})

    return jsonify({"results": results})


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    if detector is None:
        print("Exiting: Detector could not be initialised.")
        return

    print(f"\n  AI Detector Server v{MODEL_VERSION}")
    print(f"  HEIC support : {'✓ enabled' if _HEIC_SUPPORT else '✗ disabled'}")
    print("  POST /analyze          — full analysis (JPEG, PNG, HEIC, WEBP…)")
    print("  POST /analyze/heic     — HEIC/ProRAW direct (Swift-optimised)")
    print("  POST /analyze/frame    — fast frame (no heatmap)")
    print("  POST /analyze/batch    — up to 8 images")
    print("  GET  /health           — server status")
    print("  GET  /capabilities     — formats and model info")
    print("  Listening on http://0.0.0.0:8080\n")
    app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)


if __name__ == '__main__':
    main()

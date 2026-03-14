"""
AI Detector Server v4.0
Flask REST API with full Apple format support (HEIC, HEVC, ProRAW).
Now with RAG (Retrieval-Augmented Generation) and CAG (Cache-Augmented Generation).

Endpoints:
  POST /analyze          — full image analysis (JPEG, PNG, HEIC, WEBP…)
  POST /analyze/frame    — fast single-frame (no Grad-CAM), for video streaming
  POST /analyze/heic     — dedicated HEIC/ProRAW endpoint (Swift optimised)
  POST /analyze/batch    — analyse multiple images in one request
  GET  /health           — server and model status
  GET  /capabilities     — list supported formats and model info

  RAG endpoints:
  GET  /rag/stats        — RAG vector store statistics
  POST /rag/add          — manually add a labeled example to the RAG store
  POST /rag/clear        — clear all RAG store entries

  CAG endpoints:
  GET  /cache/stats      — detection cache statistics
  POST /cache/clear      — clear all cached results

  RAG query params (on /analyze and /analyze/batch):
  ?rag=true              — retrieve similar past cases and augment probability
  ?explain=true          — add LLM explanation to the response

  LLM config (environment variables):
  OLLAMA_URL             — Ollama base URL  (default: http://localhost:11434)
  OLLAMA_MODEL           — Ollama model     (default: llama3.2)
  OPENAI_API_KEY         — enables OpenAI backend (fallback if Ollama absent)
"""

import sys
import os
import io
import base64
import time
import traceback
import numpy as np
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
from rag_store import ImageRAGStore
from cag_cache import DetectionCache
from llm_explainer import LLMExplainer
from umfre import ForensicRecoverer

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200 MB (video + HEIC files)

# ── Model version ─────────────────────────────────────────────────────────────
MODEL_VERSION = "4.0-RAG+CAG"

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

# ── RAG + CAG init ────────────────────────────────────────────────────────────
rag_store = ImageRAGStore()
det_cache = DetectionCache()
umfre_engine = ForensicRecoverer(device='cpu', denoise_method='auto',
                                 tile_size=1024, verbose=False)
try:
    llm_explainer = LLMExplainer()
    print(f"✓ LLMExplainer ready — backend={llm_explainer.info()['backend']}, "
          f"chunks={llm_explainer.info()['knowledge_chunks']}")
except Exception as _e:
    print(f"⚠  LLMExplainer init failed: {_e}")
    llm_explainer = None


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


def _run_analysis(
    image_data: bytes,
    want_heatmap: bool,
    platform: str = None,
    use_rag: bool = False,
    want_explain: bool = False,
) -> dict:
    """
    Run detector with optional CAG cache lookup, RAG augmentation, and LLM explanation.

    Flow
    ----
    1. CAG cache lookup  →  hit: return instantly (no model inference)
    2. Detector inference  (with optional 768-dim embedding capture for RAG)
    3. RAG retrieve similar past cases  →  augment probability
    4. LLM explanation  →  natural-language explanation field
    5. Store result in CAG cache + auto-index embedding in RAG store
    """
    t0 = time.perf_counter()

    # ── 1. CAG: check cache ───────────────────────────────────────────────────
    cached = det_cache.lookup(image_data)
    if cached is not None:
        cached["inference_ms"]   = round((time.perf_counter() - t0) * 1000, 1)
        cached["server_version"] = MODEL_VERSION
        # Still run explain on cached result if requested and not already there
        if want_explain and "explanation" not in cached and llm_explainer:
            try:
                cached["explanation"] = llm_explainer.explain(cached)
            except Exception:
                pass
        return cached

    # ── 2. Detector inference (optionally capture fusion embedding) ───────────
    embedding  = None
    import torch

    if use_rag and detector and detector.use_fusion and hasattr(detector.model, 'fusion'):
        _captured: dict = {}

        def _emb_hook(module, inp, out):
            if isinstance(out, torch.Tensor):
                _captured['emb'] = out.detach().cpu().float().numpy()

        _handle = detector.model.fusion.register_forward_hook(_emb_hook)
        try:
            result = detector.predict(image_data, compute_heatmap=want_heatmap,
                                      platform=platform)
        finally:
            _handle.remove()

        raw = _captured.get('emb')
        if raw is not None:
            embedding = raw[0] if raw.ndim == 2 else raw   # (768,)
    else:
        result = detector.predict(image_data, compute_heatmap=want_heatmap,
                                  platform=platform)

    result["inference_ms"]   = round((time.perf_counter() - t0) * 1000, 1)
    result["server_version"] = MODEL_VERSION

    # ── 3. RAG: retrieve neighbors + augment probability ─────────────────────
    neighbors = []
    if use_rag and embedding is not None and "ai_probability" in result:
        neighbors = rag_store.retrieve(embedding, k=5)
        if neighbors:
            aug_prob = rag_store.augment_probability(
                result["ai_probability"], neighbors
            )
            result["ai_probability_raw"] = result["ai_probability"]
            result["ai_probability"]     = round(aug_prob, 4)
            result["conclusion"]         = "AI-Generated" if aug_prob > 0.5 else "REAL"
            result["rag_neighbors"] = [
                {
                    "verdict":    n["verdict"],
                    "confidence": round(n["confidence"], 3),
                    "distance":   round(n["distance"], 2),
                }
                for n in neighbors
            ]

    # ── 4. LLM explanation ────────────────────────────────────────────────────
    if want_explain and llm_explainer:
        try:
            result["explanation"] = llm_explainer.explain(result, neighbors or None)
        except Exception as e:
            result["explanation"] = f"[explanation unavailable: {e}]"

    # ── 5. Store in CAG cache + auto-index embedding in RAG store ─────────────
    det_cache.store(image_data, result)

    if embedding is not None and "ai_probability" in result:
        try:
            rag_store.add(
                embedding=embedding,
                verdict=result["conclusion"],
                confidence=result.get("ai_probability", 0.5),
            )
        except Exception:
            pass

    return result


# ── Health ────────────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    if detector is None:
        return jsonify({"status": "error", "message": "Detector not initialised"}), 503
    return jsonify({
        "status":           "ok",
        "model":            _model_type,
        "model_version":    MODEL_VERSION,
        "heic_support":     _HEIC_SUPPORT,
        "prnu_enabled":     True,
        "gradcam":          True,
        "rag":              rag_store.stats(),
        "cache":            det_cache.stats(),
        "llm_explainer":    llm_explainer.info() if llm_explainer else None,
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
    use_rag      = request.args.get('rag',     'false').lower() == 'true'
    want_explain = request.args.get('explain', 'false').lower() == 'true'

    try:
        image_data = _decode_image(request.files['image'])
        return jsonify(_run_analysis(
            image_data, want_heatmap,
            platform=platform, use_rag=use_rag, want_explain=want_explain,
        ))
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

    platform     = request.args.get('platform', None)
    use_rag      = request.args.get('rag',     'false').lower() == 'true'
    want_explain = request.args.get('explain', 'false').lower() == 'true'
    results = []
    for f in files:
        try:
            image_data = _decode_image(f)
            r = _run_analysis(
                image_data, want_heatmap=False,
                platform=platform, use_rag=use_rag, want_explain=want_explain,
            )
            entry = {
                "filename":             f.filename,
                "ai_probability":       r.get("ai_probability"),
                "conclusion":           r.get("conclusion"),
                "inference_ms":         r.get("inference_ms"),
                "prnu_analysis":        r.get("prnu_analysis"),
                "platform_compression": r.get("platform_compression"),
                "cache_hit":            r.get("cache_hit", False),
            }
            if use_rag and "rag_neighbors" in r:
                entry["rag_neighbors"]        = r["rag_neighbors"]
                entry["ai_probability_raw"]   = r.get("ai_probability_raw")
            if want_explain and "explanation" in r:
                entry["explanation"] = r["explanation"]
            results.append(entry)
        except Exception as e:
            results.append({"filename": f.filename, "error": str(e)})

    return jsonify({"results": results})


# ── Compression analysis (UMFRE) ──────────────────────────────────────────────

@app.route('/analyze/compression', methods=['POST'])
def analyze_compression():
    """
    Analyze the compression history of an uploaded image or video.

    Accepts
    -------
    Multipart form-data  field 'file' or 'image'  (images and videos)
    Raw bytes body       Content-Type: application/octet-stream

    Supported formats
    -----------------
    Images:  JPEG, PNG, WebP, HEIC, HEIF, AVIF
    Videos:  MP4 (H.264/H.265/AV1), WebM (VP8/VP9), MOV, MKV

    Returns JSON with:
      format, dimensions, media_type         — basic media info
      ── JPEG ────────────────────────────────────────────────
      jpeg_quality, chroma_subsampling        — quality factor + chroma
      is_progressive, has_restart_markers     — bitstream scan flags
      huffman_optimized                       — non-standard Huffman tables
      encoder_signature, encoder_confidence   — encoder fingerprint
      encoder_candidates                      — ranked encoder list
      double_jpeg, estimated_q1               — double-compression detection
      dct_periodicity_score                   — DCT histogram peak strength
      quantization_tables                     — embedded Q-tables
      ── PNG ─────────────────────────────────────────────────
      png_compression                         — zlib compression level (0-9)
      ── WebP ────────────────────────────────────────────────
      webp_type, webp_lossless               — VP8 | VP8L | VP8X
      webp_has_alpha, webp_animated          — extended WebP flags
      webp_quality_estimate                  — estimated lossy quality
      ── HEIC / HEIF / AVIF ──────────────────────────────────
      heif_brand                             — heic | heix | avif | mif1
      heif_chroma, heif_bit_depth            — chroma subsampling, bit depth
      av1_film_grain, av1_seq_profile        — AV1-specific flags
      ── Video (H.264 / H.265 / AV1 / VP9 / VP8) ─────────────
      video_codec, video_profile, video_level — codec identity
      video_bit_depth, pixel_format           — sample format
      bitrate_kbps, bits_per_pixel            — bitrate metrics
      quality_tier                            — high | medium | low
      container_format, streams_info          — container + all streams
      re_encoding_detected, re_encoding_evidence — transcoding detection
      color_primaries, transfer_characteristics  — colour space info
      gop_mean, gop_std, transcoding_detected — GOP structure (video)
      mv_entropy                              — motion-vector entropy
      ── Common ──────────────────────────────────────────────
      metadata_stripped, stripping_evidence  — EXIF stripping detection
      notes                                  — human-readable summary
      inference_ms                           — server processing time (ms)
    """
    import tempfile

    try:
        # Accept raw body or multipart ('file' or 'image' field)
        if request.content_type and 'octet-stream' in request.content_type:
            raw = request.get_data()
            ext = '.bin'
        elif 'file' in request.files:
            f   = request.files['file']
            raw = f.read()
            ext = os.path.splitext(f.filename or '.bin')[1] or '.bin'
        elif 'image' in request.files:
            f   = request.files['image']
            raw = f.read()
            ext = os.path.splitext(f.filename or '.jpg')[1] or '.jpg'
        else:
            return jsonify({"error": "Send file as body (octet-stream) or 'file'/'image' form field."}), 400

        t0 = time.perf_counter()

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name

        try:
            media_info = umfre_engine.ingest(tmp_path)
            comp       = umfre_engine.analyze_compression(media_info)
        finally:
            os.unlink(tmp_path)

        elapsed = round((time.perf_counter() - t0) * 1000, 1)

        return jsonify({
            # ── Media basics ─────────────────────────────────────────────────
            "format":               media_info.format,
            "media_type":           media_info.media_type,
            "dimensions":           f"{media_info.width}×{media_info.height}",
            "color_space":          media_info.color_space,
            "frame_count":          media_info.frame_count,
            "fps":                  media_info.fps or None,
            # ── JPEG-specific ────────────────────────────────────────────────
            "jpeg_quality":         comp.jpeg_quality_current,
            "chroma_subsampling":   comp.chroma_subsampling if comp.chroma_subsampling != "unknown" else None,
            "is_progressive":       comp.is_progressive,
            "has_restart_markers":  comp.has_restart_markers,
            "huffman_optimized":    comp.huffman_optimized,
            "encoder_signature":    comp.encoder_signature if comp.encoder_signature != "unknown" else None,
            "encoder_confidence":   comp.encoder_confidence or None,
            "encoder_candidates":   comp.encoder_candidates or None,
            # ── Double-JPEG detection ────────────────────────────────────────
            "double_jpeg":          comp.double_jpeg_detected,
            "estimated_q1":         comp.estimated_q1,
            "dct_periodicity_score": comp.dct_periodicity_score or None,
            "quantization_tables":  comp.quantization_tables or None,
            # ── PNG ──────────────────────────────────────────────────────────
            "png_compression":      comp.png_compression,
            # ── WebP ─────────────────────────────────────────────────────────
            "webp_type":            comp.webp_type,
            "webp_lossless":        comp.webp_lossless,
            "webp_has_alpha":       comp.webp_has_alpha or None,
            "webp_animated":        comp.webp_animated or None,
            "webp_quality_estimate": comp.webp_quality_estimate,
            # ── HEIC / HEIF / AVIF ───────────────────────────────────────────
            "heif_brand":           comp.heif_brand,
            "heif_chroma":          comp.heif_chroma,
            "heif_bit_depth":       comp.heif_bit_depth,
            "av1_film_grain":       comp.av1_film_grain or None,
            "av1_seq_profile":      comp.av1_seq_profile,
            # ── Video codec ──────────────────────────────────────────────────
            "video_codec":          comp.video_codec,
            "video_profile":        comp.video_profile,
            "video_level":          comp.video_level,
            "video_bit_depth":      comp.video_bit_depth,
            "pixel_format":         comp.pixel_format,
            "bitrate_kbps":         comp.bitrate_kbps,
            "bits_per_pixel":       comp.bits_per_pixel,
            "quality_tier":         comp.quality_tier,
            "container_format":     comp.container_format,
            "streams_info":         comp.streams_info or None,
            # ── Re-encoding / transcoding ────────────────────────────────────
            "re_encoding_detected": comp.re_encoding_detected or None,
            "re_encoding_evidence": comp.re_encoding_evidence or None,
            # ── Color ────────────────────────────────────────────────────────
            "color_primaries":      comp.color_primaries,
            "transfer_characteristics": comp.transfer_characteristics,
            # ── Video GOP ────────────────────────────────────────────────────
            "gop_mean":             comp.gop_mean or None,
            "gop_std":              comp.gop_std or None,
            "transcoding_detected": comp.transcoding_detected or None,
            "mv_entropy":           comp.mv_entropy or None,
            # ── Metadata ─────────────────────────────────────────────────────
            "metadata_stripped":    media_info.metadata_stripped,
            "stripping_evidence":   media_info.stripping_evidence or None,
            # ── Summary ──────────────────────────────────────────────────────
            "notes":                comp.notes,
            "inference_ms":         elapsed,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── RAG management ────────────────────────────────────────────────────────────

@app.route('/rag/stats', methods=['GET'])
def rag_stats():
    """Return RAG vector store statistics."""
    return jsonify(rag_store.stats())


@app.route('/rag/add', methods=['POST'])
def rag_add():
    """
    Manually add a labeled image to the RAG store.

    Multipart form:
        image    — image file
        verdict  — "AI-Generated" or "REAL"

    Query params:
        ?rag=true  must be set (safety gate)
    """
    if request.args.get('rag', 'false').lower() != 'true':
        return jsonify({"error": "Pass ?rag=true to confirm RAG indexing."}), 400

    if 'image' not in request.files:
        return jsonify({"error": "No 'image' field."}), 400

    verdict = (request.form.get('verdict') or '').strip()
    if verdict not in ('AI-Generated', 'REAL'):
        return jsonify({"error": "verdict must be 'AI-Generated' or 'REAL'"}), 400

    if detector is None:
        return jsonify({"error": "Detector not initialised"}), 500

    try:
        import torch
        image_data = _decode_image(request.files['image'])

        # Extract embedding via fusion hook
        _captured: dict = {}
        if detector.use_fusion and hasattr(detector.model, 'fusion'):
            def _hook(module, inp, out):
                if isinstance(out, torch.Tensor):
                    _captured['emb'] = out.detach().cpu().float().numpy()
            handle = detector.model.fusion.register_forward_hook(_hook)
            try:
                result = detector.predict(image_data, compute_heatmap=False)
            finally:
                handle.remove()
        else:
            result = detector.predict(image_data, compute_heatmap=False)

        raw = _captured.get('emb')
        if raw is None:
            return jsonify({"error": "Could not extract embedding (non-fusion model)."}), 500

        emb    = raw[0] if raw.ndim == 2 else raw
        row_id = rag_store.add(
            emb, verdict=verdict,
            confidence=result.get("ai_probability", 0.5),
            metadata={"source": "manual"},
        )
        return jsonify({
            "ok":         True,
            "row_id":     row_id,
            "verdict":    verdict,
            "rag_stats":  rag_store.stats(),
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/rag/clear', methods=['POST'])
def rag_clear():
    """Clear all RAG store entries."""
    rag_store.clear()
    return jsonify({"ok": True, "rag_stats": rag_store.stats()})


# ── CAG (cache) management ────────────────────────────────────────────────────

@app.route('/cache/stats', methods=['GET'])
def cache_stats():
    """Return detection cache statistics."""
    return jsonify(det_cache.stats())


@app.route('/cache/clear', methods=['POST'])
def cache_clear():
    """Clear all cached detection results."""
    det_cache.clear()
    return jsonify({"ok": True, "cache_stats": det_cache.stats()})


# ── LLM explainer info ────────────────────────────────────────────────────────

@app.route('/explain/info', methods=['GET'])
def explain_info():
    """Return LLM explainer backend info."""
    if llm_explainer is None:
        return jsonify({"error": "LLMExplainer not initialised"}), 503
    return jsonify(llm_explainer.info())


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    if detector is None:
        print("Exiting: Detector could not be initialised.")
        return

    print(f"\n  AI Detector Server v{MODEL_VERSION}")
    print(f"  HEIC support   : {'✓ enabled' if _HEIC_SUPPORT else '✗ disabled'}")
    _llm_be = llm_explainer.info()['backend'] if llm_explainer else 'unavailable'
    print(f"  LLM backend    : {_llm_be}")
    print(f"  RAG store      : {rag_store.stats()['total_cases']} cases indexed")
    print(f"  Cache          : {det_cache.stats()['db_entries']} entries persisted")
    print()
    print("  POST /analyze/compression  — compression history (JPEG/PNG/WebP/HEIC/HEIF/AVIF/MP4/WebM/H.264/H.265/AV1/VP9/VP8)")
    print("  POST /analyze              — full analysis (?rag=true &explain=true)")
    print("  POST /analyze/heic         — HEIC/ProRAW direct (Swift-optimised)")
    print("  POST /analyze/frame        — fast frame (no heatmap)")
    print("  POST /analyze/batch        — up to 8 images (?rag=true &explain=true)")
    print("  GET  /health               — server + RAG + cache status")
    print("  GET  /capabilities         — formats and model info")
    print("  GET  /rag/stats            — RAG vector store stats")
    print("  POST /rag/add?rag=true     — manually index a labeled image")
    print("  POST /rag/clear            — clear RAG store")
    print("  GET  /cache/stats          — cache stats")
    print("  POST /cache/clear          — clear cache")
    print("  GET  /explain/info         — LLM explainer info")
    print("  Listening on http://0.0.0.0:8080\n")
    app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)


if __name__ == '__main__':
    main()

# AI Detector Core

An advanced AI-generated content detection system for images and videos, featuring:

- 🧠 **PRNUFusionNet** — dual-branch neural network that trains on both visual features AND PRNU (Photo Response Non-Uniformity) noise forensics simultaneously
- 🔥 **Grad-CAM heatmaps** — visual highlighting of AI-detected regions, returned as base64 PNG from the REST API
- 📱 **Kotlin Multiplatform mobile app** — iOS + Android targets with Share Extension support (in progress)
- ⚡ **Fast REST API** — Flask server with `/analyze`, `/analyze/frame`, and `/health` endpoints

## Architecture

```
Image Input ──► CNN Branch (256-dim) ──► ┐
                                          ├──► Classifier ──► AI Probability
PRNU Features ──► MLP Branch (32-dim) ──► ┘
```

PRNU features: noise strength, uniformity, blockiness, frequency energy, skewness, kurtosis, high-freq ratio, JPEG quality

## Quick Start

```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Train
python src/train_pytorch.py

# Run server
python server.py

# Analyze an image
curl -X POST http://localhost:8080/analyze -F "image=@your_image.jpg"
```

## API Response

```json
{
  "ai_probability": 0.87,
  "conclusion": "AI-Generated",
  "model_type": "PRNUFusionNet",
  "heatmap_base64": "iVBOR...",
  "heatmap_width": 1024,
  "heatmap_height": 768,
  "prnu_analysis": { "noise_strength": 0.003, "noise_uniformity": 0.91 }
}
```

## Project Structure

```
src/
  prnu_features.py      # 8-dim PRNU feature extractor
  model_prnu.py         # PRNUFusionNet dual-branch model
  detect.py             # Inference + Grad-CAM
  train_pytorch.py      # Training pipeline
  prnu.py               # PRNU forensic algorithms
server.py               # Flask REST API
AIDetectorApp/          # Kotlin Multiplatform (Android + iOS)
```

## Technologies

- Python, PyTorch, Flask
- scikit-image (wavelet denoising), scipy (FFT)
- Kotlin Multiplatform, Compose Multiplatform
- C++ / LibTorch (video inference)

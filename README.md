# Livestock Tracking / Biometric Passport System

This project is a livestock biometric verification prototype built around a FastAPI backend, YOLO-based muzzle detection, and biometric vector matching for cattle passport registration and validation.

## Project purpose

- Register cattle with a digital passport
- Detect muzzle region from uploaded images
- Extract biometric feature vectors
- Compare a live image against the stored biometric signature
- Save registration metadata in JSON files

## Main files

- `app.py` — older FastAPI prototype with in-memory passport registration flow
- `main_app.py` — current biometric passport API with Qdrant-style vector logic
- `core/` — biometric extraction and matcher logic
- `models/` — model assets such as YOLO weights
- `passports_data/` — generated passport JSON and crop outputs
- `run_pipeline.py` — placeholder CLI runner

## Environment

This project depends on Python packages such as:

- `fastapi`
- `uvicorn`
- `opencv-python`
- `ultralytics`
- `torch` / `torchvision`
- `qdrant-client`

## Deployment note

This code is meant for a Python backend environment and is not a standard static frontend app. Vercel can host a FastAPI service only in a constrained server-like deployment setup, and this project currently relies on local model files, CPU/GPU inference, and file-based local storage.

For Vercel, the codebase should be treated as a backend-only deployment candidate or restructured for a container/Render/Railway-style hosting approach.

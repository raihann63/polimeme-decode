"""FastAPI app for the PoliMeme Decode frontend (local development)."""
from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import UnidentifiedImageError

from .model_service import inference_service

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
SUPPORTED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}

app = FastAPI(title="PoliMeme Decode API", version="0.1.0")

origins = os.environ.get(
    "POLIMEME_FRONTEND_ORIGINS",
    "http://localhost:5500,http://127.0.0.1:5500",
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in origins if origin.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {
        "status": "ok",                # API reachable; prediction may initialize lazily
        "model_file_found": inference_service.checkpoint_path.is_file(),
        "model_loaded": inference_service.model_loaded,
        "device": str(inference_service.device),
    }


@app.post("/api/predict")
def predict(file: UploadFile = File(...)):
    if file.content_type not in SUPPORTED_MIME_TYPES:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, or WebP image")

    contents = file.file.read(MAX_UPLOAD_BYTES + 1)
    if not contents:
        raise HTTPException(status_code=400, detail="The image file is empty")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Maximum image upload size is 8 MB")

    try:
        return inference_service.predict_bytes(contents)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc
    except FileNotFoundError as exc:
        LOGGER.exception("Model file is missing")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("The inference pipeline failed")
        raise HTTPException(
            status_code=503,
            detail="Model inference could not be initialized or completed. Check the backend terminal.",
        ) from exc

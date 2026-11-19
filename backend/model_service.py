
"""PoliMeme Decode — Multimodal AI Inference Service.

Architecture:
- ViT: vit_base_patch16_224
- Text Encoder: roberta-base
- Fusion Dimension: 512
- OCR: EasyOCR (Bangla + English)
- Labels: NonPolitical=0, Political=1

Default: Dynamic INT8 quantized checkpoint.
Optional: Original FP32 checkpoint via POLIMEME_MODEL_VARIANT=original.
"""

from __future__ import annotations

import io
import logging
import os
import threading
import time
from pathlib import Path

import easyocr
import numpy as np
import timm
import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision import transforms
from transformers import AutoConfig, AutoModel, AutoTokenizer


LOGGER = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent

ORIGINAL_MODEL_PATH = ROOT / "models" / "best_model.pt"
QUANTIZED_MODEL_PATH = ROOT / "models" / "best_model_quantized.pt"

MODEL_VARIANT = os.environ.get(
    "POLIMEME_MODEL_VARIANT", "quantized"
).strip().lower()

if MODEL_VARIANT not in ("quantized", "original"):
    raise ValueError(
        "POLIMEME_MODEL_VARIANT must be 'quantized' or 'original'"
    )

MODEL_PATH = (
    QUANTIZED_MODEL_PATH
    if MODEL_VARIANT == "quantized"
    else ORIGINAL_MODEL_PATH
)

MODEL_NAME = "roberta-base"
IMAGE_MODEL_NAME = "vit_base_patch16_224"

MAX_LEN = 64
IMG_SIZE = 224
MAX_PIXELS = 20_000_000

Image.MAX_IMAGE_PIXELS = MAX_PIXELS

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])


class MultimodalMemeClassifier(nn.Module):
    """Architecture matched to the original training notebook."""

    def __init__(self, text_config):
        super().__init__()

        self.image_encoder = timm.create_model(
            IMAGE_MODEL_NAME,
            pretrained=False,
            num_classes=0
        )

        img_feat_dim = self.image_encoder.num_features

        self.text_encoder = AutoModel.from_config(text_config)
        text_feat_dim = self.text_encoder.config.hidden_size

        fusion_dim = 512
        dropout = 0.3

        self.img_proj = nn.Sequential(
            nn.Linear(img_feat_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.txt_proj = nn.Sequential(
            nn.Linear(text_feat_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.classifier = nn.Linear(fusion_dim, 2)

        self.confidence_head = nn.Sequential(
            nn.Linear(fusion_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, image, input_ids, attention_mask):

        img_feat = self.image_encoder(image)

        txt_out = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        txt_feat = txt_out.last_hidden_state[:, 0, :]

        img_feat = self.img_proj(img_feat)
        txt_feat = self.txt_proj(txt_feat)

        fused = self.fusion(
            torch.cat([img_feat, txt_feat], dim=1)
        )

        logits = self.classifier(fused)

        conf = self.confidence_head(fused).squeeze(-1)

        return logits, conf


class InferenceService:

    def __init__(self, checkpoint_path: Path = MODEL_PATH):

        self.checkpoint_path = Path(checkpoint_path)

        self._lock = threading.Lock()

        self._model = None
        self._tokenizer = None
        self._reader = None

        self.is_quantized = (
            self.checkpoint_path.name == "best_model_quantized.pt"
        )

        device_setting = os.environ.get(
            "POLIMEME_DEVICE", "cpu"
        ).lower()

        if self.is_quantized and device_setting != "cpu":
            raise RuntimeError(
                "Dynamic INT8 quantized model requires CPU. "
                "Set POLIMEME_DEVICE=cpu."
            )

        if device_setting == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but CUDA is unavailable."
            )

        self.device = torch.device(device_setting)

    @property
    def model_loaded(self) -> bool:
        return (
            self._model is not None
            and self._tokenizer is not None
            and self._reader is not None
        )

    def _prepare_once(self):

        if self.model_loaded:
            return

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Model checkpoint not found: {self.checkpoint_path}"
            )

        LOGGER.info(
            "Initializing PoliMeme model: %s",
            self.checkpoint_path.name
        )

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        config = AutoConfig.from_pretrained(MODEL_NAME)

        model = MultimodalMemeClassifier(config)

        model.eval()

        # IMPORTANT:
        # Quantize the architecture BEFORE loading INT8 weights.
        # The quantization settings reproduce the Colab code.
        if self.is_quantized:

            LOGGER.info(
                "Applying Dynamic INT8 Quantization to Linear layers"
            )

            model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear},
                dtype=torch.qint8
            )

        LOGGER.info("Loading checkpoint weights")

        weights = torch.load(
            str(self.checkpoint_path),
            map_location="cpu",
            weights_only=True,
            mmap=True
        )

        if not isinstance(weights, dict):
            raise TypeError(
                "Expected a PyTorch state_dict checkpoint."
            )

        model.load_state_dict(
            weights,
            strict=True
        )

        del weights

        model.to(self.device)
        model.eval()

        LOGGER.info(
            "Model loaded successfully | Quantized: %s",
            self.is_quantized
        )

        LOGGER.info(
            "Initializing Bangla/English EasyOCR"
        )

        reader = easyocr.Reader(
            ["bn", "en"],
            gpu=(self.device.type == "cuda")
        )

        self._tokenizer = tokenizer
        self._model = model
        self._reader = reader

        LOGGER.info(
            "PoliMeme inference model and OCR reader ready"
        )

    def predict_bytes(self, raw: bytes):

        # Validate uploaded image.
        with Image.open(io.BytesIO(raw)) as source:

            if source.width * source.height > MAX_PIXELS:
                raise ValueError(
                    "Image dimensions are too large"
                )

            image = ImageOps.exif_transpose(
                source
            ).convert("RGB")

            image.load()

        start = time.perf_counter()

        # One inference at a time to limit memory spikes.
        with self._lock:

            self._prepare_once()

            # OCR
            ocr_result = self._reader.readtext(
                np.asarray(image),
                detail=1
            )

            text = " ".join(
                str(item[1])
                for item in ocr_result
            ).strip()

            # Text preprocessing
            encoded = self._tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=MAX_LEN,
                return_tensors="pt"
            )

            # Image preprocessing
            image_tensor = IMAGE_TRANSFORM(
                image
            ).unsqueeze(0).to(self.device)

            input_ids = encoded[
                "input_ids"
            ].to(self.device)

            attention_mask = encoded[
                "attention_mask"
            ].to(self.device)

            # Prediction
            with torch.inference_mode():

                logits, _ = self._model(
                    image_tensor,
                    input_ids,
                    attention_mask
                )

                probs = torch.softmax(
                    logits.float(),
                    dim=1
                )[0].cpu().tolist()

        non_political = float(probs[0])
        political = float(probs[1])

        predicted_political = (
            political >= non_political
        )

        label = (
            "Political"
            if predicted_political
            else "Non-Political"
        )

        confidence = (
            political
            if predicted_political
            else non_political
        )

        return {
            "label": label,
            "confidence": confidence,
            "probabilities": {
                "Political": political,
                "Non-Political": non_political
            },
            "ocr_text": text,
            "latency_ms": round(
                (time.perf_counter() - start) * 1000,
                1
            )
        }


inference_service = InferenceService()

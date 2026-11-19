"""Inference implementation matched to 4_Poli_Meme_Decode_fixed.ipynb.

Training details reproduced:
- timm vit_base_patch16_224 (num_classes=0)
- Hugging Face roberta-base, CLS token representation
- 512-dimensional image/text projections and fusion, dropout 0.3
- 224x224 RGB images with ImageNet normalization
- EasyOCR ['bn', 'en'], RoBERTa max_length=64
- NonPolitical=0 / Political=1

The model state_dict is loaded from models/best_model.pt. No fake predictions.
"""

from __future__ import annotations

import io
import logging
import os
import threading
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torch import nn
from torchvision import transforms
from transformers import AutoConfig, AutoModel, AutoTokenizer
import easyocr
import timm

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "best_model.pt"
MODEL_NAME = "roberta-base"
IMAGE_MODEL_NAME = "vit_base_patch16_224"
MAX_LEN = 64
IMG_SIZE = 224
MAX_PIXELS = 20_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class MultimodalMemeClassifier(nn.Module):
    """Exact structural counterpart of the user's notebook model."""

    def __init__(self, text_config):
        super().__init__()
        self.image_encoder = timm.create_model(IMAGE_MODEL_NAME, pretrained=False, num_classes=0)
        img_feat_dim = self.image_encoder.num_features
        # Avoid downloading the original pretraining weights again: the full
        # trained state_dict includes the image and text encoders already.
        self.text_encoder = AutoModel.from_config(text_config)
        text_feat_dim = self.text_encoder.config.hidden_size
        fusion_dim, dropout = 512, 0.3
        self.img_proj = nn.Sequential(nn.Linear(img_feat_dim, fusion_dim), nn.ReLU(), nn.Dropout(dropout))
        self.txt_proj = nn.Sequential(nn.Linear(text_feat_dim, fusion_dim), nn.ReLU(), nn.Dropout(dropout))
        self.fusion = nn.Sequential(nn.Linear(fusion_dim * 2, fusion_dim), nn.ReLU(), nn.Dropout(dropout))
        self.classifier = nn.Linear(fusion_dim, 2)
        self.confidence_head = nn.Sequential(nn.Linear(fusion_dim, 1), nn.Sigmoid())

    def forward(self, image, input_ids, attention_mask):
        img_feat = self.image_encoder(image)
        txt_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        txt_feat = txt_out.last_hidden_state[:, 0, :]
        img_feat = self.img_proj(img_feat)
        txt_feat = self.txt_proj(txt_feat)
        fused = self.fusion(torch.cat([img_feat, txt_feat], dim=1))
        logits = self.classifier(fused)
        conf = self.confidence_head(fused).squeeze(-1)
        return logits, conf


class InferenceService:
    def __init__(self, checkpoint_path: Path = MODEL_PATH):
        self.checkpoint_path = checkpoint_path
        self._lock = threading.Lock()
        self._model = None
        self._tokenizer = None
        self._reader = None
        device_setting = os.environ.get("POLIMEME_DEVICE", "cpu").lower()
        if device_setting == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("POLIMEME_DEVICE=cuda requested but CUDA is not available")
        self.device = torch.device(device_setting)

    @property
    def model_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None and self._reader is not None

    def _prepare_once(self):
        if self.model_loaded:
            return
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Model checkpoint was not found: {self.checkpoint_path}")

        LOGGER.info("Initializing PoliMeme model on %s", self.device)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        config = AutoConfig.from_pretrained(MODEL_NAME)
        model = MultimodalMemeClassifier(config)
        # mmap avoids an extra eager 807 MB copy when supported by PyTorch.
        weights = torch.load(str(self.checkpoint_path), map_location="cpu", weights_only=True, mmap=True)
        if not isinstance(weights, dict):
            raise TypeError("Expected a state_dict checkpoint saved with torch.save(model.state_dict(), ...)")
        model.load_state_dict(weights, strict=True)
        del weights
        model.to(self.device)
        model.eval()

        LOGGER.info("Initializing Bangla/English EasyOCR reader (first run may download OCR weights)")
        reader = easyocr.Reader(['bn', 'en'], gpu=(self.device.type == 'cuda'))
        self._tokenizer = tokenizer
        self._model = model
        self._reader = reader
        LOGGER.info("PoliMeme inference model and OCR reader ready")

    def predict_bytes(self, raw: bytes):
        # PIL validates actual image bytes; don't trust a filename extension alone.
        with Image.open(io.BytesIO(raw)) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
            image.load()
        if image.width * image.height > MAX_PIXELS:
            raise ValueError("Image dimensions are too large")

        start = time.perf_counter()
        # Serialize first-time initialization and prediction on CPU to avoid
        # simultaneous 807 MB checkpoint loads / memory spikes.
        with self._lock:
            self._prepare_once()
            ocr_result = self._reader.readtext(np.asarray(image), detail=1)
            text = ' '.join(str(item[1]) for item in ocr_result).strip()
            encoded = self._tokenizer(
                text, truncation=True, padding='max_length', max_length=MAX_LEN,
                return_tensors='pt',
            )
            image_tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(self.device)
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            with torch.inference_mode():
                logits, _ = self._model(image_tensor, input_ids, attention_mask)
                probs = torch.softmax(logits.float(), dim=1)[0].cpu().tolist()

        non_political, political = (float(probs[0]), float(probs[1]))
        predicted_political = political >= non_political
        label = "Political" if predicted_political else "Non-Political"
        # Softmax confidence is consistent with the displayed class probabilities.
        confidence = political if predicted_political else non_political
        return {
            "label": label,
            "confidence": confidence,
            "probabilities": {"Political": political, "Non-Political": non_political},
            "ocr_text": text,
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
        }


inference_service = InferenceService()

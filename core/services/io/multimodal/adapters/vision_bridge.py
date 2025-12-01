# core/services/multimodal/vision_bridge.py
"""
VisionBridge V2.1 — Ponte para modelos de Visão (embeddings e tags)
Objetivos:
    • Sanitização completa (formatos, tamanho, corrupção)
    • Auditoria e metadados (UUID, timestamps, preview simbólico)
    • Logging estruturado JSON (para PerceptionAPI V4.2)
    • PRAG & PCVS friendly (hashes de embedding, rollback)
    • Testes unitários fáceis de integrar
"""

from __future__ import annotations
import logging
import uuid
from time import time
from typing import Any, Dict, List, Tuple, Optional
from io import BytesIO

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

print("Pillow OK:", Image)
print("Torch OK:", torch.__version__)
print("Torchvision OK:", T)

from core.orchestration.mch import setup_logger

# ---------------------------------------------------------------
# Logger
# ---------------------------------------------------------------
logger = setup_logger("VisionBridge")

# ---------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------
DEFAULT_EMBEDDING_DIM = 768
SUPPORTED_FORMATS = ["jpg", "jpeg", "png"]
MIN_FILE_SIZE = 100  # bytes
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB

# ---------------------------------------------------------------
# VisionBridge
# ---------------------------------------------------------------
class VisionBridge:
    def __init__(
        self,
        vision_encoder: Optional[torch.nn.Module] = None,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        device: str = "cpu"
    ):
        """
        :param vision_encoder: modelo plugável (CLIP, ONNX, etc.)
        :param embedding_dim: tamanho do embedding final
        :param device: 'cpu' ou 'cuda'
        """
        self.encoder = vision_encoder.to(device) if vision_encoder else None
        self.device = device
        self.embedding_dim = embedding_dim

        # Transforms padrão (imagem → tensor normalizado)
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

        logger.info(f"VisionBridge V2.1 initialized (device={device})")

    # ------------------------------------------------------------------
    # Validação e sanitização de imagem
    # ------------------------------------------------------------------
    def _load_image(self, image_data: bytes, image_format: str) -> Image.Image:
        fmt = image_format.lower()
        if fmt not in SUPPORTED_FORMATS:
            logger.error(f"Formato não suportado: {fmt}")
            raise ValueError("Unsupported image format.")

        if not (MIN_FILE_SIZE <= len(image_data) <= MAX_FILE_SIZE):
            logger.error(f"Tamanho inválido: {len(image_data)} bytes")
            raise ValueError("Invalid image file size.")

        try:
            img = Image.open(BytesIO(image_data)).convert("RGB")
            # Validação de magic bytes (JPEG, PNG)
            header = image_data[:8]
            if fmt in ["jpg", "jpeg"] and header[:2] != b'\xff\xd8':
                raise ValueError("JPEG header mismatch")
            if fmt == "png" and header[:4] != b'\x89PNG':
                raise ValueError("PNG header mismatch")
            return img
        except Exception as e:
            logger.error(f"Falha ao decodificar a imagem: {e}")
            raise ValueError("Failed to decode image file.")

    # ------------------------------------------------------------------
    # Geração de embedding
    # ------------------------------------------------------------------
    def _encode(self, tensor: torch.Tensor) -> np.ndarray:
        if self.encoder is None:
            # Fallback: embedding simulado determinístico
            rng = np.random.default_rng(int(tensor.sum().item()) % 999999)
            embedding = rng.random(self.embedding_dim).astype(np.float32)
        else:
            with torch.no_grad():
                features = self.encoder(tensor.to(self.device))
                if isinstance(features, tuple):
                    features = features[0]
                embedding = features.flatten().cpu().numpy().astype(np.float32)
            # Ajustar para embedding_dim
            if embedding.shape[0] > self.embedding_dim:
                embedding = embedding[:self.embedding_dim]
            elif embedding.shape[0] < self.embedding_dim:
                pad = self.embedding_dim - embedding.shape[0]
                embedding = np.pad(embedding, (0, pad))
        return embedding

    # ------------------------------------------------------------------
    # Geração de tags simbólicas
    # ------------------------------------------------------------------
    def _generate_tags(self, img: Image.Image) -> List[str]:
        """
        Placeholder para:
            • BLIP caption
            • OCR/Tesseract
            • Classificadores ImageNet/CLIP
        """
        # Tags simuladas
        tags = ["objeto_visual", "estrutura", "conteudo_simbolico"]
        return [t.lower().replace(" ", "_") for t in tags]

    # ------------------------------------------------------------------
    # Pipeline público
    # ------------------------------------------------------------------
    def process_image(
        self, image_data: bytes, image_format: str
    ) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """
        :return: embedding, tags, metadados
        """
        start_ts = time()
        image_id = str(uuid.uuid4())
        error_flag = False
        preview = None

        try:
            img = self._load_image(image_data, image_format)
            tensor = self.transform(img).unsqueeze(0)  # shape: (1,3,224,224)
            embedding = self._encode(tensor)
            tags = self._generate_tags(img)
            preview = "[IMAGE DATA]"  # Placeholder para preview
        except Exception as e:
            logger.error(f"Erro ao processar imagem {image_id}: {e}")
            error_flag = True
            embedding = np.zeros(self.embedding_dim, dtype=np.float32)
            tags = []

        end_ts = time()
        latency_ms = int((end_ts - start_ts) * 1000)

        # Metadados estruturados
        metadata = {
            "image_id": image_id,
            "error_flag": error_flag,
            "received_ts": start_ts,
            "processed_ts": end_ts,
            "latency_ms": latency_ms,
            "tags": tags,
            "embedding_dim": self.embedding_dim,
            "preview": preview,
            "hash_embedding": hash(embedding.tobytes())
        }

        logger.info(f"Image processed. ID: {image_id}, Tags: {tags}, Latency: {latency_ms}ms")
        return embedding, tags, metadata

# core/services/perception.py (Versão Final com Integração Multimodal)
"""
PerceptionAPI – V4.3 (Suporte Multimodal Completo)
"""
import numpy as np
import time
import logging
import uuid
from typing import Any, Dict, Optional, Tuple

from .security import Security

# Tentativa de importar os Bridges reais
try:
    from .multimodal.ovi_service.adapters.audio_bridge import AudioBridge
    from .multimodal.ovi_service.adapters.vision_bridge import VisionBridge
except ImportError:
    # Fallback Mocks com log de aviso
    class MockAudioBridge:
        def __init__(self):
            logging.warning("[PerceptionAPI] AudioBridge em modo simulado (Mock ativo).")
        def transcribe(self, data, fmt):
            return "Simulação de transcrição de áudio.", 0.9

    class MockVisionBridge:
        def __init__(self):
            logging.warning("[PerceptionAPI] VisionBridge em modo simulado (Mock ativo).")
        def process_image(self, data, fmt):
            return np.zeros(768), ["mock_tag"]

    AudioBridge, VisionBridge = MockAudioBridge, MockVisionBridge

logger = logging.getLogger("PerceptionAPI")
MAX_INPUT_LENGTH = 4096


class PerceptionAPI:
    """
    Serviço de Percepção.
    Entrada → Sanitização → Hash → Metadados
    """

    def __init__(self, security_service: Optional[Any] = None):
        self.sec = security_service or Security()
        self.audio_bridge = AudioBridge()
        self.vision_bridge = VisionBridge()
        logger.info("PerceptionAPI inicializado (Suporte Multimodal Ativo).")

    # ---------------------------------------------------------------------
    # Funções Auxiliares
    # ---------------------------------------------------------------------

    def get_raw_input(self, input_data: Any, source_type: str) -> Any:
        """
        Captura e normaliza o input multimodal.
        - Texto → str
        - Áudio/Imagem → bytes
        """
        if source_type == "text":
            if not isinstance(input_data, str):
                raise TypeError("get_raw_input: esperado str para texto.")
            return input_data.strip()

        elif source_type in ["audio", "image"]:
            if not isinstance(input_data, bytes):
                raise TypeError(f"get_raw_input: esperado bytes para {source_type}.")
            return input_data

        else:
            raise ValueError(f"get_raw_input: tipo desconhecido {source_type}")

    def validate_and_sanitize(self, raw_input: Any, source_type: str) -> Tuple[Any, bool]:
        """
        Valida e sanitiza o input.
        Retorna (input_sanitizado, status_sanitizado).
        """
        if source_type == "text":
            processed = self.sec.sanitize_input(raw_input)
            if len(processed) > MAX_INPUT_LENGTH:
                return processed[:MAX_INPUT_LENGTH], False
            return processed, True

        elif source_type in ["audio", "image"]:
            # Para multimodal, a validação é delegada aos Bridges
            return raw_input, True

        return raw_input, False

    # ---------------------------------------------------------------------
    # PERCEPÇÃO PRINCIPAL
    # ---------------------------------------------------------------------

    def perceive(
        self,
        input_data: Any,
        source_type: str = "text",
        file_format: Optional[str] = None,
        context_meta: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Dict[str, Any]]:

        is_sanitized = True
        extra_multimodal_data: Dict[str, Any] = {}

        # 1. Captura e sanitização inicial
        raw_input = self.get_raw_input(input_data, source_type)
        processed_input, is_sanitized = self.validate_and_sanitize(raw_input, source_type)

        # ============================
        # 1. Percepção de TEXTO
        # ============================
        if source_type == "text":
            original_length = len(raw_input)
            processed_text = processed_input
            processed_length = len(processed_text)
            input_hash = self.sec.hash_state(processed_text)

            logger.info(
                f"[Perception] Texto recebido. len={original_length} → {processed_length}, hash={input_hash[:12]}..."
            )

        # ============================
        # 2. Percepção MULTIMODAL
        # ============================
        elif source_type in ["audio", "image"]:
            try:
                if source_type == "audio":
                    transcribed_text, confidence = self.audio_bridge.transcribe(processed_input, file_format or "wav")
                    processed_text = f"[ÁUDIO TRANSCRITO]: {transcribed_text}"
                    extra_multimodal_data["stt_confidence"] = float(confidence)
                    extra_multimodal_data["transcribed_text"] = transcribed_text

                elif source_type == "image":
                    embedding_vector, tags = self.vision_bridge.process_image(processed_input, file_format or "jpg")
                    if isinstance(embedding_vector, np.ndarray):
                        embedding_vector = embedding_vector.tolist()
                    processed_text = f"[IMAGEM DESCRITA]: Tags: {', '.join(tags)}"
                    extra_multimodal_data["image_tags"] = tags
                    extra_multimodal_data["vision_embedding"] = embedding_vector

                original_length = len(processed_input)
                processed_length = len(processed_text)
                input_hash = self.sec.hash_state(processed_text)

                logger.info(
                    f"[Perception] Multimodal '{source_type}' processado. hash={input_hash[:12]}..."
                )

            except ValueError as e:
                is_sanitized = False
                processed_text = f"[ERROR: Falha na validação binária de {source_type}]"
                input_hash = self.sec.hash_state(processed_text)
                original_length = len(processed_input)
                processed_length = len(processed_text)
                logger.error(f"[Perception] Falha na validação binária: {e}")

        # ============================
        # 3. Tipo desconhecido
        # ============================
        else:
            raise ValueError(f"Unknown source type: {source_type}")

        # ============================
        # 4. Metadados finais
        # ============================
        meta = {
            "source_type": source_type,
            "timestamp": time.time(),
            "uuid": str(uuid.uuid4()),
            "processed_hash": input_hash,
            "sanitized": is_sanitized,
            "context": context_meta or {},
            "original_length": original_length,
            "processed_length": processed_length,
        }

        meta["context"].update(extra_multimodal_data)

        return processed_text, meta
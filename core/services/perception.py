# core/services/perception.py (Versão com Integração Multimodal)
"""
PerceptionAPI – V4.3 (Com Suporte Arquitetural a Multimodal)
"""
import numpy as np
import time
import logging
import uuid # Adicionado para futuros metadados
from typing import Any, Dict, Optional, Tuple

from .security import Security

# ... (imports existentes) ...
# Simular importação dos novos Bridges
try:
    # Assumindo nova estrutura core/services/multimodal/
    from .multimodal.audio_bridge import AudioBridge 
    from .multimodal.vision_bridge import VisionBridge
except ImportError:
    # Fallback Mocks (mantido para execução local)
    class MockAudioBridge:
        def transcribe(self, data, fmt): return "Simulação de transcrição de áudio.", 0.9
    class MockVisionBridge:
        def process_image(self, data, fmt): return np.zeros(768), ["mock_tag"]
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
        self.audio_bridge = AudioBridge() # <-- NOVO
        self.vision_bridge = VisionBridge() # <-- NOVO
        logger.info("PerceptionAPI inicializado (Suporte Multimodal Ativo).")

    # ---------------------------------------------------------------------
    # PERCEPÇÃO PRINCIPAL
    # ---------------------------------------------------------------------

    def perceive(
        self,
        input_data: Any,
        source_type: str = "text",
        file_format: Optional[str] = None, # <-- NOVO: Auxilia na validação binária
        context_meta: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Dict[str, Any]]:

        is_sanitized = True
        extra_multimodal_data: Dict[str, Any] = {} # Para embeddings, confiança, etc.
        
        # ============================
        # 1. Percepção de TEXTO
        # ============================
        if source_type == "text":
            if not isinstance(input_data, str):
                raise TypeError("PerceptionAPI: input_data deve ser str quando source_type='text'")

            raw_text = input_data.strip()
            # ... (Restante da lógica de sanitização de texto) ...
            original_length = len(raw_text)
            processed_text = self.sec.sanitize_input(raw_text)

            if len(processed_text) > MAX_INPUT_LENGTH:
                processed_text = processed_text[:MAX_INPUT_LENGTH]
                is_sanitized = False 
            
            processed_length = len(processed_text)
            input_hash = self.sec.hash_state(processed_text)

            logger.info(
                f"[Perception] Texto recebido. len={original_length} → {processed_length}, hash={input_hash[:12]}..."
            )

        # ============================
        # 2. Percepção MULTIMODAL
        # ============================
        elif source_type in ["audio", "image"]:
            
            if not isinstance(input_data, bytes):
                raise TypeError(f"PerceptionAPI: input_data deve ser bytes para source_type='{source_type}'")

            try:
                if source_type == "audio":
                    # 🔒 Validação Binária e Sanitização Real (via Bridge)
                    transcribed_text, confidence = self.audio_bridge.transcribe(input_data, file_format or "wav")
                    
                    # O texto processado é o resultado do STT
                    processed_text = f"[ÁUDIO TRASNCRITO]: {transcribed_text}"
                    extra_multimodal_data["stt_confidence"] = float(confidence)
                    extra_multimodal_data["transcribed_text"] = transcribed_text # Original transcription
                    
                elif source_type == "image":
                    # 🔒 Validação Binária e Sanitização Real (via Bridge)
                    embedding_vector, tags = self.vision_bridge.process_image(input_data, file_format or "jpg")
                    
                    # O texto processado é uma descrição/tags para o OL
                    processed_text = f"[IMAGEM DESCRITA]: Tags: {', '.join(tags)}"
                    extra_multimodal_data["image_tags"] = tags
                    # É vital persistir o vetor no Hippocampus, não apenas o texto:
                    extra_multimodal_data["vision_embedding"] = embedding_vector.tolist() 

                original_length = len(input_data)
                processed_length = len(processed_text)
                
                # O Hash é feito no texto processado para fins de PRAG/Governança
                input_hash = self.sec.hash_state(processed_text)

                logger.info(
                    f"[Perception] Multimodal '{source_type}' processado. Tags/Confiança anexadas. hash={input_hash[:12]}..."
                )

            except ValueError as e:
                # Captura erros de formato/integridade do arquivo binário (Validação)
                is_sanitized = False # Falha na sanitização/validação binária
                processed_text = f"[ERROR: Falha na validação binária de {source_type}]"
                input_hash = self.sec.hash_state(processed_text)
                original_length = len(input_data)
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
            "processed_hash": input_hash,
            "sanitized": is_sanitized, 
            "context": context_meta or {},
            "original_length": original_length,
            "processed_length": processed_length,
        }
        
        # Adiciona dados multimodais ao contexto
        meta["context"].update(extra_multimodal_data)

        return processed_text, meta
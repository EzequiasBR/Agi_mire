# core/services/nlp_bridge.py
"""
NLPBridge - Módulo de Comunicação com Modelos NLP Externos.
Atua como um adaptador de API para serviços de NLP/LLM, garantindo que o MCH
e o OL possam se comunicar de forma agnóstica a modelos.
"""
from __future__ import annotations
import time
import logging
from typing import Any, Dict, List, Optional
from threading import Lock
import numpy as np

# Importar setup_logger da mesma pasta services
try:
    from .utils import setup_logger
except ImportError:
    # Fallback simples
    def setup_logger(name):
        logger = logging.getLogger(name)
        if not logger.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s"))
            logger.addHandler(ch)
        logger.setLevel(logging.INFO)
        return logger

logger = setup_logger("NLPBridgeService")

class NLPBridge:
    def __init__(self, 
                 model_endpoint: str = "http://nlp_provider/v1/model",
                 embedding_dim: int = 768,
                 default_model: str = "Mihe-Llama-7B"):
        
        self._lock = Lock()
        self.model_endpoint = model_endpoint
        self.embedding_dim = embedding_dim
        self.default_model = default_model
        
        # Simula o estado de cache/conexão (para PCVS)
        self.last_successful_call_ts: float = 0.0

    # --------------------------------
    # Funções de Comunicação (Mock para ambiente sem serviço real)
    # --------------------------------

    def _call_external_service(self, payload: Dict[str, Any], endpoint_suffix: str = "") -> Dict[str, Any]:
        """
        Mock de uma chamada HTTP/gRPC para o serviço NLP externo.
        Em um ambiente real, esta função faria o request e tratamento de erros.
        """
        # Simula latência de rede e processamento
        time.sleep(0.005) 
        
        # Base de resposta
        response = {
            "model_id": payload.get("model", self.default_model),
            "latency_s": 0.005 + np.random.rand() * 0.05,
            "success": True,
        }

        if endpoint_suffix == "/embed":
            # Retorna um vetor de embedding aleatório (para simulação)
            response["vector"] = np.random.randn(self.embedding_dim).tolist()
        elif endpoint_suffix == "/generate":
            # Retorna uma string de texto gerada
            response["text"] = f"Generated text based on prompt: '{payload.get('prompt', '')[:50]}...'"
        
        self.last_successful_call_ts = time.time()
        return response

    # --------------------------------
    # Interfaces Públicas
    # --------------------------------

    def get_embedding_dimension(self) -> int:
        """Retorna a dimensão do vetor de embedding configurada."""
        return self.embedding_dim

    def get_embedding(self, 
                      text: str, 
                      context: Optional[str] = None, 
                      model: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Converte texto e contexto em um vetor de embedding. Usado pelo OL.
        """
        try:
            payload = {
                "text": text,
                "context": context,
                "model": model or self.default_model
            }
            result = self._call_external_service(payload, endpoint_suffix="/embed")
            
            if result.get("success") and "vector" in result:
                # O OL espera um np.ndarray
                return np.asarray(result["vector"], dtype=float)
            
            logger.error("Embedding generation failed for model %s.", payload["model"])
            return None
        except Exception as e:
            logger.exception("NLPBridge: Failed during get_embedding call.")
            return None

    def generate_response(self, 
                          prompt: str, 
                          context: Optional[str] = None, 
                          max_tokens: int = 100) -> Optional[str]:
        """
        Gera uma resposta textual baseada em um prompt e contexto.
        Pode ser usado para resumo, diálogo ou formatação de saída.
        """
        try:
            payload = {
                "prompt": prompt,
                "context": context,
                "max_tokens": max_tokens,
                "model": self.default_model
            }
            result = self._call_external_service(payload, endpoint_suffix="/generate")
            
            if result.get("success") and "text" in result:
                return str(result["text"])
            
            logger.error("Text generation failed.")
            return None
        except Exception as e:
            logger.exception("NLPBridge: Failed during generate_response call.")
            return None

    # --------------------------------
    # Snapshot (Compatibilidade com PCVS)
    # --------------------------------
    def snapshot_state(self) -> Dict[str, Any]:
        """Retorna o estado completo para persistência PCVS."""
        with self._lock:
            return {
                "model_endpoint": self.model_endpoint,
                "embedding_dim": self.embedding_dim,
                "default_model": self.default_model,
                "last_successful_call_ts": self.last_successful_call_ts
            }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Carrega o estado a partir de um snapshot PCVS."""
        with self._lock:
            if not state:
                return
            
            self.model_endpoint = state.get("model_endpoint", self.model_endpoint)
            self.embedding_dim = int(state.get("embedding_dim", self.embedding_dim))
            self.default_model = state.get("default_model", self.default_model)
            self.last_successful_call_ts = float(state.get("last_successful_call_ts", 0.0))
            
            logger.info("NLPBridge state loaded. Dim: %d", self.embedding_dim)

# --------------------------------
# Teste Rápido
# --------------------------------
if __name__ == "__main__":
    import json
    
    # Criamos um bridge com 128 dimensões para simular um modelo menor
    bridge = NLPBridge(embedding_dim=128)
    
    test_text = "O cisne negro é um evento de baixa probabilidade e alto impacto."
    
    print("--- 1. Teste de Geração de Embedding ---")
    embedding = bridge.get_embedding(test_text, context="Teoria da Probabilidade")
    
    if embedding is not None:
        print(f"Embedding retornado: {type(embedding)}")
        print(f"Dimensão: {embedding.shape}")
        assert embedding.shape == (128,)
    else:
        print("Falha ao obter embedding.")

    print("\n--- 2. Teste de Geração de Resposta ---")
    response = bridge.generate_response("O que é o efeito Mandela?", max_tokens=50)
    print("Resposta Gerada:", response)
    
    print("\nParâmetros Snapshot:", json.dumps(bridge.snapshot_state(), indent=2))
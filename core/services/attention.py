# core/services/attention.py
"""
Attention - Módulo de Mecanismo de Atenção e Priorização.
Calcula o Nível de Atenção (A) do sistema com base na Divergência (D), 
Confiança (C) e Volatilidade (Vol) do Analytics. 
A é usado para modular a profundidade do processamento (ex: busca no Hippocampus).
"""
from __future__ import annotations
import time
import logging
from typing import Any, Dict, Optional
from threading import Lock
import numpy as np

# Importar setup_logger da mesma pasta services
try:
    from .utils import setup_logger
except ImportError:
    def setup_logger(name):
        logger = logging.getLogger(name)
        if not logger.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s"))
            logger.addHandler(ch)
        logger.setLevel(logging.INFO)
        return logger

logger = setup_logger("AttentionService")

class Attention:
    def __init__(self,
                 w_D: float = 0.4,       # Peso para Divergência (D)
                 w_volatility: float = 0.3, # Peso para Volatilidade do Analytics
                 w_uncertainty: float = 0.3, # Peso para Incerteza (1 - C)
                 base_prioritization: float = 0.5): # Atenção base
        
        self._lock = Lock()
        
        # Pesos (ajustáveis se Adaptation quiser)
        self.weights = {
            "D": w_D,
            "volatility": w_volatility,
            "uncertainty": w_uncertainty
        }
        self.base_prioritization = base_prioritization
        self.last_attention_level: float = base_prioritization

    # --------------------------------
    # Lógica de Pontuação
    # --------------------------------
    
    def _sigmoid_normalize(self, value: float) -> float:
        """Normaliza um valor para a faixa (0, 1) usando a função Sigmoid."""
        # Multiplicamos por um fator (ex: 5) para dar mais inclinação e garantir 
        # que valores altos de entrada gerem valores de atenção próximos a 1.
        return 1.0 / (1.0 + np.exp(-5.0 * value))

    def calculate_attention(self, 
                            D: float, 
                            C: float, 
                            analytics_report: Dict[str, Any]) -> float:
        """
        Calcula o Nível de Atenção (A) com base nos fatores de criticidade.
        Retorna um float A onde 0 <= A <= 1.
        """
        
        with self._lock:
            # 1. Obter Volatilidade
            volatility_D = analytics_report.get("metrics_volatility", {}).get("volatility_index_D", 0.0)
            
            # 2. Calcular Incerteza (1 - C)
            uncertainty = 1.0 - C
            
            # 3. Combinação Ponderada (Input Score)
            # Normalizamos os inputs D, Vol e Uncertainty para evitar que um domine os outros.
            # Assumimos que D e C estão entre 0 e 1. Volatilidade pode ser maior.
            
            input_score = (
                self.weights["D"] * D +
                self.weights["uncertainty"] * uncertainty +
                self.weights["volatility"] * volatility_D 
            )
            
            # 4. Normalização Final para a faixa [0, 1]
            # Usamos uma normalização simples para garantir que A seja um float válido.
            # O Max(0, ...) garante que o nível base é o mínimo.
            A = max(self.base_prioritization, input_score)
            
            # Normalização final para a faixa [0, 1]
            A = np.clip(A, 0.0, 1.0)
            
            self.last_attention_level = A
            logger.info("Attention Level (A) calculated: %.4f", A)
            
            return float(A)

    # --------------------------------
    # Snapshot (Compatibilidade com PCVS)
    # --------------------------------
    def snapshot_state(self) -> Dict[str, Any]:
        """Retorna o estado completo para persistência PCVS."""
        with self._lock:
            return {
                "weights": self.weights,
                "base_prioritization": self.base_prioritization,
                "last_attention_level": self.last_attention_level,
            }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Carrega o estado a partir de um snapshot PCVS."""
        with self._lock:
            if not state:
                return
            
            self.weights = state.get("weights", self.weights)
            self.base_prioritization = float(state.get("base_prioritization", self.base_prioritization))
            self.last_attention_level = float(state.get("last_attention_level", self.base_prioritization))
            
            logger.info("Attention state loaded. Last A: %.4f", self.last_attention_level)

# --------------------------------
# Teste Rápido
# --------------------------------
if __name__ == "__main__":
    import json
    
    attention = Attention(w_D=0.5, w_volatility=0.2, w_uncertainty=0.3, base_prioritization=0.2)
    
    # Mock Analytics Report 1: ESTÁVEL E CONFIANTE (Baixa Atenção)
    stable_analytics = {
        "metrics_volatility": {"volatility_index_D": 0.01}, # Baixa Volatilidade
    }

    print("--- 1. Cenário ESTÁVEL (Baixa Atenção) ---")
    A1 = attention.calculate_attention(D=0.1, C=0.9, analytics_report=stable_analytics)
    # A1 deve ser próximo ao base_prioritization (0.2)
    print(f"D=0.1, C=0.9, Vol=0.01 -> Atenção A: {A1:.4f}")
    assert A1 < 0.3
    
    # Mock Analytics Report 2: INSTÁVEL E INCERTO (Alta Atenção)
    unstable_analytics = {
        "metrics_volatility": {"volatility_index_D": 0.20}, # Alta Volatilidade
    }
    
    print("\n--- 2. Cenário CRÍTICO (Alta Atenção) ---")
    A2 = attention.calculate_attention(D=0.8, C=0.4, analytics_report=unstable_analytics)
    # A2 deve ser próximo a 1.0
    print(f"D=0.8, C=0.4, Vol=0.20 -> Atenção A: {A2:.4f}")
    assert A2 > 0.8
    
    print("\nParâmetros Snapshot:", json.dumps(attention.snapshot_state(), indent=2))
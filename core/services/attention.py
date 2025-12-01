"""
Attention - Módulo de Mecanismo de Atenção e Priorização.
Calcula o Nível de Atenção (A) do sistema com base na Divergência (D), 
Confiança (C) e Volatilidade (Vol) do Analytics. 
"""
from __future__ import annotations
import time
import logging
from typing import Any, Dict, Optional, TYPE_CHECKING
from threading import Lock
import numpy as np
import asyncio # Necessário para rodar o bloco de teste e a função async

# Mock de dependências para o código ser executável (Ajuste 3)
class SystemEvents:
    ATTENTION_UPDATED = "ATTENTION_UPDATED"
if TYPE_CHECKING:
    class ControlBus:
        async def publish(self, event: str, payload: dict[str, Any], source_module: str) -> None: ...
        def subscribe(self, event: str, handler: Any) -> None: ...
# Fim do Mock de dependências

# Importar setup_logger
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
                 control_bus: 'ControlBus', # Injeção de dependência
                 w_D: float = 0.4,          # Peso para Divergência (D)
                 w_volatility: float = 0.3, # Peso para Volatilidade do Analytics
                 w_uncertainty: float = 0.3, # Peso para Incerteza (1 - C)
                 base_prioritization: float = 0.5): # Atenção base
        
        self._lock = Lock()
        self._control_bus = control_bus
        
        self.weights = {
            "D": w_D,
            "volatility": w_volatility,
            "uncertainty": w_uncertainty
        }
        self.base_prioritization = base_prioritization
        self.last_attention_level: float = base_prioritization

        # Ajuste 2: Campos para Telemetria de Entrada
        self.last_inputs: Dict[str, float] = {"D": 0.0, "C": 1.0, "volatility_D": 0.0}

    # --------------------------------
    # Lógica de Pontuação
    # --------------------------------
    
    def _sigmoid_normalize(self, value: float) -> float:
        """Normaliza um valor para a faixa (0, 1) usando a função Sigmoid."""
        # Fator de inclinação de 5.0
        return 1.0 / (1.0 + np.exp(-5.0 * value))

    async def calculate_attention(self, # Função alterada para ASYNC
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
            input_score = (
                self.weights["D"] * D +
                self.weights["uncertainty"] * uncertainty +
                self.weights["volatility"] * volatility_D 
            )
            
            # 4. Normalização Final para a faixa [0, 1]
            
            # Ajuste 1: Aplicar Sigmoid ao input_score antes de combinar com a base
            normalized_score = self._sigmoid_normalize(input_score)

            # Combinação: A base é o mínimo, o score suavizado eleva
            A = max(self.base_prioritization, normalized_score)
            
            # Normalização final para a faixa [0, 1]
            A = np.clip(A, 0.0, 1.0)
            
            self.last_attention_level = A
            self.last_inputs = {"D": D, "C": C, "volatility_D": volatility_D} # Ajuste 2
            
            logger.info("Attention Level (A) calculated: %.4f (Base: %.2f, Score: %.4f)", 
                        A, self.base_prioritization, normalized_score)

            # Ajuste 3: Publicar evento no ControlBus
            await self._control_bus.publish(
                SystemEvents.ATTENTION_UPDATED, 
                {"A": float(A), "timestamp": time.time()}, 
                source_module="AttentionService"
            )
            
            return float(A)

    # --------------------------------
    # Snapshot (Compatibilidade com PCVS)
    # --------------------------------
    def snapshot_state(self) -> Dict[str, Any]:
        """Retorna o estado completo para persistência PCVS."""
        with self._lock:
            # Ajuste 2: Incluir valores de entrada no snapshot
            return {
                "weights": self.weights,
                "base_prioritization": self.base_prioritization,
                "last_attention_level": self.last_attention_level,
                "last_inputs": self.last_inputs, # Inclusão da telemetria de entrada
            }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Carrega o estado a partir de um snapshot PCVS."""
        with self._lock:
            if not state:
                return
            
            self.weights = state.get("weights", self.weights)
            self.base_prioritization = float(state.get("base_prioritization", self.base_prioritization))
            self.last_attention_level = float(state.get("last_attention_level", self.base_prioritization))
            # Ajuste 2: Carregar a telemetria de entrada
            self.last_inputs = state.get("last_inputs", {"D": 0.0, "C": 1.0, "volatility_D": 0.0}) 
            
            logger.info("Attention state loaded. Last A: %.4f", self.last_attention_level)

# --------------------------------
# Teste Rápido (Ajustado para Assincronia)
# --------------------------------
if __name__ == "__main__":
    import json
    
    # Mock do ControlBus
    class MockControlBus:
        def __init__(self):
            self.events_published = []
        async def publish(self, event: str, payload: dict[str, Any], source_module: str) -> None:
            print(f"MockControlBus: Publicado {event} de {source_module} (A={payload['A']:.4f})")
            self.events_published.append({"event": event, "payload": payload, "source": source_module})
        def subscribe(self, event: str, handler: Any) -> None:
            pass

    async def main():
        control_bus_mock = MockControlBus()
        attention = Attention(
            control_bus=control_bus_mock, # Injeção
            w_D=0.5, w_volatility=0.2, w_uncertainty=0.3, base_prioritization=0.2
        )
        
        # Mock Analytics Report 1: ESTÁVEL E CONFIANTE (Baixa Atenção)
        stable_analytics = {
            "metrics_volatility": {"volatility_index_D": 0.01}, # Baixa Volatilidade
        }

        print("--- 1. Cenário ESTÁVEL (Baixa Atenção) ---")
        A1 = await attention.calculate_attention(D=0.1, C=0.9, analytics_report=stable_analytics)
        print(f"D=0.1, C=0.9, Vol=0.01 -> Atenção A: {A1:.4f}")
        assert A1 < 0.3
        
        # Mock Analytics Report 2: INSTÁVEL E INCERTO (Alta Atenção)
        unstable_analytics = {
            "metrics_volatility": {"volatility_index_D": 0.20}, # Alta Volatilidade
        }
        
        print("\n--- 2. Cenário CRÍTICO (Alta Atenção) ---")
        A2 = await attention.calculate_attention(D=0.8, C=0.4, analytics_report=unstable_analytics)
        print(f"D=0.8, C=0.4, Vol=0.20 -> Atenção A: {A2:.4f}")
        assert A2 > 0.8
        
        print("\nParâmetros Snapshot (com telemetria de entrada):")
        print(json.dumps(attention.snapshot_state(), indent=2))

    asyncio.run(main())
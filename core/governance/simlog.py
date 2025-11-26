import numpy as np
import logging
from typing import Dict, Any, List, Optional
import time

from core.memory.hippocampus import setup_logger
from core.services.utils import cosine_similarity, divergence_from_cosine, hash_state, timestamp_id

# Importações de funções utilitárias


logger = setup_logger("SIMLOG")

class SimLog:
    """
    SimLog (Simulated Log): Fornece funções de avaliação lógica e cálculo de divergência (D)
    entre o vetor de representação (embedding) e a 'tríade lógica' associada.
    """

    def __init__(self):
        # Histórico de log e divergência
        self._divergence_history: List[Dict[str, Any]] = []
        self._tripla_logica_log: List[Dict[str, Any]] = []
        self.max_history_size = 1000 # Limite para evitar estouro de memória
        logger.info("SimLog inicializado. Logs e histórico de divergência prontos.")

    # --- Métodos Críticos de Cálculo ---

    def calculate_divergence(self, vector: np.ndarray, tripla_logica: Dict[str, Any]) -> float:
        """
        Calcula a divergência D do vetor (embedding) em relação à lógica esperada (tripla_logica).
        
        A lógica de cálculo é a seguinte:
        1. O vetor do MCH (vector) é comparado com o vetor de intenção esperado da tripla lógica.
        2. A similaridade de cosseno (cos_sim) é calculada.
        3. A divergência D é derivada de cos_sim, mapeada para [0, 1].
        
        Args:
            vector: Vetor de representação (embedding) gerado pelo MCH.
            tripla_logica: Dicionário contendo a lógica esperada, incluindo 'expected_vector'.
        
        Returns:
            O valor da divergência D (float).
        """
        try:
            expected_vector = tripla_logica.get('expected_vector')
            if expected_vector is None:
                logger.error("Tripla Lógica não contém 'expected_vector' para cálculo de divergência.")
                return 1.0 # Retorna divergência máxima em caso de erro

            # 1. Cálculo da Similaridade de Cosseno (usa core/utils)
            cos_sim = cosine_similarity(vector, expected_vector)
            
            # 2. Mapeamento para Divergência D (usa core/utils)
            divergence_D = divergence_from_cosine(cos_sim)

            # 3. Log e Histórico
            log_entry = {
                "cycle_id": tripla_logica.get('cycle_id', timestamp_id('diverg')),
                "timestamp": time.time(),
                "divergence_D": divergence_D,
                "cos_similarity": cos_sim,
                "vector_hash": hash_state({"vector": vector.tolist()}) # Hash do vetor para auditoria
            }
            self._divergence_history.append(log_entry)
            
            # Limpeza do histórico
            if len(self._divergence_history) > self.max_history_size:
                self._divergence_history.pop(0)

            return divergence_D
            
        except Exception as e:
            logger.error(f"Erro no cálculo da divergência: {e}")
            return 1.0

    def log_tripla(self, cycle_id: str, tripla_logica: Dict[str, Any]):
        """
        Registra a tríade lógica de cada ciclo.
        """
        log_entry = {
            "cycle_id": cycle_id,
            "timestamp": time.time(),
            "tripla": tripla_logica,
            "tripla_hash": hash_state(tripla_logica) # Garante a integridade da tríade logada
        }
        self._tripla_logica_log.append(log_entry)
        
        if len(self._tripla_logica_log) > self.max_history_size:
            self._tripla_logica_log.pop(0)
        
        logger.debug(f"Tríade lógica {cycle_id} logada.")

    # --- Métodos de Histórico e Persistência ---

    def get_recent_divergence_history(self, window_size: int = 100) -> List[float]:
        """
        Retorna o histórico recente de valores de divergência D.
        """
        # Retorna apenas os valores de D dos últimos 'window_size' elementos
        return [entry['divergence_D'] for entry in self._divergence_history[-window_size:]]

    def serialize_state(self) -> Dict[str, Any]:
        """
        Serializa o estado interno do SimLog para persistência via PCVS.
        """
        state = {
            "divergence_history": self._divergence_history,
            "tripla_logica_log": self._tripla_logica_log,
            "max_history_size": self.max_history_size
        }
        return state

    def load_state(self, state: Dict[str, Any]):
        """
        Restaura o estado do SimLog a partir de dados previamente salvos.
        """
        try:
            self._divergence_history = state.get("divergence_history", [])
            self._tripla_logica_log = state.get("tripla_logica_log", [])
            self.max_history_size = state.get("max_history_size", 1000)
            logger.warning("Estado do SimLog restaurado com sucesso.")
        except Exception as e:
            logger.error(f"Falha ao carregar estado do SimLog: {e}")

    # --- Métodos Auxiliares de Análise (Opcional, mas Útil) ---

    def get_average_divergence(self, window_size: int = 100) -> float:
        """ Calcula a média de divergência na janela recente. """
        history = self.get_recent_divergence_history(window_size)
        if not history:
            return 0.0
        return float(np.mean(history))
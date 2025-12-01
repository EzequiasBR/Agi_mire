"""
core/services/vector_index.py — VectorIndex V2.0
Integrado com:
- PRAG (PCVS, rollback)
- RegVet (sanidade vetorial)
- SymbolTable (explicabilidade)
- SimLog (auditoria multimodal)
- ControlBus (eventos do ecossistema)
"""

import numpy as np
import time
import hashlib
import logging
from typing import Dict, Any, Optional, Tuple, List


DEFAULT_DIM = 768


class VectorIndex:
    """
    Índice Vetorial com PCVS + integração multimódulo.
    Mantém:
    - armazenamento seguro
    - validação RegVet-like
    - integração com PRAG (rollback)
    - explicabilidade via SymbolTable
    """

    _pcvs_store: Dict[str, Dict[str, Any]] = {}
    _hippocampal_index: List[str] = []

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        prag=None,
        symbol_table=None,
        simlog=None,
        event_bus=None
    ):
        self.dimension = config.get("encoding", {}).get("vector_dimension", DEFAULT_DIM)
        self.logger = logger
        self.prag = prag
        self.symbol_table = symbol_table
        self.simlog = simlog
        self.event_bus = event_bus

        if not VectorIndex._pcvs_store:
            self.logger.info(f"VectorIndex initialized with dimension {self.dimension}")

    # -----------------------------------------------------
    # Helpers de validação (RegVet)
    # -----------------------------------------------------
    def _validate_vector(self, v: np.ndarray):
        if not isinstance(v, np.ndarray):
            raise ValueError("Vector must be a NumPy array.")

        if v.shape != (self.dimension,):
            raise ValueError(f"Vector dimension mismatch. Expected {self.dimension}")

        if v.dtype != np.float32:
            v = v.astype(np.float32)

        norm = np.linalg.norm(v)
        if norm == 0 or np.isnan(norm) or np.isinf(norm):
            raise ValueError("Invalid vector: zero, NaN or infinite norm.")

        return v / norm  # normalizado

    # -----------------------------------------------------
    # Hashes
    # -----------------------------------------------------
    def _hash_vector(self, v: np.ndarray) -> str:
        return hashlib.sha256(np.asarray(v, dtype=np.float32).tobytes(order='C')).hexdigest()

    def _create_pcvs_hash(self, vector: np.ndarray, cycle_id: str) -> str:
        return hashlib.sha256(
            f"{self._hash_vector(vector)}:{cycle_id}:{time.time()}".encode("utf-8")
        ).hexdigest()

    # -----------------------------------------------------
    # PCVS — Save
    # -----------------------------------------------------
    def save_pcvs(self, vector_state: np.ndarray, cycle_id: str, metadata: Dict[str, Any]) -> str:
        vector_state = self._validate_vector(vector_state)

        pcvs_hash = self._create_pcvs_hash(vector_state, cycle_id)

        VectorIndex._pcvs_store[pcvs_hash] = {
            "vector": vector_state,
            "cycle_id": cycle_id,
            "timestamp": time.time(),
            "metadata": metadata
        }
        VectorIndex._hippocampal_index.append(pcvs_hash)

        # Integração SymbolTable (se houver)
        tripla_id = metadata.get("tripla_id")
        if self.symbol_table and tripla_id:
            self.symbol_table.register_mapping(pcvs_hash, tripla_id)

        # SimLog
        if self.simlog:
            self.simlog.log_pcvs_save(pcvs_hash, metadata)

        # Eventos
        if self.event_bus:
            self.event_bus.publish("PCVS_SAVED", {"pcvs_hash": pcvs_hash, "cycle": cycle_id})

        self.logger.debug(f"PCVS saved: {pcvs_hash[:12]} cycle={cycle_id}")
        return pcvs_hash

    # -----------------------------------------------------
    # Load PCVS
    # -----------------------------------------------------
    def load_pcvs(self, pcvs_hash: str) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        record = VectorIndex._pcvs_store.get(pcvs_hash)

        if not record:
            self.logger.warning(f"PCVS hash {pcvs_hash[:12]} not found.")
            return None

        # SimLog
        if self.simlog:
            self.simlog.log_pcvs_load(pcvs_hash)

        # Evento
        if self.event_bus:
            self.event_bus.publish("PCVS_LOADED", {"pcvs_hash": pcvs_hash})

        return record['vector'], record['metadata']

    # -----------------------------------------------------
    # Get latest
    # -----------------------------------------------------
    def get_latest_pcvs_hash(self) -> Optional[str]:
        if not VectorIndex._pcvs_store:
            return None
        return max(
            VectorIndex._pcvs_store,
            key=lambda k: VectorIndex._pcvs_store[k]['timestamp']
        )

    # -----------------------------------------------------
    # Busca Top-K
    # -----------------------------------------------------
    def find_nearest_pcvs(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        query_vector = self._validate_vector(query_vector)

        if not VectorIndex._pcvs_store:
            return []

        results: List[Tuple[str, float]] = []

        for pcvs_hash, record in VectorIndex._pcvs_store.items():
            stored = self._validate_vector(record["vector"])
            sim = float(np.dot(query_vector, stored))
            results.append((pcvs_hash, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    # Compatibilidade
    def add_to_index(self, key: str, vector: np.ndarray):
        metadata = {"cycle_id": key, "protocol": "TEST"}
        return self.save_pcvs(vector, key, metadata)

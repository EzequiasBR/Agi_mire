# core/storage/bridge/symbol_table.py
"""
Symbol Table V2.0 — Integrado, Seguro e Compatível com PRAG / SimLog / OA / RegVet

Funções principais:
- Mapeamento bidirecional V-ID ↔ T-ID
- Verificação de Round-Trip
- Snapshots para PRAG (PCVS)
- Emissão de eventos SimLog estruturados
"""

import logging
from typing import Optional, Dict, Tuple


class SymbolTable:
    """
    Gerencia a coerência simbólico–vetorial do Agi_mire.

    Responsabilidades:
    - Mapeamento bidirecional (round-trip)
    - Validação de IDs simbólicos e vetoriais
    - Eventos de auditoria para SimLog
    - Suporte a snapshots PRAG (rollback)
    """

    def __init__(self, config: Dict, logger: logging.Logger, simlog_api=None):
        self.config = config
        self.logger = logger
        self.simlog = simlog_api

        # Agora são atributos de instância (correto)
        self._v_to_t: Dict[str, str] = {}
        self._t_to_v: Dict[str, str] = {}

        self.logger.info("Symbol Table V2.0 inicializada.")

    # ===========================================================
    # Validação de IDs
    # ===========================================================

    def _validate_id(self, id_value: str, id_type: str):
        if not isinstance(id_value, str) or len(id_value) == 0:
            raise ValueError(f"ID inválido para {id_type}: '{id_value}'")

    # ===========================================================
    # Registro de Mapeamento
    # ===========================================================

    def register_mapping(self, vector_id: str, tripla_id: str) -> bool:
        self._validate_id(vector_id, "vector_id")
        self._validate_id(tripla_id, "tripla_id")

        conflict = vector_id in self._v_to_t and self._v_to_t[vector_id] != tripla_id

        if conflict:
            self.logger.warning(f"Conflito V-ID '{vector_id}': sobrescrevendo mapeamento.")
            if self.simlog:
                self.simlog.emit("SIMLOG_SYMBOL_CONFLICT", {
                    "vector_id": vector_id,
                    "old_tripla_id": self._v_to_t[vector_id],
                    "new_tripla_id": tripla_id
                })

        self._v_to_t[vector_id] = tripla_id
        self._t_to_v[tripla_id] = vector_id

        if self.simlog:
            self.simlog.emit("SIMLOG_SYMBOL_REGISTERED", {
                "vector_id": vector_id,
                "tripla_id": tripla_id
            })

        return True

    # ===========================================================
    # Acesso
    # ===========================================================

    def get_tripla_id(self, vector_id: str) -> Optional[str]:
        self._validate_id(vector_id, "vector_id")
        tid = self._v_to_t.get(vector_id)

        if self.simlog:
            self.simlog.emit("SIMLOG_SYMBOL_LOOKUP", {
                "vector_id": vector_id,
                "found": tid is not None
            })

        return tid

    def get_vector_id(self, tripla_id: str) -> Optional[str]:
        self._validate_id(tripla_id, "tripla_id")
        vid = self._t_to_v.get(tripla_id)

        if self.simlog:
            self.simlog.emit("SIMLOG_SYMBOL_LOOKUP", {
                "tripla_id": tripla_id,
                "found": vid is not None
            })

        return vid

    # ===========================================================
    # Coerência Round-Trip
    # ===========================================================

    def check_round_trip_coherence(self, vector_id: str, tripla_id: str) -> bool:
        self._validate_id(vector_id, "vector_id")
        self._validate_id(tripla_id, "tripla_id")

        ok = (
            self.get_tripla_id(vector_id) == tripla_id and
            self.get_vector_id(tripla_id) == vector_id
        )

        if self.simlog:
            self.simlog.emit("SIMLOG_SYMBOL_RT_CHECK", {
                "vector_id": vector_id,
                "tripla_id": tripla_id,
                "coherent": ok
            })

        return ok

    # ===========================================================
    # Snapshots (PRAG V1.0)
    # ===========================================================

    def export_state(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        return self._v_to_t.copy(), self._t_to_v.copy()

    def load_state(self, v_to_t_state: Dict[str, str], t_to_v_state: Dict[str, str]):
        self._v_to_t = v_to_t_state
        self._t_to_v = t_to_v_state
        self.logger.info("Estado do SymbolTable restaurado via PRAG.")
        if self.simlog:
            self.simlog.emit("SIMLOG_SYMBOL_RESTORED", {})

# core/prag.py (PRAG V3 - Pylance friendly)
"""
PRAG (Ponto de Revalidação de Ação e Governança) V3
Controle de integridade do estado com rollback multimodal.
Integração com ControlBus e telemetria.
"""
from __future__ import annotations
import logging
import time
import json
from typing import Any, Dict, Optional, List

# Simulação de importações (ControlBus e SystemEvents)
try:
    from ..services.control_bus import ControlBus, SystemEvents
except ImportError:
    class SystemEvents:
        PARAM_ADJUSTED = "PARAM_ADJUSTED"
        ROLLBACK_INITIATED = "ROLLBACK_INITIATED"

    class ControlBus:
        def subscribe(self, *args, **kwargs): pass
        def publish(self, *args, **kwargs): pass

logger = logging.getLogger("PRAG")

class PRAG:
    def __init__(self, initial_threshold: float = 0.05, event_bus: Optional["ControlBus"] = None):
        """
        Inicializa o PRAG.
        :param initial_threshold: Limiar delta H (0-1)
        :param event_bus: Instância do ControlBus
        """
        self.rollback_threshold: float = self._validate_threshold(initial_threshold)
        self.last_H: float = 1.0
        self.event_bus = event_bus
        self.threshold_history: List[float] = [self.rollback_threshold]  # Histórico de ajustes

        if self.event_bus:
            self.event_bus.subscribe(SystemEvents.PARAM_ADJUSTED, self._handle_param_adjustment)

        logger.info(json.dumps({
            "event": "PRAG_initialized",
            "rollback_threshold": self.rollback_threshold,
            "timestamp": time.time()
        }))

    # ---------------------------------------------------------
    # Helpers de Validação
    # ---------------------------------------------------------
    @staticmethod
    def _validate_threshold(value: float) -> float:
        """Garante que o threshold esteja entre 0 e 1"""
        if not isinstance(value, (float, int)):
            raise ValueError("Threshold deve ser numérico.")
        if value < 0.0 or value > 1.0:
            raise ValueError("Threshold deve estar entre 0 e 1.")
        return float(value)

    # ---------------------------------------------------------
    # Handlers do ControlBus
    # ---------------------------------------------------------
    def _handle_param_adjustment(self, payload: Dict[str, Any]):
        """Atualiza o limiar de rollback dinamicamente via BUS"""
        new_threshold = payload.get("prag_threshold")
        if new_threshold is not None:
            try:
                validated = self._validate_threshold(new_threshold)
                old_threshold = self.rollback_threshold
                self.rollback_threshold = validated
                self.threshold_history.append(validated)
                logger.info(json.dumps({
                    "event": "PRAG_threshold_updated",
                    "old_threshold": old_threshold,
                    "new_threshold": validated,
                    "timestamp": time.time()
                }))
            except ValueError as ve:
                logger.error(json.dumps({
                    "event": "PRAG_invalid_threshold",
                    "attempted_value": new_threshold,
                    "error": str(ve),
                    "timestamp": time.time()
                }))

    # ---------------------------------------------------------
    # Governança e Decisão
    # ---------------------------------------------------------
    def update_rollback_threshold(self, new_threshold: float):
        """Atualiza threshold manualmente se não houver BUS"""
        validated = self._validate_threshold(new_threshold)
        self.rollback_threshold = validated
        self.threshold_history.append(validated)

    def should_rollback(self, current_H: float, last_state_hash: Optional[str], scope: Optional[List[str]] = None) -> bool:
        """
        Determina se o rollback deve ser iniciado.
        :param current_H: Saúde atual do sistema (0-1)
        :param last_state_hash: Hash do último estado salvo
        :param scope: Lista de subsistemas para rollback parcial
        :return: True se rollback necessário, False caso contrário
        """
        delta_H = self.last_H - current_H
        rollback_needed = delta_H > self.rollback_threshold

        # Monta log estruturado
        log_entry = {
            "event": "PRAG_rollback_check",
            "delta_H": delta_H,
            "rollback_threshold": self.rollback_threshold,
            "last_H": self.last_H,
            "current_H": current_H,
            "rollback_needed": rollback_needed,
            "scope": scope,
            "timestamp": time.time()
        }

        if rollback_needed:
            logger.warning(json.dumps(log_entry))
            if self.event_bus:
                self.event_bus.publish(
                    SystemEvents.ROLLBACK_INITIATED,
                    {
                        "reason": "DeltaH_Exceeded",
                        "delta_H": delta_H,
                        "snapshot_hash": last_state_hash,
                        "current_H": current_H,
                        "scope": scope
                    }
                )
        else:
            logger.info(json.dumps(log_entry))

        # Atualiza H para o próximo ciclo
        self.last_H = current_H
        return rollback_needed

    def validate_hash_consistency(self, current_input_hash: str, last_snapshot_hash: Optional[str] = None):
        """
        Validação de hash dupla
        :param current_input_hash: hash do estado atual
        :param last_snapshot_hash: hash do snapshot
        """
        log_entry = {
            "event": "PRAG_validate_hash",
            "current_input_hash": current_input_hash[:10],
            "last_snapshot_hash": (last_snapshot_hash[:10] if last_snapshot_hash else None),
            "timestamp": time.time()
        }

        if last_snapshot_hash and current_input_hash != last_snapshot_hash:
            log_entry["warning"] = "Hash mismatch detected!"
            logger.warning(json.dumps(log_entry))
        else:
            logger.info(json.dumps(log_entry))

    def snapshot_state(self) -> Dict[str, Any]:
        """Retorna snapshot do estado para PCVS"""
        return {
            "rollback_threshold": self.rollback_threshold,
            "last_H": self.last_H,
            "threshold_history": self.threshold_history.copy(),
            "timestamp": time.time()
        }

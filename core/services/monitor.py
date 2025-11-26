# core/services/monitor.py
"""
Monitor – V1.0
Responsável por registrar métricas, eventos de ciclos e fornecer relatórios
do estado operacional do Agi_mire.
"""

import time
import logging
import json
from typing import Dict, Any, List

logger = logging.getLogger("Monitor")


class Monitor:
    def __init__(self, window_size: int = 100):
        # Histórico de eventos
        self.event_log: List[Dict[str, Any]] = []

        # Métricas agregadas
        self.metrics: Dict[str, Any] = {
            "cycle_count": 0,
            "avg_cycle_duration": 0.0,
            "avg_divergence_D": 0.0,
            "rollback_count": 0,
            "lo_trigger_count": 0,
        }

        # Configuração
        self.window_size = window_size
        logger.info(f"Monitor inicializado com window_size={window_size}")

    # ------------------------------------------------------------------
    # Registro de ciclo
    # ------------------------------------------------------------------
    def register_cycle_end(self, result_data: Dict[str, Any]) -> None:
        """
        Registra o resultado de cada ciclo.
        result_data deve conter: D, C, H, V, E, acionamentos de LO/rollback,
        hash do snapshot e duração do ciclo.
        """
        result_data["timestamp"] = time.time()
        self.event_log.append(result_data)

        # Mantém histórico limitado
        if len(self.event_log) > self.window_size:
            self.event_log.pop(0)

        # Atualiza métricas agregadas
        self.metrics["cycle_count"] += 1
        if result_data.get("rollback", False):
            self.metrics["rollback_count"] += 1
        if result_data.get("lo_trigger", False):
            self.metrics["lo_trigger_count"] += 1

        logger.info(f"[Monitor] Ciclo registrado: {result_data}")

    # ------------------------------------------------------------------
    # Atualização de métricas
    # ------------------------------------------------------------------
    def update_metrics(self, cycle_duration: float, divergence_D: float, cycle_count: int) -> None:
        """
        Atualiza métricas agregadas do sistema.
        """
        # Média móvel simples
        prev_count = self.metrics["cycle_count"]
        self.metrics["avg_cycle_duration"] = (
            (self.metrics["avg_cycle_duration"] * prev_count + cycle_duration) / (prev_count + 1)
        )
        self.metrics["avg_divergence_D"] = (
            (self.metrics["avg_divergence_D"] * prev_count + divergence_D) / (prev_count + 1)
        )

        logger.debug(
            f"[Monitor] Métricas atualizadas: duration={cycle_duration}, divergence_D={divergence_D}, total_cycles={cycle_count}"
        )

    # ------------------------------------------------------------------
    # Relatórios
    # ------------------------------------------------------------------
    def export_report(self) -> None:
        """
        Exporta relatório consolidado em JSON.
        """
        report = {
            "metrics": self.metrics,
            "event_log": self.event_log,
            "generated_at": time.time(),
        }
        try:
            with open("monitor_report.json", "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            logger.info("[Monitor] Relatório exportado para monitor_report.json")
        except Exception as e:
            logger.error(f"[Monitor] Falha ao exportar relatório: {e}")

    def get_status_report(self) -> Dict[str, Any]:
        """
        Retorna estado atual das métricas e contagem de eventos.
        """
        return {
            "metrics": self.metrics,
            "last_event": self.event_log[-1] if self.event_log else None,
        }

    # ------------------------------------------------------------------
    # Persistência e Rollback
    # ------------------------------------------------------------------
    def load_snapshot(self, state: Dict[str, Any]) -> None:
        """
        Restaura estado do monitor a partir de snapshot (PCVS).
        """
        self.metrics = state.get("metrics", self.metrics)
        self.event_log = state.get("event_log", self.event_log)
        logger.warning("[Monitor] Estado restaurado a partir de snapshot.")

    def serialize_state(self) -> Dict[str, Any]:
        """
        Serializa estado interno para persistência (PCVS).
        """
        return {
            "metrics": self.metrics,
            "event_log": self.event_log,
            "timestamp": time.time(),
        }
from __future__ import annotations
import uuid
import time
import json
import logging
from typing import Dict, Any, Optional
import numpy as np

from core.orchestration.control_bus import ControlBus, SystemEvents

def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s"))
        logger.addHandler(ch)
    logger.setLevel(logging.INFO)
    return logger

logger = setup_logger("PRAG")

class PRAG:
    """
    Protocolo de Auditoria e Governança (PRAG).
    Registra ciclos de decisão, divergências e gerencia eventos de segurança.
    """

    def __init__(self, simlog: Any, control_bus: Optional[ControlBus], thresholds: Optional[Dict[str, Any]] = None):
        self.simlog = simlog
        self.control_bus = control_bus
        self.active_cycles: Dict[str, Dict[str, Any]] = {}
        self.thresholds = thresholds or {
            "max_divergence_D": 0.9,
            "max_runtime_s": 10.0,
            "feedback_dim": 3
        }

        logger.info(json.dumps({
            "event": "PRAG_initialized",
            "control_bus_attached": bool(self.control_bus),
            "thresholds": self.thresholds
        }))

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------
    async def _safe_publish(self, event_type: SystemEvents, payload: Dict[str, Any], source_module: str, request_id: str):
        if not self.control_bus:
            logger.warning(json.dumps({
                "event": "publish_skipped_no_bus",
                "type": str(event_type),
                "request_id": request_id
            }))
            return
        try:
            await self.control_bus.publish(
                event_type=event_type,
                payload=payload,
                source_module=source_module,
                request_id=request_id
            )
        except Exception as e:
            logger.exception(f"PRAG publish failure: {e}")
            logger.error(json.dumps({
                "event": "publish_failure",
                "type": str(event_type),
                "error": str(e),
                "request_id": request_id
            }))

    def _feedback_vec(self, kind: str = "success") -> np.ndarray:
        dim = int(self.thresholds.get("feedback_dim", 3))
        v = np.zeros(dim, dtype=np.float32)
        if kind == "success":
            v[0] = 1.0
        elif kind == "failure":
            v[min(1, dim - 1)] = 1.0
        else:
            v[min(2, dim - 1)] = 1.0
        return v

    # ---------------------------------------------------------
    # Gerenciamento de Ciclos
    # ---------------------------------------------------------
    def start_new_cycle(self) -> str:
        session_id = str(uuid.uuid4())
        ts = time.time()
        cycle_record = {
            "start_ts": ts,
            "status": "ACTIVE",
            "logs": [],
            "divergence_D": 0.0,
            "final_hypothesis": None,
            "feedback_vector": None,
            "rollback_triggered": False
        }
        self.active_cycles[session_id] = cycle_record
        logger.info(json.dumps({"event": "PRAG_cycle_started", "session_id": session_id, "ts": ts}))
        return session_id

    def end_cycle(self, session_id: str):
        record = self.active_cycles.pop(session_id, None)
        if record is None:
            logger.warning(json.dumps({"event": "PRAG_cycle_end_missing", "session_id": session_id}))
            return

        record["end_ts"] = time.time()
        record["status"] = "COMPLETED"
        record["runtime_s"] = record["end_ts"] - record["start_ts"]

        if hasattr(self.simlog, 'persist_prag_cycle'):
            try:
                self.simlog.persist_prag_cycle(session_id, record)
            except Exception as e:
                logger.exception(f"persist_prag_cycle failed: {e}")
                logger.error(json.dumps({
                    "event": "persist_prag_cycle_failure",
                    "session_id": session_id,
                    "error": str(e)
                }))
        else:
            logger.info(json.dumps({
                "event": "persist_prag_cycle_skipped",
                "reason": "method_missing",
                "session_id": session_id
            }))

        logger.info(json.dumps({
            "event": "PRAG_cycle_completed",
            "session_id": session_id,
            "status": record["status"],
            "runtime_s": round(record["runtime_s"], 4),
            "divergence_D": round(record["divergence_D"], 4)
        }))

    # ---------------------------------------------------------
    # Logs e Métricas
    # ---------------------------------------------------------
    def log_divergence(self, session_id: str, D: float) -> None:
        record = self.active_cycles.get(session_id)
        if record:
            record["divergence_D"] = float(D)
            record["logs"].append({"ts": time.time(), "type": "divergence_log", "value": D})
            logger.info(json.dumps({
                "event": "PRAG_divergence_logged",
                "session_id": session_id,
                "D": round(float(D), 6)
            }))

    def log_cycle_success(self, session_id: str, hypothesis_triple: Any) -> np.ndarray:
        record = self.active_cycles.get(session_id)
        if record:
            record["final_hypothesis"] = hypothesis_triple
            feedback_vector = self._feedback_vec("success")
            record["feedback_vector"] = feedback_vector.tolist()
            logger.info(json.dumps({
                "event": "PRAG_cycle_success",
                "session_id": session_id
            }))
            return feedback_vector
        return self._feedback_vec("none")

    def log_cycle_failure(self, session_id: str, V_desvio: np.ndarray) -> np.ndarray:
        record = self.active_cycles.get(session_id)
        if record:
            record["status"] = "FAILED_VALIDATION"
            feedback_vector = self._feedback_vec("failure")
            record["feedback_vector"] = feedback_vector.tolist()
            record["V_desvio"] = np.asarray(V_desvio, dtype=np.float32).tolist()
            logger.warning(json.dumps({
                "event": "PRAG_cycle_failure",
                "session_id": session_id
            }))
            return feedback_vector
        return self._feedback_vec("none")

    # ---------------------------------------------------------
    # Supervisão de runtime
    # ---------------------------------------------------------
    def check_runtime(self, session_id: str) -> bool:
        record = self.active_cycles.get(session_id)
        if not record:
            return True
        runtime = time.time() - record["start_ts"]
        max_rt = float(self.thresholds.get("max_runtime_s", 10.0))
        if runtime > max_rt:
            record["logs"].append({"ts": time.time(), "type": "runtime_exceeded", "runtime_s": runtime})
            logger.warning(json.dumps({
                "event": "PRAG_runtime_exceeded",
                "session_id": session_id,
                "runtime_s": round(runtime, 4),
                "max_runtime_s": max_rt
            }))
            return False
        return True

    # ---------------------------------------------------------
    # Eventos de segurança
    # ---------------------------------------------------------
    async def trigger_rollback(self, session_id: str, rollback_level: str):
        record = self.active_cycles.get(session_id)
        if record:
            record["rollback_triggered"] = True
            record["status"] = f"ROLLBACK_{rollback_level.upper()}"
            record["logs"].append({"ts": time.time(), "type": "rollback_trigger", "level": rollback_level})

        logger.warning(json.dumps({
            "event": "PRAG_rollback_triggered",
            "session_id": session_id,
            "level": rollback_level
        }))

        await self._safe_publish(
            event_type=SystemEvents.ROLLBACK_INITIATED,
            payload={"session_id": session_id, "rollback_level": rollback_level, "timestamp_prag": time.time()},
            source_module="PRAG",
            request_id=session_id
        )

    async def trigger_fail_safe(self, session_id: str, error_details: str):
        record = self.active_cycles.get(session_id)
        if record:
            record["status"] = "FAIL_SAFE_ACTIVE"
            record["logs"].append({"ts": time.time(), "type": "fail_safe", "error": error_details})

        logger.critical(json.dumps({
            "event": "PRAG_fail_safe_triggered",
            "session_id": session_id,
            "details": error_details
        }))

        await self._safe_publish(
            event_type=SystemEvents.INTEGRITY_VIOLATION,
            payload={
                "session_id": session_id,
                "violation_type": "CRITICAL_RUNTIME_ERROR",
                "details": error_details,
                "timestamp_prag": time.time()
            },
            source_module="PRAG",
            request_id=session_id
        )

        if session_id in self.active_cycles:
            self.active_cycles.pop(session_id)
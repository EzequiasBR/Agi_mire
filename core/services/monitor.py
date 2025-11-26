# core/services/monitor.py
"""
MonitorService V1 - Telemetria, Observabilidade e Governança Multimodal

Funções:
- Registrar métricas de desempenho e eventos críticos.
- Produzir logs estruturados em JSON.
- Integrar com ControlBus para telemetria distribuída.
- Snapshots multimodais com hash/checksum.
- Thread-safe e configurável.
"""
from __future__ import annotations
import time
import json
import logging
import os
import hashlib
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

# Tentativa de importar utilitários
try:
    from .utils import setup_logger, save_json
    from .control_bus import ControlBus, SystemEvents
except Exception:
    # Fallback simples
    def setup_logger(name: str) -> logging.Logger:
        logger = logging.getLogger(name)
        if not logger.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s"))
            logger.addHandler(ch)
        logger.setLevel(logging.INFO)
        return logger

    def save_json(path: str, data: Dict[str, Any]) -> None:
        tmp = f"{path}.tmp"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp, path)

    class SystemEvents:
        ROLLBACK_INITIATED = "RollbackInitiated"
        STATE_PERSISTED = "StatePersisted"
        PARAM_ADJUSTED = "AdaptationParametersAdjusted"

    class ControlBus:
        def publish(self, event_type: str, payload: Dict[str, Any]): pass

logger = setup_logger("MonitorService")


class Monitor:
    def __init__(self, control_bus: Optional[ControlBus] = None, max_series_len: int = 50000):
        self._lock = Lock()
        self._series: Dict[str, List[Tuple[float, float, Dict[str, Any]]]] = {}
        self.metrics: Dict[str, Any] = self._get_initial_metrics()
        self.event_log: List[Dict[str, Any]] = []
        self.max_series_len = max_series_len
        self.control_bus = control_bus

    def _get_initial_metrics(self) -> Dict[str, Any]:
        return {
            "cycle_count": 0,
            "total_duration_s": 0.0,
            "successful_roundtrip_count": 0,
            "failed_roundtrip_count": 0,
            "rollback_total_count": 0,
            "rollback_partial_count": 0,
            "ppo_trigger_count": 0,
            "regvet_enforced_count": 0,
            "last_cycle_duration_s": 0.0,
            "last_cycle_ts": 0.0
        }

    # -----------------------------
    # Observability primitives
    # -----------------------------
    def observe(self, name: str, value: float, ctx: Optional[Dict[str, Any]] = None) -> None:
        if not isinstance(value, (float, int)):
            raise ValueError(f"Metric value must be numeric. Got {value!r}")
        ts = time.time()
        ctx = ctx or {}
        with self._lock:
            self._series.setdefault(name, []).append((ts, float(value), ctx))
            if len(self._series[name]) > self.max_series_len:
                self._series[name] = self._series[name][-int(self.max_series_len * 0.8):]
        logger.debug(json.dumps({
            "event": "observe",
            "metric": name,
            "value": float(value),
            "timestamp": ts,
            "ctx": ctx
        }))

    def summary(self, name: str, window_s: float = 300.0) -> Dict[str, Optional[float]]:
        now = time.time()
        with self._lock:
            series = self._series.get(name, [])
            vals = [v for ts, v, _ in series if now - ts <= window_s]
        if not vals:
            return {"count": 0, "avg": None, "p95": None, "max": None}
        vals_sorted = sorted(vals)
        count = len(vals_sorted)
        avg = sum(vals_sorted) / count
        p95 = vals_sorted[int(0.95 * (count - 1))]
        return {"count": count, "avg": float(avg), "p95": float(p95), "max": float(max(vals_sorted))}

    def clear_series(self, name: Optional[str] = None) -> None:
        with self._lock:
            if name is None:
                self._series.clear()
            else:
                self._series.pop(name, None)
            logger.info(f"Monitor.clear_series: cleared {name or 'all'}")

    # -----------------------------
    # Registro de Ciclos e Eventos
    # -----------------------------
    def register_cycle_end(self, result_data: Dict[str, Any]) -> None:
        ts = time.time()
        action = result_data.get("action", "unknown")
        duration = float(result_data.get("duration_s", 0.0))
        valid_rt = bool(result_data.get("valid_roundtrip", False))
        regvet_enforced = bool(result_data.get("regvet", {}).get("enforced", result_data.get("regvet_enforced", False)))
        with self._lock:
            # Atualiza métricas
            self.metrics["cycle_count"] += 1
            self.metrics["total_duration_s"] += duration
            self.metrics["last_cycle_duration_s"] = duration
            self.metrics["last_cycle_ts"] = ts

            if valid_rt:
                self.metrics["successful_roundtrip_count"] += 1
            else:
                self.metrics["failed_roundtrip_count"] += 1

            if action == "rollback_total":
                self.metrics["rollback_total_count"] += 1
                self._publish_event(SystemEvents.ROLLBACK_INITIATED, {"type": "total"})
            elif action == "rollback_partial":
                self.metrics["rollback_partial_count"] += 1
                self._publish_event(SystemEvents.ROLLBACK_INITIATED, {"type": "partial"})
            elif action == "trigger_ppo":
                self.metrics["ppo_trigger_count"] += 1

            if regvet_enforced:
                self.metrics["regvet_enforced_count"] += 1

            # Registra log estruturado
            evt = {
                "ts": ts,
                "cycle_num": self.metrics["cycle_count"],
                "action": action,
                "duration_s": duration,
                "D": result_data.get("D"),
                "C": result_data.get("C"),
                "pcvs_hash": result_data.get("pcvs_hash"),
                "extra": result_data.get("extra", {})
            }
            self.event_log.append(evt)
            # Observe métricas pontuais
            if "D" in result_data:
                try:
                    self.observe("D", float(result_data["D"]), {"cycle_num": self.metrics["cycle_count"]})
                except Exception:
                    logger.exception("Monitor.register_cycle_end: failed to observe D")
            if "error_rt" in result_data:
                try:
                    self.observe("error_rt", float(result_data["error_rt"]), {"cycle_num": self.metrics["cycle_count"]})
                except Exception:
                    logger.exception("Monitor.register_cycle_end: failed to observe error_rt")

    def _publish_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        if self.control_bus:
            try:
                self.control_bus.publish(event_type, payload)
            except Exception as e:
                logger.error("Monitor.publish_event failed: %s", str(e))

    # -----------------------------
    # Snapshot e Persistência
    # -----------------------------
    def export_snapshot(self, path: str = "monitor_snapshot.json") -> None:
        with self._lock:
            data = {
                "timestamp": time.time(),
                "metrics": dict(self.metrics),
                "event_log": list(self.event_log),
                "series_keys": list(self._series.keys())
            }
            # Gerar hash do snapshot
            snapshot_json = json.dumps(data, sort_keys=True).encode("utf-8")
            data["snapshot_hash"] = hashlib.sha256(snapshot_json).hexdigest()
        save_json(path, data)
        logger.info(f"Monitor snapshot exportado para {path}")

    def load_snapshot(self, state: Dict[str, Any]) -> None:
        if not state or not isinstance(state, dict):
            raise ValueError("Invalid snapshot structure")
        with self._lock:
            self.metrics = state.get("metrics", self._get_initial_metrics())
            self.event_log = state.get("event_log", [])
        logger.info(f"Monitor snapshot loaded. Cycle count: {self.metrics.get('cycle_count', 0)}")

    # -----------------------------
    # Relatórios e Utilitários
    # -----------------------------
    def get_status_report(self) -> Dict[str, Any]:
        with self._lock:
            cycles = int(self.metrics.get("cycle_count", 0))
            total_dur = float(self.metrics.get("total_duration_s", 0.0))
            avg_duration = total_dur / cycles if cycles > 0 else 0.0
            rt_success_rate = (float(self.metrics.get("successful_roundtrip_count", 0)) / cycles * 100) if cycles > 0 else 0.0
            return {
                "system_status": "OPERATIONAL",
                "uptime_s": time.time() - self.metrics["last_cycle_ts"] if self.metrics["last_cycle_ts"] > 0 else 0,
                "metrics": dict(self.metrics),
                "calculated_metrics": {
                    "avg_duration_s": float(avg_duration),
                    "roundtrip_success_rate_%": float(rt_success_rate)
                }
            }

    def clear_logs(self) -> None:
        with self._lock:
            self.event_log = []
            logger.info("Monitor event log cleared.")

    def peek_events(self, n: int = 10) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self.event_log[-n:])

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "metrics": dict(self.metrics),
                "series_count": {k: len(v) for k, v in self._series.items()},
                "event_log_len": len(self.event_log)
            }

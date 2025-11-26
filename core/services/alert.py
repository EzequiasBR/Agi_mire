# core/services/alert.py
"""
Alert - Módulo de Notificação e Resposta a Anomalias.
Consome dados do Monitor e Analytics para gerar alertas de segurança 
e desempenho em tempo real, com mecanismo de throttling.
"""
from __future__ import annotations
import time
import logging
from typing import Any, Dict, List, Optional
from threading import Lock

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

logger = setup_logger("AlertService")

class Alert:
    def __init__(self, 
                 min_alert_interval_s: float = 30.0,
                 volatile_threshold: float = 0.10,
                 failure_ratio_threshold: float = 0.30):
        
        self._lock = Lock()
        self.min_alert_interval_s = min_alert_interval_s
        self.volatile_threshold = volatile_threshold
        self.failure_ratio_threshold = failure_ratio_threshold
        
        self.last_alert_ts: float = 0.0
        self.alert_log: List[Dict[str, Any]] = []

    # --------------------------------
    # Avaliação de Gatilhos
    # --------------------------------

    def _should_throttle(self, key: str) -> bool:
        """Verifica se o alerta deve ser silenciado (throttled)."""
        now = time.time()
        if (now - self.last_alert_ts) < self.min_alert_interval_s:
            logger.debug("Alert throttled: too soon since last alert.")
            return True
        return False

    def check_for_alerts(self, monitor_report: Dict[str, Any], analytics_report: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Principal função de avaliação de anomalias.
        Retorna o alerta gerado ou None.
        """
        alert_reason = None
        
        # Extrair métricas
        metrics = monitor_report.get("metrics", {})
        calc_metrics = monitor_report.get("calculated_metrics", {})
        volatility = analytics_report.get("metrics_volatility", {}).get("volatility_index_D", 0.0)
        
        # Regra 1: Alta Volatilidade
        if volatility > self.volatile_threshold:
            alert_reason = "HIGH_VOLATILITY"
        
        # Regra 2: Alta Taxa de Falha de Round-Trip
        rt_success_rate = calc_metrics.get("roundtrip_success_rate_%", 100.0)
        rt_failure_ratio = 1.0 - (rt_success_rate / 100.0)
        if rt_failure_ratio >= self.failure_ratio_threshold:
            if not alert_reason:
                alert_reason = "HIGH_RT_FAILURE_RATIO"
            else:
                alert_reason += " & HIGH_RT_FAILURE_RATIO"

        # Regra 3: Rollbacks Consecutivos (Exemplo: 3+ rollbacks nas últimas 10 entradas)
        last_events = monitor_report.get("event_log", [])[-10:]
        recent_rollbacks = sum(1 for event in last_events if "rollback" in event.get("action", ""))
        if recent_rollbacks >= 3 and len(last_events) >= 5:
            if not alert_reason:
                alert_reason = "CONSECUTIVE_ROLLBACKS"
            else:
                alert_reason += " & CONSECUTIVE_ROLLBACKS"

        if alert_reason:
            return self._generate_alert(alert_reason, monitor_report, analytics_report)
            
        return None

    # --------------------------------
    # Geração de Alertas
    # --------------------------------

    def _generate_alert(self, reason: str, monitor_report: Dict[str, Any], analytics_report: Dict[str, Any]) -> Dict[str, Any]:
        """Cria e registra o alerta se não estiver throttled."""
        
        # Check throttling
        if self._should_throttle(reason):
            return None 

        with self._lock:
            ts = time.time()
            alert_data = {
                "ts": ts,
                "reason": reason,
                "severity": "CRITICAL" if "HIGH_VOLATILITY" in reason else "WARNING",
                "monitor_snapshot": monitor_report["metrics"],
                "analytics_snapshot": analytics_report["metrics_volatility"],
                "last_cycle_info": monitor_report.get("event_log", [{}])[-1]
            }
            
            self.alert_log.append(alert_data)
            self.last_alert_ts = ts
            logger.warning("ALERT GENERATED (%s): Reason=%s", alert_data["severity"], reason)
            
            return alert_data

    # --------------------------------
    # Snapshot (Compatibilidade com PCVS)
    # --------------------------------

    def snapshot_state(self) -> Dict[str, Any]:
        """Retorna o estado completo para persistência PCVS."""
        with self._lock:
            return {
                "min_alert_interval_s": self.min_alert_interval_s,
                "volatile_threshold": self.volatile_threshold,
                "failure_ratio_threshold": self.failure_ratio_threshold,
                "last_alert_ts": self.last_alert_ts,
                "alert_log_count": len(self.alert_log)
                # O log completo não é salvo por performance
            }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Carrega o estado a partir de um snapshot PCVS."""
        with self._lock:
            if not state:
                return
            self.min_alert_interval_s = float(state.get("min_alert_interval_s", self.min_alert_interval_s))
            self.volatile_threshold = float(state.get("volatile_threshold", self.volatile_threshold))
            self.failure_ratio_threshold = float(state.get("failure_ratio_threshold", self.failure_ratio_threshold))
            self.last_alert_ts = float(state.get("last_alert_ts", 0.0))
            # O log de alerta é reconstruído ou ignorado na carga, dependendo da política de auditoria externa.
            logger.info("Alert state loaded. Last alert timestamp: %f", self.last_alert_ts)

# --------------------------------
# Teste Rápido
# --------------------------------
if __name__ == "__main__":
    import json
    
    # Mock de relatórios necessários
    mock_monitor_report = {
        "metrics": {"cycle_count": 10},
        "calculated_metrics": {"roundtrip_success_rate_%": 65.0}, # Falha (35% > 30%)
        "event_log": [{"action": "continue"}, {"action": "rollback_partial"}, 
                      {"action": "rollback_total"}, {"action": "rollback_partial"},
                      {"action": "continue"}, {"action": "continue"}],
    }
    mock_analytics_report = {
        "metrics_volatility": {"volatility_index_D": 0.15, "std_dev_D": 0.2}, # Volátil (> 0.10)
    }

    alert_service = Alert(min_alert_interval_s=5.0)

    print("--- 1. Verificação de Alerta Crítico ---")
    alert1 = alert_service.check_for_alerts(mock_monitor_report, mock_analytics_report)
    print("Alerta 1 (Gerado):", json.dumps(alert1, indent=2) if alert1 else "None")

    print("\n--- 2. Verificação com Throttling (deve ser None) ---")
    alert2 = alert_service.check_for_alerts(mock_monitor_report, mock_analytics_report)
    print("Alerta 2 (Throttled):", alert2)

    print("\n--- 3. Verificação com DADOS ESTÁVEIS (deve ser None) ---")
    stable_monitor = {"metrics": {"cycle_count": 20}, "calculated_metrics": {"roundtrip_success_rate_%": 95.0}, "event_log": [{"action": "continue"}]*10}
    stable_analytics = {"metrics_volatility": {"volatility_index_D": 0.01}}
    
    # Simular passagem de tempo
    time.sleep(5.1) 
    
    alert3 = alert_service.check_for_alerts(stable_monitor, stable_analytics)
    print("Alerta 3 (Estável):", alert3)
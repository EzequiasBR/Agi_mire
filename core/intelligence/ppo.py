# PPO V1.3 - Integração completa com MonitorService
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import time
import logging
import numpy as np
import uuid

logger = logging.getLogger("PPO")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s PPO %(levelname)s: %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)


class PPO:
    def __init__(self,
                 tau: float = 0.5,
                 cost_budget: float = 1.0,
                 failure_persist_window: int = 50,
                 failure_persist_threshold: float = 0.3,
                 cooldown_s: float = 300.0,
                 min_improvement_expected: float = 0.05,
                 rng_seed: Optional[int] = None,
                 monitor_service: Optional[Any] = None):
        """
        PPO V1.3 - Integração total com MonitorService para telemetria contínua
        """
        self.session_id = str(uuid.uuid4())
        self.monitor_service = monitor_service

        # Thresholds validados
        self.tau = self._validate_threshold(tau, "tau")
        self.cost_budget = float(cost_budget)
        self.failure_persist_window = int(failure_persist_window)
        self.failure_persist_threshold = self._validate_threshold(failure_persist_threshold, "failure_persist_threshold")
        self.cooldown_s = float(cooldown_s)
        self.min_improvement_expected = float(min_improvement_expected)

        self.last_trigger_time: Optional[float] = None
        self.last_reason: Optional[str] = None
        self.history: List[Dict[str, Any]] = []
        # Garante seed válido para RandomState
        valid_seed = (rng_seed if rng_seed is not None else int(time.time() * 1000)) % (2**32 - 1)
        self.rng = np.random.RandomState(valid_seed)

        logger.info(f"PPO session {self.session_id} initialized | tau={self.tau}, cost_budget={self.cost_budget}")

    # -----------------------------
    # Threshold validation
    # -----------------------------
    def _validate_threshold(self, value: float, name: str) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"PPO: {name} must be between 0 and 1, got {value}")
        return float(value)

    # -----------------------------
    # History helpers
    # -----------------------------
    def register_failure(self, info: Optional[Dict[str, Any]] = None) -> None:
        self.history.append({"ts": time.time(), "type": "failure", "info": info or {}})
        self._trim_history()

    def register_success(self, info: Optional[Dict[str, Any]] = None) -> None:
        self.history.append({"ts": time.time(), "type": "success", "info": info or {}})
        self._trim_history()

    def _trim_history(self) -> None:
        if len(self.history) > self.failure_persist_window:
            self.history = self.history[-self.failure_persist_window:]

    def failure_persistence(self) -> float:
        if not self.history:
            return 0.0
        failures = sum(1 for e in self.history if e["type"] == "failure")
        return float(failures) / len(self.history)

    # -----------------------------
    # Cost justification
    # -----------------------------
    def justify_ontogenesis(self, cost_estimate: Optional[float]) -> Tuple[bool, Dict[str, Any]]:
        details: Dict[str, Any] = {"cost_estimate": cost_estimate, "cost_budget": self.cost_budget}
        if cost_estimate is None:
            fp = self.failure_persistence()
            allowed = fp >= self.failure_persist_threshold
            details.update({"reason": "unknown_cost", "failure_persistence": fp, "allowed_by_persistence": allowed})
            return allowed, details

        cost = float(cost_estimate)
        if cost <= self.cost_budget:
            details.update({"reason": "within_budget", "allowed": True})
            return True, details

        fp = self.failure_persistence()
        if fp >= self.failure_persist_threshold:
            details.update({"reason": "over_budget_allowed_by_persistence", "failure_persistence": fp, "allowed": True})
            return True, details

        details.update({"reason": "over_budget_and_not_persistent", "failure_persistence": fp, "allowed": False})
        return False, details

    # -----------------------------
    # Core trigger
    # -----------------------------
    def trigger(self, D: float, C: float, E_sistemica: float, cost_estimate: Optional[float] = None) -> bool:
            # Valida inputs
            D = self._validate_threshold(D, "D")
            C = self._validate_threshold(C, "C")
            E_sistemica = self._validate_threshold(E_sistemica, "E_sistemica")

            # 1. Condição de Estagnação (Trigger de Inovação Forçada)
            # (Divergência Baixa E Certeza Alta) -> Sistema está 'preso' em uma resposta.
            stagnation_condition = (D < 0.20 and C > 0.80) 

            # 2. Condição de Instabilidade (Trigger de Correção)
            # (Erro Sistêmico Acima do Limiar) -> O sistema está falhando.
            system_error_condition = (E_sistemica > self.tau)
            
            # O PPO é acionado se houver Estagnação OU Erro.
            logical_trigger = stagnation_condition or system_error_condition

            if not logical_trigger:
                self.last_reason = "logical_conditions_not_met"
                self._report_monitor(False, D, C, E_sistemica)
                return False

            allowed, details = self.justify_ontogenesis(cost_estimate)
            if not allowed:
                self.last_reason = f"cost_rejected:{details.get('reason')}"
                self._report_monitor(False, D, C, E_sistemica, details)
                return False

            now = time.time()
            if self.last_trigger_time and (now - self.last_trigger_time) < self.cooldown_s:
                self.last_reason = "cooldown_active"
                self._report_monitor(False, D, C, E_sistemica, details)
                return False

            if stagnation_condition:
                reason = "stagnation_innovation_forced"
            elif system_error_condition:
                reason = "system_error_correction"
            else:
                reason = "triggered_by_unknown" # Fallback
                
            reason += ";cost_ok" if allowed else ";cost_denied"
            self.last_reason = reason

            self.last_trigger_time = now # Aplicar o cooldown somente se o trigger for executado.
            self._report_monitor(True, D, C, E_sistemica, details)
            return True

    # -----------------------------
    # MonitorService reporting
    # -----------------------------
    def _report_monitor(self, triggered: bool, D: float, C: float, E_sistemica: float, details: Optional[Dict[str, Any]] = None):
        if self.monitor_service:
            try:
                self.monitor_service.observe(
                    "PPO_triggered",
                    int(triggered),
                    {
                        "session_id": self.session_id,
                        "D": D, "C": C, "E_sistemica": E_sistemica,
                        "reason": self.last_reason,
                        "details": details or {}
                    }
                )
            except Exception as e:
                logger.warning(f"PPO.monitor_service.observe failed: {e}")

    # -----------------------------
    # Proposal / apply
    # -----------------------------
    def propose_changes(self, context: Optional[Dict[str, Any]] = None, n_proposals: int = 3) -> List[Dict[str, Any]]:
        proposals: List[Dict[str, Any]] = []
        # Seed corrigido para intervalo válido
        base_seed = (int(time.time() * 1000) ^ (len(self.history) << 4)) % (2**32 - 1)
        rng = np.random.RandomState(base_seed)
        for i in range(n_proposals):
            est_cost = float(max(0.01, rng.rand() * (self.cost_budget * 3)))
            est_impr = float(rng.rand() * 0.3)
            desc = f"proposal_{int(time.time())}_{i}"
            steps = [f"mutate_rule_set_step_{j}" for j in range(1, rng.randint(2, 5))]
            proposals.append({
                "id": desc,
                "description": f"Auto-generated structural change #{i}",
                "est_cost": est_cost,
                "est_improvement": est_impr,
                "steps": steps,
                "context": context or {}
            })
        return proposals

    def apply_changes(self, proposals: List[Dict[str, Any]], dry_run: bool = True) -> Dict[str, Any]:
        total_cost = 0.0
        total_impr = 0.0
        details = []
        for p in proposals:
            cost = float(p.get("est_cost", 0.0))
            impr = float(p.get("est_improvement", 0.0))
            success_prob = min(0.95, 0.1 + impr * 3.0)
            success = self.rng.rand() < success_prob
            details.append({"proposal_id": p.get("id"), "cost": cost, "est_improvement": impr, "success_sim": bool(success)})
            total_cost += cost
            if success:
                total_impr += impr

        summary = {"total_cost": total_cost, "achieved_improvement": total_impr, "applied": len(proposals), "details": details}

        if not dry_run:
            if total_impr >= self.min_improvement_expected:
                self.last_trigger_time = time.time()
                self.register_success({"summary": summary})
                logger.info(f"PPO.apply_changes: applied proposals, improvement={total_impr:.4f} cost={total_cost:.4f}")
            else:
                self.register_failure({"summary": summary})
                logger.warning(f"PPO.apply_changes: insufficient improvement -> failure (impr={total_impr:.4f})")

        # Report to MonitorService
        if self.monitor_service:
            try:
                self.monitor_service.observe(
                    "PPO_apply_summary",
                    1,
                    {
                        "session_id": self.session_id,
                        "total_cost": total_cost,
                        "achieved_improvement": total_impr,
                        "applied_count": len(proposals),
                        "dry_run": dry_run,
                        "details": details
                    }
                )
            except Exception as e:
                logger.warning(f"PPO.monitor_service.observe failed: {e}")

        return summary

    # -----------------------------
    # State serialization
    # -----------------------------
    def serialize_state(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "tau": self.tau,
            "cost_budget": self.cost_budget,
            "failure_persist_window": self.failure_persist_window,
            "failure_persist_threshold": self.failure_persist_threshold,
            "cooldown_s": self.cooldown_s,
            "min_improvement_expected": self.min_improvement_expected,
            "last_trigger_time": self.last_trigger_time,
            "last_reason": self.last_reason,
            "history": self.history
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        if not state:
            return
        self.session_id = state.get("session_id", self.session_id)
        self.tau = self._validate_threshold(state.get("tau", self.tau), "tau")
        self.cost_budget = float(state.get("cost_budget", self.cost_budget))
        self.failure_persist_window = int(state.get("failure_persist_window", self.failure_persist_window))
        self.failure_persist_threshold = self._validate_threshold(state.get("failure_persist_threshold", self.failure_persist_threshold), "failure_persist_threshold")
        self.cooldown_s = float(state.get("cooldown_s", self.cooldown_s))
        self.min_improvement_expected = float(state.get("min_improvement_expected", self.min_improvement_expected))
        self.last_trigger_time = state.get("last_trigger_time", self.last_trigger_time)
        self.last_reason = state.get("last_reason", self.last_reason)
        self.history = state.get("history", [])
        self._trim_history()
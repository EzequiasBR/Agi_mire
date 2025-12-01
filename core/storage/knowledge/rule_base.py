# core/storage/knowledge/rule_base.py
"""
RuleBase V3.0 — Base de regras lógicas e guardrails éticos para Agi_mire

Principais características:
- Regras armazenadas como callables (mantendo API original)
- Métodos: add_rule, check_violation (preservados)
- Nova função check(triple, meta) compatível com OEA contract:
    returns {"violated": bool, "gravity": float, "rule_id": str, "severity": float, "meta": dict}
- Integração com ControlBus/PRAG (event emission) quando guardrails disparados
- Support for severity/weight per rule
"""

from __future__ import annotations
import logging
import time
import hashlib
from typing import Callable, Dict, List, Tuple, Optional, Any

logger = logging.getLogger("RuleBase")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s RuleBase %(levelname)s: %(message)s"))
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)

RuleFunction = Callable[[str, str, str], bool]


class RuleBase:
    """
    Stores logical rules and ethical guardrails.
    Maintains:
      self.rules: mapping rule_name -> dict { 'is_ethical': bool, 'func': RuleFunction, 'severity': float }
    """

    def __init__(self, control_bus: Optional[Any] = None, config: Optional[Dict[str, Any]] = None):
        self.rules: Dict[str, Dict[str, Any]] = {}
        self.control_bus = control_bus
        self.config = config or {}
        # Add default ethical guardrails
        self._add_ethical_guardrails()

    # ---------------------------
    # Internal helpers
    # ---------------------------
    def _make_rule_id(self, name: str) -> str:
        return hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]

    # ---------------------------
    # Public API (names preserved)
    # ---------------------------
    def _add_ethical_guardrails(self):
        """Add baseline ethical guardrails (non-replaceable unless explicitly removed)."""

        def check_irreversible_damage(s: str, p: str, o: str) -> bool:
            return not (p == "causa_dano_irreversível")

        # severity 1.0 (max); is_ethical True
        self.add_rule("NoDamageRule", check_irreversible_damage, is_ethical=True, severity=1.0)

    def add_rule(self, name: str, func: RuleFunction, is_ethical: bool = False, severity: float = 0.5):
        """
        Add a new rule.
        - name: user-friendly
        - func: callable(s, p, o) -> bool (True means allowed / passes)
        - is_ethical: whether the rule is ethical guardrail
        - severity: float in [0,1], higher => more severe when violated
        """
        if not callable(func):
            raise ValueError("func must be callable")
        sid = self._make_rule_id(name)
        self.rules[name] = {"id": sid, "func": func, "is_ethical": bool(is_ethical), "severity": max(0.0, min(1.0, float(severity)))}
        logger.info({"event": "RULE_ADDED", "name": name, "id": sid, "is_ethical": is_ethical, "severity": severity})
        if self.control_bus:
            try:
                self.control_bus.publish("PRAG_RULE_ADDED", {"name": name, "id": sid, "severity": severity})
            except Exception:
                logger.debug("control_bus.publish failed for RULE_ADDED", exc_info=True)

    def check_violation(self, subject: str, predicate: str, obj: str) -> List[str]:
        """
        Legacy function preserved: returns list of rule names that are violated.
        """
        violated_rules: List[str] = []
        triple = (subject, predicate, obj)
        for name, data in self.rules.items():
            try:
                ok = bool(data["func"](*triple))
            except Exception as e:
                logger.exception("RuleBase.check_violation: rule function exception for %s", name)
                ok = True  # treat as pass to avoid false positive
            if not ok:
                violated_rules.append(f"{name} (is_ethical={data.get('is_ethical')})")
                # emit an event for each violation (best-effort)
                if self.control_bus:
                    try:
                        self.control_bus.publish("PRAG_RULE_VIOLATION", {"rule": name, "subject": subject, "predicate": predicate, "object": obj})
                    except Exception:
                        logger.debug("control_bus.publish PRAG_RULE_VIOLATION failed", exc_info=True)
        return violated_rules

    # ---------------------------
    # New contract used by OEA: check(triplet, meta) -> dict
    # ---------------------------
    def check(self, triplet: Dict[str, Any], meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate the triplet against rules and return structured contract:
            {"violated": bool, "gravity": float, "rule_id": str | None, "severity": float, "meta": dict}
        Contract details:
            - violated: True if any rule was violated
            - gravity: aggregated gravity in [0,1] (max severity among violated rules weighted)
            - rule_id: id of principal rule triggered (or None)
            - severity: severity of the principal violation
            - meta: extra information (violated_rules list, timestamp, input_meta)
        """
        meta = meta or {}
        subject = triplet.get("subject") or triplet.get("s") or ""
        predicate = triplet.get("predicate") or triplet.get("p") or ""
        obj = triplet.get("object") or triplet.get("o") or ""

        violated = False
        max_severity = 0.0
        principal_rule_id = None
        violated_rules = []

        for name, data in self.rules.items():
            try:
                ok = bool(data["func"](subject, predicate, obj))
            except Exception as e:
                logger.exception("Rule evaluation exception for %s", name)
                ok = True  # safe default: rule passed

            if not ok:
                violated = True
                sev = float(data.get("severity", 0.5))
                violated_rules.append({"name": name, "id": data.get("id"), "severity": sev, "is_ethical": data.get("is_ethical", False)})
                # principal = highest severity
                if sev > max_severity:
                    max_severity = sev
                    principal_rule_id = data.get("id")

                # emit guardrail event when ethical guardrail violated
                if data.get("is_ethical", False):
                    try:
                        if self.control_bus:
                            self.control_bus.publish("PRAG_GUARDRAIL_TRIGGERED", {"rule": name, "severity": sev, "triplet": (subject, predicate, obj)})
                    except Exception:
                        logger.debug("control_bus.publish PRAG_GUARDRAIL_TRIGGERED failed", exc_info=True)

        # Compute gravity: map severity to gravity metric (simple identity or mapping)
        gravity = float(max_severity)

        result = {
            "violated": bool(violated),
            "gravity": float(gravity),
            "rule_id": principal_rule_id,
            "severity": float(max_severity),
            "meta": {"violated_rules": violated_rules, "timestamp": time.time(), **(meta or {})}
        }

        # structured logging
        logger.info({"event": "RULE_CHECK", "triplet": (subject, predicate, obj), "result": {"violated": violated, "gravity": gravity}})

        return result

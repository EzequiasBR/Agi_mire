# core/regvet.py
"""
Reg-Vet V1.1 - Razão -> Vetor (Coerção Vetorial)
Versão final: robusta, determinística e auditável.

Recursos principais:
- normalização rigorosa de vetores
- coerção vetorial com projeção controlada
- seleção determinística de regras (strength > certainty > angle_score)
- impacto angular opcional
- integração segura com projector externo
- recomendação metacognitiva: severity correction
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import copy
import json
import hashlib

# util oficial do Agi_mire
from core.services.utils import setup_logger
logger = setup_logger("RegVet")


# ---------------------------------------------------------
# Utilidades matemáticas
# ---------------------------------------------------------
def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n <= 1e-12:
        return np.zeros_like(v)
    return v / n


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_n = _normalize(a)
    b_n = _normalize(b)
    if np.all(a_n == 0) or np.all(b_n == 0):
        return 0.0
    return float(np.dot(a_n, b_n))


def _divergence_from_cosine_sim(cos_sim: float) -> float:
    return float(max(0.0, min(1.0, (1.0 - cos_sim) / 2.0)))


def _angular_impact_from_cosine(cos_sim: float) -> float:
    c = float(np.clip(cos_sim, -1.0, 1.0))
    return float(np.arccos(c) / np.pi)


# ---------------------------------------------------------
# Classe Principal
# ---------------------------------------------------------
class RegVet:
    def __init__(
        self,
        enforcement_threshold: float = 1e-6,
        default_severity: float = 0.6,
        max_strength: float = 1.0,
        projector: Optional[Any] = None,
        impact_angular: bool = False,
        adjustment_clamp: Tuple[float, float] = (-0.2, 0.2)
    ):
        self.enforcement_threshold = float(enforcement_threshold)
        self.default_severity = float(default_severity)
        self.max_strength = float(max_strength)
        self.projector = projector
        self.impact_angular = bool(impact_angular)
        self.adjustment_clamp = (
            float(adjustment_clamp[0]),
            float(adjustment_clamp[1])
        )

    # -----------------------------------------------------
    # Normalização de regras
    # -----------------------------------------------------
    def _normalize_rules(self, rules: Any) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []

        if rules is None:
            return normalized

        # Estrutura OA: {"level1": [...], "level2": [...]}
        if isinstance(rules, dict) and ("level1" in rules or "level2" in rules):
            for lvl in ("level1", "level2"):
                for r in rules.get(lvl, []):
                    nr = {
                        "id": r.get("id") or hashlib.sha256(
                            json.dumps(r, sort_keys=True).encode()
                        ).hexdigest(),
                        "anchor_vector": r.get("meta", {}).get("anchor_vector"),
                        "certainty": float(r.get("meta", {}).get("certainty", 1.0)),
                        "severity": float(r.get("meta", {}).get("severity", self.default_severity)),
                        "active": r.get("meta", {}).get("active", True),
                        "meta": r.get("meta", {}),
                        "raw_rule": r
                    }
                    normalized.append(nr)
            return normalized

        # Lista de regras
        if isinstance(rules, (list, tuple)):
            for r in rules:
                if isinstance(r, dict):
                    nr = {
                        "id": r.get("id") or hashlib.sha256(
                            json.dumps(r, sort_keys=True, default=str).encode()
                        ).hexdigest(),
                        "anchor_vector": r.get("anchor_vector", None),
                        "certainty": float(r.get("certainty", 1.0)),
                        "severity": float(r.get("severity", self.default_severity)),
                        "active": r.get("active", True),
                        "meta": r.get("meta", {}),
                        "raw_rule": r
                    }
                    normalized.append(nr)
            return normalized

        # Regra única
        if isinstance(rules, dict):
            nr = {
                "id": rules.get("id") or hashlib.sha256(
                    json.dumps(rules, sort_keys=True, default=str).encode()
                ).hexdigest(),
                "anchor_vector": rules.get("anchor_vector", None),
                "certainty": float(rules.get("certainty", 1.0)),
                "severity": float(rules.get("severity", self.default_severity)),
                "active": rules.get("active", True),
                "meta": rules.get("meta", {}),
                "raw_rule": rules
            }
            normalized.append(nr)

        return normalized

    # -----------------------------------------------------
    # Cálculo interno de coerção
    # -----------------------------------------------------
    def _compute_coercion(self, embedding: np.ndarray, anchor: np.ndarray, strength: float) -> Tuple[np.ndarray, np.ndarray]:
        emb = _normalize(embedding)
        anc = _normalize(anchor)

        if np.all(emb == 0) or np.all(anc == 0):
            return emb, np.zeros_like(emb)

        proj_coeff = float(np.dot(emb, anc))
        proj_component = proj_coeff * anc
        coerced = _normalize(emb - strength * proj_component)

        if np.linalg.norm(proj_component) <= 1e-12:
            rep = np.zeros_like(emb)
        else:
            rep = _normalize(proj_component)

        return coerced, rep

    # -----------------------------------------------------
    # API pública
    # -----------------------------------------------------
    def enforce(self, embedding: np.ndarray, rules: Any) -> Dict[str, Any]:
        emb = _normalize(np.asarray(embedding, dtype=float))
        candidate_rules = self._normalize_rules(rules)

        candidates = []
        for r in candidate_rules:
            if not r.get("active", True):
                continue

            anc = r.get("anchor_vector")
            if anc is None:
                continue

            try:
                anc_arr = np.asarray(anc, dtype=float)
            except Exception:
                continue

            if anc_arr.size == 0:
                continue

            severity = float(r.get("severity", self.default_severity))
            certainty = float(r.get("certainty", 1.0))
            strength = min(self.max_strength, max(0.0, severity * certainty))

            cos_sim = _cosine_similarity(emb, anc_arr)
            angle_score = abs(cos_sim)

            candidates.append(
                (strength, certainty, angle_score, anc_arr, r)
            )

        if not candidates:
            return {
                "vector": emb,
                "enforced": False,
                "impact": 0.0,
                "applied_rule": None,
                "repulsion_vector": np.zeros_like(emb),
                "recommendation": {"adjustment": 0.0, "reason": "no_anchor_rule"}
            }

        candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        strength, certainty, angle_score, anchor_vec, rule_meta = candidates[0]

        if strength < self.enforcement_threshold:
            return {
                "vector": emb,
                "enforced": False,
                "impact": 0.0,
                "applied_rule": rule_meta,
                "repulsion_vector": np.zeros_like(emb),
                "recommendation": {"adjustment": 0.0, "reason": "below_threshold"}
            }

        used_projector = False
        coerced_vec = None
        repulsion_vec = None

        # Tentativa com projector externo
        if self.projector is not None:
            try:
                proj_input = {
                    "embedding": emb.copy(),
                    "anchor": anchor_vec.copy(),
                    "strength": float(strength),
                    "rule": copy.deepcopy(rule_meta)
                }
                out = self.projector(proj_input)
                coerced_vec = _normalize(np.asarray(out, dtype=float))
                delta = emb - coerced_vec
                repulsion_vec = _normalize(delta) if np.linalg.norm(delta) > 1e-12 else np.zeros_like(emb)
                used_projector = True
                logger.info(f"RegVet: projector aplicado rule_id={rule_meta.get('id')}")
            except Exception:
                logger.exception("RegVet: projector falhou, usando fallback interno.")
                coerced_vec = None

        # Fallback interno
        if coerced_vec is None:
            coerced_vec, repulsion_vec = self._compute_coercion(emb, anchor_vec, strength)

        cos_sim = _cosine_similarity(emb, coerced_vec)
        cos_sim_anchor = _cosine_similarity(coerced_vec, anchor_vec)

        if self.impact_angular:
            impact = _angular_impact_from_cosine(cos_sim)
        else:
            impact = _divergence_from_cosine_sim(cos_sim)

        # Recomendação metacognitiva
        expected_scale = strength
        adjustment = 0.0
        reason = "within_expected"

        if impact > min(1.0, expected_scale * 1.5):
            adjustment = -0.1 * (impact / (expected_scale + 1e-12))
            reason = "overly_strong_enforcement"
        elif impact < max(1e-4, expected_scale * 0.1):
            adjustment = 0.1 * (1.0 - impact)
            reason = "ineffective_enforcement"

        adjustment = float(np.clip(adjustment, self.adjustment_clamp[0], self.adjustment_clamp[1]))

        recommendation = {
            "adjustment": adjustment,
            "reason": reason,
            "observed_impact": float(impact),
            "raw_adjustment": float(adjustment),
            "strength": float(strength),
            "severity": float(rule_meta.get("severity", self.default_severity)),
            "certainty": float(rule_meta.get("certainty", 1.0)),
            "rule_id": rule_meta.get("id"),
            "angle_score": float(angle_score),
            "used_projector": used_projector,
            "cos_sim": float(cos_sim),
            "cos_sim_anchor": float(cos_sim_anchor)
        }

        logger.info(
            "RegVet.enforce -> id=%s str=%.4f cert=%.4f impact=%.4f adj=%.4f projector=%s",
            str(rule_meta.get("id"))[:12],
            strength, certainty, impact, adjustment, used_projector
        )

        return {
            "vector": coerced_vec,
            "enforced": True,
            "impact": float(impact),
            "applied_rule": rule_meta,
            "repulsion_vector": repulsion_vec if np.linalg.norm(repulsion_vec) > 1e-12 else np.zeros_like(emb),
            "recommendation": recommendation,
            "used_projector": used_projector
        }

    # -----------------------------------------------------
    # Serialização (PCVS)
    # -----------------------------------------------------
    def serialize_state(self) -> Dict[str, Any]:
        return {
            "enforcement_threshold": self.enforcement_threshold,
            "default_severity": self.default_severity,
            "max_strength": self.max_strength,
            "has_projector": self.projector is not None,
            "impact_angular": self.impact_angular,
            "adjustment_clamp": list(self.adjustment_clamp)
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        if not state:
            return
        self.enforcement_threshold = float(state.get("enforcement_threshold", self.enforcement_threshold))
        self.default_severity = float(state.get("default_severity", self.default_severity))
        self.max_strength = float(state.get("max_strength", self.max_strength))
        self.impact_angular = bool(state.get("impact_angular", self.impact_angular))
        ac = state.get("adjustment_clamp")
        if ac:
            self.adjustment_clamp = (float(ac[0]), float(ac[1]))

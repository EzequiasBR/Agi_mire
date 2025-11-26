"""
core/regvet.py

Reg-Vet V1.1 - Razão -> Vetor (Coerção Vetorial)
Finalized: incorpora políticas de robustez, determinismo e metacognição.

Principais melhorias e garantias:
- Desempate determinístico: prioriza (strength, certainty, -abs(cos_sim(emb,anchor))).
- Tratamento robusto de norma zero: _normalize retorna zeros_like(v) quando norm == 0.
- Threshold inclusivo: enforcement ocorre quando strength >= enforcement_threshold.
- Clamping de ajuste metacognitivo: adjustment é clamped para [-0.2, 0.2].
- Métrica de impacto angular opcional (impact_angular=True).
- Validação e logging do projector externo, com fallback seguro.
- Serialização de parâmetros para PCVS/auditoria.

API principal:
    enforce(embedding, rules) -> dict {
      "vector": np.ndarray,
      "enforced": bool,
      "impact": float,
      "applied_rule": {...},
      "repulsion_vector": np.ndarray,
      "recommendation": {...}
    }
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import copy
import json
import hashlib
import logging

logger = logging.getLogger("RegVet")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s RegVet %(levelname)s: %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)


# -----------------------
# Utility math helpers
# -----------------------
def _normalize(v: np.ndarray) -> np.ndarray:
    """Normalize vector; if norm is zero, return zeros_like to be deterministic."""
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n <= 1e-12:
        return np.zeros_like(v)
    return v / n


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [-1,1]."""
    a_n = _normalize(a)
    b_n = _normalize(b)
    # if either is zero-vector, define similarity as 0.0 for deterministic behavior
    if np.all(a_n == 0) or np.all(b_n == 0):
        return 0.0
    return float(np.dot(a_n, b_n))


def _divergence_from_cosine_sim(cos_sim: float) -> float:
    """
    Map cosine similarity (-1..1) to divergence [0..1]:
      D = (1 - cos_sim) / 2
    """
    return float(max(0.0, min(1.0, (1.0 - cos_sim) / 2.0)))


def _angular_impact_from_cosine(cos_sim: float) -> float:
    """
    Angular impact normalized to [0,1]:
      impact = arccos(cos_sim) / pi
    More sensitive around orthogonal angles.
    """
    c = float(np.clip(cos_sim, -1.0, 1.0))
    return float(np.arccos(c) / np.pi)


# -----------------------
# Reg-Vet class
# -----------------------
class RegVet:
    def __init__(self,
                 enforcement_threshold: float = 1e-6,
                 default_severity: float = 0.6,
                 max_strength: float = 1.0,
                 projector: Optional[Any] = None,
                 impact_angular: bool = False,
                 adjustment_clamp: Tuple[float, float] = (-0.2, 0.2)):
        """
        :param enforcement_threshold: minimal combined strength to consider enforcement (inclusive).
        :param default_severity: fallback severity if rule omits it.
        :param max_strength: clamp for strength (<=1.0).
        :param projector: optional callable(project_info) -> projected_vector
            project_info = {"embedding": emb, "anchor": anchor, "strength": strength, "rule": rule_meta}
            If provided, projector must return a numpy vector-like.
        :param impact_angular: whether to compute impact using angular metric (arccos/pi).
        :param adjustment_clamp: (min, max) clamp for metacognitive adjustment.
        """
        self.enforcement_threshold = float(enforcement_threshold)
        self.default_severity = float(default_severity)
        self.max_strength = float(max_strength)
        self.projector = projector
        self.impact_angular = bool(impact_angular)
        self.adjustment_clamp = (float(adjustment_clamp[0]), float(adjustment_clamp[1]))

    # -----------------------
    # Internal: normalize rules input into canonical list
    # -----------------------
    def _normalize_rules(self, rules: Any) -> List[Dict[str, Any]]:
        """
        Accepts flexible 'rules' formats and returns a list of rule dicts with expected fields.
        Fields ensured:
          - id, anchor_vector (or None), certainty, severity, active, meta, raw_rule
        """
        normalized: List[Dict[str, Any]] = []

        if rules is None:
            return normalized

        # If rules is a dict of level1/level2 as OA.get_rules(), attempt to extract anchor info in meta
        if isinstance(rules, dict) and ("level1" in rules or "level2" in rules):
            for lvl in ("level1", "level2"):
                for r in rules.get(lvl, []):
                    nr = {
                        "id": r.get("id") or hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest(),
                        "anchor_vector": r.get("meta", {}).get("anchor_vector"),
                        "certainty": float(r.get("meta", {}).get("certainty", 1.0)),
                        "severity": float(r.get("meta", {}).get("severity", self.default_severity)),
                        "active": r.get("meta", {}).get("active", True),
                        "meta": r.get("meta", {}),
                        "raw_rule": r
                    }
                    normalized.append(nr)
            return normalized

        # If rules is list-like
        if isinstance(rules, (list, tuple)):
            for r in rules:
                if not isinstance(r, dict):
                    continue
                nr = {
                    "id": r.get("id") or hashlib.sha256(json.dumps(r, sort_keys=True, default=str).encode()).hexdigest(),
                    "anchor_vector": r.get("anchor_vector", None),
                    "certainty": float(r.get("certainty", 1.0)),
                    "severity": float(r.get("severity", self.default_severity)),
                    "active": r.get("active", True),
                    "meta": r.get("meta", {}),
                    "raw_rule": r
                }
                normalized.append(nr)
            return normalized

        # If rules is a single dict
        if isinstance(rules, dict):
            nr = {
                "id": rules.get("id") or hashlib.sha256(json.dumps(rules, sort_keys=True, default=str).encode()).hexdigest(),
                "anchor_vector": rules.get("anchor_vector", None),
                "certainty": float(rules.get("certainty", 1.0)),
                "severity": float(rules.get("severity", self.default_severity)),
                "active": rules.get("active", True),
                "meta": rules.get("meta", {}),
                "raw_rule": rules
            }
            normalized.append(nr)
            return normalized

        # otherwise unsupported format
        return normalized

    # -----------------------
    # Core: build repulsion vector and apply it
    # -----------------------
    def _compute_coercion(self, embedding: np.ndarray, anchor: np.ndarray, strength: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute coerced vector given embedding and anchor and scalar strength in [0..max_strength].
        Approach:
          - Remove the component of 'embedding' along anchor by factor 'strength'.
          - coerced = normalize(embedding - strength * (projection of embedding onto anchor))
        Returns (coerced_vector_normalized, repulsion_vector_normalized_or_zero).
        """
        emb = _normalize(embedding)
        anc = _normalize(anchor)
        # If either is zero vector, return embedding and zero repulsion deterministically
        if np.all(emb == 0) or np.all(anc == 0):
            return emb, np.zeros_like(emb)

        proj_coeff = float(np.dot(emb, anc))
        proj_component = proj_coeff * anc
        coerced = emb - float(strength) * proj_component
        coerced = _normalize(coerced)
        # repulsion direction is the projection component removed; if near-zero, return zero vector
        if np.linalg.norm(proj_component) <= 1e-12:
            repulsion_vector = np.zeros_like(emb)
        else:
            repulsion_vector = _normalize(proj_component)
        return coerced, repulsion_vector

    # -----------------------
    # Public API: enforce
    # -----------------------
    def enforce(self, embedding: np.ndarray, rules: Any) -> Dict[str, Any]:
        """
        Apply Reg-Vet enforcement on 'embedding' given 'rules'.
        Returns a dictionary with enforcement results and metacognitive feedback.
        """
        emb = _normalize(np.asarray(embedding, dtype=float))
        candidate_rules = self._normalize_rules(rules)

        # Build candidate list: (strength, certainty, angle_score, anchor_vec, rule_meta)
        candidates = []
        for r in candidate_rules:
            if not r.get("active", True):
                continue
            anc = r.get("anchor_vector")
            if anc is None:
                # skip rules without anchor for enforcement, but include metadata for auditability
                continue
            try:
                anc_arr = np.asarray(anc, dtype=float)
                if anc_arr.size == 0:
                    continue
            except Exception:
                continue
            severity = float(r.get("severity", self.default_severity))
            certainty = float(r.get("certainty", 1.0))
            strength = min(self.max_strength, max(0.0, severity * certainty))
            # compute angle score (abs cos sim) for tie-break; if anc/emb zero => 0
            cos_sim = _cosine_similarity(emb, anc_arr)
            angle_score = abs(cos_sim)
            candidates.append((strength, certainty, angle_score, anc_arr, r))

        # If no candidate rules with anchors, return no-op result
        if not candidates:
            return {
                "vector": emb,
                "enforced": False,
                "impact": 0.0,
                "applied_rule": None,
                "repulsion_vector": np.zeros_like(emb),
                "recommendation": {"adjustment": 0.0, "reason": "no_anchor_rule"}
            }

        # Sorting: deterministic tie-break:
        # primary: strength desc; secondary: certainty desc; tertiary: angle closeness desc (angle_score)
        # We want rules that are strongest, most certain, and most relevant (largest abs(cos_sim)).
        candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)

        # Select top candidate
        strength, certainty, angle_score, anchor_vec, rule_meta = candidates[0]

        # Enforce only if strength >= enforcement_threshold (inclusive)
        if strength < self.enforcement_threshold:
            return {
                "vector": emb,
                "enforced": False,
                "impact": 0.0,
                "applied_rule": rule_meta,
                "repulsion_vector": np.zeros_like(emb),
                "recommendation": {"adjustment": 0.0, "reason": "below_threshold"}
            }

        # Try projector if provided (validate output)
        coerced_vec = None
        repulsion_vec = None
        used_projector = False
        if self.projector is not None:
            try:
                # projector may expect a dict; provide clear interface
                proj_input = {"embedding": emb.copy(), "anchor": anchor_vec.copy(), "strength": float(strength), "rule": copy.deepcopy(rule_meta)}
                proj_out = self.projector(proj_input)
                coerced_vec = _normalize(np.asarray(proj_out, dtype=float))
                delta = emb - coerced_vec
                repulsion_vec = _normalize(delta) if np.linalg.norm(delta) > 1e-12 else np.zeros_like(emb)
                used_projector = True
                logger.info(f"RegVet: external projector used for rule_id={str(rule_meta.get('id'))[:8]}")
            except Exception:
                logger.exception("RegVet: external projector failed; falling back to internal coercion")
                coerced_vec = None
                repulsion_vec = None

        # Internal coercion fallback
        if coerced_vec is None:
            coerced_vec, repulsion_vec = self._compute_coercion(emb, anchor_vec, strength)

        # Impact metric: either angular or divergence depending on config
        cos_sim = _cosine_similarity(emb, coerced_vec)
        if self.impact_angular:
            impact = _angular_impact_from_cosine(cos_sim)
        else:
            impact = _divergence_from_cosine_sim(cos_sim)

        # Metacognitive recommendation computation
        expected_scale = strength  # heuristic
        adjustment = 0.0
        reason = "within_expected"
        # if impact is much larger than expected_scale * 1.5 => reduce severity
        if impact > min(1.0, expected_scale * 1.5):
            adjustment = -0.1 * (impact / (expected_scale + 1e-12))
            reason = "overly_strong_enforcement"
        elif impact < max(1e-4, expected_scale * 0.1):
            adjustment = 0.1 * (1.0 - impact)
            reason = "ineffective_enforcement"
        else:
            adjustment = 0.0
            reason = "within_expected"

        # Clamp adjustment to avoid oscillations
        adjustment_clamped = float(np.clip(adjustment, self.adjustment_clamp[0], self.adjustment_clamp[1]))

        recommendation = {
            "adjustment": adjustment_clamped,
            "raw_adjustment": float(adjustment),
            "reason": reason,
            "observed_impact": float(impact),
            "strength": float(strength),
            "severity": float(rule_meta.get("severity", self.default_severity)),
            "certainty": float(rule_meta.get("certainty", 1.0)),
            "rule_id": rule_meta.get("id")
        }

        result = {
            "vector": coerced_vec,
            "enforced": True,
            "impact": float(impact),
            "applied_rule": rule_meta,
            "repulsion_vector": _normalize(np.asarray(repulsion_vec, dtype=float)) if np.linalg.norm(repulsion_vec) > 1e-12 else np.zeros_like(emb),
            "recommendation": recommendation,
            "used_projector": used_projector
        }

        # Audit logging (INFO)
        logger.info(f"RegVet.enforce -> rule_id={str(rule_meta.get('id'))[:12]} "
                    f"strength={strength:.4f} certainty={certainty:.4f} angle_score={angle_score:.4f} "
                    f"impact={impact:.4f} adj={adjustment_clamped:.4f} projector={used_projector}")

        return result

    # -----------------------
    # Serialization (for PCVS/auditoria)
    # -----------------------
    def serialize_state(self) -> Dict[str, Any]:
        """Return serializable param state for auditing."""
        return {
            "enforcement_threshold": self.enforcement_threshold,
            "default_severity": self.default_severity,
            "max_strength": self.max_strength,
            "has_projector": self.projector is not None,
            "impact_angular": self.impact_angular,
            "adjustment_clamp": list(self.adjustment_clamp)
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Restore parameters from a serialized state."""
        if not state:
            return
        self.enforcement_threshold = float(state.get("enforcement_threshold", self.enforcement_threshold))
        self.default_severity = float(state.get("default_severity", self.default_severity))
        self.max_strength = float(state.get("max_strength", self.max_strength))
        self.impact_angular = bool(state.get("impact_angular", self.impact_angular))
        ac = state.get("adjustment_clamp", None)
        if ac:
            self.adjustment_clamp = (float(ac[0]), float(ac[1]))

# -----------------------
# Quick demo / self-test
# -----------------------
if __name__ == "__main__":
    # Deterministic tests demonstrating tie-break and adjustments
    try:
        import numpy as _np

        rv = RegVet(impact_angular=True)

        # Two anchors: one aligned with embedding, one opposite; different severity/certainty to test tie-break
        emb = _np.zeros(64, dtype=float)
        emb[0:2] = _np.array([0.6, 0.8])  # direction (0,1)
        emb = _normalize(emb)

        # anchor A: very aligned, medium severity/certainty
        ancA = _np.zeros(64, dtype=float); ancA[0:2] = _np.array([0.59, 0.81]); ancA = _normalize(ancA)
        ruleA = {"id": "A", "anchor_vector": ancA.tolist(), "certainty": 0.9, "severity": 0.7, "active": True}

        # anchor B: slightly less aligned but higher strength (severity*certainty)
        ancB = _np.zeros(64, dtype=float); ancB[0:2] = _np.array([0.5, 0.8660254]); ancB = _normalize(ancB)
        ruleB = {"id": "B", "anchor_vector": ancB.tolist(), "certainty": 1.0, "severity": 0.63, "active": True}

        out = rv.enforce(emb, [ruleA, ruleB])
        print("Selected rule id:", out["applied_rule"]["id"])
        print("Impact:", out["impact"])
        print("Recommendation:", out["recommendation"])

        # Test anchor zero handling
        ruleC = {"id": "C", "anchor_vector": [0.0]*64, "certainty": 1.0, "severity": 1.0, "active": True}
        out2 = rv.enforce(emb, [ruleC])
        print("Enforce with zero anchor -> enforced:", out2["enforced"], "impact:", out2["impact"])

        # Test threshold behavior: very small strength
        small_rule = {"id": "small", "anchor_vector": ancA.tolist(), "certainty": 0.000001, "severity": 0.000001, "active": True}
        rv_low = RegVet(enforcement_threshold=1e-8)
        out3 = rv_low.enforce(emb, [small_rule])
        print("Small strength enforced (threshold 1e-8)?", out3["enforced"], "impact:", out3["impact"])

    except Exception:
        logger.exception("RegVet demo failed")

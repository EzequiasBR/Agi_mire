# core/intelligence/oea.py
"""
OEA V4.9 — Observador Ético e Avaliador (refatorado, integrado, seguro)

Principais melhorias:
- validação robusta de vetores (dtype/shape/L2)
- contrato validado para rule_base_api.check(...)
- memória ética com aging/decay/pruning
- prevenção de rollbacks em loop (cooldown)
- publicação de eventos padronizados OEA_* via control_bus/monitor/prag (best-effort)
- integração segura com regvet_api (clamp, repulsion merge)
- homeostase integrada e decisão com telemetria
- logs estruturados (JSON-like) via setup_logger
- vetores float32 e normalizados; clamps aplicados onde necessário
- todas as classes originais mantidas (EmotionalSignal, EthicalVerdict, HomeostasisAction, RiskPreventiveVector, OEAConfig, EmotionalEvaluator, EthicalValidator, HomeostasisController, EthicalMemory, OEAEngine)
"""

from __future__ import annotations

import time
import math
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Iterable, Tuple

import numpy as np

# Prefer project utilities; provide safe fallbacks
try:
    from core.services.utils import setup_logger, normalize_vector, deterministic_hash
except Exception:
    # minimal fallbacks
    import hashlib, json as _json

    def setup_logger(name: str) -> logging.Logger:
        logger = logging.getLogger(name)
        if not logger.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s"))
            logger.addHandler(ch)
            logger.setLevel(logging.INFO)
        return logger

    def normalize_vector(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        a = np.asarray(v, dtype=np.float32)
        n = np.linalg.norm(a)
        if n <= eps:
            return np.zeros_like(a)
        return a / n

    def deterministic_hash(obj: Any) -> str:
        try:
            s = _json.dumps(obj, sort_keys=True, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o), separators=(",", ":"))
        except Exception:
            s = str(obj)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

logger = setup_logger("OEA")

# Standardized event names
OEA_EVENT_VIOLATION = "OEA_VIOLATION"
OEA_EVENT_MEMORY_STORED = "OEA_MEMORY_STORED"
OEA_EVENT_HOMEOSTASIS = "OEA_HOMEOSTASIS"
OEA_EVENT_PREVENTIVE_VECTOR = "OEA_PREVENTIVE_VECTOR"
OEA_EVENT_ERROR = "OEA_ERROR"

# =========================
# Tipos de Dados (mantidos)
# =========================
@dataclass
class EmotionalSignal:
    weight: float
    vector: np.ndarray
    meta: Dict[str, Any]


@dataclass
class EthicalVerdict:
    gravity: float
    violated: bool
    repulsion_vector: Optional[np.ndarray]
    rule_id: Optional[str]
    details: Dict[str, Any]


@dataclass
class HomeostasisAction:
    reduce_creativity: bool
    apply_pathogen: bool
    pathogen_intensity: float
    meta: Dict[str, Any]


@dataclass
class RiskPreventiveVector:
    vector: np.ndarray
    magnitude: float
    source_id: Optional[str]


# =========================
# Configuração (ampliada)
# =========================
@dataclass
class OEAConfig:
    ethical_soft_min: float = 0.1
    ethical_partial_min: float = 0.5
    ethical_partial_max: float = 0.7
    ethical_hard_min: float = 0.8
    max_pathogen_intensity: float = 0.8
    preventive_magnitude_max: float = 0.35
    emotion_sigmoid_k: float = 6.0
    emotion_sigmoid_mid: float = 0.5
    alpha_recurring_error: float = 0.15
    ethical_memory_decay_per_sec: float = 0.001  # decay per second for gravity
    ethical_memory_prune_threshold: float = 1e-4  # gravity below this will be pruned
    preventive_similarity_threshold: float = 0.4
    rollback_cooldown_seconds: float = 5.0  # avoid repeated rollbacks
    clamp_preventive_magnitude: float = 0.35
    clamp_repulsion_magnitude: float = 0.8


# =========================
# Avaliador Emocional
# =========================
class EmotionalEvaluator:
    def __init__(self, config: OEAConfig):
        self.config = config

    def evaluate_context(self, vector_context: np.ndarray, meta: Dict[str, Any]) -> EmotionalSignal:
        # validate input
        vc = self._validate_vector(vector_context, "context_vector")
        # simple axes: arousal ~ projection on +1 vector, valence ~ projection on -1 vector
        if vc.size == 0:
            arousal, valence = 0.0, 0.0
        else:
            arousal_axis = self._unit(np.ones(vc.shape, dtype=np.float32))
            valence_axis = self._unit(-np.ones(vc.shape, dtype=np.float32))
            arousal = float(np.dot(self._unit(vc), arousal_axis))
            valence = float(np.dot(self._unit(vc), valence_axis))

        raw_weight = 0.5 * abs(arousal) + 0.5 * (1.0 - abs(valence))
        weight = self._sigmoid(raw_weight)

        emo_vec = normalize_vector(np.array([arousal, valence], dtype=np.float32))
        details = {"raw_weight": raw_weight, **(meta or {})}
        logger.debug({"event": "OEA_EMOTIONAL_EVALUATION", "raw_weight": raw_weight, "weight": weight, "meta": meta})
        return EmotionalSignal(weight=weight, vector=emo_vec, meta=details)

    def _sigmoid(self, x: float) -> float:
        k = float(self.config.emotion_sigmoid_k)
        mid = float(self.config.emotion_sigmoid_mid)
        # numerically stable sigmoid
        try:
            z = -k * (x - mid)
            if z >= 0:
                return float(1.0 / (1.0 + math.exp(z)))
            else:
                ez = math.exp(z)
                return float(ez / (1.0 + ez))
        except Exception:
            return float(0.5)

    def _unit(self, v: np.ndarray) -> np.ndarray:
        return self._validate_vector(v, "arousal/valence_axis", allow_zero=True) / (np.linalg.norm(v) + 1e-12)

    def _validate_vector(self, v: np.ndarray, name: str = "vector", allow_zero: bool = True) -> np.ndarray:
        arr = np.asarray(v, dtype=np.float32)
        if arr.ndim != 1:
            # flatten to 1D if possible
            arr = arr.flatten()
        if not allow_zero and np.linalg.norm(arr) <= 1e-12:
            raise ValueError(f"{name} is zero vector")
        return arr


# =========================
# Validador Ético
# =========================
class EthicalValidator:
    def __init__(self, config: OEAConfig, rule_base_api):
        self.config = config
        self.rule_base_api = rule_base_api

    def validate_trajectory(self, logical_triplet: Dict[str, Any], context_vector: np.ndarray, emotion_weight: float) -> EthicalVerdict:
        # Validate inputs
        if not isinstance(logical_triplet, dict):
            raise ValueError("logical_triplet must be a dict")

        # Call rule_base_api in a safe, contract-checked manner
        try:
            result = self.rule_base_api.check(logical_triplet, {"context_vector_shape": int(np.asarray(context_vector).size)})
        except Exception as e:
            logger.exception("rule_base_api.check raised exception")
            # Fail-safe: no violation assumed if rules check fails
            return EthicalVerdict(0.0, False, None, None, {"checked": False, "error": str(e)})

        # Contract normalization and validation
        if not isinstance(result, dict):
            logger.error("rule_base_api.check returned unexpected type; expected dict")
            return EthicalVerdict(0.0, False, None, None, {"checked": False, "error": "invalid_contract"})

        violated = bool(result.get("violated", False))
        gravity = float(result.get("gravity", 0.0)) if result.get("gravity") is not None else 0.0
        gravity = float(np.clip(gravity, 0.0, 1.0))
        rule_id = result.get("rule_id", None)
        details = result.get("meta", {}) if isinstance(result.get("meta", {}), dict) else {}

        if not violated:
            return EthicalVerdict(0.0, False, None, rule_id, {"checked": True})

        # Build repulsion vector: direction opposite to context_vector, magnitude scaled by gravity & emotion weight
        cv = np.asarray(context_vector, dtype=np.float32)
        if cv.size == 0 or np.linalg.norm(cv) <= 1e-12:
            # fallback direction: zero-vector repulsion (can't determine direction)
            repulsion = np.zeros_like(cv, dtype=np.float32)
        else:
            direction = -self._unit(cv)
            magnitude = float(np.clip(gravity * emotion_weight, 0.0, self.config.clamp_repulsion_magnitude))
            repulsion = direction * magnitude

        repulsion = repulsion.astype(np.float32)
        logger.info({"event": "OEA_VERDICT", "rule_id": rule_id, "violated": True, "gravity": gravity, "magnitude": float(np.linalg.norm(repulsion))})
        return EthicalVerdict(gravity, True, repulsion, rule_id, {"checked": True, **details})

    def _unit(self, v: np.ndarray) -> np.ndarray:
        a = np.asarray(v, dtype=np.float32)
        n = np.linalg.norm(a) + 1e-12
        return a / n


# =========================
# Homeostase
# =========================
class HomeostasisController:
    def __init__(self, config: OEAConfig):
        self.config = config

    def decide(self, metrics: Dict[str, Any]) -> HomeostasisAction:
        volatility = float(metrics.get("volatility", 0.0))
        avg_D = float(metrics.get("avg_D", 0.0))
        rollback_rate = float(metrics.get("rollback_rate", 0.0))
        buffer_saturation = float(metrics.get("buffer_saturation", 0.0))

        # Rules for homeostasis — tunable
        reduce_creativity = (volatility > 0.10) or (rollback_rate > 0.15) or (buffer_saturation > 0.80)
        apply_pathogen = (avg_D > 0.20 and volatility > 0.08)

        pathogen_intensity = 0.0
        if apply_pathogen:
            pathogen_intensity = min(self.config.max_pathogen_intensity, 0.3 + avg_D * 0.5)

        action = HomeostasisAction(reduce_creativity, apply_pathogen, float(pathogen_intensity), {"volatility": volatility, "avg_D": avg_D})
        logger.debug({"event": "OEA_HOMEOSTASIS_DECISION", **action.__dict__})
        return action


# =========================
# Memória Ética (aging/decay/prune)
# =========================
class EthicalMemory:
    def __init__(self, config: OEAConfig):
        self.config = config
        # internal store: list of dicts {"ts": float, "vector": np.ndarray, "gravity": float, "rule_id": str}
        self._store: List[Dict[str, Any]] = []

    def store_violation_vector(self, vector: np.ndarray, gravity: float, rule_id: Optional[str] = None):
        v = np.asarray(vector, dtype=np.float32)
        # normalize stored vector for stable comparisons
        if v.size == 0:
            return
        v_n = normalize_vector(v)
        entry = {"ts": time.time(), "vector": v_n.astype(np.float32), "gravity": float(np.clip(gravity, 0.0, 1.0)), "rule_id": rule_id}
        self._store.append(entry)
        logger.info({"event": "OEA_MEMORY_STORE", "rule_id": rule_id, "gravity": entry["gravity"], "ts": entry["ts"]})

    def anticipate_risk(self, current_vector: np.ndarray) -> Optional[RiskPreventiveVector]:
        if not self._store:
            return None
        u_curr = self._unit(current_vector)
        best = None
        best_sim = -1.0
        for item in self._store:
            u_vec = self._unit(item["vector"])
            sim = float(np.dot(u_curr, u_vec))
            if sim > best_sim:
                best_sim = sim
                best = item
        if best is None or best_sim < float(self.config.preventive_similarity_threshold):
            return None
        direction = -self._unit(best["vector"])
        magnitude = min(self.config.preventive_magnitude_max, 0.2 + best["gravity"] * 0.3)
        preventive = direction * magnitude
        logger.debug({"event": "OEA_PREVENTIVE_COMPUTED", "best_rule": best.get("rule_id"), "sim": best_sim, "magnitude": magnitude})
        return RiskPreventiveVector(vector=preventive.astype(np.float32), magnitude=float(magnitude), source_id=best.get("rule_id"))

    def apply_decay_and_prune(self):
        """Decay gravity over time and prune negligible entries."""
        now = time.time()
        new_store = []
        dec_per_sec = float(self.config.ethical_memory_decay_per_sec)
        prune_thr = float(self.config.ethical_memory_prune_threshold)
        for item in self._store:
            age = max(0.0, now - float(item.get("ts", now)))
            decayed = float(item["gravity"] * (max(0.0, 1.0 - dec_per_sec * age)))
            if decayed >= prune_thr:
                new_item = dict(item)
                new_item["gravity"] = decayed
                new_store.append(new_item)
        removed = len(self._store) - len(new_store)
        if removed > 0:
            logger.debug({"event": "OEA_MEMORY_PRUNED", "removed": removed})
        self._store = new_store

    def iter_items(self) -> Iterable[Dict[str, Any]]:
        for item in list(self._store):
            yield item

    def delete(self, item: Dict[str, Any]):
        try:
            self._store.remove(item)
        except ValueError:
            pass

    def update(self, item: Dict[str, Any]):
        # replace by id match if present; otherwise ignore
        for idx, it in enumerate(self._store):
            if it is item:
                self._store[idx] = item
                return

    def _unit(self, v: np.ndarray) -> np.ndarray:
        a = np.asarray(v, dtype=np.float32)
        n = np.linalg.norm(a) + 1e-12
        return a / n


# =========================
# Engine Principal do OEA (V4.9)
# =========================
class OEAEngine:
    def __init__(self, config: Optional[OEAConfig] = None, rule_base_api=None, regvet_api=None, prag_api=None, monitor_api=None, control_bus=None):
        self.config = config or OEAConfig()
        self.rule_base = rule_base_api
        self.regvet = regvet_api
        self.prag = prag_api
        self.monitor = monitor_api
        self.control_bus = control_bus

        self.evaluator = EmotionalEvaluator(self.config)
        self.validator = EthicalValidator(self.config, self.rule_base)
        self.homeostasis = HomeostasisController(self.config)
        self.ethical_memory = EthicalMemory(self.config)

        # cooldown for rollbacks to avoid loops
        self._last_rollback_ts: float = 0.0

    # Public API: process_cycle maintains method name compatibility
    def process_cycle(self, cycle_id: str, context_vector: np.ndarray, logical_triplet: Dict[str, Any], system_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing pipeline for a single cognitive cycle:
          1) emotional evaluation
          2) ethical validation (contract-checked)
          3) store violation (with decay/prune)
          4) anticipate risk (preventive vector)
          5) homeostasis decision
          6) possibly request rollback via PRAG (with cooldown)
          7) publish events and telemetry
        Returns a dictionary with structured outputs.
        """
        ts_start = time.time()
        # Input validation
        try:
            cv = self._validate_vector(context_vector, "context_vector")
        except Exception as e:
            logger.exception("Invalid context_vector in OEA.process_cycle")
            self._publish_event(OEA_EVENT_ERROR, {"cycle_id": cycle_id, "error": str(e)})
            return {"error": "invalid_context_vector"}

        # 1) Emotional signal
        emo = self.evaluator.evaluate_context(cv, {"cycle_id": cycle_id})

        # 2) Ethical validation (contract with rule_base_api)
        try:
            verdict = self.validator.validate_trajectory(logical_triplet, cv, emo.weight)
        except Exception as e:
            logger.exception("EthicalValidator failed", exc_info=True)
            # safe fallback: treat as no violation but log
            verdict = EthicalVerdict(0.0, False, None, None, {"checked": False, "error": str(e)})

        # 3) If violated -> store repulsion vector (clamped) and publish
        preventive = None
        if verdict.violated:
            rep = verdict.repulsion_vector if verdict.repulsion_vector is not None else np.zeros_like(cv)
            # clamp magnitude
            mag = float(min(self.config.clamp_repulsion_magnitude, np.linalg.norm(rep)))
            rep_unit = np.zeros_like(rep) if np.linalg.norm(rep) <= 1e-12 else (rep / (np.linalg.norm(rep) + 1e-12))
            rep_clamped = (rep_unit * mag).astype(np.float32)
            # store with gravity
            try:
                self.ethical_memory.store_violation_vector(rep_clamped, float(verdict.gravity), verdict.rule_id)
                self._publish_event(OEA_EVENT_MEMORY_STORED, {"cycle_id": cycle_id, "rule_id": verdict.rule_id, "gravity": float(verdict.gravity)})
            except Exception:
                logger.exception("ethical_memory.store_violation_vector failed", exc_info=True)

            # optionally invoke regvet to check repulsion compatibility (best-effort)
            try:
                if self.regvet and hasattr(self.regvet, "enforce"):
                    # regvet may accept repulsion vectors to validate or propose a mitigator
                    rv = self.regvet.enforce(rep_clamped, [{"id": f"OEA:{verdict.rule_id}", "anchor_vector": rep_clamped.tolist(), "certainty": 1.0, "severity": verdict.gravity, "active": True}])
                    # combine or log regvet suggestion as meta
                    self._publish_event("OEA_REGVET_CHECK", {"cycle_id": cycle_id, "regvet_result": rv})
            except Exception:
                logger.debug("regvet integration failed", exc_info=True)

        # 4) anticipate risk (preventive vector)
        try:
            preventive = self.ethical_memory.anticipate_risk(cv)
            if preventive:
                # clamp magnitude
                preventive.vector = normalize_vector(preventive.vector) * min(preventive.magnitude, self.config.clamp_preventive_magnitude)
                self._publish_event(OEA_EVENT_PREVENTIVE_VECTOR, {"cycle_id": cycle_id, "magnitude": preventive.magnitude, "source": preventive.source_id})
        except Exception:
            logger.exception("anticipate_risk failed", exc_info=True)
            preventive = None

        # 5) homeostasis decision
        try:
            h_action = self.homeostasis.decide(system_metrics)
            self._publish_event(OEA_EVENT_HOMEOSTASIS, {"cycle_id": cycle_id, "action": h_action.__dict__})
        except Exception:
            logger.exception("homeostasis.decide failed", exc_info=True)
            h_action = HomeostasisAction(False, False, 0.0, {})

        # 6) Aging / decay of ethical memory (periodic)
        try:
            self.ethical_memory.apply_decay_and_prune()
        except Exception:
            logger.exception("ethical_memory.apply_decay_and_prune failed", exc_info=True)

        # 7) If severe violation -> request PRAG rollback (with cooldown)
        rollback_requested = False
        rollback_info = None
        try:
            if verdict.violated and self.prag:
                gravity = float(verdict.gravity)
                now = time.time()
                if gravity >= self.config.ethical_hard_min and (now - self._last_rollback_ts) >= self.config.rollback_cooldown_seconds:
                    # call prag.rollback_total with guarded arguments (best-effort)
                    try:
                        # prag.rollback_total(snapshot_hash) signature differs across implementations:
                        # we try heuristics: prefer rollback_total(snapshot_hash) or rollback_total(scope)
                        snapshot_hash = f"ETHICAL_HARD:{verdict.rule_id}"
                        # call and record
                        ok = False
                        try:
                            ok = bool(self.prag.rollback_total(snapshot_hash))
                        except TypeError:
                            # maybe signature: rollback_total(snapshot_hash, metadata)
                            try:
                                ok = bool(self.prag.rollback_total(snapshot_hash, {"gravity": gravity}))
                            except Exception:
                                ok = False
                        except Exception:
                            ok = False

                        if ok:
                            rollback_requested = True
                            rollback_info = {"type": "hard", "rule_id": verdict.rule_id, "gravity": gravity}
                            self._last_rollback_ts = now
                            logger.warning({"event": OEA_EVENT_VIOLATION, "cycle_id": cycle_id, "rule_id": verdict.rule_id, "gravity": gravity, "action": "prag_rollback_total"})
                            self._publish_event(OEA_EVENT_VIOLATION, {"cycle_id": cycle_id, "rule_id": verdict.rule_id, "gravity": gravity, "action": "prag_rollback_total"})
                    except Exception:
                        logger.exception("prag.rollback_total failed", exc_info=True)
                elif gravity >= self.config.ethical_partial_min and gravity <= self.config.ethical_partial_max and (now - self._last_rollback_ts) >= self.config.rollback_cooldown_seconds:
                    # partial rollback
                    try:
                        scope = [f"OEA:{verdict.rule_id}"]
                        ok = False
                        try:
                            ok = bool(self.prag.rollback_partial(scope, f"ETHICAL_PARTIAL:{verdict.rule_id}"))
                        except TypeError:
                            try:
                                ok = bool(self.prag.rollback_partial(scope))
                            except Exception:
                                ok = False
                        if ok:
                            rollback_requested = True
                            rollback_info = {"type": "partial", "rule_id": verdict.rule_id, "gravity": gravity}
                            self._last_rollback_ts = now
                            logger.warning({"event": OEA_EVENT_VIOLATION, "cycle_id": cycle_id, "rule_id": verdict.rule_id, "gravity": gravity, "action": "prag_rollback_partial"})
                            self._publish_event(OEA_EVENT_VIOLATION, {"cycle_id": cycle_id, "rule_id": verdict.rule_id, "gravity": gravity, "action": "prag_rollback_partial"})
                    except Exception:
                        logger.exception("prag.rollback_partial failed", exc_info=True)
        except Exception:
            logger.exception("Error evaluating rollback conditions", exc_info=True)

        # 8) Telemetry / monitor ingest (best-effort)
        try:
            telemetry = {
                "cycle_id": cycle_id,
                "emotional_weight": float(emo.weight),
                "violated": bool(verdict.violated),
                "gravity": float(verdict.gravity) if verdict.violated else 0.0,
                "preventive_present": bool(preventive),
                "rollback_requested": bool(rollback_requested),
                "processing_time_ms": (time.time() - ts_start) * 1000.0
            }
            if self.monitor and hasattr(self.monitor, "ingest"):
                try:
                    self.monitor.ingest("OEA_telemetry", telemetry)
                except Exception:
                    # fallback: log
                    logger.debug("monitor.ingest failed", exc_info=True)
            # publish structured log via control_bus if available
            self._publish_event("OEA_TELEMETRY", telemetry)
        except Exception:
            logger.exception("telemetry publish failed", exc_info=True)

        outputs = {
            "emotional_signal": emo,
            "ethical_verdict": verdict,
            "homeostasis_action": h_action,
            "preventive_vector": preventive,
            "rollback_info": rollback_info,
            "telemetry": telemetry
        }
        return outputs

    # =========================
    # helpers
    # =========================
    def _validate_vector(self, v: np.ndarray, name: str = "vector") -> np.ndarray:
        arr = np.asarray(v, dtype=np.float32)
        if arr.ndim == 0:
            arr = arr.reshape(-1)
        if arr.ndim != 1:
            # flatten conservatively
            arr = arr.flatten()
        # don't accept NaN/Inf
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            raise ValueError(f"{name} contains NaN/Inf")
        return arr

    def _publish_event(self, name: str, payload: Dict[str, Any]):
        payload = dict(payload)
        payload.setdefault("ts", time.time())
        try:
            if self.control_bus and hasattr(self.control_bus, "publish"):
                self.control_bus.publish(name, payload)
        except Exception:
            # attempt monitor fallback
            try:
                if self.monitor and hasattr(self.monitor, "ingest"):
                    self.monitor.ingest(name, payload)
            except Exception:
                logger.debug("No event sink available for OEA event", exc_info=True)

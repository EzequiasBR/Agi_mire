# core/mch.py
"""
Master Control Hub (MCH) V3 - integrando Analytics, Adaptation, Monitor e PCVS.

Responsabilidades:
 - Orquestrar o ciclo cognitivo completo (percepção, recuperação, validação, adaptação).
 - Integrar Analytics para calcular H, V, E (feedback sistêmico).
 - Persistir snapshots via PCVS e coordenar rollback via PRAG.
 - Expor APIs simples: execute_cycle(input_data) e shutdown().
"""

from __future__ import annotations
import time
import logging
import json
from typing import Any, Dict, Optional, Tuple, List

import numpy as np

# --------------------------
# Imports dos módulos reais (com fallbacks para desenvolvimento local)
# --------------------------
try:
    from ..ol import OL
    from ..governance.prag import PRAG
    from ..intelligence.ppo import PPO
    from ..memory.hippocampus import Hippocampus
    from .adaptation import Adaptation
    from ..pcvs import PCVS
    from .security import Security
    from .nlp_bridge import NLPBridge
    from .analytics import Analytics
    from .monitor import Monitor
    from .utils import setup_logger
except Exception as _exc:
    # Fallbacks simples e seguros para desenvolvimento sem toda a infra
    import hashlib

    def setup_logger(name: str):
        logger = logging.getLogger(name)
        if not logger.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s"))
            logger.addHandler(ch)
        logger.setLevel(logging.INFO)
        return logger

    logger = setup_logger("MCH-Fallback")

    # Mock classes implementing minimal required interface
    class OL:
        def __init__(self, *args, **kwargs):
            self.dim = kwargs.get("dim", 768)
        def embed_and_hash(self, text):
            digest = hashlib.sha256(str(text).encode("utf-8")).hexdigest()
            seed = int(digest[:16], 16) % (2 ** 31 - 1)
            rng = np.random.RandomState(seed)
            v = rng.randn(self.dim).astype(np.float32)
            v /= (np.linalg.norm(v) + 1e-12)
            return v, digest
        def serialize_state(self): return {"dim": self.dim}
        def load_state(self, s): pass

    class PRAG:
        def __init__(self, *args, **kwargs):
            self.divergence_threshold = 0.85
            self.partial_threshold = 0.70
            self.last_hash = None
            self.last_trigger = False
            self.rollback_threshold = 0.05
        def update_rollback_threshold(self, val): self.rollback_threshold = val
        def should_rollback(self, H_sist, last_hash): 
            # simple: rollback if H below threshold
            return H_sist < 0.2
        def snapshot_state(self): return {"divergence_threshold": self.divergence_threshold}
        def load_state(self, s): pass

    class PPO:
        def __init__(self, tau=0.90):
            self.tau = float(tau)
            self.tau_sist = 0.75
            self.last_reason = None
            self.last_lo_triggered = False
        def update_tau(self, tau): self.tau = tau
        def should_trigger(self, D, C, E_sist):
            if E_sist >= self.tau_sist:
                self.last_reason = f"E_SYSTEMIC({E_sist:.3f})"; return True
            if C >= self.tau:
                self.last_reason = f"PRIMAL_CONFIDENCE(C={C:.3f})"; return True
            if D <= (1.0 - self.tau) / 2.0:
                self.last_reason = f"STRUCTURAL_STABILITY(D={D:.4f})"; return True
            self.last_reason = None; return False
        def snapshot_state(self): return {"tau": self.tau}

    class Hippocampus:
        def __init__(self, dim=768, decay_lambda=1e-4):
            self.dim = dim
            self.decay_lambda = decay_lambda
            self.memory_store = {}  # key -> (vec, meta)
        def top_k(self, q, k=5):
            if not self.memory_store: return []
            qn = q / (np.linalg.norm(q) + 1e-12)
            sims = []
            for k_id, rec in self.memory_store.items():
                v = np.asarray(rec["vec"], dtype=float)
                s = float(np.dot(qn, v / (np.linalg.norm(v) + 1e-12)))
                sims.append((k_id, s))
            sims.sort(key=lambda x: x[1], reverse=True)
            return sims[:k]
        def store(self, key, P0=0.5, payload=None, vec=None):
            v = np.asarray(vec if vec is not None else np.zeros(self.dim), dtype=float)
            v = v / (np.linalg.norm(v) + 1e-12)
            self.memory_store[key] = {"vec": v.tolist(), "meta": payload or {}, "p0": float(P0)}
        def confidence(self, key):
            rec = self.memory_store.get(key)
            if not rec: return 0.0
            # simulate decay-based confidence
            return min(1.0, rec.get("p0", 0.5))
        def snapshot_state(self):
            return {"memory_store": self.memory_store.copy()}
        def load_state(self, state):
            self.memory_store = state.get("memory_store", {}).copy()
        def save_checkpoint(self, pcvs=None):
            # simple metadata
            return {"index_meta": {"count": len(self.memory_store)}}

    class Adaptation:
        def __init__(self): self._tau_ppo = 0.90; self._delta_h_prag = 0.05; self._lambda_hippo = 1e-4
        def update_parameters(self, H, V): pass
        def get_ppo_tau(self): return self._tau_ppo
        def get_prag_rollback_threshold(self): return self._delta_h_prag
        def get_hippocampus_lambda(self): return self._lambda_hippo
        def snapshot_state(self): return {"tau_ppo": self._tau_ppo}

    class PCVS:
        def __init__(self, base_dir="pcvs"):
            self.base_dir = base_dir
            self._store = {}
        def save(self, snapshot):
            import hashlib, time
            payload = json.dumps(snapshot, sort_keys=True, default=str).encode("utf-8")
            sha = hashlib.sha256(payload).hexdigest()
            self._store[sha] = snapshot
            return sha
        def load(self, sha256=None):
            if sha256 is None: return None
            return self._store.get(sha256)
        def rollback(self, hippocampus, sha):
            st = self._store.get(sha)
            if st and "hippocampus" in st:
                hippocampus.load_state(st["hippocampus"])

    class Security:
        def hash_vector(self, v):
            import hashlib
            return hashlib.sha256(np.asarray(v, dtype=np.float32).tobytes()).hexdigest()

    class NLPBridge:
        def __init__(self, dim=768): self._dim = dim
        def get_embedding_dimension(self): return self._dim
        def get_embedding(self, text, normalize=True):
            digest = hashlib.sha256(str(text).encode("utf-8")).hexdigest()
            seed = int(digest[:16], 16) % (2 ** 31 - 1)
            rng = np.random.RandomState(seed)
            v = rng.randn(self._dim).astype(np.float32)
            if normalize:
                v /= (np.linalg.norm(v) + 1e-12)
            return v

    class Analytics:
        def __init__(self):
            self.window_size = 100
            self.confidence_history = []
            self.divergence_history = []
            self.lo_trigger_history = []
            self.rollback_history = []
            self.last_metrics = {"H": 0.5, "V": 0.5, "E": 0.5}
        def compute_metrics(self, new_C, new_D, rollback_triggered, lo_triggered):
            import numpy as _np
            self.confidence_history.append(new_C)
            self.divergence_history.append(new_D)
            self.rollback_history.append(1 if rollback_triggered else 0)
            self.lo_trigger_history.append(1 if lo_triggered else 0)
            self.confidence_history = self.confidence_history[-self.window_size:]
            self.divergence_history = self.divergence_history[-self.window_size:]
            C_med = _np.mean(self.confidence_history) if self.confidence_history else 0.5
            D_med = _np.mean(self.divergence_history) if self.divergence_history else 0.5
            C_var = _np.var(self.confidence_history) if len(self.confidence_history) > 1 else 0.0
            rollback_rate = _np.mean(self.rollback_history) if self.rollback_history else 0.0
            lo_rate = _np.mean(self.lo_trigger_history) if self.lo_trigger_history else 0.0
            H = min(1.0, max(0.0, 0.6 * C_med + 0.3 * (1.0 - D_med) + 0.1 * (1.0 - rollback_rate)))
            V = min(1.0, max(0.0, C_var * 5.0 + rollback_rate * 0.5))
            E = min(1.0, max(0.0, lo_rate * 0.7 + (1.0 - C_med) * 0.3))
            self.last_metrics = {"H": float(H), "V": float(V), "E": float(E)}
            return self.last_metrics

    class Monitor:
        def __init__(self):
            self.metrics = {"cycle_count": 0}
            self.event_log = []
        def register_cycle_end(self, result_data):
            self.metrics["cycle_count"] += 1
            self.event_log.append(result_data)
        def get_status_report(self):
            return {"metrics": self.metrics, "events": len(self.event_log)}
        def load_snapshot(self, state): pass

    # ensure we have a logger in fallback
    logger = setup_logger("MCH-Fallback")

# --------------------------
# Logger for the MCH
# --------------------------
logger = setup_logger("MCH")

# --------------------------
# Helper numeric utilities
# --------------------------
def _normalize_vector(v: np.ndarray) -> np.ndarray:
    a = np.asarray(v, dtype=float)
    n = np.linalg.norm(a)
    if n <= 1e-12:
        return np.zeros_like(a)
    return a / n

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_n = _normalize_vector(a)
    b_n = _normalize_vector(b)
    if a_n.shape != b_n.shape:
        # adapt shape by trunc/pad
        min_len = min(a_n.size, b_n.size)
        return float(np.dot(a_n[:min_len], b_n[:min_len]))
    return float(np.dot(a_n, b_n))

def _divergence_from_cosine_sim(cos_sim: float) -> float:
    return float(max(0.0, min(1.0, (1.0 - cos_sim) / 2.0)))

# --------------------------
# MCH Class
# --------------------------
class MCH:
    def __init__(self, persistence_dir: str = "pcvs_data", vector_dim: int = 768):
        logger.info("MCH v3 initializing...")
        # Services
        self.nlp = NLPBridge()          # embedding provider
        self.sec = Security()           # security/hashing
        self.adapt = Adaptation()       # adaptation service
        self.pcvs = PCVS(base_dir=f"{persistence_dir}/pcvs")  # pcvs persistence
        self.analytics = Analytics()    # analytics service
        self.monitor = Monitor()        # monitor service

        # Core modules
        self.ol = OL(dim=vector_dim)    # organism linguistic
        self.prag = PRAG()              # governance
        self.ppo = PPO(tau=self.adapt.get_ppo_tau() if hasattr(self.adapt, "get_ppo_tau") else 0.90)
        self.hippocampus = Hippocampus(dim=vector_dim, decay_lambda=self.adapt.get_hippocampus_lambda())

        # Operational state
        self.cycle_count = 0
        self.last_state_hash: Optional[str] = None
        self.H_sist = 0.5
        self.V_sist = 0.5
        self.E_sist = 0.5

        logger.info("MCH v3 ready. vector_dim=%d", vector_dim)

    # --------------------------
    # Compose PCVS snapshot
    # --------------------------
    def _compose_system_state(self) -> Dict[str, Any]:
        return {
            "timestamp": time.time(),
            "ol": getattr(self.ol, "serialize_state", lambda: {})(),
            "prag": getattr(self.prag, "snapshot_state", lambda: {})(),
            "ppo": getattr(self.ppo, "snapshot_state", lambda: {})(),
            "adapt": getattr(self.adapt, "snapshot_state", lambda: {})(),
            "hippocampus": getattr(self.hippocampus, "snapshot_state", lambda: {})(),
            "analytics": getattr(self.analytics, "snapshot_state", lambda: {})(),
            "monitor": getattr(self.monitor, "get_status_report", lambda: {})()
        }

    def save_pcvs_snapshot(self) -> Optional[str]:
        try:
            snap = self._compose_system_state()
            h = self.pcvs.save(snap)
            self.last_state_hash = h
            logger.info("PCVS snapshot saved: %s", h[:12])
            return h
        except Exception:
            logger.exception("PCVS snapshot save failed")
            return None

    def load_pcvs_snapshot(self, sha: Optional[str] = None) -> Optional[Dict[str, Any]]:
        sha = sha or self.last_state_hash
        if not sha:
            logger.warning("No PCVS hash available to load")
            return None
        try:
            st = self.pcvs.load(sha256=sha)
            logger.info("PCVS snapshot loaded: %s", sha[:12])
            return st
        except Exception:
            logger.exception("PCVS.load failed")
            return None

    # --------------------------
    # Rollback helpers
    # --------------------------
    def rollback_total(self) -> Dict[str, Any]:
        logger.warning("Performing rollback TOTAL (attempt)")
        if not self.last_state_hash:
            logger.error("No last_state_hash available for rollback")
            return {"restored": False, "reason": "no_hash"}
        st = self.load_pcvs_snapshot(self.last_state_hash)
        if not st:
            return {"restored": False, "reason": "load_failed"}
        try:
            if "hippocampus" in st and hasattr(self.hippocampus, "load_state"):
                self.hippocampus.load_state(st["hippocampus"])
            if "ol" in st and hasattr(self.ol, "load_state"):
                self.ol.load_state(st["ol"])
            if "monitor" in st and hasattr(self.monitor, "load_snapshot"):
                self.monitor.load_snapshot(st["monitor"])
            logger.info("Rollback TOTAL executed from hash %s", self.last_state_hash[:12])
            return {"restored": True, "pcvs_hash": self.last_state_hash}
        except Exception:
            logger.exception("Rollback TOTAL failed during restore")
            return {"restored": False, "reason": "restore_exception"}

    def rollback_partial(self) -> Dict[str, Any]:
        logger.warning("Performing rollback PARTIAL (attempt)")
        if not self.last_state_hash:
            logger.error("No last_state_hash available for rollback_partial")
            return {"restored": False, "reason": "no_hash"}
        st = self.load_pcvs_snapshot(self.last_state_hash)
        if not st:
            return {"restored": False, "reason": "load_failed"}
        try:
            if "hippocampus" in st and hasattr(self.hippocampus, "load_state"):
                self.hippocampus.load_state(st["hippocampus"])
            if "monitor" in st and hasattr(self.monitor, "load_snapshot"):
                self.monitor.load_snapshot(st["monitor"])
            logger.info("Rollback PARTIAL executed from hash %s", self.last_state_hash[:12])
            return {"restored": True, "pcvs_hash": self.last_state_hash}
        except Exception:
            logger.exception("Rollback PARTIAL failed")
            return {"restored": False, "reason": "restore_exception"}

    # --------------------------
    # Core cycle
    # --------------------------
    def execute_cycle(self, input_data: Any, inject_pathogen: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Runs one cognitive cycle:
         - embed (OL)
         - retrieve (Hippocampus top_k)
         - compute C and D
         - analytics.compute_metrics -> (H, V, E)
         - PRAG decision (rollback)
         - PPO decision (LO)
         - adaptation update
         - persistence (PCVS) periodically
         - monitor register
        Returns result dict (for logging/testing).
        """
        cycle_start = time.time()
        self.cycle_count += 1
        logger.info("Cycle %d start. input='%s'...", self.cycle_count, str(input_data)[:80])

        # 1) Perception / Embedding
        try:
            vec, vec_hash = self.ol.embed_and_hash(input_data)
            vec = _normalize_vector(vec)
        except Exception:
            logger.exception("OL.embed_and_hash failed; using deterministic fallback")
            vec_hash = str(time.time())
            vec = np.zeros(self.hippocampus.dim, dtype=float)

        # optional pathogen injection (simulate noise)
        if inject_pathogen:
            intensity = float(inject_pathogen.get("intensity", 0.0))
            seed = inject_pathogen.get("seed", None)
            if intensity > 0.0:
                rng = np.random.RandomState(int(seed) if seed is not None else int(time.time()*1000) % (2**31-1))
                noise = rng.randn(*vec.shape).astype(float)
                noise = noise / (np.linalg.norm(noise) + 1e-12) * intensity * (np.linalg.norm(vec) + 1e-12)
                vec = _normalize_vector(vec + noise)
                logger.info("Pathogen noise injected intensity=%.4f", intensity)

        # 2) Memory retrieval (top-k)
        try:
            topk = self.hippocampus.top_k(vec, k=5)
        except Exception:
            logger.exception("Hippocampus.top_k failed; using empty results")
            topk = []

        # compute C and D (primal signals)
        if topk:
            top_key, top_score = topk[0]
            try:
                # If hippocampus returns key as (id,score) tuple in fallback, handle gracefully
                if isinstance(top_key, (list, tuple)) and len(top_key) == 2:
                    top_key, top_score = top_key
            except Exception:
                pass
            C_primal = float(min(1.0, top_score + self.hippocampus.confidence(top_key)))
            D_primal = float(max(0.0, 1.0 - top_score))
        else:
            C_primal = 0.5
            D_primal = 0.5

        # 3) Analytics: compute H, V, E using previous triggers (stored in modules)
        prev_rollback = bool(getattr(self.prag, "last_trigger", False))
        prev_lo = bool(getattr(self.ppo, "last_lo_triggered", False))
        try:
            metrics = self.analytics.compute_metrics(new_C=C_primal, new_D=D_primal,
                                                     rollback_triggered=prev_rollback,
                                                     lo_triggered=prev_lo)
            H_sist = float(metrics.get("H", self.H_sist))
            V_sist = float(metrics.get("V", self.V_sist))
            E_sist = float(metrics.get("E", self.E_sist))
        except Exception:
            logger.exception("Analytics.compute_metrics failed; using fallback previous metrics")
            H_sist, V_sist, E_sist = self.H_sist, self.V_sist, self.E_sist

        # update internal copy
        self.H_sist, self.V_sist, self.E_sist = H_sist, V_sist, E_sist
        logger.info("Metrics: H=%.3f V=%.3f E=%.3f | C=%.3f D=%.3f", H_sist, V_sist, E_sist, C_primal, D_primal)

        # 4) PRAG: determine rollback action
        try:
            # allow PRAG to be updated with adaptation's rollback threshold if method exists
            prag_threshold = getattr(self.adapt, "get_prag_rollback_threshold", lambda: None)()
            if prag_threshold is not None and hasattr(self.prag, "update_rollback_threshold"):
                self.prag.update_rollback_threshold(prag_threshold)
        except Exception:
            logger.exception("Failed to update PRAG threshold from Adaptation")

        rollback_decision = False
        try:
            # abuse-compatible call: PRAG may implement evaluate_action or should_rollback
            if hasattr(self.prag, "evaluate_action"):
                action, reason = self.prag.evaluate_action(D_primal, V_sist)
                if action in ("ROLLBACK_TOTAL", "ROLLBACK_PARTIAL"):
                    rollback_decision = True
                    self.prag.last_trigger = True
                    self.prag.last_action_reason = reason
                    logger.warning("PRAG decided: %s (%s)", action, reason)
                    if action == "ROLLBACK_TOTAL":
                        rb = self.rollback_total()
                    else:
                        rb = self.rollback_partial()
                    # register and return early with rollback result
                    result = {
                        "action": action.lower(),
                        "pcvs_hash": rb.get("pcvs_hash"),
                        "D": D_primal, "C": C_primal, "H": H_sist, "V": V_sist, "E": E_sist,
                        "duration_s": time.time() - cycle_start
                    }
                    self.monitor.register_cycle_end(result)
                    return result
            else:
                # older interface: should_rollback(H_sist, last_hash)
                rollback_decision = bool(self.prag.should_rollback(H_sist, self.last_state_hash))
                self.prag.last_trigger = rollback_decision
                if rollback_decision:
                    logger.warning("PRAG.should_rollback -> True, performing rollback_partial by default")
                    rb = self.rollback_partial()
                    result = {
                        "action": "rollback_partial",
                        "pcvs_hash": rb.get("pcvs_hash"),
                        "D": D_primal, "C": C_primal, "H": H_sist, "V": V_sist, "E": E_sist,
                        "duration_s": time.time() - cycle_start
                    }
                    self.monitor.register_cycle_end(result)
                    return result
        except Exception:
            logger.exception("PRAG decision flow failed; skipping rollback check")

        # 5) PPO: decide Learning Optimization (LO)
        try:
            # update PPO tau from adaptation
            ppo_tau = getattr(self.adapt, "get_ppo_tau", lambda: getattr(self.ppo, "tau", 0.90))()
            if ppo_tau is not None and hasattr(self.ppo, "update_tau"):
                self.ppo.update_tau(ppo_tau)
        except Exception:
            logger.exception("Failed to update PPO tau from Adaptation")

        lo_decision = False
        try:
            lo_decision = bool(self.ppo.should_trigger(D_primal, C_primal, E_sist))
            self.ppo.last_lo_triggered = lo_decision
            if lo_decision:
                # perform learning optimization action: store into hippocampus and consolidate
                key = f"MCH_C{self.cycle_count}_{vec_hash[:8]}"
                try:
                    self.hippocampus.store(key, P0=float(C_primal), payload={"input": str(input_data)}, vec=vec)
                except Exception:
                    logger.exception("Hippocampus.store during LO failed")
                logger.info("PPO triggered LO: stored key=%s", key)
        except Exception:
            logger.exception("PPO decision failed; proceeding")

        # 6) Adaptation: update parameters based on H and V
        try:
            if hasattr(self.adapt, "update_parameters"):
                self.adapt.update_parameters(H_sist, V_sist)
            # ensure hippocampus has updated decay_lambda if applicable
            if hasattr(self.hippocampus, "decay_lambda") and hasattr(self.adapt, "get_hippocampus_lambda"):
                try:
                    self.hippocampus.decay_lambda = self.adapt.get_hippocampus_lambda()
                except Exception:
                    pass
        except Exception:
            logger.exception("Adaptation update failed")

        # 7) Periodic persistence (PCVS)
        try:
            if self.cycle_count % 5 == 0:
                h = self.save_pcvs_snapshot()
                logger.info("Periodic PCVS snapshot saved: %s", (h or "none"))
        except Exception:
            logger.exception("Periodic PCVS snapshot failed")

        # 8) Register cycle result in Monitor
        result = {
            "action": "continue",
            "D": D_primal, "C": C_primal,
            "H": H_sist, "V": V_sist, "E": E_sist,
            "lo_triggered": lo_decision,
            "rollback_triggered": rollback_decision,
            "pcvs_hash": self.last_state_hash,
            "duration_s": time.time() - cycle_start,
            "timestamp": time.time()
        }
        try:
            self.monitor.register_cycle_end(result)
        except Exception:
            logger.exception("Monitor.register_cycle_end failed")

        logger.info("Cycle %d finished in %.3f s", self.cycle_count, result["duration_s"])
        return result

    # --------------------------
    # Admin
    # --------------------------
    def inspect_last_pcvs(self) -> Dict[str, Any]:
        return {"last_hash": self.last_state_hash, "pcvs_state": self.load_pcvs_snapshot(self.last_state_hash)}

    def force_save_snapshot(self) -> Optional[str]:
        return self.save_pcvs_snapshot()

    def shutdown(self):
        logger.info("MCH shutdown: saving final snapshot and cleaning up")
        try:
            self.save_pcvs_snapshot()
        except Exception:
            logger.exception("Final PCVS save failed")
        logger.info("MCH shutdown complete")

# --------------------------
# Quick local demo (if executed directly)
# --------------------------
if __name__ == "__main__":
    mch = MCH(persistence_dir="/tmp/agi_mire_run", vector_dim=768)

    # Example cycles (these will exercise analytics/adaptation/ppo/prag)
    inputs = [
        ("O clima hoje está agradável com sol moderado.", None),
        ("Uma anomalia grave foi detectada no cluster de dados principal.", None),
        ("Analisando o relatório de anomalia, o impacto é isolado.", None),
        ("A tentativa de correção causou uma nova falha.", None),
        ("O clima hoje está agradável com sol moderado.", None),
    ]

    for i, (txt, inj) in enumerate(inputs, start=1):
        out = mch.execute_cycle(txt, inject_pathogen=inj)
        print(f"\n--- Cycle {i} result ---")
        print(json.dumps(out, indent=2))

    mch.shutdown()

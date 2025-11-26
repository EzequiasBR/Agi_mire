"""
core/simlog.py

Sim-Log (Híbrido patch) - foco em submódulos críticos:
- round_trip, round_trip_valid
- divergence (D) unificada
- deterministic fallback (seeded behavior)
- tensorlog / audit trail (in-memory + optional file sink)
- zero-norm handling and safe normalization
- angular impact option

API principais usadas pelo MCH:
- simlog.round_trip_valid(embedding, recon_embedding) -> (bool, error_rt)
- simlog.round_trip(embedding, recon_embedding, mode="full") -> dict (trace + metrics)
- simlog.max_error (atributo configurável)
- simlog.serialize_state() / load_state()
- simlog.tensorlog_event(event_dict)
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import numpy as np
import time
import hashlib
import json
import logging
import os

logger = logging.getLogger("SimLog")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s SimLog %(levelname)s: %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)


# -----------------------
# Helpers
# -----------------------
def _safe_normalize(v: np.ndarray) -> np.ndarray:
    """Normalize vector deterministically: zero vector => zeros_like."""
    a = np.asarray(v, dtype=float)
    n = np.linalg.norm(a)
    if n <= 1e-12:
        return np.zeros_like(a)
    return a / n


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity in [-1,1]; zero-vector rules yield 0.0 deterministically."""
    a_n = _safe_normalize(a)
    b_n = _safe_normalize(b)
    if np.all(a_n == 0) or np.all(b_n == 0):
        return 0.0
    return float(np.dot(a_n, b_n))


def divergence_from_cosine(cos_sim: float) -> float:
    """Map cosine similarity (-1..1) to divergence [0..1]: D = (1 - cos_sim) / 2"""
    return float(max(0.0, min(1.0, (1.0 - float(cos_sim)) / 2.0)))


def angular_impact_from_cosine(cos_sim: float) -> float:
    """Normalized angular impact in [0,1]: arccos(cos_sim) / pi"""
    c = float(np.clip(cos_sim, -1.0, 1.0))
    return float(np.arccos(c) / np.pi)


def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _embedding_digest(vec: np.ndarray) -> str:
    """Deterministic digest for an embedding (useful for audit)."""
    return _hash_bytes(vec.astype(np.float32).tobytes())


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# -----------------------
# SimLog class
# -----------------------
class SimLog:
    def __init__(self,
                 max_error: float = 0.05,
                 use_angular_impact: bool = False,
                 deterministic_seed: Optional[int] = None,
                 tensorlog_path: Optional[str] = None):
        """
        :param max_error: threshold (error_rt) for valid round-trip (default 0.05)
        :param use_angular_impact: whether to compute impact using angular metric
        :param deterministic_seed: optional seed controlling deterministic fallbacks (if needed)
        :param tensorlog_path: optional file path to append tensorlog events (newline-delimited JSON)
        """
        self.max_error = float(max_error)
        self.use_angular_impact = bool(use_angular_impact)
        self.deterministic_seed = deterministic_seed
        self.tensorlog_path = tensorlog_path

        # in-memory tensorlog buffer for quick inspection; append-only
        self._tensorlog_buffer: list[Dict[str, Any]] = []

        # ensure file exists if path given
        if tensorlog_path:
            try:
                os.makedirs(os.path.dirname(tensorlog_path), exist_ok=True)
                open(tensorlog_path, "a").close()
            except Exception:
                logger.exception("SimLog: could not prepare tensorlog file")

    # -----------------------
    # Core metrics
    # -----------------------
    def divergence(self, a: np.ndarray, b: np.ndarray) -> float:
        """Unified divergence metric D in [0,1] based on cosine similarity."""
        cos = _cosine_similarity(a, b)
        return divergence_from_cosine(cos)

    def impact(self, a: np.ndarray, b: np.ndarray) -> float:
        """Impact metric either angular or divergence depending on config."""
        cos = _cosine_similarity(a, b)
        if self.use_angular_impact:
            return angular_impact_from_cosine(cos)
        else:
            return divergence_from_cosine(cos)

    # -----------------------
    # Round-trip operations
    # -----------------------
    def round_trip(self, embedding: np.ndarray, recon_embedding: np.ndarray, trace: bool = True) -> Dict[str, Any]:
        """
        Compute full round-trip trace and metrics.
        Returns a dict with:
          - embedding_digest, recon_digest
          - cos_sim, error_rt, divergence_D, impact
          - valid_rt (bool)
          - trace (timings + optional extra)
        """
        start = time.time()
        a = np.asarray(embedding, dtype=float)
        b = np.asarray(recon_embedding, dtype=float)

        # normalize safely for internal computations but preserve originals for digest
        a_n = _safe_normalize(a)
        b_n = _safe_normalize(b)

        cos_sim = _cosine_similarity(a, b)
        error_rt = divergence_from_cosine(cos_sim)  # primary error metric
        divergence_D = error_rt  # alias for compatibility
        impact_val = angular_impact_from_cosine(cos_sim) if self.use_angular_impact else error_rt

        valid_rt = error_rt <= self.max_error

        end = time.time()
        trace_obj = {
            "timings": {
                "start_ts": start,
                "end_ts": end,
                "duration_s": end - start
            }
        } if trace else {}

        result = {
            "embedding_digest": _embedding_digest(a),
            "recon_digest": _embedding_digest(b),
            "cos_sim": float(cos_sim),
            "error_rt": float(error_rt),
            "D": float(divergence_D),
            "impact": float(impact_val),
            "valid_rt": bool(valid_rt),
            "trace": trace_obj
        }

        # store in tensorlog for audit
        self.tensorlog_event({
            "event": "round_trip",
            "time": _now_iso(),
            "embedding_digest": result["embedding_digest"],
            "recon_digest": result["recon_digest"],
            "cos_sim": result["cos_sim"],
            "error_rt": result["error_rt"],
            "D": result["D"],
            "impact": result["impact"],
            "valid_rt": result["valid_rt"]
        })

        return result

    def round_trip_valid(self, embedding: np.ndarray, recon_embedding: np.ndarray) -> Tuple[bool, float]:
        """
        Backward-compatible function used by MCH:
        returns (valid_rt: bool, error_rt: float)

        Implements robust fallback: if any exception occurs, compute cosine-based fallback deterministically.
        """
        try:
            res = self.round_trip(embedding, recon_embedding, trace=False)
            return bool(res["valid_rt"]), float(res["error_rt"])
        except Exception:
            # deterministic fallback (no raising)
            try:
                cos = _cosine_similarity(embedding, recon_embedding)
                error_rt = divergence_from_cosine(cos)
                valid_rt = error_rt <= self.max_error
                logger.exception("SimLog.round_trip failed; using cosine-based fallback")
                # log fallback event
                self.tensorlog_event({
                    "event": "round_trip_fallback",
                    "time": _now_iso(),
                    "cos_sim": float(cos),
                    "error_rt": float(error_rt),
                    "valid_rt": bool(valid_rt)
                })
                return bool(valid_rt), float(error_rt)
            except Exception:
                # last resort: safe return
                logger.exception("SimLog.round_trip fallback also failed; returning safe defaults")
                return False, 1.0

    # -----------------------
    # Semantic reconstruction helpers (placeholders)
    # -----------------------
    def reconstruct_text_from_embedding(self, embedding: np.ndarray) -> str:
        """
        Placeholder deterministic semantic reconstruction:
        - Produces a short pseudo-text from embedding digest for traceability.
        - Not intended as a semantic true decoder; OA should provide true reconstruction.
        """
        digest = _embedding_digest(_safe_normalize(embedding))
        # build reproducible pseudo-text (compact)
        return f"<recon:{digest[:12]}>"

    def reconstruct_canonical(self, embedding: np.ndarray) -> Dict[str, Any]:
        """
        Returns a canonical reconstruction artifact useful for Sim-Log trace:
        {
          "text": str,
          "digest": str,
          "meta": {...}
        }
        """
        emb_n = _safe_normalize(embedding)
        digest = _embedding_digest(emb_n)
        txt = self.reconstruct_text_from_embedding(emb_n)
        return {"text": txt, "digest": digest, "meta": {"norm": float(np.linalg.norm(embedding))}}

    # -----------------------
    # Tensorlog / audit trail
    # -----------------------
    def tensorlog_event(self, event: Dict[str, Any]) -> None:
        """
        Append an event to in-memory tensorlog and optionally to file (newline-delimited JSON).
        Event will be shallow-copied and timestamped if not present.
        """
        evt = dict(event)  # shallow copy
        if "time" not in evt:
            evt["time"] = _now_iso()
        # compute concise event id
        evt_id_source = json.dumps(evt, sort_keys=True, default=str).encode("utf-8")
        evt["event_id"] = hashlib.sha256(evt_id_source).hexdigest()
        # append to buffer
        self._tensorlog_buffer.append(evt)
        # optional file append
        if self.tensorlog_path:
            try:
                with open(self.tensorlog_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(evt, ensure_ascii=False) + "\n")
            except Exception:
                logger.exception("SimLog.tensorlog_event: failed to write to file")

    def tensorlog_peek(self, n: int = 10) -> list:
        """Return last n events (copy)."""
        return list(self._tensorlog_buffer[-n:])

    def tensorlog_clear(self) -> None:
        self._tensorlog_buffer.clear()
        if self.tensorlog_path:
            try:
                open(self.tensorlog_path, "w").close()
            except Exception:
                logger.exception("SimLog.tensorlog_clear: failed to clear file")

    # -----------------------
    # State persistence (for PCVS)
    # -----------------------
    def serialize_state(self) -> Dict[str, Any]:
        return {
            "max_error": self.max_error,
            "use_angular_impact": self.use_angular_impact,
            "deterministic_seed": self.deterministic_seed,
            "tensorlog_len": len(self._tensorlog_buffer)
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        if not state:
            return
        self.max_error = float(state.get("max_error", self.max_error))
        self.use_angular_impact = bool(state.get("use_angular_impact", self.use_angular_impact))
        self.deterministic_seed = state.get("deterministic_seed", self.deterministic_seed)
        # note: tensorlog buffer not restored from this simple state

# -----------------------
# Quick demo when executed as script
# -----------------------
if __name__ == "__main__":
    try:
        sim = SimLog(max_error=0.05, use_angular_impact=False)
        import numpy as _np
        v = _np.random.RandomState(0).randn(64)
        v = v / (np.linalg.norm(v) + 1e-12)
        recon_same = v.copy()
        recon_inv = -v.copy()

        print("Roundtrip same:", sim.round_trip_valid(v, recon_same))
        print("Roundtrip inverse:", sim.round_trip_valid(v, recon_inv))
        print("Roundtrip trace:", sim.round_trip(v, recon_same))
        print("Tensorlog peek:", sim.tensorlog_peek())
    except Exception:
        logger.exception("SimLog demo failed")

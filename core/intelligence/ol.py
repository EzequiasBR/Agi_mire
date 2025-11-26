# core/ol.py
"""
Organismo Linguístico (OL) - Refatorado (v3)
Integrações:
 - NLPBridge (embed real, get_embedding_dimension)
 - SecurityService (hash_vector)
 - VectorIndex (opcional) para top-k / persistência (FAISS)
 - Deterministic fallback baseado em SHA256 -> seed -> RandomState
 - Telemetria mínima e serialização compatível com PCVS
"""

from __future__ import annotations
import hashlib
import json
import time
import logging
from typing import Any, Callable, Dict, Optional, List, Tuple

import numpy as np

# Try to import service helpers if available (project layout core/services)
try:
    from .services.utils import setup_logger, _normalize_vector, save_json, load_json
except Exception:
    # Minimal fallbacks
    def setup_logger(name: str) -> logging.Logger:
        logger = logging.getLogger(name)
        if not logger.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s"))
            logger.addHandler(ch)
        logger.setLevel(logging.INFO)
        return logger

    def _normalize_vector(v: np.ndarray) -> np.ndarray:
        a = np.asarray(v, dtype=np.float32)
        n = np.linalg.norm(a)
        if n <= 1e-12:
            return np.zeros_like(a)
        return a / n

    def save_json(path: str, data: Dict[str, Any]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    def load_json(path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


logger = setup_logger("OL")

DEFAULT_DIM = 768


def _sha256_to_int_seed(s: str) -> int:
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:16], 16) % (2 ** 31 - 1)


def _deterministic_embedding(text: str, dim: int = DEFAULT_DIM) -> np.ndarray:
    """
    Deterministic embedding fallback:
      sha256(text) -> seed -> RandomState(seed).randn(dim) -> L2 normalize
    """
    if text is None:
        text = ""
    digest = hashlib.sha256(str(text).encode("utf-8")).hexdigest()
    seed = _sha256_to_int_seed(digest)
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return _normalize_vector(v)


class OL:
    def __init__(self,
                 nlp_bridge: Optional[Any] = None,
                 security_service: Optional[Any] = None,
                 vector_index: Optional[Any] = None,
                 dim: int = DEFAULT_DIM,
                 use_projector_only: bool = False):
        """
        :param nlp_bridge: instance implementing .get_embedding(text) and optionally .get_embedding_dimension()
        :param security_service: instance implementing .hash_vector(np.ndarray) -> str
        :param vector_index: optional VectorIndex-like providing .add(vector, meta) and .top_k(query,k)
        :param dim: desired embedding dimensionality
        :param use_projector_only: if True, fail when nlp_bridge returns None
        """
        self.nlp_bridge = nlp_bridge
        self.security_service = security_service
        self.vector_index = vector_index
        self.dim = int(dim)
        self.use_projector_only = bool(use_projector_only)

        # Store: key -> {"vec": list[float], "ts": float, "meta": dict}
        self.vector_store: Dict[str, Dict[str, Any]] = {}

        # Telemetry / events
        self.telemetry: Dict[str, List[Dict[str, Any]]] = {"events": []}

        # If bridge exposes dimension, reconcile (prefer bridge)
        try:
            if self.nlp_bridge and hasattr(self.nlp_bridge, "get_embedding_dimension"):
                bd = int(self.nlp_bridge.get_embedding_dimension())
                if bd != self.dim:
                    logger.info("OL: NLPBridge dimension %d differs from configured %d. Adopting bridge dim.", bd, self.dim)
                    self.dim = bd
        except Exception:
            logger.exception("OL: failed to query NLPBridge dimension; keeping configured dim=%d", self.dim)

        logger.info("OL initialized (dim=%d) projector=%s use_projector_only=%s idx=%s",
                    self.dim, bool(self.nlp_bridge), self.use_projector_only, bool(self.vector_index))

    # -----------------------
    # Embedding + Hash (central)
    # -----------------------
    def embed_and_hash(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, str]:
        """
        Returns (normalized_vector, vector_hash).
        Tries NLPBridge first; deterministic fallback when necessary.
        """
        text = input_data if isinstance(input_data, str) else str(input_data)

        # 1) NLPBridge attempt
        vector = None
        if self.nlp_bridge is not None:
            try:
                # prefer call signature get_embedding(text, context)
                if hasattr(self.nlp_bridge, "get_embedding"):
                    vector = self.nlp_bridge.get_embedding(text) if context is None else self.nlp_bridge.get_embedding(text, context=context)
                elif hasattr(self.nlp_bridge, "embed"):
                    vector = self.nlp_bridge.embed(text)
                else:
                    logger.debug("OL: NLPBridge present but has no get_embedding/embed method")
            except Exception:
                logger.exception("OL: NLPBridge failed, will fallback to deterministic embedding")
                vector = None

        # If the projector returned None and use_projector_only is set -> error-handling by caller
        if vector is None:
            if self.use_projector_only:
                logger.error("OL: projector-only mode and NLPBridge returned None")
                # produce deterministic zero-hash to indicate failure but be deterministic
                vec = np.zeros(self.dim, dtype=np.float32)
                h = self._hash_vector(vec)
                self._record_telemetry("embed_fail_projector_only", {"input_sample": text[:120], "hash": h[:10]})
                return vec, h
            # fallback deterministic
            vec = _deterministic_embedding(text, dim=self.dim)
            h = self._hash_vector(vec)
            self._record_telemetry("embed_fallback", {"input_sample": text[:120], "hash": h[:10]})
            return vec, h

        # Ensure vector shape/dtype and adapt if needed
        vec_np = np.asarray(vector, dtype=np.float32)
        if vec_np.ndim != 1:
            # try flatten / squeeze
            vec_np = vec_np.reshape(-1)
        # Adapt dimension if different (deterministic resample/pad/truncate)
        if vec_np.shape[0] != self.dim:
            logger.warning("OL: embedding dim mismatch (%d != %d). Adapting deterministically.", vec_np.shape[0], self.dim)
            # Create a deterministic vector derived from the original + text: hash(original_bytes + text)
            digest = hashlib.sha256(vec_np.tobytes() + text.encode("utf-8")).hexdigest()
            vec_np = _deterministic_embedding(digest, dim=self.dim)

        vec_np = _normalize_vector(vec_np)
        h = self._hash_vector(vec_np)
        self._record_telemetry("embed_success", {"input_sample": text[:120], "hash": h[:10]})
        return vec_np, h

    def _hash_vector(self, vector: np.ndarray) -> str:
        """
        Use SecurityService if available, else sha256 of float32 bytes (deterministic).
        """
        try:
            if self.security_service and hasattr(self.security_service, "hash_vector"):
                return self.security_service.hash_vector(vector)
        except Exception:
            logger.exception("OL: SecurityService.hash_vector failed; falling back to local hash")

        # local deterministic hash
        return hashlib.sha256(np.asarray(vector, dtype=np.float32).tobytes(order='C')).hexdigest()

    # -----------------------
    # CRUD vector store
    # -----------------------
    def upsert_vector(self, key: str, vector: Any, meta: Optional[Dict[str, Any]] = None, index_add: bool = True) -> None:
        v = np.asarray(vector, dtype=np.float32)
        v = _normalize_vector(v)
        self.vector_store[key] = {"vec": v.tolist(), "ts": time.time(), "meta": meta or {}}
        self._record_telemetry("upsert", {"key": key})
        logger.info("OL upsert_vector: %s", key)
        # optionally add to external vector_index for scalable top-k
        if index_add and self.vector_index is not None:
            try:
                meta_idx = {"key": key}
                self.vector_index.add(v, meta=meta_idx)
            except Exception:
                logger.exception("OL: vector_index.add failed for key=%s", key)

    def get_vector(self, key: str) -> Optional[np.ndarray]:
        rec = self.vector_store.get(key)
        if not rec:
            return None
        return np.asarray(rec["vec"], dtype=np.float32)

    def delete_vector(self, key: str) -> bool:
        if key in self.vector_store:
            del self.vector_store[key]
            self._record_telemetry("delete", {"key": key})
            logger.info("OL delete_vector: %s", key)
            # Note: external vector index deletion not handled (depends on index implementation)
            return True
        return False

    def update_adaptive_vector(self, key: str, target_vector: Any, lr: float = 0.1, momentum: float = 0.9) -> bool:
        rec = self.vector_store.get(key)
        tgt = _normalize_vector(np.asarray(target_vector, dtype=np.float32))
        if rec is None:
            self.upsert_vector(key, tgt)
            return True
        old = np.asarray(rec["vec"], dtype=np.float32)
        delta = tgt - old
        candidate = old + float(lr) * delta
        blended = (1.0 - float(momentum)) * old + float(momentum) * candidate
        new = _normalize_vector(blended)
        rec["vec"] = new.tolist()
        rec["ts"] = time.time()
        self._record_telemetry("update_adaptive", {"key": key, "lr": float(lr), "momentum": float(momentum)})
        logger.info("OL update_adaptive_vector: %s (lr=%.3f momentum=%.3f)", key, lr, momentum)
        # Optionally update vector_index (if supported)
        try:
            if self.vector_index is not None and hasattr(self.vector_index, "add"):
                # best-effort: add new vector (duplicates may exist)
                self.vector_index.add(new, {"key": key})
        except Exception:
            logger.exception("OL: vector_index update failed for key=%s", key)
        return True

    # -----------------------
    # kNN / top-k: prefers vector_index else numpy fallback
    # -----------------------
    def top_k(self, query_vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        q = _normalize_vector(np.asarray(query_vector, dtype=np.float32))
        # 1) vector_index if present
        if self.vector_index is not None and hasattr(self.vector_index, "top_k"):
            try:
                results = self.vector_index.top_k(q, k)
                # Expected format from index: list of {"key": key, "id": id, "score": float, "meta": {...}}
                return results
            except Exception:
                logger.exception("OL: vector_index.top_k failed; falling back to numpy kNN")

        # 2) numpy brute-force fallback
        out: List[Tuple[str, float, Dict[str, Any]]] = []
        for key, rec in self.vector_store.items():
            v = np.asarray(rec["vec"], dtype=np.float32)
            if v.size != q.size:
                continue
            score = float(np.dot(q, v))
            out.append((key, score, rec.get("meta", {})))
        out.sort(key=lambda x: x[1], reverse=True)
        return [{"key": k, "score": float(s), "meta": m} for k, s, m in out[:k]]

    # -----------------------
    # Serialization / state
    # -----------------------
    def serialize_state(self) -> Dict[str, Any]:
        return {
            "dim": int(self.dim),
            "vector_store": {k: {"vec": v["vec"], "ts": v["ts"], "meta": v.get("meta", {})}
                             for k, v in self.vector_store.items()},
            "telemetry": dict(self.telemetry)
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        if not state:
            return
        self.dim = int(state.get("dim", self.dim))
        vs = state.get("vector_store", {})
        self.vector_store = {}
        for k, rec in vs.items():
            vec = rec.get("vec")
            if vec is None:
                continue
            arr = np.asarray(vec, dtype=np.float32)
            if arr.shape[-1] != self.dim:
                logger.warning("OL.load_state: vector %s has wrong dim (%d != %d); skipping", k, arr.shape[-1], self.dim)
                continue
            self.vector_store[k] = {"vec": _normalize_vector(arr).tolist(), "ts": float(rec.get("ts", time.time())), "meta": rec.get("meta", {})}
        self.telemetry = state.get("telemetry", {"events": []})
        logger.info("OL.load_state: loaded %d vectors", len(self.vector_store))

    def dump_state_json(self) -> str:
        return json.dumps(self.serialize_state(), ensure_ascii=False, indent=2)

    # -----------------------
    # Utilities / Telemetry
    # -----------------------
    def _record_telemetry(self, event: str, payload: Optional[Dict[str, Any]] = None) -> None:
        e = {"event": event, "ts": time.time(), "payload": payload or {}}
        self.telemetry.setdefault("events", []).append(e)

    def stats(self) -> Dict[str, Any]:
        return {
            "vectors": len(self.vector_store),
            "dim": self.dim,
            "telemetry_events": len(self.telemetry.get("events", [])),
            "has_vector_index": bool(self.vector_index)
        }

# -----------------------
# Quick integration test snippet (run as script)
# -----------------------
if __name__ == "__main__":
    # Minimal mock services if core/services not available
    class MockNLP:
        def get_embedding(self, text, context=None):
            # deterministic but different from OL fallback
            seed = int(hashlib.sha256((text + "_mock").encode()).hexdigest()[:16], 16) % (2**31 - 1)
            rng = np.random.RandomState(seed)
            return rng.randn(DEFAULT_DIM).astype(np.float32)

        def get_embedding_dimension(self):
            return DEFAULT_DIM

    class MockSecurity:
        def hash_vector(self, v: np.ndarray) -> str:
            return hashlib.sha256(np.asarray(v, dtype=np.float32).tobytes(order='C')).hexdigest()

    mock_nlp = MockNLP()
    mock_sec = MockSecurity()
    ol = OL(nlp_bridge=mock_nlp, security_service=mock_sec, dim=DEFAULT_DIM)

    text = "um cachorro late na rua"
    vec, h = ol.embed_and_hash(text)
    print("Embed norm:", float(np.linalg.norm(vec)))
    print("Hash (prefix):", h[:12])

    key = "evt:1"
    ol.upsert_vector(key, vec, meta={"source": "demo"})
    # adaptive update
    ol.update_adaptive_vector(key, vec * 0.95, lr=0.05, momentum=0.8)
    print("Top-1:", ol.top_k(vec, k=1))
    print("Stats:", ol.stats())

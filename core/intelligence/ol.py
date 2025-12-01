# core/intelligence/ol.py
"""
Organismo Linguístico (OL) — V2.0 (Produção, PRAG-ready)

Melhorias aplicadas em relação à versão anterior:
 - validação estrita do retorno do NLPBridge
 - opção de lançar erro em projector-only mode (raise_on_projector_only)
 - fallback determinístico com Generator moderno (default_rng) opcional
 - validação do hash retornado por SecurityService
 - telemetry enriquecida (eventos padronizados)
 - LRU in-memory store opcional para evitar vazamento de memória
 - eventos publicados no ControlBus: OL_EMBED_FAIL, OL_EMBED_FALLBACK,
   OL_EMBED_SUCCESS, OL_PLASTICITY_APPLIED, OL_VECTOR_EVICTED
 - contratos de vector_index esperados documentados (best-effort)
 - serialize/load state compatíveis com PCVS
 - logging estruturado
"""

from __future__ import annotations
import hashlib
import json
import time
import logging
from typing import Any, Callable, Dict, Optional, List, Tuple

import numpy as np
from collections import OrderedDict

# Fallback utility imports (if project utils are available they will be used)
try:
    from ..services.utils import setup_logger, normalize_vector as _normalize_vector
except Exception:
    # minimal fallback implementations
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
        n = float(np.linalg.norm(a))
        if n <= 1e-12:
            return np.zeros_like(a)
        return a / n

logger = setup_logger("OL_V2")

DEFAULT_DIM = 768


def _sha256_to_int_seed(s: str) -> int:
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    # use 32-bit-compatible seed range for RNG
    return int(h[:16], 16) % (2 ** 31 - 1)


def _deterministic_embedding_fallback(text: str, dim: int = DEFAULT_DIM, use_modern_rng: bool = True) -> np.ndarray:
    """
    Deterministic fallback embedding.
    If use_modern_rng True, uses numpy.default_rng with fixed seed; else uses RandomState.
    """
    if text is None:
        text = ""
    seed = _sha256_to_int_seed(str(text))
    if use_modern_rng:
        # modern Generator deterministic for a fixed seed
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(dim).astype(np.float32)
    else:
        rng = np.random.RandomState(seed)
        v = rng.randn(dim).astype(np.float32)
    return _normalize_vector(v)


class OL:
    """
    Organismo Linguístico - V2.0

    :param nlp_bridge: object with get_embedding(text[, context]) and optional get_embedding_dimension()
    :param security_service: object with hash_vector(np.ndarray) -> str
    :param vector_index: optional index with .add(vector, meta) and .top_k(query, k)
    :param dim: target embedding dimension
    :param use_projector_only: if True, treat missing projector output as fatal (configurable)
    :param raise_on_projector_only: when True and use_projector_only True, raise RuntimeError when projector absent/returns None
    :param max_in_memory_vectors: LRU capacity; None => unlimited
    :param use_modern_rng: True uses numpy.default_rng for fallback determinism; False uses RandomState
    """

    def __init__(
        self,
        nlp_bridge: Optional[Any] = None,
        security_service: Optional[Any] = None,
        vector_index: Optional[Any] = None,
        dim: int = DEFAULT_DIM,
        use_projector_only: bool = False,
        raise_on_projector_only: bool = False,
        max_in_memory_vectors: Optional[int] = None,
        use_modern_rng: bool = True,
        control_bus: Optional[Any] = None,
    ):
        self.nlp_bridge = nlp_bridge
        self.security_service = security_service
        self.vector_index = vector_index
        self.dim = int(dim)
        self.use_projector_only = bool(use_projector_only)
        self.raise_on_projector_only = bool(raise_on_projector_only)
        self.max_in_memory_vectors = max_in_memory_vectors
        self.use_modern_rng = bool(use_modern_rng)
        self.control_bus = control_bus

        # LRU store for vectors: key -> {"vec": list, "ts": float, "meta": dict}
        if self.max_in_memory_vectors is None:
            # simple dict if unlimited
            self.vector_store: Dict[str, Dict[str, Any]] = {}
        else:
            self.vector_store = OrderedDict()

        # telemetry
        self.telemetry: Dict[str, List[Dict[str, Any]]] = {"events": []}

        # reconcile dim with bridge if available
        try:
            if self.nlp_bridge and hasattr(self.nlp_bridge, "get_embedding_dimension"):
                bd = int(self.nlp_bridge.get_embedding_dimension())
                if bd != self.dim:
                    logger.info("OL: adopting bridge dimension %d (configured %d)", bd, self.dim)
                    self.dim = bd
        except Exception:
            logger.exception("OL: failed to query NLPBridge dimension; keeping configured dim=%d", self.dim)

        logger.info("OL V2.0 initialized dim=%d projector=%s use_projector_only=%s max_mem=%s RNG_modern=%s idx=%s",
                    self.dim, bool(self.nlp_bridge), self.use_projector_only, str(self.max_in_memory_vectors), self.use_modern_rng, bool(self.vector_index))

    # -----------------------
    # Embedding + Hash (central)
    # -----------------------
    def embed_and_hash(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, str]:
        """
        Retorna (normalized_vector, vector_hash).
        Ordem de tentativa:
         1) NLPBridge.get_embedding(text, context) or .embed(text)
         2) Deterministic fallback (_deterministic_embedding_fallback)

        Regras principais:
         - Validate shape: must be 1-D
         - Adapt dimension deterministically if mismatch (hash(original_bytes + text))
         - If use_projector_only and projector returns None: either raise or return deterministic zero-vector depending on config
        """
        text = input_data if isinstance(input_data, str) else str(input_data)

        vector = None
        # Attempt projector / bridge
        if self.nlp_bridge is not None:
            try:
                # prefer signature get_embedding(text, context=...)
                if hasattr(self.nlp_bridge, "get_embedding"):
                    # some bridges accept context kw, others don't; try both safely
                    try:
                        vector = self.nlp_bridge.get_embedding(text) if context is None else self.nlp_bridge.get_embedding(text, context=context)
                    except TypeError:
                        vector = self.nlp_bridge.get_embedding(text)
                elif hasattr(self.nlp_bridge, "embed"):
                    vector = self.nlp_bridge.embed(text)
                else:
                    logger.debug("OL: nlp_bridge present but lacking get_embedding/embed")
            except Exception:
                logger.exception("OL: nlp_bridge raised; falling back to deterministic embedding")
                vector = None

        # Handle projector-only mode
        if vector is None:
            if self.use_projector_only:
                # fatal or deterministic signaling depending on config
                msg = "OL: projector-only mode and nlp_bridge returned None"
                logger.error(msg)
                self._record_telemetry("embed_fail_projector_only", {"input_sample": text[:120]})
                if self.control_bus:
                    try:
                        self.control_bus.publish("OL_EMBED_FAIL", {"reason": "projector_only", "sample": text[:120]})
                    except Exception:
                        logger.debug("control_bus.publish OL_EMBED_FAIL failed", exc_info=True)
                if self.raise_on_projector_only:
                    raise RuntimeError(msg)
                # deterministic zero-vector with stable hash (caller can detect)
                vec = np.zeros(self.dim, dtype=np.float32)
                h = self._hash_vector(vec)
                self._record_telemetry("embed_projector_only_zero", {"hash": h[:12]})
                return vec, h
            # deterministic fallback
            vec = _deterministic_embedding_fallback(text, dim=self.dim, use_modern_rng=self.use_modern_rng)
            h = self._hash_vector(vec)
            self._record_telemetry("embed_fallback", {"input_sample": text[:120], "hash": h[:12]})
            if self.control_bus:
                try:
                    self.control_bus.publish("OL_EMBED_FALLBACK", {"sample": text[:120], "hash": h})
                except Exception:
                    logger.debug("control_bus.publish OL_EMBED_FALLBACK failed", exc_info=True)
            return vec, h

        # Validate and normalize output from projector
        vec_np = None
        try:
            vec_np = np.asarray(vector, dtype=np.float32)
        except Exception:
            logger.exception("OL: failed to convert projector output to ndarray; falling back")
            vec_np = None

        if vec_np is None or vec_np.ndim != 1:
            # try to flatten if possible
            try:
                vec_np = np.asarray(vector, dtype=np.float32).reshape(-1)
            except Exception:
                # fallback deterministic
                logger.warning("OL: projector returned incompatible shape; falling back deterministically")
                vec = _deterministic_embedding_fallback(text, dim=self.dim, use_modern_rng=self.use_modern_rng)
                h = self._hash_vector(vec)
                self._record_telemetry("embed_fallback_shape", {"sample": text[:120], "hash": h[:12]})
                if self.control_bus:
                    try:
                        self.control_bus.publish("OL_EMBED_FALLBACK", {"sample": text[:120], "reason": "shape_mismatch", "hash": h})
                    except Exception:
                        logger.debug("control_bus.publish OL_EMBED_FALLBACK failed", exc_info=True)
                return vec, h

        # If dim mismatch, deterministically adapt
        if vec_np.shape[0] != self.dim:
            logger.warning("OL: embedding dim mismatch (%d != %d). Generating deterministic adaptation.", vec_np.shape[0], self.dim)
            # derive deterministic string from original bytes + text
            digest = hashlib.sha256(vec_np.tobytes(order='C') + text.encode('utf-8')).hexdigest()
            vec = _deterministic_embedding_fallback(digest, dim=self.dim, use_modern_rng=self.use_modern_rng)
            h = self._hash_vector(vec)
            self._record_telemetry("embed_adapt_dim", {"orig_dim": int(vec_np.shape[0]), "target_dim": self.dim, "hash": h[:12]})
            if self.control_bus:
                try:
                    self.control_bus.publish("OL_EMBED_ADAPT_DIM", {"sample": text[:120], "orig_dim": int(vec_np.shape[0]), "target_dim": self.dim, "hash": h})
                except Exception:
                    logger.debug("control_bus.publish OL_EMBED_ADAPT_DIM failed", exc_info=True)
            return vec, h

        # Normal path: normalize and hash
        vec = _normalize_vector(vec_np)
        h = self._hash_vector(vec)
        self._record_telemetry("embed_success", {"sample": text[:120], "hash": h[:12]})
        if self.control_bus:
            try:
                self.control_bus.publish("OL_EMBED_SUCCESS", {"sample": text[:120], "hash": h})
            except Exception:
                logger.debug("control_bus.publish OL_EMBED_SUCCESS failed", exc_info=True)
        return vec, h

    def _hash_vector(self, vector: np.ndarray) -> str:
        """
        Prefer security_service.hash_vector; validate returned value (hex string).
        Fallback to sha256 of float32 bytes.
        """
        # try external service
        try:
            if self.security_service and hasattr(self.security_service, "hash_vector"):
                h = self.security_service.hash_vector(vector)
                if not isinstance(h, str) or len(h) < 8:
                    logger.warning("OL: security_service.hash_vector returned invalid value; falling back.")
                else:
                    return h
        except Exception:
            logger.exception("OL: security_service.hash_vector raised; falling back")

        # local fallback: sha256 of float32 bytes
        return hashlib.sha256(np.asarray(vector, dtype=np.float32).tobytes(order='C')).hexdigest()

    # -----------------------
    # CRUD vector store (LRU optional)
    # -----------------------
    def _ensure_capacity_and_insert(self, key: str, record: Dict[str, Any]) -> None:
        """
        Insert into vector_store respecting LRU capacity if configured.
        """
        if self.max_in_memory_vectors is None:
            self.vector_store[key] = record
            return

        # OrderedDict path
        # if key exists, move to end; else insert and evict if needed
        if key in self.vector_store:
            self.vector_store.move_to_end(key)
            self.vector_store[key] = record
            return

        self.vector_store[key] = record
        # evict oldest if exceeding capacity
        if len(self.vector_store) > self.max_in_memory_vectors:
            evicted_key, _ = self.vector_store.popitem(last=False)
            logger.info("OL: evicted vector %s due to memory cap (%d)", evicted_key, self.max_in_memory_vectors)
            self._record_telemetry("vector_evicted", {"key": evicted_key})
            if self.control_bus:
                try:
                    self.control_bus.publish("OL_VECTOR_EVICTED", {"key": evicted_key})
                except Exception:
                    logger.debug("control_bus.publish OL_VECTOR_EVICTED failed", exc_info=True)

    def upsert_vector(self, key: str, vector: Any, meta: Optional[Dict[str, Any]] = None, index_add: bool = True) -> None:
        v = np.asarray(vector, dtype=np.float32)
        v = _normalize_vector(v)
        rec = {"vec": v.tolist(), "ts": time.time(), "meta": meta or {}}
        self._ensure_capacity_and_insert(key, rec)
        self._record_telemetry("upsert", {"key": key})
        logger.info("OL upsert_vector: %s", key)
        # optionally add to external vector_index for scalable top-k; expected contract:
        # vector_index.add(vector: np.ndarray, meta: dict) -> returns id or None
        if index_add and self.vector_index is not None:
            try:
                if hasattr(self.vector_index, "add"):
                    self.vector_index.add(v, meta={"key": key})
            except Exception:
                logger.exception("OL: vector_index.add failed for key=%s", key)

    def get_vector(self, key: str) -> Optional[np.ndarray]:
        rec = self.vector_store.get(key)
        if not rec:
            return None
        # move-to-end for LRU
        if isinstance(self.vector_store, OrderedDict) and key in self.vector_store:
            self.vector_store.move_to_end(key)
        return np.asarray(rec["vec"], dtype=np.float32)

    def delete_vector(self, key: str) -> bool:
        if key in self.vector_store:
            del self.vector_store[key]
            self._record_telemetry("delete", {"key": key})
            logger.info("OL delete_vector: %s", key)
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
                self.vector_index.add(new, {"key": key})
        except Exception:
            logger.exception("OL: vector_index update failed for key=%s", key)
        return True

    # -----------------------
    # kNN / top-k
    # -----------------------
    def top_k(self, query_vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        q = _normalize_vector(np.asarray(query_vector, dtype=np.float32))
        # prefer vector_index if available and implements top_k(query,k)
        if self.vector_index is not None and hasattr(self.vector_index, "top_k"):
            try:
                results = self.vector_index.top_k(q, k)
                return results
            except Exception:
                logger.exception("OL: vector_index.top_k failed; falling back to numpy")

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
    # Cycle helpers (unchanged names preserved)
    # -----------------------
    def generate_vector_adaptativo(self, current_input: Any) -> Tuple[np.ndarray, str]:
        logger.info("OL: Generating V_adapt (insight).")
        V_adapt, h_adapt = self.embed_and_hash(current_input)
        return V_adapt, h_adapt

    def apply_plasticity_correction(self, key_adapt: str, V_desvio: np.ndarray, D: float, lr: float = 0.5) -> bool:
        V_adapt_old = self.get_vector(key_adapt)
        if V_adapt_old is None:
            logger.warning("OL.apply_plasticity_correction: key %s not found.", key_adapt[:10])
            self._record_telemetry("plasticity_fail_key_missing", {"key": key_adapt[:10], "D": D})
            return False

        V_corrected = V_adapt_old + np.asarray(V_desvio, dtype=np.float32) * float(lr)
        self.upsert_vector(key_adapt, V_corrected, index_add=True)
        self._record_telemetry("plasticity_applied", {"key": key_adapt[:10], "D": D, "lr_used": float(lr)})
        logger.info("OL plasticity applied: %s (D=%.3f, LR=%.2f)", key_adapt[:10], D, lr)
        if self.control_bus:
            try:
                self.control_bus.publish("OL_PLASTICITY_APPLIED", {"key": key_adapt, "D": D, "lr": lr})
            except Exception:
                logger.debug("control_bus.publish OL_PLASTICITY_APPLIED failed", exc_info=True)
        return True

    # -----------------------
    # Serialization / state
    # -----------------------
    def serialize_state(self) -> Dict[str, Any]:
        # emit stable serializable state for PCVS
        return {
            "dim": int(self.dim),
            "vector_store": {k: {"vec": v["vec"], "ts": v["ts"], "meta": v.get("meta", {})} for k, v in (self.vector_store.items() if not isinstance(self.vector_store, OrderedDict) else dict(self.vector_store).items())},
            "telemetry": dict(self.telemetry),
            "config": {"max_in_memory_vectors": self.max_in_memory_vectors, "use_modern_rng": self.use_modern_rng}
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        if not state:
            return
        self.dim = int(state.get("dim", self.dim))
        vs = state.get("vector_store", {})
        if self.max_in_memory_vectors is not None:
            self.vector_store = OrderedDict()
        else:
            self.vector_store = {}
        for k, rec in vs.items():
            vec = rec.get("vec")
            if vec is None:
                continue
            arr = np.asarray(vec, dtype=np.float32)
            if arr.shape[-1] != self.dim:
                logger.warning("OL.load_state: skipping vector %s with wrong dim (%d != %d)", k, arr.shape[-1], self.dim)
                continue
            self._ensure_capacity_and_insert(k, {"vec": _normalize_vector(arr).tolist(), "ts": float(rec.get("ts", time.time())), "meta": rec.get("meta", {})})
        self.telemetry = state.get("telemetry", {"events": []})
        logger.info("OL.load_state: loaded %d vectors", len(self.vector_store))

    def dump_state_json(self) -> str:
        return json.dumps(self.serialize_state(), ensure_ascii=False, indent=2)

    # -----------------------
    # Utilities / Telemetry / Stats
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

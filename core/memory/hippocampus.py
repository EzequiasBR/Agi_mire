"""
Hippocampus V5.0 — Armazenamento híbrido (AOL + FAISS + PCVS)
Todos os métodos principais assíncronos, compatível com AsyncMock.
"""

from __future__ import annotations
import asyncio
import os
import time
import math
import json
import hashlib
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import faiss
except Exception as e:
    raise ImportError("FAISS não encontrado. Instale faiss-cpu antes de usar Hippocampus V5.") from e

from core.orchestration.control_bus import ControlBus, SystemEvents

# -------------------------
# Utils helpers
# -------------------------
def _normalize_vector(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.zeros_like(v)

def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s"))
        logger.addHandler(ch)
    logger.setLevel(logging.INFO)
    return logger

def hash_state(obj: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()

def save_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)

def load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Snapshot inválido: deve ser um dict")
    if "memory_store" not in data or "long_term_store" not in data:
        raise ValueError("Snapshot inválido: chaves obrigatórias ausentes")
    return data

logger = setup_logger("HippocampusV5")

# -------------------------
# Hippocampus
# -------------------------
class Hippocampus:
    def __init__(self, dim: int, decay_lambda: float = 1e-4, event_bus: Optional[ControlBus] = None):
        self.dim = int(dim)
        self.decay_lambda = float(decay_lambda)
        self.memory_store: Dict[str, Dict[str, Any]] = {}
        self.long_term_store: Dict[str, Dict[str, Any]] = {}
        self.telemetry: List[Dict[str, Any]] = []
        self._store_count: int = 0
        self._consolidated_count: int = 0
        self.last_snapshot_hash: Optional[str] = None

        self.faiss_index: Optional[faiss.IndexFlatL2] = None
        self.faiss_next_id: int = 0
        self._faiss_vectors_by_id: Dict[int, List[float]] = {}
        self.faiss_index_path: Optional[str] = None

        self.event_bus = event_bus

        self._init_faiss_index()

        # Subscrição de eventos segura (compatível com AsyncMock)
        if self.event_bus:
            coro_lo = self.event_bus.subscribe(SystemEvents.LO_TRIGGERED, self._handle_lo_trigger)
            coro_param = self.event_bus.subscribe(SystemEvents.PARAM_ADJUSTED, self._handle_param_adjusted)
            if asyncio.iscoroutine(coro_lo):
                asyncio.create_task(coro_lo)
            if asyncio.iscoroutine(coro_param):
                asyncio.create_task(coro_param)

    # -------------------------
    # FAISS helpers
    # -------------------------
    def _init_faiss_index(self):
        self.faiss_index = faiss.IndexFlatL2(self.dim)
        self.faiss_next_id = 0
        self._faiss_vectors_by_id = {}
        logger.info("FAISS IndexFlatL2 inicializado (dim=%d)", self.dim)

    def _faiss_add(self, vec: np.ndarray) -> int:
        v = _normalize_vector(vec).astype(np.float32)
        self.faiss_index.add(v.reshape(1, -1))
        fid = self.faiss_next_id
        self._faiss_vectors_by_id[fid] = v.tolist()
        self.faiss_next_id += 1
        return fid

    def _faiss_remove(self, fid: int) -> bool:
        if fid >= 0 and self.faiss_index and fid in self._faiss_vectors_by_id:
            try:
                self.faiss_index.remove_ids(np.array([fid], dtype=np.int64))
                del self._faiss_vectors_by_id[fid]
                return True
            except Exception:
                logger.exception(f"Falha ao remover faiss_id {fid}")
                return False
        return False

    # -------------------------
    # Eventos seguros
    # -------------------------
    async def _safe_publish(self, event_type: str, payload: Dict[str, Any], request_id: str = ""):
        if self.event_bus is None:
            return
        try:
            await self.event_bus.publish(
                event_type=event_type,
                payload=payload,
                source_module="HippocampusV5",
                request_id=request_id
            )
        except Exception as e:
            logger.error(f"Falha CRÍTICA ao publicar {event_type}: {e}")

    # -------------------------
    # Store / Upsert
    # -------------------------
    async def store(self, key: str, P0: float, payload: Dict[str, Any],
                    vec: Optional[np.ndarray] = None,
                    meta: Optional[Dict[str, Any]] = None) -> str:
        ts = time.time()
        entry: Dict[str, Any] = {
            "P0": float(P0),
            "payload": payload,
            "vec": None,
            "ts": ts,
            "faiss_id": -1,
            "consolidated": False,
            "meta": meta or {}
        }
        if vec is not None:
            try:
                fid = self._faiss_add(vec)
                entry["vec"] = vec.tolist()
                entry["faiss_id"] = fid
            except Exception:
                logger.exception("Falha ao adicionar vetor ao FAISS")

        self.memory_store[key] = entry
        self._store_count += 1
        self.telemetry.append({"event": "store", "key": key, "ts": ts, "has_vec": vec is not None})

        if self.event_bus:
            await self._safe_publish(SystemEvents.NEW_MEMORY_STORED, {"key": key, "P0": P0})
        return key

    # -------------------------
    # API de busca
    # -------------------------
    async def top_k_records(self, query: np.ndarray, k: int = 10) -> List[Tuple[Dict[str, Any], float]]:
        q = _normalize_vector(query.astype(np.float32))
        results: List[Tuple[Dict[str, Any], float]] = []

        if self.faiss_index is None or not self._faiss_vectors_by_id:
            return results

        D, I = self.faiss_index.search(q.reshape(1, -1), k)
        for fid, score in zip(I[0].tolist(), D[0].tolist()):
            if int(fid) < 0:
                continue
            key = next((k_ for k_, rec in self.memory_store.items() if rec.get("faiss_id") == int(fid)), None)
            if key:
                results.append((self.memory_store[key], float(score)))
        return results

    # -------------------------
    # Confidence decay
    # -------------------------
    async def confidence(self, key: str, now_ts: Optional[float] = None) -> float:
        rec = self.memory_store.get(key) or self.long_term_store.get(key)
        if not rec:
            return 0.0
        P0 = float(rec.get("P0", 0.0))
        ts = float(rec.get("ts", time.time()))
        now = float(now_ts) if now_ts else time.time()
        dt = max(0.0, now - ts)
        return P0 * math.exp(-self.decay_lambda * dt)

    # -------------------------
    # Consolidate
    # -------------------------
    async def consolidate(self, key: str) -> bool:
        rec = self.memory_store.pop(key, None)
        if not rec:
            return False
        fid = rec.get("faiss_id", -1)
        if fid != -1:
            self._faiss_remove(int(fid))
        rec["consolidated"] = True
        self.long_term_store[key] = rec
        self._consolidated_count += 1
        return True

    # -------------------------
    # Snapshot / Persistência
    # -------------------------
    async def snapshot_state(self) -> Dict[str, Any]:
        snap = {
            "dim": self.dim,
            "decay_lambda": self.decay_lambda,
            "memory_store": self.memory_store.copy(),
            "long_term_store": self.long_term_store.copy(),
            "faiss_vectors_by_id": self._faiss_vectors_by_id.copy(),
            "snapshot_hash": hash_state({"memory_keys": sorted(list(self.memory_store.keys())), "ts": time.time()})
        }
        self.last_snapshot_hash = snap["snapshot_hash"]
        return snap

    async def save_checkpoint(self, pcvs: Any = None, base_dir: str = "data/pcvs_snapshots") -> Dict[str, Any]:
        os.makedirs(base_dir, exist_ok=True)
        index_path = os.path.join(base_dir, f"faiss_index_{int(time.time()*1000)}.index")
        faiss.write_index(self.faiss_index, index_path)
        self.faiss_index_path = index_path

        snapshot = await self.snapshot_state()
        snapshot["index_path"] = index_path
        snapshot_fname = os.path.join(base_dir, f"hippocampus_snapshot_{int(time.time()*1000)}.json")
        save_json(snapshot_fname, snapshot)

        pcvs_hash = None
        if pcvs and hasattr(pcvs, "save"):
            try:
                pcvs_hash = pcvs.save(snapshot)
            except Exception:
                logger.exception("Falha ao salvar snapshot no PCVS")

        if self.event_bus:
            await self._safe_publish(SystemEvents.SNAPSHOT_SAVED, {"file": snapshot_fname}, request_id=pcvs_hash or str(uuid.uuid4()))
        return {"snapshot_file": snapshot_fname, "index_path": index_path, "pcvs_hash": pcvs_hash}

    async def restore_from_pcvs_snapshot(self, snapshot_path: str) -> None:
        try:
            snapshot = load_json(snapshot_path)
            await self.load_state(snapshot)
            if self.event_bus:
                await self._safe_publish("HIPPOCAMPUS_LOAD_SUCCESS", {"file": snapshot_path})
        except Exception:
            if self.event_bus:
                await self._safe_publish("HIPPOCAMPUS_LOAD_FAIL", {"file": snapshot_path})
            raise

    # -------------------------
    # Event handlers
    # -------------------------
    async def _handle_lo_trigger(self, payload: Dict[str, Any]):
        key = payload.get("id", str(uuid.uuid4()))
        P0 = float(payload.get("D_primal", 1.0))
        vec = payload.get("vec")
        meta = payload.get("meta", {})
        await self.store(key, P0, payload, vec=vec, meta=meta)

    async def _handle_param_adjusted(self, payload: Dict[str, Any]):
        logger.info(f"Parâmetro ajustado: {payload}")

    # -------------------------
    # Load State (helper)
    # -------------------------
    async def load_state(self, snapshot: Dict[str, Any]):
        self.memory_store = snapshot.get("memory_store", {})
        self.long_term_store = snapshot.get("long_term_store", {})
        self._faiss_vectors_by_id = snapshot.get("faiss_vectors_by_id", {})
        self.dim = snapshot.get("dim", self.dim)
        self.decay_lambda = snapshot.get("decay_lambda", self.decay_lambda)

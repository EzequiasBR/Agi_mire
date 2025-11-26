# core/hippocampus.py
"""
Hipocampo V4 — memória com decaimento + FAISS CPU + PCVS nativo + snapshots automáticos
Integração opcional com ControlBus para eventos LO_TRIGGERED e PARAM_ADJUSTED

Principais garantias:
- Store / Top-K (FAISS) com fallback in-memory seguro
- Decaimento da confiança: C(t) = P0 * exp(-lambda * dt)
- Persistência determinística do índice FAISS (faiss.write_index) + SHA-256 do binário
- Snapshots automáticos (default folder: pcvs/snapshots/) e integração com PCVS.save()
- Rebuild determinístico a partir de faiss_vectors_by_id
"""

from __future__ import annotations
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
    raise ImportError("FAISS não encontrado. Instale faiss-cpu antes de usar Hipocampo V4.") from e

# utils helpers
try:
    from ..services.utils import _normalize_vector, save_json, load_json, setup_logger, hash_state
except Exception:
    def _normalize_vector(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float32)
        n = np.linalg.norm(v)
        if n <= 1e-12:
            return np.zeros_like(v)
        return v / n

    def save_json(path: str, data: Dict[str, Any]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    def load_json(path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

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


logger = setup_logger("HippocampusV4")


def _sha256_of_file(path: str) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        logger.exception("Falha ao calcular sha256 do arquivo: %s", path)
        return None


class Hippocampus:
    def __init__(self, dim: int, decay_lambda: float = 1e-4, use_faiss: bool = True, event_bus: Optional[Any] = None):
        self.dim: int = int(dim)
        self.decay_lambda: float = float(decay_lambda)
        self.memory_store: Dict[str, Dict[str, Any]] = {}
        self.long_term_store: Dict[str, Dict[str, Any]] = {}
        self.telemetry: List[Dict[str, Any]] = []
        self._store_count: int = 0
        self._consolidated_count: int = 0
        self.last_snapshot_hash: Optional[str] = None

        self.faiss_index: Optional[faiss.IndexFlatIP] = None
        self.faiss_next_id: int = 0
        self._faiss_vectors_by_id: Dict[int, List[float]] = {}
        self.faiss_index_path: Optional[str] = None

        self.event_bus = event_bus

        if use_faiss:
            self._init_faiss_index()

        # subscrição de eventos
        if self.event_bus:
            try:
                self.event_bus.subscribe("LO_TRIGGERED", self._handle_lo_trigger)
                logger.info("Hippocampus: subscribed to LO_TRIGGERED")
            except Exception as e:
                logger.warning(f"Falha ao subscrever LO_TRIGGERED: {e}")
            try:
                self.event_bus.subscribe("PARAM_ADJUSTED", self._handle_param_adjusted)
                logger.info("Hippocampus: subscribed to PARAM_ADJUSTED")
            except Exception:
                pass

    # -------------------------
    # FAISS helpers
    # -------------------------
    def _init_faiss_index(self) -> None:
        self.faiss_index = faiss.IndexFlatIP(self.dim)
        self.faiss_next_id = 0
        self._faiss_vectors_by_id = {}
        logger.info("FAISS index inicializado (dim=%d)", self.dim)

    def _faiss_add(self, vec: np.ndarray) -> int:
        if self.faiss_index is None:
            self._init_faiss_index()
        v = _normalize_vector(vec).astype(np.float32)
        self.faiss_index.add(v.reshape(1, -1))
        fid = self.faiss_next_id
        self._faiss_vectors_by_id[fid] = v.tolist()
        self.faiss_next_id += 1
        return fid

    # -------------------------
    # Store / Upsert
    # -------------------------
    def store(self, key: str, P0: float, payload: Dict[str, Any], vec: Optional[np.ndarray] = None, meta: Optional[Dict[str, Any]] = None) -> str:
        ts = time.time()
        entry: Dict[str, Any] = {
            "P0": float(P0),
            "payload": payload,
            "vec": None,
            "ts": ts,
            "faiss_id": None,
            "consolidated": False,
            "meta": meta or {}
        }
        if vec is not None:
            v = _normalize_vector(np.asarray(vec, dtype=np.float32))
            entry["vec"] = v.tolist()
            try:
                fid = self._faiss_add(v)
                entry["faiss_id"] = int(fid)
            except Exception:
                logger.exception("Falha ao adicionar vetor ao FAISS")
                entry["faiss_id"] = None

        self.memory_store[key] = entry
        self._store_count += 1
        self.telemetry.append({"event": "store", "key": key, "ts": ts, "has_vec": vec is not None})
        logger.info("Evento armazenado: %s", key)
        return key

    # -------------------------
    # Top-K retrieval
    # -------------------------
    def top_k(self, query: np.ndarray, k: int = 10, include_payload: bool = False) -> List[Tuple[str, float]]:
        q = _normalize_vector(np.asarray(query, dtype=np.float32))
        results: List[Tuple[str, float]] = []

        if self.faiss_index is not None and self._faiss_vectors_by_id:
            try:
                D, I = self.faiss_index.search(q.reshape(1, -1), k)
                for fid, score in zip(I[0].tolist(), D[0].tolist()):
                    if int(fid) < 0:
                        continue
                    key = next((k_ for k_, rec in self.memory_store.items() if rec.get("faiss_id") == int(fid)), None)
                    if key is None:
                        continue
                    if include_payload:
                        results.append((key, float(score), self.memory_store[key]["payload"]))
                    else:
                        results.append((key, float(score)))
                return results
            except Exception:
                logger.exception("FAISS search falhou; fallback in-memory")

        items = [(k_, np.asarray(rec["vec"], dtype=np.float32)) for k_, rec in self.memory_store.items() if rec.get("vec") is not None]
        if not items:
            return []
        keys, vecs = zip(*items)
        M = np.vstack(vecs)
        sims = M.dot(q)
        idx_sorted = np.argsort(-sims)[:k]
        out: List[Tuple[str, float]] = []
        for i in idx_sorted:
            key = keys[int(i)]
            score = float(sims[int(i)])
            if include_payload:
                out.append((key, score, self.memory_store[key]["payload"]))
            else:
                out.append((key, score))
        return out

    # -------------------------
    # Confidence decay C(t)
    # -------------------------
    def confidence(self, key: str, now_ts: Optional[float] = None) -> float:
        rec = self.memory_store.get(key) or self.long_term_store.get(key)
        if not rec:
            return 0.0
        P0 = float(rec.get("P0", 0.0))
        ts = float(rec.get("ts", time.time()))
        now = float(now_ts) if now_ts is not None else time.time()
        dt = max(0.0, now - ts)
        val = float(P0 * math.exp(-self.decay_lambda * dt))
        logger.info("Confiança %s: %.4f", key, val)
        return val

    # -------------------------
    # Consolidate -> move to LTM
    # -------------------------
    def consolidate(self, key: str) -> bool:
        rec = self.memory_store.pop(key, None)
        if not rec:
            return False
        rec["consolidated"] = True
        self.long_term_store[key] = rec
        self._consolidated_count += 1
        self.telemetry.append({"event": "consolidate", "key": key, "ts": time.time()})
        logger.info("Consolidado para LTM: %s", key)
        return True

    # -------------------------
    # Snapshot
    # -------------------------
    def snapshot_state(self) -> Dict[str, Any]:
        snap: Dict[str, Any] = {
            "dim": int(self.dim),
            "decay_lambda": float(self.decay_lambda),
            "memory_store": {},
            "long_term_store": {},
            "telemetry": list(self.telemetry),
            "store_count": int(self._store_count),
            "consolidated_count": int(self._consolidated_count),
            "faiss_index_path": self.faiss_index_path,
            "faiss_vectors_by_id": dict(self._faiss_vectors_by_id),
            "snapshot_hash": hash_state({"memory_keys": sorted(list(self.memory_store.keys())), "ts": time.time()})
        }
        for k, rec in self.memory_store.items():
            snap["memory_store"][k] = rec.copy()
        for k, rec in self.long_term_store.items():
            snap["long_term_store"][k] = rec.copy()
        self.last_snapshot_hash = snap["snapshot_hash"]
        return snap

    def persist_index(self, base_dir: str, index_name: Optional[str] = None) -> Dict[str, Any]:
        if self.faiss_index is None:
            raise RuntimeError("Índice FAISS não inicializado")
        os.makedirs(base_dir, exist_ok=True)
        if index_name is None:
            index_name = f"faiss_index_{int(time.time()*1000)}.index"
        index_path = os.path.join(base_dir, index_name)
        faiss.write_index(self.faiss_index, index_path)
        index_hash = _sha256_of_file(index_path) or ""
        self.faiss_index_path = index_path
        meta = {"index_path": index_path, "index_hash": index_hash, "faiss_next_id": int(self.faiss_next_id), "count_vectors": len(self._faiss_vectors_by_id)}
        logger.info("FAISS index persistido em %s (hash=%s)", index_path, index_hash)
        return meta

    def save_checkpoint(self, pcvs: Any = None, base_dir: str = "pcvs/snapshots", index_name: Optional[str] = None) -> Dict[str, Any]:
        os.makedirs(base_dir, exist_ok=True)
        index_meta: Optional[Dict[str, Any]] = None
        if self.faiss_index is not None:
            index_meta = self.persist_index(base_dir, index_name=index_name)
        snapshot = self.snapshot_state()
        snapshot["index_meta"] = index_meta
        snapshot_fname = os.path.join(base_dir, f"hippocampus_snapshot_{int(time.time()*1000)}.json")
        save_json(snapshot_fname, snapshot)
        logger.info("Snapshot salvo: %s", snapshot_fname)
        pcvs_hash: Optional[str] = None
        if pcvs is not None and hasattr(pcvs, "save"):
            try:
                pcvs_hash = pcvs.save(snapshot)
                logger.info("PCVS.save hash: %s", pcvs_hash)
            except Exception:
                logger.exception("Falha ao chamar pcvs.save")
        return {"snapshot_file": snapshot_fname, "index_meta": index_meta, "pcvs_hash": pcvs_hash}

    def load_index(self, index_path: str) -> None:
        if not os.path.exists(index_path):
            raise FileNotFoundError(index_path)
        self.faiss_index = faiss.read_index(index_path)
        self.faiss_index_path = index_path
        logger.info("FAISS index carregado de %s", index_path)

    def rebuild_index(self) -> None:
        vec_map = getattr(self, "_faiss_vectors_by_id", None)
        if not vec_map:
            items = [np.asarray(rec["vec"], dtype=np.float32) for _, rec in self.memory_store.items() if rec.get("vec") is not None]
            if not items:
                self._init_faiss_index()
                return
            M = np.vstack(items)
            self._init_faiss_index()
            self.faiss_index.add(M)
            self._faiss_vectors_by_id = {i: M[i].tolist() for i in range(M.shape[0])}
            self.faiss_next_id = M.shape[0]
            return
        ids_sorted = sorted(int(i) for i in vec_map.keys())
        mats = [np.asarray(vec_map[i], dtype=np.float32) for i in ids_sorted]
        if not mats:
            self._init_faiss_index()
            return
        M = np.vstack(mats)
        self._init_faiss_index()
        self.faiss_index.add(M)
        self.faiss_next_id = max(ids_sorted) + 1

    def restore_from_pcvs_snapshot(self, snapshot: Dict[str, Any], load_index_first: bool = True) -> None:
        if not snapshot:
            raise ValueError("snapshot vazio")
        index_meta = snapshot.get("index_meta")
        if load_index_first and index_meta and index_meta.get("index_path"):
            try:
                self.load_index(index_meta["index_path"])
            except Exception:
                logger.exception("Falha ao carregar índice antes do load_state")
        self.load_state(snapshot)
        fm = snapshot.get("faiss_vectors_by_id")
        if fm:
            self._faiss_vectors_by_id = {int(k): v for k, v in fm.items()}
            if len(self._faiss_vectors_by_id) > 0:
                self.faiss_next_id = max(self._faiss_vectors_by_id.keys()) + 1

    def load_state(self, state: Dict[str, Any]) -> None:
        if not state:
            return
        self.dim = int(state.get("dim", self.dim))
        self.decay_lambda = float(state.get("decay_lambda", self.decay_lambda))
        self.memory_store = {}
        for k, rec in state.get("memory_store", {}).items():
            self.memory_store[k] = {
                "P0": float(rec.get("P0", 0.0)),
                "payload": rec.get("payload"),
                "vec": rec.get("vec"),
                "ts": float(rec.get("ts", time.time())),
                "faiss_id": rec.get("faiss_id"),
                "consolidated": bool(rec.get("consolidated", False)),
                "meta": rec.get("meta", {})
            }
        self.long_term_store = {}
        for k, rec in state.get("long_term_store", {}).items():
            self.long_term_store[k] = {
                "P0": float(rec.get("P0", 0.0)),
                "payload": rec.get("payload"),
                "vec": rec.get("vec"),
                "ts": float(rec.get("ts", time.time())),
                "faiss_id": rec.get("faiss_id"),
                "consolidated": bool(rec.get("consolidated", True)),
                "meta": rec.get("meta", {})
            }
        self.telemetry = state.get("telemetry", [])
        self._store_count = int(state.get("store_count", 0))
        self._consolidated_count = int(state.get("consolidated_count", 0))
        self.last_snapshot_hash = state.get("snapshot_hash")
        if "faiss_vectors_by_id" in state:
            self._faiss_vectors_by_id = {int(k): v for k, v in state["faiss_vectors_by_id"].items()}
            if len(self._faiss_vectors_by_id) > 0:
                self.faiss_next_id = max(self._faiss_vectors_by_id.keys()) + 1

    def serialize_state(self) -> Dict[str, Any]:
        return {
            "decay_lambda": float(self.decay_lambda),
            "store_count": int(self._store_count),
            "consolidated_count": int(self._consolidated_count),
            "last_snapshot_hash": self.last_snapshot_hash,
            "faiss_index_path": self.faiss_index_path
        }

    def stats(self) -> Dict[str, Any]:
        return {
            "working_size": len(self.memory_store),
            "long_term_size": len(self.long_term_store),
            "store_count": self._store_count,
            "consolidated_count": self._consolidated_count,
            "faiss_index_present": self.faiss_index is not None,
            "faiss_next_id": int(self.faiss_next_id)
        }

    def dump_state_json(self) -> str:
        return json.dumps(self.snapshot_state(), ensure_ascii=False, indent=2)

    # -------------------------
    # Event handlers
    # -------------------------
    def _handle_lo_trigger(self, payload: Dict[str, Any]) -> None:
        try:
            key = payload.get("id", str(uuid.uuid4()))
            P0 = float(payload.get("D_primal", 1.0))
            vec = payload.get("vec")
            meta = payload.get("meta", {})
            self.store(key, P0, payload, vec=vec, meta=meta)
            logger.info("LO_TRIGGERED armazenado: %s", key)
        except Exception as e:
            logger.exception("Falha ao processar LO_TRIGGERED: %s", e)

    def _handle_param_adjusted(self, payload: Dict[str, Any]) -> None:
        new_lambda = payload.get("decay_lambda")
        if new_lambda is not None:
            self.update_lambda(float(new_lambda))
            logger.info("decay_lambda atualizado para %.6f", self.decay_lambda)

    def update_lambda(self, new_lambda: float) -> None:
        self.decay_lambda = float(new_lambda)

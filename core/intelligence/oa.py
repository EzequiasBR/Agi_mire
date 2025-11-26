# core/oa.py
"""
Organismo Analítico (OA) — raciocínio simbólico e módulo PRM

Características principais:
- Dual Mode:
    * simbólico: tripla (S, P, O) com KG interno (triples + meta)
    * raciocínio curto (CoT): até 3 passos; estrutura, não texto longo
- PRM: escolha preferencial de relação com combinação de fatores:
    score = alpha * sem + beta * anchor_coherence + gamma * (1 - divergence) + delta * confidence
- Integração SimLog:
    * emite eventos: simlog.emit("oa.triple", ...), simlog.emit("oa.reasoning", ...), simlog.emit("oa.error", ...)
    * pode receber tripla reconstruída via reconstruct_embedding (compatibilidade)
- Exposes:
    * symbolize(embedding) -> sym_pkg (tripla, conf, cot)
    * reconstruct_embedding(sym_pkg) -> embedding (np.ndarray)
    * add_triple(s,p,o, meta)
    * query_triples(subject=None,predicate=None,object=None)
    * get_rules() -> list[dict] (compatible com RegVet)
    * serialize_state() / load_state()
    * set_external_refs(simlog=..., hippocampus=..., regvet=..., ppo=...) (optional)
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import time
import json
import hashlib
import logging

import numpy as np

# Try to reuse utils
try:
    from .utils import _normalize_vector, hash_state, setup_logger
except Exception:
    def _normalize_vector(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float32)
        n = np.linalg.norm(v)
        if n <= 1e-12:
            return np.zeros_like(v)
        return v / n

    def hash_state(obj: Dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()

    def setup_logger(name: str) -> logging.Logger:
        logger = logging.getLogger(name)
        if not logger.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s"))
            logger.addHandler(ch)
        logger.setLevel(logging.INFO)
        return logger

logger = setup_logger("OA")

DEFAULT_DIM = 768  # keep consistent with OL baseline


# -----------------------
# Helper utilities
# -----------------------
def _sha_to_seed(s: str) -> int:
    h = hashlib.sha256(str(s).encode("utf-8")).hexdigest()
    return int(h[:16], 16) % (2 ** 31 - 1)


def _deterministic_vector_from_text(text: str, dim: int = DEFAULT_DIM) -> np.ndarray:
    seed = _sha_to_seed(text)
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return _normalize_vector(v)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_n = _normalize_vector(a)
    b_n = _normalize_vector(b)
    if np.all(a_n == 0) or np.all(b_n == 0):
        return 0.0
    return float(np.dot(a_n, b_n))


# -----------------------
# Knowledge Graph storage (Option C - Hybrid)
# -----------------------
# We store triples as canonical dicts and maintain indices for fast lookup.
# Triple representation:
# {
#   "s": str,
#   "p": str,
#   "o": str,
#   "meta": { "anchor_vector": [...], "certainty": 1.0, "severity": 0.5, "active": True, ... },
#   "ts": float
# }
#
# Indices:
#  - subject_index: {subject -> [triple_ids]}
#  - object_index: {object -> [triple_ids]}
#  - triples: {id -> triple_dict}
#

class OA:
    def __init__(self, dim: int = DEFAULT_DIM):
        self.dim = int(dim)
        self.triples: Dict[str, Dict[str, Any]] = {}
        self.subject_index: Dict[str, List[str]] = {}
        self.object_index: Dict[str, List[str]] = {}
        self.next_id = 1

        # configurable PRM weights (can be tuned)
        self.prm_alpha = 0.4  # semantic relevance
        self.prm_beta = 0.25  # hippocampal anchor coherence
        self.prm_gamma = 0.2  # divergence penalty
        self.prm_delta = 0.15  # confidence weight

        # external references (optional, set by orchestrator)
        self.simlog = None
        self.hippocampus = None
        self.regvet = None
        self.ppo = None

        logger.info("OA initialized (dim=%d)", self.dim)

    # -----------------------
    # KG management
    # -----------------------
    def _new_id(self) -> str:
        nid = f"t{self.next_id}"
        self.next_id += 1
        return nid

    def add_triple(self, s: str, p: str, o: str, meta: Optional[Dict[str, Any]] = None) -> str:
        meta = meta.copy() if meta else {}
        # normalize meta defaults
        meta.setdefault("anchor_vector", None)  # optional list[float]
        meta.setdefault("certainty", float(meta.get("certainty", 1.0)))
        meta.setdefault("severity", float(meta.get("severity", 0.5)))
        meta.setdefault("active", bool(meta.get("active", True)))
        tid = self._new_id()
        triple = {
            "s": str(s),
            "p": str(p),
            "o": str(o),
            "meta": meta,
            "ts": time.time()
        }
        self.triples[tid] = triple
        # indexes
        self.subject_index.setdefault(triple["s"], []).append(tid)
        self.object_index.setdefault(triple["o"], []).append(tid)
        logger.info("OA.add_triple: %s | %s | %s (id=%s)", s, p, o, tid)
        # emit simlog event if available
        try:
            if self.simlog and hasattr(self.simlog, "emit"):
                self.simlog.emit("oa.triple", {"id": tid, "triple": triple})
        except Exception:
            logger.exception("OA.add_triple: simlog.emit failed")
        return tid

    def query_triples(self, subject: Optional[str] = None, predicate: Optional[str] = None, object: Optional[str] = None) -> List[Dict[str, Any]]:
        # Basic query engine using indices when possible
        results: List[Dict[str, Any]] = []
        if subject is not None:
            ids = self.subject_index.get(subject, [])
            for tid in ids:
                t = self.triples.get(tid)
                if t and (predicate is None or t["p"] == predicate) and (object is None or t["o"] == object):
                    results.append(t)
            return results
        if object is not None:
            ids = self.object_index.get(object, [])
            for tid in ids:
                t = self.triples.get(tid)
                if t and (predicate is None or t["p"] == predicate):
                    results.append(t)
            return results
        # full scan fallback
        for t in self.triples.values():
            if (subject is None or t["s"] == subject) and (predicate is None or t["p"] == predicate) and (object is None or t["o"] == object):
                results.append(t)
        return results

    # -----------------------
    # get_rules - for RegVet integration
    # -----------------------
    def get_rules(self) -> List[Dict[str, Any]]:
        """
        Return list of active rules in the KG with the structure expected by RegVet:
        [{ "rule_id": id, "meta": {...}, "anchor_vector": [...], "severity":..., "certainty":... }, ...]
        """
        rules = []
        for tid, t in self.triples.items():
            meta = t.get("meta", {})
            if meta.get("active") and meta.get("is_rule"):  # rules flagged by 'is_rule' in meta
                rules.append({
                    "rule_id": tid,
                    "s": t["s"],
                    "p": t["p"],
                    "o": t["o"],
                    "meta": meta,
                    "anchor_vector": meta.get("anchor_vector"),
                    "certainty": float(meta.get("certainty", 1.0)),
                    "severity": float(meta.get("severity", 0.5))
                })
        return rules

    # -----------------------
    # Symbolization (Vector -> Symbolic)
    # -----------------------
    def symbolize(self, embedding: Any) -> Dict[str, Any]:
        """
        Map embedding (vector) to a symbolic hypothesis package (sym_pkg).
        sym_pkg contains:
            - 'triples': list of candidate triples (one preferred)
            - 'conf': confidence [0,1]
            - 'cot': optional chain-of-thought (short)
            - 'anchor_scores': per-candidate anchor coherence
        """
        emb = _normalize_vector(np.asarray(embedding, dtype=np.float32))
        candidates: List[Tuple[str, Dict[str, Any], float]] = []  # (tid, triple, sem_score)

        # 1) Semantic match: compare to anchor vectors in triples (if present)
        for tid, t in self.triples.items():
            if not t.get("meta", {}).get("active", True):
                continue
            anchor = t.get("meta", {}).get("anchor_vector")
            if anchor is not None:
                try:
                    anchor_v = np.asarray(anchor, dtype=np.float32)
                    sem = _cosine_similarity(emb, anchor_v)
                except Exception:
                    sem = 0.0
            else:
                # if no anchor, produce a semantic proxy by hashing triple text deterministically
                proxy_text = f"{t['s']}|{t['p']}|{t['o']}"
                proxy_v = _deterministic_vector_from_text(proxy_text, dim=self.dim)
                sem = _cosine_similarity(emb, proxy_v)
            candidates.append((tid, t, float(sem)))

        # 2) If no candidates (empty KG), fallback: generate one synthetic triple deterministically
        if not candidates:
            subj = f"entity_{hashlib.sha256(emb.tobytes()).hexdigest()[:8]}"
            pred = "is_related_to"
            obj = "unknown"
            tid = self.add_triple(subj, pred, obj, meta={"anchor_vector": emb.tolist(), "is_rule": False})
            triple = self.triples[tid]
            sym_pkg = {"triples": [triple], "conf": 0.5, "cot": ["generated_fallback"], "anchor_scores": [1.0]}
            # emit event
            try:
                if self.simlog and hasattr(self.simlog, "emit"):
                    self.simlog.emit("oa.reasoning", {"reason": "fallback_generation", "sym_pkg": sym_pkg})
            except Exception:
                logger.exception("OA.symbolize: simlog.emit failed on fallback")
            return sym_pkg

        # 3) Score candidates semantically and with PRM
        scored: List[Tuple[str, Dict[str, Any], float]] = []
        for tid, t, sem in candidates:
            # anchor coherence (hippocampus): if hip present, compute anchor coherence by retrieving nearest memories
            anchor_vec = t.get("meta", {}).get("anchor_vector")
            anchor_coherence = 0.0
            if anchor_vec is not None and self.hippocampus is not None:
                try:
                    # use top_k with anchor vector; check top result similarity
                    top = self.hippocampus.top_k(np.asarray(anchor_vec, dtype=np.float32), k=1)
                    if top:
                        # top returns (key, score) possibly with payload; pick score
                        top_score = float(top[0][1])
                        anchor_coherence = top_score
                except Exception:
                    anchor_coherence = 0.0

            # divergence penalty: if regvet available, compute divergence between emb and anchor (approx)
            divergence = 0.0
            if anchor_vec is not None:
                divergence = 1.0 - _cosine_similarity(np.asarray(anchor_vec, dtype=np.float32), emb)

            # confidence estimate (from triple meta.certainty)
            conf_est = float(t.get("meta", {}).get("certainty", 1.0))

            # PRM combined score
            score = (self.prm_alpha * sem +
                     self.prm_beta * anchor_coherence +
                     self.prm_gamma * (1.0 - divergence) +
                     self.prm_delta * conf_est)
            scored.append((tid, t, float(score)))

        # sort by score desc, deterministic tie-break by tid
        scored.sort(key=lambda x: (x[2], x[0]), reverse=True)

        # Build CoT (short): up to 3 steps: identify entity, check anchors, select triple
        top_tid, top_triple, top_score = scored[0]
        cot_steps = [
            "identify_entities",
            "match_anchor_vectors" if top_triple.get("meta", {}).get("anchor_vector") else "use_proxy_semantics",
            "select_preferred_triple"
        ][:3]

        # Build sym_pkg
        sym_pkg = {
            "triples": [t for _, t, _ in scored],
            "preferred": top_triple,
            "preferred_id": top_tid,
            "scores": [s for _, _, s in scored],
            "conf": float(min(1.0, max(0.0, top_score))),
            "cot": cot_steps,
            "anchor_scores": [float(min(1.0, max(0.0, _))) for _, _, _ in [(x[0], x[1], x[2]) for x in scored]]
        }

        # emit simlog events
        try:
            if self.simlog and hasattr(self.simlog, "emit"):
                self.simlog.emit("oa.reasoning", {"preferred_id": top_tid, "preferred_triple": top_triple, "sym_pkg": sym_pkg})
                self.simlog.emit("oa.triple", {"preferred_id": top_tid, "preferred_triple": top_triple})
        except Exception:
            logger.exception("OA.symbolize: simlog.emit failed")

        return sym_pkg

    # -----------------------
    # Reconstruct embedding from symbolic package or triple
    # -----------------------
    def reconstruct_embedding(self, sym_pkg: Dict[str, Any]) -> np.ndarray:
        """
        Map symbolic hypothesis (preferred triple or triple list) back to embedding
        Prefer anchor_vector if present, else deterministic vector from triple text.
        """
        if not sym_pkg:
            return _deterministic_vector_from_text("empty", dim=self.dim)
        pref = sym_pkg.get("preferred") or (sym_pkg.get("triples") or [None])[0]
        if pref is None:
            return _deterministic_vector_from_text(json.dumps(sym_pkg, sort_keys=True), dim=self.dim)
        anchor = pref.get("meta", {}).get("anchor_vector")
        if anchor is not None:
            try:
                v = np.asarray(anchor, dtype=np.float32)
                return _normalize_vector(v)
            except Exception:
                logger.exception("OA.reconstruct_embedding: anchor vector invalid; fallback to deterministic")
        # deterministic fallback based on triple text
        txt = f"{pref.get('s')}|{pref.get('p')}|{pref.get('o')}"
        return _deterministic_vector_from_text(txt, dim=self.dim)

    # -----------------------
    # PRM: expose a helper to score candidate triples explicitly
    # -----------------------
    def score_candidate(self, embedding: np.ndarray, triple: Dict[str, Any]) -> float:
        emb = _normalize_vector(np.asarray(embedding, dtype=np.float32))
        anchor = triple.get("meta", {}).get("anchor_vector")
        sem = 0.0
        if anchor is not None:
            sem = _cosine_similarity(emb, np.asarray(anchor, dtype=np.float32))
        else:
            proxy = _deterministic_vector_from_text(f"{triple['s']}|{triple['p']}|{triple['o']}", dim=self.dim)
            sem = _cosine_similarity(emb, proxy)
        anchor_coherence = 0.0
        if anchor is not None and self.hippocampus is not None:
            try:
                top = self.hippocampus.top_k(np.asarray(anchor, dtype=np.float32), k=1)
                anchor_coherence = float(top[0][1]) if top else 0.0
            except Exception:
                anchor_coherence = 0.0
        divergence = 1.0 - sem
        conf_est = float(triple.get("meta", {}).get("certainty", 1.0))
        score = (self.prm_alpha * sem +
                 self.prm_beta * anchor_coherence +
                 self.prm_gamma * (1.0 - divergence) +
                 self.prm_delta * conf_est)
        return float(score)

    # -----------------------
    # External refs
    # -----------------------
    def set_external_refs(self, simlog: Any = None, hippocampus: Any = None, regvet: Any = None, ppo: Any = None) -> None:
        self.simlog = simlog
        self.hippocampus = hippocampus
        self.regvet = regvet
        self.ppo = ppo

    # -----------------------
    # Serialization / state
    # -----------------------
    def serialize_state(self) -> Dict[str, Any]:
        state = {
            "dim": int(self.dim),
            "triples": self.triples,
            "subject_index": self.subject_index,
            "object_index": self.object_index,
            "next_id": int(self.next_id),
            "prm_weights": {
                "alpha": float(self.prm_alpha),
                "beta": float(self.prm_beta),
                "gamma": float(self.prm_gamma),
                "delta": float(self.prm_delta)
            }
        }
        return state

    def load_state(self, state: Dict[str, Any]) -> None:
        if not state:
            return
        self.dim = int(state.get("dim", self.dim))
        self.triples = dict(state.get("triples", {}))
        self.subject_index = dict(state.get("subject_index", {}))
        self.object_index = dict(state.get("object_index", {}))
        self.next_id = int(state.get("next_id", self.next_id))
        w = state.get("prm_weights", {})
        self.prm_alpha = float(w.get("alpha", self.prm_alpha))
        self.prm_beta = float(w.get("beta", self.prm_beta))
        self.prm_gamma = float(w.get("gamma", self.prm_gamma))
        self.prm_delta = float(w.get("delta", self.prm_delta))

    # -----------------------
    # Utilities
    # -----------------------
    def dump_state_json(self) -> str:
        return json.dumps(self.serialize_state(), ensure_ascii=False, indent=2, default=str)

    def stats(self) -> Dict[str, Any]:
        return {
            "triples": len(self.triples),
            "subjects": len(self.subject_index),
            "objects": len(self.object_index),
            "prm_weights": {"alpha": self.prm_alpha, "beta": self.prm_beta, "gamma": self.prm_gamma, "delta": self.prm_delta}
        }


# -----------------------
# quick self-test when run directly
# -----------------------
if __name__ == "__main__":
    oa = OA(dim=768)
    # add some triples with anchor vectors
    v1 = _deterministic_vector_from_text("dog|is|animal", dim=768).tolist()
    t1 = oa.add_triple("dog", "is", "animal", meta={"anchor_vector": v1, "certainty": 0.95, "is_rule": True})
    t2 = oa.add_triple("bark", "related_to", "dog", meta={"anchor_vector": _deterministic_vector_from_text("bark|dog", dim=768).tolist(), "certainty": 0.8})
    # symbolize a vector similar to dog
    query_vec = _deterministic_vector_from_text("dog", dim=768)
    sym = oa.symbolize(query_vec)
    print("Preferred triple:", sym.get("preferred"))
    recon = oa.reconstruct_embedding(sym)
    print("Reconstructed norm:", float(np.linalg.norm(recon)))

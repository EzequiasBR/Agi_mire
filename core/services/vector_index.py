# vector_index.py (FAISS-CPU)
import faiss
import numpy as np
from typing import Dict, List, Tuple, Any


class VectorIndex:
    """
    FAISS Index Wrapper for scalable vector similarity search.
    - Uses L2-normalized vectors + IndexFlatIP (cosine similarity)
    - Stores Python-side metadata for each id
    """

    def __init__(self, dim: int):
        self.dim = dim

        # FAISS index using inner product (works as cosine if normalized)
        self.index = faiss.IndexFlatIP(dim)

        # Python-side map: id -> metadata dict
        self.metadata: Dict[int, Dict[str, Any]] = {}

        # Track next ID
        self.next_id = 0

        # Cache of vectors to allow full snapshot reconstruction
        self._vectors: Dict[int, np.ndarray] = {}

    # -----------------------------------------------------------
    # Normalization Helper
    # -----------------------------------------------------------
    def _normalize(self, v: np.ndarray) -> np.ndarray:
        return v / (np.linalg.norm(v) + 1e-12)

    # -----------------------------------------------------------
    # Add Vector
    # -----------------------------------------------------------
    def add(self, vector: np.ndarray, meta: Dict[str, Any]) -> int:
        vector = np.asarray(vector, dtype=np.float32)
        vector = self._normalize(vector)

        vec_id = self.next_id
        self.next_id += 1

        # Save raw vector
        self._vectors[vec_id] = vector

        # Save metadata
        self.metadata[vec_id] = meta

        # Add to FAISS index
        self.index.add(vector.reshape(1, -1))

        return vec_id

    # -----------------------------------------------------------
    # Search Top-K
    # -----------------------------------------------------------
    def top_k(self, query: np.ndarray, k: int = 5) -> List[Tuple[int, float, Dict]]:
        if len(self.metadata) == 0:
            return []

        query = np.asarray(query, dtype=np.float32)
        query = self._normalize(query)

        q = query.reshape(1, -1)
        scores, ids = self.index.search(q, k)

        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue

            meta = self.metadata.get(idx, {})
            results.append((idx, float(score), meta))

        return results

    # -----------------------------------------------------------
    # Snapshot full state  (PCVS full-dump)
    # -----------------------------------------------------------
    def snapshot_state(self) -> Dict[str, Any]:
        return {
            "dim": self.dim,
            "next_id": self.next_id,
            "vectors": {k: v.tolist() for k, v in self._vectors.items()},
            "metadata": self.metadata.copy(),
        }

    # -----------------------------------------------------------
    # Light serialization (only parameters, no vectors)
    # Useful for fast PCVS cycle snapshots
    # -----------------------------------------------------------
    def serialize_state(self) -> Dict[str, Any]:
        return {
            "dim": self.dim,
            "next_id": self.next_id,
            "count": len(self.metadata),
        }

    # -----------------------------------------------------------
    # Load full state (PCVS rollback)
    # -----------------------------------------------------------
    def load_state(self, state: Dict[str, Any]):
        self.dim = state["dim"]
        self.next_id = state["next_id"]

        # Restore metadata
        self.metadata = {int(k): v for k, v in state["metadata"].items()}

        # Restore vectors
        self._vectors = {int(k): np.array(v, dtype=np.float32) for k, v in state["vectors"].items()}

        # Rebuild FAISS index
        self.index = faiss.IndexFlatIP(self.dim)

        if len(self._vectors) > 0:
            mat = np.stack([self._vectors[i] for i in sorted(self._vectors.keys())], axis=0)
            self.index.add(mat)

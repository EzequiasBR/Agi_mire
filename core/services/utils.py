# core/services/utils.py
import numpy as np
import logging
import hashlib
import time
import json
from typing import Union, Any, List

DEFAULT_DIM = 768

# -------------------------
# Vetores e Embeddings
# -------------------------
def _normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normaliza um vetor para a norma L2 (tamanho unitário)."""
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    if n <= 1e-12:
        return np.zeros_like(v)
    return v / n

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calcula a similaridade de cosseno entre dois vetores."""
    a_n = _normalize_vector(a)
    b_n = _normalize_vector(b)
    if np.all(a_n == 0) or np.all(b_n == 0):
        return 0.0
    return float(np.dot(a_n, b_n))

def _sha_to_seed(s: str) -> int:
    """Converte string em seed determinística para geração de vetores."""
    h = hashlib.sha256(str(s).encode("utf-8")).hexdigest()
    return int(h[:16], 16) % (2 ** 31 - 1)

def deterministic_vector_from_text(text: str, dim: int = DEFAULT_DIM) -> np.ndarray:
    """Gera vetor normalizado e determinístico a partir de texto."""
    seed = _sha_to_seed(text)
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return _normalize_vector(v)

# -------------------------
# Hash / Auditoria (PRAG)
# -------------------------
def hash_state(data: Any) -> str:
    """
    Gera SHA-256 para auditoria de estado.
    Suporta dicts, listas, bytes, strings, números.
    """
    if isinstance(data, (dict, list)):
        data_bytes = json.dumps(data, sort_keys=True).encode("utf-8")
    elif isinstance(data, str):
        data_bytes = data.encode("utf-8")
    elif isinstance(data, bytes):
        data_bytes = data
    else:
        data_bytes = str(data).encode("utf-8")
    return hashlib.sha256(data_bytes).hexdigest()

# -------------------------
# Identificadores / Timestamp
# -------------------------
def timestamp_id(prefix: str) -> str:
    """Gera ID único baseado em timestamp de alta precisão."""
    return f"{prefix}_{time.time():.5f}".replace('.', '')

def deterministic_id_from_text(text: str, prefix: str = "ID") -> str:
    """Gera ID determinístico baseado em string."""
    seed = _sha_to_seed(text)
    return f"{prefix}_{seed}"

# -------------------------
# Logging
# -------------------------
def setup_logger(name: str) -> logging.Logger:
    """Configura logger padronizado para módulos principais."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s"))
        logger.addHandler(ch)
    logger.setLevel(logging.INFO)
    return logger

# -------------------------
# JSON Utils / Conversões
# -------------------------
def ensure_json_serializable(obj: Any) -> Any:
    """
    Converte tipos complexos (np.ndarray, np.float32) para JSON-serializable.
    Útil para descriptors do OVI.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: ensure_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [ensure_json_serializable(x) for x in obj]
    else:
        return obj

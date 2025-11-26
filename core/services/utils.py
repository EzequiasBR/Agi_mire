# core/utils.py
"""
Módulo de utilidades transversais para MIHE/AGI
Inclui: normalização, similaridade, hashing, snapshots, ruído, decaimento e logging.
"""

import numpy as np
import hashlib, json, time, random, logging
from typing import Optional

# ---------------------------
# Vetores e métricas
# ---------------------------
def _normalize_vector(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n <= 1e-12:
        return np.zeros_like(v, dtype=float)
    return v / n

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(_normalize_vector(a), _normalize_vector(b)))

def divergence_from_cosine(cos_sim: float) -> float:
    """
    Mapeia [-1,1] -> [0,1] para divergência
    """
    return max(0.0, min(1.0, (1.0 - cos_sim) / 2.0))

# ---------------------------
# Hashing e auditoria
# ---------------------------
def hash_state(state: dict) -> str:
    payload = json.dumps(state, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()

def timestamp_id(prefix: str = "evt") -> str:
    return f"{prefix}:{int(time.time()*1000)}:{random.randint(0,999999):06d}"

# ---------------------------
# Persistência leve
# ---------------------------
def save_json(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------------------------
# Ruído / perturbação
# ---------------------------
def inject_noise(vec: np.ndarray, intensity: float = 0.01, seed: Optional[int] = None) -> np.ndarray:
    rng = np.random.RandomState(seed if seed is not None else int(time.time()*1000))
    noise = rng.randn(*vec.shape)
    vec_norm = np.linalg.norm(vec) + 1e-12
    perturb = noise / (np.linalg.norm(noise) + 1e-12) * (intensity * vec_norm)
    return _normalize_vector(vec + perturb)

# ---------------------------
# Tempo e decaimento
# ---------------------------
def exp_decay(p0: float, delta_t: float, lambda_: float) -> float:
    return p0 * np.exp(-lambda_ * delta_t)

# ---------------------------
# Logging utilitário
# ---------------------------
def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s"))
        logger.addHandler(ch)
    logger.setLevel(logging.INFO)
    return logger

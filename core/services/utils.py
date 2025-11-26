# core/utils_ultra_safe.py
"""
Módulo de utilidades ULTRA seguras para MIHE/AGI
Inclui:
- Normalização e similaridade com validação completa de vetores
- Hashing seguro com captura de exceções e checagem de tipo
- Persistência com proteção contra falhas de I/O e verificação de dicionários
- Ruído com validação de dimensões e tipos
- Decaimento exponencial seguro
- Logging reforçado
- Restauro de snapshots com validação de integridade e tipos
"""

import numpy as np
import hashlib
import json
import time
import random
import logging
from typing import Optional, Dict, Any, Union

# ---------------------------
# Logging reforçado
# ---------------------------
def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s"))
        logger.addHandler(ch)
    logger.setLevel(logging.INFO)
    return logger

logger = setup_logger("UtilsUltraSafe")

# ---------------------------
# Funções utilitárias de verificação
# ---------------------------
def assert_vector(vec: np.ndarray):
    if not isinstance(vec, np.ndarray):
        raise TypeError(f"Entrada não é np.ndarray: {type(vec)}")
    if vec.ndim != 1:
        raise ValueError(f"Vetor deve ser 1D, encontrado {vec.ndim}D")

def assert_dict(obj: Any):
    if not isinstance(obj, dict):
        raise TypeError(f"Entrada não é dict: {type(obj)}")

# ---------------------------
# Vetores e métricas ultra seguras
# ---------------------------
def normalize_vector(v: Union[np.ndarray, list, tuple]) -> np.ndarray:
    try:
        v = np.asarray(v, dtype=float)
        assert_vector(v)
        n = np.linalg.norm(v)
        if n <= 1e-12:
            return np.zeros_like(v, dtype=float)
        return v / n
    except Exception as e:
        logger.error(f"normalize_vector falhou: {e}")
        return np.zeros((len(v) if hasattr(v, '__len__') else 1,), dtype=float)


def cosine_similarity(a: Union[np.ndarray, list, tuple], b: Union[np.ndarray, list, tuple]) -> float:
    try:
        a_n = normalize_vector(a)
        b_n = normalize_vector(b)
        return float(np.dot(a_n, b_n))
    except Exception as e:
        logger.error(f"cosine_similarity falhou: {e}")
        return 0.0


def divergence_from_cosine(cos_sim: float) -> float:
    try:
        cos_sim = float(cos_sim)
        if cos_sim < -1.0 or cos_sim > 1.0:
            raise ValueError(f"cos_sim fora do intervalo [-1,1]: {cos_sim}")
        return (1.0 - cos_sim) / 2.0
    except Exception as e:
        logger.error(f"divergence_from_cosine falhou: {e}")
        return 1.0

# ---------------------------
# Hashing ultra seguro
# ---------------------------
def hash_state(state: Dict[str, Any]) -> str:
    try:
        assert_dict(state)
        payload = json.dumps(state, sort_keys=True, default=str).encode()
        return hashlib.sha256(payload).hexdigest()
    except Exception as e:
        logger.error(f"hash_state falhou: {e}")
        return ""

def timestamp_id(prefix: str = "evt") -> str:
    try:
        return f"{prefix}:{int(time.time() * 1000)}:{random.randint(0, 999999):06d}"
    except Exception as e:
        logger.error(f"timestamp_id falhou: {e}")
        return f"{prefix}:0:000000"

# ---------------------------
# Persistência ultra segura
# ---------------------------
def save_json(path: str, data: Dict[str, Any]) -> bool:
    try:
        assert_dict(data)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"save_json falhou ({path}): {e}")
        return False

def load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        data = json.load(open(path, "r", encoding="utf-8"))
        assert_dict(data)
        return data
    except Exception as e:
        logger.error(f"load_json falhou ({path}): {e}")
        return None

# ---------------------------
# Ruído ultra seguro
# ---------------------------
def inject_noise(vec: Union[np.ndarray, list, tuple], intensity: float = 0.01, seed: Optional[int] = None) -> np.ndarray:
    try:
        vec = np.asarray(vec, dtype=float)
        assert_vector(vec)
        rng = np.random.RandomState(seed if seed is not None else int(time.time() * 1000))
        noise = rng.randn(*vec.shape)
        vec_norm = np.linalg.norm(vec) + 1e-12
        perturb = noise / (np.linalg.norm(noise) + 1e-12) * (intensity * vec_norm)
        return normalize_vector(vec + perturb)
    except Exception as e:
        logger.error(f"inject_noise falhou: {e}")
        return np.zeros_like(vec)

# ---------------------------
# Decaimento ultra seguro
# ---------------------------
def exp_decay(p0: float, delta_t: float, lambda_: float) -> float:
    try:
        p0 = float(p0)
        delta_t = float(delta_t)
        lambda_ = float(lambda_)
        return p0 * np.exp(-lambda_ * delta_t)
    except Exception as e:
        logger.error(f"exp_decay falhou: {e}")
        return p0

# ---------------------------
# Restauro ultra seguro de snapshots
# ---------------------------
def load_pcvs_snapshot(self, sha: Optional[str] = None) -> Optional[Dict[str, Any]]:
    try:
        target_sha = sha or getattr(self, "last_state_hash", None)
        if not target_sha:
            logger.warning("Nenhum hash alvo fornecido ou last_state_hash vazio.")
            return None

        if not hasattr(self, "pcvs"):
            logger.error("PCVS não encontrado no objeto.")
            return None

        snapshot = self.pcvs.load_snapshot(target_sha)
        if snapshot:
            loaded_hash = hash_state(snapshot)
            if loaded_hash != target_sha:
                logger.critical(
                    f"❌ FALHA DE INTEGRIDADE PCVS! Esperado {target_sha[:10]}, encontrado {loaded_hash[:10]}"
                )
                if hasattr(self, "monitor"):
                    self.monitor.register_event("PCVS_CORRUPTION_FAIL",
                                                {"expected_hash": target_sha, "loaded_hash": loaded_hash})
                return None

            logger.warning(f"✅ Snapshot restaurado com sucesso: {target_sha[:10]}")
            # Validar campos críticos
            self.cycle_count = int(snapshot.get('cycle_count', 0))
            mch_metrics = snapshot.get('mch_metrics', {})
            self.H_sist = float(mch_metrics.get('H', 0))
            self.V_sist = float(mch_metrics.get('V', 0))
            self.E_sist = float(mch_metrics.get('E', 0))

            if hasattr(self, "hippocampus") and "hippocampus_state" in snapshot:
                self.hippocampus.restore_state(snapshot['hippocampus_state'])

            return snapshot

        logger.warning(f"Snapshot não encontrado para hash: {target_sha[:10]}")
        return None
    except Exception as e:
        logger.error(f"load_pcvs_snapshot falhou: {e}")
        return None

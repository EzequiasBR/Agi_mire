# core/pcvs.py
"""
PCVS — Ponto de Controle de Validação de Segurança

Funcionalidades:
- Salva snapshots de memória/estado com hash para auditoria.
- Permite rollback determinístico a qualquer ponto salvo.
- Compatível com Hippocampus V4.
- Snapshots armazenados em `snapshots/` por padrão.
"""

from __future__ import annotations
import os
import time
import json
import hashlib
import logging
from typing import Any, Dict, Optional

from .utils import save_json, load_json, setup_logger

logger = setup_logger("PCVS")

class PCVS:
    def __init__(self, base_dir: str = "snapshots"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        # Map: hash -> snapshot_path
        self._snapshots: Dict[str, str] = {}

    # -------------------------
    # Save snapshot
    # -------------------------
    def save(self, snapshot: Dict[str, Any]) -> str:
        """
        Salva um snapshot e retorna o hash SHA-256 como referência.
        """
        ts = int(time.time() * 1000)
        snapshot["pcvs_ts"] = ts
        snapshot_json = json.dumps(snapshot, ensure_ascii=False, sort_keys=True)
        sha256 = hashlib.sha256(snapshot_json.encode("utf-8")).hexdigest()

        fname = os.path.join(self.base_dir, f"pcvs_{sha256}_{ts}.json")
        save_json(fname, snapshot)
        self._snapshots[sha256] = fname
        logger.info("Snapshot PCVS salvo: %s", fname)
        return sha256

    # -------------------------
    # Load snapshot by hash
    # -------------------------
    def load(self, sha256: str) -> Dict[str, Any]:
        """
        Recupera snapshot a partir do hash SHA-256.
        """
        fname = self._snapshots.get(sha256)
        if not fname or not os.path.exists(fname):
            raise FileNotFoundError(f"Snapshot PCVS não encontrado para hash {sha256}")
        snap = load_json(fname)
        logger.info("Snapshot PCVS carregado: %s", fname)
        return snap

    # -------------------------
    # List snapshots
    # -------------------------
    def list_snapshots(self) -> Dict[str, str]:
        """
        Retorna todos os snapshots salvos: {hash: arquivo}
        """
        return dict(self._snapshots)

    # -------------------------
    # Rollback helper
    # -------------------------
    def rollback(self, hippocampus: Any, sha256: str):
        """
        Restaura estado do Hippocampus V4 a partir do snapshot.
        """
        snap = self.load(sha256)
        hippocampus.restore_from_pcvs_snapshot(snap)
        logger.info("Hippocampus restaurado do snapshot PCVS %s", sha256)

# -------------------------
# Self-test
# -------------------------
if __name__ == "__main__":
    from memory.hippocampus import Hippocampus
    import numpy as np

    pcvs = PCVS(base_dir="/tmp/pcvs_test")
    hip = Hippocampus(dim=64)

    # Store eventos
    v = np.random.randn(64).astype(np.float32)
    hip.store("evt1", 0.9, {"note": "evento 1"}, vec=v)
    hip.store("evt2", 0.7, {"note": "evento 2"}, vec=v)

    # Salvar snapshot
    snap_hash = pcvs.save(hip.snapshot_state())
    print("Snapshot hash:", snap_hash)

    # Modificar estado
    hip.store("evt3", 0.8, {"note": "evento 3"}, vec=v)
    print("Antes rollback:", list(hip.memory_store.keys()))

    # Rollback
    pcvs.rollback(hip, snap_hash)
    print("Após rollback:", list(hip.memory_store.keys()))

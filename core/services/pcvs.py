import os
import logging
import time
import asyncio
from typing import Dict, Any, Optional, List

from core.memory.hippocampus import load_json, save_json

from ..services.utils import setup_logger, hash_state
from core.orchestration.control_bus import ControlBus, SystemEvents

logger = setup_logger("PCVS")


class PCVS:
    """
    PCVS — Persistent Control & Verification System
    Responsável por:
      - Captura e persistência de snapshots 
      - Verificação de integridade (hash)
      - Gestão do histórico de snapshots
      - Sinalização de rollback ao ControlBus
    """

    def __init__(self, control_bus: ControlBus, base_dir: str = "pcvs_data"):
        self.control_bus = control_bus
        self.base_dir = base_dir

        self.snapshot_history: List[Dict[str, Any]] = []
        self.last_snapshot_hash: Optional[str] = None

        os.makedirs(self.base_dir, exist_ok=True)
        self._load_history()

        logger.info(f"PCVS inicializado. Diretório base: {self.base_dir}")

    # ============================================================== #
    # AUXILIARES DE ARQUIVO E HISTÓRICO
    # ============================================================== #

    def _get_file_path(self, sha256: str) -> str:
        return os.path.join(self.base_dir, f"{sha256}.json")

    async def _save_history(self) -> None:
        """
        Persistência do histórico.
        Usa asyncio.to_thread para evitar travamento do event loop.
        """
        history_path = os.path.join(self.base_dir, "history.json")
        payload = {"history": self.snapshot_history}

        try:
            await asyncio.to_thread(save_json, history_path, payload)
        except Exception as e:
            logger.error(f"Falha ao persistir o histórico do PCVS: {e}")

    def _load_history(self) -> None:
        """
        Carrega histórico salvo.
        Garante estrutura consistente com {"history": [...]}.
        """
        history_path = os.path.join(self.base_dir, "history.json")

        if not os.path.exists(history_path):
            return

        try:
            data = load_json(history_path)
            self.snapshot_history = data.get("history", [])
            if self.snapshot_history:
                self.last_snapshot_hash = self.snapshot_history[-1]["hash"]
                logger.info(f"Histórico carregado. Último hash: {self.last_snapshot_hash[:10]}")
        except Exception as e:
            logger.error(f"Falha ao carregar histórico do PCVS: {e}")

    # ============================================================== #
    # SNAPSHOTS (SAVE / LOAD)
    # ============================================================== #

    async def persist_snapshot(self, snapshot_data: Dict[str, Any], reason: str) -> Optional[str]:
        """
        Persiste snapshot com metadados + SHA256 de integridade.
        Retorna o hash persistido.
        """

        # 1) Adiciona metadados
        snapshot_to_save = {
            "metadata": {
                "timestamp": time.time(),
                "reason": reason,
            },
            "system_state": snapshot_data,
        }

        # 2) Hash consistente
        sha256 = hash_state(snapshot_to_save)
        file_path = self._get_file_path(sha256)

        try:
            # 3) I/O assíncrono real
            await asyncio.to_thread(save_json, file_path, snapshot_to_save)

            # 4) Atualiza histórico completo
            self.last_snapshot_hash = sha256
            self.snapshot_history.append({
                "hash": sha256,
                "timestamp": snapshot_to_save["metadata"]["timestamp"],
                "reason": reason,
            })

            # Persiste histórico auxiliar
            await self._save_history()

            # 5) Publica evento
            await self.control_bus.publish(
                SystemEvents.SNAPSHOT_SAVED,
                {"hash": sha256, "reason": reason}
            )

            logger.info(f"Snapshot '{reason}' salvo. Hash: {sha256[:10]}")
            return sha256

        except Exception as e:
            logger.error(f"Erro ao salvar snapshot no PCVS: {e}")
            return None

    async def load_snapshot(self, sha256: str) -> Optional[Dict[str, Any]]:
        """
        Carrega e verifica integridade do snapshot.
        Retorna o conteúdo ou None se houver corrupção/ausência.
        """

        file_path = self._get_file_path(sha256)

        if not os.path.exists(file_path):
            logger.warning(f"Snapshot não encontrado: {sha256[:10]}")
            return None

        try:
            # I/O assíncrono real
            loaded_snapshot = await asyncio.to_thread(load_json, file_path)

            # Recalcula hash para integridade (confiança zero)
            rehashed = hash_state(loaded_snapshot)

            if rehashed != sha256:
                logger.critical(
                    f"Falha de integridade: arquivo={rehashed[:10]} != esperado={sha256[:10]}"
                )

                await self.control_bus.publish(
                    SystemEvents.INTEGRITY_VIOLATION,
                    {"file_hash": sha256, "reason": "Snapshot hash mismatch"}
                )
                return None

            logger.debug(f"Snapshot {sha256[:10]} carregado e verificado.")
            return loaded_snapshot

        except Exception as e:
            logger.error(f"Erro ao carregar snapshot {sha256[:10]}: {e}")
            return None

    # ============================================================== #
    # CONSULTA
    # ============================================================== #

    def get_last_snapshot_hash(self) -> Optional[str]:
        return self.last_snapshot_hash

    # ============================================================== #
    # ROLLBACK (SINALIZAÇÃO)
    # ============================================================== #

    async def rollback(self, target_hash: str) -> Dict[str, Any]:
        """
        Sinaliza rollback ao ControlBus.  
        Retorna dict com rastreabilidade.
        """
        logger.warning(f"Sinalizando rollback para hash: {target_hash[:10]}")

        await self.control_bus.publish(
            SystemEvents.ROLLBACK_REQUESTED,
            {"target_hash": target_hash, "source": "PCVS_SIGNAL"}
        )

        return {
            "status": "signaled",
            "target_hash": target_hash,
        }

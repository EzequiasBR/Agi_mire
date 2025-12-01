import time
import logging
import asyncio
import json
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import uuid

from core.orchestration.control_bus import SystemEvents
from core.services.utils import hash_state, timestamp_id

try:
    import trimesh
except ImportError:
    trimesh = None

# -------------------------
# Logger estruturado
# -------------------------
logger = logging.getLogger("OVI_Core")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s OVI_Core: %(levelname)s: %(message)s"))
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)
logger.setLevel(logging.INFO)


class OVICore:
    """
    Núcleo do OVI para produção real:
    - Recebe estado do Hippocampus e gera scene descriptors JSON.
    - Publica eventos async para OVIRenderer via ControlBus.
    """

    def __init__(self, hippocampus: Any, control_bus: Any, config: Optional[Dict[str, Any]] = None):
        self.hippocampus = hippocampus
        self.control_bus = control_bus
        self.config = config or {}

        self.semantic_to_mesh = self.config.get("SEMANTIC_MAP", {"unknown": "builtin_cube"})
        self.default_camera = self.config.get("DEFAULT_CAMERA", {"position": [0, 5, 10], "target": [0, 0, 0]})

        # Subscrever evento async do ControlBus
        if self.control_bus:
            asyncio.create_task(
                self.control_bus.subscribe(SystemEvents.VISUALIZATION_REQUESTED, self.handle_ovi_request)
            )
            logger.info("OVICore pronto para escutar VISUALIZATION_REQUESTED")

    # -------------------------
    # Scene Descriptor
    # -------------------------
    def generate_scene_descriptor(self, query_vector: np.ndarray, fidelity: str = "placeholder", top_k: int = 50) -> Dict[str, Any]:
        """
        Gera descriptor de cena a partir dos top_k objetos do Hippocampus.
        """

        # API pública necessária
        if not hasattr(self.hippocampus, "top_k_records"):
            raise RuntimeError("Hippocampus API 'top_k_records' não encontrada.")

        # Obter registros
        memory_records = self._get_scene_objects(query_vector=query_vector, top_k=top_k)

        scene_objects = []
        for record, score in memory_records:
            meta = record.get("meta", {})
            semantic_label = meta.get("semantic_label", "unknown")

            obj = {
                "id": record.get("key", str(uuid.uuid4())),
                "semantic_label": semantic_label,
                "position": meta.get("position", [0, 0, 0]),
                "rotation": meta.get("rotation", [0, 0, 0]),
                "scale": meta.get("scale", [1, 1, 1]),
                "mesh": self.semantic_to_mesh.get(semantic_label, "builtin_cube"),
                "relevance_score": float(score)
            }
            scene_objects.append(obj)

        descriptor = {
            "scene_id": timestamp_id("scene"),
            "timestamp": time.time(),
            "camera": self.default_camera,
            "objects": scene_objects,
            "metadata": {
                "fidelity": fidelity
            }
        }

        # Hash determinístico para auditoria
        descriptor["metadata"]["scene_hash"] = hash_state(
            json.dumps(descriptor, sort_keys=True)
        )

        logger.info(json.dumps({
            "event": "scene_descriptor_generated",
            "scene_id": descriptor["scene_id"],
            "num_objects": len(scene_objects),
            "hash": descriptor["metadata"]["scene_hash"]
        }))
        return descriptor

    # -------------------------
    # Integração com Hippocampus
    # -------------------------
    def _get_scene_objects(self, query_vector: np.ndarray, top_k: int = 50) -> List[Tuple[Dict[str, Any], float]]:
        """
        Consulta o Hippocampus pela API pública top_k_records.
        """
        try:
            return self.hippocampus.top_k_records(query=query_vector, k=top_k)
        except Exception as e:
            logger.error(json.dumps({
                "event": "hippocampus_topk_error",
                "error": str(e),
                "top_k": top_k
            }))
            # Propaga a falha para que handle_ovi_request capture
            raise

    # -------------------------
    # Event handler async
    # -------------------------
    async def handle_ovi_request(self, event_payload: Dict[str, Any]):
        """
        Evento VISUALIZATION_REQUESTED -> gera descriptor -> publica para OVIRenderer.
        Garante publicação de SERVICE_FAILURE em caso de erro.
        """
        request_id = event_payload.get("request_id", str(uuid.uuid4()))
        request_data = event_payload.get("data", {})

        fidelity = request_data.get("fidelity", "placeholder")
        top_k = request_data.get("top_k", 50)
        query_vec = request_data.get("query_vector")

        # Garante um vetor dummy se não fornecido
        if query_vec is None:
            query_vec = np.random.rand(128).astype(np.float32)
            logger.debug(f"Usando query_vector dummy para {request_id}.")

        start_time = time.time()

        try:
            descriptor = self.generate_scene_descriptor(query_vector=query_vec, fidelity=fidelity, top_k=top_k)

            if self.control_bus:
                await self.control_bus.publish(
                    event_type=SystemEvents.VISUALIZATION_READY,
                    payload={
                        "scene_descriptor": descriptor,
                        "scene_creation_latency_ms": int((time.time() - start_time) * 1000)
                    },
                    source_module="OVICore",
                    request_id=request_id
                )

        except Exception as e:
            # Log completo
            logger.exception(f"Falha ao gerar descriptor no OVICore para {request_id}: {e}")
            # Publica SERVICE_FAILURE
            await self._publish_failure(request_id, str(e), "OVICORE_ERROR")

    # -------------------------
    # Publicação de falha padronizada
    # -------------------------
    async def _publish_failure(self, request_id: str, error_message: str, code: str):
        """Publica falha no ControlBus com evento SERVICE_FAILURE."""
        if self.control_bus:
            await self.control_bus.publish(
                event_type=SystemEvents.SERVICE_FAILURE,
                payload={
                    "request_id_ref": request_id,
                    "error_message": error_message,
                    "error_code": code,
                },
                source_module="OVICore",
                request_id=request_id
            )
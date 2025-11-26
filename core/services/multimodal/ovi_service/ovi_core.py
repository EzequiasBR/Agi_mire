# core/services/multimodal/ovi_core.py

import time
import logging
from typing import Dict, Any, List, Optional

# Dependência opcional de geometria
try:
    import trimesh
except ImportError:
    trimesh = None  # Pode usar placeholders

# Utils internas
from ...utils import hash_state, timestamp_id

logger = logging.getLogger("OVI_Core")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s OVI_Core: %(levelname)s: %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)


class OVICore:
    """
    Núcleo do OVI: recebe estado do Hippocampus e gera scene descriptors JSON.
    """

    def __init__(self, hippocampus: Any, control_bus: Any, config: Optional[Dict[str, Any]] = None):
        self.hippocampus = hippocampus
        self.control_bus = control_bus
        self.config = config or {}
        self.semantic_to_mesh = self.config.get(
            "SEMANTIC_MAP", {"unknown": "builtin_cube"}
        )  # Ex: {"car": "lowpoly_car", ...}
        self.default_camera = self.config.get(
            "DEFAULT_CAMERA", {"position": [0, 5, 10], "target": [0, 0, 0]}
        )

        # Subscrever evento do ControlBus para gerar cenas
        if self.control_bus:
            self.control_bus.subscribe("OVI_REQUESTED", self.handle_ovi_request)
            logger.info("OVICore pronto para escutar OVI_REQUESTED")

    # -------------------------
    # Scene Descriptor
    # -------------------------
    def generate_scene_descriptor(self, fidelity: str = "placeholder", top_k: int = 50) -> Dict[str, Any]:
        """
        Gera scene descriptor a partir dos top_k objetos do Hippocampus.
        """
        if not self.hippocampus:
            raise RuntimeError("Hippocampus não configurado")

        scene_objects = self._get_scene_objects(top_k=top_k)

        # Converte semantic_label em mesh type
        for obj in scene_objects:
            obj["mesh"] = self.semantic_to_mesh.get(obj.get("semantic_label", ""), "builtin_cube")

        descriptor = {
            "scene_id": timestamp_id("scene"),
            "timestamp": time.time(),
            "camera": self.default_camera,
            "objects": scene_objects,
            "metadata": {
                "fidelity": fidelity
            }
        }

        # Gera hash para auditoria
        descriptor["metadata"]["scene_hash"] = hash_state(descriptor)
        return descriptor

    # -------------------------
    # Integração com Hippocampus
    # -------------------------
    def _get_scene_objects(self, top_k: int = 50) -> List[Dict[str, Any]]:
        """
        Retorna objetos do Hippocampus prontos para renderização.
        """
        if not hasattr(self.hippocampus, "memory_store"):
            logger.warning("Hippocampus não tem memory_store")
            return []

        objects = []
        # Seleciona top_k objetos
        for idx, (key, rec) in enumerate(list(self.hippocampus.memory_store.items())[:top_k]):
            meta = rec.get("meta", {})
            obj = {
                "id": key,
                "semantic_label": meta.get("semantic_label", "unknown"),
                "position": meta.get("position", [0, 0, 0]),
                "rotation": meta.get("rotation", [0, 0, 0]),
                "scale": meta.get("scale", [1, 1, 1])
            }
            objects.append(obj)
        return objects

    # -------------------------
    # Event handler
    # -------------------------
    def handle_ovi_request(self, payload: Dict[str, Any]):
        """
        Recebe evento OVI_REQUESTED, gera descriptor e publica VISUALIZATION_REQUESTED.
        """
        fidelity = payload.get("fidelity", "placeholder")
        top_k = payload.get("top_k", 50)

        descriptor = self.generate_scene_descriptor(fidelity=fidelity, top_k=top_k)
        logger.info("Scene descriptor gerado: %s objetos", len(descriptor["objects"]))

        # Publica para OVI_Renderer
        if self.control_bus:
            self.control_bus.publish("VISUALIZATION_REQUESTED", {
                "scene_descriptor": descriptor
            })
            logger.info("Evento VISUALIZATION_REQUESTED publicado no ControlBus")

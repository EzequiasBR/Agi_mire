# core/services/multimodal/ovi_renderer.py

import logging
import os
import time
from typing import Dict, Any

try:
    import numpy as np
    import trimesh
    import pyrender
except ImportError:
    trimesh = None
    pyrender = None

from ...utils import timestamp_id

logger = logging.getLogger("OVI_Renderer")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s OVI_Renderer: %(levelname)s: %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)


class OVIRenderer:
    """
    Camada responsável por receber scene descriptors e gerar imagens/vetores.
    """

    def __init__(self, control_bus: Any, output_dir: str = "/tmp/ovi_renders"):
        self.control_bus = control_bus
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Subscrever eventos
        if self.control_bus:
            self.control_bus.subscribe("VISUALIZATION_REQUESTED", self.handle_render_request)
            logger.info("OVIRenderer pronto para escutar VISUALIZATION_REQUESTED")

    # -------------------------
    # Event handler
    # -------------------------
    def handle_render_request(self, payload: Dict[str, Any]):
        scene_descriptor = payload.get("scene_descriptor", {})
        fidelity = scene_descriptor.get("metadata", {}).get("fidelity", "placeholder")
        scene_id = scene_descriptor.get("scene_id", timestamp_id("scene"))

        try:
            if fidelity == "placeholder":
                render_path = self._render_placeholder(scene_descriptor)
            else:
                render_path = self._render_lowpoly(scene_descriptor)

            # Publica resultado
            if self.control_bus:
                self.control_bus.publish("VISUALIZATION_READY", {
                    "request_id": scene_id,
                    "render_path": render_path,
                    "fidelity": fidelity,
                    "timestamp": time.time()
                })
                logger.info("Render pronto: %s", render_path)

        except Exception as e:
            logger.exception("Falha na renderização: %s", e)
            if self.control_bus:
                self.control_bus.publish("VISUALIZATION_FAILED", {
                    "request_id": scene_id,
                    "error": str(e)
                })

    # -------------------------
    # Renderização Placeholder (rápida)
    # -------------------------
    def _render_placeholder(self, descriptor: Dict[str, Any]) -> str:
        """
        Renderização rápida usando Pyrender ou mock de imagem.
        """
        if pyrender is None or trimesh is None:
            # Fallback: gera arquivo dummy
            path = os.path.join(self.output_dir, f"{descriptor['scene_id']}_placeholder.png")
            with open(path, "wb") as f:
                f.write(b"\x89PNG\r\n\x1a\n")  # Cabeçalho PNG vazio
            return path

        # Construir cena Pyrender
        scene = pyrender.Scene()
        for obj in descriptor.get("objects", []):
            mesh_name = obj.get("mesh", "builtin_cube")
            mesh = trimesh.creation.box()  # Placeholder para todos os objetos
            scene.add(pyrender.Mesh.from_trimesh(mesh), pose=np.eye(4))

        # Configura câmera
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        cam_pose = np.eye(4)
        cam_pose[0:3, 3] = descriptor.get("camera", {}).get("position", [0, 5, 10])
        scene.add(camera, pose=cam_pose)

        # Render offscreen
        r = pyrender.OffscreenRenderer(viewport_width=320, viewport_height=240)
        color, _ = r.render(scene)
        r.delete()

        # Salva imagem
        import imageio
        path = os.path.join(self.output_dir, f"{descriptor['scene_id']}_placeholder.png")
        imageio.imwrite(path, color)
        return path

    # -------------------------
    # Renderização Low-Poly (futuro)
    # -------------------------
    def _render_lowpoly(self, descriptor: Dict[str, Any]) -> str:
        """
        Renderização low-poly (pode usar Blender ou Pyrender avançado).
        """
        # Por enquanto, usa mesmo placeholder para simplificar
        return self._render_placeholder(descriptor)

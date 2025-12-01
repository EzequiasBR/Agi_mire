# core/services/multimodal/ovi_renderer.py

import logging
import time
from typing import Dict, Any

logger = logging.getLogger("OVI_Renderer")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s OVI_Renderer: %(levelname)s: %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)

# Import de eventos do ControlBus
from core.orchestration.control_bus import SystemEvents


class OVIRenderer:
    def __init__(self, control_bus=None):
        """
        Inicializa o OVIRenderer.
        :param control_bus: Instância do ControlBus para publicação de eventos
        """
        self.control_bus = control_bus

    # -------------------------
    # Métodos de renderização
    # -------------------------
    def _render_placeholder(self, scene_descriptor: dict) -> str:
        """
        Renderização de placeholder fictícia.
        :param scene_descriptor: Descrição da cena
        :return: Caminho do arquivo renderizado
        """
        scene_id = scene_descriptor.get("scene_id", "unknown")
        logger.debug(f"Rendering placeholder for scene {scene_id}")
        return f"/tmp/placeholder_{scene_id}.png"

    def _render_lowpoly(self, scene_descriptor: dict) -> str:
        """
        Renderização lowpoly fictícia.
        :param scene_descriptor: Descrição da cena
        :return: Caminho do arquivo renderizado
        """
        scene_id = scene_descriptor.get("scene_id", "unknown")
        logger.debug(f"Rendering lowpoly for scene {scene_id}")
        return f"/tmp/lowpoly_{scene_id}.png"

    # -------------------------
    # Event handler (ASINCRONO)
    # -------------------------
    async def handle_render_request(self, event_payload: Dict[str, Any]):
        """
        Trata eventos de renderização de cena recebidos pelo ControlBus.
        Publica sucesso ou falha após a execução.
        """
        request_id = event_payload.get("request_id", str(time.time()))
        source_module = event_payload.get("source_module", "unknown")

        scene_descriptor = event_payload.get("data", {}).get("scene_descriptor", {})
        fidelity = scene_descriptor.get("metadata", {}).get("fidelity", "placeholder")
        scene_id = scene_descriptor.get("scene_id", request_id)

        render_start_time = time.time()

        try:
            if fidelity == "placeholder":
                render_path = self._render_placeholder(scene_descriptor)
            else:
                render_path = self._render_lowpoly(scene_descriptor)

            # Publica evento de sucesso
            if self.control_bus:
                await self.control_bus.publish(
                    event_type=SystemEvents.VISUALIZATION_READY,
                    payload={
                        "render_path": render_path,
                        "fidelity": fidelity,
                        "render_latency_ms": int((time.time() - render_start_time) * 1000)
                    },
                    source_module="OVIRenderer",
                    request_id=request_id
                )
                logger.info("Render pronto: %s", render_path)

        except Exception as e:
            logger.error("Falha na renderização de %s: %s", request_id, e)
            if self.control_bus:
                await self.control_bus.publish(
                    event_type=SystemEvents.SERVICE_FAILURE,
                    payload={
                        "error_type": "RENDER_EXCEPTION",
                        "error_message": str(e),
                        "component": "OVIRenderer",
                        "related_id": scene_id
                    },
                    source_module="OVIRenderer",
                    request_id=request_id
                )

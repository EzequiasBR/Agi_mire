"""
ControlBus V2 - Barramento de Controle de Eventos (Pub/Sub)
Versão estendida com unsubscribe e suporte assíncrono.
PADRONIZADO para Auditoria.
"""
from __future__ import annotations
import logging
import time
import json
import asyncio
import uuid  # Necessário para gerar request_id
from typing import Any, Dict, Callable, List

# Fallback Logger (mantido)
def setup_logger(name: str):
    l = logging.getLogger(name)
    if not l.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s"))
        l.addHandler(ch)
    l.setLevel(logging.INFO)
    return l

logger = setup_logger("ControlBus")


# -------------------------------------------------------------------
# Definição de eventos críticos (Reforçada)
# -------------------------------------------------------------------
class SystemEvents:
    # Eventos de Governança e Memória
    LO_TRIGGERED = "LearningOptimizationTriggered"
    ROLLBACK_INITIATED = "RollbackInitiated"
    STATE_PERSISTED = "StatePersisted"
    NEW_MEMORY_STORED = "NewMemoryStored"
    PARAM_ADJUSTED = "AdaptationParametersAdjusted"
    SNAPSHOT_SAVED = "SnapshotSaved"
    ROLLBACK_REQUESTED = "RollbackRequested"
    
    # Eventos de Visualização (OVI)
    VISUALIZATION_REQUESTED = "VisualizationRequested"
    VISUALIZATION_READY = "VisualizationReady"
    DEBUG_VISUALIZATION_REQUESTED = "DebugVisualizationRequested"
    
    # Eventos de Falha e Segurança (Padronizado)
    SERVICE_FAILURE = "ServiceFailure"
    INTEGRITY_VIOLATION = "INTEGRITY_VIOLATION"


# -------------------------------------------------------------------
# ControlBus Final
# -------------------------------------------------------------------
class ControlBus:
    def __init__(self):
        self._subscribers: Dict[str, List[Callable[[Dict[str, Any]], None]]] = {}
        self.event_history: List[Dict[str, Any]] = []  # Histórico para auditoria
        self._valid_events = set(vars(SystemEvents).values())

        logger.info(json.dumps({
            "event": "ControlBus_initialized",
            "timestamp": time.time()
        }))

    # ---------------------------------------------------------
    # Inscrição (Subscription)
    # ---------------------------------------------------------
    def subscribe(self, event_type: str, handler: Callable[[Dict[str, Any]], None]):
        if event_type not in self._valid_events:
            logger.warning(f"Event type '{event_type}' not standardized in SystemEvents.")

        if event_type not in self._subscribers:
            self._subscribers[event_type] = []

        if handler not in self._subscribers[event_type]:
            self._subscribers[event_type].append(handler)
            logger.debug(json.dumps({
                "event": "ControlBus_subscribe",
                "event_type": event_type,
                "handler": handler.__qualname__,
                "timestamp": time.time()
            }))

    # ---------------------------------------------------------
    # Cancelar inscrição (Unsubscribe)
    # ---------------------------------------------------------
    def unsubscribe(self, event_type: str, handler: Callable[[Dict[str, Any]], None]):
        if event_type in self._subscribers:
            if handler in self._subscribers[event_type]:
                self._subscribers[event_type].remove(handler)
                logger.debug(json.dumps({
                    "event": "ControlBus_unsubscribe",
                    "event_type": event_type,
                    "handler": handler.__qualname__,
                    "timestamp": time.time()
                }))

    # ---------------------------------------------------------
    # Publicação (Publication) - Assíncrona e Padronizada
    # ---------------------------------------------------------
    async def publish(
        self, event_type: str, payload: Dict[str, Any], source_module: str = "unknown", request_id: str = ""
    ):
        if not isinstance(payload, dict):
            raise ValueError(f"Payload must be dict, got {type(payload)}")

        final_payload = {
            "request_id": request_id or str(uuid.uuid4()),
            "source_module": source_module,
            "timestamp_publish": time.time(),
            "event_type": event_type,
            "data": payload.copy()
        }

        listeners = self._subscribers.get(event_type, [])

        event_record = {
            "event": "ControlBus_publish",
            "event_type": event_type,
            "payload_size": len(json.dumps(final_payload)),
            "listeners_count": len(listeners),
            "source_module": source_module,
            "timestamp": time.time()
        }

        self.event_history.append(final_payload)

        if listeners:
            logger.info(json.dumps(event_record))
        else:
            logger.debug(json.dumps(event_record))

        # ---------------------------------------------------------
        # Wrapper para executar handlers com captura de exceção
        # ---------------------------------------------------------
        async def _run_handler(handler, payload):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(payload)
                else:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, handler, payload)
            except Exception as e:
                logger.error(json.dumps({
                    "event": "ControlBus_handler_error",
                    "event_type": event_type,
                    "handler": handler.__qualname__,
                    "error": str(e),
                    "timestamp": time.time(),
                    "request_id": payload.get("request_id", "")
                }))

        tasks = [_run_handler(handler, final_payload) for handler in listeners]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

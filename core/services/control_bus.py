# core/services/control_bus.py
"""
ControlBus V2 - Barramento de Controle de Eventos (Pub/Sub)
Versão estendida com unsubscribe e suporte assíncrono.
"""
from __future__ import annotations
import logging
import time
import json
import asyncio
from typing import Any, Dict, Callable, List

# Fallback Logger
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
# Definição de eventos críticos
# -------------------------------------------------------------------
class SystemEvents:
    LO_TRIGGERED = "LearningOptimizationTriggered"
    ROLLBACK_INITIATED = "RollbackInitiated"
    STATE_PERSISTED = "StatePersisted"
    NEW_MEMORY_STORED = "NewMemoryStored"
    PARAM_ADJUSTED = "AdaptationParametersAdjusted"
    SNAPSHOT_SAVED = "SnapshotSaved"
    ROLLBACK_REQUESTED = "RollbackRequested"
    VISUALIZATION_REQUESTED = "VisualizationRequested"
    VISUALIZATION_READY = "VisualizationReady"
    VISUALIZATION_FAILED = "VisualizationFailed"
    DEBUG_VISUALIZATION_REQUESTED = "DebugVisualizationRequested"


# -------------------------------------------------------------------
# ControlBus Final
# -------------------------------------------------------------------
class ControlBus:
    def __init__(self):
        self._subscribers: Dict[str, List[Callable[[Dict[str, Any]], None]]] = {}
        self.event_history: List[Dict[str, Any]] = []  # Histórico para auditoria
        logger.info(json.dumps({
            "event": "ControlBus_initialized",
            "timestamp": time.time()
        }))

    # ---------------------------------------------------------
    # Inscrição (Subscription)
    # ---------------------------------------------------------
    def subscribe(self, event_type: str, handler: Callable[[Dict[str, Any]], None]):
        if event_type not in vars(SystemEvents).values():
            raise ValueError(f"Event type '{event_type}' is not defined in SystemEvents.")

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
    # Publicação (Publication) - Assíncrona
    # ---------------------------------------------------------
    async def publish(self, event_type: str, payload: Dict[str, Any], source_module: str = "unknown"):
        if not isinstance(payload, dict):
            raise ValueError(f"Payload must be dict, got {type(payload)}")

        listeners = self._subscribers.get(event_type, [])
        event_record = {
            "event": "ControlBus_publish",
            "event_type": event_type,
            "payload_size": len(payload),
            "listeners_count": len(listeners),
            "source_module": source_module,
            "timestamp": time.time()
        }

        # Armazena histórico
        self.event_history.append({
            "event_type": event_type,
            "payload": payload.copy(),
            "source_module": source_module,
            "timestamp": time.time()
        })

        # Log estruturado
        if listeners:
            logger.info(json.dumps(event_record))
        else:
            logger.debug(json.dumps(event_record))

        # Executa handlers de forma assíncrona
        tasks = []
        for handler in listeners:
            try:
                if asyncio.iscoroutinefunction(handler):
                    tasks.append(asyncio.create_task(handler(payload)))
                else:
                    # Executa em thread separada para não bloquear
                    loop = asyncio.get_event_loop()
                    tasks.append(loop.run_in_executor(None, handler, payload))
            except Exception as e:
                logger.error(json.dumps({
                    "event": "ControlBus_handler_error",
                    "event_type": event_type,
                    "handler": handler.__qualname__,
                    "error": str(e),
                    "timestamp": time.time()
                }))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
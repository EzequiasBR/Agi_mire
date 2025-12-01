"""
AlertService V3.1 (Ajustado)
- Persistência física de estado para auditoria offline.
- Contexto enriquecido com source_module.
- Integração completa com o ciclo de governança via STATE_PERSISTED.
"""

from __future__ import annotations
import time
import json
import logging
from typing import Any, Dict, Optional
import asyncio
import os
import aiofiles # Biblioteca para I/O assíncrono
import asyncio # Biblioteca para I/O assíncrono

# Mock de ControlBus e SystemEvents (Se necessário para rodar o arquivo como main, senão o import real deve funcionar)
try:
    from core.orchestration.control_bus import ControlBus, SystemEvents
except ImportError:
    # Fallback Mocks para teste
    class SystemEvents:
        STATE_PERSISTED = "STATE_PERSISTED"
    
    class ControlBus:
        def __init__(self):
            self.events_published = []
            self.subscriptions = {}
        async def publish(self, event_type: str, payload: Dict[str, Any], source_module: str = "unknown"):
            print(f"[ControlBus:{source_module}] Publicado: {event_type}...")
            self.events_published.append({"event": event_type, "payload": payload, "source": source_module})
        def subscribe(self, event: str, handler: Any):
            self.subscriptions[event] = handler # Mock simples

    class MockFile:
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc, tb):
            pass
        async def write(self, content):
            print(f"[Aiofiles Mock] Salvando estado em {ALERT_STATE_FILE}...")

    class MockAiofiles:
        @staticmethod
        async def open(filename, mode):
            return MockFile()

    aiofiles = MockAiofiles


# -------------------------------------------------------------------------
# Configuração
# -------------------------------------------------------------------------
ALERT_LOG_DIR = "data/logs"
ALERT_STATE_FILE = os.path.join(ALERT_LOG_DIR, "alert_service_state.json")


# -------------------------------------------------------------------------
# Logger
# -------------------------------------------------------------------------
def setup_logger(name: str):
    l = logging.getLogger(name)
    if not l.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s"))
        l.addHandler(ch)
    l.setLevel(logging.INFO)
    return l


logger = setup_logger("AlertService")


# -------------------------------------------------------------------------
# Eventos internos do AlertService
# -------------------------------------------------------------------------
class AlertEvents:
    ALERT_RAISED = "ALERT_RAISED"
    ATTENTION_UPDATED = "ATTENTION_UPDATED"
    STATE_PERSISTED = SystemEvents.STATE_PERSISTED


# -------------------------------------------------------------------------
# AlertService
# -------------------------------------------------------------------------
class AlertService:
    def __init__(self, control_bus: ControlBus, attention_event: str = "ATTENTION_UPDATED"):
        self.control_bus = control_bus

        # contadores e estado
        self.critical_count = 0
        self.warning_count = 0
        self.info_count = 0
        self.total_alerts = 0

        self.last_alert: Optional[Dict[str, Any]] = None
        self.last_update_ts: Optional[float] = None
        self.state_version: int = 0
        
        # Cria diretório de logs se não existir
        os.makedirs(ALERT_LOG_DIR, exist_ok=True) 
        
        # Tenta carregar estado anterior (opcional)
        asyncio.run(self._load_state_from_file()) # Carrega de forma síncrona/assíncrona na inicialização
        
        # subscrição
        control_bus.subscribe(attention_event, self._on_attention_updated)

        logger.info("AlertService initialized. State file: %s", ALERT_STATE_FILE)

    # -----------------------------------------------------------------
    # Persistência Física (Ajuste 1)
    # -----------------------------------------------------------------
    async def _save_state_to_file(self):
        """Salva o estado completo em arquivo JSON para auditoria offline."""
        try:
            state_data = self.get_state()
            # Certifica-se de que o estado é serializável (np.array não deve estar aqui)
            state_json = json.dumps(state_data, indent=2)

            async with aiofiles.open(ALERT_STATE_FILE, mode='w') as f:
                await f.write(state_json)
            logger.debug("State persisted to file successfully.")
        except Exception as e:
            logger.error("Failed to save state to file %s: %s", ALERT_STATE_FILE, str(e))
            
    async def _load_state_from_file(self):
        """Tenta carregar o estado a partir do arquivo na inicialização."""
        try:
            async with aiofiles.open(ALERT_STATE_FILE, mode='r') as f:
                state_json = await f.read()
            state_data = json.loads(state_json)
            
            # Carrega dados
            self.critical_count = state_data["alert_counts"].get("critical_count", 0)
            self.warning_count = state_data["alert_counts"].get("warning_count", 0)
            self.info_count = state_data["alert_counts"].get("info_count", 0)
            self.total_alerts = state_data.get("total_alerts", 0)
            self.last_update_ts = state_data.get("last_update_ts")
            self.state_version = state_data.get("state_version", 0)
            self.last_alert = state_data.get("last_alert")
            
            logger.info("State loaded from file. Total alerts: %d", self.total_alerts)

        except FileNotFoundError:
            logger.warning("State file not found. Starting with clean slate.")
        except Exception as e:
            logger.error("Failed to load state from file: %s", str(e))

    # -----------------------------------------------------------------
    # Handler de atualização de atenção
    # -----------------------------------------------------------------
    async def _on_attention_updated(self, payload: Dict[str, Any]):
        try:
            A = float(payload.get("A", 0.0))
            # Ajuste 2: Rastrear source_module do evento original
            source_module = payload.get("source_module", "unknown_monitor") 
        except Exception:
            return

        if A >= 0.8:
            await self.raise_warning(
                message=f"Attention Level (A={A:.4f}) exceeded threshold (0.80). System requires deep processing or MCH review.",
                context={"A_level": A, "threshold": 0.8},
                # Ajuste 2: Passar source real
                source=f"AlertService/AttentionMonitor({source_module})" 
            )

    # -----------------------------------------------------------------
    # Raise CRITICAL / WARNING / INFO
    # -----------------------------------------------------------------
    async def raise_critical(self, message: str, context: Dict[str, Any], source: str = "unknown"):
        self.critical_count += 1
        await self._emit("CRITICAL", message, context, source)

    async def raise_warning(self, message: str, context: Dict[str, Any], source: str = "unknown"):
        self.warning_count += 1
        await self._emit("WARNING", message, context, source)

    async def raise_info(self, message: str, context: Dict[str, Any], source: str = "unknown"):
        self.info_count += 1
        await self._emit("INFO", message, context, source)

    # -----------------------------------------------------------------
    # Núcleo do sistema de alertas
    # -----------------------------------------------------------------
    async def _emit(self, level: str, message: str, context: Dict[str, Any], source: str):
        now = time.time()

        event_payload = {
            "event_type": AlertEvents.ALERT_RAISED,
            "level": level,
            "message": message,
            "context": context,
            "timestamp": now,
            # Ajuste 2: Contexto ampliado com source_module no payload do evento
            "source_module": source, 
            "counts": {
                "critical_count": self.critical_count,
                "warning_count": self.warning_count,
                "info_count": self.info_count
            }
        }

        self.total_alerts += 1
        self.last_alert = event_payload
        self.last_update_ts = now
        self.state_version += 1

        # log estruturado (mantido)
        logger.log(
            logging.CRITICAL if level == "CRITICAL" else logging.WARNING if level == "WARNING" else logging.INFO,
            json.dumps(event_payload)
        )

        # 1. Publish ALERT_RAISED
        await self.control_bus.publish(
            event_type=AlertEvents.ALERT_RAISED,
            payload=event_payload,
            source_module=source # A source real (que chamou raise_x)
        )

        # 2. Persistência física (Ajuste 1)
        await self._save_state_to_file()
        
        # 3. Publish STATE_PERSISTED (Ajuste 3: Fecha o ciclo de governança/auditoria)
        await self.control_bus.publish(
            event_type=AlertEvents.STATE_PERSISTED,
            # Inclui o estado completo (get_state) no payload para que o Monitor possa registrar as métricas
            payload={"module": "AlertService", "state": self.get_state()}, 
            source_module="AlertService"
        )

    # -----------------------------------------------------------------
    # Estado completo para monitoramento
    # -----------------------------------------------------------------
    def get_state(self) -> Dict[str, Any]:
        """Retorna o estado completo para persistência PCVS/ControlBus/Arquivo."""
        return {
            "last_alert": self.last_alert,
            "alert_counts": {
                "critical_count": self.critical_count,
                "warning_count": self.warning_count,
                "info_count": self.info_count
            },
            "total_alerts": self.total_alerts,
            "last_update_ts": self.last_update_ts,
            "state_version": self.state_version
        }

# -------------------------------------------------------------------------
# Teste Rápido
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Mock do aiofiles para simular I/O assíncrono
    class MockAiofiles:
        _content = ""
        def __init__(self, filename, mode):
            self.filename = filename
            self.mode = mode
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc, tb):
            pass
        async def write(self, content):
            MockAiofiles._content = content # Simula a escrita
            print(f"[Aiofiles Mock] Simulação de Salvamento em: {self.filename}")
        async def read(self):
            # Simula a leitura
            if self.filename.endswith("state.json") and self._content:
                print(f"[Aiofiles Mock] Simulação de Leitura de: {self.filename}")
                return self._content
            raise FileNotFoundError # Simula arquivo não encontrado na 1ª execução

    aiofiles = MockAiofiles
    
    async def demo():
        control_bus_mock = ControlBus()
        alert_service = AlertService(control_bus=control_bus_mock)
        
        print("\n--- 1. Simular Alerta Crítico (Ajustes 1, 2 e 3) ---")
        # source: "SecurityService" (Ajuste 2)
        await alert_service.raise_critical(
            message="Vector Integrity FAILED", 
            context={"vector_hash": "A1B2C3D4"}, 
            source="SecurityService"
        )
        
        # O _save_state_to_file foi chamado (Ajuste 1)
        # O STATE_PERSISTED foi publicado (Ajuste 3)
        assert control_bus_mock.events_published[-1]["event"] == SystemEvents.STATE_PERSISTED
        assert control_bus_mock.events_published[-2]["payload"]["source_module"] == "SecurityService" # Verifica Ajuste 2

        print("\n--- 2. Simular Evento de Atenção (Trigger Warning) ---")
        # Simula um evento de atenção vindo do Monitor
        monitor_payload = {"A": 0.85, "source_module": "MonitorService/A_Metrics"}
        await alert_service._on_attention_updated(monitor_payload)
        
        # Verifica se o source do Warning está correto (Ajuste 2 no _on_attention_updated)
        print(f"Source do Warning: {control_bus_mock.events_published[-2]['payload']['source_module']}")
        
        # Verifica contadores
        print(f"Contador Crítico: {alert_service.critical_count}, Contador Warning: {alert_service.warning_count}")
        assert alert_service.critical_count == 1
        assert alert_service.warning_count == 1
        
        print("\n--- 3. Auditoria Offline (Simulação de Reload) ---")
        # Reload/Auditoria
        print(f"Estado salvo (total alerts): {json.loads(MockAiofiles._content)['total_alerts']}")
        
        # Simula o início de um novo serviço lendo o estado salvo
        new_alert_service = AlertService(control_bus=ControlBus())
        assert new_alert_service.total_alerts == 2 # Deve ter carregado o estado anterior
        

    asyncio.run(demo())
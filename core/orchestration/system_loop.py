import time
import logging
import json
from typing import Dict, Any
from datetime import datetime

# Importações reais (assumindo estrutura final)

from core.governance.prag import PRAG
from core.governance.simlog import SimLog
from core.intelligence.ppo import PPO
from core.orchestration.mch import MCH
from core.services.control_bus import ControlBus, SystemEvents
from core.services.monitor import Monitor
from core.services.pcvs import PCVS
from core.services.perception import PerceptionAPI  
from ..services.utils import timestamp_id, hash_state, setup_logger, divergence_from_cosine

# Configuração do logger principal
logger = setup_logger("SystemLoop")

class SystemLoop:
    """
    Orquestrador contínuo do Agi_mire.
    Coordena Perception, MCH, PRAG, PPO, PCVS, Monitor e ControlBus.
    """

    def __init__(self, components: Dict[str, Any]):
        self.components = components

        # Módulos críticos
        self.mch: MCH = components['mch']
        self.prag: PRAG = components['prag']
        self.ppo = components.get('ppo', None)
        self.control_bus: ControlBus = components['control_bus']
        self.pcvs: PCVS = components['pcvs']
        self.monitor: Monitor = components['monitor']
        self.perception:  PerceptionAPI = components['perception']
        self.simlog: SimLog = components['simlog']

        # Parâmetros de controle
        self.cycle_count = 0
        self.last_report_time = time.time()
        self.report_interval_seconds = 600
        self.snapshot_frequency = 100

        logger.info("SystemLoop inicializado com todos os componentes injetados.")

    # --- Governança ---
    def _handle_rollback(self, cycle_id: str, snapshot_hash: str):
        logger.critical(f"⚠️ ROLLBACK INICIADO (Cycle ID: {cycle_id})")
        self.control_bus.publish(SystemEvents.ROLLBACK_INITIATED, {'target_hash': snapshot_hash})
        self.pcvs.rollback(target_hash=snapshot_hash)
        logger.info(f"Rollback concluído. Estado restaurado para Hash: {snapshot_hash}")

    def _handle_ontogenesis(self, cycle_id: str):
        if not self.ppo:
            return
        logger.warning(f"🚀 ONTOGÊNESE (PPO) ACIONADA (Cycle ID: {cycle_id})")
        self.control_bus.publish(SystemEvents.LO_TRIGGERED, {'reason': self.ppo.last_reason})
        self.ppo.execute_ontogenesis()
        self._persist_current_state(cycle_id)

    def _persist_current_state(self, cycle_id: str):
        snapshot_data = self.mch.get_full_state_snapshot()
        snapshot_hash = hash_state(snapshot_data)
        self.pcvs.persist_snapshot(snapshot_data, reason=f"Cycle_{cycle_id}")
        logger.info(f"[PCVS] Snapshot salvo. Hash: {snapshot_hash[:10]}")

    # --- Loop principal ---
    def _run_cycle(self):
        self.cycle_count += 1
        start_time = time.time()
        cycle_id = timestamp_id("cycle")

        logger.info(f"\n--- INÍCIO DO CICLO {self.cycle_count} (ID: {cycle_id}) ---")

        # 1. Obter e validar input
        try:
            raw_input = self.perception.get_raw_input()
            validated_input = self.perception.validate_and_sanitize(raw_input)
        except Exception as e:
            logger.error(f"[SEGURANÇA] Falha na validação do Input: {e}")
            return

        # 2. Processamento Cognitivo (MCH)
        try:
            result = self.mch.process(validated_input)
            divergence_D = divergence_from_cosine(
                self.simlog.calculate_divergence(result.vector, result.tripla_logica)
            )
        except Exception as e:
            logger.error(f"[EXECUÇÃO] Erro crítico no MCH: {e}")
            return

        # 3. Avaliação PRAG
        prag_decision = self.prag.evaluate_state(divergence_D)
        if prag_decision['action'] == 'ROLLBACK':
            self._handle_rollback(cycle_id, prag_decision['target_hash'])
            return
        elif prag_decision['action'] == 'PPO_TRIGGER':
            self._handle_ontogenesis(cycle_id)

        logger.info(f"[PRAG RESULT] Ação: {prag_decision['action']}, Divergência (D): {divergence_D:.4f}")

        # 4. Persistência periódica
        if self.cycle_count % self.snapshot_frequency == 0:
            self._persist_current_state(cycle_id)

        # 5. Telemetria
        cycle_duration = time.time() - start_time
        self.monitor.update_metrics(cycle_duration, divergence_D, self.cycle_count)

        # 6. Exportar relatório
        if time.time() - self.last_report_time > self.report_interval_seconds:
            self.monitor.export_report()
            self.last_report_time = time.time()
            logger.info("[MONITOR] Relatório exportado.")

        logger.info(f"--- FIM DO CICLO {self.cycle_count}. Duração: {cycle_duration:.4f}s ---")

    def start_loop(self, max_cycles: int = -1):
        logger.info("Iniciando loop do sistema...")
        while max_cycles == -1 or self.cycle_count < max_cycles:
            self._run_cycle()
            time.sleep(0.01)
        logger.info("Loop do sistema finalizado.")


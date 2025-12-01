# core/system_loop.py

import time
import logging
from typing import Dict, Any, Optional


from core.config_loader import ConfigLoader
from core.governance.prag import PRAG
from core.governance.simlog import SimLog
from core.intelligence.ppo import PPO
from core.orchestration.mch import MCH
from core.orchestration.control_bus import ControlBus, SystemEvents
from core.services.monitoring.monitor import Monitor
from core.services.pcvs import PCVS
from core.services.io.perception import PerceptionAPI

from ..services.utils import timestamp_id, hash_state, divergence_from_cosine


class SystemLoop:
    """
    Orquestrador contínuo do Agi_mire.
    Recupera últimos snapshots PCVS de forma incremental, garantindo integridade.
    """

    def __init__(self, components: Dict[str, Any], config_dir: str = "configs"):
        self.config_loader = ConfigLoader(config_dir)
        self.config_loader.load_all()
        self.logger: logging.Logger = self.config_loader.logger

        self.components = components
        self.mch: MCH = components['mch']
        self.prag: PRAG = components['prag']
        self.ppo: Optional[PPO] = components.get('ppo', None)
        self.control_bus: ControlBus = components['control_bus']
        self.pcvs: PCVS = components['pcvs']
        self.monitor: Monitor = components['monitor']
        self.perception: PerceptionAPI = components['perception']
        self.simlog: SimLog = components['simlog']

        # Configuração de thresholds
        if self.ppo:
            self.ppo.set_thresholds(**self.config_loader.get_module_thresholds("PPO"))
        self.prag.set_thresholds(**self.config_loader.get_module_thresholds("PRAG"))

        # Parâmetros de controle
        self.cycle_count = 0
        self.last_report_time = time.time()
        self.report_interval_seconds = 600
        self.snapshot_frequency = 100

        # Tentativa de restauração incremental
        self.logger.info("🔄 Tentando restaurar últimos snapshots PCVS...")
        restored = self.restore_incremental_snapshots(max_attempts=5)
        if restored:
            self.logger.info("✅ Snapshot PCVS restaurado com sucesso.")
        else:
            self.logger.info("⚠️ Nenhum snapshot válido encontrado. Inicializando com estado limpo.")

    # -----------------------------
    # Restauração incremental
    # -----------------------------
    def restore_incremental_snapshots(self, max_attempts: int = 5) -> bool:
        """
        Tenta restaurar até `max_attempts` últimos snapshots em ordem decrescente.
        Se um snapshot estiver corrompido, tenta o anterior.
        """
        last_hashes = self.pcvs.get_last_n_hashes(max_attempts)
        for sha in last_hashes:
            try:
                snapshot = self.pcvs.load_snapshot(sha)
                if snapshot:
                    if hash_state(snapshot) != sha:
                        self.logger.critical(f"❌ Snapshot corrompido ({sha[:10]}), tentando anterior...")
                        self.monitor.register_event("PCVS_CORRUPTION_FAIL",
                                                   {"expected_hash": sha})
                        continue
                    # Restaurar métricas MCH
                    self.cycle_count = snapshot.get('cycle_count', 0)
                    self.H_sist = snapshot.get('mch_metrics', {}).get('H', 0)
                    self.V_sist = snapshot.get('mch_metrics', {}).get('V', 0)
                    self.E_sist = snapshot.get('mch_metrics', {}).get('E', 0)
                    # Restaurar módulos
                    if hasattr(self, "hippocampus") and "hippocampus_state" in snapshot:
                        self.hippocampus.restore_state(snapshot['hippocampus_state'])
                    self.logger.warning(f"✅ Estado restaurado com sucesso do snapshot: {sha[:10]}")
                    return True
            except Exception as e:
                self.logger.error(f"[PCVS] Erro ao carregar snapshot {sha[:10]}: {e}")
        return False

    # -----------------------------
    # Governança
    # -----------------------------
    def _handle_rollback(self, cycle_id: str, snapshot_hash: str):
        self.logger.critical(f"⚠️ ROLLBACK INICIADO (Cycle ID: {cycle_id})")
        self.control_bus.publish(SystemEvents.ROLLBACK_INITIATED, {'target_hash': snapshot_hash})
        self.pcvs.rollback(target_hash=snapshot_hash)
        self.logger.info(f"Rollback concluído. Estado restaurado para Hash: {snapshot_hash}")

    def _handle_ontogenesis(self, cycle_id: str):
        if not self.ppo:
            return
        self.logger.warning(f"🚀 ONTOGÊNESE (PPO) ACIONADA (Cycle ID: {cycle_id})")
        self.control_bus.publish(SystemEvents.LO_TRIGGERED, {'reason': self.ppo.last_reason})
        self.ppo.execute_ontogenesis()
        self._persist_current_state(cycle_id)

    def _persist_current_state(self, cycle_id: str):
        snapshot_data = self.mch.get_full_state_snapshot()
        snapshot_hash = hash_state(snapshot_data)
        self.pcvs.persist_snapshot(snapshot_data, reason=f"Cycle_{cycle_id}")
        self.logger.info(f"[PCVS] Snapshot salvo. Hash: {snapshot_hash[:10]}")

    # -----------------------------
    # Loop principal
    # -----------------------------
    def _run_cycle(self):
        self.cycle_count += 1
        start_time = time.time()
        cycle_id = timestamp_id("cycle")

        self.logger.info(f"\n--- INÍCIO DO CICLO {self.cycle_count} (ID: {cycle_id}) ---")

        try:
            raw_input = self.perception.get_raw_input()
            validated_input = self.perception.validate_and_sanitize(raw_input)
        except Exception as e:
            self.logger.error(f"[SEGURANÇA] Falha na validação do Input: {e}")
            return

        try:
            result = self.mch.process(validated_input)
            divergence_D = divergence_from_cosine(
                self.simlog.calculate_divergence(result.vector, result.tripla_logica)
            )
        except Exception as e:
            self.logger.error(f"[EXECUÇÃO] Erro crítico no MCH: {e}")
            return

        prag_decision = self.prag.evaluate_state(divergence_D)
        if prag_decision['action'] == 'ROLLBACK':
            self._handle_rollback(cycle_id, prag_decision['target_hash'])
            return
        elif prag_decision['action'] == 'PPO_TRIGGER':
            self._handle_ontogenesis(cycle_id)

        self.logger.info(f"[PRAG RESULT] Ação: {prag_decision['action']}, Divergência (D): {divergence_D:.4f}")

        if self.cycle_count % self.snapshot_frequency == 0:
            self._persist_current_state(cycle_id)

        cycle_duration = time.time() - start_time
        self.monitor.update_metrics(cycle_duration, divergence_D, self.cycle_count)

        if time.time() - self.last_report_time > self.report_interval_seconds:
            self.monitor.export_report()
            self.last_report_time = time.time()
            self.logger.info("[MONITOR] Relatório exportado.")

        self.logger.info(f"--- FIM DO CICLO {self.cycle_count}. Duração: {cycle_duration:.4f}s ---")

    def start_loop(self, max_cycles: int = -1):
        self.logger.info("Iniciando loop do sistema...")
        while max_cycles == -1 or self.cycle_count < max_cycles:
            self._run_cycle()
            time.sleep(0.01)
        self.logger.info("Loop do sistema finalizado.")

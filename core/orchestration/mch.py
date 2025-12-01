"""
MasterControllerHybrid (MCH FULL) — Ciclo assíncrono completo
Integração com: OL, OEAEngine, OA, SimLog, Hippocampus, PRAG e ControlBus V2
Versão revisada para produção com robustez, logging e contratos assíncronos consistentes.
"""

from typing import Dict, Any, Optional
import time
import numpy as np
import logging
import asyncio
import uuid

# Módulos centrais
from core.orchestration.control_bus import ControlBus, SystemEvents
from core.config_loader import ConfigLoader
from core.intelligence.oa import OA
from core.intelligence.oea import OEAEngine
from core.intelligence.ol import OL
from core.memory.hippocampus import Hippocampus
from core.governance.prag import PRAG
from core.governance.simlog import SimLog


class MasterControllerHybrid:
    """
    MasterControllerHybrid FULL: ciclo assíncrono coeso com I/O, eventos e integração total de módulos.
    """

    def __init__(self, modules: Dict[str, Any], config_dir: str = "configs"):
        self.logger = logging.getLogger("MCH")
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter("%(asctime)s MCH %(levelname)s: %(message)s"))
            self.logger.addHandler(ch)
        self.logger.setLevel(logging.INFO)

        # Módulos centrais
        self.ol: OL = modules['ol']
        self.hippocampus: Hippocampus = modules['hippocampus']
        self.oea: OEAEngine = modules['oea']
        self.oa: OA = modules['oa']
        self.prag: PRAG = modules['prag']
        self.simlog: SimLog = modules['simlog']
        self.control_bus: Optional[ControlBus] = modules.get('control_bus')
        if not self.control_bus:
            self.logger.error("ControlBus não fornecido. O MCH não poderá orquestrar eventos.")

        # Configuração dinâmica
        self.config_loader = ConfigLoader(config_dir)
        self.config_loader.load_all()
        self.module_thresholds = self._load_module_thresholds()

    # -------------------------
    # Helpers de configuração
    # -------------------------
    def _load_module_thresholds(self) -> Dict[str, Dict[str, Any]]:
        thresholds = {}
        for module_name in ["ol", "oea", "oa", "prag", "simlog", "hippocampus"]:
            thresholds[module_name] = self.config_loader.get_module_thresholds(module_name)
            self.logger.info(f"Thresholds carregados para {module_name}: {thresholds[module_name]}")
        return thresholds

    def _l2_normalize(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float32)
        n = np.linalg.norm(v)
        return v if n < 1e-12 else v / n

    def _calculate_divergence(self, v_adapt: np.ndarray, v_control: np.ndarray) -> float:
        va = self._l2_normalize(v_adapt)
        vc = self._l2_normalize(v_control)
        return float(1.0 - np.dot(va, vc))

    # -------------------------
    # OVI Trigger Assíncrono
    # -------------------------
    async def _trigger_ovi_visualization(self, request_id: str, V_adapt: np.ndarray):
        if self.control_bus is None:
            self.logger.warning(f"OVI não disparada (ControlBus ausente) para sessão {request_id}")
            return
        try:
            await self.control_bus.publish(
                event_type=SystemEvents.VISUALIZATION_REQUESTED,
                payload={"fidelity": "high", "top_k": 50, "query_vector": V_adapt.tolist()},
                source_module="MCH",
                request_id=request_id
            )
            self.logger.debug(f"OVI visualization triggered for session {request_id}")
        except Exception as e:
            self.logger.error(f"Falha ao disparar OVI para session {request_id}: {e}")

    # -------------------------
    # Ciclo Coeso Assíncrono
    # -------------------------
    async def run_cohesive_cycle(self, current_input: Any) -> Dict[str, Any]:
        start_time = time.time()
        session_id = self.prag.start_new_cycle() 
        context_data = {}

        try:
            # -------------------------
            # 1. OL Vector Adaptativo
            # -------------------------
            if not isinstance(current_input, (str, bytes, dict, np.ndarray)):
                raise ValueError("current_input inválido para OL")
            try:
                V_adapt = self.ol.generate_vector_adaptativo(current_input)
            except Exception as e:
                self.prag.trigger_fail_safe(session_id, f"OL_ERROR: {e}")
                raise

            # -------------------------
            # 2. Hippocampus memórias
            # -------------------------
            if not hasattr(self.hippocampus, "top_k_records"):
                raise RuntimeError("Hippocampus API desatualizada. Necessário 'top_k_records'.")
            try:
                if asyncio.iscoroutinefunction(self.hippocampus.top_k_records):
                    context_memories = await self.hippocampus.top_k_records(query=V_adapt, k=5)
                else:
                    context_memories = self.hippocampus.top_k_records(query=V_adapt, k=5)
                context_data['memories'] = context_memories
            except Exception as e:
                self.prag.trigger_fail_safe(session_id, f"Hippocampus_ERROR: {e}")
                raise

            # -------------------------
            # 3. OEA Engine
            # -------------------------
            system_metrics = {
                "volatility": 0.05, 
                "avg_D": 0.1,
                "rollback_rate": 0.0,
                "buffer_saturation": 0.2
            }
            logical_triplet = {"input": current_input} 
            try:
                oea_output = self.oea.process_cycle(
                    cycle_id=session_id,
                    context_vector=V_adapt,
                    logical_triplet=logical_triplet,
                    system_metrics=system_metrics
                )
            except Exception as e:
                self.prag.trigger_fail_safe(session_id, f"OEA_ERROR: {e}")
                raise

            preventive_vector = oea_output.get("preventive_vector").vector if oea_output.get("preventive_vector") else np.zeros_like(V_adapt)

            # -------------------------
            # 4. Divergência normalizada
            # -------------------------
            D = self._calculate_divergence(V_adapt, preventive_vector)
            rollback_threshold = float(self.module_thresholds.get("prag", {}).get("rollback_divergence", 0.9))
            self.prag.log_divergence(session_id, D)
            if D > rollback_threshold:
                self.prag.trigger_rollback(session_id, 'Parcial')
                self.logger.warning(f"Rollback Parcial acionado por divergência {D:.4f}")

            # -------------------------
            # 5. Simbolização & OA
            # -------------------------
            hypothesis_triple, C_base = self.simlog.vector_to_triple(V_adapt, oea_output.get("emotional_signal"))
            is_valid, C_final, violations = self.oa.validate_hypothesis(
                hypothesis_triple, C_base,
                thresholds=self.module_thresholds.get("oa", {})
            )

            # -------------------------
            # 6. Aprendizado & feedback
            # -------------------------
            try:
                if hasattr(self.hippocampus, "write_memory"):
                    if asyncio.iscoroutinefunction(self.hippocampus.write_memory):
                        await self.hippocampus.write_memory(vector=V_adapt, certainty=C_final, meta={"cycle_id": session_id})
                    else:
                        self.hippocampus.write_memory(vector=V_adapt, certainty=C_final, meta={"cycle_id": session_id})
                self.simlog.consolidate_knowledge(hypothesis_triple, C_final)
                feedback_vector = self.prag.log_cycle_success(session_id, hypothesis_triple) if is_valid else self.prag.log_cycle_failure(session_id, hypothesis_triple)
            except Exception as e:
                self.prag.trigger_fail_safe(session_id, f"LEARNING_ERROR: {e}")
                raise

            self.prag.end_cycle(session_id)

            # -------------------------
            # 7. OVI visualization trigger
            # -------------------------
            if self.control_bus is None:
                self.prag.log_event(session_id, "OVI_SKIPPED_NO_BUS")
                self.logger.warning(f"OVI visualização ignorada (ControlBus ausente) para sessão {session_id}")
            else:
                await self._trigger_ovi_visualization(session_id, V_adapt)

            return {
                "status": "Ciclo Concluído e OVI Disparado" if self.control_bus else "Ciclo Concluído (OVI Skipped)",
                "divergence": D,
                "hypothesis": hypothesis_triple,
                "certainty": C_final,
                "feedback_vector": feedback_vector,
                "oea_output": oea_output,
                "time_s": time.time() - start_time
            }

        except Exception as e:
            self.prag.trigger_fail_safe(session_id, str(e))
            self.logger.exception(f"Falha crítica no MCH FULL para sessão {session_id}")
            return {"status": "Falha Crítica (Fail-Safe Ativado)", "error": str(e)}

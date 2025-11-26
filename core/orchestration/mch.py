import numpy as np
import time, logging
from typing import Dict, Any, Optional

# Importações de dependências
# Utilizamos as funções utilitárias que acabamos de padronizar.
from ..services.utils import hash_state, timestamp_id
from ..services.utils import normalize_vector, cosine_similarity, divergence_from_cosine
from ..services.control_bus import SystemEvents # Para comunicação assíncrona

# O MCH deve ser inicializado com todas as dependências injetadas pelo main.py
logger = logging.getLogger("MCH")

class MCH:
    """
    Mecanismo de Ciclo Coeso (MCH): Orquestrador central que executa o ciclo
    cognitivo (Sense -> Plan -> Act -> Govern -> Learn).
    """

    # Atributos principais (conforme a especificação)
    
    # Módulos Cognitivos e Governança
    # Recebidos via injeção de dependência no __init__
    
    def __init__(self, components: Dict[str, Any], config: Dict[str, Any]):
        
        # Módulos de Dependência Injetada
        self.pcvs = components['pcvs']
        self.monitor = components['monitor']
        self.prag = components['prag']
        self.ppo = components['ppo']
        self.hippocampus = components['hippocampus']
        self.oa = components['oa']
        self.ol = components['ol']
        self.analytics = components['analytics'] # O módulo que calculará H, V, E
        self.adaptation = components['adaptation'] # Módulo de ajuste de hiperparâmetros
        self.control_bus = components['control_bus']
        self.perception = components['perception']
        
        # Configurações do Ciclo
        self.config = config
        self.cycle_count = 0
        self.pcvs_save_interval = config.get('PCVS_SAVE_INTERVAL', 100) # Salvar a cada 100 ciclos
        
        # Estados e Métricas
        self.last_state_hash: Optional[str] = None
        self.H_sist: float = 0.0 # Hierarquia (Consciência)
        self.V_sist: float = 0.0 # Validade (Coerência)
        self.E_sist: float = 0.0 # Eficiência (Performance)
        
        logger.info("MCH inicializado com injeção de todas as dependências.")

    # --- Métodos de Estado e Persistência (Status: Implementado/Ajustar) ---

    # Nota: As funções auxiliares (_normalize_vector, etc.) não precisam ser redefinidas, 
    # pois foram implementadas diretamente no core/utils.py e são acessadas via import.

    def _compose_system_state(self) -> Dict[str, Any]:
        """
        Compõe o dicionário de estado completo do sistema para snapshots PCVS.
        """
        state = {
            "cycle_count": self.cycle_count,
            "mch_metrics": {"H": self.H_sist, "V": self.V_sist, "E": self.E_sist},
            "hippocampus_state": self.hippocampus.get_state(),
            "ppo_state": self.ppo.get_state(),
            "prag_state": self.prag.get_state(),
            "ol_state": self.ol.get_state(),
            # Incluir o estado de outros módulos (Analytics, Adaptation, etc.)
            "timestamp": time.time()
        }
        return state

    def save_pcvs_snapshot(self) -> Optional[str]:
        """ Persiste o estado atual no PCVS e atualiza last_state_hash. """
        current_state = self._compose_system_state()
        self.last_state_hash = hash_state(current_state) # Usa a função de utilidade
        self.pcvs.save_snapshot(self.last_state_hash, current_state)
        return self.last_state_hash
    
    def force_save_snapshot(self) -> Optional[str]:
        """ Implementação do método que força o salvamento de snapshot atual. """
        return self.save_pcvs_snapshot()

    def load_pcvs_snapshot(self, sha: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """ Recupera snapshot e restaura estados dos subsistemas. """
        snapshot = self.pcvs.load_snapshot(sha or self.last_state_hash)
        if snapshot:
            logger.warning(f"Restaurando estado para o hash: {sha or self.last_state_hash[:10]}")
            
            # 1. Restaurar MCH
            self.cycle_count = snapshot['cycle_count']
            self.H_sist = snapshot['mch_metrics']['H']
            # ... (Restaurar V_sist, E_sist)
            
            # 2. Delegar Restauração aos Módulos
            self.hippocampus.restore_state(snapshot['hippocampus_state'])
            # ... (Delegar restauração a PPO, PRAG, OL, etc.)
            
            return snapshot
        return None

    def rollback_total(self) -> Dict[str, Any]:
        """ Realiza um Rollback Completo para o último snapshot PCVS válido. """
        logger.critical("🚨 ROLLBACK TOTAL INICIADO!")
        self.control_bus.publish(SystemEvents.ROLLBACK_INITIATED, {"type": "total"})
        
        # 1. Carregar e Restaurar o último estado válido
        self.load_pcvs_snapshot() 
        
        # 2. Informar o Monitor
        self.monitor.register_event("ROLLBACK", {"type": "total", "hash": self.last_state_hash})
        
        return self._compose_system_state()

    def rollback_partial(self) -> Dict[str, Any]:
        """ Realiza um Rollback Parcial (ex: apenas memória e monitor). """
        logger.warning("⚠️ ROLLBACK PARCIAL INICIADO!")
        self.control_bus.publish(SystemEvents.ROLLBACK_INITIATED, {"type": "partial"})
        
        # 1. Rollback na Memória (ex: reverter últimas N entradas)
        self.hippocampus.rollback_partial() 
        
        # 2. Rollback em outros estados voláteis
        # ... (lógica específica para PPO ou OL, se necessário)
        
        # 3. Informar o Monitor
        self.monitor.register_event("ROLLBACK", {"type": "partial"})
        
        return self._compose_system_state()
    
    def inspect_last_pcvs(self) -> Dict[str, Any]:
        """ Apenas retorna o estado carregado do último hash. """
        return self.pcvs.load_snapshot(self.last_state_hash) or {}

    def shutdown(self):
        """ Salva snapshot final e encerra. """
        self.save_pcvs_snapshot()
        logger.info(f"MCH encerrado no ciclo {self.cycle_count}. Snapshot final salvo.")
        # O ControlBus e o SystemLoop cuidarão de fechar outros recursos (workers).


    # --- O MÉTODO CRÍTICO: execute_cycle() (Status: Parcial/Implementar) ---

    def execute_cycle(self, input_data: Any, inject_pathogen: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Executa um ciclo completo do agente (Sense -> Plan -> Govern -> Learn).
        
        Args:
            input_data: Dados de entrada multimodal (simulados ou reais do Perception).
            inject_pathogen: Dados para simular ruído ou falha (para testes).
        """
        self.cycle_count += 1
        start_time = time.time()
        
        logger.info(f"--- INICIANDO CICLO {self.cycle_count} ---")
        
        # 0. INJEÇÃO DE PATÓGENOS (Debugging / Stress Testing)
        if inject_pathogen:
            logger.warning(f"🧬 Patógeno injetado no ciclo {self.cycle_count}.")
            # Lógica para corromper input ou métricas aqui, se necessário.

        # 1. PERCEPÇÃO (Sense & Embedding)
        # O Perception API fornece a representação vetorial (embedding) do input multimodal.
        # Ajuste: Assumimos que o Perception já retorna um vetor normalizado (ou o normalizamos aqui).
        current_embedding = self.perception.process_input(input_data)
        if current_embedding is None:
            logger.error("Perception falhou ao gerar embedding. Ignorando ciclo.")
            return {"status": "FAILED", "reason": "Perception error"}
        
        # 2. RECUPERAÇÃO DE MEMÓRIA (Recall)
        # O OL (Ontogenia) ou OA (Agente Operacional) decide qual top_k recuperar.
        # Aqui, o MCH orquestra a recuperação para o PPO/PRAG.
        top_k_memories, top_k_embeddings = self.hippocampus.retrieve_top_k(
            query_embedding=current_embedding,
            k=self.config['HIPPOCAMPUS']['TOP_K']
        )
        
        # 3. CÁLCULO DE MÉTRICAS PRIMAIS (C_primal, D_primal)
        # O MCH precisa do vetor atual e das memórias (top_k) para calcular as métricas.
        
        # C_primal (Coherence): Similaridade média com as top K memórias.
        # D_primal (Divergence): Média da divergência com as top K memórias.
        
        # [Ajuste]: O cálculo destas métricas é feito no Analytics.
        C_primal, D_primal = self.analytics.calculate_primal_metrics(
            current_embedding, top_k_embeddings
        )
        
        # 4. GOVERNANÇA (PRAG - Decisão de Rollback)
        # O PRAG verifica se a D_primal ultrapassa o 'rollback_threshold' dinâmico.
        
        # [Ajuste]: O PRAG precisa dos thresholds atualizados pela Adaptation.
        rollback_threshold = self.adaptation.get_parameter('rollback_threshold')
        rollback_decision = self.prag.check_for_rollback(D_primal, rollback_threshold)
        
        if rollback_decision['action'] == "TOTAL":
            self.rollback_total()
            return {"status": "ROLLBACK_TOTAL", "reason": rollback_decision['reason']}
        elif rollback_decision['action'] == "PARTIAL":
            self.rollback_partial()
            # Reinicia o ciclo para tentar novamente com a memória limpa
            # return self.execute_cycle(input_data, inject_pathogen) # Ou apenas loga e segue
        
        # 5. PLANEJAMENTO E AÇÃO (OA/PPO)
        
        # O OA gera a intenção e submete ao PPO
        action_vector = self.oa.generate_action(current_embedding)
        
        # O PPO avalia o vetor de ação no contexto das métricas primais.
        # O PPO decide se aciona o LO (Learning Optimization).
        ppo_result = self.ppo.process_cycle(
            action_vector, C_primal, D_primal, self.adaptation.get_parameter('tau_ppo')
        )
        
        # 6. APRENDIZAGEM (LO - Ação do OL)
        if ppo_result['trigger_lo']:
            logger.warning("🧠 PPO acionou o Learning Optimization (LO).")
            # O OL faz a otimização e gera um novo vetor otimizado
            optimized_vector = self.ol.execute_lo(
                ppo_result['vector_to_optimize'],
                self.adaptation.get_parameter('hippocampus_lambda')
            )
            # [Ajuste]: Armazenar o vetor otimizado no Hippocampus.
            self.hippocampus.store_experience(optimized_vector, ppo_result['metadata'])
            self.monitor.register_event("PPO_LO_TRIGGER", ppo_result)
        
        # 7. GOVERNANÇA (Atualização de Métricas Sistêmicas e Adaptação)
        
        # [Ajuste]: O Analytics calcula as métricas sistêmicas com base nos resultados.
        self.H_sist, self.V_sist, self.E_sist = self.analytics.calculate_system_metrics(
            C_primal, D_primal, ppo_result['performance_E'] 
        )
        
        # A Adaptation ajusta os hiperparâmetros com base nas métricas sistêmicas.
        self.adaptation.adjust_parameters(self.H_sist, self.V_sist, self.E_sist)
        
        # 8. PERSISTÊNCIA PERIÓDICA PCVS
        if self.cycle_count % self.pcvs_save_interval == 0:
            self.save_pcvs_snapshot()
            logger.info("💾 Snapshot PCVS periódico salvo.")

        # 9. TELEMETRIA E FIM DE CICLO
        end_time = time.time()
        self.monitor.register_cycle(
            self.cycle_count,
            {"H_sist": self.H_sist, "V_sist": self.V_sist, "D_primal": D_primal},
            duration=end_time - start_time
        )
        
        logger.info(f"--- CICLO {self.cycle_count} CONCLUÍDO (Tempo: {end_time - start_time:.4f}s) ---")
        
        return {
            "status": "COMPLETED",
            "metrics": {"H": self.H_sist, "V": self.V_sist, "E": self.E_sist}
        }
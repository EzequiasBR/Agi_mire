# core/governance/simlog.py
"""
SimLog V2.0-PROD
Sistema de Simbolização, Divergência e Auditoria Cognitiva

Responsabilidades principais:
- Converter vetores em triplas (tradução intuitiva → simbólica)
- Determinar expected_vector (top-K centróide ou fallback KG)
- Calcular Divergência (D) real e Saúde Cognitiva (H)
- Integrar penalidades RegVet + saúde ética OEA
- Arquivar estado (PCVS)
- Emitir eventos estruturados no ControlBus
- Garantir execução assíncrona resiliente durante ciclos contínuos
"""

from __future__ import annotations
import numpy as np
import time
import json
import logging
import asyncio
from typing import Any, Dict, Optional, List, Tuple
from uuid import uuid4

# ==============================================================================
# CONFIGURAÇÃO DE LOGGING
# ==============================================================================

# Criação de um logger específico para o módulo SimLog
logger = logging.getLogger("SimLog")
if not logger.handlers:
    # Configura um StreamHandler para exibir logs no console
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s SIMLOG %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ==============================================================================
# DEPENDÊNCIAS (MOCKS PARA AMBIENTE DE TESTE/DESENVOLVIMENTO)
# Em ambiente de produção, estas classes seriam APIs assíncronas reais.
# ==============================================================================

class ControlBusMock:
    """Mock do ControlBus para simular a publicação assíncrona de eventos."""
    async def publish(self, event: str, payload: Dict[str, Any]):
        """Simula a publicação de um evento no barramento de controle."""
        # Em um sistema real, este método enviaria a payload para um broker de mensagens (e.g., Kafka, Redis Pub/Sub)
        logger.debug(f"ControlBus: Publicado evento '{event}' com payload: {payload['cycle_id']}")

class PCVSMock:
    """Mock do Persistent Control and Vector Store (PCVS) para persistência de estado."""
    async def store_module_state(self, module_name: str, state: Dict[str, Any]):
        """Simula o armazenamento do estado serializado do módulo no PCVS."""
        # Em um sistema real, isto seria um I/O para um DB ou disco
        logger.debug(f"PCVS: Estado de '{module_name}' serializado e armazenado. Tamanho do histórico: {len(state.get('history', []))}")

class VectorClientMock:
    """Mock do cliente de armazenamento vetorial (e.g., FAISS) para busca de centróides."""
    def __init__(self, vector_dim: int = 768):
        self.vector_dim = vector_dim

    async def query_topk(self, vector: np.ndarray, k: int = 5) -> Optional[List[Dict[str, Any]]]:
        """
        Simula a consulta Top-K.
        Retorna K vetores próximos (embeddings) de memória para calcular o centróide esperado.
        """
        # Simula o retorno de K vetores aleatórios (centróides próximos)
        if np.all(vector == 0): # Simula falha/fallback se o vetor de entrada for nulo
            return None
        
        # Gera K vetores simulados (centróides)
        embeddings = [
            (np.random.rand(self.vector_dim).astype(np.float32) - 0.5) * 2  # Vetor aleatório normalizado
            for _ in range(k)
        ]
        
        # Simula um resultado Top-K, onde cada item tem um campo 'embedding'
        return [{"embedding": emb.tolist()} for emb in embeddings]
        
class KGMock:
    """Mock do Knowledge Graph (KG) para inferência de vetores e tradução simbólica."""
    async def predict_embedding(self, vector: np.ndarray) -> Optional[List[float]]:
        """
        Simula a inferência de um vetor esperado com base em regras do KG.
        Em sistemas reais, seria usado para prever vetores ausentes.
        """
        if np.linalg.norm(vector) < 0.1: # Simula falha do KG
            return None
        
        # Retorna um vetor esperado ligeiramente diferente do input
        expected = vector * 0.95 + np.random.rand(vector.shape[0]).astype(np.float32) * 0.05
        return expected.tolist()

    async def vector_to_tripla(self, vector: np.ndarray, context: List[str]) -> Optional[Dict[str, Any]]:
        """
        Simula a conversão de um vetor de estado em uma Tripla Lógica (S, P, O).
        """
        # Simula a tradução em uma tripla lógica e um score de confiança
        return {
            "s": "Agent",
            "p": "Is_Coherent",
            "o": "With_Intent",
            "certainty": float(np.clip(1.0 - np.linalg.norm(vector - np.mean(vector)) * 0.1, 0.5, 0.99))
        }

class RegVetMock:
    """Mock do RegVet (Regulatory Vector) para penalidade de coerção."""
    async def get_penalty(self, cycle_id: str, context: List[str]) -> float:
        """Simula o retorno da penalidade imposta pelo RegVet (0.0 a 1.0)."""
        # Simula uma penalidade baseada no ID do ciclo (para teste)
        if cycle_id.endswith("CRITICAL_REGVET"):
            return 0.8
        return 0.1

class OEAMock:
    """Mock do OEA (Observador Ético e Avaliador) para score de saúde ética."""
    async def get_ethics_score(self, cycle_id: str, context: List[str]) -> float:
        """Simula o retorno do score de saúde ética (0.0 a 1.0, 1.0 é ótimo)."""
        # Simula um score de saúde baseado no ID do ciclo (para teste)
        if cycle_id.endswith("CRITICAL_OEA"):
            return 0.15
        return 0.95


# ==============================================================================
# CLASSE PRINCIPAL: SIMLOG
# ==============================================================================

class SimLog:
    """
    SimLog — Protocolo de Simbolização, Divergência e Governança Cognitiva.

    Orquestra o ciclo de auditoria, calculando a divergência vetorial (D),
    a saúde cognitiva (H) e integrando métricas de governança (RegVet, OEA).
    """

    def __init__(
        self,
        vector_client: Any,
        kg: Any,
        regvet_client: Optional[Any] = None,
        oea_client: Optional[Any] = None,
        pcvs: Optional[Any] = None,
        control_bus: Optional[Any] = None,
        *,
        max_history_size: int = 500,
        divergence_threshold: float = 0.90,
        health_threshold: float = 0.30,
        vector_dim: int = 768
    ):
        """
        Inicializa o SimLog com suas dependências e parâmetros de controle.

        Args:
            vector_client: Cliente para busca vetorial (Top-K centróide).
            kg: Cliente do Knowledge Graph para inferência e tradução simbólica.
            regvet_client: Cliente do Regulatory Vector (RegVet) para penalidades.
            oea_client: Cliente do Observador Ético (OEA) para score de saúde ética.
            pcvs: Cliente PCVS para persistência de estado.
            control_bus: Cliente ControlBus para publicação de eventos.
            max_history_size (int): Tamanho máximo do buffer de histórico de ciclos.
            divergence_threshold (float): Limiar de divergência (D) para eventos críticos.
            health_threshold (float): Limiar de saúde (H) para eventos críticos.
            vector_dim (int): Dimensionalidade dos vetores, usado para fallbacks.
        """
        self.vector_client = vector_client
        self.kg = kg
        self.regvet_client = regvet_client
        self.oea_client = oea_client
        self.pcvs = pcvs
        self.control_bus = control_bus

        self.max_history_size = max_history_size
        self.divergence_threshold = divergence_threshold
        self.health_threshold = health_threshold
        self.vector_dim = vector_dim

        self.history: List[Dict[str, Any]] = []

    # --------------------------------------------------------------------------
    # NORMALIZAÇÃO
    # --------------------------------------------------------------------------
    def _normalize(self, v: np.ndarray) -> np.ndarray:
        """
        Aplica a normalização L2 (norma unitária) a um vetor.
        Essencial para o cálculo de Divergência (D).
        """
        # Coerção defensiva para float32 para garantir consistência e performance
        v_f32 = v.astype(np.float32)
        
        n = np.linalg.norm(v_f32)
        
        # Previne divisão por zero: se a norma for muito próxima de zero, retorna o vetor original (zeros)
        if n < 1e-12:
            return v_f32
        
        return v_f32 / n

    # --------------------------------------------------------------------------
    # EXPECTED VECTOR
    # --------------------------------------------------------------------------
    async def _compute_expected_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        Calcula o expected_vector (vetor esperado/racional) com base na ordem de prioridade.
        
        Ordem de prioridade:
        1. vector_client.query_topk → Centróide dos vetores vizinhos (contexto)
        2. knowledge_graph.predict_embedding → Vetor inferido por regras simbólicas
        3. fallback → Vetor de zeros (último recurso)
        """
        # 1. TOP-K centróide
        try:
            topk = await self.vector_client.query_topk(vector, k=5)
            if topk and isinstance(topk, list):
                # Extrai os embeddings e converte para array numpy (garantindo float32)
                embeddings = [np.array(item["embedding"], dtype=np.float32) for item in topk if "embedding" in item]
                
                if embeddings:
                    # Calcula o centróide (média) dos vetores encontrados
                    centroid = np.mean(embeddings, axis=0)
                    return centroid.astype(np.float32)
        except Exception as e:
            logger.error(f"ExpectedVector via top-k falhou: {e}")

        # 2. Fallback KG
        try:
            emb = await self.kg.predict_embedding(vector)
            if emb is not None:
                # Retorna o vetor inferido pelo KG (garantindo float32)
                return np.array(emb, dtype=np.float32)
        except Exception as e:
            logger.error(f"ExpectedVector via KG falhou: {e}")

        # 3. Fallback final (vetor de zeros)
        # Este é o pior cenário, indicando falta total de conhecimento ou contexto.
        logger.warning("ExpectedVector fallback: Retornando vetor de zeros.")
        return np.zeros(self.vector_dim, dtype=np.float32)

    # --------------------------------------------------------------------------
    # TRADUÇÃO VETOR → TRIPLA
    # --------------------------------------------------------------------------
    async def _translate_vector_to_tripla(self, vector: np.ndarray, context: List[str]) -> Optional[Dict[str, Any]]:
        """Converte o vetor de estado em sua representação simbólica (Tripla Lógica S-P-O)."""
        try:
            return await self.kg.vector_to_tripla(vector.astype(np.float32), context)
        except Exception as e:
            logger.error(f"Erro na tradução vetor→tripla (KG.vector_to_tripla): {e}")
            return None

    # --------------------------------------------------------------------------
    # PUBLICAÇÃO ASSÍNCRONA NO CONTROLBUS
    # --------------------------------------------------------------------------
    async def _publish_event_async(self, event: str, payload: Dict[str, Any]):
        """Publica um evento estruturado no ControlBus, tratando exceções de forma segura."""
        if not self.control_bus:
            return
        try:
            # Não fazemos "await" para garantir que a publicação não bloqueie o ciclo,
            # mas usamos asyncio.create_task para garantir que a corrotina seja executada.
            # No mock, o await é seguro. Em um ambiente real com ControlBus de rede, isto seria um task
            await self.control_bus.publish(event, payload)
        except Exception as e:
            logger.error(f"Erro ao publicar evento '{event}' no ControlBus: {e}")

    # --------------------------------------------------------------------------
    # CICLO PRINCIPAL
    # --------------------------------------------------------------------------
    async def process_cycle(
        self,
        vector: np.ndarray,
        context_memories: Optional[List[str]] = None,
        cycle_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executa um ciclo completo de SimLog: calcula D e H, integra governança e persiste o estado.

        Args:
            vector (np.ndarray): O vetor de estado/ação atual do sistema (float32).
            context_memories (Optional[List[str]]): Memórias contextuais para RegVet/OEA.
            cycle_id (Optional[str]): ID do ciclo, gerado se não fornecido.

        Returns:
            Dict[str, Any]: Dados de auditoria do ciclo processado.
        """
        cycle_id = cycle_id or str(uuid4())
        context_memories = context_memories or []
        start = time.time()

        # 0. Coerção inicial para dtype padrão do sistema
        vector = vector.astype(np.float32)

        # 1. expected_vector real
        expected_vector = await self._compute_expected_vector(vector)

        # 2. Normalização para cálculo de Divergência (D)
        v_norm = self._normalize(vector)
        exp_norm = self._normalize(expected_vector)

        # Cálculo da Divergência (D) - Distância Euclidiana entre vetores normalizados
        # D < 1.0 (para vetores totalmente alinhados), D > 1.0 (para vetores em direções opostas)
        D = float(np.linalg.norm(v_norm - exp_norm))
        # Saúde Cognitiva (H) - H = max(0, 1 - D) é uma simplificação para alinhamento
        H = float(max(0.0, 1.0 - D))

        # 3. Penalidade RegVet (Coerção)
        reg_penalty = 0.0
        if self.regvet_client:
            try:
                # Solicita a penalidade imposta pelo RegVet (0.0 = sem penalidade, 1.0 = penalidade máxima)
                value = await self.regvet_client.get_penalty(
                    cycle_id=cycle_id,
                    context=context_memories
                )
                reg_penalty = float(value or 0.0)
            except Exception as e:
                logger.error(f"Erro RegVet na integração: {e}")

        # 4. Score ético do OEA
        oea_health = 1.0
        if self.oea_client:
            try:
                # Solicita o score de saúde ética (1.0 = ótimo, 0.0 = crítico)
                value = await self.oea_client.get_ethics_score(
                    cycle_id=cycle_id,
                    context=context_memories
                )
                oea_health = float(value or 1.0)
            except Exception as e:
                logger.error(f"Erro OEA na integração: {e}")

        # 5. Tripla simbólica (tradução lógica)
        tripla = await self._translate_vector_to_tripla(vector, context_memories)

        latency_ms = int((time.time() - start) * 1000)

        cycle_data = {
            "cycle_id": cycle_id,
            "timestamp": time.time(),
            "vector": vector.tolist(),
            "expected_vector": expected_vector.tolist(),
            "D": D, # Divergência
            "H": H, # Saúde (Divergência)
            "regvet_penalty": reg_penalty, # Penalidade de coerção (inverso de H)
            "oea_health": oea_health, # Saúde ética
            "tripla": tripla, # Tripla lógica (S, P, O)
            "latency_ms": latency_ms,
        }

        # ----------------------------------------------------------------------
        # Gerenciamento de Histórico (LRU)
        # ----------------------------------------------------------------------
        self.history.append(cycle_data)
        if len(self.history) > self.max_history_size:
            self.history.pop(0)

        # ----------------------------------------------------------------------
        # Persistência PCVS (Estado do Módulo)
        # ----------------------------------------------------------------------
        if self.pcvs:
            try:
                await self.pcvs.store_module_state("SimLog", self.serialize_state())
            except Exception as e:
                logger.error(f"Erro PCVS na persistência: {e}")

        # ----------------------------------------------------------------------
        # Emissão de Eventos de Auditoria e Críticos
        # ----------------------------------------------------------------------
        await self._publish_event_async("SIMLOG_CYCLE", cycle_data)

        if D > self.divergence_threshold:
            # Divergência Alta: A ação do sistema se desvia muito do que é esperado.
            await self._publish_event_async("SIMLOG_DIVERGENCE_HIGH", cycle_data)
            logger.warning(f"Divergência Alta detectada: D={D:.4f} > {self.divergence_threshold}")

        if H < self.health_threshold:
            # Saúde Baixa: A métrica de saúde interna (baseada em D) está abaixo do limiar.
            await self._publish_event_async("SIMLOG_HEALTH_LOW", cycle_data)
            logger.critical(f"Saúde Cognitiva Baixa detectada: H={H:.4f} < {self.health_threshold}")

        # ----------------------------------------------------------------------
        # Log estruturado (JSON) para monitoramento
        # ----------------------------------------------------------------------
        # Envia um log de auditoria limpo e estruturado para o sistema de monitoramento central
        logger.info(json.dumps({
            "event": "simlog_cycle",
            "cycle_id": cycle_id,
            "D": D,
            "H": H,
            "regvet": reg_penalty,
            "oea": oea_health,
            "latency_ms": latency_ms
        }))

        return cycle_data

    # --------------------------------------------------------------------------
    # SERIALIZAÇÃO
    # --------------------------------------------------------------------------
    def serialize_state(self) -> Dict[str, Any]:
        """Retorna o estado interno do SimLog para persistência no PCVS."""
        return {
            "history": self.history[-100:],  # Persiste apenas os últimos 100 itens para segurança/eficiência
            "max_history_size": self.max_history_size,
            "divergence_threshold": self.divergence_threshold,
            "health_threshold": self.health_threshold,
            "vector_dim": self.vector_dim,
        }
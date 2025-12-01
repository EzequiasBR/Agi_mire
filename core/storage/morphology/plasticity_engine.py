# core/storage/morphology/plasticity_engine.py
import numpy as np
import time
from typing import List
from datetime import datetime, timedelta

# Importa a Camada 2 para manipular o índice
from core.storage.bridge.vector_index import VectorIndex 
# Importa o IndexPointer, pois o peso está no VectorID
from core.storage.index_pointer import IndexPointer 

class PlasticityEngine:
    """
    Motor que calcula e aplica o Decaimento Temporal e o Reforço/Correção (Delta W).
    Funciona como a Camada 5 (Morphology).
    """
    def __init__(self, vector_index: VectorIndex, pointer_index: IndexPointer, config: dict):
        self.vector_index = vector_index
        self.pointer_index = pointer_index
        # Taxa de Decaimento Lambda (λ) - Variável, ajustada pelo PPO
        self.base_lambda = config.get('base_lambda', 0.001) 
        # Fator de Correção Mínimo
        self.min_reinforce_factor = config.get('min_reinforce_factor', 0.1)

    # ------------------
    # Ação 1: Decaimento Temporal (Esquecimento)
    # ------------------

    def calculate_decay_factor(self, timestamp: float, current_lambda: float = None) -> float:
        """
        Calcula o fator de decaimento de um peso baseado no tempo e em Lambda (λ).
        Fórmula: peso_t = peso_0 * exp(-λ * Δt)
        """
        if current_lambda is None:
            current_lambda = self.base_lambda
            
        time_elapsed = time.time() - timestamp # Tempo decorrido em segundos
        
        # O fator de decaimento (e^(-λ * Δt)) deve ser um valor entre 0 e 1.
        decay_factor = np.exp(-current_lambda * time_elapsed)
        return float(decay_factor)

    def apply_decay_to_index(self, vector_id: int):
        """
        Simula o ajuste de peso no FAISS sem re-indexar o vetor.
        Nota: Em um sistema FAISS real, isso é complexo. Em nosso Blueprint,
        o peso pode ser embutido como uma dimensão extra ou o VectorID
        pode ser removido e re-adicionado com um peso novo/reduzido,
        ou usado como um fator de filtro pós-busca.
        """
        # Para o nosso Blueprint: A lógica de decay é usada pelo Hipocampo
        # para decidir se o vetor é "apagado" (descartado) ou mantido.
        
        # A implementação real aqui pode ser apenas a sinalização para o Hipocampo.
        # Ex: Sinaliza o Hippocampus que o vetor está 'stale' (obsoleto).
        pass

    # ------------------
    # Ação 2: Reforço e Correção (Aprendizado)
    # ------------------

    def apply_reinforcement(self, vector_id: int, correction_vector: np.ndarray, intensity_desvio: float):
        """
        Aplica reforço ou correção a um vetor existente.
        O 'correction_vector' é o Vetor de Desvio (HVD) ou Vetor de Sucesso.
        A 'intensity_desvio' é a I_desvio do OEA/Reg-Vet.
        """
        # 1. Recuperar o vetor original (Simulação: lendo a memória original)
        # Em produção, o Hipocampo enviaria o vetor original (V_old).
        # Simulamos uma recuperação:
        V_old_dummy = np.random.rand(self.vector_index.vector_dim) 
        
        # 2. Calcular o fator de reforço/correção (Δw)
        # O fator de correção é uma mistura ponderada do vetor original (V_old)
        # e o vetor de correção (correction_vector), modulado pela intensidade.
        
        # intensity_desvio: Alta intensidade significa que a correção deve ser forte.
        
        # O novo vetor (V_new) é uma combinação que move V_old na direção de V_correction
        # V_new = V_old + (V_correction - V_old) * (intensity_desvio * factor)
        
        # Simplificação: Novo vetor é a média ponderada
        weight = np.clip(intensity_desvio, self.min_reinforce_factor, 1.0)
        V_new = (1 - weight) * V_old_dummy + weight * correction_vector
        
        # 3. Normalizar e Re-indexar
        # Crucial: Vetores devem ser L2 Normalizados para busca IP (Produto Interno)
        V_new = V_new / np.linalg.norm(V_new)
        
        # 4. Atualização no Índice FAISS (Morfologia)
        # Na maioria das estruturas FAISS, a forma mais segura de atualizar é:
        # a) Remover o VectorID antigo.
        # b) Adicionar o novo vetor V_new (com o mesmo VectorID)
        #    OU
        # c) Adicionar o novo vetor V_new (com um novo VectorID) e sinalizar o antigo como obsoleto no IndexPointer.
        
        # Estratégia do Blueprint: Adicionar a versão corrigida como uma *nova* memória
        # e sinalizar a versão antiga no AOL/IndexPointer para auditoria, mantendo a trilha.
        # Isso garante que a trilha de auditoria PRAG (PRAG V1.0) possa rastrear a correção.
        
        # O MCH/Hippocampus fará a inserção. A PlasticityEngine fornece o novo vetor.
        return V_new
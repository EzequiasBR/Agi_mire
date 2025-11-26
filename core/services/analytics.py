# core/services/analytics.py
"""
Analytics Service V3 - Cálculo de Métricas Sistêmicas

Funções:
- compute_system_health(C_med, D_var) -> H_sist
- compute_volatility(C_var, rollback_rate) -> V_sist
- compute_system_energy(LO_rate, E_primal) -> E_sist

"""
from __future__ import annotations
import logging
import time
from typing import Any, Dict, Optional, List, Tuple
import numpy as np

try:
    from .utils import setup_logger
except ImportError:
    # Minimal Fallback
    def setup_logger(name):
        l = logging.getLogger(name)
        if not l.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s"))
            l.addHandler(ch)
        l.setLevel(logging.INFO)
        return l

logger = setup_logger("AnalyticsService")

class Analytics:
    """
    Calcula as métricas sistêmicas H (Saúde), V (Volatilidade) e E (Energia)
    baseado em telemetria do sistema (Confiança, Divergência, Taxas de Ação).
    """

    def __init__(self):
        # Janela de análise móvel para o cálculo de métricas
        self.window_size: int = 100
        self.confidence_history: List[float] = []
        self.divergence_history: List[float] = []
        self.lo_trigger_history: List[int] = [] # 1=triggered, 0=skipped
        self.rollback_history: List[int] = [] # 1=triggered, 0=skipped
        self.last_metrics: Dict[str, float] = {"H": 0.5, "V": 0.5, "E": 0.5}

        logger.info("Analytics Service initialized.")

    # ---------------------------------------------------------
    # Funções de Cálculo Principal (Normalizadas 0 a 1)
    # ---------------------------------------------------------

    def compute_metrics(self, new_C: float, new_D: float, rollback_triggered: bool, lo_triggered: bool) -> Dict[str, float]:
        """
        Atualiza o histórico e calcula as novas métricas sistêmicas.
        """
        # Atualiza Histórico
        self.confidence_history.append(new_C)
        self.divergence_history.append(new_D)
        self.rollback_history.append(1 if rollback_triggered else 0)
        self.lo_trigger_history.append(1 if lo_triggered else 0)

        # Trunca janelas móveis
        self.confidence_history = self.confidence_history[-self.window_size:]
        self.divergence_history = self.divergence_history[-self.window_size:]
        self.rollback_history = self.rollback_history[-self.window_size:]
        self.lo_trigger_history = self.lo_trigger_history[-self.window_size:]

        # Estatísticas Básicas da Janela
        C_med = np.mean(self.confidence_history) if self.confidence_history else 0.5
        D_med = np.mean(self.divergence_history) if self.divergence_history else 0.5
        C_var = np.var(self.confidence_history) if len(self.confidence_history) > 1 else 0.0
        
        rollback_rate = np.mean(self.rollback_history) # Taxa de rollbacks
        lo_rate = np.mean(self.lo_trigger_history)     # Taxa de otimizações

        # Cálculo de H, V, E
        H_sist = self._compute_system_health(C_med, D_med, rollback_rate)
        V_sist = self._compute_volatility(C_var, rollback_rate)
        E_sist = self._compute_system_energy(lo_rate, C_med)

        self.last_metrics = {"H": H_sist, "V": V_sist, "E": E_sist}
        
        logger.debug("Metrics computed: H=%.3f, V=%.3f, E=%.3f", H_sist, V_sist, E_sist)
        return self.last_metrics

    def _compute_system_health(self, C_med: float, D_med: float, rollback_rate: float) -> float:
        """
        Saúde Sistêmica (H): Alta se a Confiança Média for alta, a Divergência Média for baixa
        e a Taxa de Rollback for baixa.
        H ≈ 0.6 * C_med + 0.3 * (1 - D_med) + 0.1 * (1 - rollback_rate)
        """
        # (1 - D_med) é a Coerência Média
        # (1 - rollback_rate) é a Estabilidade Operacional
        health = 0.6 * C_med + 0.3 * (1.0 - D_med) + 0.1 * (1.0 - rollback_rate)
        return np.clip(health, 0.0, 1.0)

    def _compute_volatility(self, C_var: float, rollback_rate: float) -> float:
        """
        Volatilidade Sistêmica (V): Alta se a Confiança tiver alta variância (C_var)
        ou se a Taxa de Rollback for alta (instabilidade).
        V ≈ C_var * 5 + rollback_rate * 0.5 (normalizado)
        """
        # C_var (variância da confiança) é tipicamente pequeno, por isso o fator 5
        volatility = np.clip(C_var * 5.0 + rollback_rate * 0.5, 0.0, 1.0)
        return np.clip(volatility, 0.0, 1.0)

    def _compute_system_energy(self, lo_rate: float, C_med: float) -> float:
        """
        Energia Sistêmica (E): Alta se a taxa de Otimização (LO_rate) for alta OU
        se a Confiança Média for muito baixa (necessidade urgente de adaptação).
        E ≈ lo_rate * 0.7 + (1 - C_med) * 0.3
        """
        # (1 - C_med) é a necessidade de aprendizado (Exploração)
        energy = lo_rate * 0.7 + (1.0 - C_med) * 0.3
        return np.clip(energy, 0.0, 1.0)
    
    # ---------------------------------------------------------
    # API e Serialização
    # ---------------------------------------------------------

    def get_metrics(self) -> Dict[str, float]:
        """Retorna as últimas métricas calculadas."""
        return self.last_metrics

    def snapshot_state(self) -> Dict[str, Any]:
        """Retorna snapshot do estado para PCVS."""
        return {
            "window_size": self.window_size,
            "confidence_history": self.confidence_history,
            "divergence_history": self.divergence_history,
            "lo_trigger_history": self.lo_trigger_history,
            "rollback_history": self.rollback_history,
            "last_metrics": self.last_metrics
        }
    
# ---------------------------------------------------------
# Teste Rápido
# ---------------------------------------------------------
if __name__ == "__main__":
    analytics = Analytics()
    
    # Simulação de 100 ciclos estáveis (C alto, D baixo, sem ações)
    for i in range(50):
        metrics = analytics.compute_metrics(new_C=0.95 + 0.01 * np.random.randn(), 
                                            new_D=0.05 + 0.01 * np.random.randn(), 
                                            rollback_triggered=False, 
                                            lo_triggered=False)
    print("--- Cenário 1: Estabilidade Total ---")
    print(f"H (Saúde) esperada alta: {metrics['H']:.3f}")
    print(f"V (Volatilidade) esperada baixa: {metrics['V']:.3f}")
    print(f"E (Energia) esperada baixa: {metrics['E']:.3f}")
    
    # Simulação de 10 ciclos de crise (C baixo, D alto, com rollback e LO forçado)
    for i in range(10):
        metrics = analytics.compute_metrics(new_C=0.40, 
                                            new_D=0.60, 
                                            rollback_triggered=(i % 3 == 0), 
                                            lo_triggered=True)
        
    print("\n--- Cenário 2: Crise/Caos ---")
    print(f"H (Saúde) esperada baixa: {metrics['H']:.3f}")
    print(f"V (Volatilidade) esperada alta: {metrics['V']:.3f}")
    print(f"E (Energia) esperada alta: {metrics['E']:.3f}")
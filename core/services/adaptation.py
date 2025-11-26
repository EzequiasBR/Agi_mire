# core/services/adaptation.py
"""
Adaptation Service V3
Serviço de ajuste paramétrico autônomo baseado em métricas de saúde e volatilidade.

Funções:
- update_parameters(H_sist, V_sist)
- get_ppo_tau()
- get_prag_rollback_threshold()
- get_hippocampus_lambda()
"""

from __future__ import annotations
import logging
import time
from typing import Any, Dict

# Reutilizar logger
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

logger = setup_logger("AdaptationService")

# Parâmetros Iniciais (Defaults)
DEFAULT_TAU_PPO = 0.90       # PPO: Limiar inicial de otimização (tau)
DEFAULT_DELTA_H_PRAG = 0.05  # PRAG: Limiar inicial de rollback (Delta H)
DEFAULT_LAMBDA_HIPPO = 1e-4  # HIPPO: Fator inicial de decaimento (lambda)

# Constantes de Sensibilidade (Hyperparameters)
K_PPO = 0.1 # Sensibilidade do PPO ao V (Volatilidade)
K_PRAG = 0.05 # Sensibilidade do PRAG ao H (Saúde)
K_HIPPO = 1.0 # Sensibilidade do Hippocampus ao H (Saúde)

# Limites de Clipping (para garantir estabilidade)
TAU_PPO_MIN_MAX = (0.70, 0.99)
DELTA_H_PRAG_MIN_MAX = (0.01, 0.20)
LAMBDA_HIPPO_MIN_MAX = (1e-5, 1e-3)


class Adaptation:
    """
    Gerencia a adaptação autônoma dos parâmetros de controle do sistema.
    As métricas H (Saúde) e V (Volatilidade) devem ser normalizadas entre 0 e 1.
    """

    def __init__(self):
        # Parâmetros Adaptáveis (Valores atuais)
        self._tau_ppo: float = DEFAULT_TAU_PPO
        self._delta_h_prag: float = DEFAULT_DELTA_H_PRAG
        self._lambda_hippo: float = DEFAULT_LAMBDA_HIPPO
        
        # Telemetria
        self.last_update_ts: float = 0.0
        self.telemetry: Dict[str, Any] = {}

        logger.info("Adaptation Service initialized with defaults.")

    # ---------------------------------------------------------
    # Função Principal de Adaptação
    # ---------------------------------------------------------

    def update_parameters(self, H_sist: float, V_sist: float) -> None:
        """
        Calcula e atualiza todos os parâmetros de controle com base nas métricas
        sistêmicas (H = Saúde, V = Volatilidade), normalizadas entre 0 e 1.
        
        :param H_sist: Saúde Sistêmica (0=Péssimo, 1=Ótimo).
        :param V_sist: Volatilidade Sistêmica (0=Estável, 1=Caótico).
        """
        H = float(H_sist)
        V = float(V_sist)
        
        # 1. Ajuste do PPO Tau (τ) - Limiar de Otimização
        # Fator de cautela: τ deve subir (mais cautela) se V for alto (caos).
        # Fórmula: τ_novo = clip( DEFAULT_TAU + K_PPO * V )
        new_tau_ppo = DEFAULT_TAU_PPO + K_PPO * V
        new_tau_ppo = min(max(new_tau_ppo, TAU_PPO_MIN_MAX[0]), TAU_PPO_MIN_MAX[1])
        
        # 2. Ajuste do PRAG Delta H (ΔH) - Limiar de Rollback
        # Fator de sensibilidade: ΔH deve descer (mais sensível) se H for baixo (saúde ruim).
        # Fórmula: ΔH_novo = clip( DEFAULT_DELTA_H * (1 - K_PRAG * (1 - H)) )
        # Aumentamos o fator (1 - H) para que a saúde ruim tenha um efeito maior
        new_delta_h_prag = DEFAULT_DELTA_H_PRAG * (1.0 - K_PRAG * (1.0 - H))
        new_delta_h_prag = min(max(new_delta_h_prag, DELTA_H_PRAG_MIN_MAX[0]), DELTA_H_PRAG_MIN_MAX[1])
        
        # 3. Ajuste do Hippocampus Lambda (λ) - Fator de Decaimento (Esquecimento)
        # Fator de foco: λ deve subir (esquecer mais rápido) se H for alto (saúde boa, foco na novidade).
        # Fórmula: λ_novo = clip( DEFAULT_LAMBDA * (1 + K_HIPPO * H) )
        new_lambda_hippo = DEFAULT_LAMBDA_HIPPO * (1.0 + K_HIPPO * H)
        new_lambda_hippo = min(max(new_lambda_hippo, LAMBDA_HIPPO_MIN_MAX[0]), LAMBDA_HIPPO_MIN_MAX[1])
        
        # Aplica a atualização
        self._update(new_tau_ppo, new_delta_h_prag, new_lambda_hippo, H, V)

    def _update(self, tau_ppo: float, delta_h_prag: float, lambda_hippo: float, H: float, V: float) -> None:
        """Aplica os novos valores e registra a telemetria."""
        
        # Registro antes da atualização
        old_tau, old_delta, old_lambda = self._tau_ppo, self._delta_h_prag, self._lambda_hippo

        self._tau_ppo = tau_ppo
        self._delta_h_prag = delta_h_prag
        self._lambda_hippo = lambda_hippo
        self.last_update_ts = time.time()

        # Telemetria
        log_data = {
            "ts": self.last_update_ts,
            "H_sist": H,
            "V_sist": V,
            "tau_ppo_old": old_tau,
            "tau_ppo_new": tau_ppo,
            "delta_h_prag_old": old_delta,
            "delta_h_prag_new": delta_h_prag,
            "lambda_hippo_old": old_lambda,
            "lambda_hippo_new": lambda_hippo,
        }
        self.telemetry = log_data

        logger.info("Adaptation applied: τ_ppo=%.4f, ΔH_prag=%.4f, λ_hippo=%.6f", 
                    self._tau_ppo, self._delta_h_prag, self._lambda_hippo)
        
    # ---------------------------------------------------------
    # API para Módulos (Getters)
    # ---------------------------------------------------------

    def get_ppo_tau(self) -> float:
        """Retorna o limiar de otimização (τ) para o PPO."""
        return self._tau_ppo

    def get_prag_rollback_threshold(self) -> float:
        """Retorna o limiar de rollback (ΔH) para o PRAG."""
        return self._delta_h_prag

    def get_hippocampus_lambda(self) -> float:
        """Retorna o fator de decaimento (λ) para o Hippocampus."""
        return self._lambda_hippo

    # ---------------------------------------------------------
    # Snapshot PCVS
    # ---------------------------------------------------------
    
    def snapshot_state(self) -> Dict[str, Any]:
        """Retorna um snapshot serializável para PCVS."""
        return {
            "tau_ppo": self._tau_ppo,
            "delta_h_prag": self._delta_h_prag,
            "lambda_hippo": self._lambda_hippo,
            "last_update_ts": self.last_update_ts,
            "telemetry": self.telemetry 
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Carrega estado a partir de um snapshot PCVS."""
        if not state:
            return
            
        self._tau_ppo = float(state.get("tau_ppo", DEFAULT_TAU_PPO))
        self._delta_h_prag = float(state.get("delta_h_prag", DEFAULT_DELTA_H_PRAG))
        self._lambda_hippo = float(state.get("lambda_hippo", DEFAULT_LAMBDA_HIPPO))
        self.last_update_ts = float(state.get("last_update_ts", 0.0))
        self.telemetry = state.get("telemetry", {})
        
        logger.info("Adaptation state loaded: τ_ppo=%.4f, ΔH_prag=%.4f, λ_hippo=%.6f", 
                    self._tau_ppo, self._delta_h_prag, self._lambda_hippo)

# ---------------------------------------------------------
# Teste Rápido
# ---------------------------------------------------------
if __name__ == "__main__":
    
    adapter = Adaptation()
    
    print("--- 1. Estado Inicial ---")
    print(f"PPO τ: {adapter.get_ppo_tau():.4f}")
    print(f"PRAG ΔH: {adapter.get_prag_rollback_threshold():.4f}")
    print(f"HIPPO λ: {adapter.get_hippocampus_lambda():.6f}")

    # Cenário A: SAÚDE EXCELENTE (H=1.0) e VOLATILIDADE BAIXA (V=0.1)
    # Resultado esperado: Otimização mais rápida (τ baixa), Rollback menos sensível (ΔH alto), Esquecimento mais rápido (λ alto).
    print("\n--- 2. Cenário A: H=1.0, V=0.1 (Estabilidade/Foco) ---")
    adapter.update_parameters(H_sist=1.0, V_sist=0.1)
    print(f"PPO τ (otimização): {adapter.get_ppo_tau():.4f} (Quase padrão, V baixo)")
    print(f"PRAG ΔH (rollback): {adapter.get_prag_rollback_threshold():.4f} (Alto, menos sensível)")
    print(f"HIPPO λ (decaimento): {adapter.get_hippocampus_lambda():.6f} (Alto, mais foco/esquecimento)")

    # Cenário B: SAÚDE RUIM (H=0.3) e VOLATILIDADE ALTA (V=0.9)
    # Resultado esperado: Otimização mais cautelosa (τ alto), Rollback muito sensível (ΔH baixo), Esquecimento lento (λ baixo).
    print("\n--- 3. Cenário B: H=0.3, V=0.9 (Caos/Sobrevivência) ---")
    adapter.update_parameters(H_sist=0.3, V_sist=0.9)
    print(f"PPO τ (otimização): {adapter.get_ppo_tau():.4f} (Alto, muita cautela)")
    print(f"PRAG ΔH (rollback): {adapter.get_prag_rollback_threshold():.4f} (Baixo, máxima sensibilidade)")
    print(f"HIPPO λ (decaimento): {adapter.get_hippocampus_lambda():.6f} (Baixo, reter memória)")
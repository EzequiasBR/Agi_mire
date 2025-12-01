from __future__ import annotations
import logging
import time
from typing import Any, Dict, Callable, Awaitable
import asyncio # Necessário para rodar o bloco de teste e para a natureza assíncrona
# import numpy as np # Importe numpy no seu ambiente para usar np.clip

# Reutilizar logger (código omitido para concisão, mas mantido na classe)
try:
    from .utils import setup_logger
except ImportError:
    def setup_logger(name):
        l = logging.getLogger(name)
        if not l.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s"))
            l.addHandler(ch)
        l.setLevel(logging.INFO)
        return l

logger = setup_logger("AdaptationService")

# --- Constantes do Sistema ---
# Eventos para o ControlBus (Mock)
class SystemEvents:
    PARAM_ADJUSTED = "PARAM_ADJUSTED"

# Parâmetros Iniciais (Defaults)
DEFAULT_TAU_PPO = 0.90       # PPO: Limiar inicial de otimização (tau)
DEFAULT_DELTA_H_PRAG = 0.05  # PRAG: Limiar inicial de rollback (Delta H)
DEFAULT_LAMBDA_HIPPO = 1e-4  # HIPPO: Fator inicial de decaimento (lambda)

# Constantes de Sensibilidade (Hyperparameters)
K_PPO = 0.1     # Sensibilidade do PPO ao V (Volatilidade)
K_PRAG = 0.05   # Sensibilidade do PRAG ao H (Saúde)
K_HIPPO = 1.0   # Sensibilidade do Hippocampus ao H (Saúde)

# Limites de Clipping (para garantir estabilidade)
TAU_PPO_MIN_MAX = (0.70, 0.99)
DELTA_H_PRAG_MIN_MAX = (0.01, 0.20)
LAMBDA_HIPPO_MIN_MAX = (1e-5, 1e-3)

# Função auxiliar para clipping sem numpy (assumindo que o usuário usará numpy)
def _clip(value, min_max):
    """Substitui np.clip para que o código seja executável sem numpy."""
    return max(min(value, min_max[1]), min_max[0])


class Adaptation:
    """
    Gerencia a adaptação autônoma dos parâmetros de controle do sistema.
    As métricas H (Saúde) e V (Volatilidade) devem ser normalizadas entre 0 e 1.
    """

    def __init__(self, control_bus: Any): # O ControlBus deve ser injetado
        self._control_bus = control_bus
        
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

    async def update_parameters(self, H_sist: float, V_sist: float) -> None:
        """
        Calcula e atualiza todos os parâmetros de controle com base nas métricas
        sistêmicas (H = Saúde, V = Volatilidade), normalizadas entre 0 e 1.
        
        :param H_sist: Saúde Sistêmica (0=Péssimo, 1=Ótimo).
        :param V_sist: Volatilidade Sistêmica (0=Estável, 1=Caótico).
        """
        H = float(H_sist)
        V = float(V_sist)
        
        # 1. Ajuste do PPO Tau (τ) - Limiar de Otimização
        new_tau_ppo = DEFAULT_TAU_PPO + K_PPO * V
        # Ajuste 1: np.clip para legibilidade
        # new_tau_ppo = np.clip(new_tau_ppo, *TAU_PPO_MIN_MAX)
        new_tau_ppo = _clip(new_tau_ppo, TAU_PPO_MIN_MAX) # Usando a função auxiliar

        
        # 2. Ajuste do PRAG Delta H (ΔH) - Limiar de Rollback
        new_delta_h_prag = DEFAULT_DELTA_H_PRAG * (1.0 - K_PRAG * (1.0 - H))
        # Ajuste 1: np.clip para legibilidade
        # new_delta_h_prag = np.clip(new_delta_h_prag, *DELTA_H_PRAG_MIN_MAX)
        new_delta_h_prag = _clip(new_delta_h_prag, DELTA_H_PRAG_MIN_MAX)

        
        # 3. Ajuste do Hippocampus Lambda (λ) - Fator de Decaimento (Esquecimento)
        new_lambda_hippo = DEFAULT_LAMBDA_HIPPO * (1.0 + K_HIPPO * H)
        # Ajuste 1: np.clip para legibilidade
        # new_lambda_hippo = np.clip(new_lambda_hippo, *LAMBDA_HIPPO_MIN_MAX)
        new_lambda_hippo = _clip(new_lambda_hippo, LAMBDA_HIPPO_MIN_MAX)

        
        # Aplica a atualização
        await self._update(new_tau_ppo, new_delta_h_prag, new_lambda_hippo, H, V)

    async def _update(self, tau_ppo: float, delta_h_prag: float, lambda_hippo: float, H: float, V: float) -> None:
        """Aplica os novos valores, registra a telemetria e publica o evento."""
        
        # --------------------------------------------------
        # Ajuste 2: Determinação do Cenário para Telemetria
        # --------------------------------------------------
        if H >= 0.8 and V <= 0.3:
            scenario = "Estabilidade/Foco"
        elif H <= 0.4 and V >= 0.7:
            scenario = "Caos/Sobrevivência"
        elif H <= 0.4 and V <= 0.3:
            scenario = "Estagnação/Risco"
        else:
            scenario = "Crescimento Volátil/Transição"
        
        # Registro antes da atualização
        old_tau, old_delta, old_lambda = self._tau_ppo, self._delta_h_prag, self._lambda_hippo

        self._tau_ppo = tau_ppo
        self._delta_h_prag = delta_h_prag
        self._lambda_hippo = lambda_hippo
        self.last_update_ts = time.time()

        # Telemetria enriquecida
        log_data = {
            "ts": self.last_update_ts,
            "scenario": scenario, # Ajuste 2: Cenário incluído
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

        # --------------------------------------------------
        # Ajuste 3: Publicação do Evento no ControlBus
        # --------------------------------------------------
        try:
            await self._control_bus.publish(
                SystemEvents.PARAM_ADJUSTED, 
                self.snapshot_state(), 
                source_module="AdaptationService"
            )
        except Exception as e:
            logger.error("Falha ao publicar PARAM_ADJUSTED no ControlBus: %s", e)

        logger.info("[%s] Adaptation applied: τ_ppo=%.4f, ΔH_prag=%.4f, λ_hippo=%.6f", 
                    scenario, self._tau_ppo, self._delta_h_prag, self._lambda_hippo)
        
    # ---------------------------------------------------------
    # API para Módulos (Getters, não alterados)
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
    # Snapshot PCVS (não alterado)
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
        # ... (implementação omitida, mas mantida) ...
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
# Teste Rápido (Ajustado para Assincronia)
# ---------------------------------------------------------
if __name__ == "__main__":
    
    # Mock do ControlBus
    class MockControlBus:
        def __init__(self):
            self.events_published = []
        async def publish(self, event: str, payload: dict[str, Any], source_module: str) -> None:
            log_payload = {k: v for k, v in payload.items() if k not in ['telemetry']}
            print(f"MockControlBus: Publicado {event} de {source_module} com payload chave: {log_payload.keys()}")
            self.events_published.append({"event": event, "payload": payload, "source": source_module})
        def subscribe(self, event: str, handler: Callable[[dict[str, Any]], Awaitable[None]]) -> None:
            pass # Ignorado no mock de teste

    async def main():
        control_bus_mock = MockControlBus()
        adapter = Adaptation(control_bus=control_bus_mock)
        
        print("\n--- 1. Estado Inicial ---")
        print(f"PPO τ: {adapter.get_ppo_tau():.4f}")
        print(f"PRAG ΔH: {adapter.get_prag_rollback_threshold():.4f}")
        print(f"HIPPO λ: {adapter.get_hippocampus_lambda():.6f}")

        # Cenário A: SAÚDE EXCELENTE (H=1.0) e VOLATILIDADE BAIXA (V=0.1)
        print("\n--- 2. Cenário A: H=1.0, V=0.1 (Estabilidade/Foco) ---")
        await adapter.update_parameters(H_sist=1.0, V_sist=0.1)
        print(f"PPO τ (otimização): {adapter.get_ppo_tau():.4f}")
        print(f"PRAG ΔH (rollback): {adapter.get_prag_rollback_threshold():.4f}")
        print(f"HIPPO λ (decaimento): {adapter.get_hippocampus_lambda():.6f}")
        print(f"Telemetry Scenario: {adapter.telemetry.get('scenario')}")

        # Cenário B: SAÚDE RUIM (H=0.3) e VOLATILIDADE ALTA (V=0.9)
        print("\n--- 3. Cenário B: H=0.3, V=0.9 (Caos/Sobrevivência) ---")
        await adapter.update_parameters(H_sist=0.3, V_sist=0.9)
        print(f"PPO τ (otimização): {adapter.get_ppo_tau():.4f}")
        print(f"PRAG ΔH (rollback): {adapter.get_prag_rollback_threshold():.4f}")
        print(f"HIPPO λ (decaimento): {adapter.get_hippocampus_lambda():.6f}")
        print(f"Telemetry Scenario: {adapter.telemetry.get('scenario')}")

        # Cenário C: SAÚDE BAIXA (H=0.3) e VOLATILIDADE BAIXA (V=0.1)
        print("\n--- 4. Cenário C: H=0.3, V=0.1 (Estagnação/Risco) ---")
        await adapter.update_parameters(H_sist=0.3, V_sist=0.1)
        print(f"Telemetry Scenario: {adapter.telemetry.get('scenario')}")

    asyncio.run(main())
"""
Security - Módulo de Segurança e Integridade de Dados.
Fornece utilitários para sanitização de entrada, verificação de integridade 
vetorial (hashing) e filtragem de saída para conformidade e robustez.
"""

from __future__ import annotations
import time
import logging
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from threading import Lock
import numpy as np
import hashlib
import json
import asyncio  # Necessário para rodar o bloco de teste

# Mocks de dependências (úteis em TYPE_CHECKING)
class SystemEvents:
    INTEGRITY_VIOLATION = "INTEGRITY_VIOLATION"
    CONFIG_ADJUSTED = "CONFIG_ADJUSTED"

if TYPE_CHECKING:
    class ControlBus:
        async def publish(self, event_type: str, payload: Dict[str, Any], source_module: str = "unknown") -> None: ...
        def subscribe(self, event: str, handler: Any) -> None: ...
# Fim do Mock de dependências

# Importar setup_logger da mesma pasta services
try:
    from .utils import setup_logger
except Exception:
    # Fallback seguro
    def setup_logger(name: str) -> logging.Logger:
        logger = logging.getLogger(name)
        if not logger.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s"))
            logger.addHandler(ch)
        logger.setLevel(logging.INFO)
        return logger

logger = setup_logger("SecurityService")


class Security:
    def __init__(
        self,
        control_bus: 'ControlBus' | None,
        pii_keywords: Optional[List[str]] = None,
        vector_hash_alg: str = 'sha256',
        max_input_length: int = 4096
    ):
        self._lock = Lock()
        self._control_bus = control_bus
        self.vector_hash_alg = vector_hash_alg
        self.max_input_length = int(max_input_length)

        # Palavras-chave PII (simulação)
        self.pii_keywords = [
            "cpf", "rg", "telefone", "endereço", "email",
            "conta bancária", "data de nascimento"
        ] if pii_keywords is None else pii_keywords

        # Contadores para telemetria
        self.rejection_counts = {
            "pii_rejections": 0,
            "length_rejections": 0
        }

        # Inscrição em CONFIG_ADJUSTED (melhor usar API de subscribe se disponível)
        try:
            if self._control_bus is not None:
                if hasattr(self._control_bus, "subscribe"):
                    # registra handler async
                    try:
                        self._control_bus.subscribe(SystemEvents.CONFIG_ADJUSTED, self._handle_config_update)
                    except Exception:
                        # fallback: se subscribe não existir ou falhar, tente acessar dicionário 'subscriptions'
                        subs = getattr(self._control_bus, "subscriptions", None)
                        if isinstance(subs, dict):
                            subs.setdefault(SystemEvents.CONFIG_ADJUSTED, []).append(self._handle_config_update)
                else:
                    # fallback para API minimalista que usa dict 'subscriptions'
                    subs = getattr(self._control_bus, "subscriptions", None)
                    if isinstance(subs, dict):
                        subs.setdefault(SystemEvents.CONFIG_ADJUSTED, []).append(self._handle_config_update)
            logger.info("SecurityService initialized and (if available) subscribed to CONFIG_ADJUSTED.")
        except Exception:
            logger.exception("SecurityService initialization: subscribe fallback failed.")

    # -----------------------
    # Handler de atualização de configuração
    # -----------------------
    async def _handle_config_update(self, payload: Dict[str, Any]):
        """
        Ajustes dinâmicos via ControlBus.
        Expects payload possibly containing 'max_input_length' and/or 'vector_hash_alg'.
        """
        try:
            with self._lock:
                if "max_input_length" in payload:
                    new_len = int(payload["max_input_length"])
                    if new_len > 0:
                        self.max_input_length = new_len
                        logger.warning("Config update: max_input_length set to %d.", new_len)

                if "vector_hash_alg" in payload:
                    new_alg = str(payload["vector_hash_alg"])
                    # Verifica se hashlib suporta o algoritmo pedido
                    try:
                        hashlib.new(new_alg)
                        self.vector_hash_alg = new_alg
                        logger.warning("Config update: vector_hash_alg set to %s.", new_alg)
                    except Exception:
                        logger.error("Config update: requested hash algorithm '%s' is not supported.", new_alg)
        except Exception:
            logger.exception("Failed to apply config update in SecurityService.")

    # --------------------------------
    # 1. Sanitização de Entrada
    # --------------------------------
    def sanitize_input(self, input_data: Any) -> Tuple[bool, str]:
        """
        Verifica a validade e segurança dos dados de entrada.
        Retorna (is_safe, sanitized_data_or_reason).
        """
        try:
            data_str = str(input_data).lower().strip()
        except Exception:
            return False, "Invalid input (cannot coerce to string)."

        with self._lock:
            # Verificação de comprimento
            if len(data_str) > self.max_input_length:
                self.rejection_counts["length_rejections"] += 1
                logger.warning(
                    "Input rejected: exceeds max length (%d). Count: %d",
                    self.max_input_length, self.rejection_counts["length_rejections"]
                )
                return False, "Input length exceeded."

            # Detecção PII (simples)
            for keyword in self.pii_keywords:
                if keyword in data_str:
                    self.rejection_counts["pii_rejections"] += 1
                    logger.warning(
                        "Input rejected: possible PII detected (keyword: %s). Count: %d",
                        keyword, self.rejection_counts["pii_rejections"]
                    )
                    return False, f"Input contains potential sensitive data: {keyword}."

        # Limpeza simples
        sanitized_data = data_str.replace('\n', ' ').replace('\r', '')
        return True, sanitized_data

    # --------------------------------
    # 2. Verificação de Integridade Vetorial
    # --------------------------------
    def hash_vector(self, vector: Optional[np.ndarray]) -> str:
        """
        Gera um hash (configurável) de um vetor NumPy normalizado.
        Retorna hex digest ou 'HASH_ERROR' em falha.
        """
        if vector is None:
            return ""

        try:
            arr = np.asarray(vector, dtype=np.float64)
            # use C-order bytes for determinismo
            vector_bytes = arr.tobytes(order='C')
            # compute hash
            hasher = hashlib.new(self.vector_hash_alg)
            hasher.update(vector_bytes)
            return hasher.hexdigest()
        except Exception as e:
            logger.error("Failed to hash vector with %s: %s", self.vector_hash_alg, str(e))
            return "HASH_ERROR"

    async def check_vector_integrity(self, vector: Optional[np.ndarray], expected_hash: str) -> bool:
        """
        Compara o hash e publica INTEGRITY_VIOLATION se falhar.
        Esta função é async para poder publicar eventos async no ControlBus.
        """
        if not expected_hash:
            logger.warning("Integrity check: expected_hash is empty.")
            return False

        current_hash = self.hash_vector(vector)
        is_valid = (current_hash == expected_hash)

        if not is_valid:
            logger.critical("Integrity check FAILED. Hash mismatch detected: %s != %s", current_hash[:16], expected_hash[:16])
            # publica se control_bus disponível e possuir publish
            try:
                if self._control_bus is not None and hasattr(self._control_bus, "publish"):
                    await self._control_bus.publish(
                        SystemEvents.INTEGRITY_VIOLATION,
                        {
                            "current_hash_preview": current_hash[:16],
                            "expected_hash_preview": expected_hash[:16],
                            "hash_alg": self.vector_hash_alg,
                            "timestamp": time.time()
                        },
                        source_module="SecurityService"
                    )
            except Exception:
                logger.exception("Failed to publish INTEGRITY_VIOLATION to ControlBus.")
        return is_valid

    # --------------------------------
    # 3. Filtragem de Saída
    # --------------------------------
    def filter_output(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove campos sensíveis ou internos do resultado final antes da exposição externa.
        """
        filtered_result = dict(result)  # shallow copy

        keys_to_remove = [
            "pcvs_hash",
            "event_key",
            "embedding",
            "recon_embedding",
            "telemetry_id"
        ]

        for key in keys_to_remove:
            filtered_result.pop(key, None)

        # sanitize nested 'sym_pkg.vector' if present
        sym_pkg = filtered_result.get("sym_pkg")
        if isinstance(sym_pkg, dict) and "vector" in sym_pkg:
            sym_pkg["vector"] = "VECTOR_HIDDEN"
            filtered_result["sym_pkg"] = sym_pkg

        return filtered_result

    # --------------------------------
    # Snapshot (Compatibilidade com PCVS)
    # --------------------------------
    def snapshot_state(self) -> Dict[str, Any]:
        """Retorna o estado completo para persistência PCVS."""
        with self._lock:
            return {
                "vector_hash_alg": self.vector_hash_alg,
                "max_input_length": self.max_input_length,
                "rejection_counts": dict(self.rejection_counts)
            }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Carrega o estado a partir de um snapshot PCVS."""
        if not state:
            return
        with self._lock:
            try:
                self.vector_hash_alg = state.get("vector_hash_alg", self.vector_hash_alg)
                self.max_input_length = int(state.get("max_input_length", self.max_input_length))
                self.rejection_counts = state.get("rejection_counts", dict(self.rejection_counts))
                logger.info("Security state loaded. Hash Algo: %s, Max Length: %d",
                            self.vector_hash_alg, self.max_input_length)
            except Exception:
                logger.exception("Failed to load security state from snapshot.")

# --------------------------------
# Teste Rápido (demo)
# --------------------------------
if __name__ == "__main__":
    # Mock do ControlBus para teste e simulação de assinatura
    class MockControlBus:
        def __init__(self):
            self.events_published = []
            self.subscriptions: Dict[str, List[Any]] = {}
        def subscribe(self, event: str, handler: Any):
            self.subscriptions.setdefault(event, []).append(handler)
        async def publish(self, event_type: str, payload: Dict[str, Any], source_module: str = "unknown"):
            print(f"[ControlBus: {source_module}] Publicado: {event_type} -> {payload}")
            self.events_published.append({"event": event_type, "payload": payload})
            # dispatch to subscribers if exist (call async handlers)
            handlers = self.subscriptions.get(event_type, [])
            for h in handlers:
                try:
                    if asyncio.iscoroutinefunction(h):
                        await h(payload)
                    else:
                        # run sync handler in executor to avoid blocking
                        loop = asyncio.get_running_loop()
                        await loop.run_in_executor(None, h, payload)
                except Exception:
                    logger.exception("MockControlBus handler failed.")

    async def demo():
        cb = MockControlBus()
        sec = Security(control_bus=cb, vector_hash_alg='sha1')  # inicia com sha1

        # 1. Teste de Configuração Dinâmica (Ajuste 3)
        print("--- 1. Teste de Configuração Dinâmica ---")
        await cb.publish(SystemEvents.CONFIG_ADJUSTED, {"max_input_length": 100, "vector_hash_alg": "sha512"}, source_module="Test")
        print(f"Novos Parâmetros: Max Len={sec.max_input_length}, Hash Alg={sec.vector_hash_alg}")

        # 2. Teste de Sanitização e Contagem (Ajuste 2)
        print("\n--- 2. Teste de Sanitização e Contagem ---")
        safe, data1 = sec.sanitize_input("Qual o meu telefone? (11) 9999-8888")
        print(f"PII: Safe={safe}. Rejeições PII: {sec.rejection_counts['pii_rejections']}")
        safe, data_len = sec.sanitize_input("A" * 101)
        print(f"Length: Safe={safe}. Rejeições Length: {sec.rejection_counts['length_rejections']}")

        # 3. Teste de Integridade (Ajuste 1: Publicar Violação)
        print("\n--- 3. Teste de Integridade (Falha) ---")
        vector_a = np.array([0.12345678, 0.98765432, 0.5])
        hash_a = sec.hash_vector(vector_a)
        # Simular corrupção (mudar um valor)
        vector_c_corrupt = np.array([0.12345679, 0.98765432, 0.5])
        integrity_c = await sec.check_vector_integrity(vector_c_corrupt, hash_a)
        print(f"Integridade Corrompida (espera FALSE): {integrity_c}")

        # 4. Snapshot Final (com contadores)
        print("\n--- 4. Snapshot Final (com contadores) ---")
        print(json.dumps(sec.snapshot_state(), indent=2))

        # Verificações rápidas (asserts)
        assert cb.events_published[-1]["event"] == SystemEvents.INTEGRITY_VIOLATION
        assert sec.rejection_counts['pii_rejections'] == 1
        assert sec.rejection_counts['length_rejections'] == 1

    asyncio.run(demo())

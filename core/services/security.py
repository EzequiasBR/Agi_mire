# core/services/security.py
"""
Security - Módulo de Segurança e Integridade de Dados.
Fornece utilitários para sanitização de entrada, verificação de integridade 
vetorial (hashing) e filtragem de saída para conformidade e robustez.
"""
from __future__ import annotations
import time
import logging
from typing import Any, Dict, List, Optional, Tuple
from threading import Lock
import numpy as np
import hashlib
import json

# Importar setup_logger da mesma pasta services
try:
    from .utils import setup_logger
except ImportError:
    # Fallback simples
    def setup_logger(name):
        logger = logging.getLogger(name)
        if not logger.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s"))
            logger.addHandler(ch)
        logger.setLevel(logging.INFO)
        return logger

logger = setup_logger("SecurityService")

class Security:
    def __init__(self, 
                 pii_keywords: List[str] = None,
                 vector_hash_alg: str = 'sha256',
                 max_input_length: int = 4096):
        
        self._lock = Lock()
        self.vector_hash_alg = vector_hash_alg
        self.max_input_length = max_input_length
        
        # Lista de palavras-chave simuladas para dados PII (Personally Identifiable Information)
        self.pii_keywords = [
            "cpf", "rg", "telefone", "endereço", "email", 
            "conta bancária", "data de nascimento"
        ] if pii_keywords is None else pii_keywords

    # --------------------------------
    # 1. Sanitização de Entrada
    # --------------------------------

    def sanitize_input(self, input_data: Any) -> Tuple[bool, str]:
        """
        Verifica a validade e segurança dos dados de entrada.
        Retorna (is_safe, sanitized_data).
        """
        data_str = str(input_data).lower().strip()
        
        # 1.1. Verificação de Comprimento
        if len(data_str) > self.max_input_length:
            logger.warning("Input rejected: Exceeds max length (%d).", self.max_input_length)
            return False, "Input length exceeded."

        # 1.2. Detecção de PII/Conteúdo Sensível (Simulação)
        for keyword in self.pii_keywords:
            if keyword in data_str:
                logger.warning("Input rejected: Possible PII detected (keyword: %s).", keyword)
                return False, f"Input contains potential sensitive data: {keyword}."
        
        # 1.3. Limpeza simples (remoção de caracteres de controle, etc.)
        sanitized_data = data_str.replace('\n', ' ').replace('\r', '')
        
        return True, sanitized_data

    # --------------------------------
    # 2. Verificação de Integridade Vetorial
    # --------------------------------
    
    def hash_vector(self, vector: np.ndarray) -> str:
        """
        Gera um hash SHA256 (ou configurável) de um vetor NumPy normalizado.
        """
        if vector is None:
            return ""

        try:
            # Converte o vetor para uma string de bytes usando a precisão float64
            # A ordem dos bytes é crucial para hashes consistentes
            vector_bytes = vector.astype(np.float64).tobytes(order='C')
            
            hasher = hashlib.new(self.vector_hash_alg)
            hasher.update(vector_bytes)
            return hasher.hexdigest()
            
        except Exception as e:
            logger.error("Failed to hash vector: %s", str(e))
            return "HASH_ERROR"

    def check_vector_integrity(self, vector: np.ndarray, expected_hash: str) -> bool:
        """
        Compara o hash calculado do vetor com um hash esperado.
        """
        if not expected_hash:
            logger.warning("Integrity check failed: Expected hash is empty.")
            return False
            
        current_hash = self.hash_vector(vector)
        is_valid = current_hash == expected_hash
        
        if not is_valid:
            logger.warning("Integrity check FAILED. Current: %s, Expected: %s", current_hash[:10], expected_hash[:10])
            
        return is_valid

    # --------------------------------
    # 3. Filtragem de Saída
    # --------------------------------

    def filter_output(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove campos sensíveis ou internos do resultado final antes da exposição externa.
        (Ex: remove hashes internos, estados brutos de memória)
        """
        # Cria uma cópia para evitar modificar o resultado interno original
        filtered_result = result.copy()
        
        # Lista de chaves a serem removidas ou ofuscadas (simulação)
        keys_to_remove = [
            "pcvs_hash", 
            "event_key", 
            "embedding", 
            "recon_embedding",
            "telemetry_id" 
        ]
        
        # Remove chaves de alto risco
        for key in keys_to_remove:
            if key in filtered_result:
                del filtered_result[key]
                
        # Ofusca dados em sub-estruturas (simulação para 'sym_pkg')
        if "sym_pkg" in filtered_result:
            if "vector" in filtered_result["sym_pkg"]:
                filtered_result["sym_pkg"]["vector"] = "VECTOR_HIDDEN"

        return filtered_result

    # --------------------------------
    # Snapshot (Compatibilidade com PCVS)
    # --------------------------------
    
    def snapshot_state(self) -> Dict[str, Any]:
        """Retorna o estado completo para persistência PCVS."""
        with self._lock:
            return {
                "vector_hash_alg": self.vector_hash_alg,
                "max_input_length": self.max_input_length
                # Não salva a lista de PII por motivos de segurança e simplicidade
            }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Carrega o estado a partir de um snapshot PCVS."""
        with self._lock:
            if not state:
                return
            
            self.vector_hash_alg = state.get("vector_hash_alg", self.vector_hash_alg)
            self.max_input_length = int(state.get("max_input_length", self.max_input_length))
            logger.info("Security state loaded. Hash Algo: %s", self.vector_hash_alg)

# --------------------------------
# Teste Rápido
# --------------------------------
if __name__ == "__main__":
    
    security = Security()
    
    # 1. Teste de Sanitização
    print("--- 1. Teste de Sanitização ---")
    safe, data1 = security.sanitize_input("Qual o meu CPF? 123.456.789-00")
    print(f"PII (CPF): Safe={safe}, Data='{data1}'")
    assert not safe

    safe, data2 = security.sanitize_input("Pergunta válida sobre o clima.\n\r")
    print(f"Clean: Safe={safe}, Data='{data2}'")
    assert safe
    
    # 2. Teste de Hashing e Integridade
    print("\n--- 2. Teste de Hashing e Integridade ---")
    vector_a = np.array([0.12345678, 0.98765432, 0.5])
    hash_a = security.hash_vector(vector_a)
    print("Hash A:", hash_a[:10])
    
    # Vetor idêntico (deve passar)
    vector_b = np.array([0.12345678, 0.98765432, 0.5]) 
    integrity_b = security.check_vector_integrity(vector_b, hash_a)
    print(f"Integridade B (Idêntico): {integrity_b}")
    assert integrity_b

    # Vetor ligeiramente diferente (deve falhar)
    vector_c = np.array([0.12345679, 0.98765432, 0.5]) 
    integrity_c = security.check_vector_integrity(vector_c, hash_a)
    print(f"Integridade C (Corrompido): {integrity_c}")
    assert not integrity_c
    
    # 3. Teste de Filtragem de Saída
    print("\n--- 3. Teste de Filtragem de Saída ---")
    mock_result = {
        "action": "continue",
        "D": 0.1,
        "pcvs_hash": "a1b2c3d4e5f6g7h8",
        "sym_pkg": {"id": 1, "vector": [0.1, 0.2, 0.3]},
        "telemetry_id": 55
    }
    
    filtered_output = security.filter_output(mock_result)
    print("Saída Filtrada (chaves pcvs_hash/telemetry_id/vector removidas):")
    print(json.dumps(filtered_output, indent=2))
    
    assert "pcvs_hash" not in filtered_output
    assert filtered_output["sym_pkg"]["vector"] == "VECTOR_HIDDEN"
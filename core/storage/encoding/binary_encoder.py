"""
core/storage/encoding/binary_encoder.py

BinaryEncoder V1.2 — Produção
Camada 1 do Hipocampo Neuromórfico.
"""

import numpy as np
import struct
import hashlib
import logging
from typing import Dict, Any, Tuple
from numpy.typing import NDArray

# Constantes da Camada 1
from core.storage.encoding.constants import (
    MAGIC_HEADER, PROTOCOL_VERSION, HASH_SIZE_BYTES,
    EMBEDDING_DIMENSION, DTYPE_VECTOR, PAYLOAD_SIZE_BYTES,
    HEADER_FORMAT, HEADER_SIZE_BYTES, TOTAL_SHOT_SIZE
)


class IntegrityError(Exception):
    """ Erros de integridade da Camada 1 (PRAG). """
    pass


class BinaryEncoder:
    """
    Encoder/Decoder binário para o Banco de Dados Neuromórfico.
    Responsável por:
    - Header
    - Payload
    - Footer (Hash SHA-256)
    - Checagens de integridade PRAG
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.logger.info("BinaryEncoder V1.2 inicializado e validado.")

    # ---------------------------------------------------------------------
    # Hash
    # ---------------------------------------------------------------------
    def _generate_hash(self, data: bytes) -> bytes:
        return hashlib.sha256(data).digest()

    # ---------------------------------------------------------------------
    # ENCODE
    # ---------------------------------------------------------------------
    def encode(self, vector: NDArray[np.float64], vector_id: int, timestamp: int) -> bytes:
        """
        Gera um disparo binário completo (Header + Payload + Footer).
        """
        # 1. Validação
        if not isinstance(vector, np.ndarray):
            raise TypeError("Vector deve ser um numpy.ndarray.")

        if vector.dtype != DTYPE_VECTOR:
            raise TypeError(f"Vector dtype inválido. Esperado: {DTYPE_VECTOR}, recebido: {vector.dtype}")

        if vector.ndim != 1:
            raise ValueError("O vetor deve ser unidimensional.")

        if vector.shape[0] != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Dimensão do vetor incompatível: {vector.shape[0]} != {EMBEDDING_DIMENSION}"
            )

        # 2. Payload
        payload = vector.tobytes()

        if len(payload) != PAYLOAD_SIZE_BYTES:
            raise IntegrityError(
                f"Tamanho do payload ({len(payload)}) diferente do esperado ({PAYLOAD_SIZE_BYTES})."
            )

        # 3. Header
        header = struct.pack(
            HEADER_FORMAT,
            MAGIC_HEADER,
            PROTOCOL_VERSION,
            timestamp,
            vector_id
        )

        # 4. Footer (Hash)
        data_to_hash = header + payload
        footer = self._generate_hash(data_to_hash)

        # 5. Disparo completo
        shot = data_to_hash + footer

        self.logger.debug({
            "event": "ENCODE_COMPLETED",
            "vector_id": vector_id,
            "timestamp": timestamp,
            "binary_size": len(shot),
            "hash_prefix": footer.hex()[:16]
        })

        return shot

    # ---------------------------------------------------------------------
    # DECODE
    # ---------------------------------------------------------------------
    def decode(self, binary_shot: bytes) -> Tuple[NDArray[np.float64], Dict[str, Any]]:
        """
        Decodifica um disparo binário completo e verifica sua integridade.
        """
        if len(binary_shot) != TOTAL_SHOT_SIZE:
            raise IntegrityError(
                f"Tamanho inválido: {len(binary_shot)} != {TOTAL_SHOT_SIZE}"
            )

        # 1. Footer esperado
        expected_footer = binary_shot[-HASH_SIZE_BYTES:]
        data_to_check = binary_shot[:-HASH_SIZE_BYTES]

        # 2. Recalcular hash
        actual_footer = self._generate_hash(data_to_check)

        if expected_footer != actual_footer:
            self.logger.error({
                "event": "PRAG_INTEGRITY_FAIL",
                "expected": expected_footer.hex()[:16],
                "actual": actual_footer.hex()[:16]
            })
            raise IntegrityError("Hash inválido. PRAG violado.")

        # 3. Header
        header_bytes = binary_shot[:HEADER_SIZE_BYTES]
        magic, version, timestamp, vector_id = struct.unpack(HEADER_FORMAT, header_bytes)

        if magic != MAGIC_HEADER:
            raise IntegrityError("Magic Header inválido.")

        # 4. Payload
        payload_bytes = binary_shot[HEADER_SIZE_BYTES:-HASH_SIZE_BYTES]

        if len(payload_bytes) != PAYLOAD_SIZE_BYTES:
            raise IntegrityError("Tamanho de payload inválido.")

        vector = np.frombuffer(payload_bytes, dtype=DTYPE_VECTOR)

        self.logger.debug({
            "event": "PRAG_INTEGRITY_PASS",
            "vector_id": vector_id,
            "timestamp": timestamp,
            "hash_prefix": actual_footer.hex()[:16]
        })

        return vector, {
            "version": version,
            "timestamp": timestamp,
            "vector_id": vector_id
        }

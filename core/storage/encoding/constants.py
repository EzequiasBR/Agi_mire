"""
core/storage/encoding/constants.py

Constantes oficiais da Camada 1 (Codificação Binária Neuromórfica).
Compatível com PRAG V1.0, AOL, PCVS e BinaryEncoder V1.2.
"""

import numpy as np

# ---------------------------------------------------------------------
# Identidade do Disparo Binário
# ---------------------------------------------------------------------

# Magic Bytes — assinatura fixa "MIHE"
MAGIC_HEADER = b'\x4d\x49\x48\x45'

# Versão do Protocolo
PROTOCOL_VERSION: int = 1

# Tamanho do hash SHA-256
HASH_SIZE_BYTES = 32

# ---------------------------------------------------------------------
# Estrutura do Vetor Adaptativo
# ---------------------------------------------------------------------

# Dimensão típica de modelos Transformer-base
EMBEDDING_DIMENSION = 768

# dtype real do vetor (usado pelo NumPy)
DTYPE_VECTOR = np.float32
DTYPE_SIZE_BYTES = 4  # float32 = 4 bytes

# Tamanho do payload (vetor bruto)
PAYLOAD_SIZE_BYTES = EMBEDDING_DIMENSION * DTYPE_SIZE_BYTES

# ---------------------------------------------------------------------
# Formato Binário do Header
# ---------------------------------------------------------------------

# < = little-endian
# 4s = Magic Header
# B = Versão (1 byte)
# Q = Timestamp (8 bytes)
# Q = VectorID/PCVS CycleID (8 bytes)
HEADER_FORMAT = '<4s B Q Q'
HEADER_SIZE_BYTES = 4 + 1 + 8 + 8

# ---------------------------------------------------------------------
# Tamanho total do Disparo Binário
# ---------------------------------------------------------------------

TOTAL_SHOT_SIZE = HEADER_SIZE_BYTES + PAYLOAD_SIZE_BYTES + HASH_SIZE_BYTES

# ---------------------------------------------------------------------
# Estrutura Consolidada (opcional, usada por PCVS, AOL e OEA)
# ---------------------------------------------------------------------

ENCODING_METADATA = {
    "embedding_dimension": EMBEDDING_DIMENSION,
    "dtype": str(DTYPE_VECTOR),
    "payload_bytes": PAYLOAD_SIZE_BYTES,
    "header_bytes": HEADER_SIZE_BYTES,
    "footer_bytes": HASH_SIZE_BYTES,
    "total_shot_bytes": TOTAL_SHOT_SIZE,
    "protocol_version": PROTOCOL_VERSION,
}

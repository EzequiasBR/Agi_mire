# core/storage/index_pointer.py
from typing import Dict, Tuple

class IndexPointer:
    """
    Armazena o mapeamento VectorID (ID Lógico) para o Offset Físico (Posição no AOL).
    Esta estrutura deve residir na RAM para acesso rápido.
    """
    def __init__(self):
        # Mapeamento: VectorID -> (start_offset, pattern_size)
        self.pointer_map: Dict[int, Tuple[int, int]] = {}
        self.next_vector_id = 1

    def register_pattern(self, offset: int, size: int) -> int:
        """
        Registra um novo padrão e retorna o VectorID gerado.
        """
        vector_id = self.next_vector_id
        self.pointer_map[vector_id] = (offset, size)
        self.next_vector_id += 1
        return vector_id

    def get_offset_and_size(self, vector_id: int) -> Tuple[int, int]:
        """
        Retorna a posição de disco e o tamanho de um VectorID.
        """
        if vector_id not in self.pointer_map:
            raise KeyError(f"VectorID {vector_id} não encontrado no mapa de ponteiros.")
        return self.pointer_map[vector_id]
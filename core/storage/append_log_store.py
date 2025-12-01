from typing import Dict, Tuple


class IndexPointerError(Exception):
    """Erro levantado para inconsistências na tabela de ponteiros."""
    pass


class IndexPointer:
    """
    IndexPointer V2.1 — Ponteiro Lógico → Offset Físico, seguro e consistente.

    Responsabilidades:
    - Criar VectorID incremental e estável.
    - Registrar pares (offset, size).
    - Garantir consultas seguras.
    - Detectar IDs inexistentes.
    """

    def __init__(self):
        # Estrutura direta: VectorID → (offset, size)
        self._map: Dict[int, Tuple[int, int]] = {}
        self._next_id: int = 1

    # ------------------------------------------------------------------
    # Operações principais
    # ------------------------------------------------------------------
    def register_pattern(self, offset: int, size: int) -> int:
        """
        Registra um novo padrão persistido no AOL.

        Retorna:
            vector_id gerado (int)
        """
        if offset < 0:
            raise IndexPointerError("Offset não pode ser negativo.")
        if size <= 0:
            raise IndexPointerError("Tamanho inválido para registro.")

        vector_id = self._next_id
        self._map[vector_id] = (offset, size)
        self._next_id += 1

        return vector_id

    def get_offset_and_size(self, vector_id: int) -> Tuple[int, int]:
        """
        Consulta segura do par (offset, size).
        """
        try:
            return self._map[vector_id]
        except KeyError:
            raise IndexPointerError(f"VectorID {vector_id} não encontrado.")

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------
    def contains(self, vector_id: int) -> bool:
        return vector_id in self._map

    def count(self) -> int:
        return len(self._map)

    def all_entries(self):
        return dict(self._map)

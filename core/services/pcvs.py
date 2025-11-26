# core/services/pcvs.py (Trecho ajustado)

import os
import logging
from typing import Dict, Any, Optional, List
import time

# Importações de utilitários e serviços
from ..services.utils import setup_logger, hash_state, save_json, load_json
from ..services.control_bus import ControlBus, SystemEvents

logger = setup_logger("PCVS")

class PCVS:
    # ... __init__ e _get_file_path permanecem os mesmos ...
    
    def __init__(self, control_bus: ControlBus, base_dir: str = "pcvs_data"):
        self.control_bus = control_bus
        self.base_dir = base_dir
        self.snapshot_history: List[Dict[str, Any]] = []
        self.last_snapshot_hash: Optional[str] = None
        
        os.makedirs(self.base_dir, exist_ok=True)
        self._load_history()
        
        logger.info(f"PCVS inicializado. Diretório base: {self.base_dir}")

    # --- Métodos de Persistência e Auditoria ---
    
    def _get_file_path(self, sha256: str) -> str:
        """ Constrói o caminho completo do arquivo de snapshot. """
        return os.path.join(self.base_dir, f"{sha256}.json")
    
    def _save_history(self) -> None:
        """ Persiste o histórico leve de metadados dos snapshots. """
        try:
            # Salvar o histórico para que possa ser carregado rapidamente na inicialização
            save_json(os.path.join(self.base_dir, "history.json"), self.snapshot_history)
        except Exception as e:
            logger.error(f"Falha ao persistir o histórico do PCVS: {e}")

    def _load_history(self) -> None:
        """ Carrega o histórico leve de metadados dos snapshots. """
        history_path = os.path.join(self.base_dir, "history.json")
        if os.path.exists(history_path):
            try:
                self.snapshot_history = load_json(history_path)
                # Atualizar o último hash para o mais recente no histórico
                if self.snapshot_history:
                    self.last_snapshot_hash = self.snapshot_history[-1]['hash']
                    logger.info(f"Histórico do PCVS carregado. Último hash: {self.last_snapshot_hash[:10]}")
            except Exception as e:
                logger.error(f"Falha ao carregar o histórico do PCVS: {e}")


    def persist_snapshot(self, snapshot_data: Dict[str, Any], reason: str) -> Optional[str]:
        """
        Salva o snapshot de forma "documentada", incluindo metadados e motivo.
        Gera um hash único SHA256 do conteúdo do snapshot.
        """
        
        # 1. Adicionar metadados de auditoria
        snapshot_to_save = {
            "metadata": {
                "timestamp": time.time(),
                "reason": reason,
            },
            "system_state": snapshot_data
        }
        
        # 2. Gerar Hash (o hash é a chave única e a verificação de integridade)
        sha256 = hash_state(snapshot_to_save)
        file_path = self._get_file_path(sha256)
        
        try:
            # 3. Persistir o arquivo no disco (Usa save_json do utils)
            save_json(file_path, snapshot_to_save)
            
            # 4. Atualizar o Histórico Interno
            self.last_snapshot_hash = sha256
            self.snapshot_history.append({
                'hash': sha256, 
                'timestamp': snapshot_to_save['metadata']['timestamp'],
                'reason': reason
            })
            # Garantir que o histórico interno do PCVS também seja persistido (leve)
            self._save_history()

            self.control_bus.publish(SystemEvents.SNAPSHOT_SAVED, 
                                     {'hash': sha256, 'reason': reason})
            logger.info(f"Snapshot '{reason}' salvo. Hash: {sha256[:10]}")
            return sha256
            
        except Exception as e:
            logger.error(f"Erro crítico ao salvar snapshot no PCVS: {e}")
            return None


    def load_snapshot(self, sha256: str) -> Optional[Dict[str, Any]]:
        """
        Recupera snapshot específico pelo hash. 
        Retorna o dicionário completo, incluindo metadados e estado.
        """
        file_path = self._get_file_path(sha256)
        
        if not os.path.exists(file_path):
            logger.warning(f"Snapshot não encontrado para o hash: {sha256[:10]}")
            return None
            
        try:
            # 1. Carregar o arquivo (Usa load_json do utils)
            loaded_snapshot = load_json(file_path)
            
            # 2. VERIFICAÇÃO DE INTEGRIDADE
            # O hash do conteúdo carregado DEVE ser igual ao hash do nome do arquivo
            # O MCH, ao chamar load, já faz a verificação externa.
            
            # **Nota de design:** No fluxo MCH -> PCVS.load, o MCH já faz a verificação do hash.
            # O PCVS apenas retorna o dado carregado.
            
            return loaded_snapshot

        except Exception as e:
            logger.error(f"Erro ao carregar snapshot {sha256[:10]}: {e}")
            return None


    def get_last_snapshot_hash(self) -> Optional[str]:
        """ Retorna hash do último snapshot salvo; útil para rollbacks rápidos. """
        return self.last_snapshot_hash

    # --- Métodos de Controle (Delegados ao MCH/SystemLoop) ---

    def rollback(self, target_hash: str):
        """
        [DELEGADO] Sinaliza ao ControlBus para iniciar o Rollback,
        mas a execução real da restauração de estado é feita pelo MCH.
        """
        logger.warning(f"Sinalizando Rollback para o hash: {target_hash[:10]}")
        self.control_bus.publish(SystemEvents.ROLLBACK_REQUESTED, 
                                 {'target_hash': target_hash, 'source': 'PCVS'})
        # O SystemLoop ou o MCH irá escutar este evento e executar MCH.load_pcvs_snapshot().
        
        # (NÃO É RESPONSABILIDADE DO PCVS RESTAURAR O ESTADO INTERNO DE OUTROS MÓDULOS)
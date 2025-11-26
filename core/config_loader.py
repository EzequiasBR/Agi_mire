# core/config_loader.py
"""
Utilitário para carregar e inicializar configurações do MIHE-AGI.
Inclui: system.yaml, thresholds.json e logging.json.
Integra logger e fornece objetos de configuração prontos para uso nos módulos core.
"""

import json
import yaml
import logging
import logging.config
from pathlib import Path
from typing import Dict, Any

def load_yaml(path: str) -> Dict[str, Any]:
    """Carrega um arquivo YAML e retorna um dicionário."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_json(path: str) -> Dict[str, Any]:
    """Carrega um arquivo JSON e retorna um dicionário."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def setup_logging(logging_json_path: str) -> logging.Logger:
    """Configura logger global a partir de arquivo JSON de logging."""
    config_path = Path(logging_json_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Arquivo de logging não encontrado: {logging_json_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    logging.config.dictConfig(config_dict)
    return logging.getLogger("MIHE-AGI")  # logger raiz principal

class ConfigLoader:
    """
    Classe utilitária para carregar todas as configurações do MIHE-AGI.
    - system.yaml
    - thresholds.json
    - logging.json
    """

    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.system: Dict[str, Any] = {}
        self.thresholds: Dict[str, Any] = {}
        self.logger: logging.Logger = None

    def load_all(self):
        """Carrega todos os arquivos de configuração e inicializa logger."""
        # 1️⃣ system.yaml
        system_path = self.config_dir / "system.yaml"
        if not system_path.exists():
            raise FileNotFoundError(f"system.yaml não encontrado em {system_path}")
        self.system = load_yaml(str(system_path))

        # 2️⃣ thresholds.json
        thresholds_path = self.config_dir / "thresholds.json"
        if not thresholds_path.exists():
            raise FileNotFoundError(f"thresholds.json não encontrado em {thresholds_path}")
        self.thresholds = load_json(str(thresholds_path))

        # 3️⃣ logging.json
        logging_path = self.config_dir / "logging.json"
        if not logging_path.exists():
            raise FileNotFoundError(f"logging.json não encontrado em {logging_path}")
        self.logger = setup_logging(str(logging_path))
        self.logger.info("✅ Configurações carregadas com sucesso.")

    def get_module_thresholds(self, module_name: str) -> Dict[str, Any]:
        """Retorna thresholds específicos para um módulo (PPO, PRAG, etc.)"""
        return self.thresholds.get(module_name, {})

    def get_system_param(self, param_path: str, default=None):
        """
        Busca um parâmetro dentro de system.yaml usando notação 'modules.OL.dim'
        """
        keys = param_path.split(".")
        current = self.system
        try:
            for k in keys:
                current = current[k]
            return current
        except KeyError:
            return default

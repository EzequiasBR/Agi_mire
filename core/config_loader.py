import os
import json
import logging
import yaml
from typing import Any, Dict, Optional


class ConfigLoader:
    """
    Carrega arquivos de configuração JSON e YAML de um diretório específico.
    Permite acesso rápido a thresholds ou outros parâmetros por módulo.
    """

    SUPPORTED_EXTENSIONS = (".json", ".yaml", ".yml")

    def __init__(self, config_dir: str = "configs"):
        self.config_dir = os.path.abspath(config_dir)
        self.logger = logging.getLogger("ConfigLoader")
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s"))
            self.logger.addHandler(ch)
        self.logger.setLevel(logging.INFO)
        self.configs: Dict[str, Dict[str, Any]] = {}

    def load_all(self):
        """Carrega todos os arquivos JSON/YAML do diretório."""
        if not os.path.isdir(self.config_dir):
            self.logger.warning(f"Diretório de configs não existe: {self.config_dir}")
            return

        for fname in os.listdir(self.config_dir):
            path = os.path.join(self.config_dir, fname)
            if not fname.endswith(self.SUPPORTED_EXTENSIONS):
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    if fname.endswith(".json"):
                        data = json.load(f)
                    else:
                        data = yaml.safe_load(f)
                    self.configs[fname] = data
                    self.logger.info(f"Config carregada: {fname}")
            except Exception as e:
                self.logger.error(f"Erro ao carregar config {fname}: {e}")

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        """Retorna toda a configuração de um arquivo específico, sem extensão."""
        for fname, data in self.configs.items():
            base = os.path.splitext(fname)[0]
            if base == name:
                return data
        self.logger.warning(f"Config {name} não encontrada")
        return None

    def get_module_thresholds(self, module_name: str) -> Dict[str, Any]:
        """Retorna thresholds específicos de um módulo, vazio se não existir."""
        cfg = self.get(module_name)
        if cfg and isinstance(cfg, dict):
            return cfg.get("thresholds", {})
        return {}

    def reload(self):
        """Recarrega todas as configs do diretório."""
        self.configs.clear()
        self.load_all()

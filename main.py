#!/usr/bin/env python3
"""
main.py — Orquestrador operacional MIHE/AGI (Modo: Completo / Canary / Interativo)

Funcionalidades:
 - inicializa módulos reais de core/
 - cria diretórios (logs, snapshots, checkpoints, audit/evidence)
 - carrega configs YAML/JSON em configs/
 - configura logging via configs/logging.json (se existir)
 - executa cenários automatizados (normal, parcial, crítico)
 - mede métricas (D, C, durations)
 - persiste snapshots via PCVS (pcvs.save)
 - modo interativo opcional (CLI)
 - hooks robustos e fallbacks caso alguma interface falhe
"""

import os
import sys
import time
import json
import argparse
import logging
import logging.config
import yaml
import numpy as np
import inspect

from typing import Any, Dict, Optional, Tuple
from pathlib import Path
from core.orchestration.mch import MCH
from core.services.pcvs import PCVS
from core.governance.prag import PRAG
from core.governance.regvet import RegVet
from core.governance.simlog import SimLog
from core.intelligence.oa import OA
from core.intelligence.ol import OL
from core.intelligence.ppo import PPO
from core.memory.hippocampus import Hippocampus
from core.services.pcvs import PCVS



# -----------------------
# Paths & Directories
# -----------------------
ROOT = Path(".").resolve()
CONFIG_DIR = ROOT / "configs"
LOGS_DIR = ROOT / "logs"
SNAPSHOTS_DIR = ROOT / "snapshots"
CHECKPOINTS_DIR = ROOT / "checkpoints"
AUDIT_DIR = ROOT / "audit" / "evidence"

for d in (LOGS_DIR, SNAPSHOTS_DIR, CHECKPOINTS_DIR, AUDIT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# -----------------------
# Config loaders
# -----------------------
def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        logging.warning("Config not found: %s", path)
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        logging.warning("Config not found: %s", path)
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

# -----------------------
# Logging setup
# -----------------------
def setup_logging():
    cfg_path = CONFIG_DIR / "logging.json"
    if cfg_path.exists():
        try:
            with open(cfg_path, "r", encoding="utf-8") as fh:
                cfg = json.load(fh)
            logging.config.dictConfig(cfg)
            logging.getLogger(__name__).info("Loaded logging config from %s", cfg_path)
            return
        except Exception as e:
            print("Failed to load logging config:", e)
    # Default logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOGS_DIR / "system.log", encoding="utf-8")
        ]
    )
    logging.getLogger(__name__).info("Default logging configured (stdout + logs/system.log)")

# -----------------------
# Helper utilities
# -----------------------
def timestamp_ms() -> int:
    return int(time.time() * 1000)

def safe_snapshot_save(pcvs: PCVS, payload: Dict[str, Any]) -> Optional[str]:
    try:
        h = pcvs.save(payload)
        logging.getLogger("main").info("Snapshot saved PCVS hash=%s", str(h)[:12])
        # persist snapshot json human-readable
        p = SNAPSHOTS_DIR / f"pcvs_{str(h)[:12]}_{timestamp_ms()}.json"
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        return h
    except Exception:
        logging.exception("Failed to save snapshot via PCVS")
        return None

def save_audit_evidence(name: str, data: Any):
    p = AUDIT_DIR / f"{name}_{timestamp_ms()}.json"
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    logging.getLogger("main").info("Audit evidence saved %s", p)

# -----------------------
# System initializer
# -----------------------
def init_system(system_cfg, thresholds_cfg):
    """
    Inicializa TODOS os módulos reais da pasta core/
    usando detecção dinâmica de assinatura.
    """
    

    log = logging.getLogger("main")

    def construct(module_cls, **kwargs):
        """Inicialização robusta: tenta apenas argumentos que existem."""
        sig = inspect.signature(module_cls.__init__)
        accepted = {
            k: v for k, v in kwargs.items()
            if k in sig.parameters and k != "self"
        }
        try:
            return module_cls(**accepted)
        except Exception as e:
            log.exception(f"Falha ao inicializar {module_cls.__name__}: {e}")
            raise

    # --- Preparar parâmetros reais ---
    ol_cfg  = system_cfg["modules"]["OL"]
    oa_cfg  = system_cfg["modules"]["OA"]
    hip_cfg = system_cfg["modules"]["Hippocampus"]
    sim_cfg = system_cfg["modules"]["SimLog"]
    reg_cfg = system_cfg["modules"]["RegVet"]
    ppo_cfg = thresholds_cfg["PPO"]
    prag_cfg = thresholds_cfg["PRAG"]

    # --- Inicializar módulos reais ---
    ol  = construct(OL, **ol_cfg)
    oa  = construct(OA, **oa_cfg)
    hip = construct(Hippocampus, **hip_cfg)
    sim = construct(SimLog, **sim_cfg)
    reg = construct(RegVet, **reg_cfg)
    ppo = construct(PPO, **ppo_cfg)
    prag = construct(PRAG, **prag_cfg)
    pcvs = construct(PCVS, base_dir="snapshots")

    log.info("Todos os módulos centrais foram inicializados corretamente.")

    return {
        "ol": ol,
        "oa": oa,
        "hip": hip,
        "simlog": sim,
        "regvet": reg,
        "ppo": ppo,
        "prag": prag,
        "pcvs": pcvs,
    }

# -----------------------
# Scenarios & runners
# -----------------------
def run_scenarios(mods: Dict[str, Any], interactive: bool = False, cycles:int = 3):
    mch: MCH = mods["mch"]
    ol = mods["ol"]
    hip = mods["hip"]
    pcvs = mods["pcvs"]
    ppo = mods["ppo"]
    prag = mods["prag"]

    logger = logging.getLogger("main")
    results = []

    # baseline snapshot
    try:
        baseline_hash = mch.force_save_snapshot()
        logger.info("Baseline snapshot saved: %s", str(baseline_hash)[:12])
    except Exception:
        baseline_hash = None
        logger.exception("Failed to save baseline snapshot via MCH")

    # Scenario definitions (normal, partial, critical)
    scenarios = [
        {"name": "normal", "divergence": 0.005, "C": 0.95, "E": 0.05},
        {"name": "partial", "divergence": 0.75, "C": 0.50, "E": 0.05},
        {"name": "critical_error", "divergence": 0.50, "C": 0.50, "E": 0.95}
    ]

    # run core cycles
    for s in scenarios:
        logger.info(">> Running scenario: %s", s["name"])
        # monkeypatch regvet.enforce for scenario (safe - MCH will call it)
        try:
            original_enforce = mods["regvet"].enforce
            mods["regvet"].enforce = lambda v, rules: {"divergence": s["divergence"]}
        except Exception:
            original_enforce = None

        # run cycles for this scenario
        for i in range(cycles):
            start = time.time()
            out = mch.process(f"[{s['name']}] example input #{i}", E_sistemica=s["E"], inject_pathogen=None)
            dur = time.time() - start
            record = {
                "scenario": s["name"],
                "iteration": i,
                "action": out.get("action", out.get("Action", "unknown")),
                "D": out.get("D", None),
                "C": out.get("C", None),
                "pcvs_hash": out.get("pcvs_hash", None),
                "duration_s": dur,
                "timestamp": time.time()
            }
            results.append(record)
            logger.info("Cycle result: %s", record)
            # periodic snapshot after each cycle (best-effort)
            try:
                payload = mch._compose_system_state() if hasattr(mch, "_compose_system_state") else {}
                s_hash = safe_snapshot_save(pcvs, payload)
                if s_hash:
                    record["snapshot_hash"] = s_hash
            except Exception:
                logger.exception("Failed to persist periodic snapshot")

        # restore original enforce
        if original_enforce is not None:
            mods["regvet"].enforce = original_enforce

    # Post-run evidence
    save_audit_evidence("scenarios_run", results)
    return results, baseline_hash

# -----------------------
# Interactive loop
# -----------------------
def interactive_loop(mods: Dict[str, Any]):
    logger = logging.getLogger("main")
    mch: MCH = mods["mch"]

    logger.info("Entering interactive mode. Type 'exit' to quit.")
    while True:
        try:
            text = input("MIHE> ").strip()
            if text in ("exit", "quit"):
                break
            if text == "":
                continue
            start = time.time()
            out = mch.process(text, E_sistemica=0.1)
            dt = time.time() - start
            print("=> action:", out.get("action", out.get("Action", "unknown")), "| duration_s:", f"{dt:.3f}")
        except KeyboardInterrupt:
            logger.info("Interactive interrupted by user")
            break
        except Exception:
            logger.exception("Interactive cycle failed")

# -----------------------
# CLI / Entrypoint
# -----------------------
def parse_args():
    p = argparse.ArgumentParser(description="MIHE/AGI - Main Orchestrator")
    p.add_argument("--config", default="system.yaml", help="system YAML config filename in configs/")
    p.add_argument("--thresholds", default="thresholds.json", help="thresholds JSON filename in configs/")
    p.add_argument("--interactive", action="store_true", help="enter interactive loop after scenarios")
    p.add_argument("--cycles", type=int, default=3, help="number of cycles per scenario")
    p.add_argument("--run-audit", action="store_true", help="run audit_longitudinal script after scenarios (if present)")
    return p.parse_args()

def main():
    setup_logging()
    logger = logging.getLogger("main")
    args = parse_args()

    # load configs
    system_cfg = load_yaml(CONFIG_DIR / args.config)
    thresholds_cfg = load_json(CONFIG_DIR / args.thresholds)
    if not system_cfg:
        logger.error("System config not found or invalid: %s. Exiting.", args.config)
        sys.exit(1)
    if not thresholds_cfg:
        logger.error("Thresholds config not found or invalid: %s. Exiting.", args.thresholds)
        sys.exit(1)

    # init system with real modules
    try:
        mods = init_system(system_cfg, thresholds_cfg)
    except Exception:
        logger.exception("Failed to initialize system modules")
        sys.exit(2)

    # run deterministic scenarios and persist evidence
    try:
        results, baseline_hash = run_scenarios(mods, interactive=args.interactive, cycles=args.cycles)
        logger.info("Scenarios completed. Baseline: %s", str(baseline_hash)[:12])
    except Exception:
        logger.exception("Scenarios execution failed")
        sys.exit(3)

    # optional interactive loop
    if args.interactive:
        interactive_loop(mods)

    # optional run audit_longitudinal if flagged and file exists
    if args.run_audit:
        audit_script = ROOT / "audit" / "audit_longitudinal.py"
        if audit_script.exists():
            logger.info("Running audit_longitudinal.py...")
            try:
                import runpy
                runpy.run_path(str(audit_script), run_name="__main__")
            except Exception:
                logger.exception("audit_longitudinal failed")
        else:
            logger.warning("audit_longitudinal.py not found; skipping.")

    logger.info("Main run finished. Evidence stored in %s", AUDIT_DIR)

if __name__ == "__main__":
    main()

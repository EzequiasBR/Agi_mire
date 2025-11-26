# demo/rollback_demo.py
# Demonstração de Rollback (Total e Parcial) do MCH

import logging
import random
import numpy as np

# --- Mocks simplificados dos módulos ---

def divergence_from_cosine(cos_sim: float) -> float:
    return max(0.0, min(1.0, (1.0 - cos_sim) / 2.0))

class MockOL:
    def __init__(self, dim=128, metric="cosine"):
        self.dim = dim
        self.metric = metric
        logging.info(f"OL inicializado com dim={dim}, metric={metric}.")
    def embed(self, text):
        return np.random.randn(self.dim)

class MockOA:
    def __init__(self, ruleset="default"):
        logging.info(f"OA inicializado com ruleset={ruleset}.")

class MockHippocampus:
    def __init__(self):
        self.memory_count = 0
        logging.info("Hipocampo inicializado.")
    def store(self, item):
        self.memory_count += 1
        logging.info(f"Hipocampo armazenou item. Total={self.memory_count}")

class MockSimLog:
    def __init__(self, max_error=0.05):
        self.max_error = max_error
        logging.info(f"SimLog inicializado com max_error={max_error}.")
    def round_trip_valid(self, v1, v2):
        error_rt = random.uniform(0.0, self.max_error * 2.0)
        return error_rt <= self.max_error, error_rt

class MockRegVet:
    def __init__(self):
        logging.info("RegVet inicializado.")
    def enforce(self, v, rules):
        sim_score = random.uniform(-0.1, 0.99)
        return {"divergence": divergence_from_cosine(sim_score)}

class MockPRAG:
    def __init__(self, d_threshold=0.85, p_threshold=0.70):
        self.divergence_threshold = d_threshold
        self.partial_threshold = p_threshold
        self.audit_log = []
    def should_rollback_total(self, D):
        return D > self.divergence_threshold
    def should_rollback_partial(self, D):
        return self.partial_threshold < D <= self.divergence_threshold
    def log_cycle(self, D, **kwargs):
        self.audit_log.append({"D": D, **kwargs})
        logging.info(f"PRAG Log: D={D:.4f}, Total={self.should_rollback_total(D)}, Parcial={self.should_rollback_partial(D)}")

class MockPCVS:
    def __init__(self):
        self.snapshots = {}
    def save(self, state):
        key = f"PCVS-{len(self.snapshots)+1}"
        self.snapshots[key] = state
        logging.info(f"PCVS salvo: {key}")
        return key
    def load(self, key):
        return self.snapshots.get(key, {})

# --- Classe Rollback Demo ---

class RollbackDemo:
    def __init__(self):
        self.ol = MockOL()
        self.oa = MockOA()
        self.hip = MockHippocampus()
        self.simlog = MockSimLog()
        self.regvet = MockRegVet()
        self.prag = MockPRAG()
        self.pcvs = MockPCVS()

    def snapshot_state(self):
        """Constrói o estado atual em forma de dicionário."""
        return {
            "hippocampus": {"memory_count": self.hip.memory_count},
            "prag": {"audit_log_len": len(self.prag.audit_log)},
            "pcvs": list(self.pcvs.snapshots.keys())
        }

    # Alias sem recursão infinita
    serialize_state = snapshot_state

    def run_cycle(self, text, C, E):
        logging.info(f"\n--- Ciclo Coeso para: '{text[:20]}...' ---")
        v = self.ol.embed(text)
        D = self.regvet.enforce(v, rules=[])["divergence"]
        valid_rt, err_rt = self.simlog.round_trip_valid(v, v)
        self.prag.log_cycle(D, C=C, E_sistemica=E, valid_rt=valid_rt)

        if self.prag.should_rollback_total(D):
            logging.critical("Rollback TOTAL acionado.")
            self.pcvs.save(self.snapshot_state())
            return "ROLLBACK TOTAL"
        elif self.prag.should_rollback_partial(D):
            logging.warning("Rollback PARCIAL acionado.")
            self.pcvs.save(self.snapshot_state())
            return "ROLLBACK PARCIAL"

        self.hip.store(text)
        return "APRENDIZADO"

# --- Execução Demo ---

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    demo = RollbackDemo()

    # Cenário 1: Rollback total (D > 0.85)
    demo.regvet.enforce = lambda v, rules: {"divergence": 0.90}
    demo.run_cycle("Texto com divergência extrema...", C=0.50, E=0.10)

    # Cenário 2: Rollback parcial (0.70 < D <= 0.85)
    demo.regvet.enforce = lambda v, rules: {"divergence": 0.75}
    demo.run_cycle("Texto com divergência média...", C=0.50, E=0.10)

    # Cenário 3: Ciclo normal (D baixo, sem rollback)
    demo.regvet.enforce = lambda v, rules: {"divergence": 0.10}
    demo.run_cycle("Texto estável e coerente...", C=0.95, E=0.05)
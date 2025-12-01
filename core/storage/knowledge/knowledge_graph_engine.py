# core/storage/knowledge/knowledge_graph_engine.py
"""
KnowledgeGraphEngine V5.0 — Implementação refatorada para Agi_mire

Principais características:
- Normalização PRAG-C na atualização de certeza
- Aging / decay temporal e pruning
- Hash multimodal por fato
- Auditoria via ControlBus / SimLog (best-effort)
- Integração com RegVet (validação vetorial) e RuleBase (checagens lógicas)
- Snapshots / rollback (PCVS compatible)
- APIs adicionais: export_state, load_state, snapshot, rollback_to_snapshot
- Métodos existentes preservados: add_or_update_triple, query_subject, check_fact_certainty
"""

from __future__ import annotations
import time
import hashlib
import logging
from typing import Dict, Tuple, List, Optional, Any

PredicateData = Tuple[str, str, float, str, float]
# (predicate, object, certainty_C, fact_hash, last_updated_ts)

logger = logging.getLogger("KnowledgeGraphEngine")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s KG %(levelname)s: %(message)s"))
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)


class KnowledgeGraphEngine:
    # PRAG consolidation weights (can be tuned / moved to config)
    PRAG_ALPHA = 0.62  # weight old certainty
    PRAG_BETA = 0.38   # weight new evidence
    DECAY_PER_SEC = 0.9999  # multiplicative decay per second (very light)
    PRUNE_THRESHOLD = 1e-4

    def __init__(
        self,
        prag: Optional[Any] = None,
        simlog: Optional[Any] = None,
        regvet: Optional[Any] = None,
        symbol_table: Optional[Any] = None,
        control_bus: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            prag: PRAG controller (optional) — used for emitting events / snapshots / rollback
            simlog: SimLog integration (optional) — used for semantic validation and audit logging
            regvet: RegVet integration (optional) — used for vector validation if available
            symbol_table: SymbolTable integration (optional) — for V-ID <-> T-ID mapping
            control_bus: event bus (optional) — publish events
            config: optional tuning parameters
        """
        self.prag = prag
        self.simlog = simlog
        self.regvet = regvet
        self.symbol_table = symbol_table
        self.control_bus = control_bus
        self.config = config or {}

        # Core storage: subject -> list of PredicateData
        self.graph: Dict[str, List[PredicateData]] = {}

        # PCVS-like snapshot history: snapshot_id -> state dict
        self._snapshots: Dict[str, Dict[str, Any]] = {}

        # internal last-update timestamp
        self._last_update_ts = time.time()

        logger.info("KnowledgeGraphEngine initialized")

    # ---------------------------
    # Utilities
    # ---------------------------
    def _now(self) -> float:
        return time.time()

    def _normalize_certainty(self, c: float) -> float:
        """Clamp certainty to [0.0, 1.0]."""
        try:
            cf = float(c)
        except Exception:
            cf = 0.0
        return max(0.0, min(1.0, cf))

    def _hash_fact(self, subject: str, predicate: str, obj: str) -> str:
        raw = f"{subject}::{predicate}::{obj}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _prag_consolidate(self, old_c: float, new_c: float) -> float:
        """Apply PRAG consolidation rule: C_new = alpha*old + beta*incoming"""
        return max(0.0, min(1.0, old_c * self.PRAG_ALPHA + new_c * self.PRAG_BETA))

    def _apply_decay_and_prune_subject(self, subject: str) -> None:
        """Decay certainties for a subject and prune tiny facts."""
        now = self._now()
        updated_list: List[PredicateData] = []
        changed = False
        for (p, o, c, h, ts) in self.graph.get(subject, []):
            # time-based decay: apply multiplicative decay per second elapsed
            age = max(0.0, now - ts)
            decayed = float(c * (self.DECAY_PER_SEC ** age))
            if decayed >= self.PRUNE_THRESHOLD:
                updated_list.append((p, o, decayed, h, ts))
            else:
                changed = True
        if changed:
            logger.debug({"event": "KG_PRUNE", "subject": subject, "remaining": len(updated_list)})
        self.graph[subject] = updated_list

    # ---------------------------
    # Public API (existing names preserved)
    # ---------------------------
    def add_or_update_triple(self, subject: str, predicate: str, obj: str, certainty: float) -> bool:
        """
        Add or update a triple (subject, predicate, object) with evidence certainty.
        Returns True if saved/updated, False if rejected (e.g., regvet/simlog veto).
        """
        # Basic sanitization
        if not all(isinstance(x, str) and x for x in (subject, predicate, obj)):
            logger.error("KG.add_or_update_triple: invalid subject/predicate/object types")
            return False

        certainty = self._normalize_certainty(certainty)
        now = self._now()

        # 1) Optional RegVet validation (vector/semantic coherence)
        try:
            if self.regvet and hasattr(self.regvet, "validate_fact"):
                ok, score = self.regvet.validate_fact(subject, predicate, obj)
                if not ok:
                    logger.warning({"event": "KG_REJECTED_REGVET", "subject": subject, "predicate": predicate})
                    if self.control_bus:
                        self.control_bus.publish("PRAG_KG_REJECTED_REGVET", {"subject": subject, "predicate": predicate})
                    return False
        except Exception:
            logger.exception("KG.regvet.validate_fact failed; proceeding defensively")

        # 2) Optional SimLog consistency check
        try:
            if self.simlog and hasattr(self.simlog, "validate_kg_fact"):
                ok, score = self.simlog.validate_kg_fact(subject, predicate, obj)
                if not ok:
                    logger.warning({"event": "KG_REJECTED_SIMLOG", "subject": subject, "predicate": predicate})
                    if self.control_bus:
                        self.control_bus.publish("PRAG_KG_INCONSISTENCY", {"subject": subject, "predicate": predicate})
                    return False
        except Exception:
            logger.exception("KG.simlog.validate_kg_fact failed; proceeding defensively")

        # ensure subject exists
        if subject not in self.graph:
            self.graph[subject] = []

        # compute fact hash
        fact_hash = self._hash_fact(subject, predicate, obj)

        # find existing fact
        found_index = None
        for i, (p, o, c_old, h_old, ts_old) in enumerate(self.graph[subject]):
            if p == predicate and o == obj:
                found_index = i
                break

        if found_index is not None:
            # Consolidate certainties (PRAG rule)
            p_old, o_old, c_old, h_old, ts_old = self.graph[subject][found_index]
            c_new = self._prag_consolidate(c_old, certainty)
            self.graph[subject][found_index] = (p_old, o_old, c_new, h_old, now)
            logger.info({"event": "PRAG_KG_C_UPDATED", "subject": subject, "predicate": predicate, "old_c": c_old, "new_c": c_new})
            if self.control_bus:
                self.control_bus.publish("PRAG_KG_C_UPDATED", {"subject": subject, "predicate": predicate, "old_c": c_old, "new_c": c_new})
        else:
            # insert new fact
            self.graph[subject].append((predicate, obj, certainty, fact_hash, now))
            logger.info({"event": "PRAG_KG_NEW_FACT", "subject": subject, "predicate": predicate, "object": obj, "certainty": certainty})
            if self.control_bus:
                self.control_bus.publish("PRAG_KG_NEW_FACT", {"subject": subject, "predicate": predicate, "object": obj, "certainty": certainty})

        # update last ts and optionally notify simlog / symbol_table
        self._last_update_ts = now
        try:
            if self.simlog and hasattr(self.simlog, "log_kg_event"):
                self.simlog.log_kg_event(subject, predicate, obj, certainty)
        except Exception:
            logger.debug("simlog.log_kg_event failed", exc_info=True)

        try:
            if self.symbol_table and hasattr(self.symbol_table, "register_mapping"):
                # register a mapping between a fact hash and a generated v-id placeholder (best-effort)
                # Prefer to register mapping only if symbol_table expects tripla_id <-> v-id; use fact_hash as tripla_id
                v_id = f"KG:{fact_hash}"
                self.symbol_table.register_mapping(v_id, fact_hash)
        except Exception:
            logger.debug("symbol_table.register_mapping failed", exc_info=True)

        return True

    def query_subject(self, subject: str) -> List[PredicateData]:
        """Return list of predicate tuples for a subject (with current certainties)."""
        if subject not in self.graph:
            return []
        # apply decay pruning lazily on query
        self._apply_decay_and_prune_subject(subject)
        return list(self.graph.get(subject, []))

    def check_fact_certainty(self, subject: str, predicate: str, obj: str) -> float:
        """Return certainty C for a specific fact, or 0.0 if not present."""
        if subject not in self.graph:
            return 0.0
        for p, o, c, h, ts in self.graph[subject]:
            if p == predicate and o == obj:
                # apply small decay based on elapsed time
                age = max(0.0, self._now() - ts)
                decayed = float(c * (self.DECAY_PER_SEC ** age))
                return max(0.0, min(1.0, decayed))
        return 0.0

    # ---------------------------
    # Snapshot / PCVS helpers
    # ---------------------------
    def export_state(self) -> Dict[str, Any]:
        """Export a deep-serializable snapshot of current KG state (PCVS-ready)."""
        state = {}
        for s, facts in self.graph.items():
            state[s] = [(p, o, float(c), h, float(ts)) for (p, o, c, h, ts) in facts]
        return {"graph": state, "ts": self._now()}

    def snapshot(self, snapshot_id: Optional[str] = None) -> str:
        """Save a snapshot and return its id (pcvs-like)."""
        sid = snapshot_id or hashlib.sha256(f"{time.time()}".encode("utf-8")).hexdigest()
        self._snapshots[sid] = self.export_state()
        logger.info({"event": "PRAG_KG_SNAPSHOT", "snapshot_id": sid})
        if self.control_bus:
            self.control_bus.publish("PRAG_KG_SNAPSHOT", {"snapshot_id": sid})
        return sid

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load state exported by export_state()."""
        raw = state.get("graph", {})
        new_graph: Dict[str, List[PredicateData]] = {}
        for s, facts in raw.items():
            new_graph[s] = [(p, o, float(c), str(h), float(ts)) for (p, o, c, h, ts) in facts]
        self.graph = new_graph
        self._last_update_ts = self._now()
        logger.info({"event": "PRAG_KG_RESTORED"})

    def rollback_to_snapshot(self, snapshot_id: str) -> bool:
        """Rollback KG to a previously saved snapshot. Returns True if success."""
        snap = self._snapshots.get(snapshot_id)
        if not snap:
            logger.error({"event": "PRAG_KG_ROLLBACK_FAIL", "snapshot_id": snapshot_id})
            return False
        try:
            self.load_state(snap)
            logger.warning({"event": "PRAG_KG_ROLLED_BACK", "snapshot_id": snapshot_id})
            if self.control_bus:
                self.control_bus.publish("PRAG_KG_ROLLED_BACK", {"snapshot_id": snapshot_id})
            return True
        except Exception:
            logger.exception("PRAG_KG_ROLLBACK_ERROR")
            return False

    # ---------------------------
    # Introspection and maintenance
    # ---------------------------
    def get_all_facts(self) -> Dict[str, List[PredicateData]]:
        """Return a shallow copy of the knowledge graph (for inspection)."""
        # apply decay globally before exposing
        for s in list(self.graph.keys()):
            self._apply_decay_and_prune_subject(s)
        return {s: list(facts) for s, facts in self.graph.items()}

    def remove_fact(self, subject: str, predicate: str, obj: str) -> bool:
        """Remove matching fact if exists; return True if removed."""
        if subject not in self.graph:
            return False
        removed = False
        newlist = []
        for (p, o, c, h, ts) in self.graph[subject]:
            if not (p == predicate and o == obj):
                newlist.append((p, o, c, h, ts))
            else:
                removed = True
        self.graph[subject] = newlist
        if removed:
            logger.info({"event": "PRAG_KG_FACT_REMOVED", "subject": subject, "predicate": predicate})
            if self.control_bus:
                self.control_bus.publish("PRAG_KG_FACT_REMOVED", {"subject": subject, "predicate": predicate})
        return removed

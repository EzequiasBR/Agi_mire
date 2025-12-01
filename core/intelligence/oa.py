# core/oa.py
"""
Organismo Analítico (OA) — raciocínio simbólico, KG e MVE (Motor de Validação Ética)

Função Chave no Ciclo (Passo 5):
    - validate_hypothesis(triple, certainty): Verifica coerência factual no KG e violação ética na Rule Base.
    - find_contradictions(): detecta inconsistências semânticas/vestigiais no KG.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import time
import json
import hashlib
import logging

import numpy as np

# Tenta reusar utilities, senão define as essenciais (compatibilidade)
try:
    # utils deve expor uma função de normalização (nome _normalize_vector no arquivo original)
    from ..services.utils import _normalize_vector  # pragma: no cover
except Exception:  # pragma: no cover
    def _normalize_vector(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float32)
        n = np.linalg.norm(v)
        if n <= 1e-12:
            return np.zeros_like(v)
        return v / n

try:
    # preferível ter uma função determinística de hash no utils
    from ..services.utils import deterministic_hash  # type: ignore  # pragma: no cover
except Exception:  # pragma: no cover
    def deterministic_hash(obj: Any) -> str:
        def _numpy_serializer(o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.floating, np.integer)):
                return float(o)
            raise TypeError(f"not serializable: {type(o)}")
        s = json.dumps(obj, sort_keys=True, default=_numpy_serializer, separators=(",", ":"))
        return hashlib.sha256(s.encode("utf-8")).hexdigest()


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s"))
        logger.addHandler(ch)
    logger.setLevel(logging.INFO)
    return logger


logger = setup_logger("OA")

DEFAULT_DIM = 768  # Manter consistente com o OL baseline

# -----------------------
# Helper utilities
# -----------------------
def _sha_to_seed(s: str) -> int:
    h = hashlib.sha256(str(s).encode("utf-8")).hexdigest()
    return int(h[:16], 16) % (2 ** 31 - 1)


def _deterministic_vector_from_text(text: str, dim: int = DEFAULT_DIM) -> np.ndarray:
    seed = _sha_to_seed(text)
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return _normalize_vector(v)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_n = _normalize_vector(a)
    b_n = _normalize_vector(b)
    if np.all(a_n == 0) or np.all(b_n == 0):
        return 0.0
    # clip to avoid tiny fp overflow
    return float(np.dot(a_n, b_n))


def _is_negation_token(token: str) -> bool:
    negs = {"não", "nao", "not", "never", "no", "não_e", "nãoé"}
    return token.strip().lower() in negs


def _safe_list_get(d: dict, k: str, default=None):
    v = d.get(k)
    return v if v is not None else default


# -----------------------
# Knowledge Graph storage
# -----------------------
class OA:
    def __init__(self, dim: int = DEFAULT_DIM):
        self.dim = int(dim)
        self.triples: Dict[str, Dict[str, Any]] = {}
        self.subject_index: Dict[str, List[str]] = {}
        self.object_index: Dict[str, List[str]] = {}
        self.next_id = 1

        # configurable PRM weights (mantidos para o método symbolize)
        self.prm_alpha = 0.4
        self.prm_beta = 0.25
        self.prm_gamma = 0.2
        self.prm_delta = 0.15

        # external references (opcionais, set by orchestrator)
        # simlog: should implement emit(event_name, payload)
        # hippocampus: should implement top_k(vector,k) and store_triple(...)
        # regvet: should implement check_vector(vector)->{"risk": "LOW|MEDIUM|HIGH", ...}
        # ppo: optional policy module
        # prag, oea may be attached externally and will be used if present
        self.simlog: Any = None
        self.hippocampus: Any = None
        self.regvet: Any = None
        self.ppo: Any = None
        self.prag: Any = None
        self.oea: Any = None

        # Base de Regras Éticas (MVE - Motor de Validação Ética)
        # Em um ambiente real, esta seria uma classe dedicada (RuleBase)
        self.ethical_rules: Dict[str, Tuple[bool, Any]] = {}
        self._setup_default_ethical_rules()

        logger.info("OA initialized (dim=%d)", self.dim)

    # -----------------------
    # KG management
    # -----------------------
    def _new_id(self) -> str:
        nid = f"t{self.next_id}"
        self.next_id += 1
        return nid

    def _standardize_meta(self, meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        meta = meta.copy() if meta else {}
        # ensure anchor_vector is normalized if present
        anchor = meta.get("anchor_vector")
        if anchor is not None:
            try:
                arr = np.asarray(anchor, dtype=np.float32)
                meta["anchor_vector"] = _normalize_vector(arr).tolist()
            except Exception:
                meta["anchor_vector"] = None
        meta.setdefault("certainty", float(meta.get("certainty", 1.0)))
        meta.setdefault("severity", float(meta.get("severity", 0.5)))
        meta.setdefault("active", bool(meta.get("active", True)))
        meta.setdefault("is_rule", bool(meta.get("is_rule", False)))
        return meta

    def _triple_hash(self, s: str, p: str, o: str, meta: Dict[str, Any]) -> str:
        payload = {"s": s, "p": p, "o": o, "meta": meta}
        return deterministic_hash(payload)

    def add_triple(self, s: str, p: str, o: str, meta: Optional[Dict[str, Any]] = None) -> str:
        # Lógica de adição e indexação (mantida, com meta padronizada)
        meta = self._standardize_meta(meta)
        tid = self._new_id()
        triple = {
            "s": str(s),
            "p": str(p),
            "o": str(o),
            "meta": meta,
            "ts": time.time(),
            "hash": self._triple_hash(s, p, o, meta)
        }
        self.triples[tid] = triple
        self.subject_index.setdefault(triple["s"], []).append(tid)
        self.object_index.setdefault(triple["o"], []).append(tid)
        logger.info("OA.add_triple: %s | %s | %s (id=%s)", s, p, o, tid)
        try:
            if self.simlog and hasattr(self.simlog, "emit"):
                self.simlog.emit("oa.triple", {"id": tid, "triple": triple})
        except Exception:
            logger.exception("OA.add_triple: simlog.emit failed")
        return tid

    def query_triples(self, subject: Optional[str] = None, predicate: Optional[str] = None, object: Optional[str] = None) -> List[Dict[str, Any]]:
        # Lógica de query (mantida)
        results: List[Dict[str, Any]] = []
        if subject is not None:
            ids = self.subject_index.get(subject, [])
            for tid in ids:
                t = self.triples.get(tid)
                if t and (predicate is None or t["p"] == predicate) and (object is None or t["o"] == object):
                    results.append(t)
            return results
        if object is not None:
            ids = self.object_index.get(object, [])
            for tid in ids:
                t = self.triples.get(tid)
                if t and (predicate is None or t["p"] == predicate):
                    results.append(t)
            return results
        for t in self.triples.values():
            if (subject is None or t["s"] == subject) and (predicate is None or t["p"] == predicate) and (object is None or t["o"] == object):
                results.append(t)
        return results

    # -----------------------
    # MVE: Motor de Validação Ética (guardrails)
    # -----------------------
    def _setup_default_ethical_rules(self):
        """
        Define as Restrições Éticas Fundamentais (Guardrails - C=1.0).
        """
        # Guardrail 1: Proibir o Predicado "causa_dano_irreversível" (MVE Nível 1.0)
        def check_irreversible_damage(s, p, o):
            # Retorna True se a regra NÃO for violada
            return not (p.lower() == "causa_dano_irreversível")

        # Guardrail 2: Proibir o Predicado "cria_arma_ilegal"
        def check_illegal_weapon(s, p, o):
            return not (p.lower() == "cria_arma_ilegal")

        # Regra 3 (Moderada): Aviso sobre Predicados de "risco_alto"
        def check_high_risk(s, p, o):
            return not (p.lower() == "risco_alto" and o.lower() == "incontrolável")

        self.ethical_rules["NoIrreversibleDamage"] = (True, check_irreversible_damage)  # True=Ética Crítica
        self.ethical_rules["NoIllegalWeapon"] = (True, check_illegal_weapon)
        self.ethical_rules["HighRiskWarning"] = (False, check_high_risk)  # False=Regra Lógica/Segurança

    def _check_ethical_violations(self, s: str, p: str, o: str) -> List[Dict[str, Any]]:
        """
        Verifica se a tripla viola as Restrições Éticas (Guardrails).
        Retorna uma lista de dicionários com info de violação:
            {"rule": name, "is_critical": bool, "message": str}
        """
        violations: List[Dict[str, Any]] = []
        triple = (s, p, o)

        for name, (is_critical, rule_func) in self.ethical_rules.items():
            try:
                # Se a regra retornar False, houve violação
                if not rule_func(*triple):
                    violations.append({"rule": name, "is_critical": bool(is_critical), "message": f"Rule {name} violated"})
            except Exception as e:
                logger.error("Erro ao processar regra %s: %s", name, e)
                violations.append({"rule": name, "is_critical": bool(is_critical), "message": f"Rule execution error: {e}"})

        return violations

    # -----------------------
    # Validação Racional (Passo 5 do Ciclo Coeso)
    # -----------------------
    def validate_hypothesis(self, hypothesis_triple: Tuple[str, str, str], initial_certainty: float) -> Tuple[bool, float, List[Dict[str, Any]]]:
        """
        Executa a Validação Racional da Tripla (Sim-Log V1.2) com integrações:
          - Checagem Ética via MVE local e via OEA se disponível
          - Modulação por Reg-Vet (risco vetorial)
          - Coerência KG (c_old) e combinação ponderada
          - Emissão para SimLog, PRAG e Hippocampus quando aplicável

        Args:
            hypothesis_triple: Tripla (S, P, O) gerada pelo Sim-Log/OL.
            initial_certainty: Certeza C base gerada pelo OEA/Sim-Log.

        Returns:
            (is_valid, final_certainty, list_of_violations)
        """
        s, p, o = hypothesis_triple
        # Emit: hypothesis received
        try:
            if self.simlog and hasattr(self.simlog, "emit"):
                self.simlog.emit("oa.hypothesis.received", {"triple": hypothesis_triple, "initial_certainty": float(initial_certainty)})
        except Exception:
            logger.exception("OA.validate_hypothesis: simlog.emit(received) failed")

        # Basic input validation
        if not isinstance(s, str) or not isinstance(p, str) or not isinstance(o, str):
            logger.error("OA.validate_hypothesis: invalid hypothesis types: %s", hypothesis_triple)
            return False, 0.0, [{"rule": "invalid_input", "is_critical": True, "message": "Hypothesis elements must be strings"}]

        # 1) Ethical checks (local MVE)
        violations = self._check_ethical_violations(s, p, o)
        if violations:
            # If any critical violation exists -> immediate block
            crits = [v for v in violations if v.get("is_critical")]
            try:
                if crits:
                    logger.warning("OA.validate_hypothesis: critical ethical violation %s for %s", crits, hypothesis_triple)
                    if self.simlog and hasattr(self.simlog, "emit"):
                        self.simlog.emit("oa.ethic.block", {"triple": hypothesis_triple, "violations": crits})
                else:
                    if self.simlog and hasattr(self.simlog, "emit"):
                        self.simlog.emit("oa.ethic.warn", {"triple": hypothesis_triple, "violations": violations})
            except Exception:
                logger.exception("OA.validate_hypothesis: simlog.emit(ethic) failed")

            if crits:
                # notify PRAG for governance / rollback if available
                try:
                    if self.prag and hasattr(self.prag, "request_rollback"):
                        self.prag.request_rollback({"source": "OA", "triple": hypothesis_triple, "reason": "critical_ethic_violation", "violations": crits})
                except Exception:
                    # best-effort notify; don't raise
                    logger.exception("OA.validate_hypothesis: prag.request_rollback failed")
                return False, 0.0, violations

        # 2) KG coherence check (c_old)
        c_old = 0.0
        related = self.query_triples(subject=s, predicate=p, object=o)
        for t in related:
            c_old = max(c_old, float(_safe_list_get(t.get("meta", {}), "certainty", 0.0)))

        # 3) Evaluate vector-based risks via Reg-Vet (if anchor vector available in context)
        adjusted_certainty = float(initial_certainty)
        anchor_vector: Optional[np.ndarray] = None
        # attempt to derive anchor vector from existing KG or deterministic mapping
        # prefer an anchor in KG if present (first match)
        if related:
            meta_anchor = related[0].get("meta", {}).get("anchor_vector")
            if meta_anchor is not None:
                try:
                    anchor_vector = np.asarray(meta_anchor, dtype=np.float32)
                    anchor_vector = _normalize_vector(anchor_vector)
                except Exception:
                    anchor_vector = None
        # else fallback: deterministic mapping of triple text
        if anchor_vector is None:
            try:
                txt = f"{s}|{p}|{o}"
                anchor_vector = _deterministic_vector_from_text(txt, dim=self.dim)
            except Exception:
                anchor_vector = None

        # consult regvet if available
        try:
            if self.regvet and hasattr(self.regvet, "check_vector") and anchor_vector is not None:
                rv = self.regvet.check_vector(anchor_vector)
                # expected rv to include at least {"risk":"LOW|MEDIUM|HIGH", "score":float}
                risk = str(rv.get("risk", "LOW")).upper() if isinstance(rv, dict) else "LOW"
                if risk == "HIGH":
                    adjusted_certainty *= 0.5
                elif risk == "MEDIUM":
                    adjusted_certainty *= 0.8
                # emit event
                try:
                    if self.simlog and hasattr(self.simlog, "emit"):
                        self.simlog.emit("oa.regvet_check", {"triple": hypothesis_triple, "regvet": rv})
                except Exception:
                    logger.exception("OA.validate_hypothesis: simlog.emit(regvet) failed")
        except Exception:
            logger.exception("OA.validate_hypothesis: regvet.check_vector failed")

        # 4) Consult OEA (ethics engine) if present for nuanced decisioning
        try:
            if self.oea and hasattr(self.oea, "process_cycle") and anchor_vector is not None:
                oea_meta = {"expected_dim": anchor_vector.shape[0], "source": "OA.validate_hypothesis"}
                oea_resp = self.oea.process_cycle(anchor_vector, oea_meta)
                # oea_resp might contain "violated", "gravity", "repulsion_vector"
                if isinstance(oea_resp, dict) and oea_resp.get("violated"):
                    # if OEA flags a violation with high gravity, reduce certainty or block
                    gravity = float(oea_resp.get("gravity", 0.0))
                    if gravity >= 0.9:
                        # critical ethical veto coming from specialized engine
                        try:
                            if self.simlog and hasattr(self.simlog, "emit"):
                                self.simlog.emit("oa.oea.critical", {"triple": hypothesis_triple, "oea": oea_resp})
                        except Exception:
                            logger.exception("OA.validate_hypothesis: simlog.emit(oea critical) failed")
                        # request PRAG rollback if available and block
                        try:
                            if self.prag and hasattr(self.prag, "request_rollback"):
                                self.prag.request_rollback({"source": "OA", "triple": hypothesis_triple, "reason": "oea_critical", "oea": oea_resp})
                        except Exception:
                            logger.exception("OA.validate_hypothesis: prag.request_rollback failed (oea)")
                        return False, 0.0, [{"rule": "oea_critical", "is_critical": True, "message": "OEA critical violation"}]
                    else:
                        # soften certainty proportional to gravity
                        adjusted_certainty *= max(0.0, 1.0 - gravity * 0.8)
                        try:
                            if self.simlog and hasattr(self.simlog, "emit"):
                                self.simlog.emit("oa.oea.warn", {"triple": hypothesis_triple, "oea": oea_resp})
                        except Exception:
                            logger.exception("OA.validate_hypothesis: simlog.emit(oea warn) failed")
        except Exception:
            logger.exception("OA.validate_hypothesis: calling OEA failed")

        # 5) Combine KG coherence (c_old) with adjusted certainty
        # if KG already strongly supports the triple, nudge toward c_old
        try:
            # weights: more weight to KG support as c_old increases
            w_kg = min(0.8, c_old)  # cap influence
            final_certainty = float((1.0 - w_kg) * adjusted_certainty + w_kg * c_old)
        except Exception:
            final_certainty = float(adjusted_certainty)

        # 6) Final validity decision
        # thresholding: require >0.1 and no critical violations detected earlier
        is_valid = (final_certainty > 0.1) and not any(v.get("is_critical") for v in violations)

        # 7) Emissões finais e persistência recomendada
        try:
            if self.simlog and hasattr(self.simlog, "emit"):
                self.simlog.emit("oa.validation.result", {"triple": hypothesis_triple, "is_valid": is_valid, "final_certainty": float(final_certainty), "violations": violations})
        except Exception:
            logger.exception("OA.validate_hypothesis: simlog.emit(result) failed")

        # If valid, suggest persistence to hippocampus and notify PRAG for governance check
        if is_valid:
            try:
                # persist to hippocampus if available
                if self.hippocampus and hasattr(self.hippocampus, "store_triple"):
                    store_meta = {"certainty": float(final_certainty), "severity": 0.0, "tags": ["oa_validated"], "anchor_vector": anchor_vector.tolist() if anchor_vector is not None else None}
                    # keep hippocampus interface flexible: attempt to call store_triple
                    try:
                        self.hippocampus.store_triple(s, p, o, meta=store_meta)
                        if self.simlog and hasattr(self.simlog, "emit"):
                            self.simlog.emit("oa.memory.persisted", {"triple": hypothesis_triple, "meta": store_meta})
                    except Exception:
                        # if hippocampus API different, try generic write_memory
                        if hasattr(self.hippocampus, "write_memory"):
                            self.hippocampus.write_memory({"s": s, "p": p, "o": o, "meta": store_meta})
                            if self.simlog and hasattr(self.simlog, "emit"):
                                self.simlog.emit("oa.memory.persisted", {"triple": hypothesis_triple, "meta": store_meta})
            except Exception:
                logger.exception("OA.validate_hypothesis: hippocampus persistence failed")

            # notify PRAG (best-effort)
            try:
                if self.prag and hasattr(self.prag, "check"):
                    self.prag.check("OA", {"triple": hypothesis_triple, "certainty": float(final_certainty)})
                elif self.prag and hasattr(self.prag, "notify"):
                    self.prag.notify("OA", {"triple": hypothesis_triple, "certainty": float(final_certainty)})
            except Exception:
                logger.exception("OA.validate_hypothesis: prag notification failed")

        return is_valid, float(final_certainty), violations

    # -----------------------
    # get_rules - para integração com RegVet
    # -----------------------
    def get_rules(self) -> List[Dict[str, Any]]:
        rules = []
        # Adiciona as regras éticas do OA para o RegVet (OEA)
        for name, (is_critical, rule_func) in self.ethical_rules.items():
            anchor_text = f"rule|{name}|{is_critical}"
            anchor_v = _deterministic_vector_from_text(anchor_text, dim=self.dim).tolist()

            rules.append({
                "rule_id": name,
                "s": name,
                "p": "is_ethical_guardrail" if is_critical else "is_logical_rule",
                "o": "True",
                "meta": {"is_critical": is_critical, "is_rule": True},
                "anchor_vector": anchor_v,
                "certainty": 1.0 if is_critical else 0.8,
                "severity": 1.0 if is_critical else 0.5
            })

        # Adiciona regras baseadas em triplas KG (se houver a flag 'is_rule')
        for tid, t in self.triples.items():
            meta = t.get("meta", {})
            if meta.get("active") and meta.get("is_rule"):
                rules.append({
                    "rule_id": tid,
                    "s": t["s"],
                    "p": t["p"],
                    "o": t["o"],
                    "meta": meta,
                    "anchor_vector": meta.get("anchor_vector"),
                    "certainty": float(meta.get("certainty", 1.0)),
                    "severity": float(meta.get("severity", 0.5))
                })
        return rules

    # -----------------------
    # Symbolization (Vector -> Symbolic) - Auxiliar do Sim-Log
    # -----------------------
    def symbolize(self, embedding: Any) -> Dict[str, Any]:
        """
        Mapeia o embedding para um pacote de hipótese simbólica (sym_pkg) usando PRM.
        Esta função é o coração do PRM (Preferred Relation Module) do OA,
        que o Sim-Log V1.2 consulta para a tradução.
        """
        emb = _normalize_vector(np.asarray(embedding, dtype=np.float32))
        candidates: List[Tuple[str, Dict[str, Any], float]] = []

        # 1) Semantic match: compare to anchor vectors in triples (if present)
        for tid, t in self.triples.items():
            if not t.get("meta", {}).get("active", True):
                continue
            anchor = t.get("meta", {}).get("anchor_vector")
            if anchor is not None:
                try:
                    anchor_v = np.asarray(anchor, dtype=np.float32)
                    sem = _cosine_similarity(emb, anchor_v)
                except Exception:
                    sem = 0.0
            else:
                proxy_text = f"{t['s']}|{t['p']}|{t['o']}"
                proxy_v = _deterministic_vector_from_text(proxy_text, dim=self.dim)
                sem = _cosine_similarity(emb, proxy_v)
            candidates.append((tid, t, float(sem)))

        # 2) Fallback (mantido)
        if not candidates:
            subj = f"entity_{hashlib.sha256(emb.tobytes()).hexdigest()[:8]}"
            pred = "is_related_to"
            obj = "unknown"
            tid = self.add_triple(subj, pred, obj, meta={"anchor_vector": emb.tolist(), "is_rule": False})
            triple = self.triples[tid]
            sym_pkg = {"triples": [triple], "conf": 0.5, "cot": ["generated_fallback"], "anchor_scores": [1.0]}
            try:
                if self.simlog and hasattr(self.simlog, "emit"):
                    self.simlog.emit("oa.reasoning", {"reason": "fallback_generation", "sym_pkg": sym_pkg})
            except Exception:
                logger.exception("OA.symbolize: simlog.emit failed on fallback")
            return sym_pkg

        # 3) Score candidates semantically and with PRM (mantido)
        scored: List[Tuple[str, Dict[str, Any], float]] = []
        for tid, t, sem in candidates:
            anchor_vec = t.get("meta", {}).get("anchor_vector")
            anchor_coherence = 0.0
            if anchor_vec is not None and self.hippocampus is not None:
                try:
                    top = self.hippocampus.top_k(np.asarray(anchor_vec, dtype=np.float32), k=1)
                    if top:
                        top_score = float(top[0][1])
                        anchor_coherence = top_score
                except Exception:
                    anchor_coherence = 0.0

            divergence = 0.0
            if anchor_vec is not None:
                divergence = 1.0 - _cosine_similarity(np.asarray(anchor_vec, dtype=np.float32), emb)

            conf_est = float(t.get("meta", {}).get("certainty", 1.0))

            # PRM combined score
            score = (self.prm_alpha * sem +
                     self.prm_beta * anchor_coherence +
                     self.prm_gamma * (1.0 - divergence) +
                     self.prm_delta * conf_est)
            scored.append((tid, t, float(score)))

        scored.sort(key=lambda x: (x[2], x[0]), reverse=True)

        # Build CoT (short)
        top_tid, top_triple, top_score = scored[0]
        cot_steps = [
            "identify_entities",
            "match_anchor_vectors" if top_triple.get("meta", {}).get("anchor_vector") else "use_proxy_semantics",
            "select_preferred_triple"
        ][:3]

        # Build sym_pkg
        sym_pkg = {
            "triples": [t for _, t, _ in scored],
            "preferred": top_triple,
            "preferred_id": top_tid,
            "scores": [s for _, _, s in scored],
            "conf": float(min(1.0, max(0.0, top_score))),
            "cot": cot_steps,
            "anchor_scores": [float(min(1.0, max(0.0, _))) for _, _, _ in [(x[0], x[1], x[2]) for x in scored]]
        }

        # emit simlog events
        try:
            if self.simlog and hasattr(self.simlog, "emit"):
                self.simlog.emit("oa.reasoning", {"preferred_id": top_tid, "preferred_triple": top_triple, "sym_pkg": sym_pkg})
                self.simlog.emit("oa.triple", {"preferred_id": top_tid, "preferred_triple": top_triple})
        except Exception:
            logger.exception("OA.symbolize: simlog.emit failed")

        return sym_pkg

    # -----------------------
    # Reconstruct embedding from symbolic package or triple
    # -----------------------
    def reconstruct_embedding(self, sym_pkg: Dict[str, Any]) -> np.ndarray:
        """
        Mapeia a hipótese simbólica de volta para o embedding (Vetor Lógico).
        """
        # Lógica original mantida.
        if not sym_pkg:
            return _deterministic_vector_from_text("empty", dim=self.dim)
        pref = sym_pkg.get("preferred") or (sym_pkg.get("triples") or [None])[0]
        if pref is None:
            return _deterministic_vector_from_text(json.dumps(sym_pkg, sort_keys=True), dim=self.dim)
        anchor = pref.get("meta", {}).get("anchor_vector")
        if anchor is not None:
            try:
                v = np.asarray(anchor, dtype=np.float32)
                return _normalize_vector(v)
            except Exception:
                logger.exception("OA.reconstruct_embedding: anchor vector invalid; fallback to deterministic")
        txt = f"{pref.get('s')}|{pref.get('p')}|{pref.get('o')}"
        return _deterministic_vector_from_text(txt, dim=self.dim)

    # -----------------------
    # Contradiction detection
    # -----------------------
    def find_contradictions(self, sensitivity: float = 0.75) -> List[Dict[str, Any]]:
        """
        Detecta contradições no KG.

        Heurísticas:
          - mesmo (s,p) com objetos diferentes;
          - predicados/oposições textuais (tokens de negação);
          - vetorial: âncoras com similaridade negativa forte (repulsão).

        Args:
            sensitivity: float em (0,1], quanto maior mais sensível (usa limiar -sensitivity para oposição vetorial)

        Returns:
            lista de dicionários descrevendo contradições e evidências
        """
        contradictions: List[Dict[str, Any]] = []
        # 1) Same (s,p) different o
        seen_sp: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
        for tid, t in self.triples.items():
            key = (t["s"], t["p"])
            seen_sp.setdefault(key, []).append((tid, t["o"]))
        for (s, p), vals in seen_sp.items():
            if len(vals) > 1:
                objects = [o for _, o in vals]
                if len(set(objects)) > 1:
                    contradictions.append({
                        "type": "same_sp_different_o",
                        "s": s,
                        "p": p,
                        "objects": objects,
                        "evidence": vals
                    })

        # 2) textual negation conflicts (cheap)
        # for each triple, look for reversed assertion with negation tokens
        for tid, t in self.triples.items():
            o_tok = str(t["o"]).lower()
            # naive tokenization
            tokens = set(o_tok.replace("_", " ").split())
            if any(_is_negation_token(tok) for tok in tokens):
                # search for triple with same s and p but object without negation form
                s = t["s"]; p = t["p"]
                for tid2 in self.subject_index.get(s, []):
                    t2 = self.triples.get(tid2)
                    if not t2 or tid2 == tid:
                        continue
                    if t2["p"] == p:
                        # consider contradiction
                        contradictions.append({
                            "type": "textual_negation",
                            "s": s,
                            "p": p,
                            "object_neg": t["o"],
                            "object_pos": t2["o"],
                            "evidence": (tid, tid2)
                        })

        # 3) vector opposition detection
        # build list of (tid, anchor_vec)
        vec_list: List[Tuple[str, np.ndarray]] = []
        for tid, t in self.triples.items():
            anchor = t.get("meta", {}).get("anchor_vector")
            if anchor is None:
                continue
            try:
                v = np.asarray(anchor, dtype=np.float32)
                vn = _normalize_vector(v)
                if np.all(vn == 0):
                    continue
                vec_list.append((tid, vn))
            except Exception:
                continue
        # compare pairwise (O(n^2) but KG expected smaller; sensitivity maps to negative threshold)
        neg_threshold = -1.0 * float(min(max(sensitivity, 0.1), 0.99))
        n = len(vec_list)
        for i in range(n):
            tid_i, vi = vec_list[i]
            for j in range(i + 1, n):
                tid_j, vj = vec_list[j]
                sim = _cosine_similarity(vi, vj)
                if sim <= neg_threshold:
                    contradictions.append({
                        "type": "vector_opposition",
                        "pairs": (tid_i, tid_j),
                        "similarity": sim,
                        "evidence": (self.triples.get(tid_i), self.triples.get(tid_j))
                    })

        # deduplicate (simple)
        seen = set()
        uniq: List[Dict[str, Any]] = []
        for c in contradictions:
            key = (c.get("type"), json.dumps(c.get("evidence", {}), sort_keys=True, default=str))
            if key in seen:
                continue
            seen.add(key)
            uniq.append(c)

        # emit event
        try:
            if self.simlog and hasattr(self.simlog, "emit"):
                self.simlog.emit("oa.contradictions.detected", {"count": len(uniq), "contradictions": uniq})
        except Exception:
            logger.exception("OA.find_contradictions: simlog.emit failed")

        return uniq

    # -----------------------
    # External refs e Serialização
    # -----------------------
    def set_external_refs(self, simlog: Any = None, hippocampus: Any = None, regvet: Any = None, ppo: Any = None, prag: Any = None, oea: Any = None) -> None:
        # mantemos compatibilidade com a assinatura anterior; permitem injetar prag e oea tambem
        self.simlog = simlog
        self.hippocampus = hippocampus
        self.regvet = regvet
        self.ppo = ppo
        self.prag = prag
        self.oea = oea

    def serialize_state(self) -> Dict[str, Any]:
        state = {
            "dim": int(self.dim),
            "triples": self.triples,
            "subject_index": self.subject_index,
            "object_index": self.object_index,
            "next_id": int(self.next_id),
            "prm_weights": {
                "alpha": float(self.prm_alpha),
                "beta": float(self.prm_beta),
                "gamma": float(self.prm_gamma),
                "delta": float(self.prm_delta)
            },
            # Inclui regras éticas (serialização simplificada)
            "ethical_rules_keys": list(self.ethical_rules.keys())
        }
        return state

    def load_state(self, state: Dict[str, Any]) -> None:
        if not state:
            return
        self.dim = int(state.get("dim", self.dim))
        self.triples = dict(state.get("triples", {}))
        self.subject_index = dict(state.get("subject_index", {}))
        self.object_index = dict(state.get("object_index", {}))
        self.next_id = int(state.get("next_id", self.next_id))
        w = state.get("prm_weights", {})
        self.prm_alpha = float(w.get("alpha", self.prm_alpha))
        self.prm_beta = float(w.get("beta", self.prm_beta))
        self.prm_gamma = float(w.get("gamma", self.prm_gamma))
        self.prm_delta = float(w.get("delta", self.prm_delta))

        # Recria as regras éticas, pois a função não é serializada.
        self._setup_default_ethical_rules()

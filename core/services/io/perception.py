# core/services/perception_api.py
"""
PerceptionAPI V4.5 — Multimodal perception with persistence, PCVS integration,
adaptive configuration, accumulated metrics and safe-mode transitions.

Correções da versão 4.5:
- _build_meta adicionado e integrado de forma padronizada
- ajustes em safe_mode + multimodal
- correções de serialização de snapshot
- correções estruturais no fluxo de publish
- estabilidade geral + consistência interna
"""

from __future__ import annotations
import time
import uuid
import json
import logging
import asyncio
from typing import Any, Dict, Optional, Tuple
from threading import Lock

# try imports; fallback to mocks
try:
    from ..utils import setup_logger
    from core.orchestration.control_bus import ControlBus, SystemEvents
    from ..security import Security
    from ..pcvs import PCVS
    from .multimodal.adapters.audio_bridge import AudioBridge
    from .multimodal.adapters.vision_bridge import VisionBridge
    from core.memory.hippocampus import save_json, load_json
except Exception:
    import hashlib

    def setup_logger(name: str) -> logging.Logger:
        logger = logging.getLogger(name)
        if not logger.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s"))
            logger.addHandler(ch)
        logger.setLevel(logging.INFO)
        return logger

    def save_json(path: str, data: Dict[str, Any]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    def load_json(path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    class SystemEvents:
        NEW_MEMORY_STORED = "NEW_MEMORY_STORED"
        STATE_PERSISTED = "STATE_PERSISTED"
        CONFIG_ADJUSTED = "CONFIG_ADJUSTED"
        INTEGRITY_VIOLATION = "INTEGRITY_VIOLATION"

    class ControlBus:
        def __init__(self):
            self.subscriptions = {}
            self.events_published = []

        def subscribe(self, event: str, handler):
            self.subscriptions.setdefault(event, []).append(handler)

        def unsubscribe(self, event: str, handler):
            if event in self.subscriptions and handler in self.subscriptions[event]:
                self.subscriptions[event].remove(handler)

        async def publish(self, event_type: str, payload: Dict[str, Any], source_module: str = "unknown"):
            self.events_published.append({"event": event_type, "payload": payload})
            handlers = self.subscriptions.get(event_type, [])
            for h in handlers:
                try:
                    if asyncio.iscoroutinefunction(h):
                        await h(payload)
                    else:
                        loop = asyncio.get_running_loop()
                        await loop.run_in_executor(None, h, payload)
                except Exception:
                    logging.exception("ControlBus mock handler failed")

    class Security:
        def sanitize_input(self, input_data: Any) -> Tuple[bool, str]:
            s = str(input_data).strip()
            if "cpf" in s.lower():
                return False, "PII detected"
            return True, s

        def hash_state(self, data: Any) -> str:
            return hashlib.sha256(str(data).encode()).hexdigest()

    class PCVS:
        def __init__(self, control_bus: ControlBus, base_dir: str = "pcvs_demo"):
            self.control_bus = control_bus
            self.base_dir = base_dir

        async def persist_snapshot(self, snapshot_data: Dict[str, Any], reason: str) -> Optional[str]:
            sha = hashlib.sha256(
                json.dumps(snapshot_data, sort_keys=True, default=str).encode()
            ).hexdigest()
            path = f"{self.base_dir}_{sha[:8]}.json"
            await asyncio.to_thread(save_json, path, {"reason": reason, "snapshot": snapshot_data})
            try:
                await self.control_bus.publish(
                    SystemEvents.STATE_PERSISTED,
                    {"hash": sha, "reason": reason},
                    source_module="PCVS",
                )
            except Exception:
                pass
            return sha

    class AudioBridge:
        def __init__(self):
            logging.getLogger("PerceptionAPI").warning("Mock AudioBridge in use")

        def transcribe(self, data: bytes, fmt: str):
            time.sleep(0.02)
            return "simulated transcription", 0.92

    class VisionBridge:
        def __init__(self):
            logging.getLogger("PerceptionAPI").warning("Mock VisionBridge in use")

        def process_image(self, data: bytes, fmt: str):
            time.sleep(0.05)
            import numpy as _np
            return _np.zeros(768), ["mock_tag"]


logger = setup_logger("PerceptionAPIV4.5")

DEFAULT_MAX_INPUT_LENGTH = 4096


class PerceptionAPI:
    def __init__(
        self,
        security_service: Security,
        control_bus: ControlBus,
        pcvs: Optional[PCVS] = None,
        audio_bridge: Optional[AudioBridge] = None,
        vision_bridge: Optional[VisionBridge] = None,
        max_input_length: int = DEFAULT_MAX_INPUT_LENGTH,
    ):
        self.sec = security_service
        self.cb = control_bus
        self.pcvs = pcvs
        self.audio_bridge = audio_bridge or AudioBridge()
        self.vision_bridge = vision_bridge or VisionBridge()

        self.max_input_length = int(max_input_length)
        self.safe_mode = False
        self.safe_mode_reason = None

        self._lock = Lock()

        self.metrics = {
            "total_inputs": 0,
            "rejected_inputs": 0,
            "per_source": {"text": 0, "audio": 0, "image": 0},
            "latency_ms": {"count": 0, "sum": 0.0, "min": None, "max": None},
            "integrity_failures": 0,
            "pii_rejections": 0,
        }

        self.adaptation_policy = {
            "safe_mode_enabled": True,
            "pii_rate_threshold": 0.2,
            "integrity_failure_threshold": 1,
        }

        try:
            self.cb.subscribe(SystemEvents.CONFIG_ADJUSTED, self.handle_adaptation_update)
        except Exception:
            logger.debug("Failed to subscribe to CONFIG_ADJUSTED")

        self.last_snapshot_hash = None
        self.last_snapshot_ts = None

        logger.info("PerceptionAPI V4.5 initialized.")

    # --------------------------------------
    # helper: build meta
    # --------------------------------------
    def _build_meta(self, msg: str, source: str, start: float, processed: bool) -> Dict[str, Any]:
        end = time.perf_counter()
        latency = (end - start) * 1000.0
        self._update_latency_metrics(latency)
        uuid_ = str(uuid.uuid4())
        meta = {
            "uuid": uuid_,
            "timestamp": time.time(),
            "source_type": source,
            "processed_text": msg if processed else None,
            "processed": processed,
            "processed_hash": self.sec.hash_state(msg),
            "latency_ms": latency,
            "context": {},
            "metrics": {
                "total_inputs": self.metrics["total_inputs"],
                "rejected_inputs": self.metrics["rejected_inputs"],
                "per_source": dict(self.metrics["per_source"]),
                **self._get_latency_summary(),
            },
        }
        return meta

    # --------------------------------------
    def _update_latency_metrics(self, lat_ms: float) -> None:
        with self._lock:
            m = self.metrics["latency_ms"]
            m["count"] += 1
            m["sum"] += lat_ms
            m["min"] = lat_ms if m["min"] is None else min(m["min"], lat_ms)
            m["max"] = lat_ms if m["max"] is None else max(m["max"], lat_ms)

    def _get_latency_summary(self) -> Dict[str, Any]:
        m = self.metrics["latency_ms"]
        avg = m["sum"] / m["count"] if m["count"] > 0 else None
        return {"count": m["count"], "avg_ms": avg, "min_ms": m["min"], "max_ms": m["max"]}

    def _increment_source(self, source: str):
        with self._lock:
            self.metrics["total_inputs"] += 1
            self.metrics["per_source"][source] += 1

    def _register_rejection(self, kind: str):
        with self._lock:
            self.metrics["rejected_inputs"] += 1
            if kind == "pii":
                self.metrics["pii_rejections"] += 1

    def _maybe_update_safe_mode(self):
        with self._lock:
            total = max(1, self.metrics["total_inputs"])
            pii_rate = self.metrics["pii_rejections"] / total

            if (
                self.adaptation_policy["safe_mode_enabled"]
                and pii_rate >= self.adaptation_policy["pii_rate_threshold"]
            ):
                if not self.safe_mode:
                    self.safe_mode = True
                    self.safe_mode_reason = f"pii_rate={pii_rate:.3f}"
                    logger.warning("Safe mode ENABLED: %s", self.safe_mode_reason)

            if self.metrics["integrity_failures"] >= self.adaptation_policy["integrity_failure_threshold"]:
                if not self.safe_mode:
                    self.safe_mode = True
                    self.safe_mode_reason = "integrity_failures"
                    logger.warning("Safe mode ENABLED (integrity failures)")

    # --------------------------------------
    # main pipeline
    # --------------------------------------
    async def perceive(
        self,
        input_data: Any,
        source_type: str = "text",
        file_format: Optional[str] = None,
        context_meta: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        start = time.perf_counter()
        source = source_type.lower()
        context_meta = context_meta or {}

        # SAFE MODE: multimodal restrito
        if self.safe_mode and source in ("audio", "image"):
            msg = "[SAFE_MODE] multimodal processing deferred"
            meta = self._build_meta(msg, source, start, processed=False)
            await self._publish_new_memory(meta)
            return msg, meta

        # valida entrada
        if source == "text":
            if not isinstance(input_data, str):
                raise TypeError("Text input must be str")
            raw = input_data
        elif source in ("audio", "image"):
            if not isinstance(input_data, (bytes, bytearray)):
                raise TypeError("Binary input expected for audio/image")
            raw = input_data
        else:
            raise ValueError(f"Unknown source_type: {source_type}")

        self._increment_source(source)

        # sanitização texto
        if source == "text":
            is_ok, out = self.sec.sanitize_input(raw)
            if not is_ok:
                self._register_rejection("pii")
                self._maybe_update_safe_mode()

                processed = False
                processed_text = f"[REJEIÇÃO DE SEGURANÇA]: {out}"
            else:
                processed = True
                processed_text = out[: self.max_input_length]
        else:
            processed = True
            processed_text = ""

        extra_meta: Dict[str, Any] = {}

        # multimodal
        if source == "audio" and processed:
            try:
                if asyncio.iscoroutinefunction(self.audio_bridge.transcribe):
                    txt, conf = await self.audio_bridge.transcribe(raw, file_format or "wav")
                else:
                    loop = asyncio.get_running_loop()
                    txt, conf = await loop.run_in_executor(None, self.audio_bridge.transcribe, raw, file_format or "wav")

                processed_text = f"[ÁUDIO TRANSCRITO]: {txt}"
                extra_meta["stt_confidence"] = float(conf)
                extra_meta["transcribed_text"] = txt
            except Exception as e:
                processed = False
                processed_text = f"[ERROR: audio processing failed: {e}]"

        elif source == "image" and processed:
            try:
                if asyncio.iscoroutinefunction(self.vision_bridge.process_image):
                    emb, tags = await self.vision_bridge.process_image(raw, file_format or "jpg")
                else:
                    loop = asyncio.get_running_loop()
                    emb, tags = await loop.run_in_executor(None, self.vision_bridge.process_image, raw, file_format or "jpg")

                emb_ser = emb.tolist() if hasattr(emb, "tolist") else emb

                processed_text = f"[IMAGEM DESCRITA]: Tags: {', '.join(tags)}"
                extra_meta["image_tags"] = tags
                extra_meta["vision_embedding_preview"] = (emb_ser[:8] if isinstance(emb_ser, list) else None)
            except Exception as e:
                processed = False
                processed_text = f"[ERROR: image processing failed: {e}]"

        # hash
        try:
            input_hash = await asyncio.get_running_loop().run_in_executor(None, self.sec.hash_state, processed_text)
        except Exception:
            input_hash = "HASH_ERROR"

        end = time.perf_counter()
        latency = (end - start) * 1000.0
        self._update_latency_metrics(latency)

        meta = {
            "uuid": str(uuid.uuid4()),
            "timestamp": time.time(),
            "source_type": source,
            "processed_text": processed_text if processed else None,
            "processed": processed,
            "processed_hash": input_hash,
            "latency_ms": latency,
            "sanitized": (source != "text") or (processed and "REJEIÇÃO" not in processed_text),
            "context": {**context_meta, **extra_meta},
            "metrics": {
                "total_inputs": self.metrics["total_inputs"],
                "rejected_inputs": self.metrics["rejected_inputs"],
                "per_source": dict(self.metrics["per_source"]),
                **self._get_latency_summary(),
            },
        }

        await self._publish_new_memory(meta)

        # persist snapshot
        if self.pcvs:
            try:
                reason = f"perception_snapshot_{int(time.time())}"
                sha = await self.pcvs.persist_snapshot(self.snapshot_state(), reason)
                self.last_snapshot_hash = sha
                self.last_snapshot_ts = time.time()
            except Exception:
                logger.exception("PCVS persist failed")
        else:
            try:
                await self.cb.publish(
                    SystemEvents.STATE_PERSISTED,
                    {"component": "PerceptionAPI", "state": self.snapshot_state()},
                    source_module="PerceptionAPI",
                )
            except Exception:
                logger.debug("Fallback STATE_PERSISTED publish failed")

        self._maybe_update_safe_mode()

        return processed_text, meta

    # --------------------------------------
    async def _publish_new_memory(self, meta: Dict[str, Any]):
        try:
            await self.cb.publish(SystemEvents.NEW_MEMORY_STORED, meta, source_module="PerceptionAPI")
        except Exception:
            logger.exception("publish NEW_MEMORY_STORED failed")

    # --------------------------------------
    async def handle_adaptation_update(self, payload: Dict[str, Any]) -> None:
        try:
            if "max_input_length" in payload:
                self.max_input_length = int(payload["max_input_length"])

            if "safe_mode_enabled" in payload:
                self.adaptation_policy["safe_mode_enabled"] = bool(payload["safe_mode_enabled"])

            if "pii_rate_threshold" in payload:
                self.adaptation_policy["pii_rate_threshold"] = float(payload["pii_rate_threshold"])

            if "integrity_failure_threshold" in payload:
                self.adaptation_policy["integrity_failure_threshold"] = int(payload["integrity_failure_threshold"])

            if "force_safe_mode" in payload:
                forced = bool(payload["force_safe_mode"])
                self.safe_mode = forced
                self.safe_mode_reason = "forced_by_policy" if forced else None
        except Exception:
            logger.exception("Adaptation update failed")

    # --------------------------------------
    def snapshot_state(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "max_input_length": self.max_input_length,
                "safe_mode": self.safe_mode,
                "safe_mode_reason": self.safe_mode_reason,
                "metrics": json.loads(json.dumps(self.metrics, default=str)),
                "adaptation_policy": dict(self.adaptation_policy),
                "last_snapshot_hash": self.last_snapshot_hash,
                "last_snapshot_ts": self.last_snapshot_ts,
            }

    def load_state(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        with self._lock:
            self.max_input_length = int(state.get("max_input_length", self.max_input_length))
            self.safe_mode = bool(state.get("safe_mode", self.safe_mode))
            self.safe_mode_reason = state.get("safe_mode_reason", self.safe_mode_reason)
            self.metrics = state.get("metrics", self.metrics)
            self.adaptation_policy = state.get("adaptation_policy", self.adaptation_policy)
            self.last_snapshot_hash = state.get("last_snapshot_hash", self.last_snapshot_hash)
            self.last_snapshot_ts = state.get("last_snapshot_ts", self.last_snapshot_ts)
            logger.info("State restored from snapshot")

    # --------------------------------------
    def get_state(self) -> Dict[str, Any]:
        s = self.snapshot_state()
        s["latency_summary"] = self._get_latency_summary()
        return s

    # --------------------------------------
    async def start(self):
        try:
            self.cb.subscribe(SystemEvents.CONFIG_ADJUSTED, self.handle_adaptation_update)
        except Exception:
            pass
        logger.info("PerceptionAPI started.")

    async def stop(self):
        try:
            self.cb.unsubscribe(SystemEvents.CONFIG_ADJUSTED, self.handle_adaptation_update)
        except Exception:
            pass
        logger.info("PerceptionAPI stopped.")


# -------------------------
# Demo / quick test (runs when module executed)
# -------------------------
if __name__ == "__main__":
    import asyncio, hashlib

    async def demo():
        cb = ControlBus()
        sec = Security()
        pcvs = PCVS(cb)  # mock pcvs
        api = PerceptionAPI(security_service=sec, control_bus=cb, pcvs=pcvs)

        # start
        await api.start()

        # text ok
        txt, meta = await api.perceive("Olá, tudo bem?", source_type="text", context_meta={"user": "demo"})
        print("TEXT OK ->", txt, meta["latency_ms"])

        # text PII -> should be rejected
        txt2, meta2 = await api.perceive("Meu CPF é 123.456.789-00", source_type="text")
        print("TEXT PII ->", txt2, meta2["sanitized"])

        # audio (mock)
        audio_bytes = b"\x00" * 200
        txt3, meta3 = await api.perceive(audio_bytes, source_type="audio")
        print("AUDIO ->", txt3, meta3["latency_ms"])

        # image (mock)
        img_bytes = b"\x00" * 500
        txt4, meta4 = await api.perceive(img_bytes, source_type="image")
        print("IMAGE ->", txt4, meta4["latency_ms"])

        # adapt config: reduce max length and force safe mode off
        await cb.publish(SystemEvents.CONFIG_ADJUSTED, {"max_input_length": 32, "force_safe_mode": False}, source_module="demo")

        # persist snapshot explicitly
        snap = api.snapshot_state()
        sha = await pcvs.persist_snapshot(snap, "manual_demo_snapshot")
        print("Snapshot persisted:", sha)

        # view state
        print("State:", json.dumps(api.get_state(), indent=2, default=str))

        await api.stop()

    asyncio.run(demo())
# await asyncio.to_thread(save_json, path, {"reason": reason, "snapshot": snapshot_data})
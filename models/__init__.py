"""
models/__init__.py
~~~~~~~~~~~~~~~~~~
Model registry and subprocess-based model proxy.

Each model runs inside its own conda environment.  The coordinator
(run_pipeline.py) never imports model-specific packages; instead it
spawns a worker subprocess using the per-model conda Python binary and
communicates over stdin/stdout with newline-delimited JSON.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry: model name → (module_path, class_name)
# Used by model_worker.py (runs inside conda env) to import the right class.
# ---------------------------------------------------------------------------
MODEL_REGISTRY: Dict[str, tuple] = {
    "CosyVoice3": ("models.cosyvoice_model", "CosyVoiceModel"),
    "Qwen3TTS":   ("models.qwen3_tts_model",  "Qwen3TTSModel"),
    "VoxCPM":     ("models.voxcpm_model",      "VoxCPMModel"),
    "MossTTS":    ("models.moss_tts_model",    "MossTTSModel"),
}


# ---------------------------------------------------------------------------
# Subprocess proxy
# ---------------------------------------------------------------------------

class SubprocessModelProxy:
    """
    Wraps a TTS model that lives inside a separate conda environment.

    Lifecycle
    ---------
    1. ``ensure_loaded()``  — spawns the conda-env Python running
       ``model_worker.py``, sends the model config, waits for "loaded".
    2. ``generate()``       — sends a JSON generate request, reads the
       audio temp-file path from the JSON response, loads & returns the
       numpy waveform.
    3. ``close()``          — sends {"action":"stop"}, waits for the
       worker process to exit.
    """

    def __init__(
        self,
        name: str,
        model_cfg: Dict[str, Any],
        conda_python: str,
    ) -> None:
        self.MODEL_NAME: str = name
        self._model_cfg = model_cfg
        self._conda_python = conda_python
        self._proc: Optional[subprocess.Popen] = None
        self._sample_rate: int = 22_050
        self._supported_languages: List[str] = model_cfg.get("languages", [])
        self._is_loaded: bool = False

    # ------------------------------------------------------------------ #
    # BaseTTSModel-compatible interface                                    #
    # ------------------------------------------------------------------ #

    @property
    def name(self) -> str:
        return self.MODEL_NAME

    def supports_language(self, lang: str) -> bool:
        return lang in self._supported_languages

    def ensure_loaded(self) -> None:
        if self._is_loaded:
            return
        logger.info("[%s] Spawning worker in %s …", self.MODEL_NAME, self._conda_python)
        worker = Path(__file__).parent.parent / "model_worker.py"
        env = os.environ.copy()
        device = str(self._model_cfg.get("device", "")).strip().lower()
        if device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        elif device.startswith("cuda:"):
            if self.MODEL_NAME == "MossTTS":
                extra = self._model_cfg.get("extra", {}) or {}
                visible_devices: List[str] = []

                def _add_cuda(dev: Any) -> None:
                    dev = str(dev).strip().lower()
                    if dev.startswith("cuda:"):
                        idx = dev.split(":", 1)[1]
                        if idx not in visible_devices:
                            visible_devices.append(idx)

                _add_cuda(device)
                _add_cuda(extra.get("audio_tokenizer_device", ""))
                for gpu_id, budget in (extra.get("max_memory", {}) or {}).items():
                    budget_str = str(budget).strip().lower()
                    if budget_str and not budget_str.startswith("0"):
                        idx = str(gpu_id).strip()
                        if idx not in visible_devices:
                            visible_devices.append(idx)

                if visible_devices:
                    env["CUDA_VISIBLE_DEVICES"] = ",".join(visible_devices)
            else:
                # Many third-party model loaders ignore explicit cuda:N args and
                # just use "cuda". Restricting visible devices at subprocess spawn
                # time makes their internal cuda:0 map to the configured GPU.
                env["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]
        self._proc = subprocess.Popen(
            [self._conda_python, str(worker)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,   # inherit — worker logs go to the same terminal
            text=True,
            bufsize=1,
            env=env,
        )
        # Send config as first line
        self._send({"model_name": self.MODEL_NAME, "model_cfg": self._model_cfg})
        # Wait for loaded confirmation
        resp = self._recv()
        if resp.get("status") != "loaded":
            raise RuntimeError(
                f"[{self.MODEL_NAME}] Worker failed to load: {resp}"
            )
        self._sample_rate = resp.get("sample_rate", 22_050)
        self._is_loaded = True
        logger.info("[%s] Worker ready.  sample_rate=%d Hz.", self.MODEL_NAME, self._sample_rate)

    def generate(
        self,
        text: str,
        ref_audio_path: str,
        ref_text: str,
        language: str,
    ) -> np.ndarray:
        self._send({
            "action": "generate",
            "text": text,
            "ref_audio_path": ref_audio_path,
            "ref_text": ref_text,
            "language": language,
        })
        resp = self._recv()
        if resp.get("status") == "error":
            raise RuntimeError(resp.get("message", "unknown worker error"))
        audio_path = resp["audio_path"]
        try:
            wav, _ = sf.read(audio_path, dtype="float32")
        finally:
            try:
                os.unlink(audio_path)
            except OSError:
                pass
        return wav

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def close(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                self._send({"action": "stop"})
                self._proc.wait(timeout=10)
            except Exception:
                self._proc.kill()

    def __repr__(self) -> str:
        return f"SubprocessModelProxy({self.MODEL_NAME!r}, env={self._conda_python!r})"

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _send(self, obj: dict) -> None:
        self._proc.stdin.write(json.dumps(obj) + "\n")
        self._proc.stdin.flush()

    def _recv(self) -> dict:
        line = self._proc.stdout.readline()
        if not line:
            rc = self._proc.poll()
            raise RuntimeError(
                f"[{self.MODEL_NAME}] Worker process exited unexpectedly (rc={rc})"
            )
        return json.loads(line.strip())


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(name: str, model_cfg: Dict[str, Any]) -> SubprocessModelProxy:
    """
    Build a SubprocessModelProxy for *name* using the conda Python declared
    in *model_cfg['conda_python']*.

    Raises
    ------
    ValueError  if the model name is not in MODEL_REGISTRY.
    KeyError    if 'conda_python' is missing from model_cfg.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'.  Known: {list(MODEL_REGISTRY)}"
        )
    conda_python = model_cfg.get("conda_python")
    if not conda_python:
        raise KeyError(
            f"Model '{name}' is missing 'conda_python' in config.yaml. "
            f"Add it under models.{name}.conda_python"
        )
    return SubprocessModelProxy(name=name, model_cfg=model_cfg, conda_python=conda_python)

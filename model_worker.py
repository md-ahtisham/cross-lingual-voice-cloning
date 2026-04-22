#!/usr/bin/env python
"""
model_worker.py
~~~~~~~~~~~~~~~
Subprocess worker that runs INSIDE a model-specific conda environment.

The coordinator (run_pipeline.py) spawns this script using the conda
env's Python binary, e.g.:

    /Data/deepakkumar/miniconda3/envs/cosyvoice/bin/python model_worker.py

Protocol (newline-delimited JSON over stdin/stdout)
----------------------------------------------------
1. Coordinator sends one JSON config line on stdin:
       {"model_name": "CosyVoice3", "model_cfg": {...}}

2. Worker imports the model class, calls load_model(), then responds:
       {"status": "loaded", "sample_rate": 22050}
   or on failure:
       {"status": "error", "message": "...", "traceback": "..."}

3. Coordinator sends generate requests (one per line):
       {"action": "generate", "text": "...", "ref_audio_path": "...",
        "ref_text": "...", "language": "fr"}

4. Worker calls model.generate(), saves the waveform to a temp WAV,
   and responds:
       {"status": "ok", "audio_path": "/tmp/xxx.wav", "sample_rate": 22050}
   or on error:
       {"status": "error", "message": "...", "traceback": "..."}

5. Coordinator sends {"action": "stop"} to shut down cleanly.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import sys
import tempfile
import traceback
from pathlib import Path

# ── Redirect stdout → stderr BEFORE any imports print anything ────────────
# This ensures library warnings/prints (e.g. transformers FutureWarning)
# never pollute the JSON IPC pipe.  All intentional JSON goes through
# _IPC (the saved original stdout fd).
_IPC = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
sys.stdout.flush()
os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
# Now sys.stdout and print() both go to stderr; _IPC is the real pipe.

# ── Add the project root to sys.path so "models.*" can be imported ────────
_PROJ_ROOT = str(Path(__file__).parent.resolve())
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

# ── Model name → (module, class) mapping ──────────────────────────────────
_MODEL_CLASS_MAP = {
    "CosyVoice3": ("models.cosyvoice_model", "CosyVoiceModel"),
    "Qwen3TTS":   ("models.qwen3_tts_model",  "Qwen3TTSModel"),
    "VoxCPM":     ("models.voxcpm_model",      "VoxCPMModel"),
    "MossTTS":    ("models.moss_tts_model",    "MossTTSModel"),
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [WORKER/%(levelname)s] %(name)s — %(message)s",
    stream=sys.stderr,   # keep stdout clean for JSON protocol
)
logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────

def _send(obj: dict) -> None:
    _IPC.write(json.dumps(obj) + "\n")
    _IPC.flush()


def _recv() -> dict:
    line = sys.stdin.readline()
    if not line:
        raise EOFError("stdin closed unexpectedly")
    return json.loads(line.strip())


def _build_model(model_name: str, model_cfg: dict):
    """Import and instantiate the model class for *model_name*."""
    if model_name not in _MODEL_CLASS_MAP:
        raise ValueError(
            f"Unknown model '{model_name}'.  Known: {list(_MODEL_CLASS_MAP)}"
        )
    module_path, class_name = _MODEL_CLASS_MAP[model_name]
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    init_kwargs = dict(
        model_path=model_cfg.get("model_path", ""),
        device=model_cfg.get("device", "cuda:0"),
        languages=model_cfg.get("languages"),
    )
    # Pass model-specific extra kwargs
    init_kwargs.update(model_cfg.get("extra", {}))
    # CosyVoice3 needs repo_path at top level
    if "repo_path" in model_cfg:
        init_kwargs["repo_path"] = model_cfg["repo_path"]

    return cls(**init_kwargs)


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    # Step 1 — receive config
    try:
        config = _recv()
    except Exception as exc:
        _send({"status": "error", "message": f"Failed to read config: {exc}"})
        sys.exit(1)

    model_name = config.get("model_name", "")
    model_cfg  = config.get("model_cfg", {})

    # Step 2 — build and load model
    try:
        model = _build_model(model_name, model_cfg)
        model.load_model()
        sample_rate = getattr(model, "sample_rate", 22_050)
        _send({"status": "loaded", "sample_rate": sample_rate})
        logger.info("[%s] Model loaded.  sample_rate=%d Hz.", model_name, sample_rate)
    except Exception as exc:
        _send({
            "status": "error",
            "message": str(exc),
            "traceback": traceback.format_exc(),
        })
        sys.exit(1)

    # Step 3 — serve generate requests
    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        try:
            req = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            _send({"status": "error", "message": f"JSON parse error: {exc}"})
            continue

        action = req.get("action", "generate")

        if action == "stop":
            logger.info("[%s] Received stop signal.  Exiting.", model_name)
            break

        # Generate request
        try:
            wav = model.generate(
                text=req["text"],
                ref_audio_path=req["ref_audio_path"],
                ref_text=req.get("ref_text", ""),
                language=req["language"],
            )

            import numpy as np
            import soundfile as sf

            tmp = tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False, prefix=f"worker_{model_name}_"
            )
            tmp.close()
            sf.write(tmp.name, np.asarray(wav, dtype=np.float32),
                     getattr(model, "sample_rate", 22_050), subtype="PCM_16")

            _send({
                "status": "ok",
                "audio_path": tmp.name,
                "sample_rate": getattr(model, "sample_rate", 22_050),
            })

        except Exception as exc:
            _send({
                "status": "error",
                "message": str(exc),
                "traceback": traceback.format_exc(),
            })


if __name__ == "__main__":
    main()

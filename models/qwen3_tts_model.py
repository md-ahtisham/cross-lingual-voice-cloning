"""
models/qwen3_tts_model.py
~~~~~~~~~~~~~~~~~~~~~~~~~
Wrapper around Qwen3-TTS for zero-shot cross-lingual voice cloning.

Requires:
    pip install qwen-tts   (inside the qwen3-tts conda env)
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from .base import BaseTTSModel

logger = logging.getLogger(__name__)

_LANG_MAP: dict[str, str] = {
    "en": "English",
    "zh": "Chinese",
    "fr": "French",
    "de": "German",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "ar": "Arabic",
    "es": "Spanish",
    "ru": "Russian",
    "it": "Italian",
}


class Qwen3TTSModel(BaseTTSModel):
    """Qwen3-TTS (1.7B Base) zero-shot voice cloning wrapper."""

    MODEL_NAME = "Qwen3TTS"

    _DEFAULT_LANGUAGES: List[str] = list(_LANG_MAP.keys())

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:1",
        languages: Optional[List[str]] = None,
        dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
        **kwargs,
    ) -> None:
        super().__init__(
            model_path=model_path,
            device=device,
            languages=languages or self._DEFAULT_LANGUAGES,
            **kwargs,
        )
        self._dtype_str = dtype
        self._attn_impl = attn_implementation
        self._sample_rate: int = 24_000

    # ------------------------------------------------------------------ #
    # BaseTTSModel interface                                               #
    # ------------------------------------------------------------------ #

    def load_model(self) -> None:
        import torch
        from qwen_tts import Qwen3TTSModel as _QwenModel  # type: ignore

        resolved_device = self.resolved_device()
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16":  torch.float16,
            "float32":  torch.float32,
        }
        dtype = dtype_map.get(self._dtype_str, torch.bfloat16)

        self._model = _QwenModel.from_pretrained(
            self.model_path,
            device_map=resolved_device,
            dtype=dtype,
            attn_implementation=self._attn_impl,
        )
        logger.info(
            "[Qwen3TTS] Loaded on configured=%s resolved=%s (dtype=%s).",
            self.device,
            resolved_device,
            dtype,
        )

    def generate(
        self,
        text: str,
        ref_audio_path: str,
        ref_text: str,
        language: str,
    ) -> np.ndarray:
        lang_name = _LANG_MAP.get(language, "English")

        wavs, sr = self._model.generate_voice_clone(
            text=text,
            language=lang_name,
            ref_audio=ref_audio_path,
            ref_text=ref_text,
            x_vector_only_mode=True,
        )
        self._sample_rate = sr
        wav = np.array(wavs[0], dtype=np.float32)
        return wav.flatten()

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

"""
models/cosyvoice_model.py
~~~~~~~~~~~~~~~~~~~~~~~~~
Wrapper around CosyVoice3 (0.5B) for zero-shot cross-lingual voice cloning.

Expects:
    config.models.CosyVoice3.repo_path  – root of the cloned CosyVoice repo
    config.models.CosyVoice3.model_path – pretrained weights directory
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

from .base import BaseTTSModel

logger = logging.getLogger(__name__)


class CosyVoiceModel(BaseTTSModel):
    """CosyVoice3 0.5B zero-shot voice cloning wrapper."""

    MODEL_NAME = "CosyVoice3"
    _CV3_PREFIX = "You are a helpful assistant.<|endofprompt|>"

    _DEFAULT_LANGUAGES: List[str] = [
        "en", "ar", "de", "fa", "fr", "ja", "nl", "pt", "ru", "tr", "zh",
    ]

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        languages: Optional[List[str]] = None,
        repo_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            model_path=model_path,
            device=device,
            languages=languages or self._DEFAULT_LANGUAGES,
            **kwargs,
        )
        self._repo_path = repo_path

    # ------------------------------------------------------------------ #
    # BaseTTSModel interface                                               #
    # ------------------------------------------------------------------ #

    def load_model(self) -> None:
        if self._repo_path:
            repo = Path(self._repo_path)
            for p in [repo, repo / "third_party" / "Matcha-TTS"]:
                s = str(p)
                if s not in sys.path:
                    sys.path.insert(0, s)

        from cosyvoice.cli.cosyvoice import CosyVoice3  # type: ignore
        from cosyvoice.utils.file_utils import load_wav  # type: ignore

        self._load_wav = load_wav
        self._cosyvoice = CosyVoice3(self.model_path)
        self._sample_rate: int = self._cosyvoice.sample_rate

    def generate(
        self,
        text: str,
        ref_audio_path: str,
        ref_text: str,
        language: str,
    ) -> np.ndarray:
        import torch

        chunks = []
        if ref_text and ref_text.strip():
            if "<|endofprompt|>" not in ref_text:
                ref_text = f"{self._CV3_PREFIX}{ref_text}"
            iterator = self._cosyvoice.inference_zero_shot(
                text,
                ref_text,
                ref_audio_path,
                stream=False,
            )
        else:
            if "<|endofprompt|>" not in text:
                text = f"{self._CV3_PREFIX}{text}"
            iterator = self._cosyvoice.inference_cross_lingual(
                text,
                ref_audio_path,
                stream=False,
            )

        for chunk in iterator:
            chunks.append(chunk["tts_speech"])

        if not chunks:
            raise RuntimeError("CosyVoice3 produced no audio chunks")

        audio_tensor = torch.concat(chunks, dim=1).squeeze(0)
        return audio_tensor.cpu().numpy().astype(np.float32)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

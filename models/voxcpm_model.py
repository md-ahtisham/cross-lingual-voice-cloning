"""
models/voxcpm_model.py
~~~~~~~~~~~~~~~~~~~~~~
Wrapper around VoxCPM 1.5 for zero-shot voice cloning (ZH + EN only).

Requires:
    pip install voxcpm   (inside the voxcpm_env conda env)
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from .base import BaseTTSModel

logger = logging.getLogger(__name__)


class VoxCPMModel(BaseTTSModel):
    """VoxCPM 1.5 tokenizer-free diffusion TTS wrapper."""

    MODEL_NAME = "VoxCPM"

    _DEFAULT_LANGUAGES: List[str] = ["en", "zh"]

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        languages: Optional[List[str]] = None,
        optimize: bool = False,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        retry_badcase: bool = True,
        retry_badcase_max_times: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(
            model_path=model_path,
            device=device,
            languages=languages or self._DEFAULT_LANGUAGES,
            **kwargs,
        )
        self._optimize = optimize
        self._cfg_value = cfg_value
        self._inference_timesteps = inference_timesteps
        self._retry_badcase = retry_badcase
        self._retry_badcase_max_times = retry_badcase_max_times
        self._sample_rate: int = 44_100

    # ------------------------------------------------------------------ #
    # BaseTTSModel interface                                               #
    # ------------------------------------------------------------------ #

    def load_model(self) -> None:
        from voxcpm import VoxCPM  # type: ignore

        self._model = VoxCPM.from_pretrained(
            self.model_path,
            optimize=self._optimize,
        )
        self._sample_rate = self._model.tts_model.sample_rate
        logger.info("[VoxCPM] Loaded.  Sample rate: %d Hz.", self._sample_rate)

    def generate(
        self,
        text: str,
        ref_audio_path: str,
        ref_text: str,
        language: str,
    ) -> np.ndarray:
        wav = self._model.generate(
            text=text,
            prompt_wav_path=ref_audio_path,
            prompt_text=ref_text,
            cfg_value=self._cfg_value,
            inference_timesteps=self._inference_timesteps,
            normalize=False,
            denoise=False,
            retry_badcase=self._retry_badcase,
            retry_badcase_max_times=self._retry_badcase_max_times,
        )
        return np.asarray(wav, dtype=np.float32).flatten()

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

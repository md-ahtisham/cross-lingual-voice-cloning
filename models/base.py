"""
models/base.py
~~~~~~~~~~~~~~
Abstract base class that every TTS model wrapper must implement.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class BaseTTSModel(ABC):
    """
    Unified interface for all TTS/voice-cloning backends.

    Subclasses must implement:
        load_model()     – load weights onto the configured device
        generate()       – produce a cloned waveform
        supports_language() – declare which language codes are supported
    """

    # Subclasses should override this with a human-readable name used in
    # output paths and metadata.
    MODEL_NAME: str = "BaseTTSModel"

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        languages: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self._supported_languages: List[str] = languages or []
        self.kwargs = kwargs
        self._model = None          # populated by load_model()
        self._is_loaded: bool = False

    # ------------------------------------------------------------------ #
    # Abstract interface                                                   #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def load_model(self) -> None:
        """Load model weights into memory on self.device."""
        ...

    @abstractmethod
    def generate(
        self,
        text: str,
        ref_audio_path: str,
        ref_text: str,
        language: str,
    ) -> np.ndarray:
        """
        Synthesise *text* in *language* with the voice from *ref_audio_path*.

        Returns
        -------
        np.ndarray
            1-D float32 waveform (mono, normalised to [-1, 1]).
        """
        ...

    def supports_language(self, lang: str) -> bool:
        """Return True if this model can synthesise *lang*."""
        return lang in self._supported_languages

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def ensure_loaded(self) -> None:
        """Call load_model() exactly once."""
        if not self._is_loaded:
            logger.info("[%s] Loading model from %s onto %s …",
                        self.MODEL_NAME, self.model_path, self.device)
            self.load_model()
            self._is_loaded = True
            logger.info("[%s] Model ready.", self.MODEL_NAME)

    def resolved_device_for(self, device: str) -> str:
        """
        Return the device string to use inside the current process.

        When the worker subprocess is launched with ``CUDA_VISIBLE_DEVICES``
        narrowed to a single GPU, libraries inside that subprocess should use
        the worker-local ordinals rather than the original global ordinals
        from config.
        """
        device = str(device).strip().lower()
        if device == "cpu":
            return "cpu"
        if device.startswith("cuda:"):
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
            if visible:
                visible_ids = [item.strip() for item in visible.split(",") if item.strip()]
                try:
                    requested = device.split(":", 1)[1]
                    if requested in visible_ids:
                        return f"cuda:{visible_ids.index(requested)}"
                except Exception:
                    pass
        return device

    def resolved_device(self) -> str:
        return self.resolved_device_for(self.device)

    @property
    def name(self) -> str:
        return self.MODEL_NAME

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model_path={self.model_path!r}, "
            f"device={self.device!r}, "
            f"languages={self._supported_languages})"
        )

"""
utils/audio.py
~~~~~~~~~~~~~~
Audio I/O helpers shared across the pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf


def load_audio(path: str, target_sr: int = 16_000) -> Tuple[np.ndarray, int]:
    """
    Load an audio file and optionally resample to *target_sr*.

    Returns
    -------
    (waveform, sample_rate)
        waveform : float32 numpy array, shape (n_samples,)
        sample_rate : actual sample rate after resampling (== target_sr)
    """
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)   # stereo → mono

    if sr != target_sr:
        try:
            import resampy  # type: ignore
            wav = resampy.resample(wav, sr, target_sr)
            sr = target_sr
        except ImportError:
            pass   # return at native rate if resampy not installed

    return wav, sr


def save_audio(path: str, wav: np.ndarray, sr: int) -> None:
    """Save *wav* (float32, mono) to *path* as 16-bit PCM WAV."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), wav.astype(np.float32), sr, subtype="PCM_16")


def validate_audio(wav: np.ndarray) -> bool:
    """Return True if the waveform looks valid (non-empty, finite values)."""
    if wav is None or wav.size == 0:
        return False
    return bool(np.isfinite(wav).all())

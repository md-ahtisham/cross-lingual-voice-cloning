"""
pipeline/dataset_processor.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Download, parse, and persist the HuggingFace dataset to local disk.

Outputs
-------
<output_dir>/reference_audio/sample_<id>.wav
<output_dir>/target_texts/<lang>/sample_<id>.txt
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


class DatasetProcessor:
    """
    Handles downloading and caching the HuggingFace dataset, extracting
    reference audio clips and multilingual target texts, and writing them
    to structured directories.
    """

    def __init__(
        self,
        dataset_name: str,
        split: str,
        text_columns: Dict[str, str],
        output_dir: Path,
        target_languages: List[str],
        max_samples: Optional[int] = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.split = split
        self.text_columns = text_columns
        self.output_dir = output_dir
        self.target_languages = target_languages
        self.max_samples = max_samples

        self._ref_audio_dir = output_dir / "reference_audio"
        self._target_texts_dir = output_dir / "target_texts"
        self._dataset = None

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def prepare(self) -> None:
        """
        Full preparation pass:
        1. Download/load dataset.
        2. Extract & save reference audio.
        3. Extract & save target texts.
        """
        self._create_dirs()
        self._load_dataset()
        self._extract_all()

    def iter_samples(
        self,
    ) -> Iterator[Tuple[str, str, int, Dict[str, str]]]:
        """
        Yield ``(ref_audio_path, ref_text_en, sample_id, {lang: text, …})`` tuples.

        ``ref_text_en`` is the ground-truth English transcription (``text_en``)
        for the audio in that row — passed directly to model.generate() so
        models can map the speaker voice correctly.
        """
        if self._dataset is None:
            self._load_dataset()

        ds = self._dataset
        n = min(len(ds), self.max_samples) if self.max_samples else len(ds)
        en_col = self.text_columns.get("en", "text_en")

        for idx in range(n):
            row = ds[idx]
            ref_path = self._ref_audio_dir / f"sample_{idx:05d}.wav"
            if not ref_path.exists():
                logger.warning("Reference audio missing for sample %d, skipping.", idx)
                continue

            ref_text_en: str = str(row.get(en_col, "") or "").strip()

            texts: Dict[str, str] = {}
            for lang in self.target_languages:
                col = self.text_columns.get(lang)
                if col and col in row and row[col]:
                    texts[lang] = str(row[col]).strip()

            if texts:
                yield str(ref_path), ref_text_en, idx, texts

    # ------------------------------------------------------------------ #
    # Internals                                                            #
    # ------------------------------------------------------------------ #

    def _create_dirs(self) -> None:
        self._ref_audio_dir.mkdir(parents=True, exist_ok=True)
        for lang in self.target_languages:
            (self._target_texts_dir / lang).mkdir(parents=True, exist_ok=True)
        logger.debug("Output directories created under %s.", self.output_dir)

    def _load_dataset(self) -> None:
        from datasets import load_dataset  # type: ignore

        logger.info(
            "Loading dataset '%s' (split=%s) …", self.dataset_name, self.split
        )
        self._dataset = load_dataset(self.dataset_name, split=self.split)
        total = len(self._dataset)
        effective = min(total, self.max_samples) if self.max_samples else total
        logger.info("Dataset loaded: %d rows total, processing %d.", total, effective)

    def _extract_all(self) -> None:
        ds = self._dataset
        n = min(len(ds), self.max_samples) if self.max_samples else len(ds)

        audio_saved = 0
        text_saved: Dict[str, int] = {lang: 0 for lang in self.target_languages}

        for idx in range(n):
            row = ds[idx]
            self._save_reference_audio(row, idx)
            audio_saved += 1
            for lang in self.target_languages:
                if self._save_target_text(row, idx, lang):
                    text_saved[lang] += 1

        logger.info(
            "Extraction complete.  Audio: %d clips.  Texts: %s",
            audio_saved,
            text_saved,
        )

    def _save_reference_audio(self, row: dict, idx: int) -> None:
        out_path = self._ref_audio_dir / f"sample_{idx:05d}.wav"
        if out_path.exists():
            return

        audio_data = row.get("audio")
        if audio_data is None:
            logger.warning("Row %d has no 'audio' field, skipping.", idx)
            return

        if isinstance(audio_data, dict):
            array: np.ndarray = np.asarray(audio_data["array"], dtype=np.float32)
            sr: int = int(audio_data["sampling_rate"])
        else:
            logger.warning(
                "Unexpected audio format at row %d: %s", idx, type(audio_data)
            )
            return

        sf.write(str(out_path), array, sr, subtype="PCM_16")

    def _save_target_text(self, row: dict, idx: int, lang: str) -> bool:
        col = self.text_columns.get(lang)
        if not col:
            return False
        text = row.get(col, "")
        if not text or not str(text).strip():
            return False

        out_path = self._target_texts_dir / lang / f"sample_{idx:05d}.txt"
        if out_path.exists():
            return True

        out_path.write_text(str(text).strip(), encoding="utf-8")
        return True

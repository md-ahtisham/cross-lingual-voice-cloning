"""
pipeline/generation_pipeline.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Orchestrates the full cross-lingual voice-cloning generation loop.

Flow:
    for each enabled model:
        load model subprocess (once)
        for each dataset sample:
            use the row's own English audio/text as reference
            for each target language (if model supports it):
                generate cloned audio
                save .wav
                append metadata row
    write metadata.csv
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np
import soundfile as sf
from tqdm import tqdm

logger = logging.getLogger(__name__)

_METADATA_FIELDS = [
    "sample_id",
    "model_name",
    "language",
    "reference_audio_path",
    "ref_text_en",
    "generated_audio_path",
    "text",
    "status",
    "error_msg",
    "duration_sec",
]


class GenerationPipeline:
    """
    Drives the full cross-lingual voice-cloning experiment.

    Each model is a SubprocessModelProxy; generation calls are forwarded
    to the per-model conda environment worker process.
    """

    def __init__(
        self,
        models: list,
        dataset_iter: Iterator[Tuple[str, str, int, Dict[str, str]]],
        output_dir: Path,
        target_langs: List[str],
        skip_on_error: bool = True,
    ) -> None:
        self.models = models
        self.dataset_iter = dataset_iter
        self.output_dir = output_dir
        self.target_langs = target_langs
        self.skip_on_error = skip_on_error

        self._metadata_path = output_dir / "metadata.csv"
        self._metadata_rows: List[Dict] = []

    # ------------------------------------------------------------------ #
    # Public entry point                                                   #
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        """Execute the full pipeline and write metadata.csv."""
        samples: List[Tuple[str, str, int, Dict[str, str]]] = list(self.dataset_iter)
        total_per_model = sum(len(lang_texts) for _, _, _, lang_texts in samples)
        logger.info(
            "Pipeline start:  %d models | %d samples | %d target texts",
            len(self.models),
            len(samples),
            total_per_model,
        )

        total = len(self.models) * total_per_model

        try:
            with tqdm(total=total, desc="Generating", unit="clip") as pbar:
                for model in self.models:
                    self._run_model(model, samples, pbar)
        finally:
            # Cleanly shut down all worker subprocesses
            for model in self.models:
                if hasattr(model, "close"):
                    try:
                        model.close()
                    except Exception:
                        pass

        self._write_metadata()
        logger.info(
            "Pipeline complete.  %d clips attempted.  Metadata → %s",
            len(self._metadata_rows),
            self._metadata_path,
        )

    # ------------------------------------------------------------------ #
    # Per-model loop                                                       #
    # ------------------------------------------------------------------ #

    def _run_model(self, model, samples, pbar: tqdm) -> None:
        try:
            model.ensure_loaded()
        except Exception as exc:
            logger.error(
                "[%s] Failed to load model: %s  —  skipping entire model.",
                model.name,
                exc,
                exc_info=True,
            )
            pbar.update(sum(len(lang_texts) for _, _, _, lang_texts in samples))
            return

        model_out_dir = self.output_dir / "generated_clones" / model.name
        model_out_dir.mkdir(parents=True, exist_ok=True)

        for ref_audio_path, ref_text_en, sample_id, lang_texts in samples:
            self._run_sample(model, ref_audio_path, ref_text_en, sample_id, lang_texts, model_out_dir, pbar)

    # ------------------------------------------------------------------ #
    # Per-sample loop                                                      #
    # ------------------------------------------------------------------ #

    def _run_sample(
        self,
        model,
        ref_audio_path: str,
        ref_text_en: str,
        sample_id: int,
        lang_texts: Dict[str, str],
        model_out_dir: Path,
        pbar: tqdm,
    ) -> None:
        for lang, text in lang_texts.items():
            pbar.update(1)

            if not model.supports_language(lang):
                self._record(
                    sample_id=sample_id,
                    model_name=model.name,
                    language=lang,
                    ref_audio=ref_audio_path,
                    ref_text_en=ref_text_en,
                    gen_audio="",
                    text=text,
                    status="skipped_lang",
                )
                continue

            lang_dir = model_out_dir / lang
            lang_dir.mkdir(parents=True, exist_ok=True)
            out_fname = f"sample_{sample_id:05d}_{lang}.wav"
            out_path = lang_dir / out_fname

            if out_path.exists():
                logger.debug("[%s] %s already exists, skipping.", model.name, out_path)
                self._record(
                    sample_id=sample_id,
                    model_name=model.name,
                    language=lang,
                    ref_audio=ref_audio_path,
                    ref_text_en=ref_text_en,
                    gen_audio=str(out_path),
                    text=text,
                    status="cached",
                )
                continue

            self._generate_and_save(
                model=model,
                text=text,
                ref_audio_path=ref_audio_path,
                ref_text_en=ref_text_en,
                language=lang,
                sample_id=sample_id,
                out_path=out_path,
            )

    # ------------------------------------------------------------------ #
    # Single-clip generation                                               #
    # ------------------------------------------------------------------ #

    def _generate_and_save(
        self,
        model,
        text,
        ref_audio_path,
        ref_text_en,
        language,
        sample_id,
        out_path: Path,
    ) -> None:
        t0 = time.perf_counter()
        try:
            wav: np.ndarray = model.generate(
                text=text,
                ref_audio_path=ref_audio_path,
                ref_text=ref_text_en,
                language=language,
            )
            elapsed = time.perf_counter() - t0

            wav = np.asarray(wav, dtype=np.float32)
            peak = np.abs(wav).max()
            if peak > 0:
                wav = wav / peak

            sr = getattr(model, "sample_rate", 22_050)
            sf.write(str(out_path), wav, sr, subtype="PCM_16")

            logger.debug(
                "[%s] ✓ %s | lang=%s | %.1fs",
                model.name, out_path.name, language, elapsed,
            )
            self._record(
                sample_id=sample_id,
                model_name=model.name,
                language=language,
                ref_audio=ref_audio_path,
                ref_text_en=ref_text_en,
                gen_audio=str(out_path),
                text=text,
                status="ok",
                duration_sec=round(elapsed, 2),
            )

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            logger.error(
                "[%s] ✗ sample=%d lang=%s — %s",
                model.name, sample_id, language, exc,
                exc_info=True,
            )
            self._record(
                sample_id=sample_id,
                model_name=model.name,
                language=language,
                ref_audio=ref_audio_path,
                ref_text_en=ref_text_en,
                gen_audio="",
                text=text,
                status="error",
                error_msg=str(exc),
                duration_sec=round(elapsed, 2),
            )
            if not self.skip_on_error:
                raise

    # ------------------------------------------------------------------ #
    # Metadata helpers                                                     #
    # ------------------------------------------------------------------ #

    def _record(
        self,
        sample_id, model_name, language,
        ref_audio, ref_text_en, gen_audio, text,
        status="ok", error_msg="", duration_sec=0.0,
    ) -> None:
        self._metadata_rows.append({
            "sample_id": sample_id,
            "model_name": model_name,
            "language": language,
            "reference_audio_path": ref_audio,
            "ref_text_en": ref_text_en,
            "generated_audio_path": gen_audio,
            "text": text,
            "status": status,
            "error_msg": error_msg,
            "duration_sec": duration_sec,
        })

    def _write_metadata(self) -> None:
        self._metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._metadata_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_METADATA_FIELDS)
            writer.writeheader()
            writer.writerows(self._metadata_rows)
        logger.info(
            "Metadata written: %d rows → %s",
            len(self._metadata_rows), self._metadata_path,
        )

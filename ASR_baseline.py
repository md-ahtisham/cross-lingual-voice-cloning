"""
ASR_baseline.py
~~~~~~~~~~~~~~~
Transcribes all sampled clone WAV files using Whisper large-v3 (local cache)
"""

import argparse
import csv
import gc
import logging
from collections import defaultdict
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

LANG_MAP: dict[str, str] = {
    "ar": "arabic",
    "fr": "french",
    "zh": "chinese",
}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Whisper ASR baseline for cloned audio.")
    p.add_argument("--sample_root", required=True)
    p.add_argument("--model_cache", default="/mnt/storage/hf_cache/hub/models--openai--whisper-large-v3")
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()

def load_manifest(sample_root: Path) -> list[dict]:
    manifest_path = sample_root / "manifests" / "sampled_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    logger.info("Manifest loaded: %d rows from %s", len(rows), manifest_path)
    return rows

def resolve_whisper_path(model_cache: str) -> str:
    cache = Path(model_cache)
    snapshots = cache / "snapshots"
    if snapshots.exists():
        hashes = sorted(snapshots.iterdir())
        if hashes:
            return str(hashes[-1])
    return model_cache

def load_whisper(model_path: str, device: str):
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    logger.info("Loading Whisper large-v3 from %s onto %s …", model_path, device)
    torch_dtype = torch.float16 if "cuda" in device else torch.float32

    whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    whisper_model.to(device)
    processor = AutoProcessor.from_pretrained(model_path)

    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model=whisper_model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        chunk_length_s=30, # PREVENTS OOM SPIKES
        return_timestamps=False,
    )
    logger.info("Whisper ready.")
    return asr_pipe

RESULT_FIELDS = [
    "model", "speaker_id", "lang", "line_id", "clone_wav_path",
    "target_text", "hypothesis", "ref_audio_path", "ref_text_path",
]
SUMMARY_FIELDS = ["model", "lang", "total_clips", "transcribed", "failed"]

def write_results(rows: list[dict], out_path: Path) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logger.info("ASR results written: %d rows → %s", len(rows), out_path)

def write_summary(rows: list[dict], out_path: Path) -> None:
    counts: dict[tuple, dict] = defaultdict(lambda: {"total_clips": 0, "transcribed": 0, "failed": 0})
    for row in rows:
        key = (row["model"], row["lang"])
        counts[key]["total_clips"] += 1
        if row.get("hypothesis"):
            counts[key]["transcribed"] += 1
        else:
            counts[key]["failed"] += 1

    summary_rows = [
        {"model": model, "lang": lang, "total_clips": v["total_clips"], "transcribed": v["transcribed"], "failed": v["failed"]}
        for (model, lang), v in sorted(counts.items())
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)
    logger.info("Summary written → %s", out_path)

def main() -> None:
    args = parse_args()
    sample_root = Path(args.sample_root)
    out_dir = sample_root / "asr_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    results_csv = out_dir / "asr_results.csv"
    summary_csv = out_dir / "asr_summary.csv"

    if results_csv.exists() and not args.overwrite:
        logger.info("asr_results.csv already exists. Use --overwrite to re-run. Exiting.")
        return

    manifest = load_manifest(sample_root)
    valid_manifest = [r for r in manifest if (sample_root / r["clone_wav_path"]).exists()]
    logger.info("Clips to transcribe: %d", len(valid_manifest))

    model_path = resolve_whisper_path(args.model_cache)
    asr_pipe = load_whisper(model_path, args.device)

    lang_groups = defaultdict(list)
    for row in valid_manifest:
        lang_groups[row["lang"]].append(row)

    all_results = []

    for lang_code, rows in lang_groups.items():
        whisper_lang = LANG_MAP.get(lang_code, None)
        generate_kwargs = {"task": "transcribe"}
        if whisper_lang:
            generate_kwargs["language"] = whisper_lang
            
        logger.info(f"Processing Language: {lang_code.upper()} ({len(rows)} clips)")
        
        # EXPLICIT MINI-BATCHING WITH FALLBACK
        for i in range(0, len(rows), args.batch_size):
            batch_rows = rows[i:i + args.batch_size]
            batch_paths = [str(sample_root / r["clone_wav_path"]) for r in batch_rows]
            
            try:
                # Attempt Batched Inference
                outputs = asr_pipe(batch_paths, batch_size=args.batch_size, generate_kwargs=generate_kwargs)
                for j, out in enumerate(outputs):
                    batch_rows[j]["hypothesis"] = out["text"].strip()
                    all_results.append(batch_rows[j])
                    
            except Exception as e:
                logger.warning(f"Batch {i//args.batch_size} failed ({type(e).__name__}). Falling back to batch_size=1...")
                torch.cuda.empty_cache()
                
                # Fallback: Process 1 by 1 to bypass OOM or Dictionary mismatch
                for j, single_path in enumerate(batch_paths):
                    try:
                        out = asr_pipe(single_path, generate_kwargs=generate_kwargs)
                        batch_rows[j]["hypothesis"] = out["text"].strip()
                    except Exception as inner_e:
                        logger.error(f"Failed clip {single_path}: {inner_e}")
                        batch_rows[j]["hypothesis"] = ""
                    all_results.append(batch_rows[j])
                    
            finally:
                gc.collect()
                torch.cuda.empty_cache()

    write_results(all_results, results_csv)
    write_summary(all_results, summary_csv)
    logger.info("Done. Results in: %s", out_dir.resolve())

if __name__ == "__main__":
    main()
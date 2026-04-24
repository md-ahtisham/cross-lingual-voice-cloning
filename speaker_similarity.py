"""
speaker_similarity.py
~~~~~~~~~~~~~~~~~~~~~
Calculates Voice Biometrics (Cosine Similarity) between cloned audio and reference audio.
Uses SpeechBrain's ECAPA-TDNN model.
"""

import argparse
import pandas as pd
import torch
import torchaudio
import logging
from pathlib import Path

from speechbrain.inference.speaker import EncoderClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser(description="Calculate Speaker Similarity using SpeechBrain.")
    p.add_argument("--results_csv", required=True, help="Path to your ASR or Sampled CSV containing audio paths.")
    p.add_argument("--sample_root", required=True, help="Path to your evaluation_dataset folder.")
    p.add_argument("--output_dir", required=True, help="Directory to save the similarity metrics.")
    p.add_argument("--device", default="cuda:0", help="Torch device to run ECAPA-TDNN (e.g., cuda:4).")
    return p.parse_args()

def load_and_prep_audio(filepath, target_sr=16000):
    """Loads audio, converts to mono, and resamples to 16kHz for ECAPA-TDNN."""
    sig, sr = torchaudio.load(filepath)
    
    if sig.shape[0] > 1:
        sig = sig.mean(dim=0, keepdim=True)
        
    if sr != target_sr:
        sig = torchaudio.functional.resample(sig, orig_freq=sr, new_freq=target_sr)
        
    return sig

def main():
    args = parse_args()
    input_path = Path(args.results_csv)
    sample_root = Path(args.sample_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)

    logger.info(f"Downloading/Loading ECAPA-TDNN onto {args.device}...")
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": args.device},
        savedir="tmp_speechbrain_cache" 
    )
    logger.info("Model loaded successfully.")

    ref_embedding_cache = {}
    similarity_scores = []

    logger.info("Extracting embeddings and calculating similarity...")
    
    for idx, row in df.iterrows():
        ref_path = row['ref_audio_path']
        
        # THE FIX: Combine the sample_root with the relative path from the CSV
        clone_path = str(sample_root / row['clone_wav_path'])

        if pd.isna(ref_path) or not Path(ref_path).exists():
            logger.warning(f"Row {idx}: Missing reference audio. Assigning score 0.0")
            similarity_scores.append(0.0)
            continue
            
        if pd.isna(clone_path) or not Path(clone_path).exists():
            logger.warning(f"Row {idx}: Missing clone audio ({clone_path}). Assigning score 0.0")
            similarity_scores.append(0.0)
            continue

        try:
            if ref_path not in ref_embedding_cache:
                ref_sig = load_and_prep_audio(ref_path)
                ref_emb = classifier.encode_batch(ref_sig).squeeze()
                ref_embedding_cache[ref_path] = ref_emb

            ref_emb = ref_embedding_cache[ref_path]

            clone_sig = load_and_prep_audio(clone_path)
            clone_emb = classifier.encode_batch(clone_sig).squeeze()

            score = torch.nn.functional.cosine_similarity(ref_emb, clone_emb, dim=0).item()
            score = max(0.0, score)
            similarity_scores.append(score)

        except Exception as e:
            logger.error(f"Failed to process row {idx} ({clone_path}): {e}")
            similarity_scores.append(0.0)

        if (idx + 1) % 200 == 0:
            logger.info(f"Processed {idx + 1}/{len(df)} files...")

    df['similarity_score'] = similarity_scores

    detailed_csv = out_dir / "similarity_detailed.csv"
    df.to_csv(detailed_csv, index=False, encoding='utf-8')

    logger.info("Aggregating high-level similarities...")
    model_summary = df.groupby(['model', 'lang'])['similarity_score'].mean().reset_index()
    
    model_summary_csv = out_dir / "similarity_model_summary.csv"
    model_summary.to_csv(model_summary_csv, index=False, encoding='utf-8')

    logger.info("="*50)
    logger.info(f"Done! Matrices saved to {out_dir}")
    logger.info("="*50)
    print(model_summary.to_string(index=False))

if __name__ == "__main__":
    main()
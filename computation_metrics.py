"""
compute_metrics.py
~~~~~~~~~~~~~~~~~~
Calculates WER (Word Error Rate) for Arabic and French, and CER (Character Error Rate) for Chinese.
Aggregates the scores by model and speaker for easy plotting.
"""

import argparse
import re
import pandas as pd
import jiwer
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser(description="Calculate WER/CER for ASR results.")
    p.add_argument("--results_csv", required=True, help="Path to asr_results.csv")
    p.add_argument("--output_dir", required=True, help="Directory to save metric reports")
    return p.parse_args()

def normalize_fr(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^\w\s\']', ' ', text)  # Keep apostrophes, remove other punctuation
    return " ".join(text.split())

def normalize_ar(text):
    if not isinstance(text, str): return ""
    # Strip Arabic diacritics (Tashkeel) and punctuation
    text = re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return " ".join(text.split())

def normalize_zh(text):
    if not isinstance(text, str): return ""
    # Remove all punctuation and spaces for pure character comparison
    text = re.sub(r'[^\w]', '', text)
    return text

def compute_error(row):
    lang = row['lang']
    ref = row['norm_target']
    hyp = row['norm_hypothesis']
    
    # Handle empty predictions or references
    if not ref:
        return 0.0 # Ignore if ground truth is magically empty
    if not hyp:
        return 1.0 # 100% error if model output nothing

    try:
        if lang == 'zh':
            # CER for Chinese (jiwer computes character-level if we treat characters as words)
            # By splitting into list of chars, jiwer calculates exact CER
            return jiwer.wer(list(ref), list(hyp))
        else:
            # WER for French and Arabic
            return jiwer.wer(ref, hyp)
    except Exception as e:
        logger.warning(f"Error computing metric for line {row['line_id']}: {e}")
        return 1.0

def main():
    args = parse_args()
    input_path = Path(args.results_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading ASR results from {input_path}")
    df = pd.read_csv(input_path)

    # Apply Normalization
    logger.info("Applying language-specific text normalization...")
    df['norm_target'] = ""
    df['norm_hypothesis'] = ""

    # Vectorized application based on language
    df.loc[df['lang'] == 'fr', 'norm_target'] = df.loc[df['lang'] == 'fr', 'target_text'].apply(normalize_fr)
    df.loc[df['lang'] == 'fr', 'norm_hypothesis'] = df.loc[df['lang'] == 'fr', 'hypothesis'].apply(normalize_fr)

    df.loc[df['lang'] == 'ar', 'norm_target'] = df.loc[df['lang'] == 'ar', 'target_text'].apply(normalize_ar)
    df.loc[df['lang'] == 'ar', 'norm_hypothesis'] = df.loc[df['lang'] == 'ar', 'hypothesis'].apply(normalize_ar)

    df.loc[df['lang'] == 'zh', 'norm_target'] = df.loc[df['lang'] == 'zh', 'target_text'].apply(normalize_zh)
    df.loc[df['lang'] == 'zh', 'norm_hypothesis'] = df.loc[df['lang'] == 'zh', 'hypothesis'].apply(normalize_zh)

    # Compute Metrics
    logger.info("Calculating WER and CER...")
    df['error_rate'] = df.apply(compute_error, axis=1)

    # Export Line-by-Line Detailed Metrics
    detailed_csv = out_dir / "detailed_metrics.csv"
    df.to_csv(detailed_csv, index=False, encoding='utf-8')
    logger.info(f"Detailed metrics saved to {detailed_csv}")

    # Aggregate by Model, Language, and Speaker
    logger.info("Aggregating results for graphing...")
    grouped = df.groupby(['model', 'lang', 'speaker_id'])['error_rate'].agg(['mean', 'count']).reset_index()
    grouped.rename(columns={'mean': 'avg_error_rate', 'count': 'samples_evaluated'}, inplace=True)
    
    # Split the metric column for clarity in the final report
    grouped['metric_type'] = grouped['lang'].apply(lambda x: 'CER' if x == 'zh' else 'WER')
    
    speaker_summary_csv = out_dir / "speaker_level_metrics.csv"
    grouped.to_csv(speaker_summary_csv, index=False, encoding='utf-8')
    
    # High-level Model Summary
    model_summary = df.groupby(['model', 'lang'])['error_rate'].mean().reset_index()
    model_summary['metric_type'] = model_summary['lang'].apply(lambda x: 'CER' if x == 'zh' else 'WER')
    model_summary_csv = out_dir / "model_level_summary.csv"
    model_summary.to_csv(model_summary_csv, index=False, encoding='utf-8')

    logger.info("="*50)
    logger.info(f"Done! Aggregated matrices saved to {out_dir}")
    logger.info("="*50)
    print(model_summary.to_string(index=False))

if __name__ == "__main__":
    main()
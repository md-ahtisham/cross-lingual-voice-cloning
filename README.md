# Cross-Lingual Voice Cloning Pipeline

End-to-end pipeline that takes English reference audio and multilingual text
from the `ymoslem/acl-6060` HuggingFace dataset and generates cross-lingual
cloned speech with four locally-hosted TTS models.

---

## Project structure

```
cross_lingual_vc_pipeline/
│
├── run_pipeline.py              ← CLI entry point
├── requirements.txt
│
├── config/
│   └── config.yaml              ← all paths, GPU assignments, enabled models
│
├── models/
│   ├── __init__.py              ← MODEL_REGISTRY + build_model()
│   ├── base.py                  ← BaseTTSModel (abstract)
│   ├── cosyvoice_model.py
│   ├── qwen3_tts_model.py
│   ├── voxcpm_model.py
│   ├── moss_tts_model.py
│   └── cpsy_voice_model.py      ← stub — fill in when env is ready
│
├── pipeline/
│   ├── __init__.py
│   ├── dataset_processor.py     ← HF download, audio/text extraction
│   └── generation_pipeline.py  ← main generation loop + metadata.csv
│
└── utils/
    ├── __init__.py
    ├── audio.py                 ← load/save/validate audio
    ├── config.py                ← YAML loader + validation
    └── logging_setup.py
```

Output written to `cross_lingual_voice_cloned_data/` (configurable):

```
cross_lingual_voice_cloned_data/
├── metadata.csv
├── reference_audio/             ← extracted from dataset (or your own)
├── target_texts/
│   ├── ar/  fr/  zh/ …
└── generated_clones/
    ├── CosyVoice3/  ar/  fr/  zh/
    ├── Qwen3TTS/
    ├── VoxCPM/
    └── MossTTS/
```

---

## Quick start

### 1. Edit config

Open `config/config.yaml` and set:
- `model_path` / `repo_path` for each model to your local paths
- `device` per model (e.g. `cuda:0`, `cuda:1`, `cuda:2`)
- `enabled: true/false` to toggle models
- `target_languages` to the codes you want

### 2. Add reference speakers (optional)

Place your own `.wav` files in `reference_audio/` named `spk01.wav`,
`spk02.wav`, etc.  If the directory is empty or absent the pipeline will
use the audio clips extracted from the dataset itself.

### 3. Install dependencies

Each model may need its own conda environment.  At minimum install core deps:

```bash
pip install -r requirements.txt
```

Then per-model:
```bash
# CosyVoice3
git clone --recursive https://github.com/FunAudioLLM/CosyVoice
# update repo_path in config.yaml to point here

# Qwen3-TTS
pip install qwen-tts

# VoxCPM
pip install voxcpm
```

### 4. Run

```bash
# Full run
python run_pipeline.py --config config/config.yaml

# Dry run (no model loading, just validate + count)
python run_pipeline.py --config config/config.yaml --dry-run

# Specific models + languages
python run_pipeline.py --models CosyVoice3 VoxCPM --langs fr zh

# Limit dataset size for a quick test
python run_pipeline.py --max-samples 5
```

---

## Adding a new model

1. Create `models/my_new_model.py` inheriting from `BaseTTSModel`.
2. Implement `load_model()`, `generate()`, and set `MODEL_NAME`.
3. Register it in `models/__init__.py` → `MODEL_REGISTRY`.
4. Add an entry under `models:` in `config.yaml`.

That's it — no other file needs to change.

---

## metadata.csv columns

| Column | Description |
|---|---|
| sample_id | Dataset row index |
| speaker_id | Stem of reference wav (e.g. `spk01`) |
| model_name | Model that generated this clip |
| language | ISO-639-1 code |
| reference_audio_path | Path to reference .wav used |
| generated_audio_path | Path to output .wav (empty on error/skip) |
| text | Target text synthesised |
| status | `ok` / `error` / `skipped_lang` / `skipped_no_text` / `cached` |
| error_msg | Exception message (empty on success) |
| duration_sec | Wall-clock generation time |

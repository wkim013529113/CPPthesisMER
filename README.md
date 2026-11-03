# Phonetic Lyric Representations in Multimodal Music Emotion Recognition

Code for the MS thesis *"Phonetic Lyric Representations in Multimodal Music Emotion Recognition: A Rhyme-Density Ablation Study"* (Bill Kim, Cal Poly Pomona; advisor Dr. Fatemeh Jamshidi).

The question: do four rhyme-density features carry emotion signal beyond what an audio encoder (MERT-v1-95M) and a lyric semantic encoder (XLM-RoBERTa-base) already capture?

This code extends **Music2Emo** (Kang & Herremans, 2025, arXiv:2502.03979). See `LICENSE` for the original license; changes are described below.

## Result summary (rerun-v2)

- XLM-R lyrics improve MERGE valence R² by +0.158 (95% CI [+0.101, +0.217]); arousal is unchanged.
- Rhyme density adds no detectable signal beyond XLM-R: C4 − C3 valence R² −0.002 (95% CI [−0.015, +0.010]). Real rhyme values perform the same as shuffled values, random noise, and non-rhyme lyric features.
- The same null holds on MERGE's official (artist-overlapping) split.

Tables: `runs/results/` (main matrix) and `runs/results_diag/` (official-split diagnostic).

## Repository map

| Path | Purpose |
|---|---|
| `train.py`, `test.py` | Hydra entry points for training and evaluation |
| `trainer.py` | Lightning module: multitask losses, lyrics fusion routing, metrics, saved predictions |
| `data_loader.py` | DataModule; seeded DataLoaders |
| `model/linear_mt_attn_ck.py` | Base Music2Emo multitask model (mood, VA, GlobalMood heads) |
| `model/lyrics_fusion.py` | Late fusion: shared audio repr (512) + XLM-R (768→256) + 4-dim slot → 512 |
| `dataset_loaders/` | Jamendo, GlobalMood, MERGE loaders (frozen splits, rhyme z-scoring, control modes) |
| `config/` | `base_config.yaml` (datasets, seed, split, rhyme settings), train/test configs |
| `lyrics_feature_extractor.py` | XLM-R embeddings and rhyme vectors per MERGE lyric |
| `tools/data_audit.py` | Coverage, leakage and label audit |
| `tools/make_splits.py` | Artist-disjoint MERGE split and frozen split lists |
| `tools/make_surface_features.py` | Non-rhyme lyric control features (K3) |
| `tools/analyze_v2.py` | Seed summaries, paired bootstrap, pre-registered criteria |
| `runs/` | Protocol (`README.md`), run matrices, SLURM job scripts, results |
| `splits/final/` | Frozen train/val/test ID lists and normalization statistics |

## Data (not included)

Datasets and extracted features are not in this repository. Expected layout under `dataset/`:

- `jamendo/` — MTG-Jamendo moodtheme, split-0 pickles, `mert_30s/` features
- `globalmood/` — GlobalMood labels, tags, `mert_30s/` features
- `merge/MERGE_Bimodal_Balanced/` — MERGE Bimodal Balanced (Louro et al., 2024, arXiv:2407.06060) with `mert_30s/`, `xlmr/`, `rhyme/`, `surface/`, `lyrics/`, `tvt_dataframes/`

Check each dataset's license before redistributing any derived files.

## Reproducing the results

```bash
conda env create -f environment.yml && conda activate music2emo

# 1. features (once): MERT per dataset, then lyrics features
python lyrics_feature_extractor.py
python tools/make_splits.py
python tools/make_surface_features.py

# 2. runs (SLURM): 26-run matrix, then the 6-run diagnostic
LAMBDA=0.05 sbatch --array=1-26%2 --export=ALL runs/run_one.sh
MATRIX=runs/run_matrix_diag.tsv LAMBDA=0.05 sbatch --array=1-6%2 --export=ALL runs/run_one.sh

# 3. analysis
python tools/analyze_v2.py
python tools/analyze_v2.py --matrix runs/run_matrix_diag.tsv --out runs/results_diag
```

The SLURM scripts contain cluster-specific paths (`/data03/home/wkim/...`, `--nodelist=cn01`); edit them for another machine. Runs are seeded (0, 1, 2); GPU training is not bit-exact across hardware, so reruns should fall within the reported seed SD.

Environment used: CPP HPC, Tesla P100 GPUs, PyTorch Lightning + Hydra.

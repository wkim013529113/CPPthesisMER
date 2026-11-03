# rerun-v2 run protocol (fixed before any results)

## Data (see splits/final/, tools/make_splits.py)
- Training pool: MTG-Jamendo moodtheme (mood head), GlobalMood (100-tag head), MERGE (VA head). GlobalMood_VA removed.
- MERGE split: artist-disjoint 70/15/15, stratified by quadrant (`tvt_artist_70_15_15`).
- Rhyme slot: always 4-dim when lyrics are on; z-scored with MERGE train-split statistics.

## Run matrix (runs/run_matrix.tsv, 26 runs)
| Config | Datasets | Lyrics | 4-dim slot | Seeds | Tier |
|---|---|---|---|---|---|
| C1 | jamendo, merge | off | — | 0,1,2 | core |
| C2 | jamendo, globalmood, merge | off | — | 0,1,2 | core |
| C3 | jamendo, globalmood, merge | XLM-R | zeros | 0,1,2 | core |
| C4 | jamendo, globalmood, merge | XLM-R | real rhyme | 0,1,2 | core |
| K1 | as C4 | XLM-R | rhyme vectors shuffled across songs (fixed, within split) | 0,1,2 | control |
| K2 | as C4 | XLM-R | fixed per-song N(0,1) noise | 0,1,2 | control |
| K3 | as C4 | XLM-R | non-rhyme surface features (lines, words, word length, type-token ratio) | 0,1,2 | control |
| X1 | as C4 | XLM-R | rhyme without coverage (slots 0,1,2) | 0 | exploratory |
| X2 | as C4 | XLM-R | coverage only (slot 3) | 0 | exploratory |
| X3–X5 | as C4 | XLM-R | end / internal / per-line only | 0 | exploratory |

Slot order: [end_rhyme_density, internal_rhyme_density, rhymes_per_line, cmudict_coverage].
Every lyrics run has the identical 772-dim fusion input, so parameter count is the same across C3, C4, K*, X*.
Exploratory runs are single-seed and are reported as such.

## Fixed settings
- Optimizer Adam, lr 1e-4, batch 8 per dataset, max 50 epochs, early stopping on val_loss (patience 10, min_delta 1e-4).
- Seeds 0, 1, 2 for every multi-seed configuration; the same seed set for all, so comparisons are paired.
- control_seed 1234 for K1 permutation and K2 noise, fixed across training seeds.
- VA loss weight λ: decided once by runs/lambda_check.sh (below), then fixed for all 26 runs.

## Checkpoint rule
- Jamendo metrics (PR-AUC, ROC-AUC): checkpoint with the lowest val_loss_mood.
- MERGE metrics (R², valence R², arousal R², RMSE): checkpoint with the lowest val_loss_va (MERGE validation MSE).
- Each eval writes the checkpoint path to eval_checkpoint_{mood,va}.txt in the run folder.

## λ decision rule (job 0)
Run C2, seed 0, 3 epochs at λ = 0.05. At epoch 3, compare λ·train_loss_va with train_loss_mood + train_loss_globalmood.
- If the ratio is between 1/3 and 3, keep λ = 0.05.
- Otherwise pick the λ from {0.2, 0.1, 0.05, 0.02, 0.01} that brings the ratio closest to 1.
Training losses only; no validation or test results are used to choose λ.

## Analysis (planned; tools/analyze_v2.py to follow)
- Per run: metrics JSON + per-song test predictions (test_metrics_{mood,va}_preds_*.npz).
- Report mean ± SD over seeds per configuration.
- Paired bootstrap over MERGE test songs (10,000 resamples), seed-averaged predictions, for:
  1. C4 vs C3 (rhyme beyond XLM-R semantics) — primary comparison
  2. C4 vs K1 (song-specific rhyme vs rhyme distribution)
  3. C4 vs K2 (rhyme vs any extra 4-dim input)
  4. C4 vs K3 (rhyme vs non-rhyme lyric structure)
  5. C3 vs C2 (XLM-R lyrics), C2 vs C1 (GlobalMood multitask)

## Proposed claim criteria (confirm with Dr. Jamshidi before results)
- Rhyme effect supported: C4 − C3 on MERGE valence R² has a 95% bootstrap CI above 0, AND C4 beats K1 and K2 by the same criterion.
- If C4 ≈ K3 (CI includes 0): the gain is attributed to general lyric structure, not rhyme specifically.
- If C4 − C3 CI includes 0: report as no detectable rhyme effect under these controls.
- Arousal and Jamendo metrics are reported for all configurations, without separate claims.

## Commands
```bash
python tools/make_surface_features.py               # once, before K3
sbatch runs/lambda_check.sh                         # job 0
RUN_SUFFIX=_smoke EXTRA="epochs=1 +trainer.limit_train_batches=30 +trainer.limit_val_batches=10" \
  sbatch --array=19 --export=ALL runs/run_one.sh    # smoke: K3 row, separate folder
LAMBDA=0.05 sbatch --array=1-26%2 --export=ALL runs/run_one.sh   # full matrix
```

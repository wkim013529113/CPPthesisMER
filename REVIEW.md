# Code review guide (rerun-v2)

Suggested reading order. Each item lists what to check; together they cover every change made after the advisor review.

1. **`runs/README.md`: the protocol.** Run matrix (C1–C4, K1–K3, X1–X5), seeds, checkpoint rule, λ decision rule, comparisons and claim criteria, all fixed before results.

2. **`tools/make_splits.py`: data splits.** MERGE re-split artist-disjoint (70/15/15, stratified by quadrant; same-artist and identical-audio songs grouped). GlobalMood split frozen to files (same rng-42 assignment). Jamendo: two test tracks that duplicate train audio removed. Audit evidence: `tools/data_audit.py`.

3. **`dataset_loaders/merge.py`: MERGE loader.**
   - Reads the split folder from config (`tvt_artist_70_15_15` by default).
   - Z-scores the 4 rhyme features with train-split statistics (`splits/final/rhyme_stats.json`).
   - `rhyme_mode` fills the 4-dim slot: `real` (C4), `zeros` (C3), `permute` (K1), `noise` (K2), `surface` (K3), `keepNNN` (ablations). The slot is 4-dim in every lyrics run, so all have the same parameter count.

4. **`dataset_loaders/globalmood.py`, `dataset_loaders/jamendo.py`.** Read frozen split lists from `splits/final/`.

5. **`trainer.py`: training and evaluation.**
   - `globalmood_va` removed from all dataset lists; a run that includes it raises an error.
   - `handle_dataset`: Jamendo trains the mood head only (its tag-derived VA is never added to the loss); MERGE routes through lyrics fusion when lyrics are on.
   - `on_test_end`: writes metrics JSON and per-song predictions for the bootstrap.

6. **`train.py`, `data_loader.py`, `test.py`: reproducibility.** `seed` config with `pl.seed_everything` and seeded DataLoader workers and shuffling. `run_name` gives each run its own log folder. Test uses the named run and the selection rule (mood checkpoint for Jamendo metrics, VA checkpoint for MERGE metrics).

7. **`runs/run_one.sh`: one job per matrix row.** Train, then evaluate with both checkpoints.

8. **`tools/analyze_v2.py`: statistics.** Mean ± SD over seeds; paired bootstrap (10,000 resamples of test songs) of the mean per-seed R² difference; pre-registered verdict.

## Results to review

`runs/results/results_table.md`, `runs/results/comparisons.md`, `runs/results_diag/` (official-split diagnostic).

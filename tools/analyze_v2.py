#!/usr/bin/env python
"""
analyze_v2.py -- rerun-v2 results: seed summaries, paired bootstrap, verdicts.

Reads runs/run_matrix.tsv and, for each run folder
tb_logs/train_audio_classification/<run_name>/ :
  test_metrics_mood.json            Jamendo metrics (mood-selected checkpoint)
  test_metrics_va.json              MERGE metrics   (VA-selected checkpoint)
  test_metrics_va_preds_merge.npz   per-song MERGE predictions (VA checkpoint)

Writes to runs/results/:
  results_table.md / .csv     mean ± SD over seeds per configuration
  comparisons.md / .csv       paired bootstrap over MERGE test songs
  per_run.csv                 every run's metrics

Bootstrap statistic: for each resample of test songs, the mean over seeds of
[R²(A, seed s) − R²(B, seed s)], i.e. the expected single-model difference.
Seeds are paired (same seed in A and B). 95% percentile CI; two-sided p.

Run from repo root:  python tools/analyze_v2.py [--boot 10000]
"""
import os, json, argparse
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--matrix", default="runs/run_matrix.tsv")
ap.add_argument("--logdir", default="tb_logs/train_audio_classification")
ap.add_argument("--out", default="runs/results")
ap.add_argument("--boot", type=int, default=10000)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

NAMES = {
    "C1": "Audio baseline (Jamendo + MERGE)",
    "C2": "+ GlobalMood multitask",
    "C3": "+ XLM-R lyrics (empty 4-dim slot)",
    "C4": "+ XLM-R + rhyme density",
    "K1": "Control: shuffled rhyme vectors",
    "K2": "Control: random 4-dim noise",
    "K3": "Control: non-rhyme lyric features",
    "X1": "Ablation: rhyme without coverage",
    "X2": "Ablation: coverage only",
    "X3": "Ablation: end-rhyme density only",
    "X4": "Ablation: internal-rhyme density only",
    "X5": "Ablation: rhymes per line only",
    "D3": "Official split: + XLM-R lyrics (empty slot)",
    "D4": "Official split: + XLM-R + rhyme density",
}
COMPARISONS = [  # (A, B, question) -- A minus B
    ("C4", "C3", "Rhyme beyond XLM-R semantics (primary)"),
    ("C4", "K1", "Song-specific rhyme vs. rhyme distribution"),
    ("C4", "K2", "Rhyme vs. any extra 4-dim input"),
    ("C4", "K3", "Rhyme vs. non-rhyme lyric structure"),
    ("C3", "C2", "XLM-R lyrics beyond audio multitask"),
    ("C2", "C1", "GlobalMood multitask beyond audio baseline"),
    ("D4", "D3", "Rhyme beyond XLM-R on the official (artist-overlapping) split"),
]
METRICS = [  # (column, json file, json key, label)
    ("pr_auc", "mood", "test_pr_auc_jamendo", "Jamendo PR-AUC"),
    ("roc_auc", "mood", "test_roc_auc_jamendo", "Jamendo ROC-AUC"),
    ("r2", "va", "test_r2_merge", "MERGE R²"),
    ("r2_val", "va", "test_r2_valence_merge", "MERGE valence R²"),
    ("r2_aro", "va", "test_r2_arousal_merge", "MERGE arousal R²"),
    ("rmse", "va", "test_rmse_merge", "MERGE RMSE"),
]

mx = pd.read_csv(args.matrix, sep="\t")

# ------------------------------------------------------------- collect runs
rows, preds, missing = [], {}, []
paths_ref = None
for r in mx.itertuples():
    d = os.path.join(args.logdir, r.run_name)
    js = {}
    for sel in ("mood", "va"):
        p = os.path.join(d, f"test_metrics_{sel}.json")
        if os.path.exists(p):
            js[sel] = json.load(open(p))
    if len(js) < 2:
        missing.append(r.run_name)
        continue
    row = dict(run_name=r.run_name, config=r.config, seed=r.seed, tier=r.tier)
    for col, sel, key, _ in METRICS:
        row[col] = js[sel].get(key, np.nan)
    rows.append(row)
    npz = os.path.join(d, "test_metrics_va_preds_merge.npz")
    if os.path.exists(npz):
        z = np.load(npz, allow_pickle=True)
        if paths_ref is None:
            paths_ref = z["paths"]
        assert np.array_equal(z["paths"], paths_ref), f"{r.run_name}: MERGE test order differs"
        preds[(r.config, int(r.seed))] = z["preds"].astype(np.float64)
        gt = z["gt"].astype(np.float64)

runs = pd.DataFrame(rows)
runs.to_csv(os.path.join(args.out, "per_run.csv"), index=False)
print(f"runs found: {len(runs)}/{len(mx)}" + (f"  missing: {', '.join(missing)}" if missing else ""))
if runs.empty:
    raise SystemExit("no completed runs yet")


# ----------------------------------------------------------- seed summaries
def fmt(m, s, n):
    return f"{m:.4f}" if n < 2 else f"{m:.4f} ± {s:.4f}"


order = [c for c in NAMES if c in set(runs["config"])]
summ = []
for c in order:
    g = runs[runs["config"] == c]
    rec = {"ID": c, "Configuration": NAMES[c], "Seeds": len(g)}
    for col, _, _, lab in METRICS:
        rec[lab] = fmt(g[col].mean(), g[col].std(ddof=1) if len(g) > 1 else 0, len(g))
        rec[col + "_mean"] = g[col].mean()
        rec[col + "_sd"] = g[col].std(ddof=1) if len(g) > 1 else np.nan
    summ.append(rec)
summ = pd.DataFrame(summ)
summ.to_csv(os.path.join(args.out, "results_table.csv"), index=False)

show = ["Configuration", "Seeds"] + [m[3] for m in METRICS]
with open(os.path.join(args.out, "results_table.md"), "w") as f:
    for tier, title in [("core", "Core configurations"), ("control", "Controls"),
                        ("exploratory", "Exploratory ablations (single seed)"),
                        ("diagnostic", "Diagnostic: official MERGE split (artist overlap)")]:
        ids = [c for c in order if runs.loc[runs["config"] == c, "tier"].iloc[0] == tier]
        if not ids:
            continue
        t = summ[summ["ID"].isin(ids)][show]
        f.write(f"### {title}\n\n| " + " | ".join(show) + " |\n|" + "---|" * len(show) + "\n")
        for _, rr in t.iterrows():
            f.write("| " + " | ".join(str(rr[k]) for k in show) + " |\n")
        split_note = ("official tvt_70_15_15 MERGE test split" if tier == "diagnostic"
                      else "artist-disjoint MERGE test split")
        f.write("\nMean ± SD over seeds. Jamendo: mood-selected checkpoint; "
                f"MERGE: VA-selected checkpoint; {split_note}.\n\n")
print(open(os.path.join(args.out, "results_table.md")).read())


# --------------------------------------------------------- paired bootstrap
def r2_cols(p, y):
    """R² per column; p,y: (..., N, 2) -> (..., 2)"""
    sse = ((p - y) ** 2).sum(-2)
    sst = ((y - y.mean(-2, keepdims=True)) ** 2).sum(-2)
    return 1 - sse / sst


rng = np.random.default_rng(args.seed)
comp_rows = []
if preds:
    N = gt.shape[0]
    idx = rng.integers(0, N, size=(args.boot, N))
    Yb = gt[idx]                                        # (B, N, 2)
    for a, b, q in COMPARISONS:
        seeds = sorted({s for (c, s) in preds if c == a} & {s for (c, s) in preds if c == b})
        if not seeds:
            continue
        d_full, d_boot = [], []
        for s in seeds:
            ra = r2_cols(preds[(a, s)], gt); rb = r2_cols(preds[(b, s)], gt)
            d_full.append(np.r_[ra - rb, (ra - rb).mean()])
            ba = r2_cols(preds[(a, s)][idx], Yb); bb = r2_cols(preds[(b, s)][idx], Yb)
            d = ba - bb
            d_boot.append(np.c_[d, d.mean(1)])          # (B, 3): valence, arousal, combined
        d_full = np.mean(d_full, 0); d_boot = np.mean(d_boot, 0)
        for k, lab in enumerate(["valence R²", "arousal R²", "combined R²"]):
            bs = d_boot[:, k]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            p = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))
            comp_rows.append(dict(A=a, B=b, question=q, metric=lab, seeds=len(seeds),
                                  diff=d_full[k], ci_lo=lo, ci_hi=hi, p=p,
                                  ci_excludes_0=bool(lo > 0 or hi < 0)))
comp = pd.DataFrame(comp_rows)
comp.to_csv(os.path.join(args.out, "comparisons.csv"), index=False)

with open(os.path.join(args.out, "comparisons.md"), "w") as f:
    f.write("### Paired comparisons on MERGE test (A − B)\n\n"
            "| A − B | Question | Metric | Seeds | Δ | 95% CI | p |\n|---|---|---|---|---|---|---|\n")
    for r in comp.itertuples():
        f.write(f"| {r.A} − {r.B} | {r.question} | {r.metric} | {r.seeds} | {r.diff:+.4f} | "
                f"[{r.ci_lo:+.4f}, {r.ci_hi:+.4f}] | {r.p:.4f} |\n")
    f.write(f"\nΔ = mean over paired seeds of R²(A) − R²(B); CI and p from {args.boot} "
            "paired bootstrap resamples of test songs.\n")

    # verdict against the pre-registered criteria (runs/README.md)
    def sig_pos(a, b):
        r = comp[(comp.A == a) & (comp.B == b) & (comp.metric == "valence R²")]
        return None if r.empty else bool(r.ci_lo.iloc[0] > 0)

    v = {k: sig_pos("C4", k) for k in ("C3", "K1", "K2", "K3")}
    f.write("\n### Pre-registered criteria (valence R²)\n\n")
    for k, lab in [("C3", "C4 > C3"), ("K1", "C4 > K1"), ("K2", "C4 > K2"), ("K3", "C4 > K3")]:
        f.write(f"- {lab}: {'n/a (runs missing)' if v[k] is None else ('CI above 0' if v[k] else 'CI includes 0 or below')}\n")
    if None not in (v["C3"], v["K1"], v["K2"]):
        if v["C3"] and v["K1"] and v["K2"]:
            f.write("- Rhyme effect: **supported**" + (" (also exceeds non-rhyme lyric control)" if v["K3"]
                    else " but not distinguishable from non-rhyme lyric structure (K3)") + ".\n")
        elif not v["C3"]:
            f.write("- Rhyme effect: **not detected** under these controls.\n")
        else:
            f.write("- Rhyme effect: C4 > C3, but not above the K1/K2 controls; **not attributable to rhyme**.\n")
print(open(os.path.join(args.out, "comparisons.md")).read())

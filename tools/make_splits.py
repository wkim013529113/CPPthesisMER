#!/usr/bin/env python
"""
make_splits.py -- build the final frozen splits for rerun-v2.

MERGE      : artist-disjoint 70/15/15, stratified by quadrant (new).
             Written in the shipped CSV format (Song,Quadrant) to
             <MERGE>/tvt_dataframes/tvt_artist_70_15_15/ so the loader only
             needs the split-folder name changed.
GlobalMood : exact replica of the current loader split (rng 42, 80/10/10).
Jamendo    : MTG split-0 pickles, minus test tracks that duplicate train audio.

All final ID lists go to splits/final/. Also writes splits/final/rhyme_stats.json
(train-split mean/std, .npy order) for z-scoring.

Run from the repo root:  python tools/make_splits.py
"""
import os, re, json, pickle, argparse
from collections import defaultdict
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--ratios", default="0.70,0.15,0.15")
ap.add_argument("--out", default="splits/final")
args = ap.parse_args()

OUT = args.out
os.makedirs(OUT, exist_ok=True)
SPLITS = ["train", "val", "test"]
RATIOS = [float(x) for x in args.ratios.split(",")]

M = "dataset/merge/MERGE_Bimodal_Balanced"
G = "dataset/globalmood"
JD = "dataset/jamendo"
SPLIT_NAME = "tvt_artist_70_15_15"
RCOLS_NPY = ["end_rhyme_density", "internal_rhyme_density",
             "rhymes_per_line", "cmudict_coverage"]

# Identical-audio pairs found by data_audit.py --near-dup (centred cosine >= 0.98)
MERGE_SAME_AUDIO = [("MT0009795009", "A066-140"), ("MT0011933875", "A114-120")]
JAMENDO_TEST_DUPES = [24836, 1420543]   # test copies of train tracks 17146, 1420470

LINES = []


def log(s=""):
    print(s, flush=True)
    LINES.append(s)


def write_ids(name, ids):
    with open(os.path.join(OUT, f"{name}.txt"), "w") as f:
        f.write("\n".join(map(str, ids)) + "\n")


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def primary_artist(a):
    """'X feat. Y' -> 'x'; 'The Beatles' -> 'beatles'."""
    a = re.split(r"\s+(feat\.?|ft\.?|featuring|with)\s+", str(a), flags=re.I)[0]
    a = re.sub(r"^the\s+", "", a.strip(), flags=re.I)
    return norm(a)


class UnionFind:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


# ===================================================================== MERGE
log("# Final splits — rerun-v2\n")
log("## MERGE: artist-disjoint, stratified by quadrant\n")

meta = pd.read_csv(f"{M}/merge_bimodal_balanced_metadata.csv")
av = pd.read_csv(f"{M}/merge_bimodal_balanced_av_values.csv")
df = meta[["Audio_Song", "Artist", "Title", "Quadrant"]].merge(
    av[["Audio_Song", "Lyric_Song", "Valence", "Arousal"]], on="Audio_Song")
assert len(df) == 2000, f"expected 2000 MERGE songs, got {len(df)}"
df["artist_key"] = df["Artist"].map(primary_artist)

# groups: same primary artist, or identical audio
uf = UnionFind()
for sid, ak in zip(df["Audio_Song"], df["artist_key"]):
    uf.union("song:" + sid, "artist:" + ak)
for a, b in MERGE_SAME_AUDIO:
    if a in set(df["Audio_Song"]) and b in set(df["Audio_Song"]):
        uf.union("song:" + a, "song:" + b)
    else:
        log(f"  !! same-audio pair not found: {a}, {b}")
df["group"] = [uf.find("song:" + s) for s in df["Audio_Song"]]

quads = sorted(df["Quadrant"].unique())
gq = df.groupby(["group", "Quadrant"]).size().unstack(fill_value=0)[quads]
sizes = gq.sum(axis=1)
log(f"  {len(df)} songs, {df['artist_key'].nunique()} artists, {len(gq)} groups; "
    f"largest group {int(sizes.max())} songs ({sizes.idxmax().split(':', 1)[1]})")

# greedy assignment: largest groups first (ties broken by seeded shuffle),
# each to the split that most reduces squared deviation from per-quadrant targets
rng = np.random.default_rng(args.seed)
order = list(gq.index)
rng.shuffle(order)
order.sort(key=lambda g: -sizes[g])
qtot = gq.sum(axis=0).values.astype(float)
target = np.outer(RATIOS, qtot)                  # [3, n_quads]
cur = np.zeros_like(target)
assign = {}
for g in order:
    c = gq.loc[g].values
    cost = [((cur[s] + c - target[s]) ** 2).sum() - ((cur[s] - target[s]) ** 2).sum()
            for s in range(3)]
    s = int(np.argmin(cost))
    cur[s] += c
    assign[g] = SPLITS[s]
df["split"] = df["group"].map(assign)

# write in shipped CSV format + ID lists
sdir = f"{M}/tvt_dataframes/{SPLIT_NAME}"
os.makedirs(sdir, exist_ok=True)
fname = {"train": "train", "val": "validate", "test": "test"}
for s in SPLITS:
    sub = df[df["split"] == s].sort_values("Audio_Song")
    sub[["Audio_Song", "Quadrant"]].rename(columns={"Audio_Song": "Song"}).to_csv(
        f"{sdir}/{SPLIT_NAME}_{fname[s]}_bimodal_balanced.csv", index=False)
    write_ids(f"merge_{s}", sub["Audio_Song"].tolist())

# verification
log("\n  split   n     " + "  ".join(f"{q:>4s}" for q in quads) + "   val_mean aro_mean")
for s in SPLITS:
    sub = df[df["split"] == s]
    qc = sub["Quadrant"].value_counts().reindex(quads, fill_value=0)
    log(f"  {s:5s} {len(sub):5d}   " + "  ".join(f"{v:4d}" for v in qc.values)
        + f"   {sub['Valence'].mean():.4f}   {sub['Arousal'].mean():.4f}")
for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
    A = df[df["split"] == a]; B = df[df["split"] == b]
    art = set(A["artist_key"]) & set(B["artist_key"])
    at = (set(A["artist_key"] + "|" + A["Title"].map(norm))
          & set(B["artist_key"] + "|" + B["Title"].map(norm)))
    log(f"  {a}∩{b}: artists {len(art)}  artist+title {len(at)}")
for a, b in MERGE_SAME_AUDIO:
    sa = df.loc[df["Audio_Song"] == a, "split"].tolist()
    sb = df.loc[df["Audio_Song"] == b, "split"].tolist()
    log(f"  same-audio {a} -> {sa}, {b} -> {sb}")
log(f"  wrote {sdir}/")

# rhyme stats on the NEW train split
tr = df[df["split"] == "train"]
X = np.stack([np.load(f"{M}/rhyme/{l}.npy").astype(np.float64).ravel()
              for l in tr["Lyric_Song"]])
mu, sd = X.mean(0), X.std(0)
json.dump({"columns": RCOLS_NPY, "mean": mu.tolist(), "std": sd.tolist(),
           "source": f"MERGE {SPLIT_NAME} train split ({len(tr)} songs)"},
          open(os.path.join(OUT, "rhyme_stats.json"), "w"), indent=2)
log("\n  rhyme stats (new train split, .npy order):")
for c, m_, s_ in zip(RCOLS_NPY, mu, sd):
    log(f"    {c:24s} mean {m_:.4f}  std {s_:.4f}")

# ================================================================ GlobalMood
log("\n## GlobalMood: loader replica (rng 42, 80/10/10)\n")
gdf = pd.read_csv(f"{G}/globalmood_labels.csv")
mdir = f"{G}/mert_30s"
avail = {n for n in os.listdir(mdir) if os.path.isdir(os.path.join(mdir, n))}
gdf = gdf[gdf["videoID"].isin(avail)].reset_index(drop=True)
idx = np.random.default_rng(42).permutation(len(gdf))
n = len(gdf); n_tr = int(n * 0.8); n_val = int(n * 0.1)
parts = {"train": idx[:n_tr], "val": idx[n_tr:n_tr + n_val], "test": idx[n_tr + n_val:]}
for s in SPLITS:
    ids = gdf.iloc[parts[s]]["videoID"].tolist()
    write_ids(f"globalmood_{s}", ids)
    log(f"  {s}: {len(ids)}")

# =================================================================== Jamendo
log("\n## Jamendo: MTG split-0 pickles, test duplicates removed\n")
jf = {"train": "train", "val": "validation", "test": "test"}
for s in SPLITS:
    pk = pickle.load(open(f"{JD}/splits/split-0/moodtheme_{jf[s]}_dict.pickle", "rb"))
    ids = [int(os.path.splitext(os.path.basename(v["path"]))[0]) for v in pk.values()]
    n0 = len(ids)
    if s == "test":
        ids = [i for i in ids if i not in set(JAMENDO_TEST_DUPES)]
    write_ids(f"jamendo_{s}", ids)
    log(f"  {s}: {len(ids)}" + (f"  (removed {n0 - len(ids)} duplicates)" if n0 != len(ids) else ""))

open(os.path.join(OUT, "splits_report.md"), "w").write("\n".join(LINES) + "\n")
log(f"\nwrote {OUT}/ (ID lists, rhyme_stats.json, splits_report.md)")

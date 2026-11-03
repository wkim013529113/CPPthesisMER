#!/usr/bin/env python
"""
make_surface_features.py -- K3 control: non-rhyme lyric surface features.

For every MERGE lyric, writes <MERGE>/surface/<L-ID>.npy, a 4-dim vector:
  [0] log(1 + n_lines)        non-empty lines
  [1] log(1 + n_words)        tokens matching [A-Za-z']+
  [2] mean_word_length        characters per token
  [3] type_token_ratio        unique lowercased tokens / n_words
Same 4-dim slot as the rhyme vector, but carries lyric length/structure and
vocabulary only, no phonology. Writes splits/final/surface_stats.json with
mean/std from the MERGE train split (splits/final/merge_train.txt).

Run from repo root:  python tools/make_surface_features.py
"""
import os, re, json
import numpy as np
import pandas as pd

M = "dataset/merge/MERGE_Bimodal_Balanced"
COLS = ["log1p_n_lines", "log1p_n_words", "mean_word_length", "type_token_ratio"]
TOK = re.compile(r"[A-Za-z']+")

meta = pd.read_csv(f"{M}/merge_bimodal_balanced_metadata.csv")[["Audio_Song", "Quadrant"]]
av = pd.read_csv(f"{M}/merge_bimodal_balanced_av_values.csv")[["Audio_Song", "Lyric_Song"]]
df = meta.merge(av, on="Audio_Song")
os.makedirs(f"{M}/surface", exist_ok=True)

feats, empty = {}, 0
for q, l in zip(df["Quadrant"], df["Lyric_Song"]):
    with open(f"{M}/lyrics/{q}/{l}.txt", encoding="utf-8", errors="replace") as f:
        text = f.read()
    lines = [x for x in text.splitlines() if x.strip()]
    toks = TOK.findall(text)
    if not toks:
        empty += 1
    nw = len(toks)
    v = np.array([np.log1p(len(lines)), np.log1p(nw),
                  np.mean([len(t) for t in toks]) if toks else 0.0,
                  len({t.lower() for t in toks}) / nw if nw else 0.0], dtype=np.float32)
    np.save(f"{M}/surface/{l}.npy", v)
    feats[l] = v

train_audio = [x.strip() for x in open("splits/final/merge_train.txt") if x.strip()]
train_lyr = df.set_index("Audio_Song").loc[train_audio, "Lyric_Song"]
X = np.stack([feats[l] for l in train_lyr])
st = {"columns": COLS, "mean": X.mean(0).tolist(), "std": X.std(0).tolist(),
      "source": f"MERGE tvt_artist_70_15_15 train split ({len(X)} songs)"}
json.dump(st, open("splits/final/surface_stats.json", "w"), indent=2)
print(f"wrote {len(feats)} surface vectors to {M}/surface/ (empty lyrics: {empty})")
for c, m, s in zip(COLS, st["mean"], st["std"]):
    print(f"  {c:18s} mean {m:.3f}  std {s:.3f}")
# how much the control overlaps the rhyme vector (should be partial, not total)
R = np.stack([np.load(f"{M}/rhyme/{l}.npy").ravel() for l in train_lyr])
C = np.corrcoef(np.hstack([X, R]).T)[:4, 4:]
print("  corr(surface, rhyme) rows=surface, cols=[end, internal, per_line, coverage]")
print(np.round(C, 2))

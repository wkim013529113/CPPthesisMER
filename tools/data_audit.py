#!/usr/bin/env python
"""
data_audit.py -- rerun-v2 data audit for the rhyme-density MER thesis.

Freezes train/val/test splits to text files and checks coverage, leakage,
label ranges and rhyme features for MERGE, GlobalMood and MTG-Jamendo.

Run from the repo root (via sbatch, not the login node):
    python tools/data_audit.py                 # core audit
    python tools/data_audit.py --near-dup      # + MERT near-duplicate check

Outputs (in --out, default ./splits):
    {merge,globalmood,jamendo}_{train,val,test}.txt   frozen split IDs
    rhyme_stats.json                                  train-split mean/std
    data_audit.md / data_audit.json                   report
"""
import os, re, sys, json, pickle, argparse, traceback
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=".")
ap.add_argument("--out", default="splits")
ap.add_argument("--near-dup", action="store_true")
ap.add_argument("--near-dup-thresh", type=float, default=0.98)
ap.add_argument("--layers", default="5,6")
ap.add_argument("--merge-split", default="tvt_70_15_15",
                help="MERGE split folder under tvt_dataframes/ (e.g. tvt_artist_70_15_15)")
args = ap.parse_args()

os.chdir(args.root)
OUT = args.out
os.makedirs(OUT, exist_ok=True)
LAYERS = [int(x) for x in args.layers.split(",")]
SPLITS = ["train", "val", "test"]

LINES, J = [], {}


def log(s=""):
    print(s, flush=True)
    LINES.append(s)


def write_ids(name, ids):
    with open(os.path.join(OUT, f"{name}.txt"), "w") as f:
        f.write("\n".join(map(str, ids)) + "\n")


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def nonempty_dir(p):
    return os.path.isdir(p) and any(f.endswith(".npy") for f in os.listdir(p))


def overlap_report(name, groups):
    """groups: {split: set(keys)}. Logs pairwise overlaps."""
    out = {}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        inter = groups[a] & groups[b]
        out[f"{a}&{b}"] = len(inter)
        ex = ", ".join(sorted(map(str, inter))[:5])
        log(f"  {name} {a}∩{b}: {len(inter)}" + (f"  e.g. {ex}" if inter else ""))
    return out


def section(title, fn):
    log(f"\n## {title}\n")
    try:
        fn()
    except Exception:
        log("  !! section failed:")
        log("  " + traceback.format_exc().replace("\n", "\n  "))


# ----------------------------------------------------------------- MERGE
M = "dataset/merge/MERGE_Bimodal_Balanced"
# Order of values inside rhyme/<L-ID>.npy (lyrics_feature_extractor.py docstring);
# the CSV report stores the same values in a different column order.
RCOLS = ["end_rhyme_density", "internal_rhyme_density",
         "rhymes_per_line", "cmudict_coverage"]
merge_split = {}


def audit_merge():
    av = pd.read_csv(f"{M}/merge_bimodal_balanced_av_values.csv")
    meta = pd.read_csv(f"{M}/merge_bimodal_balanced_metadata.csv")
    rep = pd.read_csv(f"{M}/lyrics_features_report.csv").set_index("lyric_id")
    fmap = {"train": "train", "val": "validate", "test": "test"}
    jm = {}

    for s in SPLITS:
        df = pd.read_csv(f"{M}/tvt_dataframes/{args.merge_split}/"
                         f"{args.merge_split}_{fmap[s]}_bimodal_balanced.csv")
        j = df.merge(av, left_on="Song", right_on="Audio_Song", how="left")
        j = j.merge(meta[["Audio_Song", "Artist", "Title", "Quadrant"]]
                    .rename(columns={"Quadrant": "Quadrant_meta"}),
                    on="Audio_Song", how="left")
        merge_split[s] = j
        write_ids(f"merge_{s}", df["Song"].tolist())

        miss_av = int(j["Lyric_Song"].isna().sum())
        ok = j.dropna(subset=["Lyric_Song"])
        mert_ok = sum(nonempty_dir(f"{M}/mert_30s/{a}") for a in ok["Audio_Song"])
        xlmr_ok = sum(os.path.exists(f"{M}/xlmr/{l}.npy") for l in ok["Lyric_Song"])
        rh_ok = sum(os.path.exists(f"{M}/rhyme/{l}.npy") for l in ok["Lyric_Song"])
        lyr_ok = sum(os.path.exists(f"{M}/lyrics/{q}/{l}.txt")
                     for q, l in zip(ok["Quadrant"], ok["Lyric_Song"]))
        q_mis = int((ok["Quadrant"] != ok["Quadrant_meta"]).sum())
        dup = int(df["Song"].duplicated().sum())

        v, a = ok["Valence"], ok["Arousal"]
        log(f"### MERGE {s}: {len(df)} songs")
        log(f"  missing AV row: {miss_av} | duplicate IDs: {dup} | "
            f"quadrant mismatch split-vs-meta: {q_mis}")
        log(f"  coverage  MERT {mert_ok}/{len(ok)}  XLM-R {xlmr_ok}/{len(ok)}  "
            f"rhyme {rh_ok}/{len(ok)}  lyric txt {lyr_ok}/{len(ok)}")
        log(f"  valence raw [{v.min():.4f}, {v.max():.4f}] mean {v.mean():.4f} "
            f"-> rescaled [{1+8*v.min():.3f}, {1+8*v.max():.3f}]")
        log(f"  arousal raw [{a.min():.4f}, {a.max():.4f}] mean {a.mean():.4f} "
            f"-> rescaled [{1+8*a.min():.3f}, {1+8*a.max():.3f}]")
        jm[s] = dict(n=len(df), missing_av=miss_av, dup=dup, quad_mismatch=q_mis,
                     mert=mert_ok, xlmr=xlmr_ok, rhyme=rh_ok, lyric=lyr_ok,
                     val_range=[float(v.min()), float(v.max())],
                     aro_range=[float(a.min()), float(a.max())])

    # --- rhyme: cached .npy vs report row, and train-split stats
    log("\n### MERGE rhyme vectors")
    vecs, diffs, shapes, not_in_rep = {}, [], set(), 0
    for s in SPLITS:
        for l in merge_split[s]["Lyric_Song"].dropna():
            p = f"{M}/rhyme/{l}.npy"
            if not os.path.exists(p):
                continue
            x = np.load(p).astype(np.float64).ravel()
            shapes.add(x.shape)
            vecs.setdefault(s, []).append(x)
            if l in rep.index:
                diffs.append(np.abs(x[:4] - rep.loc[l, RCOLS].values.astype(float)))
            else:
                not_in_rep += 1
    log(f"  vector shapes: {sorted(shapes)}")
    if diffs:
        d = np.vstack(diffs)
        log(f"  |npy - report| max per column {dict(zip(RCOLS, d.max(0).round(5)))}")
        log(f"  rows with any diff > 1e-3: {int((d.max(1) > 1e-3).sum())}/{len(d)}"
            f"  (0 => cached vector order = {RCOLS})")
    log(f"  lyric IDs absent from report: {not_in_rep}")
    tr = np.vstack(vecs["train"])
    mu, sd = tr.mean(0), tr.std(0)
    log("  train-split stats (unnormalized, as currently fed to the model):")
    for i, c in enumerate(RCOLS[:len(mu)]):
        col = tr[:, i]
        log(f"    {c:24s} mean {mu[i]:.4f}  std {sd[i]:.4f}  "
            f"min {col.min():.4f}  max {col.max():.4f}")
    # extremes: songs driving the tails (often lyric files with few line breaks)
    tr_ids = [l for l in merge_split["train"]["Lyric_Song"].dropna()
              if os.path.exists(f"{M}/rhyme/{l}.npy")]
    z = (tr - mu) / np.where(sd > 0, sd, 1)
    log("  most extreme train songs (max |z| over the 4 features):")
    for k in np.argsort(-np.abs(z).max(1))[:8]:
        l = tr_ids[k]
        nl = rep.loc[l, "n_lines"] if l in rep.index else "?"
        nw = rep.loc[l, "n_words"] if l in rep.index else "?"
        log(f"    {l:12s} z={np.round(z[k], 1).tolist()}  n_lines {nl}  n_words {nw}")
    log(f"  train songs with any |z| > 5: {int((np.abs(z) > 5).any(1).sum())}/{len(z)}")
    json.dump({"columns": RCOLS[:len(mu)], "mean": mu.tolist(), "std": sd.tolist(),
               "source": "MERGE tvt_70_15_15 train split"},
              open(os.path.join(OUT, "rhyme_stats.json"), "w"), indent=2)
    log(f"  wrote {OUT}/rhyme_stats.json")

    # --- leakage within MERGE
    log("\n### MERGE leakage")
    jm["song_id_overlap"] = overlap_report(
        "song ID", {s: set(merge_split[s]["Song"]) for s in SPLITS})
    jm["artist_overlap"] = overlap_report(
        "artist", {s: set(merge_split[s]["Artist"].dropna().map(norm)) for s in SPLITS})
    tr_art = set(merge_split["train"]["Artist"].dropna().map(norm))
    for s in ["val", "test"]:
        n = int(merge_split[s]["Artist"].map(norm).isin(tr_art).sum())
        log(f"  {s} songs whose artist is also in train: {n}/{len(merge_split[s])}")
        jm[f"{s}_songs_artist_in_train"] = n
    jm["artist_title_overlap"] = overlap_report(
        "artist+title", {s: set((merge_split[s]["Artist"].map(norm) + "|" +
                                 merge_split[s]["Title"].map(norm))) for s in SPLITS})
    J["merge"] = jm


# ------------------------------------------------------------ GlobalMood
G = "dataset/globalmood"
gm_split = {}


def audit_globalmood():
    jg = {}
    df = pd.read_csv(f"{G}/globalmood_labels.csv")
    mert_dir = f"{G}/mert_30s"   # loader's self.mert_dir; confirm vs training log
    avail = {n for n in os.listdir(mert_dir) if os.path.isdir(os.path.join(mert_dir, n))}
    n_all = len(df)
    df = df[df["videoID"].isin(avail)].reset_index(drop=True)
    log(f"  labels rows {n_all}, with MERT dir {len(df)}  "
        f"(compare with '[GlobalMoodDataset] N songs' in a training log)")

    # exact replica of dataset_loaders/globalmood.py split
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(df))
    n = len(df); n_tr = int(n * 0.8); n_val = int(n * 0.1)
    parts = {"train": idx[:n_tr], "val": idx[n_tr:n_tr + n_val], "test": idx[n_tr + n_val:]}

    meta = (pd.read_csv(f"{G}/Csvs/songmeta_GlobalMood.csv")
            .drop_duplicates("videoID"))
    tags = [c for c in df.columns if c != "videoID"]
    for s in SPLITS:
        sub = df.iloc[parts[s]].reset_index(drop=True)
        sub = sub.merge(meta, on="videoID", how="left")
        gm_split[s] = sub
        write_ids(f"globalmood_{s}", sub["videoID"].tolist())
        mert_ok = sum(nonempty_dir(f"{mert_dir}/{v}") for v in sub["videoID"])
        vals = sub[tags].values.astype(float)
        log(f"### GlobalMood {s}: {len(sub)} songs | MERT non-empty {mert_ok}/{len(sub)} "
            f"| no songmeta {int(sub['artist'].isna().sum())}")
        log(f"  tag values [{np.nanmin(vals):.4f}, {np.nanmax(vals):.4f}] "
            f"NaN {int(np.isnan(vals).sum())} | {len(tags)} tags")
        jg[s] = dict(n=len(sub), mert=mert_ok, n_tags=len(tags),
                     nan=int(np.isnan(vals).sum()))

    log("\n### GlobalMood leakage")
    jg["artist_overlap"] = overlap_report(
        "artist", {s: set(gm_split[s]["artist"].dropna().map(norm)) for s in SPLITS})
    jg["artist_title_overlap"] = overlap_report(
        "artist+title", {s: set(gm_split[s]["artist"].map(norm) + "|" +
                                gm_split[s]["song"].map(norm)) for s in SPLITS})
    J["globalmood"] = jg


# ---------------------------------------------------------------- Jamendo
JD = "dataset/jamendo"
jam_split = {}


def read_mtg_tsv(path):
    rows = []
    with open(path) as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            rows.append(dict(track=int(p[0].split("_")[1]),
                             artist=p[1], album=p[2], tags=p[5:]))
    return pd.DataFrame(rows).set_index("track")


def pickle_ids(obj):
    """Best-effort track IDs from the loader pickle (dict or list)."""
    items = obj.items() if isinstance(obj, dict) else enumerate(obj)
    ids = []
    for k, v in items:
        src = v.get("path", k) if isinstance(v, dict) else (v if isinstance(v, str) else k)
        m = re.findall(r"\d+", os.path.splitext(os.path.basename(str(src)))[0])
        ids.append(int(m[-1]) if m else None)
    return ids


def audit_jamendo():
    jj = {}
    fmap = {"train": "train", "val": "validation", "test": "test"}
    for s in SPLITS:
        tsv = read_mtg_tsv(f"{JD}/splits/split-0/autotagging_moodtheme-{fmap[s]}.tsv")
        pk = pickle.load(open(f"{JD}/splits/split-0/moodtheme_{fmap[s]}_dict.pickle", "rb"))
        if s == "train":
            k0 = next(iter(pk)) if isinstance(pk, dict) else 0
            v0 = pk[k0]
            log(f"  pickle type {type(pk).__name__}, sample key {k0!r} -> "
                f"{str(v0)[:160]}")
        ids = pickle_ids(pk)
        bad = sum(i is None for i in ids)
        ids = [i for i in ids if i is not None]
        in_tsv = sum(i in tsv.index for i in ids)
        mert_ok = sum(nonempty_dir(f"{JD}/mert_30s/{i}") for i in ids)
        jam_split[s] = tsv.loc[[i for i in ids if i in tsv.index]]
        write_ids(f"jamendo_{s}", ids)
        log(f"### Jamendo {s}: tsv {len(tsv)} | pickle {len(ids)} (unparsed {bad}) | "
            f"pickle∩tsv {in_tsv} | MERT non-empty {mert_ok}/{len(ids)}")
        jj[s] = dict(tsv=len(tsv), pickle=len(ids), in_tsv=in_tsv, mert=mert_ok)

    log("\n### Jamendo leakage (pickle IDs, artists from MTG tsv)")
    jj["track_overlap"] = overlap_report(
        "track", {s: set(jam_split[s].index) for s in SPLITS})
    jj["artist_overlap"] = overlap_report(
        "artist", {s: set(jam_split[s]["artist"]) for s in SPLITS})
    jj["album_overlap"] = overlap_report(
        "album", {s: set(jam_split[s]["album"]) for s in SPLITS})
    J["jamendo"] = jj


# ---------------------------------------------------------- cross-dataset
def audit_cross():
    jc = {}
    gm_all = pd.concat(gm_split.values())
    gm_keys = set(gm_all["artist"].map(norm) + "|" + gm_all["song"].map(norm))
    gm_art = set(gm_all["artist"].dropna().map(norm))
    for s in ["val", "test"]:
        m = merge_split[s]
        k = m["Artist"].map(norm) + "|" + m["Title"].map(norm)
        n_song = int(k.isin(gm_keys).sum())
        n_art = int(m["Artist"].map(norm).isin(gm_art).sum())
        ex = ", ".join(sorted(k[k.isin(gm_keys)])[:5])
        log(f"  MERGE {s} songs also in GlobalMood (artist+title): {n_song}"
            + (f"  e.g. {ex}" if n_song else ""))
        log(f"  MERGE {s} songs whose artist appears in GlobalMood: {n_art}")
        jc[f"merge_{s}_in_globalmood_song"] = n_song
        jc[f"merge_{s}_in_globalmood_artist"] = n_art
    J["cross"] = jc


# ------------------------------------------------------ near-dup (MERT)
def mean_mert(d):
    if not nonempty_dir(d):
        return None
    segs = []
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".npy"):
            x = np.load(os.path.join(d, fn))
            segs.append(np.squeeze(np.concatenate([x[:, l, :] for l in LAYERS], axis=1)))
    return np.mean(np.array(segs), axis=0)


def audit_neardup():
    tracks = []   # (dataset, split, id, path)
    for s in SPLITS:
        tracks += [("merge", s, a, f"{M}/mert_30s/{a}") for a in merge_split[s]["Song"]]
        tracks += [("globalmood", s, v, f"{G}/mert_30s/{v}") for v in gm_split[s]["videoID"]]
        tracks += [("jamendo", s, t, f"{JD}/mert_30s/{t}") for t in jam_split[s].index]
    embs, keep = [], []
    for i, t in enumerate(tracks):
        e = mean_mert(t[3])
        if e is not None:
            embs.append(e.astype(np.float32)); keep.append(t)
        if i % 1000 == 0:
            print(f"  loaded {i}/{len(tracks)}", flush=True)
    X = np.vstack(embs)
    X -= X.mean(0)                      # centre: MERT means are anisotropic
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    is_tr = np.array([t[1] == "train" for t in keep])
    Xtr = X[is_tr]; tr_meta = [t for t in keep if t[1] == "train"]
    ev = [i for i, t in enumerate(keep) if t[1] != "train"]
    hits = []
    for b in range(0, len(ev), 512):
        idx = ev[b:b + 512]
        S = X[idx] @ Xtr.T
        best = S.argmax(1)
        for r, i in enumerate(idx):
            sc = float(S[r, best[r]])
            if sc >= args.near_dup_thresh:
                hits.append((sc, keep[i][:3], tr_meta[best[r]][:3]))
    hits.sort(reverse=True)
    log(f"  {len(keep)} tracks embedded; val/test tracks with centred cosine "
        f">= {args.near_dup_thresh} to some train track: {len(hits)}")
    for sc, a, b in hits[:25]:
        log(f"    {sc:.4f}  {a[0]}/{a[1]}/{a[2]}  ~  {b[0]}/{b[1]}/{b[2]}")
    J["near_dup"] = dict(thresh=args.near_dup_thresh, n_hits=len(hits),
                         top=[(sc, list(a), list(b)) for sc, a, b in hits[:100]])


# ------------------------------------------------------------------ main
log("# Data audit — rerun-v2\n")
section(f"MERGE ({args.merge_split})", audit_merge)
section("GlobalMood (loader replica, rng 42, 80/10/10)", audit_globalmood)
section("MTG-Jamendo moodtheme split-0", audit_jamendo)
section("Cross-dataset: MERGE val/test vs GlobalMood (training pool)", audit_cross)
if args.near_dup:
    section("Near-duplicate audio (mean MERT, centred cosine)", audit_neardup)

open(os.path.join(OUT, "data_audit.md"), "w").write("\n".join(LINES) + "\n")
json.dump(J, open(os.path.join(OUT, "data_audit.json"), "w"), indent=2, default=str)
log(f"\nwrote {OUT}/data_audit.md, {OUT}/data_audit.json and split files")

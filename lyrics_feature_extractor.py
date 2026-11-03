"""
Stage 1 lyrics feature extraction for MERGE Bimodal Balanced.

Produces, per lyric file (lyrics/Q*/L-ID.txt):
  1. xlmr/<L-ID>.npy   -- 768-dim mean-pooled XLM-RoBERTa-base embedding
  2. rhyme/<L-ID>.npy   -- 4-dim rhyme feature vector
plus lyrics_features_report.csv for thesis notebook.

Rhyme features:
  [0] end_rhyme_density      : end-rhymed pairs in 3-line window / n_lines
  [1] internal_rhyme_density : within-line rhyme pairs / n_words
  [2] rhymes_per_line        : (end + internal pairs) / n_lines
  [3] cmudict_coverage       : fraction of tokens with CMUdict entry

Usage:
  python3 lyrics_feature_extractor.py --scan-only   # coverage scan only
  python3 lyrics_feature_extractor.py               # full extraction
"""

import argparse
import os
import re
import sys
import csv
import numpy as np

WORD_RE = re.compile(r"[a-zA-Z']+")
END_RHYME_WINDOW = 3


def build_rhyme_extractor():
    import pronouncing
    cache = {}

    def rhyming_part(word):
        w = word.lower().strip("'")
        if not w:
            return None
        if w in cache:
            return cache[w]
        phones = pronouncing.phones_for_word(w)
        rp = pronouncing.rhyming_part(phones[0]) if phones else None
        cache[w] = rp
        return rp

    def features(text):
        lines = [ln for ln in (l.strip() for l in text.splitlines()) if ln]
        if not lines:
            return np.zeros(4, dtype=np.float32), 0, 0

        line_words = [WORD_RE.findall(ln) for ln in lines]
        all_words = [w for ws in line_words for w in ws]
        n_words = len(all_words)
        n_lines = len(lines)
        if n_words == 0:
            return np.zeros(4, dtype=np.float32), n_lines, 0

        covered = sum(1 for w in all_words if rhyming_part(w) is not None)
        coverage = covered / n_words

        finals = []
        for ws in line_words:
            if ws:
                finals.append((rhyming_part(ws[-1]), ws[-1].lower()))
            else:
                finals.append((None, None))

        end_pairs = 0
        for i in range(n_lines):
            rp_i, w_i = finals[i]
            if rp_i is None:
                continue
            for j in range(i + 1, min(i + 1 + END_RHYME_WINDOW, n_lines)):
                rp_j, w_j = finals[j]
                if rp_j is not None and rp_j == rp_i and w_j != w_i:
                    end_pairs += 1

        internal_pairs = 0
        for ws in line_words:
            rps = [(rhyming_part(w), w.lower()) for w in ws]
            for a in range(len(rps)):
                rp_a, w_a = rps[a]
                if rp_a is None:
                    continue
                for b in range(a + 1, len(rps)):
                    rp_b, w_b = rps[b]
                    if rp_b is not None and rp_b == rp_a and w_b != w_a:
                        internal_pairs += 1

        feats = np.array([
            end_pairs / n_lines,
            internal_pairs / n_words,
            (end_pairs + internal_pairs) / n_lines,
            coverage,
        ], dtype=np.float32)
        return feats, n_lines, n_words

    return features


def build_xlmr_embedder(device):
    import torch
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    model = AutoModel.from_pretrained("xlm-roberta-base").to(device).eval()

    @torch.no_grad()
    def embed(text):
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            stride=64,
            return_overflowing_tokens=True,
            padding=True,
        )
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        out = model(input_ids=input_ids, attention_mask=attn).last_hidden_state
        mask = attn.unsqueeze(-1).float()
        chunk_sums = (out * mask).sum(dim=1)
        chunk_counts = mask.sum(dim=1)
        emb = chunk_sums.sum(dim=0) / chunk_counts.sum()
        return emb.cpu().numpy().astype(np.float32)

    return embed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset/merge/MERGE_Bimodal_Balanced")
    ap.add_argument("--scan-only", action="store_true",
                    help="Coverage scan only; no XLM-R, no .npy output")
    ap.add_argument("--coverage-flag", type=float, default=0.70,
                    help="Flag songs with CMUdict coverage below this")
    args = ap.parse_args()

    lyrics_dir = os.path.join(args.root, "lyrics")
    xlmr_dir = os.path.join(args.root, "xlmr")
    rhyme_dir = os.path.join(args.root, "rhyme")
    report_path = os.path.join(args.root, "lyrics_features_report.csv")

    files = []
    for quadrant in sorted(os.listdir(lyrics_dir)):
        qdir = os.path.join(lyrics_dir, quadrant)
        if not os.path.isdir(qdir):
            continue
        for fn in sorted(os.listdir(qdir)):
            if fn.endswith(".txt"):
                files.append((quadrant, fn[:-4], os.path.join(qdir, fn)))
    print("[lyrics-extract] Found {} lyric files under {}".format(len(files), lyrics_dir))
    if not files:
        sys.exit(1)

    rhyme_features = build_rhyme_extractor()

    embed = None
    if not args.scan_only:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("[lyrics-extract] Loading xlm-roberta-base on {}".format(device))
        embed = build_xlmr_embedder(device)
        os.makedirs(xlmr_dir, exist_ok=True)
        os.makedirs(rhyme_dir, exist_ok=True)

    rows = []
    flagged = []
    empty = []
    for k, (quadrant, lyric_id, path) in enumerate(files, 1):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read().strip()

        if not text:
            empty.append(lyric_id)

        feats, n_lines, n_words = rhyme_features(text)
        coverage = float(feats[3])
        if coverage < args.coverage_flag and n_words > 0:
            flagged.append((lyric_id, coverage))

        if not args.scan_only:
            np.save(os.path.join(rhyme_dir, "{}.npy".format(lyric_id)), feats)
            if text:
                emb = embed(text)
            else:
                emb = np.zeros(768, dtype=np.float32)
            np.save(os.path.join(xlmr_dir, "{}.npy".format(lyric_id)), emb)

        rows.append({
            "lyric_id": lyric_id,
            "quadrant": quadrant,
            "n_lines": n_lines,
            "n_words": n_words,
            "cmudict_coverage": round(coverage, 4),
            "end_rhyme_density": round(float(feats[0]), 4),
            "internal_rhyme_density": round(float(feats[1]), 4),
            "rhymes_per_line": round(float(feats[2]), 4),
        })
        if k % 200 == 0:
            print("[lyrics-extract] {}/{} done".format(k, len(files)))

    with open(report_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    cov = np.array([r["cmudict_coverage"] for r in rows if r["n_words"] > 0])
    print("[lyrics-extract] Report written: {}".format(report_path))
    print("[lyrics-extract] Coverage mean={:.3f} median={:.3f} min={:.3f}".format(
        cov.mean(), float(np.median(cov)), cov.min()))
    print("[lyrics-extract] Songs flagged below {:.0%} coverage: {}".format(
        args.coverage_flag, len(flagged)))
    for lid, c in flagged[:20]:
        print("    {}: {:.2%}".format(lid, c))
    if len(flagged) > 20:
        print("    ... and {} more (see report CSV)".format(len(flagged) - 20))
    if empty:
        print("[lyrics-extract] WARNING: {} empty lyric files: {}".format(len(empty), empty[:10]))


if __name__ == "__main__":
    main()

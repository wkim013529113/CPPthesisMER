# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import json
import zlib
import torch
from torch.utils import data


class MERGEDataset(data.Dataset):
    """
    MERGE Bimodal Balanced (Louro et al., 2024, arXiv 2407.06060).
    2000 samples, audio + lyrics, continuous AV labels (empirically [0,1]).
    Uses the dataset's fixed 70/15/15 splits.

    Lyrics branch (rung 5/6):
      When cfg.model.use_lyrics=True, loads pre-extracted features:
        xlmr/<L-ID>.npy   (768-dim XLM-RoBERTa-base embedding)
        rhyme/<L-ID>.npy  (4-dim rhyme density vector)
      produced by lyrics_feature_extractor.py.
    """
    def __init__(self, **task_args):
        self.task_args = task_args
        self.tr_val    = task_args.get('tr_val', 'train')
        self.root      = task_args.get('root', './dataset/merge/MERGE_Bimodal_Balanced')
        self.cfg       = task_args.get('cfg')

        self.mert_dir  = os.path.join(self.root, 'mert_30s')
        self.lyric_dir = os.path.join(self.root, 'lyrics')

        # --- Lyrics branch flags (safe defaults for backward compat) ---
        self.use_lyrics = getattr(self.cfg.model, 'use_lyrics', False) if self.cfg else False
        self.use_rhyme  = getattr(self.cfg.model, 'use_rhyme', False) if self.cfg else False
        self.xlmr_dir   = os.path.join(self.root, 'xlmr')
        self.rhyme_dir  = os.path.join(self.root, 'rhyme')

        # --- Split: artist-disjoint 70/15/15 by default (rerun-v2) ---
        # split_folder comes from config dataset.merge.split_folder;
        # 'tvt_70_15_15' reproduces the official (artist-overlapping) split.
        self.split_folder = task_args.get('split_folder', 'tvt_artist_70_15_15')
        split_name = "validate" if self.tr_val in ("val", "validation") else self.tr_val
        split_csv = os.path.join(
            self.root, 'tvt_dataframes', self.split_folder,
            f'{self.split_folder}_{split_name}_bimodal_balanced.csv'
        )
        split_df = pd.read_csv(split_csv)          # columns: Song, Quadrant

        # --- AV labels; join key: Song == Audio_Song ---
        av_csv = os.path.join(self.root, 'merge_bimodal_balanced_av_values.csv')
        av_df  = pd.read_csv(av_csv)               # Audio_Song, Lyric_Song, Arousal, Valence

        self.df = split_df.merge(
            av_df, left_on='Song', right_on='Audio_Song', how='inner'
        ).reset_index(drop=True)

        if len(self.df) != len(split_df):
            print(f"[MERGEDataset] WARNING: {len(split_df) - len(self.df)} "
                  f"split entries missing from AV csv")
        # --- Rhyme z-scoring with TRAIN-split statistics (rerun-v2) ---
        # model.rhyme_norm=false reproduces the unnormalized rungs 5-6 input.
        self.rhyme_mean = self.rhyme_std = None
        self.rhyme_norm = bool(self.cfg.model.get('rhyme_norm', True)) if self.cfg else True
        if self.use_rhyme and self.rhyme_norm:
            stats_path = task_args.get('rhyme_stats', 'splits/final/rhyme_stats.json')
            with open(stats_path) as f:
                st = json.load(f)
            self.rhyme_mean = np.asarray(st['mean'], dtype=np.float32)
            self.rhyme_std  = np.maximum(np.asarray(st['std'], dtype=np.float32), 1e-6)

        # --- Rhyme-slot control mode (rerun-v2) ---
        #   real     : the song's own rhyme vector (C4)
        #   zeros    : constant zero vector (C3: same 4-dim input, no information)
        #   permute  : another song's rhyme vector, fixed shuffle within split (K1)
        #   noise    : fixed per-song N(0,1) draw, i.e. z-space scale (K2)
        #   surface  : non-rhyme lyric features from surface/<L-ID>.npy (K3)
        #   keepNNN  : real vector with only the listed slots kept, others 0 (ablations)
        self.rhyme_mode   = str(self.cfg.model.get('rhyme_mode', 'real')) if self.cfg else 'real'
        self.control_seed = int(self.cfg.model.get('control_seed', 1234)) if self.cfg else 1234
        self.keep_idx = None
        if self.rhyme_mode.startswith('keep'):
            self.keep_idx = [int(c) for c in self.rhyme_mode[4:]]
            assert self.keep_idx and all(0 <= k < 4 for k in self.keep_idx), self.rhyme_mode
        elif self.rhyme_mode not in ('real', 'zeros', 'permute', 'noise', 'surface'):
            raise ValueError(f"unknown model.rhyme_mode: {self.rhyme_mode}")

        self.rhyme_src = list(self.df['Lyric_Song'].values)
        if self.use_rhyme and self.rhyme_mode == 'permute':
            off = {'train': 0, 'validation': 1, 'val': 1}.get(self.tr_val, 2)
            perm = np.random.default_rng(self.control_seed + off).permutation(len(self.df))
            self.rhyme_src = [self.rhyme_src[i] for i in perm]

        self.surface_dir = os.path.join(self.root, 'surface')
        if self.use_rhyme and self.rhyme_mode == 'surface':
            with open(task_args.get('surface_stats', 'splits/final/surface_stats.json')) as f:
                st = json.load(f)
            self.surface_mean = np.asarray(st['mean'], dtype=np.float32)
            self.surface_std  = np.maximum(np.asarray(st['std'], dtype=np.float32), 1e-6)

        print(f"[MERGEDataset] Split '{self.tr_val}' ({self.split_folder}): {len(self.df)} songs, "
              f"use_lyrics={self.use_lyrics}, use_rhyme={self.use_rhyme}, "
              f"rhyme_norm={self.rhyme_norm if self.use_rhyme else 'n/a'}, "
              f"rhyme_mode={self.rhyme_mode if self.use_rhyme else 'n/a'}")

        # Dummy chord/key tensors (same convention as globalmood_va.py)
        self.dummy_chord      = torch.zeros(100, dtype=torch.long)
        self.dummy_chord_root = torch.zeros(100, dtype=torch.long)
        self.dummy_chord_attr = torch.zeros(100, dtype=torch.long)
        self.dummy_key        = torch.zeros(1,   dtype=torch.long)

    def _rhyme_vector(self, index, lyric_id):
        mode = self.rhyme_mode
        if mode == 'zeros':
            return np.zeros(4, dtype=np.float32)
        if mode == 'noise':
            rng = np.random.default_rng(self.control_seed + zlib.crc32(str(lyric_id).encode()))
            return rng.standard_normal(4).astype(np.float32)
        if mode == 'surface':
            s = np.load(os.path.join(self.surface_dir, f'{lyric_id}.npy')).astype(np.float32).ravel()
            return (s - self.surface_mean) / self.surface_std
        src = self.rhyme_src[index] if mode == 'permute' else lyric_id
        r = np.load(os.path.join(self.rhyme_dir, f'{src}.npy')).astype(np.float32).ravel()
        if self.rhyme_mean is not None:
            r = (r - self.rhyme_mean) / self.rhyme_std
        if self.keep_idx is not None:
            mask = np.zeros(4, dtype=np.float32); mask[self.keep_idx] = 1.0
            r = r * mask
        return r


    @staticmethod
    def _rescale_av(x):
        # MERGE labels are empirically in [0,1] (verified 2026-07-06:
        # arousal 0.0625–0.95625, valence 0.075–0.95625).
        # Mapping: [0,1] -> [1,9] to match DEAM/PMEmo/EmoMusic regime.
        return 1.0 + 8.0 * x

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row      = self.df.iloc[index]
        audio_id = row['Audio_Song']
        lyric_id = row['Lyric_Song']
        quadrant = row['Quadrant']            # e.g. "Q4" — used for paths

        # --- MERT features (precomputed, same pattern as other loaders) ---
        fn_mert = os.path.join(self.mert_dir, audio_id)
        layers_to_extract = self.cfg.model.layers

        segment_embeddings = []
        if os.path.isdir(fn_mert):
            for filename in sorted(os.listdir(fn_mert)):
                if not filename.endswith('.npy'):
                    continue
                segment = np.load(os.path.join(fn_mert, filename))
                concatenated = np.concatenate(
                    [segment[:, layer_idx, :] for layer_idx in layers_to_extract],
                    axis=1
                )
                segment_embeddings.append(np.squeeze(concatenated))

        segment_embeddings = np.array(segment_embeddings)

        if self.tr_val == 'train' and len(segment_embeddings) > 1:
            start = np.random.randint(0, len(segment_embeddings))
            end   = np.random.randint(start + 1, len(segment_embeddings) + 1)
            final_embedding = np.mean(segment_embeddings[start:end], axis=0)
        elif len(segment_embeddings) > 0:
            final_embedding = np.mean(segment_embeddings, axis=0)
        else:
            final_embedding = np.zeros(1536)

        x_mert = torch.from_numpy(final_embedding.astype(np.float32))

        # --- Lyrics (raw text; kept for backward compat / debugging) ---
        lyric_path = os.path.join(self.lyric_dir, quadrant, lyric_id + '.txt')
        if os.path.exists(lyric_path):
            with open(lyric_path, 'r', encoding='utf-8', errors='replace') as f:
                lyric_text = f.read().strip()
        else:
            lyric_text = ""
            print(f"[MERGEDataset] WARNING: missing lyric file {lyric_path}")

        # --- Labels ---
        y_va = torch.tensor(
            [self._rescale_av(row['Valence']), self._rescale_av(row['Arousal'])],
            dtype=torch.float32
        )
        y_mood = torch.zeros(56, dtype=torch.float32)   # dummy, VA-only dataset

        audio_path = os.path.join(self.root, 'audio', quadrant, audio_id + '.mp3')

        # --- Build return dict ---
        sample = {
            "x_mert"       : x_mert,
            "x_chord"      : self.dummy_chord,
            "x_chord_root" : self.dummy_chord_root,
            "x_chord_attr" : self.dummy_chord_attr,
            "x_key"        : self.dummy_key,
            "y_va"         : y_va,
            "y_mood"       : y_mood,
            "lyrics"       : lyric_text,
            "path"         : audio_path
        }

        # --- Pre-extracted lyrics features (only when enabled) ---
        if self.use_lyrics:
            xlmr_path = os.path.join(self.xlmr_dir, f'{lyric_id}.npy')
            sample["x_xlmr"] = torch.from_numpy(np.load(xlmr_path)).float()

            if self.use_rhyme:
                sample["x_rhyme"] = torch.from_numpy(self._rhyme_vector(index, lyric_id))

        return sample

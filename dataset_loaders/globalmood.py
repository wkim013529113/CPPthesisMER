# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import torch
from torch.utils import data

class GlobalMoodDataset(data.Dataset):
    def __init__(self, **task_args):
        self.task_args = task_args
        self.tr_val    = task_args.get('tr_val', 'train')
        self.root      = task_args.get('root', './dataset/globalmood')
        self.cfg       = task_args.get('cfg')

        self.mert_dir  = os.path.join(self.root, 'mert_30s')
        self.labels_csv = os.path.join(self.root, 'globalmood_labels.csv')
        self.tag_file  = os.path.join(self.root, 'globalmood_tags.txt')

        # Load tag list
        with open(self.tag_file, 'r', encoding='utf-8') as f:
            self.tags = [line.strip() for line in f if line.strip()]
        self.num_tags = len(self.tags)  # 100

        # Load labels
        df = pd.read_csv(self.labels_csv)

        # Filter to songs that have MERT features extracted
        available_ids = {
            name for name in os.listdir(self.mert_dir)
            if os.path.isdir(os.path.join(self.mert_dir, name))
        }
        df = df[df['videoID'].isin(available_ids)].reset_index(drop=True)
        print(f"[GlobalMoodDataset] {len(df)} songs with MERT features | {self.num_tags} tags")

        # Frozen split from tools/make_splits.py (same rng-42 80/10/10 assignment,
        # but read from file so it cannot shift if the MERT folder changes).
        split_key = {'train': 'train', 'validation': 'val', 'val': 'val'}.get(self.tr_val, 'test')
        split_dir = self.cfg.get('split_dir', None) if self.cfg is not None else None
        split_file = os.path.join(split_dir, f'globalmood_{split_key}.txt') if split_dir else None
        if split_file and os.path.exists(split_file):
            with open(split_file) as f:
                ids = [l.strip() for l in f if l.strip()]
            pos = {v: i for i, v in enumerate(df['videoID'])}
            missing = [v for v in ids if v not in pos]
            if missing:
                raise ValueError(f"[GlobalMoodDataset] {len(missing)} split IDs lack MERT/labels, e.g. {missing[:3]}")
            self.df = df.iloc[[pos[v] for v in ids]].reset_index(drop=True)
            print(f"[GlobalMoodDataset] using frozen split {split_file}")
        else:
            # Legacy in-code split (80/10/10, rng 42)
            rng   = np.random.default_rng(42)
            idx   = rng.permutation(len(df))
            n     = len(df)
            n_tr  = int(n * 0.8)
            n_val = int(n * 0.1)

            if self.tr_val == 'train':
                idx = idx[:n_tr]
            elif self.tr_val == 'validation':
                idx = idx[n_tr:n_tr + n_val]
            else:  # test
                idx = idx[n_tr + n_val:]

            self.df = df.iloc[idx].reset_index(drop=True)
        print(f"[GlobalMoodDataset] Split '{self.tr_val}': {len(self.df)} songs")

        # Dummy chord/key tensors (GlobalMood has no chord/key data)
        # Matches the exact shapes expected by the model
        self.dummy_chord      = torch.zeros(100, dtype=torch.long)
        self.dummy_chord_root = torch.zeros(100, dtype=torch.long)
        self.dummy_chord_attr = torch.zeros(100, dtype=torch.long)
        self.dummy_key        = torch.zeros(1,   dtype=torch.long)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row    = self.df.iloc[index]
        vid_id = row['videoID']

        # --- MERT features (same logic as DEAMDataset) ---
        fn_mert = os.path.join(self.mert_dir, vid_id)
        layers_to_extract = self.cfg.model.layers  # e.g. [4, 5] for layers 5 & 6

        segment_embeddings = []
        for filename in sorted(os.listdir(fn_mert)):
            if not filename.endswith('.npy'):
                continue
            file_path = os.path.join(fn_mert, filename)
            segment   = np.load(file_path)
            concatenated = np.concatenate(
                [segment[:, layer_idx, :] for layer_idx in layers_to_extract], axis=1
            )
            concatenated = np.squeeze(concatenated)  # (1536,)
            segment_embeddings.append(concatenated)

        segment_embeddings = np.array(segment_embeddings)

        # Augmentation for training, mean for val/test
        if self.tr_val == 'train' and len(segment_embeddings) > 1:
            num_segments = len(segment_embeddings)
            start_idx = np.random.randint(0, num_segments)
            end_idx   = np.random.randint(start_idx + 1, num_segments + 1)
            chosen    = segment_embeddings[start_idx:end_idx]
            final_embedding = np.mean(chosen, axis=0)
        else:
            if len(segment_embeddings) > 0:
                final_embedding = np.mean(segment_embeddings, axis=0)
            else:
                final_embedding = np.zeros((1536,))

        x_mert = torch.from_numpy(final_embedding.astype(np.float32))

        # --- Mood labels (100-dim normalized [0,1]) ---
        y_mood = torch.tensor(
            row[self.tags].values.astype(np.float32),
            dtype=torch.float32
        )

        # --- Dummy valence/arousal (not used for GlobalMood head) ---
        y_va = torch.zeros(2, dtype=torch.float32)

        wav_path = os.path.join(self.root, 'wav', vid_id + '.wav')

        return {
            "x_mert"       : x_mert,
            "x_chord"      : self.dummy_chord,
            "x_chord_root" : self.dummy_chord_root,
            "x_chord_attr" : self.dummy_chord_attr,
            "x_key"        : self.dummy_key,
            "y_va"         : y_va,
            "y_mood"       : y_mood,
            "path"         : wav_path
        }


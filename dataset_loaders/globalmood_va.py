# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import torch
from torch.utils import data

class GlobalMoodVADataset(data.Dataset):
    """
    GlobalMood Option A — feeds VA regression head.
    Uses hand-crafted multilingual tag -> valence/arousal mapping.
    Compatible with DEAMDataset return format.
    """
    def __init__(self, **task_args):
        self.task_args = task_args
        self.tr_val    = task_args.get('tr_val', 'train')
        self.root      = task_args.get('root', './dataset/globalmood')
        self.cfg       = task_args.get('cfg')

        self.mert_dir  = os.path.join(self.root, 'mert_30s')
        self.va_csv    = os.path.join(self.root, 'globalmood_va_labels.csv')

        # Load VA labels
        df = pd.read_csv(self.va_csv)

        # Filter to songs with MERT features
        available_ids = {
            name for name in os.listdir(self.mert_dir)
            if os.path.isdir(os.path.join(self.mert_dir, name))
        }
        df = df[df['videoID'].isin(available_ids)].reset_index(drop=True)
        print(f"[GlobalMoodVADataset] {len(df)} songs with MERT features")

        # Reproducible 80/10/10 split
        rng   = np.random.default_rng(42)
        idx   = rng.permutation(len(df))
        n     = len(df)
        n_tr  = int(n * 0.8)
        n_val = int(n * 0.1)

        if self.tr_val == 'train':
            idx = idx[:n_tr]
        elif self.tr_val == 'validation':
            idx = idx[n_tr:n_tr + n_val]
        else:
            idx = idx[n_tr + n_val:]

        self.df = df.iloc[idx].reset_index(drop=True)
        print(f"[GlobalMoodVADataset] Split '{self.tr_val}': {len(self.df)} songs")

        # Dummy chord/key tensors
        self.dummy_chord      = torch.zeros(100, dtype=torch.long)
        self.dummy_chord_root = torch.zeros(100, dtype=torch.long)
        self.dummy_chord_attr = torch.zeros(100, dtype=torch.long)
        self.dummy_key        = torch.zeros(1,   dtype=torch.long)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row    = self.df.iloc[index]
        vid_id = row['videoID']

        # MERT features
        fn_mert = os.path.join(self.mert_dir, vid_id)
        layers_to_extract = self.cfg.model.layers

        segment_embeddings = []
        for filename in sorted(os.listdir(fn_mert)):
            if not filename.endswith('.npy'):
                continue
            segment = np.load(os.path.join(fn_mert, filename))
            concatenated = np.concatenate(
                [segment[:, layer_idx, :] for layer_idx in layers_to_extract], axis=1
            )
            segment_embeddings.append(np.squeeze(concatenated))

        segment_embeddings = np.array(segment_embeddings)

        if self.tr_val == 'train' and len(segment_embeddings) > 1:
            start = np.random.randint(0, len(segment_embeddings))
            end   = np.random.randint(start + 1, len(segment_embeddings) + 1)
            final_embedding = np.mean(segment_embeddings[start:end], axis=0)
        else:
            final_embedding = np.mean(segment_embeddings, axis=0) if len(segment_embeddings) > 0 else np.zeros(1536)

        x_mert = torch.from_numpy(final_embedding.astype(np.float32))

        # VA labels — same format as DEAM (2-dim float tensor)
        y_va = torch.tensor(
            [row['valence'], row['arousal']],
            dtype=torch.float32
        )

        # Dummy mood (not used for VA head)
        y_mood = torch.zeros(56, dtype=torch.float32)

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

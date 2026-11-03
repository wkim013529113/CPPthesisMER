import os
import numpy as np
import pickle
from torch.utils import data
import torchaudio.transforms as T
import torchaudio
import torch
import csv
import pytorch_lightning as pl
from music2latent import EncoderDecoder
import json
import math
from sklearn.preprocessing import StandardScaler

from dataset_loaders.jamendo import JamendoDataset
from dataset_loaders.pmemo import PMEmoDataset
from dataset_loaders.deam import DEAMDataset
from dataset_loaders.emomusic import EmoMusicDataset
from dataset_loaders.globalmood import GlobalMoodDataset
from dataset_loaders.globalmood_va import GlobalMoodVADataset
from dataset_loaders.merge import MERGEDataset

from omegaconf import DictConfig
import random


def _seed_worker(worker_id):
    # Give each DataLoader worker its own numpy/python RNG state derived from the
    # torch seed, so the segment-crop augmentation (np.random) is reproducible
    # and not duplicated across workers.
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)

DATASET_REGISTRY = {
    "jamendo": JamendoDataset,
    "pmemo": PMEmoDataset,
    "deam": DEAMDataset,
    "emomusic": EmoMusicDataset,
    "globalmood": GlobalMoodDataset,
    "globalmood_va": GlobalMoodVADataset,
    "merge": MERGEDataset
}

class DataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        self.train_datasets = []
        self.val_datasets = []
        self.test_datasets = []
        
    def setup(self, stage=None):
        # Clear previous dataset lists
        self.train_datasets = []
        self.val_datasets = []
        self.test_datasets = []

        # Register the datasets and load them
        for dataset_name in self.cfg.datasets:
            dataset_cfg = self.cfg.dataset[dataset_name]

            if dataset_name in DATASET_REGISTRY:
                train_dataset = DATASET_REGISTRY[dataset_name](**dataset_cfg, cfg=self.cfg, tr_val='train')
                val_dataset = DATASET_REGISTRY[dataset_name](**dataset_cfg, cfg=self.cfg, tr_val='validation')
                test_dataset = DATASET_REGISTRY[dataset_name](**dataset_cfg, cfg=self.cfg, tr_val='test')

                self.train_datasets.append(train_dataset)
                self.val_datasets.append(val_dataset)
                self.test_datasets.append(test_dataset)
            else:
                raise ValueError(f"Dataset {dataset_name} not found in registry")

    def train_dataloader(self):
        seed = int(self.cfg.get("seed", 0))
        return [data.DataLoader(ds, batch_size=self.cfg.dataset[ds_name].batch_size, 
                                shuffle=True, num_workers=self.cfg.dataset[ds_name].num_workers, 
                                persistent_workers=True,
                                worker_init_fn=_seed_worker,
                                generator=torch.Generator().manual_seed(seed + i))
                for i, (ds, ds_name) in enumerate(zip(self.train_datasets, self.cfg.datasets))]

    def val_dataloader(self):
        return [data.DataLoader(ds, batch_size=self.cfg.dataset[ds_name].batch_size, 
                                shuffle=False, num_workers=self.cfg.dataset[ds_name].num_workers, 
                                persistent_workers=True, worker_init_fn=_seed_worker)
                for ds, ds_name in zip(self.val_datasets, self.cfg.datasets)]

    def test_dataloader(self):
        return [data.DataLoader(ds, batch_size=self.cfg.dataset[ds_name].batch_size, 
                                shuffle=False, num_workers=self.cfg.dataset[ds_name].num_workers, 
                                persistent_workers=True, worker_init_fn=_seed_worker)
                for ds, ds_name in zip(self.test_datasets, self.cfg.datasets)]




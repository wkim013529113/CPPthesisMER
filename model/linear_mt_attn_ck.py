import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn import metrics
from transformers import AutoModelForAudioClassification
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len, :].to(x.device)

class FeedforwardModelMTAttnCK(nn.Module):
    def __init__(self, input_size, output_size_classification, output_size_regression,
                 output_size_globalmood=100,
                 nhead=8, num_layers=1, dropout_rate=0.1,
                 num_key=2, num_chords=158, num_chords_root=14, num_chords_attr=14,
                 key_emb_dim=4, chord_emb_dim=8, chord_root_emb_dim=4, chord_attr_emb_dim=4):
        super().__init__()
        self.d_model = 512
        self.d_model_transformer = chord_root_emb_dim + chord_attr_emb_dim

        self.chord_root_embedding = nn.Embedding(num_chords_root, chord_root_emb_dim)
        self.chord_attr_embedding = nn.Embedding(num_chords_attr, chord_attr_emb_dim)
        nn.init.xavier_uniform_(self.chord_root_embedding.weight)
        nn.init.xavier_uniform_(self.chord_attr_embedding.weight)

        self.positional_encoding = PositionalEncoding(self.d_model_transformer)

        self.chord_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.d_model_transformer, nhead=nhead,
                                       dim_feedforward=64, dropout=0.1, batch_first=True),
            num_layers=2
        )

        self.input_proj = nn.Sequential(
            nn.Linear(input_size + self.d_model_transformer + 1, self.d_model),
            nn.ReLU(),
        )

        # Head 1: Jamendo categorical (56 tags)
        self.classification_branch = nn.Sequential(
            nn.Linear(self.d_model, 256),
            nn.ReLU(),
            nn.Linear(256, output_size_classification)
        )

        # Head 2: Valence/Arousal regression (2-dim)
        self.regression_branch = nn.Sequential(
            nn.Linear(self.d_model, 256),
            nn.ReLU(),
            nn.Linear(256, output_size_regression)
        )

        # Head 3: GlobalMood multilingual tags (100-dim)
        self.globalmood_branch = nn.Sequential(
            nn.Linear(self.d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_size_globalmood),
            nn.Sigmoid()
        )

    def _compute_shared_repr(self, model_input_dic):
        """Compute the 512-dim shared representation before head routing.

        Factored out of forward() so lyrics_fusion can access the same
        representation without duplicating code. Both forward() and
        get_shared_repr() call this.
        """
        x_mert       = model_input_dic["x_mert"]
        x_chord_root = model_input_dic["x_chord_root"]
        x_chord_attr = model_input_dic["x_chord_attr"]
        x_key        = model_input_dic["x_key"]
        key_embedding = x_key.float()

        chord_root_embedding = self.chord_root_embedding(x_chord_root)
        chord_attr_embedding = self.chord_attr_embedding(x_chord_attr)
        chord_combined = torch.cat((chord_root_embedding, chord_attr_embedding), dim=-1)
        chord_combined = self.positional_encoding(chord_combined)

        cls_token = torch.zeros_like(chord_combined[:, :1, :])
        chord_with_cls = torch.cat([cls_token, chord_combined], dim=1)
        chord_transformed = self.chord_transformer(chord_with_cls)
        chord_cls = chord_transformed[:, 0, :]

        combined = torch.cat((x_mert, chord_cls, key_embedding), dim=1)
        combined = self.input_proj(combined)
        return combined

    def get_shared_repr(self, model_input_dic):
        """Return the 512-dim shared representation (for lyrics fusion).

        This is the output of input_proj -- the same tensor that feeds
        classification_branch, regression_branch, and globalmood_branch
        in forward(). Lyrics fusion concatenates its own features to this
        before passing through regression_branch.
        """
        return self._compute_shared_repr(model_input_dic)

    def forward(self, model_input_dic):
        combined = self._compute_shared_repr(model_input_dic)

        classification_output = self.classification_branch(combined)
        regression_output     = self.regression_branch(combined)
        globalmood_output     = self.globalmood_branch(combined)

        return classification_output, regression_output, globalmood_output

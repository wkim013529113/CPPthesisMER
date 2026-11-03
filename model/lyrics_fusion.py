"""
Lyrics fusion module for the Music2Emo lyrics branch.

This module lives alongside model/linear_mt_attn_ck.py and is instantiated
by trainer.py when model.use_lyrics=True. It is deliberately kept separate
from FeedforwardModelMTAttnCK so the base model code is unchanged.

Architecture (late fusion, per thesis section 3.3):
  1. The MERT shared representation (512-dim) is computed by the
     existing model as before.
  2. The pre-extracted XLM-R embedding (768-dim) is projected down to
     lyrics_proj_dim (default 256) via a linear layer + ReLU + dropout.
  3. Optionally, the 4-dim rhyme density vector is concatenated.
  4. The concatenated [shared_repr; xlmr_proj; (rhyme)] vector is
     projected back to 512-dim via a fusion layer + ReLU + dropout.
  5. The fused representation is passed to the existing regression_branch.

Dimension summary:
  Rung 5: [512 shared; 256 xlmr_proj]         = 768-dim -> fusion -> 512-dim
  Rung 6: [512 shared; 256 xlmr_proj; 4 rhyme] = 772-dim -> fusion -> 512-dim
"""

import torch
import torch.nn as nn


class LyricsFusionModule(nn.Module):
    def __init__(self, shared_dim=512, xlmr_dim=768, rhyme_dim=4,
                 lyrics_proj_dim=256, use_rhyme=False, dropout=0.1):
        """
        Args:
            shared_dim:      dimension of the model's shared representation
                             after input_proj (512 in FeedforwardModelMTAttnCK)
            xlmr_dim:        dimension of XLM-R embeddings (768 for base)
            rhyme_dim:       dimension of rhyme feature vector (4)
            lyrics_proj_dim: intermediate projection dimension for XLM-R
            use_rhyme:       whether to concatenate rhyme features
            dropout:         dropout rate for projection layers
        """
        super().__init__()
        self.use_rhyme = use_rhyme

        # Project XLM-R embedding to a lower dimension
        self.xlmr_proj = nn.Sequential(
            nn.Linear(xlmr_dim, lyrics_proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Fusion: map [shared_repr; xlmr_proj; (rhyme)] back to shared_dim
        fusion_input_dim = shared_dim + lyrics_proj_dim
        if use_rhyme:
            fusion_input_dim += rhyme_dim

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, shared_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, shared_repr, x_xlmr, x_rhyme=None):
        """
        Args:
            shared_repr: (batch, shared_dim) -- MERT backbone output
            x_xlmr:      (batch, 768)        -- pre-extracted XLM-R embedding
            x_rhyme:     (batch, 4) or None  -- pre-extracted rhyme features

        Returns:
            fused: (batch, shared_dim) -- ready for regression_branch (VA head)
        """
        xlmr_proj = self.xlmr_proj(x_xlmr)

        if self.use_rhyme and x_rhyme is not None:
            fused_input = torch.cat([shared_repr, xlmr_proj, x_rhyme], dim=-1)
        else:
            fused_input = torch.cat([shared_repr, xlmr_proj], dim=-1)

        return self.fusion(fused_input)

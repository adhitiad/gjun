# model.py
import math

import torch
import torch.nn as nn


class TimeSeriesTransformer(nn.Module):
    """Time Series Transformer Model"""

    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=3, output_dim=3):
        super().__init__()

        # Feature Embedding
        self.embedding = nn.Linear(input_dim, d_model)

        # Positional Encoding
        self.pos_encoder = nn.Parameter(torch.zeros(1, 500, d_model))

        # Transformer Encoder (Batch First = True is cleaner)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dropout=0.2
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Decoder Head
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Features)
        x = self.embedding(x)
        x = x + self.pos_encoder[:, : x.size(1), :]

        x = self.transformer(x)

        # Global Average Pooling (Contextual Summary)
        x = x.mean(dim=1)

        return self.decoder(x)

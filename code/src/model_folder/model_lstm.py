import torch
import torch.nn as nn

from .model_base.model_embed_base import EmbedBase


class LongShortTermMemory(EmbedBase):
    def __init__(self, settings):
        super().__init__(settings, settings["lstm"])

        self.n_layers = settings["lstm"]["n_layers"]
        self.output_dim = settings["lstm"]["output_dim"]

        self.lstm = nn.LSTM(
            self.input_dim, self.output_dim, self.n_layers, batch_first=True
        ).to(self.device)

        self.output_lin = nn.Linear(self.output_dim, 1).to(self.device)

    def forward(self, x):
        input_size = len(x["interaction"])

        input_x = super().forward(x)

        output_x, _ = self.lstm(input_x)

        output_x = output_x.contiguous().view(input_size, -1, self.output_dim)

        y_hat = self.output_lin(output_x).view(input_size, -1)

        return y_hat

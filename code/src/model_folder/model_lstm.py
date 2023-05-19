import torch
import torch.nn as nn

from .model_base.model_embed_base import EmbedLayer


class LongShortTermMemory(nn.Module):
    def __init__(self, settings):
        super().__init__()

        self.embedding_dim = settings["lstm"]["embedding_dim"]
        self.input_dim = settings["lstm"]["input_dim"]
        self.n_layers = settings["lstm"]["n_layers"]
        self.output_dim = settings["lstm"]["output_dim"]
        self.max_label_dict = settings["max_label_dict"]

        self.embed_layer = EmbedLayer(
            self.embedding_dim, self.input_dim, self.max_label_dict
        )

        self.lstm = nn.LSTM(
            self.input_dim, self.output_dim, self.n_layers, batch_first=True
        )

        self.output_lin = nn.Linear(self.output_dim, 1)

    def forward(self, x):
        input_size = len(x["interaction"])

        input_x = self.embed_layer(x)

        output_x, _ = self.lstm(input_x)

        output_x = output_x.contiguous().view(input_size, -1, self.output_dim)

        y_hat = self.output_lin(output_x).view(input_size, -1)

        return y_hat

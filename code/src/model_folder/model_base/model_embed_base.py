import torch
import torch.nn as nn


class EmbedBase(nn.Module):
    def __init__(self, settings, model_settings):
        super().__init__()

        self.device = settings["device"]

        self.embedding_dim = model_settings["embedding_dim"]
        self.input_dim = model_settings["input_dim"]
        self.max_label_dict = settings["max_label_dict"]

        # embedding layers
        self.embedding = dict()
        self.embedding["interaction"] = nn.Embedding(3, self.embedding_dim).to(
            self.device
        )
        for i, v in self.max_label_dict.items():
            self.embedding[i] = nn.Embedding(v + 1, self.embedding_dim).to(self.device)

        self.max_label_dict["interaction"] = 3

        self.input_lin = nn.Linear(
            len(self.embedding) * self.embedding_dim, self.input_dim
        ).to(self.device)

    def forward(self, x):
        embedded_x = torch.cat(
            [self.embedding[i](x[i].int()) for i in list(self.max_label_dict)], dim=2
        )

        input_x = self.input_lin(embedded_x)

        return input_x

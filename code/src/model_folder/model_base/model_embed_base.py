import torch
import torch.nn as nn


class EmbedLayer(nn.Module):
    def __init__(self, embedding_dim, input_dim, max_label_dict):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.input_dim = input_dim
        self.max_label_dict = max_label_dict

        # embedding layers
        self.embedding = nn.ModuleDict(
            {
                i: nn.Embedding(v + 1, self.embedding_dim)
                for i, v in self.max_label_dict.items()
            }
        )

        self.embedding["interaction"] = nn.Embedding(3, self.embedding_dim)

        self.max_label_dict["interaction"] = 3

        self.input_lin = nn.Linear(
            len(self.embedding) * self.embedding_dim, self.input_dim
        )

    def forward(self, x):
        embedded_x = torch.cat(
            [self.embedding[i](x[i].int()) for i in list(self.max_label_dict)], dim=2
        )

        input_x = self.input_lin(embedded_x)

        return input_x

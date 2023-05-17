import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertConfig, BertEncoder, BertModel

from .model_base.model_embed_base import EmbedBase


class BidirectionalEncoderRepresentationsfromTransformers(EmbedBase):
    def __init__(self, settings):
        super().__init__(settings, settings["bert"])

        self.n_layers = settings["bert"]["n_layers"]
        self.n_heads = settings["bert"]["n_heads"]

        self.config = BertConfig(
            3,  # not used
            hidden_size=self.input_dim,
            num_hidden_layers=self.n_layers,
            num_attention_heads=self.n_heads,
            max_position_embeddings=settings["bert"]["max_seq_len"],
        )

        self.encoder = BertModel(self.config)

        self.output_lin = nn.Linear(self.input_dim, 1)

    def forward(self, x):
        input_size = len(x["interaction"])

        input_x = super().forward(x)

        encoded_layers = self.encoder(inputs_embeds=input_x, attention_mask=x["mask"])
        out = encoded_layers[0]
        out = out.contiguous().view(input_size, -1, self.input_dim)
        out = self.output_lin(out).view(input_size, -1)

        return out

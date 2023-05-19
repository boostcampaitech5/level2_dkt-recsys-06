import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertConfig, BertEncoder, BertModel

from .model_base.model_embed_base import EmbedLayer


class BidirectionalEncoderRepresentationsfromTransformers(nn.Module):
    def __init__(self, settings):
        super().__init__()

        self.embedding_dim = settings["bert"]["embedding_dim"]
        self.input_dim = settings["bert"]["input_dim"]
        self.max_label_dict = settings["max_label_dict"]
        self.n_layers = settings["bert"]["n_layers"]
        self.n_heads = settings["bert"]["n_heads"]

        self.embed_layer = EmbedLayer(
            self.embedding_dim, self.input_dim, self.max_label_dict
        )

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

        input_x = self.embed_layer(x)

        encoded_layers = self.encoder(inputs_embeds=input_x, attention_mask=x["mask"])
        out = encoded_layers[0]
        out = out.contiguous().view(input_size, -1, self.input_dim)
        out = self.output_lin(out).view(input_size, -1)

        return out

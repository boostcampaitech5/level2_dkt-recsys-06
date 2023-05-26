import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertConfig, BertEncoder, BertModel

from .model_base.model_embed_base import EmbedLayer
from .model_base.model_mlp import MultiLayerPerceptron


class BidirectionalEncoderRepresentationsfromTransformers(nn.Module):
    """
    BERT model
    """

    def __init__(self, settings):
        """
        Initializes BERT Model

        Parameters:
            settings(dict): Dictionary containing the settings
        """

        super().__init__()

        # Get settings
        self.embedding_dim = settings["bert"]["embedding_dim"]
        self.input_dim = settings["bert"]["input_dim"]
        self.label_len_dict = settings["label_len_dict"]
        self.n_layers = settings["bert"]["n_layers"]
        self.n_heads = settings["bert"]["n_heads"]
        self.dense_layer_dim = settings["bert"]["dense_layer_dim"]
        self.non_embed_col = settings["non_embedding_columns"]

        # Create embedding layer
        self.embed_layer = EmbedLayer(self.embedding_dim, self.label_len_dict)

        # Create input linear layer
        embed_output_dim = self.embed_layer.get_output_dim()
        self.input_lin = nn.Linear(
            embed_output_dim + len(self.non_embed_col), self.input_dim
        )

        # Create BERT layer
        self.config = BertConfig(
            3,  # not used
            hidden_size=self.input_dim,
            num_hidden_layers=self.n_layers,
            num_attention_heads=self.n_heads,
            max_position_embeddings=settings["bert"]["max_seq_len"],
        )

        self.encoder = BertModel(self.config)

        # output dense layer
        self.output_lin = MultiLayerPerceptron(self.input_dim, self.dense_layer_dim)

        return

    def forward(self, x):
        # Get data input size
        input_size = len(x["interaction"])

        # Embedding layer
        embedded_x = self.embed_layer(x)

        # Combine non-embedding layer
        if len(self.non_embed_col) != 0:
            embedded_x = torch.cat(
                [embedded_x] + [x[i].unsqueeze(2) for i in self.non_embed_col], -1
            )

        # Input linear layer
        input_x = self.input_lin(embedded_x)

        # BERT layer
        encoded_layers = self.encoder(inputs_embeds=input_x, attention_mask=x["mask"])
        out = encoded_layers[0]

        # Dense layer
        out = out.contiguous().view(input_size, -1, self.input_dim)
        out = self.output_lin(out).view(input_size, -1)

        return out

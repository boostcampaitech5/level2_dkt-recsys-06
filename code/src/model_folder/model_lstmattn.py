import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertConfig, BertEncoder, BertModel

from .model_base.model_embed_base import EmbedLayer
from .model_base.model_mlp import MultiLayerPerceptron


class LongShortTermMemoryAttention(nn.Module):
    """
    LSTM Attention model
    """

    def __init__(self, settings):
        """
        Initializes LSTM Attention Model

        Parameters:
            settings(dict): Dictionary containing the settings
        """

        super().__init__()

        # Get settings
        self.embedding_dim = settings["lstm_attn"]["embedding_dim"]
        self.input_dim = settings["lstm_attn"]["input_dim"]
        self.label_len_dict = settings["label_len_dict"]
        self.n_layers = settings["lstm_attn"]["n_layers"]
        self.output_dim = settings["lstm_attn"]["output_dim"]
        self.dense_layer_dim = settings["lstm_attn"]["dense_layer_dim"]
        self.non_embed_col = settings["non_embedding_columns"]

        # Create embedding layer
        self.embed_layer = EmbedLayer(self.embedding_dim, self.label_len_dict)

        # Create input linear layer
        embed_output_dim = self.embed_layer.get_output_dim()
        self.input_lin = nn.Linear(
            embed_output_dim + len(self.non_embed_col), self.input_dim
        )

        # Create LSTM layer
        self.lstm = nn.LSTM(
            self.input_dim, self.output_dim, self.n_layers, batch_first=True
        )

        # Create Attention layer
        self.n_heads = settings["lstm_attn"]["n_heads"]
        self.drop_out = settings["lstm_attn"]["drop_out"]

        self.config = BertConfig(
            3,  # not used
            hidden_size=self.output_dim,
            num_hidden_layers=1,
            num_attention_heads=self.n_heads,
            intermediate_size=self.output_dim,
            hidden_dropout_prob=self.drop_out,
            attention_probs_dropout_prob=self.drop_out,
        )

        self.attn = BertEncoder(self.config)

        # Create dense layer
        self.output_lin = MultiLayerPerceptron(self.output_dim, self.dense_layer_dim)

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

        # LSTM layer
        output_x, _ = self.lstm(input_x)

        output_x = output_x.contiguous().view(input_size, -1, self.output_dim)

        extended_attention_mask = x["mask"].unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.n_layers

        # Attention layer
        encoded_layers = self.attn(
            output_x, extended_attention_mask, head_mask=head_mask
        )
        sequence_output = encoded_layers[-1]

        # Dense layer
        y_hat = self.output_lin(sequence_output).view(input_size, -1)

        return y_hat

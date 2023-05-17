import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertConfig, BertEncoder, BertModel

from .model_base.model_embed_base import EmbedBase


class LongShortTermMemoryAttention(EmbedBase):
    def __init__(self, settings):
        super().__init__(settings, settings["lstm_attn"])

        self.n_layers = settings["lstm_attn"]["n_layers"]
        self.output_dim = settings["lstm_attn"]["output_dim"]

        self.lstm = nn.LSTM(
            self.input_dim, self.output_dim, self.n_layers, batch_first=True
        )

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

        self.output_lin = nn.Linear(self.output_dim, 1)

    def forward(self, x):
        input_size = len(x["interaction"])

        input_x = super().forward(x)

        output_x, _ = self.lstm(input_x)

        output_x = output_x.contiguous().view(input_size, -1, self.output_dim)

        extended_attention_mask = x["mask"].unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.n_layers

        encoded_layers = self.attn(
            output_x, extended_attention_mask, head_mask=head_mask
        )
        sequence_output = encoded_layers[-1]

        y_hat = self.output_lin(sequence_output).view(input_size, -1)

        return y_hat

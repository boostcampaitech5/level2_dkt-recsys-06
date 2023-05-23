import torch
import torch.nn as nn

from .model_base.model_mlp import MultiLayerPerceptron
from .model_base.model_embed_base import EmbedLayer


class LongShortTermMemory(nn.Module):
    """
    LSTM model
    """

    def __init__(self, settings: dict) -> None:
        """
        Initializes LSTM Model

        Parameters:
            settings(dict): Dictionary containing the settings
        """

        super().__init__()

        # Get settings
        self.embedding_dim = settings["lstm"]["embedding_dim"]
        self.input_dim = settings["lstm"]["input_dim"]
        self.n_layers = settings["lstm"]["n_layers"]
        self.output_dim = settings["lstm"]["output_dim"]
        self.label_len_dict = settings["label_len_dict"]
        self.dense_layer_dim = settings["lstm"]["dense_layer_dim"]

        # Create embedding layer
        self.embed_layer = EmbedLayer(self.embedding_dim, self.label_len_dict)

        # Create input linear layer
        embed_output_dim = self.embed_layer.get_output_dim()
        self.input_lin = nn.Linear(embed_output_dim, self.input_dim)

        # Create LSTM layer
        self.lstm = nn.LSTM(
            self.input_dim, self.output_dim, self.n_layers, batch_first=True
        )

        # Create dense layer
        self.output_lin = MultiLayerPerceptron(self.output_dim, self.dense_layer_dim)

        return

    def forward(self, x):
        # Get data input size
        input_size = len(x["interaction"])

        # Embedding layer
        embedded_x = self.embed_layer(x)

        # Input linear layer
        input_x = self.input_lin(embedded_x)

        # LSTM layer
        output_x, _ = self.lstm(input_x)

        # Dense layer
        output_x = output_x.contiguous().view(input_size, -1, self.output_dim)
        y_hat = self.output_lin(output_x).view(input_size, -1)

        return y_hat

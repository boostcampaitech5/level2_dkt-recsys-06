import torch
import torch.nn as nn


class EmbedLayer(nn.Module):
    """
    Module used to create embedding's for each column
    """

    def __init__(
        self, embedding_dim: int, label_len_dict: dict, is_graph_embedding: bool = False
    ) -> None:
        """
        Initializes SaveSetting class

        Parameters:
            embedding_dim(int): Embedding layer output dimension
            label_len_dict(dict): Dictionary of label lengths
            is_graph_embedding : if True then Applying lgcn embedding
        """

        super().__init__()

        # Get settings
        self.embedding_dim = embedding_dim
        self.label_len_dict = label_len_dict

        # Embedding layers
        self.embedding = nn.ModuleDict(
            {
                i: nn.Embedding(v + 1, self.embedding_dim)
                for i, v in self.label_len_dict.items()
                if i != "question_id" or not is_graph_embedding
            }
        )

        if is_graph_embedding:
            self.embedding["question_id"] = nn.Linear(64, embedding_dim)

        # Add interaction layers
        self.embedding["interaction"] = nn.Embedding(3, self.embedding_dim)
        self.label_len_dict["interaction"] = 3

        return

    def forward(self, x):
        # Add Graph Embedding

        # Embed all embedding columns and concat them
        embedded_x = torch.cat(
            [self.embedding[i](x[i].int()) for i in list(self.label_len_dict)], dim=2
        )

        return embedded_x

    def get_output_dim(self):
        """
        Returns total output dim of embedding
        """

        return len(self.embedding) * self.embedding_dim

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLayerPerceptron(nn.Module):
    """
    Multilayer Perceptron (MLP) Class
    """

    def __init__(self, input_dim, layer_dim):
        super(MultiLayerPerceptron, self).__init__()

        self.layer_num = len(layer_dim)
        self.layer_dim = layer_dim

        if self.layer_num != 0:
            self.lin = nn.Sequential(nn.Linear(input_dim, self.layer_dim[0]), nn.ReLU())

            for i in range(self.layer_num - 1):
                self.lin.append(nn.Linear(self.layer_dim[i], self.layer_dim[i + 1]))
                self.lin.append(nn.ReLU())

            self.lin.append(nn.Linear(self.layer_dim[-1], 1))
        else:
            self.lin = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor):
        y_hat = self.lin(x)

        return y_hat

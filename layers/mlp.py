import unittest
from collections import OrderedDict
import torch.nn as nn


class MLP(nn.Module):
    """MLP module used in the main Koopman architecture.

    Attributes:
      input_dim: number of features
      output_dim: output dimension of encoder
      hidden_dim: hidden dimension of encoder
      num_layers: number of layers
      use_instancenorm: whether to use instance normalization
      dropout_rate: dropout rate
    """

    def __init__(
            self,
            input_dim,
            output_dim,
            hidden_dim,
            num_layers,
            use_instancenorm=False,
            dropout_rate=0
    ):
        super().__init__()

        model = [nn.Linear(input_dim, hidden_dim)]
        if use_instancenorm:
            model += [nn.InstanceNorm1d(hidden_dim)]
        model += [nn.ReLU(), nn.Dropout(dropout_rate)]

        for _ in range(num_layers - 2):
            model += [nn.Linear(hidden_dim, hidden_dim)]
            if use_instancenorm:
                model += [nn.InstanceNorm1d(hidden_dim)]
            model += [nn.ReLU(), nn.Dropout(dropout_rate)]
        model += [nn.Linear(hidden_dim, output_dim)]

        self.model = nn.Sequential(*model)

    def forward(self, inps):
        # expected input dims (batch size, sequence length, number of features)
        return self.model(inps)

from collections import OrderedDict
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self,
                 width_list,
                 prefix="mlp",
                 activation=nn.ReLU(),
                 dropout=0.01):

        """
        Create a multi-layer perceptron from the given dimension list.
        Adds dropout and activations expect the last layer.

        :param width_list: list of dimensions of the layers
        including input and output layer. Ex. [10, 32, 32, 2]
        will create a MLP with input dimension = 10 and 2 hidden layer
        with dimension 32 and output layer dimension 2.
        :param prefix: a string that will be prepended in the layer names.
        :param activation: non-linear activation after each layer of the mlp
        :param dropout: dropout value
        """

        super(MLP, self).__init__()
        layer_dict = OrderedDict()
        drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        for i in range(1, len(width_list)):
            layer_dict[prefix + "_linear_" + str(i)] = nn.Linear(width_list[i - 1], width_list[i])
            if i != len(width_list) - 2:
                layer_dict[prefix + "_act_" + str(i)] = activation
            if i <= len(width_list) - 2:
                layer_dict[prefix + "_drop_" + str(i)] = drop
        self.mlp = nn.Sequential(layer_dict)

    def forward(self, x):
        return self.mlp(x)

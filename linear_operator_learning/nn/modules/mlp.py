# TODO: Refactor the models, add docstrings, etc...
"""PyTorch Models."""

import torch
from torch.nn import Conv2d, Dropout, Linear, MaxPool2d, Module, ReLU, Sequential


class _MLPBlock(Module):
    def __init__(self, input_size, output_size, dropout=0.0, activation=ReLU, bias=True):
        super(_MLPBlock, self).__init__()
        self.linear = Linear(input_size, output_size, bias=bias)
        self.dropout = Dropout(dropout)
        self.activation = activation()

    def forward(self, x):
        out = self.linear(x)
        out = self.dropout(out)
        out = self.activation(out)
        return out


class MLP(Module):
    """Multi Layer Perceptron.

    Args:
        input_shape (int): Input shape of the MLP.
        n_hidden (int): Number of hidden layers.
        layer_size (int or list of ints): Number of neurons in each layer. If an int is
            provided, it is used as the number of neurons for all hidden layers. Otherwise,
            the list of int is used to define the number of neurons for each layer.
        output_shape (int): Output shape of the MLP.
        dropout (float): Dropout probability between layers. Defaults to 0.0.
        activation (torch.nn.Module): Activation function. Defaults to ReLU.
        iterative_whitening (bool): Whether to add an IterNorm layer at the end of the
            network. Defaults to False.
        bias (bool): Whether to include bias in the layers. Defaults to False.
    """

    def __init__(
        self,
        input_shape,
        n_hidden,
        layer_size,
        output_shape,
        dropout=0.0,
        activation=ReLU,
        iterative_whitening=False,
        bias=False,
    ):
        super(MLP, self).__init__()
        if isinstance(layer_size, int):
            layer_size = [layer_size] * n_hidden
        if n_hidden == 0:
            layers = [Linear(input_shape, output_shape, bias=False)]
        else:
            layers = []
            for layer in range(n_hidden):
                if layer == 0:
                    layers.append(
                        _MLPBlock(input_shape, layer_size[layer], dropout, activation, bias=bias)
                    )
                else:
                    layers.append(
                        _MLPBlock(
                            layer_size[layer - 1], layer_size[layer], dropout, activation, bias=bias
                        )
                    )

            layers.append(Linear(layer_size[-1], output_shape, bias=False))
            if iterative_whitening:
                # layers.append(IterNorm(output_shape))
                raise NotImplementedError("IterNorm isn't implemented")
        self.model = Sequential(*layers)

    def forward(self, x):  # noqa: D102
        return self.model(x)

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
    """Multilayer Perceptron."""

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

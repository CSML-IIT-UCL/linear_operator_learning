"""Exponential moving average of the covariance matrices."""

import torch
import torch.distributed
from torch import Tensor


class EMACovariance(torch.nn.Module):
    r"""Exponential moving average of the covariance matrices.

    Gives an online estimate of the covariances and means :math:`C` adding the batch covariance :math:`\hat{C}` via the following update forumla

    .. math::

        C \leftarrow (1 - m)C + m \hat{C}

    Args:
        feature_dim: The number of features in the input and output tensors.
        momentum: The momentum for the exponential moving average.
        center: Whether to center the data before computing the covariance matrices.
    """

    def __init__(self, feature_dim: int, momentum: float = 0.01, center: bool = True):
        super().__init__()
        self.is_centered = center
        self.momentum = momentum
        self.register_buffer("mean_X", torch.zeros(feature_dim))
        self.register_buffer("cov_X", torch.eye(feature_dim))
        self.register_buffer("mean_Y", torch.zeros(feature_dim))
        self.register_buffer("cov_Y", torch.eye(feature_dim))
        self.register_buffer("cov_XY", torch.eye(feature_dim))
        self.register_buffer("is_initialized", torch.tensor(False, dtype=torch.bool))

    @torch.no_grad()
    def forward(self, X: Tensor, Y: Tensor):
        """Update the exponential moving average of the covariance matrices.

        Args:
            X: Input tensor.
            Y: Output tensor.

        Shape:
            ``x``: :math:`(N, D)`, where :math:`N` is the batch size and :math:`D` is the number of features.

            ``y``: :math:`(N, D)`, where :math:`N` is the batch size and :math:`D` is the number of features.
        """
        if not self.training:
            return
        assert X.ndim == 2
        assert X.shape == Y.shape
        assert X.shape[1] == self.mean_X.shape[0]
        if not self.is_initialized.item():
            self._first_forward(X, Y)
        else:
            mean_X = X.mean(dim=0, keepdim=True)
            mean_Y = Y.mean(dim=0, keepdim=True)
            # Update means
            self._inplace_EMA(mean_X[0], self.mean_X)
            self._inplace_EMA(mean_Y[0], self.mean_Y)

            if self.is_centered:
                X = X - self.mean_X
                Y = Y - self.mean_Y

            cov_X = torch.mm(X.T, X) / X.shape[0]
            cov_Y = torch.mm(Y.T, Y) / Y.shape[0]
            cov_XY = torch.mm(X.T, Y) / X.shape[0]
            # Update covariances
            self._inplace_EMA(cov_X, self.cov_X)
            self._inplace_EMA(cov_Y, self.cov_Y)
            self._inplace_EMA(cov_XY, self.cov_XY)

    def _first_forward(self, X: torch.Tensor, Y: torch.Tensor):
        mean_X = X.mean(dim=0, keepdim=True)
        self._inplace_set(mean_X[0], self.mean_X)
        mean_Y = Y.mean(dim=0, keepdim=True)
        self._inplace_set(mean_Y[0], self.mean_Y)
        if self.is_centered:
            X = X - self.mean_X
            Y = Y - self.mean_Y

        cov_X = torch.mm(X.T, X) / X.shape[0]
        cov_Y = torch.mm(Y.T, Y) / Y.shape[0]
        cov_XY = torch.mm(X.T, Y) / X.shape[0]
        self._inplace_set(cov_X, self.cov_X)
        self._inplace_set(cov_Y, self.cov_Y)
        self._inplace_set(cov_XY, self.cov_XY)
        self.is_initialized = torch.tensor(True, dtype=torch.bool)

    def _inplace_set(self, update, current):
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(update, op=torch.distributed.ReduceOp.SUM)
            update /= torch.distributed.get_world_size()
        current.copy_(update)

    def _inplace_EMA(self, update, current):
        alpha = 1 - self.momentum
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(update, op=torch.distributed.ReduceOp.SUM)
            update /= torch.distributed.get_world_size()

        current.mul_(alpha).add_(update, alpha=self.momentum)


def test_EMACovariance():  # noqa: D103
    torch.manual_seed(0)

    dims = 5
    dummy_X = torch.randn(10, dims)
    dummy_Y = torch.randn(10, dims)
    cov_module = EMACovariance(feature_dim=dims)

    # Check that when model is not set to training covariance is not updated
    cov_module.eval()
    cov_module(dummy_X, dummy_Y)
    assert torch.allclose(cov_module.cov_X, torch.eye(dims))
    assert torch.allclose(cov_module.cov_Y, torch.eye(dims))
    assert torch.allclose(cov_module.cov_XY, torch.eye(dims))

    assert torch.allclose(cov_module.mean_X, torch.zeros(dims))
    assert torch.allclose(cov_module.mean_Y, torch.zeros(dims))

    # Check that the first_forward is correctly called
    cov_module.train()
    assert not cov_module.is_initialized.item()
    cov_module(dummy_X, dummy_Y)
    assert cov_module.is_initialized.item()
    assert torch.allclose(cov_module.mean_X, dummy_X.mean(dim=0))
    assert torch.allclose(cov_module.mean_Y, dummy_Y.mean(dim=0))
    if cov_module.is_centered:
        assert torch.allclose(cov_module.cov_X, torch.cov(dummy_X.T, correction=0))
        assert torch.allclose(cov_module.cov_Y, torch.cov(dummy_Y.T, correction=0))

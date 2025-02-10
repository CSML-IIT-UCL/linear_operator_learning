"""Neural Conditional Probability in PyTorch."""

import math

import torch
from torch import Tensor
from torch.nn import Module

from linear_operator_learning.nn.linalg import whitening

__all__ = ["NCP"]


class _Matrix_Layer(Module):
    """truncated operator's matrix form. See the NCP class for arguments."""

    def __init__(self, dim_u: int, dim_v: int, matrix_form: str, learnable_matrix: bool):
        super().__init__()
        if matrix_form != "dense" or not learnable_matrix:
            raise NotImplementedError(
                "This NCP implementation only supports dense learnable matrix layer."
            )
        self.weights = torch.nn.Parameter(
            torch.normal(mean=0.0, std=2.0 / math.sqrt(dim_u * dim_v), size=(dim_u, dim_v))
        )

    # @property
    # def matrix(self) -> Tensor:
    #     """Get matrix as a new tensor."""
    #     return self.weights

    def forward(self, u: Tensor, v: Tensor) -> Tensor:
        """Forward pass of the matrix layer."""
        # Sum A_ij u_i v_j, where k is the index within the batch.
        out = torch.einsum("ij,ki,kl->k", self.weights, u, v)
        return out


class NCP(Module):
    """Neural Conditional Probability Architecture.

    Args:
        embedding_x (Module): Neural embedding of x.
        embedding_dim_x (int): Latent dimension of x. I.e., dim of embedding_x.
        embedding_y (Module): Neural embedding of y.
        embedding_dim_y (int): Latent dimension of y. I.e., dim of embedding_y.
        matrix_form (str, optional): Either 'dense' of 'sparse'. Defaults to 'dense'.
        learnable_matrix (bool, optional): Whether the matrix layer is learnable. Defaults to True.
        tract_running_stats (bool, optional): Whether to track running latent variable statistics.
            Defaults to True.
        momentum_running_stats (float, optional): Momentum for exponential moving average of the
            running latent variable statistics. Defaults to 0.9.
        whitening (bool, optional): Whether to perform whitening after training. Defaults to True.
    """

    def __init__(
        self,
        embedding_x: Module,
        embedding_dim_x: int,
        embedding_y: Module,
        embedding_dim_y: int,
        matrix_form: str = "dense",
        learnable_matrix: bool = True,
        track_running_stats: bool = True,
        momentum_running_stats: float = 0.9,
        whitening: bool = True,
    ) -> None:
        super().__init__()

        self.U = embedding_x
        self.dim_u = embedding_dim_x
        self.V = embedding_y
        self.dim_v = embedding_dim_y
        self.S = _Matrix_Layer(
            dim_u=self.dim_u,
            dim_v=self.dim_v,
            matrix_form=matrix_form,
            learnable_matrix=learnable_matrix,
        )

        # Register buffers for the statistics of the latent variables u and v.
        self.track_running_stats = track_running_stats
        self.momentum_running_stats = momentum_running_stats
        if track_running_stats:
            self._register_running_stats_buffers()

        # Register buffers for whitening.
        self.whitening = whitening
        if whitening:
            self._register_whitening_buffers()

    @property
    def truncated_op_matrix(self) -> Tensor:
        """Truncated operator's matrix."""
        return self.S.matrix

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """NCP's forward pass.

        Args:
            x (Tensor): Input features for x.
            y (Tensor): Input features for y.

        Shape:
            # TODO: I just saw Dani's implementation and I really like the (..., d) style of NN coding.
        """
        u = self.U(x)
        v = self.V(y)

        if self.training and self.track_running_stats:
            self._update_running_stats_buffers(u, v)

        return u, v

    def _register_running_stats_buffers(
        self,
    ) -> None:
        """TODO: Add docs."""
        if self.dim_u != self.dim_v:
            raise NotImplementedError("dim_u != dim_v is not currently implemented.")
        # Store running means of covariance matrices:
        self.register_buffer("running_cov_u", torch.eye(self.dim_u))
        self.register_buffer("running_cov_v", torch.eye(self.dim_v))
        self.register_buffer("running_cov_uv", torch.eye(self.dim_u))

        # Store running means of u and v:
        self.register_buffer("running_mean_u", torch.zeros(self.dim_u))
        self.register_buffer("running_mean_v", torch.zeros(self.dim_v))

    def _register_whitening_buffers(
        self,
    ) -> None:
        """TODO: Add docs."""
        if self.dim_u != self.dim_v:
            raise NotImplementedError("dim_u != dim_v is not currently implemented.")
        # Store square root of inverse of covariances of u and v after whitening:
        self.register_buffer("sqrt_cov_u_inv", torch.eye(self.dim_u))
        self.register_buffer("sqrt_cov_v_inv", torch.eye(self.dim_v))

        # Store singular values and vectors after whitening:
        self.register_buffer("sing_val", torch.ones(self.dim_u))
        self.register_buffer("sing_vec_l", torch.eye(self.dim_u))
        self.register_buffer("sing_vec_r", torch.eye(self.dim_v))

    def _update_running_stats_buffers(
        self, u: Tensor, v: Tensor
    ) -> None:  # TODO: Could be a decorator
        """TODO: Add docs."""
        # TODO: Would exponential moving average be better here? In the class docstring I say we use
        #       it, but I am not familiar with this.
        momentum = self.momentum_running_stats
        self.mean_u = momentum * u.mean(dim=0, keepdim=True) + (1 - momentum) * self.running_mean_u
        self.mean_v = momentum * v.mean(dim=0, keepdim=True) + (1 - momentum) * self.running_mean_v

        # Centering before (centered/un-centered) covariance estimation is key for numerical stability.
        n_samples = u.shape[0]
        eps = 1e-6
        eps_u = eps * torch.eye(self.dim_u, device=u.device, dtype=u.dtype)
        eps_v = eps * torch.eye(self.dim_v, device=v.device, dtype=v.dtype)
        u_c, v_c = u - self.running_mean_u, v - self.running_mean_v
        self.running_cov_uv = (
            momentum * torch.einsum("nr,nc->rc", u_c, v_c) / (n_samples - 1)
            + (1 - momentum) * self.running_cov_uv
        )
        self.running_cov_u = (
            momentum * torch.einsum("nr,nc->rc", u_c, u_c) / (n_samples - 1)
            + eps_u
            + (1 - momentum) * self.running_cov_u
        )
        self.running_cov_v = (
            momentum * torch.einsum("nr,nc->rc", v_c, v_c) / (n_samples - 1)
            + eps_v
            + (1 - momentum) * self.running_cov_v
        )

    def _update_whitening_buffers(self, x: Tensor, y: Tensor) -> None:
        """TODO: Add docs."""
        u = self.U(x)
        v = self.V(y)

        (
            self.sqrt_cov_u_inv,
            self.sqrt_cov_v_inv,
            self.sing_val,
            self.sing_vec_l,
            self.sing_vec_r,
        ) = whitening(u, v)

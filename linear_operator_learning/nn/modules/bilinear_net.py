"""Bilinear Neural Network."""

import math

import torch
from torch import Tensor
from torch.nn import Module

__all__ = ["BiLinearNet"]


class _Matrix(
    Module
):  # TODO: I propose this module be private, as it will contain tricks (estimated S via cov(u,v)) which are only used for the conditional expectation operator.
    """Auxiliary Module for computing linear (v -> Sv) and bilinear (u,v -> uSv) operations with configurable weight matrix S."""

    def __init__(
        self,
        dim_u: int,
        dim_v: int,
        learnable: bool = True,
        form: str = "dense",
        symmetric: bool = True,
        rescale: str = "linear-eig",
        rescale_eig_lt: float = 1.0,
        # eps (float, optional): Small constant added to matrix diagonals for numerical stability. Defaults to 1e-4.
        # eps: float = 1e-4,
    ) -> None:
        super().__init__()
        if learnable and form == "dense":
            self._weights = torch.nn.Parameter(
                torch.normal(mean=0.0, std=2.0 / math.sqrt(dim_u * dim_v), size=(dim_u, dim_v))
            )
        else:
            raise NotImplementedError("This implementation only supports dense, learnable matrix")

        self.symmetric = symmetric
        self.rescale = rescale
        self.rescale_eig_lt = rescale_eig_lt

    @property
    def matrix(self) -> Tensor:
        """TODO: Add docs."""
        S = self._weights

        if self.rescale == "linear":
            max_eigval = torch.linalg.eigvalsh(S)[-1]
            S *= self.rescale_eig_lt / max_eigval
        elif self.rescale == "exp":
            S = torch.exp(-(S**2))

        if self.symmetric:
            S = 0.5 * (S + S.T)

        return S

    def forward(self, v: Tensor) -> Tensor:
        """TODO: Add docs."""
        out = v @ self.matrix.T
        return out

    def bilinear(self, u: Tensor, v: Tensor) -> Tensor:
        """TODO: Add docs."""
        # Sum A_ij u_i v_j, where k is the index within the batch.
        out = torch.einsum("ij,ki,kl->k", self.matrix, u, v)
        return out


class BiLinearNet(Module):
    """Bilinear Neural Network Architecture.

    # TODO: Add detailed description of what is meant by bilinear network.

    Args:
        embedding_x (Module): Neural embedding of x.
        embedding_dim_x (int): Latent dimension of x. I.e., dim of embedding_x.
        embedding_y (Module): Neural embedding of y.
        embedding_dim_y (int): Latent dimension of y. I.e., dim of embedding_y.
        forward_mode (str, optional): Either 'u,v' or 'uSv' or 'u,Sv' or 'u,v,Sv'. Defaults to 'u,Sv'.
        matrix_learnable (bool, optional): Whether the matrix layer is learnable. If not, then S
            is taken to be the covariance between u and v and is estimated from batches. Defaults to True.
        matrix_form (str, optional): Either 'diagonal' or 'dense'. Defaults to 'dense'.
        matrix_symmetric (bool, optional): Whether the matrix is symmetric.
        matrix_rescale (str, optional): Either 'linear-eig' or 'exp'. Defaults to 'linear-eig'.
        matrix_rescale_eig_lt (float, optional): Constrain scale of the maximum eigenvalue. Defaults to 1.0.
    """

    def __init__(
        self,
        embedding_x: Module,
        embedding_dim_x: int,
        embedding_y: Module,
        embedding_dim_y: int,
        # TODO: What do you guys think about forward mode?
        # Naming of the categories is very important. Check the class docstring for my proposal.
        forward_mode: str = "u,Sv",
        matrix_form: str = "dense",
        matrix_learnable: bool = True,
        matrix_symmetric: bool = True,
        matrix_rescale: str = "linear-eig",
        matrix_rescale_eig_lt: float = 1.0,
    ) -> None:
        if forward_mode not in ["u,v", "uSv", "u,Sv", "u,v,Sv"]:
            raise RuntimeError(f"Forward mode {forward_mode} not supported.")
        super.__init__()
        self.U = embedding_x
        self.dim_u = embedding_dim_x
        self.V = embedding_y
        self.dim_v = embedding_dim_y
        self.forward_mode = forward_mode
        self.S = _Matrix(
            dim_u=self.dim_u,
            dim_v=self.dim_v,
            form=matrix_form,
            learnable=matrix_learnable,
            symmetric=matrix_symmetric,
            rescale=matrix_rescale,
            rescale_eig_lt=matrix_rescale_eig_lt,
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """TODO: Add docs."""
        # TODO: Do we need to center?
        u = self.U(x)
        v = self.V(y)

        if self.forward_mode == "u,v":
            return u, v

        if self.forward_mode == "uSv":
            return self.S.bilinear(u, v)

        Sv = self.S(v)

        if self.forward_mode == "u,Sv":
            return u, Sv
        else:
            return u, v, Sv

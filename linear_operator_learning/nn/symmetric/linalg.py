"""Linear algebra utilities for symmetric vector spaces with known group representations."""

import numpy as np

# Created by Daniel Ordoñez (daniels.ordonez@gmail.com) at 13/02/25
import torch
from escnn.group import Representation, change_basis
from jinja2.lexer import TOKEN_DOT
from symm_torch.utils.rep_theory import isotypic_decomp_rep
from torch import Tensor

from linear_operator_learning.nn.symmetric.stats import isotypic_covariance


def isotypic_signal2irreducible_subspaces(x: Tensor, rep_x: Representation):
    r"""Given a random variable in an isotypic subspace, flatten the r.v. into G-irreducible subspaces.

    Given a signal of shape :math:`(n, m_x \cdot d)` where :math:`n` is the number of samples, :math:`m_x` the multiplicity of the irrep in :math:`X`, and :math:`d` the dimension of the irrep.
    :math:`X = [x_1, \ldots, x_n]` and :math:`x_i = [x_{i_{11}}, \ldots, x_{i_{1d}}, x_{i_{21}}, \ldots, x_{i_{2d}}, \ldots, x_{i_{m_x1}}, \ldots, x_{i_{m_xd}}]`

    This function returns the signal :math:`Z` of shape :math:`(n \cdot d, m_x)` where each column represents the flattened signal of a G-irreducible subspace.
    :math:`Z[:, k] = [x_{1_{k1}}, \ldots, x_{1_{kd}}, x_{2_{k1}}, \ldots, x_{2_{kd}}, \ldots, x_{n_{k1}}, \ldots, x_{n_{kd}}]`

    Args:
        x (Tensor): Shape :math:`(..., n, m_x \cdot d)` where :math:`n` is the number of samples and :math:`m_x` the multiplicity of the irrep in :math:`X`.
        rep_x (escnn.nn.Representation): Representation in the isotypic basis of a single type of irrep.

    Returns:
        Tensor: Shape :math:`(n \cdot d, m_x)` where each column represents the flattened signal of an irreducible subspace.
    """
    assert len(rep_x._irreps_multiplicities) == 1, (
        "Random variable is assumed to be in a single isotypic subspace."
    )
    irrep_id = rep_x.irreps[0]
    irrep_dim = rep_x.group.irrep(*irrep_id).size
    mk = rep_x._irreps_multiplicities[irrep_id]  # Multiplicity of the irrep in X

    Z = x.view(-1, mk, irrep_dim).permute(0, 2, 1).reshape(-1, mk)

    assert Z.shape == (x.shape[0] * irrep_dim, mk)
    return Z


def lstsq(x: Tensor, y: Tensor, rep_x: Representation, rep_y: Representation):
    r"""Computes a symmetry-aware solution to the least squares problem of a system of linear equations.

    The least squares problem for a linear system of equations :math:`Y = AX` with :math:`Y \in \mathbb{R}^{N\times n_y}`
    and :math:`X \in \mathbb{R}^{N\times n_x}`
    TODO: Finish docs
    """
    rep_x = isotypic_decomp_rep(rep_x)
    rep_y = isotypic_decomp_rep(rep_y)
    X_iso_reps = rep_x.attributes["isotypic_reps"]
    Y_iso_reps = rep_y.attributes["isotypic_reps"]
    Qx2iso = torch.tensor(rep_x.change_of_basis_inv, dtype=x.dtype, device=x.device)
    Qy2iso = torch.tensor(rep_y.change_of_basis_inv, dtype=y.dtype, device=y.device)
    Qiso2y = torch.tensor(rep_y.change_of_basis, dtype=y.dtype, device=y.device)

    x_iso = torch.einsum("ij,ni->nj", Qx2iso, x)
    y_iso = torch.einsum("ij,ni->nj", Qy2iso, y)

    # Get orthogonal projection to isotypic subspaces.
    dimx, dimy = 0, 0
    X_iso_dims, Y_iso_dims = {}, {}
    for irrep_k_id, rep_X_k in X_iso_reps.items():
        X_iso_dims[irrep_k_id] = slice(dimx, dimx + rep_X_k.size)
        dimx += rep_X_k.size
    for irrep_k_id, rep_Y_k in Y_iso_reps.items():
        Y_iso_dims[irrep_k_id] = slice(dimy, dimy + rep_Y_k.size)
        dimy += rep_Y_k.size

    A = torch.zeros((rep_y.size, rep_x.size), device=x.device, dtype=x.dtype)
    for irrep_k_id in Y_iso_reps.keys():
        if irrep_k_id not in X_iso_reps:
            continue
        d_k = rep_x.group.irrep(*irrep_k_id).size
        I_d_k = torch.eye(d_k, dtype=x.dtype, device=x.device)
        rep_X_k, rep_Y_k = X_iso_reps[irrep_k_id], Y_iso_reps[irrep_k_id]
        x_k, y_k = x_iso[..., X_iso_dims[irrep_k_id]], y_iso[..., Y_iso_dims[irrep_k_id]]

        # A_k = (Zyx_k @ Zx_k^†) ⊗ I_d_k
        # Cx_k, Zx_k = isotypic_covariance(x=x_k, y=x_k, rep_X=rep_X_k, rep_Y=rep_X_k)
        # Cyx_k, Zyx_k = isotypic_covariance(x=x_k, y=y_k, rep_X=rep_X_k, rep_Y=rep_Y_k)
        # A_k = torch.kron(Zyx_k @ torch.linalg.pinv(Zx_k), torch.eye(d_k, dtype=x.dtype, device=x.device))
        x_sing = isotypic_signal2irreducible_subspaces(x_k, rep_X_k)
        y_sing = isotypic_signal2irreducible_subspaces(y_k, rep_Y_k)
        # (Zyx_k @ Zx_k^†)
        out = torch.linalg.lstsq(x_sing, y_sing)
        A_k = torch.kron(out.solution.T, I_d_k)

        A[Y_iso_dims[irrep_k_id], X_iso_dims[irrep_k_id]] = A_k

    # Change back to the original input output basis sets
    A = Qiso2y @ A @ Qx2iso

    return A


def test_lstsq():  # noqa: D103
    import escnn
    from escnn.group import directsum

    # Icosahedral group has irreps of dimensions [1, ... 5]. Good test case.
    # G = escnn.group.Icosahedral()
    G = escnn.group.Icosahedral()
    mx, my = 2, 2
    rep_x = directsum([G.regular_representation] * mx)
    rep_y = directsum([G.regular_representation] * my)

    # Test isotypic basis
    rep_x = isotypic_decomp_rep(rep_x)
    rep_x = change_basis(rep_x, rep_x.change_of_basis_inv, f"{rep_x.name}-iso")
    rep_y = isotypic_decomp_rep(rep_y)
    rep_y = change_basis(rep_y, rep_y.change_of_basis_inv, f"{rep_y.name}-iso")

    batch_size = 256
    X = torch.randn(batch_size, rep_x.size)

    # Random G-equivariant linear map A: X -> Y
    A_gt = np.random.rand(rep_y.size, rep_x.size)
    G_gt = [np.einsum("ij,jk,kl->il", rep_y(~g), A_gt, rep_x(g)) for g in G.elements]
    A_gt = torch.tensor(np.mean(G_gt, axis=0), dtype=X.dtype)
    for g in G.elements:
        assert np.allclose(rep_y(g) @ A_gt.numpy(), A_gt.numpy() @ rep_x(g), atol=1e-5, rtol=1e-5)

    Y = torch.einsum("ij,nj->ni", A_gt, X)
    A = lstsq(X, Y, rep_x, rep_y)

    assert np.allclose(A_gt.numpy(), A.numpy(), atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    test_lstsq()

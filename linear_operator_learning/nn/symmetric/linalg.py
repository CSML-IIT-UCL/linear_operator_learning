"""Linear algebra utilities for symmetric vector spaces with known group representations."""

# Created by Daniel Ordoñez (daniels.ordonez@gmail.com) at 13/02/25
import torch
from escnn.group import Representation
from torch import Tensor


def invariant_orthogonal_projector(rep_X: Representation):
    r"""Computes the orthogonal projection to the invariant subspace.

    The input representation :math:`\rho_{\mathcal{X}}: \mathbb{G} \mapsto \mathbb{G}\mathbb{L}(\mathcal{X})` is transformed to the spectral basis given by:

    .. math::
        \rho_\mathcal{X} = \mathbf{Q} \left( \bigoplus_{i\in[1,n]} \hat{\rho}_i \right) \mathbf{Q}^T

    where :math:`\hat{\rho}_i` denotes an instance of one of the irreducible representations of the group, and :math:`\mathbf{Q}: \mathcal{X} \mapsto \mathcal{X}` is the orthogonal change of basis from the spectral basis to the original basis.

    The projection is performed by:
        1. Changing the basis to the representation spectral basis (exposing signals per irrep).
        2. Zeroing out all signals on irreps that are not trivial.
        3. Mapping back to the original basis set.

    Args:
        rep_X (Representation): The representation for which the orthogonal projection to the invariant subspace is computed.

    Returns:
        Tensor: The orthogonal projection matrix to the invariant subspace, :math:`\mathbf{Q} \mathbf{S} \mathbf{Q}^T`.
    """
    Qx_T, Qx = Tensor(rep_X.change_of_basis_inv), Tensor(rep_X.change_of_basis)

    # S is an indicator of which dimension (in the irrep-spectral basis) is associated with a trivial irrep
    S = torch.zeros((rep_X.size, rep_X.size))
    irreps_dimension = []
    cum_dim = 0
    for irrep_id in rep_X.irreps:
        irrep = rep_X.group.irrep(*irrep_id)
        # Get dimensions of the irrep in the original basis
        irrep_dims = range(cum_dim, cum_dim + irrep.size)
        irreps_dimension.append(irrep_dims)
        if irrep_id == rep_X.group.trivial_representation.id:
            # this dimension is associated with a trivial irrep
            S[irrep_dims, irrep_dims] = 1
        cum_dim += irrep.size

    inv_projector = Qx @ S @ Qx_T
    return inv_projector


def isotypic_signal2irreducible_subspaces(X: Tensor, repX: Representation):
    r"""Given a random variable in an isotypic subspace, flatten the r.v. into G-irreducible subspaces.

    Given a signal of shape :math:`(n, m_x \cdot d)` where :math:`n` is the number of samples, :math:`m_x` the multiplicity of the irrep in :math:`X`, and :math:`d` the dimension of the irrep.
    :math:`X = [x_1, \ldots, x_n]` and :math:`x_i = [x_{i_{11}}, \ldots, x_{i_{1d}}, x_{i_{21}}, \ldots, x_{i_{2d}}, \ldots, x_{i_{m_x1}}, \ldots, x_{i_{m_xd}}]`

    This function returns the signal :math:`Z` of shape :math:`(n \cdot d, m_x)` where each column represents the flattened signal of a G-irreducible subspace.
    :math:`Z[:, k] = [x_{1_{k1}}, \ldots, x_{1_{kd}}, x_{2_{k1}}, \ldots, x_{2_{kd}}, \ldots, x_{n_{k1}}, \ldots, x_{n_{kd}}]`

    Args:
        X (Tensor): Shape :math:`(..., n, m_x \cdot d)` where :math:`n` is the number of samples and :math:`m_x` the multiplicity of the irrep in :math:`X`.
        repX (escnn.nn.Representation): Representation in the isotypic basis of a single type of irrep.

    Returns:
        Tensor:

    Shape:
        :math:`(n \cdot d, m_x)`, where each column represents the flattened signal of an irreducible subspace.
    """
    assert len(repX._irreps_multiplicities) == 1, (
        "Random variable is assumed to be in a single isotypic subspace."
    )
    irrep_id = repX.irreps[0]
    irrep_dim = repX.group.irrep(*irrep_id).size
    mk = repX._irreps_multiplicities[irrep_id]  # Multiplicity of the irrep in X

    Z = X.view(-1, mk, irrep_dim).permute(0, 2, 1).reshape(-1, mk)

    assert Z.shape == (X.shape[0] * irrep_dim, mk)
    return Z

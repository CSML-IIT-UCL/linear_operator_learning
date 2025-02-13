"""Linear algebra utilities for symmetric vector spaces with known group representations."""

# Created by Daniel Ordoñez (daniels.ordonez@gmail.com) at 13/02/25
import torch
from escnn.group import Representation


def invariant_orthogonal_projector(rep_X: Representation) -> torch.Tensor:
    r"""Computes the orthogonal projection to the invariant subspace.

    The representation `rep_X` is transformed to the spectral basis using the change of basis matrix `Q`:

    .. math::
        rep_X = Q (\bigoplus_i^n \\hat{\rho}_i) Q^T

    The projection is performed by:
        1. Change basis to representation spectral basis (exposing signals per irrep).
        2. Zero out all signals on irreps that are not trivial.
        3. Map back to original basis set.

    Args:
        rep_X (Representation): The representation for which the orthogonal projection to the invariant subspace is computed.

    Returns:
        torch.Tensor: The orthogonal projection matrix to the invariant subspace. Q S Q^T
    """
    Qx_T, Qx = torch.Tensor(rep_X.change_of_basis_inv), torch.Tensor(rep_X.change_of_basis)

    # S is an indicator of which dimension (in the irrep-spectral basis) is associated with a trivial irrep
    S = torch.zeros((rep_X.size, rep_X.size))
    irreps_dimension = []
    cum_dim = 0
    for irrep_id in rep_X.irreps:
        irrep = rep_X.group.irrep(*irrep_id)
        # Get dimensions of the irrep in the original basis
        irrep_dims = range(cum_dim, cum_dim + irrep.size)
        irreps_dimension.append(irrep_dims)
        if (
            irrep_id == rep_X.group.trivial_representation.id
        ):  # this dimension is associated with a trivial irrep
            S[irrep_dims, irrep_dims] = 1
        cum_dim += irrep.size

    inv_projector = Qx @ S @ Qx_T
    return inv_projector

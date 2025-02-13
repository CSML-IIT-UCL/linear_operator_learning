"""Statistics utilities for symmetric random variables with known group representations."""

# Created by Daniel Ordoñez (daniels.ordonez@gmail.com) at 13/02/25
import torch
from escnn.group import Representation


def symmetric_moments(x: torch.Tensor, rep_X: Representation) -> [torch.Tensor, torch.Tensor]:
    """Compute the mean and variance of a symmetric random variable.

    Args:
        x: (torch.Tensor) of shape (N, dx) containing the observations of the symmetric random variable
        rep_X: (escnn.group.Representation) representation of the symmetric random variable.

    Returns:
        mean: (torch.Tensor) of shape (dx,) containing the mean of the symmetric random variable, restricted to be
                in the trivial/G-invariant subspace of the symmetric vector space.
        var: (torch.Tensor) of shape (dx,) containing the variance of the symmetric random variable, constrained to be
                the same for all dimensions of each G-irreducible subspace (i.e., each subspace associated with an
                irrep).
    """
    assert len(x.shape) == 2, f"Expected x to have shape (n_samples, n_features), got {x.shape}"
    G = rep_X.group
    # Allocate the mean and variance arrays.
    mean, var = torch.zeros(rep_X.size), torch.ones(rep_X.size)
    # Change basis of the observation to expose the irrep G-stable subspaces
    Qx_T, Qx = torch.Tensor(rep_X.change_of_basis_inv), torch.Tensor(rep_X.change_of_basis)

    # Get the dimensions of each irrep.
    S = torch.zeros((rep_X.size, rep_X.size))
    irreps_dimension = []
    cum_dim = 0
    for irrep_id in rep_X.irreps:
        irrep = G.irrep(*irrep_id)
        # Get dimensions of the irrep in the original basis
        irrep_dims = range(cum_dim, cum_dim + irrep.size)
        irreps_dimension.append(irrep_dims)
        if irrep_id == G.trivial_representation.id:
            S[irrep_dims, irrep_dims] = 1
        cum_dim += irrep.size

    # Compute the mean of the observation.
    # The mean of a symmetric random variable (rv) lives in the subspaces associated with the trivial/inv irreps.
    has_trivial_irreps = G.trivial_representation.id in rep_X.irreps
    if has_trivial_irreps:
        avg_projector = Qx @ S @ Qx_T
        # Compute the mean in a single vectorized operation
        mean_empirical = torch.mean(x, dim=0)
        # Project to the inv-subspace and map back to the original basis
        mean = torch.einsum("...ij,...j->...i", avg_projector, mean_empirical)

    # Compute the variance of the observable by computing a single variance per irrep G-stable subspace.
    # To do this, we project the observations to the basis exposing the irreps, compute the variance per
    # G-stable subspace, and map the variance back to the original basis.
    x_iso_centered = torch.einsum("...ij,...j->...i", Qx_T, x - mean)
    var_irrep_basis = torch.ones_like(var)
    for irrep_id, irrep_dims in zip(rep_X.irreps, irreps_dimension):
        irrep = G.irrep(*irrep_id)
        x_irrep_centered = x_iso_centered[..., irrep_dims]
        assert x_irrep_centered.shape[-1] == irrep.size, (
            f"Obs irrep shape {x_irrep_centered.shape} != {irrep.size}"
        )
        # Since the irreps are unitary/orthogonal transformations, we are constrained compute a unique variance
        # for all dimensions of the irrep G-stable subspace, as scaling the dimensions independently would break
        # the symmetry of the rv. As a centered rv the variance is the expectation of the squared rv.
        var_irrep = torch.mean(x_irrep_centered**2)  # Single scalar variance per G-stable subspace
        # Ensure the multipliticy of the variance is equal to the dimension of the irrep.
        var_irrep_basis[irrep_dims] = var_irrep
    # Convert the variance from the irrep/spectral basis to the original basis
    Cov = Qx @ torch.diag(var_irrep_basis) @ Qx_T
    var = torch.diagonal(Cov)

    # TODO: Move this check to Unit test as it is computationally demanding to check this at runtime.
    # Ensure the mean is equivalent to computing the mean of the orbit of the recording under the group action
    # aug_obs = []
    # for g in G.elements:
    #     g_obs = np.einsum('...ij,...j->...i', rep_obs(g), obs_original_basis)
    #     aug_obs.append(g_obs)
    #
    # aug_obs = np.concatenate(aug_obs, axis=0)   # Append over the trajectory dimension
    # mean_emp = np.mean(aug_obs, axis=(0, 1))
    # assert np.allclose(mean, mean_emp, rtol=1e-3, atol=1e-3), f"Mean {mean} != {mean_emp}"

    # var_emp = np.var(aug_obs, axis=(0, 1))
    # assert np.allclose(var, var_emp, rtol=1e-2, atol=1e-2), f"Var {var} != {var_emp}"
    return mean, var

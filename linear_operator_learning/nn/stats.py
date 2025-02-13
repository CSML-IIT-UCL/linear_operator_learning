"""Statistics utilities for multi-variate random variables."""

import torch


def cross_cov_norm_squared_unbiased(x: torch.Tensor, y: torch.Tensor, permutation=None):
    r"""Compute the unbiased estimation of :math:`\|\mathbf{C}_{xy}\|_F^2` from a batch of samples, using U-statistics.

    Given the Covariance matrix :math:`\mathbf{C}_{xy} = \mathbb{E}_p(x,y) [x^T y]`, this function computes an unbiased estimation
    of the Frobenius norm of the covariance matrix from two independent sampling sets (an effective samples size of :math:`N^2`).

    .. math::

        \begin{align}
            \|\mathbf{C}_{xy}\|_F^2 &= \text{tr}(\mathbf{C}_{xy}^T \mathbf{C}_{xy})
            = \sum_i \sum_j (\mathbb{E}_{x,y \sim p(x,y)} [x_i y_j]) (\mathbb{E}_{x',y' \sim p(x,y)} [x_j y_i']) \\
            &= \mathbb{E}_{(x,y),(x',y') \sim p(x,y)} [(x^T y') (x'^T y)] \\
            &\approx \frac{1}{N^2} \sum_n \sum_m [(x_n^T y'_m) (x'_m^T y_n)]
        \end{align}

    .. note::
    The random variable is assumed to be centered.

    Args:
        x (torch.Tensor): (n_samples, r_x) Centered realizations of a random variable x = [x_1, ..., x_{r_x}].
        y (torch.Tensor): (n_samples, r_y) Centered realizations of a random variable y = [y_1, ..., y_{r_y}].
        permutation (torch.Tensor, optional): List of integer indices of shape (n_samples,) used to permute the samples.

    Returns:
        torch.Tensor: Unbiased estimation of :math:`\|\mathbf{C}_{xy}\|_F^2`.
    """
    n_samples = x.shape[0]

    # Permute the rows independently to simulate independent sampling
    perm = permutation if permutation is not None else torch.randperm(n_samples)
    assert perm.shape == (n_samples,), f"Invalid permutation {perm.shape}!=({n_samples},)"
    xp = x[perm]  # Independent sampling of x'
    yp = y[perm]  # Independent sampling of y'

    # Compute 1/N^2 Σ_n Σ_m [(x_n.T y'_m) (x'_m.T y_n)]
    val = torch.einsum("nj,mj,mk,nk->", x, yp, xp, y)
    cov_fro_norm = val / (n_samples**2)
    return cov_fro_norm


def cov_norm_squared_unbiased(x: torch.Tensor, permutation=None):
    r"""Compute the unbiased estimation of :math:`\|\mathbf{C}_x\|_F^2` from a batch of samples.

    Given the Covariance matrix :math:`\mathbf{C}_x = \mathbb{E}_p(x) [x^T x]`, this function computes an unbiased estimation
    of the Frobenius norm of the covariance matrix from a single sampling set.

    :math:`\|\mathbf{C}_x\|_F^2 = \text{tr}(\mathbf{C}_x^T \mathbf{C}_x) = \sum_i \sum_j (\mathbb{E}_{x} [x_i x_j]) (\mathbb{E}_{x'} [x'_j x'_i])`
    :math:`= \mathbb{E}_{x,x' \sim p(x)} [(x^T x')^2]`
    :math:`\approx \frac{1}{N^2} \sum_n \sum_m [(x_n^T x'_m)^2]`

    .. note::
        The random variable is assumed to be centered.

    Args:
        x (torch.Tensor): (n_samples, r) Centered realizations of a random variable x = [x_1, ..., x_r].
        permutation (torch.Tensor, optional): List of integer indices of shape (n_samples,) used to permute the samples.

    Returns:
        torch.Tensor: Unbiased estimation of :math:`\|\mathbf{C}_x\|_F^2`.
    """
    return cross_cov_norm_squared_unbiased(x=x, y=x, permutation=permutation)

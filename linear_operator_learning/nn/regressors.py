"""NN regressors."""

import torch

from linear_operator_learning.nn.structs import FitResult


def ridge_least_squares(
    cov_X: torch.Tensor,
    tikhonov_reg: float = 0.0,
) -> FitResult:
    """Fit the ridge least squares estimator for the transfer operator.

    Args:
        cov_X (torch.Tensor): covariance matrix of the input data.
        tikhonov_reg (float, optional): Ridge regularization. Defaults to 0.0.

    Returns:
        FitResult: as defined in operator_learning.structs
    """
    dim = cov_X.shape[0]
    reg_input_covariance = cov_X + tikhonov_reg * torch.eye(
        dim, dtype=cov_X.dtype, device=cov_X.device
    )
    values, vectors = torch.linalg.eigh(reg_input_covariance)
    # Divide columns of vectors by square root of eigenvalues
    rsqrt_evals = 1.0 / torch.sqrt(values + 1e-10)
    Q = vectors @ torch.diag(rsqrt_evals)
    result: FitResult = FitResult({"U": Q, "V": Q, "svals": values})
    return result

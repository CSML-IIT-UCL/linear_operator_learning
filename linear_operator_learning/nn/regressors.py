"""NN regressors."""

from typing import Literal

import numpy as np
import scipy.linalg
import torch
from torch import Tensor

from linear_operator_learning.nn.structs import EigResult, FitResult


def ridge_least_squares(
    cov_X: Tensor,
    tikhonov_reg: float = 0.0,
) -> FitResult:
    """Fit the ridge least squares estimator for the transfer operator.

    Args:
        cov_X (Tensor): covariance matrix of the input data.
        tikhonov_reg (float, optional): Ridge regularization. Defaults to 0.0.

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


def eig(
    fit_result: FitResult,
    cov_XY: Tensor,
) -> EigResult:
    """Computes the eigendecomposition of a regressor.

    Args:
        fit_result (FitResult): Fit result as defined in ``linear_operator_learning.nn.structs``.
        cov_XY (Tensor): Cross covariance matrix between the input and output data.


    Shape:
        ``cov_XY``: :math:`(D, D)`, where :math:`D` is the number of features.

        Output: ``U, V`` of shape :math:`(D, R)`, ``svals`` of shape :math:`R`
        where :math:`D` is the number of features and  :math:`R` is the rank of the regressor.
    """
    dtype_and_device = {
        "dtype": cov_XY.dtype,
        "device": cov_XY.device,
    }
    U = fit_result["U"]
    # Using the trick described in https://arxiv.org/abs/1905.11490
    M = torch.linalg.multi_dot([U.T, cov_XY, U])
    # Convertion to numpy
    M = M.numpy(force=True)
    values, lv, rv = scipy.linalg.eig(M, left=True, right=True)
    r_perm = torch.tensor(np.argsort(values), device=cov_XY.device)
    l_perm = torch.tensor(np.argsort(values.conj()), device=cov_XY.device)
    values = values[r_perm]
    # Back to torch, casting to appropriate dtype and device
    values = torch.complex(
        torch.tensor(values.real, **dtype_and_device), torch.tensor(values.imag, **dtype_and_device)
    )
    lv = torch.complex(
        torch.tensor(lv.real, **dtype_and_device), torch.tensor(lv.imag, **dtype_and_device)
    )
    rv = torch.complex(
        torch.tensor(rv.real, **dtype_and_device), torch.tensor(rv.imag, **dtype_and_device)
    )
    # Normalization in RKHS norm
    rv = U.to(rv.dtype) @ rv
    rv = rv[:, r_perm]
    rv = rv / torch.linalg.norm(rv, axis=0)
    # # Biorthogonalization
    lv = torch.linalg.multi_dot([cov_XY.T.to(lv.dtype), U.to(lv.dtype), lv])
    lv = lv[:, l_perm]
    l_norm = torch.sum(lv * rv, axis=0)
    lv = lv / l_norm
    result: EigResult = EigResult({"values": values, "left": lv, "right": rv})
    return result


def evaluate_eigenfunction(
    eig_result: EigResult,
    which: Literal["left", "right"],
    X: Tensor,
):
    """Evaluates left or right eigenfunctions of a regressor.

    Args:
        eig_result: EigResult object containing eigendecomposition results
        which: String indicating "left" or "right" eigenfunctions.
        X: Feature map of the input data


    Shape:
        ``eig_results``: ``U, V`` of shape :math:`(D, R)`, ``svals`` of shape :math:`R`
        where :math:`D` is the number of features and  :math:`R` is the rank of the regressor.

        ``X``: :math:`(N_0, D)`, where :math:`N_0` is the number of inputs to predict and :math:`D` is the number of features.

        Output: :math:`(N_0, R)`
    """
    vr_or_vl = eig_result[which]
    return X.to(vr_or_vl.dtype) @ vr_or_vl

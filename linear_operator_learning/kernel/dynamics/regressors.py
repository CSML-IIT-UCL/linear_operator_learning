"""Kernel-based regressors for linear operators based on dynamical trajectories."""

from math import sqrt
from typing import Callable, Literal, Union
from warnings import warn

import numpy as np
import scipy.linalg
from numpy import ndarray
from scipy.sparse.linalg import eigs, eigsh

from linear_operator_learning.kernel.dynamics.structs import ModeResult
from linear_operator_learning.kernel.linalg import (
    add_diagonal_,
    stable_topk,
    weighted_norm,
)
from linear_operator_learning.kernel.regressors import evaluate_eigenfunction
from linear_operator_learning.kernel.structs import EigResult, FitResult
from linear_operator_learning.kernel.utils import sanitize_complex_conjugates

__all__ = ["reduced_rank_regression_dirichlet", "modes", "predict"]


def reduced_rank_regression_dirichlet(
    kernel_X: ndarray,  # kernel matrix of the training data
    dKernel_X: ndarray,  # derivative of the kernel: dK_X_{i,j} = <\phi(x_i),d\phi(x_j)> (matrix N in the paper)
    dKernel_dX: ndarray,  # derivative of the kernel dK_dX_{i,j} = <d\phi(x_i),d\phi(x_j)>
    shift: float,  # shift parameter of the resolvent
    tikhonov_reg: float,  # Tikhonov (ridge) regularization parameter, can be 0,
    rank: int,
):  # Rank of the estimator)
    r"""Fits the physics informed Reduced Rank Estimator from :footcite:t:`kostic2024learning`.

    Args:
        kernel_X (np.ndarray): kernel matrix of the training data
        dKernel_X (np.ndarray): (matrix N in :footcite:t:`kostic2024learning`) derivative of the kernel: :math:`N_{i,(k-1)n+j} = \langle \phi(x_i),d_k\phi(x_j) \rangle`
        dKernel_dX (np.ndarray):  (matrix M in :footcite:t:`kostic2024learning`) derivative of the kernel :math:`M_{(l-1)n+i,(k-1)n+j} = \langle d_l\phi(x_i),d_k\phi(x_j) \rangle`
        shift (float): shift parameter of the resolvent
        tikhonov_reg (float): Tikhonov (ridge) regularization parameter
        rank (int): Rank of the estimator

    Shape:
        ``kernel_X``: :math:`(N, N)`, where :math:`N` is the number of training data.

        ``dkernel_X``: :math:`(N, (d+1)N)`. where :math:`N` is the number of training data amd :math:`d` is the dimensionality of the input data.

        ``dkernel_dX``: :math:`((d+1)N, (d+1)N)`. where :math:`N` is the number of training data amd :math:`d` is the dimensionality of the input data.
    Returns: EigResult structure containing the eigenvalues and eigenvectors
    """
    npts = kernel_X.shape[0]
    sqrt_npts = np.sqrt(npts)

    dimension_derivative = dKernel_dX.shape[0]

    # We follow the notation of the paper
    J = (
        kernel_X / sqrt_npts
        - (dKernel_X / sqrt_npts)
        @ np.linalg.inv(
            dKernel_dX + tikhonov_reg * shift * sqrt_npts * np.eye(dimension_derivative)
        )
        @ dKernel_X.T
    )

    sigmas_2, vectors = eigs(
        J @ kernel_X / sqrt_npts, k=rank + 5, M=(J + tikhonov_reg * np.eye(J.shape[0])) * shift
    )

    values, stable_values_idxs = stable_topk(sigmas_2, rank, ignore_warnings=False)

    V = vectors[:, stable_values_idxs]
    # Normalization step
    V = V @ np.diag(np.sqrt(sqrt_npts) / np.sqrt(np.diag(V.T @ kernel_X @ V)))

    sigma = np.diag(sigmas_2[stable_values_idxs])

    make_U = np.block(
        [
            np.eye(npts) / np.sqrt(shift),
            -dKernel_X
            @ np.linalg.inv(
                dKernel_dX + shift * tikhonov_reg * sqrt_npts * np.eye(dimension_derivative)
            ),
        ]
    ).T

    U = make_U @ (kernel_X @ V / sqrt_npts - shift * V @ sigma) / (tikhonov_reg * shift)

    evals, vl, vr = scipy.linalg.eig(V.T @ V @ sigma, left=True)

    evals = sanitize_complex_conjugates(evals)  # To be coupled with the LOL function
    lambdas = shift - 1 / evals
    r_perm = np.argsort(-lambdas)
    vr = vr[:, r_perm]
    l_perm = np.argsort(-lambdas.conj())
    vl = vl[:, l_perm]
    evals = evals[r_perm]

    vl /= np.sqrt(evals)
    result: EigResult = {"values": lambdas[r_perm], "left": (V @ vl) * sqrt_npts, "right": U @ vr}
    return result


def modes(
    eig_result: EigResult,
    initial_conditions: ndarray,  # Feature matrix of the shape [num_initial_conditions, features] or kernel matrix of the shape [num_initial_conditions, num_training_points]
    obs_train: ndarray,  # Observable to be predicted evaluated on the trajectory data, shape [num_training_points, obs_features]
) -> ModeResult:
    r"""Computes the decomposition of an observable in the eigenmode basis.

    Given the eigenfunctions obtained from a fit function, this function computes
    the decomposition of a given observable in terms of these eigenmodes. The resulting
    modes provide insight into the dominant temporal patterns governing the system.

    Args:
        eig_result (EigResult): Eigen decomposition result containing eigenvalues and left/right eigenfunctions.
        initial_conditions (np.ndarray):
            kernel matrix of shape :math:`(N_{init}, N)`, where :math:`N_{init}` is the number
            of initial conditions and :math:`N` is the number of training points.
        obs_train (np.ndarray): Observable evaluated on the training trajectory data,
            of shape :math:`(N, d_{obs})`, where :math:`N` is the number of training points
            and :math:`d_{obs}` is the number of observable features.

    Shape:
        - ``eig_result["left"]``: :math:`(N, r)`, where :math:`r` is the rank of the decomposition.
        - ``eig_result["right"]``: :math:`(N, r)`, right eigenfunctions of the transition operator.
        - ``eig_result["values"]``: :math:`(r,)`, complex eigenvalues of the transition operator.
        - ``initial_conditions``:  :math:`(N_{init}, N)`, kernel representation or  :math:`(N_{init}, N*(d+1))` if using the dirichlet estimator.
        - ``obs_train``: :math:`(N, d_{obs})`, observable evaluated on training data.

    Returns:
        ModeResult: A dictionary containing:
            - ``decay_rates`` (:math:`(r,)`): Real part of the eigenvalues, representing decay rates.
            - ``frequencies`` (:math:`(r,)`): Imaginary part of the eigenvalues, representing oscillation frequencies.
            - ``modes`` (:math:`(r, N_{init}, d_{obs})`): Observable decomposed in the eigenmode basis.
    """
    evals = eig_result["values"]
    levecs = eig_result["left"]
    npts = levecs.shape[
        0
    ]  # We use the eigenvector to be consistent with the dirichlet estimator that does not have the same shape #obs_train.shape[0]
    if initial_conditions.ndim == 1:
        initial_conditions = np.expand_dims(initial_conditions, axis=0)
    conditioning = evaluate_eigenfunction(
        eig_result, "right", initial_conditions
    ).T  # [rank, num_initial_conditions]
    str = "abcdefgh"  # Maximum number of feature dimensions is 8
    einsum_str = "".join([str[k] for k in range(obs_train.ndim - 1)])  # string for features
    modes_ = np.einsum("nr,n" + einsum_str + "->r" + einsum_str, levecs.conj(), obs_train) / sqrt(
        npts
    )  # [rank, features]
    modes_ = np.expand_dims(modes_, axis=1)
    dims_to_add = modes_.ndim - conditioning.ndim
    if dims_to_add > 0:
        conditioning = np.expand_dims(conditioning, axis=tuple(range(-dims_to_add, 0)))
    modes = conditioning * modes_  # [rank, num_init_cond, obs_features]
    result: ModeResult = {
        "decay_rates": -evals.real,
        "frequencies": evals.imag / (2 * np.pi),
        "modes": modes,
    }
    return result


def predict(
    t: Union[float, ndarray],  # time in the same units as dt
    mode_result: ModeResult,
) -> ndarray:  # shape [num_init_cond, features] or if num_t>1 [num_init_cond, num_time, features]
    r"""Predicts the evolution of an observable using the mode decomposition.

    Given the modal decomposition of an observable, this function reconstructs
    the time evolution by propagating the eigenmodes forward in time.

    Args:
        t (Union[float, np.ndarray]): Time or array of time values in the same units as the spectral decomposition.
        mode_result (ModeResult): Result of the mode decomposition, containing decay rates, frequencies, and modes.

    Shape:
        - ``t``: :math:`(T,)` or scalar, where :math:`T` is the number of time steps.
        - ``mode_result["decay_rates"]``: :math:`(r,)`, real decay rates of the modes.
        - ``mode_result["frequencies"]``: :math:`(r,)`, imaginary frequencies of the modes.
        - ``mode_result["modes"]``: :math:`(r, N_{init}, d_{obs})`, mode coefficients.
        - ``predictions``: :math:`(N_{init}, T, d_{obs})`, predicted observable values at times :math:`t`.
          If :math:`T=1` or :math:`N_{init}=1`, unnecessary dimensions are removed.

    Returns:
        np.ndarray: Predicted observable values at the given time steps.
        If multiple time points are given, the shape is :math:`(N_{init}, T, d_{obs})`.
        If only one time step or one initial condition is provided, the output is squeezed accordingly.
    """
    if type(t) == float:  # noqa: E721
        t = np.array([t])
    evals = -mode_result["decay_rates"] + 2 * np.pi * 1j * mode_result["frequencies"]
    to_evals = np.exp(evals[:, None] * t[None, :])  # [rank,time_steps]
    str = "abcdefgh"  # Maximum number of feature dimensions is 8
    einsum_str = "".join(
        [str[k] for k in range(mode_result["modes"].ndim - 2)]
    )  # string for features
    predictions = np.einsum(
        "rs,rm" + einsum_str + "->ms" + einsum_str, to_evals, mode_result["modes"]
    )
    if (
        predictions.shape[0] == 1 or predictions.shape[1] == 1
    ):  # If only one time point or one initial condition is requested, remove unnecessary dims
        predictions = np.squeeze(predictions)
    return predictions.real

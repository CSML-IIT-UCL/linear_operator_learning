"""Kernel-based regressors for linear operators."""

from math import sqrt
from typing import Literal
from warnings import warn

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eig, eigh, lstsq, qr
from scipy.sparse.linalg import eigs, eigsh

from linear_operator_learning.kernel.linalg import add_diagonal_, stable_topk
from linear_operator_learning.kernel.structs import FitResult

__all__ = [
    "predict",
    "pcr",
    "nystroem_pcr",
    "reduced_rank",
    "nystroem_reduced_rank",
    "rand_reduced_rank",
]


def predict(
    num_steps: int,
    fit_result: FitResult,
    K_YX: np.ndarray,
    K_Xin_X: np.ndarray,
    obs_train_Y: np.ndarray,
) -> np.ndarray:
    """Predicts future states using kernel matrices and fitted results.

    Args:
        num_steps (int): Number of steps to predict forward (returns the last prediction)
        fit_result (FitResult): FitResult object containing fitted U and V matrices
        K_YX (np.ndarray): Kernel matrix between output data and input data (or inducing points for Nystroem)
        K_Xin_X (np.ndarray): Kernel matrix between initial conditions and input data (or inducing points for Nystroem)
        obs_train_Y (np.ndarray): Observable evaluated on output training data (or inducing points for Nystroem)
    """
    # G = S UV.T Z
    # G^n = (SU)(V.T K_YX U)^(n-1)(V.T Z)
    U = fit_result["U"]
    V = fit_result["V"]
    npts = U.shape[0]
    K_dot_U = K_Xin_X @ U / sqrt(npts)
    V_dot_obs = V.T @ obs_train_Y / sqrt(npts)
    V_K_YX_U = np.linalg.multi_dot([V.T, K_YX, U]) / npts
    M = np.linalg.matrix_power(V_K_YX_U, num_steps - 1)
    return np.linalg.multi_dot([K_dot_U, M, V_dot_obs])


def pcr(
    kernel_X: np.ndarray,
    tikhonov_reg: float = 0.0,
    rank: int | None = None,
    svd_solver: Literal["arnoldi", "full"] = "arnoldi",
) -> FitResult:
    """Fits the Principal Components estimator.

    Args:
        kernel_X (np.ndarray): Kernel matrix of the input data.
        tikhonov_reg (float, optional): Tikhonov (ridge) regularization parameter. Defaults to 0.0.
        rank (int | None, optional): Rank of the estimator. Defaults to None.
        svd_solver (Literal[ &quot;arnoldi&quot;, &quot;full&quot; ], optional): Solver for the generalized eigenvalue problem. Defaults to "arnoldi".

    """
    npts = kernel_X.shape[0]
    add_diagonal_(kernel_X, npts * tikhonov_reg)
    if svd_solver == "arnoldi":
        _num_arnoldi_eigs = min(rank + 5, kernel_X.shape[0])
        values, vectors = eigsh(kernel_X, k=_num_arnoldi_eigs)
    elif svd_solver == "full":
        values, vectors = eigh(kernel_X)
    else:
        raise ValueError(f"Unknown svd_solver {svd_solver}")
    add_diagonal_(kernel_X, -npts * tikhonov_reg)

    values, stable_values_idxs = stable_topk(values, rank, ignore_warnings=False)
    vectors = vectors[:, stable_values_idxs]
    Q = sqrt(npts) * vectors / np.sqrt(values)
    kernel_X_eigvalsh = np.sqrt(np.abs(values)) / npts
    result: FitResult = {"U": Q, "V": Q, "svals": kernel_X_eigvalsh}
    return result


def nystroem_pcr(
    kernel_X: np.ndarray,  # Kernel matrix of the input inducing points
    kernel_Y: np.ndarray,  # Kernel matrix of the output inducing points
    kernel_Xnys: np.ndarray,  # Kernel matrix between the input data and the input inducing points
    kernel_Ynys: np.ndarray,  # Kernel matrix between the output data and the output inducing points
    tikhonov_reg: float = 0.0,  # Tikhonov (ridge) regularization parameter (can be 0)
    rank: int | None = None,  # Rank of the estimator
    svd_solver: Literal["arnoldi", "full"] = "arnoldi",
) -> FitResult:
    """Fits the Principal Components estimator using the Nyström method from :footcite:t:`Meanti2023`.

    Args:
        kernel_X (np.ndarray): Kernel matrix of the input inducing points.
        kernel_Y (np.ndarray): Kernel matrix of the output inducing points.
        kernel_Xnys (np.ndarray): Kernel matrix between the input data and the input inducing points.
        kernel_Ynys (np.ndarray): Kernel matrix between the output data and the output inducing points.
        tikhonov_reg (float, optional): Tikhonov (ridge) regularization parameter. Defaults to 0.0.
        rank (int | None, optional): Rank of the estimator. Defaults to None.
        svd_solver (Literal[ "arnoldi", "full" ], optional): Solver for the generalized eigenvalue problem. Defaults to "arnoldi".
    """
    ncenters = kernel_X.shape[0]
    npts = kernel_Xnys.shape[0]
    eps = 1000 * np.finfo(kernel_X.dtype).eps
    reg = max(eps, tikhonov_reg)
    kernel_Xnys_sq = kernel_Xnys.T @ kernel_Xnys
    add_diagonal_(kernel_X, reg * ncenters)
    if svd_solver == "full":
        values, vectors = eigh(
            kernel_Xnys_sq, kernel_X
        )  # normalization leads to needing to invert evals
    elif svd_solver == "arnoldi":
        _oversampling = max(10, 4 * int(np.sqrt(rank)))
        _num_arnoldi_eigs = min(rank + _oversampling, ncenters)
        values, vectors = eigsh(
            kernel_Xnys_sq,
            M=kernel_X,
            k=_num_arnoldi_eigs,
            which="LM",
        )
    else:
        raise ValueError(f"Unknown svd_solver {svd_solver}")
    add_diagonal_(kernel_X, -reg * ncenters)

    values, stable_values_idxs = stable_topk(values, rank, ignore_warnings=False)
    vectors = vectors[:, stable_values_idxs]

    U = sqrt(ncenters) * vectors / np.sqrt(values)
    V = np.linalg.multi_dot([kernel_Ynys.T, kernel_Xnys, vectors])
    V = lstsq(kernel_Y, V)[0]
    V = sqrt(ncenters) * V / np.sqrt(values)

    kernel_X_eigvalsh = np.sqrt(np.abs(values)) / npts
    result: FitResult = {"U": U, "V": V, "svals": kernel_X_eigvalsh}
    return result


def reduced_rank(
    kernel_X: np.ndarray,  # Kernel matrix of the input data
    kernel_Y: np.ndarray,  # Kernel matrix of the output data
    tikhonov_reg: float,  # Tikhonov (ridge) regularization parameter, can be 0
    rank: int,  # Rank of the estimator
    svd_solver: Literal["arnoldi", "full"] = "arnoldi",
) -> FitResult:
    """Fits the Reduced Rank estimator from :footcite:t:`Kostic2022`.

    Args:
        kernel_X (np.ndarray): Kernel matrix of the input data.
        kernel_Y (np.ndarray): Kernel matrix of the output data.
        tikhonov_reg (float): Tikhonov (ridge) regularization parameter.
        rank (int): Rank of the estimator.
        svd_solver (Literal[ "arnoldi", "full" ], optional): Solver for the generalized eigenvalue problem. Defaults to "arnoldi".
    """
    # Number of data points
    npts = kernel_X.shape[0]
    eps = 1000.0 * np.finfo(kernel_X.dtype).eps
    penalty = max(eps, tikhonov_reg) * npts

    A = (kernel_Y / sqrt(npts)) @ (kernel_X / sqrt(npts))
    add_diagonal_(kernel_X, penalty)
    # Find U via Generalized eigenvalue problem equivalent to the SVD. If K is ill-conditioned might be slow.
    # Prefer svd_solver == 'randomized' in such a case.
    if svd_solver == "arnoldi":
        # Adding a small buffer to the Arnoldi-computed eigenvalues.
        num_arnoldi_eigs = min(rank + 5, npts)
        values, vectors = eigs(A, k=num_arnoldi_eigs, M=kernel_X)
    elif svd_solver == "full":  # 'full'
        values, vectors = eig(A, kernel_X, overwrite_a=True, overwrite_b=True)
    else:
        raise ValueError(f"Unknown svd_solver: {svd_solver}")
    # Remove the penalty from kernel_X (inplace)
    add_diagonal_(kernel_X, -penalty)

    values, stable_values_idxs = stable_topk(values, rank, ignore_warnings=False)
    vectors = vectors[:, stable_values_idxs]
    # Compare the filtered eigenvalues with the regularization strength, and warn if there are any eigenvalues that are smaller than the regularization strength.
    if not np.all(np.abs(values) >= tikhonov_reg):
        warn(
            f"Warning: {(np.abs(values) < tikhonov_reg).sum()} out of the {len(values)} squared singular values are smaller than the regularization strength {tikhonov_reg:.2e}. Consider redudcing the regularization strength to avoid overfitting."
        )

    # Eigenvector normalization
    kernel_X_vecs = np.dot(kernel_X / sqrt(npts), vectors)
    vecs_norm = np.sqrt(
        np.sum(
            kernel_X_vecs**2 + tikhonov_reg * kernel_X_vecs * vectors * sqrt(npts),
            axis=0,
        )
    )

    norm_rcond = 1000.0 * np.finfo(values.dtype).eps
    values, stable_values_idxs = stable_topk(vecs_norm, rank, rcond=norm_rcond)
    U = vectors[:, stable_values_idxs] / vecs_norm[stable_values_idxs]

    # Ordering the results
    V = kernel_X @ U
    svals = np.sqrt(np.abs(values))
    result: FitResult = {"U": U.real, "V": V.real, "svals": svals}
    return result


def nystroem_reduced_rank(
    kernel_X: np.ndarray,  # Kernel matrix of the input inducing points
    kernel_Y: np.ndarray,  # Kernel matrix of the output inducing points
    kernel_Xnys: np.ndarray,  # Kernel matrix between the input data and the input inducing points
    kernel_Ynys: np.ndarray,  # Kernel matrix between the output data and the output inducing points
    tikhonov_reg: float,  # Tikhonov (ridge) regularization parameter
    rank: int,  # Rank of the estimator
    svd_solver: Literal["arnoldi", "full"] = "arnoldi",
) -> FitResult:
    """Fits the Nyström Reduced Rank estimator from :footcite:t:`Meanti2023`.

    Args:
        kernel_X (np.ndarray): Kernel matrix of the input inducing points.
        kernel_Y (np.ndarray): Kernel matrix of the output inducing points.
        kernel_Xnys (np.ndarray): Kernel matrix between the input data and the input inducing points.
        kernel_Ynys (np.ndarray): Kernel matrix between the output data and the output inducing points.
        tikhonov_reg (float): Tikhonov (ridge) regularization parameter.
        rank (int): Rank of the estimator.
        svd_solver (Literal[ "arnoldi", "full" ], optional): Solver for the generalized eigenvalue problem. Defaults to "arnoldi".
    """
    num_points = kernel_Xnys.shape[0]
    num_centers = kernel_X.shape[0]

    eps = 1000 * np.finfo(kernel_X.dtype).eps * num_centers
    reg = max(eps, tikhonov_reg)

    # LHS of the generalized eigenvalue problem
    sqrt_Mn = sqrt(num_centers * num_points)
    kernel_YX_nys = (kernel_Ynys.T / sqrt_Mn) @ (kernel_Xnys / sqrt_Mn)

    _tmp_YX = lstsq(kernel_Y * (num_centers**-1), kernel_YX_nys)[0]
    kernel_XYX = kernel_YX_nys.T @ _tmp_YX

    # RHS of the generalized eigenvalue problem
    kernel_Xnys_sq = (kernel_Xnys.T / sqrt_Mn) @ (kernel_Xnys / sqrt_Mn) + reg * kernel_X * (
        num_centers**-1
    )

    add_diagonal_(kernel_Xnys_sq, eps)
    A = lstsq(kernel_Xnys_sq, kernel_XYX)[0]
    if svd_solver == "full":
        values, vectors = eigh(
            kernel_XYX, kernel_Xnys_sq
        )  # normalization leads to needing to invert evals
    elif svd_solver == "arnoldi":
        _oversampling = max(10, 4 * int(np.sqrt(rank)))
        _num_arnoldi_eigs = min(rank + _oversampling, num_centers)
        values, vectors = eigs(kernel_XYX, k=_num_arnoldi_eigs, M=kernel_Xnys_sq)
    else:
        raise ValueError(f"Unknown svd_solver {svd_solver}")
    add_diagonal_(kernel_Xnys_sq, -eps)

    values, stable_values_idxs = stable_topk(values, rank, ignore_warnings=False)
    vectors = vectors[:, stable_values_idxs]
    # Compare the filtered eigenvalues with the regularization strength, and warn if there are any eigenvalues that are smaller than the regularization strength.
    if not np.all(np.abs(values) >= tikhonov_reg):
        warn(
            f"Warning: {(np.abs(values) < tikhonov_reg).sum()} out of the {len(values)} squared singular values are smaller than the regularization strength {tikhonov_reg:.2e}. Consider redudcing the regularization strength to avoid overfitting."
        )
    # Eigenvector normalization
    vecs_norm = np.sqrt(np.abs(np.sum(vectors.conj() * (kernel_XYX @ vectors), axis=0)))
    norm_rcond = 1000.0 * np.finfo(values.dtype).eps
    values, stable_values_idxs = stable_topk(vecs_norm, rank, rcond=norm_rcond)
    vectors = vectors[:, stable_values_idxs] / vecs_norm[stable_values_idxs]
    U = A @ vectors
    V = _tmp_YX @ vectors
    svals = np.sqrt(np.abs(values))
    result: FitResult = {"U": U.real, "V": V.real, "svals": svals}
    return result


def rand_reduced_rank(
    kernel_X: np.ndarray,
    kernel_Y: np.ndarray,
    tikhonov_reg: float,
    rank: int,
    n_oversamples: int = 5,
    optimal_sketching: bool = False,
    iterated_power: int = 1,
    rng_seed: int | None = None,
    precomputed_cholesky=None,
) -> FitResult:
    """Fits the Randomized Reduced Rank Estimator from :footcite:t:`Turri2023`.

    Args:
        kernel_X (np.ndarray): Kernel matrix of the input data
        kernel_Y (np.ndarray): Kernel matrix of the output data
        tikhonov_reg (float): Tikhonov (ridge) regularization parameter
        rank (int): Rank of the estimator
        n_oversamples (int, optional): Number of Oversamples. Defaults to 5.
        optimal_sketching (bool, optional): Whether to use optimal sketching (slower but more accurate) or not.. Defaults to False.
        iterated_power (int, optional): Number of iterations of the power method. Defaults to 1.
        rng_seed (int | None, optional): Random Number Generators seed. Defaults to None.
        precomputed_cholesky (optional): Precomputed Cholesky decomposition. Should be the output of cho_factor evaluated on the regularized kernel matrix.. Defaults to None.
    """
    rng = np.random.default_rng(rng_seed)
    npts = kernel_X.shape[0]

    penalty = npts * tikhonov_reg
    add_diagonal_(kernel_X, penalty)
    if precomputed_cholesky is None:
        cholesky_decomposition = cho_factor(kernel_X)
    else:
        cholesky_decomposition = precomputed_cholesky
    add_diagonal_(kernel_X, -penalty)

    sketch_dimension = rank + n_oversamples

    if optimal_sketching:
        cov = kernel_Y / npts
        sketch = rng.multivariate_normal(
            np.zeros(npts, dtype=kernel_Y.dtype), cov, size=sketch_dimension
        ).T
    else:
        sketch = rng.standard_normal(size=(npts, sketch_dimension))

    for _ in range(iterated_power):
        # Powered randomized rangefinder
        sketch = (kernel_Y / npts) @ (sketch - penalty * cho_solve(cholesky_decomposition, sketch))
        sketch, _ = qr(sketch, mode="economic")  # QR re-orthogonalization

    kernel_X_sketch = cho_solve(cholesky_decomposition, sketch)
    _M = sketch - penalty * kernel_X_sketch

    F_0 = sketch.T @ sketch - penalty * (sketch.T @ kernel_X_sketch)  # Symmetric
    F_0 = 0.5 * (F_0 + F_0.T)
    F_1 = _M.T @ ((kernel_Y @ _M) / npts)

    values, vectors = eig(lstsq(F_0, F_1)[0])
    values, stable_values_idxs = stable_topk(values, rank, ignore_warnings=False)
    vectors = vectors[:, stable_values_idxs]

    # Remove elements in the kernel of F_0
    relative_norm_sq = np.abs(
        np.sum(vectors.conj() * (F_0 @ vectors), axis=0) / np.linalg.norm(vectors, axis=0) ** 2
    )
    norm_rcond = 1000.0 * np.finfo(values.dtype).eps
    values, stable_values_idxs = stable_topk(relative_norm_sq, rank, rcond=norm_rcond)
    vectors = vectors[:, stable_values_idxs]

    vecs_norms = (np.sum(vectors.conj() * (F_0 @ vectors), axis=0).real) ** 0.5
    vectors = vectors / vecs_norms

    U = sqrt(npts) * kernel_X_sketch @ vectors
    V = sqrt(npts) * _M @ vectors
    svals = np.sqrt(values)
    result: FitResult = {"U": U, "V": V, "svals": svals}
    return result

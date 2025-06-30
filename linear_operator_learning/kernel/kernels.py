# linear_operator_learning/kernel/dynamics/kernels.py  # noqa: D100

import numpy as np
from sklearn.gaussian_process.kernels import RBF  # type: ignore


class RBF_with_grad(RBF):  # noqa: D101
    def __init___(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5)):  # noqa: D105
        # Init code here:
        super().__init__(length_scale, length_scale_bounds)

    def grad(self, kernel_X, X, friction):
        r"""Returns the matrix :math:`N_{i,(k-1)n+j}= \langle \phi(x_i),d_k\phi(x_j) \rangle` (matrix :math:`N` :footcite:t:`kostic2024learning`
        where :math:`i = 1,\dots n, k=1, \dots d, j=1,\dots n`
        and :math:`N` is the number of training points and :math:`d` is the dimensionality of the system.

        Args:
            kernel_X (np.ndarray): kernel matrix of the training data
            X (np.ndarray): training data
            friction (np.ndarray): friction parameter of the physical model :math:`s(x)` in :footcite:t:`kostic2024learning`
        Shape:
            ``kernel_X``: :math:`(N, N)`, where :math:`N` is the number of training data.

            ``X``: :math:`(N,d)`  where :math:`N` is the number of training data and :math:`d` the dimensionality of the system.

            ``friction``: :math:`d`  where :math:`d` is the dimensionality of the system.

        Output: :math:`\langle \phi(x_i),d_k\phi(x_j) \rangle` of shape `N,Nd`, where :math:`N` is the number of training data and :math:`d` the dimension of the system.
        """  # noqa: D205
        difference = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        n = difference.shape[0]
        d = X.shape[1]
        N = np.zeros((n, n * d))
        sigma = self.length_scale
        for i in range(n):
            for j in range(n):
                for k in range(0, d):
                    N[i, k * n + j] = (
                        np.sqrt(friction[k]) * difference[i, j, k] * kernel_X[i, j] / sigma**2
                    )
        return N

    def grad2(self, kernel_X: np.ndarray, X: np.ndarray, friction: np.ndarray):
        r"""Returns the matrix :math:`M_{(k-1)n + i,(l-1)n+j}= \langle d_k\phi(x_i),d_l\phi(x_j) \rangle` (matrix :math:`M` in :footcite:t:`kostic2024learning`)
        where :math:`i = 1,\dots n, k=1, \dots d, j=1,\dots n, l=1, \dots d`
        and :math:`N` is the number of training points and :math:`d` is the dimensionality of the system.

        Args:
            kernel_X (np.ndarray): kernel matrix of the training data
            X (np.ndarray): training data
            friction (np.ndarray): friction parameter of the physical model :math:`s(x)` in :footcite:t:`kostic2024learning`
        Shape:
            ``kernel_X``: :math:`(N, N)`, where :math:`N` is the number of training data.
            ``X``: :math:`(N,d)`  where :math:`N` is the number of training data and `d` the dimensionality of the system.
            ``friction``: :math:`d`  where :math:`d` is the dimensionality of the system
        Returns: :math:`M_{(k-1)n + i,(l-1)n+j}= \langle d_k\phi(x_i),d_l\phi(x_j) \rangle` of shape `Nd,Nd`, where :math:`N` is the number of training data and :math:`d` the dimension of the system.
        """  # noqa: D205
        difference = X[:, np.newaxis, :] - X[np.newaxis, :, :]

        d = difference.shape[2]
        n = difference.shape[0]
        M = np.zeros((n * d, n * d))
        sigma = self.length_scale
        for i in range(n):
            for j in range(n):
                for k in range(0, d):
                    for m in range(0, d):
                        if m == k:
                            M[(k) * n + i, (m) * n + j] = (
                                friction[k]
                                * (1 / sigma**2 - difference[i, j, k] ** 2 / sigma**4)
                                * kernel_X[i, j]
                            )
                        else:
                            M[(k) * n + i, (m) * n + j] = (
                                np.sqrt(friction[k])
                                * np.sqrt(friction[m])
                                * (-difference[i, j, k] * difference[i, j, m] / sigma**4)
                                * kernel_X[i, j]
                            )

        return M

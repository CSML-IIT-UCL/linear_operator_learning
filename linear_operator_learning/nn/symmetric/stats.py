"""Statistics utilities for symmetric random variables with known group representations."""

from __future__ import annotations

import numpy as np
import torch
from escnn.group import Representation
from symm_torch.utils.rep_theory import isotypic_decomp_rep
from torch import Tensor

from linear_operator_learning.nn.symmetric.linalg import invariant_orthogonal_projector


def var_mean(x: Tensor, rep_X: Representation):
    """Compute the mean and variance of a symmetric random variable.

    Args:
        x: (Tensor) of shape :math:`(N, Dx)` containing the observations of the symmetric random variable
        rep_X: (escnn.group.Representation) representation of the symmetric random variable.

    Returns:
        (Tensor, Tensor): Mean and variance of the symmetric random variable. The mean isrestricted to be in the trivial/G-invariant subspace of the symmetric vector space. The variance is constrained to be the same for all dimensions of each G-irreducible subspace (i.e., each subspace associated with an irrep).

    Shape:
        :code:`x`: :math:`(N, Dx)` where N is the number of samples and Dx is the dimension of the symmetric random variable.

        Output: :math:`(Dx, Dx)`
    """
    assert len(x.shape) == 2, f"Expected x to have shape (N, n_features), got {x.shape}"
    # Compute the mean of the observation.
    mean_empirical = torch.mean(x, dim=0)
    # Project to the inv-subspace and map back to the original basis
    P_inv = invariant_orthogonal_projector(rep_X)
    mean = torch.einsum("ij,...j->...i", P_inv, mean_empirical)

    # Symmetry constrained variance computation.
    Cx = covariance(x, x, rep_X, rep_X)
    var = torch.diag(Cx)
    return var, mean


def isotypic_covariance(
    x: Tensor, y: Tensor, rep_X: Representation, rep_Y: Representation, center=True
):
    r"""Cross covariance of signals between isotypic subspaces of the same type.

    This function exploits the fact that the covariance of signals between isotypic subspaces of the same type
    is constrained to be of the block form:

    .. math::
        \mathbf{C}_{xy} = \text{Cov}(X, Y) = \mathbf{Z}_{xy} \otimes \mathbf{I}_d,

    where :math:`d = \text{dim(irrep)}` and :math:`\mathbf{Z}_{xy} \in \mathbb{R}^{m_x \times m_y}` and :math:`\mathbf{C}_{xy} \in \mathbb{R}^{(m_x \cdot d) \times (m_y \cdot d)}`.

    Being :math:`m_x` and :math:`m_y` the multiplicities of the irrep in X and Y respectively. This implies that the matrix :math:`\mathbf{Z}_{xy}`
    represents the free parameters of the covariance we are required to estimate. To do so we reshape
    the signals :math:`X \in \mathbb{R}^{N \times (m_x \cdot d)}` and :math:`Y \in \mathbb{R}^{N \times (m_y \cdot d)}` to :math:`X_{\text{sing}} \in \mathbb{R}^{(d \cdot N) \times m_x}` and :math:`Y_{\text{sing}} \in \mathbb{R}^{(d \cdot N) \times m_y}`
    respectively. Ensuring all dimensions of the irreducible subspaces associated to each multiplicity of the irrep are
    considered as a single dimension for estimating :math:`\mathbf{Z}_{xy} = \frac{1}{n \cdot d} X_{\text{sing}}^T Y_{\text{sing}}`.

    Args:
        x (Tensor): Realizations of the random variable X.
        y (Tensor): Realizations of the random variable Y.
        rep_X (escnn.nn.Representation): composed of :math:`m_x` copies of a single irrep: :math:`\rho_X = \otimes_i^{m_x} \rho_k`
        rep_Y (escnn.nn.Representation): composed of :math:`m_y` copies of a single irrep: :math:`\rho_Y = \otimes_i^{m_y} \rho_k`
        center (bool): whether to center the signals before computing the covariance.

    Returns:
        (Tensor, Tensor): :math:`\mathbf{C}_{xy}`, (:math:`m_y \cdot d, m_x \cdot d`) the covariance matrix between the isotypic subspaces of :code:`x` and :code:`y`, and :math:`\mathbf{Z}_{xy}`, (:math:`m_y, m_x`) the free parameters of the covariance matrix in the isotypic basis.

    Shape:
        :code:`x`: :math:`(..., N, m_x * d)` where N is the number of samples, :math:`d` is the dimension of the only irrep in :math:`rep_X` and :math:`m_x` is the multiplicity of the irrep in X.

        :code:`y`: :math:`(..., N, m_y * d)` where N is the number of samples, :math:`d` is the dimension of the only irrep in :math:`rep_Y` and :math:`m_y` is the multiplicity of the irrep in Y.

        Output: :math:`(m_y * d, m_x * d)`.
    """
    assert len(rep_X._irreps_multiplicities) == len(rep_Y._irreps_multiplicities) == 1, (
        f"Expected group representation of an isotypic subspace.I.e., with only one type of irrep. \nFound: "
        f"{list(rep_X._irreps_multiplicities.keys())} in rep_X, {list(rep_Y._irreps_multiplicities.keys())} in rep_Y."
    )
    assert rep_X.group == rep_Y.group, f"{rep_X.group} != {rep_Y.group}"
    irrep_id = rep_X.irreps[0]  # Irrep id of the isotypic subspace
    assert irrep_id == rep_Y.irreps[0], (
        f"Irreps {irrep_id} != {rep_Y.irreps[0]}. Hence signals are orthogonal and Cxy=0."
    )
    assert rep_X.size == x.shape[-1], (
        f"Expected signal shape to be (..., {rep_X.size}) got {x.shape}"
    )
    assert rep_Y.size == y.shape[-1], (
        f"Expected signal shape to be (..., {rep_Y.size}) got {y.shape}"
    )

    # Get information about the irreducible representation present in the isotypic subspace
    irrep_dim = rep_X.group.irrep(*irrep_id).size
    mk_X = rep_X._irreps_multiplicities[irrep_id]  # Multiplicity of the irrep in X
    mk_Y = rep_Y._irreps_multiplicities[irrep_id]  # Multiplicity of the irrep in Y

    # If required we must change bases to the isotypic bases.
    Qx_T, Qx = rep_X.change_of_basis_inv, rep_X.change_of_basis
    Qy_T, Qy = rep_Y.change_of_basis_inv, rep_Y.change_of_basis
    x_in_iso_basis = np.allclose(Qx_T, np.eye(Qx_T.shape[0]), atol=1e-6, rtol=1e-4)
    y_in_iso_basis = np.allclose(Qy_T, np.eye(Qy_T.shape[0]), atol=1e-6, rtol=1e-4)
    if x_in_iso_basis:
        x_iso = x
    else:
        Qx_T = Tensor(Qx_T).to(device=x.device, dtype=x.dtype)
        Qx = Tensor(Qx).to(device=x.device, dtype=x.dtype)
        x_iso = torch.einsum("...ij,...j->...i", Qx_T, x)  # x_iso = Q_x2iso @ x
    if np.allclose(Qy_T, np.eye(Qy_T.shape[0]), atol=1e-6, rtol=1e-4):
        y_iso = y
    else:
        Qy_T = Tensor(Qy_T).to(device=y.device, dtype=y.dtype)
        Qy = Tensor(Qy).to(device=y.device, dtype=y.dtype)
        y_iso = torch.einsum("...ij,...j->...i", Qy_T, y)  # y_iso = Q_y2iso @ y

    if irrep_dim > 1:
        # Since Cxy = Dxy ⊗ I_d  , d = dim(irrep) and D_χy ∈ R^{mχ x my}
        # We compute the constrained covariance, by estimating the matrix D_χy
        # This requires reshape X_iso ∈ R^{n x p} to X_sing ∈ R^{nd x mχ} and Y_iso ∈ R^{n x q} to Y_sing ∈ R^{nd x my}
        # Ensuring all samples from dimensions of a single irrep are flattened into a row of X_sing and Y_sing
        x_sing = x_iso.view(-1, mk_X, irrep_dim).permute(0, 2, 1).reshape(-1, mk_X)
        y_sing = y_iso.view(-1, mk_Y, irrep_dim).permute(0, 2, 1).reshape(-1, mk_Y)
    else:  # For one dimensional (real) irreps, this defaults to the standard covariance
        x_sing, y_sing = x_iso, y_iso

    is_inv_subspace = irrep_id == rep_X.group.trivial_representation.id
    if center and is_inv_subspace:  # Non-trivial isotypic subspace are centered
        x_sing = x_sing - torch.mean(x_sing, dim=0, keepdim=True)
        y_sing = y_sing - torch.mean(y_sing, dim=0, keepdim=True)

    N = x_sing.shape[0]
    assert N == x.shape[0] * irrep_dim

    c = 1 if center and is_inv_subspace else 0
    Dxy = torch.einsum("...y,...x->yx", y_sing, x_sing) / (N - c)
    if irrep_dim > 1:  # Broadcast the estimates according to Cxy = Dxy ⊗ I_d.
        I_d = torch.eye(irrep_dim, device=Dxy.device, dtype=Dxy.dtype)
        Cxy_iso = torch.kron(Dxy, I_d)
    else:
        Cxy_iso = Dxy

    # Change back to original basis if needed _______________________
    if not x_in_iso_basis:
        Cxy = Qy @ Cxy_iso
    else:
        Cxy = Cxy_iso

    if not y_in_iso_basis:
        Cxy = Cxy @ Qx_T

    return Cxy, Dxy


def covariance(X: Tensor, Y: Tensor, rep_X: Representation, rep_Y: Representation):
    r"""Compute the covariance between two symmetric random variables.

    The covariance of r.v. can be computed from the covariance of the orthogonal projections of the r.v. to each isotypic subspace. Hence in the disentangled/isotypic basis the covariance can be computed in
    block-diagonal form:

    .. math::
        \begin{align}
            \mathbf{C}_{xy} &= \mathbf{Q}_y^T (\bigoplus_{k} \mathbf{C}_{xy}^{(k)} )\mathbf{Q}_x \\
            &= \mathbf{Q}_y^T (\bigoplus_{k} \mathbf{Z}_{xy}^{(k)}  \otimes \mathbf{I}_{d_k} )\mathbf{Q}_x \\
        \end{align}
    Where :math:`\mathbf{Q}_x^T` and :math:`\mathbf{Q}_y^T` are the change of basis matrices to the isotypic basis of X and Y respectively,
    :math:`\mathbf{C}_{xy}^{(k)}` is the covariance between the isotypic subspaces of type k, :math:`\mathbf{Z}_{xy}^{(k)}` is the free parameters of the covariance matrix in the isotypic basis,
    and :math:`d_k` is the dimension of the irrep associated with the isotypic subspace of type k.

    Args:
        X (Tensor): Realizations of a random variable x.
        Y (Tensor): Realizations of a random variable y.
        rep_X (Representation): The representation for which the orthogonal projection to the invariant subspace is computed.
        rep_Y (Representation): The representation for which the orthogonal projection to the invariant subspace is computed.

    Returns:
        Tensor: The covariance matrix between the two random variables, of shape :math:`(Dy, Dx)`.

    Shape:
        X: :math:`(N, Dx)` where :math:`Dx` is the dimension of the random variable X.
        
        Y: :math:`(N, Dy)` where :math:`Dy` is the dimension of the random variable Y.
        
        Output: :math:`(Dy, Dx)`
    """
    assert X.shape[0] == Y.shape[0], "Expected equal number of samples in X and Y"
    assert X.shape[1] == rep_X.size, f"Expected X shape (N, {rep_X.size}), got {X.shape}"
    assert Y.shape[1] == rep_Y.size, f"Expected Y shape (N, {rep_Y.size}), got {Y.shape}"
    assert X.shape[-1] == rep_X.size, f"Expected X shape (..., {rep_X.size}), got {X.shape}"
    assert Y.shape[-1] == rep_Y.size, f"Expected Y shape (..., {rep_Y.size}), got {Y.shape}"

    rep_X_iso = isotypic_decomp_rep(rep_X)
    rep_Y_iso = isotypic_decomp_rep(rep_Y)
    # Changes of basis from the Disentangled/Isotypic-basis of X, and Y to the original basis.
    Qx = torch.tensor(rep_X_iso.change_of_basis, device=X.device, dtype=X.dtype)
    Qy = torch.tensor(rep_Y_iso.change_of_basis, device=Y.device, dtype=Y.dtype)

    rep_X_iso_subspaces = rep_X_iso.attributes["isotypic_reps"]
    rep_Y_iso_subspaces = rep_Y_iso.attributes["isotypic_reps"]

    # Get the dimensions of the isotypic subspaces of the same type in the input/output representations.
    iso_idx_X, iso_idx_Y = {}, {}
    x_dim = 0
    for iso_id, rep_k in rep_X_iso_subspaces.items():
        iso_idx_X[iso_id] = slice(x_dim, x_dim + rep_k.size)
        x_dim += rep_k.size
    y_dim = 0
    for iso_id, rep_k in rep_Y_iso_subspaces.items():
        iso_idx_Y[iso_id] = slice(y_dim, y_dim + rep_k.size)
        y_dim += rep_k.size

    X_iso = torch.einsum("ij,...j->...i", Qx.T, X)
    Y_iso = torch.einsum("ij,...j->...i", Qy.T, Y)
    Cxy_iso = torch.zeros((rep_Y.size, rep_X.size), dtype=X.dtype, device=X.device)
    for iso_id in rep_Y_iso_subspaces.keys():
        if iso_id not in rep_X_iso_subspaces:
            continue  # No covariance between the isotypic subspaces of different types.
        X_k = X_iso[..., iso_idx_X[iso_id]]
        Y_k = Y_iso[..., iso_idx_Y[iso_id]]
        rep_X_k = rep_X_iso_subspaces[iso_id]
        rep_Y_k = rep_Y_iso_subspaces[iso_id]
        # Cxy_k = Dxy_k ⊗ I_d [my * d x mx * d]
        Cxy_k, _ = isotypic_covariance(X_k, Y_k, rep_X_k, rep_Y_k, center=True)
        Cxy_iso[iso_idx_Y[iso_id], iso_idx_X[iso_id]] = Cxy_k

    # Change to the original basis
    Cxy = Qy.T @ Cxy_iso @ Qx
    return Cxy


#  Tests to confirm the operation of the functions is correct _________________________________________
def test_isotypic_cross_cov():  # noqa: D103
    import escnn
    from escnn.group import IrreducibleRepresentation, change_basis, directsum

    # Icosahedral group has irreps of dimensions [1, ... 5]. Good test case.
    G = escnn.group.Icosahedral()

    for irrep in G.representations.values():
        if not isinstance(irrep, IrreducibleRepresentation):
            continue
        mx, my = 2, 3
        x_rep_iso = directsum([irrep] * mx)  # ρ_Χ
        y_rep_iso = directsum([irrep] * my)  # ρ_Y

        batch_size = 500
        #  Simulate symmetric random variables
        X_iso = torch.randn(batch_size, x_rep_iso.size)
        Y_iso = torch.randn(batch_size, y_rep_iso.size)

        Cxy_iso, Dxy = isotypic_covariance(X_iso, Y_iso, x_rep_iso, y_rep_iso)
        Cxy_iso = Cxy_iso.numpy()

        assert Cxy_iso.shape == (my * irrep.size, mx * irrep.size), (
            f"Expected Cxy_iso to have shape ({my * irrep.size}, {mx * irrep.size}), got {Cxy_iso.shape}"
        )

        # Test change of basis is handled appropriately, using random change of basis.
        Qx, _ = np.linalg.qr(np.random.randn(x_rep_iso.size, x_rep_iso.size))
        Qy, _ = np.linalg.qr(np.random.randn(y_rep_iso.size, y_rep_iso.size))
        x_rep = change_basis(x_rep_iso, Qx, name=f"{x_rep_iso.name}_p")  # ρ_Χ_p = Q_Χ ρ_Χ Q_Χ^T
        y_rep = change_basis(y_rep_iso, Qy, name=f"{y_rep_iso.name}_p")  # ρ_Y_p = Q_Y ρ_Y Q_Y^T
        # Random variables NOT in irrep-spectral basis.
        X = Tensor(np.einsum("...ij,...j->...i", Qx, X_iso.numpy()))  # X_p = Q_x X
        Y = Tensor(np.einsum("...ij,...j->...i", Qy, Y_iso.numpy()))  # Y_p = Q_y Y
        Cxy_p, Dxy = isotypic_covariance(X, Y, x_rep, y_rep)
        Cxy_p = Cxy_p.numpy()

        assert np.allclose(Cxy_p, Qy @ Cxy_iso @ Qx.T, atol=1e-6, rtol=1e-4), (
            f"Expected Cxy_p - Q_y Cxy_iso Q_x^T = 0. Got \n {Cxy_p - Qy @ Cxy_iso @ Qx.T}"
        )

        # Test that computing Cxy_iso is equivalent to computing standard cross covariance using data augmentation.
        GX_iso, GY_iso = [X_iso], [Y_iso]
        for g in G.elements[1:]:
            X_g = Tensor(np.einsum("...ij,...j->...i", x_rep(g), X_iso.numpy()))
            Y_g = Tensor(np.einsum("...ij,...j->...i", y_rep(g), Y_iso.numpy()))
            GX_iso.append(X_g)
            GY_iso.append(Y_g)
        GX_iso = torch.cat(GX_iso, dim=0)

        Cx_iso, _ = isotypic_covariance(x=GX_iso, y=GX_iso, rep_X=x_rep_iso, rep_Y=x_rep_iso)
        Cx_iso = Cx_iso.numpy()
        # Compute the covariance in standard way doing data augmentation.
        Cx_iso_orbit = (GX_iso.T @ GX_iso / (GX_iso.shape[0])).numpy()
        # Project each empirical Cov to the subspace of G-equivariant linear maps, and average across orbit
        Cx_iso_orbit = np.mean(
            [
                np.einsum("ij,jk,kl->il", x_rep_iso(g), Cx_iso_orbit, x_rep_iso(~g))
                for g in G.elements
            ],
            axis=0,
        )
        # Numerical error occurs for small sample sizes
        assert np.allclose(Cx_iso, Cx_iso_orbit, atol=1e-2, rtol=1e-2), (
            "isotypic_cross_cov is not equivalent to computing the covariance using data-augmentation"
        )


def test_cross_cov():  # noqa: D103
    import escnn
    from escnn.group import IrreducibleRepresentation, change_basis, directsum

    # Icosahedral group has irreps of dimensions [1, ... 5]. Good test case.
    G = escnn.group.Icosahedral()
    # G = escnn.group.CyclicGroup(3)
    mx, my = 1, 2
    x_rep = directsum([G.regular_representation] * mx)
    y_rep = directsum([G.regular_representation] * my)

    # G = escnn.group.CyclicGroup(3)

    x_rep = isotypic_decomp_rep(x_rep)
    y_rep = isotypic_decomp_rep(y_rep)
    Qx, Qy = x_rep.change_of_basis, y_rep.change_of_basis
    x_rep_iso = change_basis(x_rep, Qx.T, name=f"{x_rep.name}_iso")  # ρ_Χ_p = Q_Χ ρ_Χ Q_Χ^T
    y_rep_iso = change_basis(y_rep, Qy.T, name=f"{y_rep.name}_iso")  # ρ_Y_p = Q_Y ρ_Y Q_Y^T

    batch_size = 500
    # Isotypic basis computation
    X_iso = torch.randn(batch_size, x_rep.size)
    Y_iso = torch.randn(batch_size, y_rep.size)
    Cxy_iso = covariance(X_iso, Y_iso, x_rep_iso, y_rep_iso).cpu().numpy()

    # Regular basis computation
    Qx = torch.tensor(x_rep.change_of_basis, dtype=X_iso.dtype)
    Qy = torch.tensor(y_rep.change_of_basis, dtype=Y_iso.dtype)
    X = torch.einsum("ij,...j->...i", Qx, X_iso)
    Y = torch.einsum("ij,...j->...i", Qy, Y_iso)
    Cxy = covariance(X, Y, x_rep, y_rep).cpu().numpy()

    assert np.allclose(Cxy, Qy.T @ Cxy_iso @ Qx, atol=1e-6, rtol=1e-4), (
        f"Expected Cxy - Q_y.T Cxy_iso Q_x = 0. Got \n {Cxy - Qy.T @ Cxy_iso @ Qx}"
    )

    # Test that r.v with different irrep types have no covariance. ===========================================
    irrep_id1, irrep_id2 = list(G._irreps.keys())[:2]
    x_rep = directsum([G._irreps[irrep_id1]] * mx)
    y_rep = directsum([G._irreps[irrep_id2]] * my)
    X = torch.randn(batch_size, x_rep.size)
    Y = torch.randn(batch_size, y_rep.size)
    Cxy = covariance(X, Y, x_rep, y_rep).cpu().numpy()
    assert np.allclose(Cxy, 0), f"Expected Cxy = 0, got {Cxy}"


def test_symmetric_moments():  # noqa: D103
    import escnn
    from escnn.group import directsum

    def compute_moments_for_rep(rep: Representation, batch_size=500):
        rep = isotypic_decomp_rep(rep)
        x = torch.randn(batch_size, rep.size)
        var, mean = var_mean(x, rep)
        return x, mean, var

    # Test that G-invariant random variables should have equivalent mean and var as standard computation
    G = escnn.group.DihedralGroup(3)
    mx = 10
    rep_x = directsum([G.trivial_representation] * mx)
    x, mean, var = compute_moments_for_rep(rep_x)
    mean_gt = torch.mean(x, dim=0)
    var_gt = torch.var(x, dim=0)
    assert torch.allclose(mean, mean_gt, atol=1e-6, rtol=1e-4), f"Mean {mean} != {mean_gt}"
    assert torch.allclose(var, var_gt, atol=1e-4, rtol=1e-4), f"Var {var} != {var_gt}"

    # Test that the variance of a G-irreducible subspace is the same for all dimensions
    G = escnn.group.DihedralGroup(3)
    mx = 10
    irrep_2d = G._irreps[(1, 1)]
    rep_x = directsum([irrep_2d] * mx)  # 2D irrep * mx
    x, mean, var = compute_moments_for_rep(rep_x)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-6, rtol=1e-4), (
        f"Mean {mean} != 0 for non-trivial space"
    )
    assert len(torch.unique(var)) == mx, (
        f"Each of the {mx} irreducible subspaces should have the same variance {var}"
    )
    # Check computing the variance on a irreducible subspace is equivalent to the returned value for that space
    x1 = x[:, : irrep_2d.size]
    var1_gt = (x1**2).mean()
    assert torch.allclose(var1_gt, var[0], atol=1e-4, rtol=1e-4), f"Var {var[0]} != {var1_gt}"

    # ____________________________________________________________
    G = escnn.group.Icosahedral()
    mx = 1
    rep_x = directsum([G.regular_representation] * mx)
    x, mean, var = compute_moments_for_rep(rep_x)
    #  Check mean is in the invariant subspace.
    mean_gt = torch.einsum(
        "ij,...j->...i", invariant_orthogonal_projector(rep_x), torch.mean(x, dim=0)
    )
    assert torch.allclose(mean, mean_gt, atol=1e-6, rtol=1e-4), f"Mean {mean} != {mean_gt}"

    # Ensure the mean is equivalent to computing the mean of the orbit of the dataset under the group action
    Gx = []
    for g in G.elements:
        g_x = torch.einsum(
            "...ij,...j->...i", torch.tensor(rep_x(g), dtype=x.dtype, device=x.device), x
        )
        Gx.append(g_x)
    Gx = torch.cat(Gx, dim=0)
    mean_Gx = torch.mean(Gx, dim=0)
    assert torch.allclose(mean, mean_Gx, atol=1e-6, rtol=1e-4), f"Mean {mean} != {mean_Gx}"

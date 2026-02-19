import torch
import numpy as np



def l2(x, y):

    nDimx = x.ndim - 1
    nDimy = y.ndim - 1

    sq_x = (x**2).sum(axis=-1).view(*x.shape[:-1], *([1]*nDimy))
    sq_y = (y**2).sum(axis=-1).view(*([1]*nDimx), *y.shape[:-1])

    prod = torch.tensordot(x, y, dims=([-1], [-1]))

    return sq_x + sq_y - 2 * prod


def inv_sqrt_matrix(A):
    """ Returns A^{-1/2} for positive matrices """
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)

    alpha = 1e-7
    S_inv_sqrt = 1.0 / (torch.sqrt(S)+alpha)
    # Reconstruct the matrix: A^(-1/2) = U * diag(1/sqrt(S)) * U.T
    A_inv_sqrt = U @ torch.diag(S_inv_sqrt) @ Vh
    
    return A_inv_sqrt




def compute_mahalanobis_distance(x, y, VI=None, inv_sigma_matrix=None):
    if inv_sigma_matrix is None and VI is None:
        raise TypeError("Either 'VI' or 'inv_sqrt_matrix' must be provided.")

    if inv_sigma_matrix is None:
        inv_sigma_matrix = inv_sqrt_matrix(VI)

    x = torch.tensordot(x, inv_sigma_matrix, dims=([-1], [1]))
    y = torch.tensordot(y, inv_sigma_matrix, dims=([-1], [1]))

    return l2(x, y)


def compute_mahalanobis_distance_batched(craft_centroids, img_centroids, VIs):
    """Batched Mahalanobis distance for N CRAFT components × M image components.

    Instead of looping over each CRAFT component and calling
    ``compute_mahalanobis_distance`` one-at-a-time, this function batches the
    SVD decomposition and the distance computation into a handful of large
    tensor operations.

    Parameters
    ----------
    craft_centroids : Tensor, shape (N, D)
    img_centroids   : Tensor, shape (M, D)
    VIs             : Tensor, shape (N, D, D)
        Per-component inertia tensors.

    Returns
    -------
    distances : Tensor, shape (N, M)
        Squared Mahalanobis distances.
    """
    # Batched SVD: A_inv_sqrt[n] = U[n] @ diag(1/sqrt(S[n])) @ Vh[n]
    U, S, Vh = torch.linalg.svd(VIs, full_matrices=False)       # each (N,D,D)/(N,D)
    S_inv_sqrt = 1.0 / (torch.sqrt(S) + 1e-7)                   # (N, D)
    A_inv_sqrt = U * S_inv_sqrt.unsqueeze(-2) @ Vh               # (N, D, D)

    # Transform craft centroids: result[n] = A_inv_sqrt[n] @ craft_centroids[n]
    t_craft = torch.einsum('nij,nj->ni', A_inv_sqrt, craft_centroids)   # (N, D)

    # Transform image centroids with each CRAFT's matrix:
    # result[n, m] = A_inv_sqrt[n] @ img_centroids[m]
    t_img = torch.einsum('nij,mj->nmi', A_inv_sqrt, img_centroids)      # (N, M, D)

    # Squared L2 in transformed space
    diff = t_craft.unsqueeze(1) - t_img                                  # (N, M, D)
    return (diff * diff).sum(-1)                                          # (N, M)


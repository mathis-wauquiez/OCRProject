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


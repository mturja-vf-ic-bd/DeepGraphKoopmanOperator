import torch
import torch.nn as nn
from typing import List

from layers.mlp import MLP


def form_complex_conjugate_block(omegas, delta_t, k=1):
    """Form a 2x2 block for a complex conj. pair of eigenvalues, but for each example, so dimension [None, 2, 2]

    2x2 Block is
    exp(mu * delta_t) * [cos(omega * delta_t), -sin(omega * delta_t)
                         sin(omega * delta_t), cos(omega * delta_t)]

    Arguments:
        omegas -- array of parameters for blocks. first column is freq. (omega) and 2nd is scaling (mu), size [None, None, 2]
        delta_t -- time step in trajectories from input data

    Returns:
        stack of 2x2 blocks, size [None, 2, 2], where first dimension matches first dimension of omegas

    Side effects:
        None
    """
    scale = 1 - nn.ReLU()(-(torch.exp(omegas[:, :, 1] * delta_t * k) - 1))
    scale = nn.ReLU()(scale)
    # scale = torch.exp(omegas[:, :, 1] * delta_t * k)
    entry11 = torch.mul(scale, torch.cos(omegas[:, :, 0] * delta_t * k))
    entry12 = torch.mul(scale, torch.sin(omegas[:, :, 0] * delta_t * k))
    row1 = torch.stack([entry11, -entry12], dim=-1)  # [None, None, 2]
    row2 = torch.stack([entry12, entry11], dim=-1)  # [None, None, 2]
    return torch.stack([row1, row2], dim=-2)  # [None, None, 2, 2] put one row below other


def varying_multiply(y, U, omegas, delta_t, num_real, num_complex_pairs, k=1):
    """Multiply y-coordinates on the left by matrix L, but let matrix vary.

    Arguments:
        y -- array of shape [None, None, m] of y-coordinates
        U -- Koopman eigenvectors of shape [None, None, n**2/2, num_real + num_complex_pairs]
        omegas -- tensor containing the omegas of shape [None, None, num_real + num_complex_pairs]
        delta_t -- time step in trajectories from input data
        num_real -- number of real eigenvalues
        num_complex_pairs -- number of pairs of complex conjugate eigenvalues

    Returns:
        array same size as input y, but advanced to next time step

    Side effects:
        None
    """
    complex_list = []
    y = torch.matmul(
        U.transpose(-1, -2).unsqueeze(1),
        y.unsqueeze(-1)).squeeze(-1)

    # first, Jordan blocks for each pair of complex conjugate eigenvalues
    for j in range(num_complex_pairs):
        L_stack = form_complex_conjugate_block(omegas[:, :, 2 * j + num_real:2 * j + 2 + num_real], delta_t, k)
        complex_list.append(torch.einsum('bfij, bfj -> bfi', L_stack, y[:, :, 2 * j + num_real: 2 * j + 2 + num_real]))

    if len(complex_list):
        # each element in list output_list is shape [None, 2]
        complex_part = torch.cat(complex_list, dim=-1)

    # next, diagonal structure for each real eigenvalue
    # faster to not explicitly create stack of diagonal matrices L
    real_list = []
    for j in range(num_real):
        real_list.append(
            torch.mul(y[:, :, j],
                      torch.exp(omegas[:, :, j] * delta_t * k)))

    if len(real_list):
        real_part = torch.stack(real_list, dim=-1)

    if len(complex_list) and len(real_list):
        y_next = torch.cat([real_part, complex_part], dim=-1)
    elif len(complex_list):
        y_next = complex_part
    else:
        y_next = real_part
    y_next = torch.einsum('bij,bfj->bfi', U, y_next)
    return y_next

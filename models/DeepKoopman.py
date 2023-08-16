import unittest
from typing import List
import torch
import torch.nn as nn

from layers.mlp import MLP


def pearson(A):
    eps = 1e-5
    m = A.mean(dim=-1, keepdim=True)
    s = A.std(dim=-1, keepdim=True)
    N = A.size(-1)
    A = A - m
    cov = (A @ A.transpose(-1, -2)) / (N - 1)
    corr = cov / (s @ s.transpose(-1, -2) + eps)
    return corr - torch.diag_embed(torch.diagonal(corr, dim1=-2, dim2=-1))


class EdgeEmbeddingNN(nn.Module):
    def __init__(self,
                 encoder_width_list: List[int],
                 decoder_width_list: List[int],
                 edge_func_width_list: List[int],
                 activation):
        super(EdgeEmbeddingNN, self).__init__()
        assert encoder_width_list[-1] == decoder_width_list[0], "Mismatched latent dimesnion"
        self.node_func = MLP(
            encoder_width_list,
            prefix="node_func",
            activation=activation
        )
        self.node_func_inv = MLP(
            decoder_width_list,
            prefix="node_func_inv",
            activation=activation
        )
        self.edge_func = MLP(
            edge_func_width_list,
            prefix="edge_func",
            activation=activation
        )

    def forward(self, x):
        """

        :param x: tensor of size B * N * T
                  where B is the batch size,
                  N is the number of nodes, and,
                  T is the number of time-stamps.
        :return: y: latent embedding of node features
                 g_e: edge measurement functions
                 recon_loss: reconstruction loss of the autoencoder
        """

        y = self.node_func(x)  # node embedding of size B * N * M
        g_e = pearson(y)
        triu_index = torch.triu_indices(g_e.shape[-2], g_e.shape[-1], offset=1)
        if len(g_e.size()) == 3:
            g_e = g_e[:, triu_index[0], triu_index[1]]
        elif len(g_e.size()) == 4:
            g_e = g_e[:, :, triu_index[0], triu_index[1]]
        g_e = self.edge_func(g_e.unsqueeze(-1)).squeeze(-1)
        x_recon = self.node_func_inv(y)
        return y, g_e, x_recon


class EigenFunctionNN(nn.Module):
    def __init__(self, n_modes,
                 encoder_width_list, decoder_width_list,
                 edge_func_width_list, activation):
        super(EigenFunctionNN, self).__init__()
        self.koopman_eigen_modes = nn.ModuleList()
        for i in range(n_modes):
            eigen_module = EdgeEmbeddingNN(
                encoder_width_list, decoder_width_list,
                edge_func_width_list, activation
            )
            self.koopman_eigen_modes.append(eigen_module)
        self.n_modes = n_modes

    def forward(self, x):
        U = []
        for i in range(self.n_modes):
            _, g_e, x_recon = self.koopman_eigen_modes[i](x)
            U.append(g_e)
        U = torch.stack(U, dim=-1)
        return U


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
    scale = torch.exp(torch.ones_like(omegas[:, :, 1]) * delta_t * k)
    entry11 = torch.mul(scale, torch.cos(omegas[:, :, 0] * delta_t * k))
    entry12 = torch.mul(scale, torch.sin(omegas[:, :, 0] * delta_t * k))
    row1 = torch.stack([entry11, -entry12], dim=-1)  # [None, None, 2]
    row2 = torch.stack([entry12, entry11], dim=-1)  # [None, None, 2]
    return torch.stack([row1, row2], dim=-2)  # [None, None, 2, 2] put one row below other


def varying_multiply(y, U, omegas, delta_t, num_real, num_complex_pairs, k=1):
    """Multiply y-coordinates on the left by matrix L, but let matrix vary.

    Arguments:
        y -- array of shape [None, None, m] of y-coordinates
        U -- Koopman eigenvectors of shape [None, None, n**2/2, num_modes]
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
    y = torch.matmul(U.transpose(-1, -2).unsqueeze(1), y.unsqueeze(-1)).squeeze(-1)

    # first, Jordan blocks for each pair of complex conjugate eigenvalues
    for j in range(num_complex_pairs):
        L_stack = form_complex_conjugate_block(omegas[:, :, 2 * j:2 * j + 2], delta_t, k)
        complex_list.append(torch.einsum('bfij, bfj -> bfi', L_stack, y[:, :, 2 * j: 2 * j + 2]))

    if len(complex_list):
        # each element in list output_list is shape [None, 2]
        complex_part = torch.cat(complex_list, dim=-1)

    # next, diagonal structure for each real eigenvalue
    # faster to not explicitly create stack of diagonal matrices L
    real_list = []
    for j in range(num_real):
        real_list.append(
            torch.mul(y[:, :, 2 * num_complex_pairs + j],
                      torch.exp(omegas[:, :, 2 * num_complex_pairs + j] * delta_t * k)))

    if len(real_list):
        real_part = torch.stack(real_list, dim=-1)

    if len(complex_list) and len(real_list):
        y_next = torch.cat([complex_part, real_part], dim=-1)
    elif len(complex_list):
        y_next = complex_part
    else:
        y_next = real_part
    y_next = torch.einsum('bij,bfj->bfi', U, y_next)
    return y_next


class KoopmanAutoencoder(nn.Module):
    def __init__(
            self,
            num_nodes,
            num_real_modes,
            num_complex_modes,
            delta_t,
            encoder_width_list: List[int],
            decoder_width_list: List[int],
            edge_proj_width_list: List[int],
            edge_func_enc_width: List[int],
            edge_func_dec_width: List[int],
            edge_func_proj_width: List[int],
            activation: nn.Module
    ):
        super(KoopmanAutoencoder, self).__init__()
        self.num_nodes = num_nodes
        self.num_real_modes = num_real_modes
        self.num_complex_modes = num_complex_modes
        self.sw_size = encoder_width_list[0]
        self.latent_dim = encoder_width_list[-1]
        self.delta_t = delta_t
        self.eigen_func_nn = EigenFunctionNN(
            num_real_modes + 2 * num_complex_modes,
            edge_func_enc_width,
            edge_func_dec_width,
            edge_func_proj_width,
            activation
        )
        self.edge_embedder = EdgeEmbeddingNN(
            encoder_width_list,
            decoder_width_list,
            edge_proj_width_list,
            activation
        )
        self.eig_val = MLP(
            width_list=[
                self.latent_dim * self.num_nodes,
                128, 128, 128,
                num_real_modes + 2 * num_complex_modes],
            prefix="eig_val"
        )

    def forward(self, x, max_k=1):
        U = self.eigen_func_nn(x)
        x = x.unfold(-1, self.sw_size, self.sw_size).permute(0, 2, 1, 3)
        y, g, x_recon = self.edge_embedder(x)
        x_recon = x_recon.permute(0, 2, 1, 3).flatten(start_dim=2)
        omega = self.eig_val(y.flatten(start_dim=2))
        g_next_list = []
        for k in range(1, max_k+1):
            g_next = varying_multiply(g, U, omega, self.delta_t,
                                      self.num_real_modes,
                                      self.num_complex_modes, k)
            g_next_list.append(g_next)
        return y, g, g_next_list, U, omega, x_recon


class testEdgeKoopmanNN(unittest.TestCase):
    def testOutputShapes3D(self):
        x = torch.rand((10, 50, 32))
        model = EdgeEmbeddingNN(
            encoder_width_list=[32, 64, 64, 8],
            decoder_width_list=[8, 64, 64, 32],
            edge_func_width_list=[1, 16, 16, 1],
            activation=nn.ReLU()
        )
        y, g_e, x_recon = model(x)
        self.assertEqual(y.shape, (10, 50, 8), "Mismatch shape in latent node embedding")
        self.assertEqual(g_e.shape, (10, 1225), "Mismatch shape in edge embedding")
        self.assertEqual(x.shape, x_recon.shape, "Mismatch shape in x_recon")

    def testOutputShapes4D(self):
        x = torch.rand((10, 3, 50, 32))
        model = EdgeEmbeddingNN(
            encoder_width_list=[32, 64, 64, 8],
            decoder_width_list=[8, 64, 64, 32],
            edge_func_width_list=[1, 16, 16, 1],
            activation=nn.ReLU()
        )
        y, g_e, x_recon = model(x)
        self.assertEqual(y.shape, (10, 3, 50, 8), "Mismatch shape in latent node embedding")
        self.assertEqual(g_e.shape, (10, 3, 1225), "Mismatch shape in edge embedding")
        self.assertEqual(x.shape, x_recon.shape, "Mismatch shape in x_recon")


class testDeepGraphKoopman(unittest.TestCase):
    def testOutputShape(self):
        x = torch.rand((10, 50, 32))
        model = EigenFunctionNN(3,
                                encoder_width_list=[32, 64, 64, 8],
                                decoder_width_list=[8, 64, 64, 32],
                                edge_func_width_list=[1, 16, 16, 1],
                                activation=nn.ReLU())
        U = model(x)
        self.assertEqual(U.shape, (10, 1225, 3))


class testVaryingMultiply(unittest.TestCase):
    def testOutputShape(self):
        U = torch.rand((10, 1225, 7))
        g = torch.rand((10, 5, 1225))
        omegas = torch.rand((10, 5, 7))
        g_next = varying_multiply(g, U, omegas, 0.72, 3, 2)
        self.assertEqual(g.shape, g_next.shape)


class testKoopmanAutoencoder(unittest.TestCase):
    def testOutputShape(self):
        model = KoopmanAutoencoder(
            50, 3, 2, 0.72,
            encoder_width_list=[4, 64, 64, 8],
            decoder_width_list=[8, 64, 64, 4],
            edge_proj_width_list=[1, 16, 16, 1],
            edge_func_enc_width=[32, 64, 64, 8],
            edge_func_dec_width=[8, 64, 64, 32],
            edge_func_proj_width=[1, 16, 16, 1],
            activation=nn.ReLU()
        )
        x = torch.rand((10, 50, 32))
        y, g, g_next_list, U, omega, x_recon = model(x, 5)
        for g_next in g_next_list:
            self.assertEqual(g.shape, g_next.shape)
        self.assertEqual(x_recon.shape, x.shape)
        self.assertEqual(y.shape, (10, 8, 50, 8))

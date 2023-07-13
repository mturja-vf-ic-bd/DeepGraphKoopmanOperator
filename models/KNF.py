import unittest

import torch
import torch.nn as nn

from layers.mlp import MLP
from models.DeepKoopman import form_complex_conjugate_block
from utils.correlation import pearson


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
    y = torch.matmul(U.transpose(-1, -2).unsqueeze(1), y.unsqueeze(-1)).squeeze(-1)

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
        y_next = torch.cat([complex_part, real_part], dim=-1)
    elif len(complex_list):
        y_next = complex_part
    else:
        y_next = real_part
    y_next = torch.einsum('bij,bfj->bfi', U, y_next)
    return y_next


class KNF_Embedder(nn.Module):
    def __init__(
            self,
            input_dim,
            input_length,
            u_window_size,
            num_steps,
            encoder_hidden_dim,
            decoder_hidden_dim,
            encoder_num_layers,
            decoder_num_layers,
            n_real_dmd_modes,
            n_complex_dmd_mode_pairs,
            transformer_dim,
            transformer_num_layers,
            num_heads,
            latent_dim,
            num_sins=-1,
            num_poly=-1,
            num_exp=-1,
            num_feats=1
    ):
        super(KNF_Embedder, self).__init__()
        self.input_dim = input_dim
        self.input_length = input_length
        self.u_window_size = u_window_size
        self.num_steps = num_steps
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.encoder_num_layers = encoder_num_layers
        self.decoder_num_layers = decoder_num_layers
        self.n_real_dmd_modes = n_real_dmd_modes
        self.n_complex_dmd_mode_pairs = n_complex_dmd_mode_pairs
        self.latent_dim = latent_dim
        self.num_feats = num_feats
        self.num_poly = num_poly
        self.num_sins = num_sins
        self.num_exp = num_exp
        self.encoder = MLP(
            width_list=[input_dim * num_feats] +
                       [self.encoder_hidden_dim] * self.encoder_num_layers +
                       [latent_dim * input_dim * num_feats],
            prefix="encoder"
        )
        self.decoder = MLP(
            width_list=[latent_dim * num_feats] +
                       [self.decoder_hidden_dim] * self.decoder_num_layers +
                       [input_dim * num_feats],
            prefix="decoder"
        )
        self.U = nn.Linear(u_window_size // input_dim,
                           n_real_dmd_modes + n_complex_dmd_mode_pairs)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=(latent_dim - 2 * num_sins) * num_feats,
            nhead=num_heads,
            dim_feedforward=transformer_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=transformer_num_layers)
        self.omegas = MLP(
            width_list=[(latent_dim - 2 * num_sins) * num_feats,
                        128, 128, n_real_dmd_modes + 2 * n_complex_dmd_mode_pairs],
            prefix="omega_net")

    def generate_embeddings(self, x):
        # Create time chunks and flatten
        inps = x.unfold(-1, self.input_dim, self.input_dim).transpose(-2, -3)
        X = inps.flatten(start_dim=2)

        # Generate weights for linear combination
        coefs = self.encoder(X).reshape(X.shape[0], X.shape[1], self.latent_dim, -1)
        coefs = coefs.reshape(X.shape[0], X.shape[1], self.latent_dim, self.num_feats, -1)

        # Generate raw embeddings for edges (learned linear combinations and then pearson corr)
        V = torch.einsum("blkfd, blfd -> blkf", coefs, inps)

        # Apply predefined functions on V (such as exponential, polynomial, sin/cos etc.)
        ################ Calculate Meausurements ############
        embedding = torch.zeros(V.shape[0], V.shape[1],
                                self.latent_dim - 2 * self.num_sins,
                                self.num_feats).to(inps.device)

        # print("Embedding shape: ", embedding.shape)
        for f in range(self.num_feats):
            # polynomials
            for i in range(self.num_poly):
                embedding[:, :, i, f] = V[:, :, i, f] ** (i + 1)

            # exponential function
            for i in range(self.num_poly, self.num_poly + self.num_exp):
                embedding[:, :, i, f] = torch.exp(V[:, :, i, f])

            # sine/cos functions
            for i in range(self.num_poly + self.num_exp,
                           self.num_poly + self.num_exp + self.num_sins):
                embedding[:, :, i, f] = V[:, :, self.num_sins * 2 + i, f] * torch.cos(
                    V[:, :, i, f])
                embedding[:, :, self.num_sins +
                                i, f] = V[:, :, self.num_sins * 3 + i, f] * torch.sin(
                    V[:, :, self.num_sins + i, f])

            # the remaining ouputs are purely data-driven measurement functions.
            embedding[:, :, self.num_poly + self.num_exp +
                            self.num_sins * 2:, f] = V[:, :, self.num_poly +
                                                             self.num_exp + self.num_sins * 4:, f]
        return embedding, X, V

    def forward(self, x, max_k=1):
        """
        The forward function breaks the time dimension into smaller chunks of size `self.input_dim`.
        It then generates the embedding as a linear combination of the observations within each chunk.
        The weights of the linear combinations are learned via an MLP. There are multiple set of such
        weight parameters --- one for each measurement function.

        :param x: Input tensor of size (batch, feat_dim, ts)
        :return:
        """

        x_cur = x[:, :, :self.u_window_size]
        x_future = x[:, :, self.u_window_size:]
        embedding_cur, X_cur, V_cur = self.generate_embeddings(x_cur)
        embedding_future, X_future, V_future = self.generate_embeddings(x_future)

        # Compute eigenvalues
        omega_cur = self.omegas(embedding_cur.flatten(start_dim=2))
        omega_future = self.omegas(embedding_future.flatten(start_dim=2))

        # Compute common DMD modes for the x_cur
        dmd_modes = self.transformer_encoder(embedding_cur.flatten(start_dim=2))
        dmd_modes = self.U(dmd_modes.transpose(1, 2))
        dmd_modes = torch.cat([
            dmd_modes[:, :, :self.n_real_dmd_modes],
            dmd_modes[:, :, self.n_real_dmd_modes:],
            dmd_modes[:, :, self.n_real_dmd_modes:]],
            dim=-1
        )

        # shift latent embeddings to the right
        pred_list_cur = []
        pred_list_next = []
        for k in range(1, max_k + 1):
            pred_cur = varying_multiply(
                embedding_cur.flatten(start_dim=2),
                U=dmd_modes,
                omegas=omega_cur,
                num_real=self.n_real_dmd_modes,
                num_complex_pairs=self.n_complex_dmd_mode_pairs,
                delta_t=0.72, k=k
            )
            pred_list_cur.append(pred_cur)
            pred_future = varying_multiply(
                embedding_future.flatten(start_dim=2),
                U=dmd_modes,
                omegas=omega_future,
                num_real=self.n_real_dmd_modes,
                num_complex_pairs=self.n_complex_dmd_mode_pairs,
                delta_t=0.72, k=k
            )
            pred_list_next.append(pred_future)

        # Reconstruct V back to original space
        X_recon_cur = self.decoder(V_cur.flatten(start_dim=2))
        X_recon_future = self.decoder(V_future.flatten(start_dim=2))

        return V_cur, V_future, X_cur, X_future, \
               X_recon_cur, X_recon_future, embedding_cur, \
               dmd_modes, omega_cur, omega_future, \
               pred_list_cur, pred_list_next


def compute_prediction_loss(x_orig, x_pred_list):
    loss = 0
    for i, x_pred in enumerate(x_pred_list):
        loss += nn.MSELoss(reduction='mean')(
            x_orig[:, i+1:], x_pred[:, :-(i+1)])
    return loss / len(x_pred_list)


class testKNF_Embedder(unittest.TestCase):
    def testShape(self):
        input_dim = 16
        ts = 1200
        u_window = 800
        batch = 32
        n_nodes = 50
        latent_dim = 24
        num_sins = 2
        model = KNF_Embedder(input_dim, ts, u_window,
                             5, 256, 256, 3, 3,
                             transformer_dim=128,
                             transformer_num_layers=3,
                             num_heads=4, latent_dim=latent_dim,
                             n_real_dmd_modes=3, n_complex_dmd_mode_pairs=4,
                             num_sins=num_sins, num_poly=5, num_exp=2,
                             num_feats=50)
        x = torch.FloatTensor(torch.randn(batch, n_nodes, ts))
        V_cur, V_future, X_cur, X_future, \
            X_recon_cur, X_recon_future, embeddings, emb_trans, \
                omega_cur, omega_future, \
                    pred_list_cur, pred_list_next = model(x, max_k=3)
        self.assertEqual(V_cur.shape, (batch, u_window // input_dim, 24, n_nodes))
        self.assertEqual(X_cur.shape, X_recon_cur.shape)
        self.assertEqual(V_future.shape, (batch, (ts - u_window) // input_dim, 24, n_nodes))
        self.assertEqual(X_recon_future.shape, X_future.shape)
        self.assertEqual(embeddings.shape, (batch, u_window // input_dim, 20, n_nodes))
        self.assertEqual(emb_trans.shape, (batch, (latent_dim - 2 * num_sins) * n_nodes, 11))
        self.assertEqual(omega_cur.shape, (batch, u_window // input_dim, 11))
        self.assertEqual(omega_future.shape, (batch, (ts - u_window) // input_dim, 11))

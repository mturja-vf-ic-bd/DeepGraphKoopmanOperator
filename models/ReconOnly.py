import unittest

import torch
import torch.nn as nn

from layers.mlp import MLP



class KNF_Embedder_Recon(nn.Module):
    def __init__(
            self,
            input_dim,
            input_length,
            encoder_hidden_dim,
            decoder_hidden_dim,
            encoder_num_layers,
            decoder_num_layers,
            latent_dim,
            num_sins=2,
            num_poly=2,
            num_exp=0,
            num_feats=1
    ):
        super(KNF_Embedder_Recon, self).__init__()
        self.input_dim = input_dim
        self.input_length = input_length
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.encoder_num_layers = encoder_num_layers
        self.decoder_num_layers = decoder_num_layers
        self.latent_dim = latent_dim
        self.num_sins = num_sins
        self.num_poly = num_poly
        self.num_exp = num_exp
        self.num_feats = num_feats
        self.encoder = MLP(
            width_list=[input_dim * num_feats] +
                       [self.encoder_hidden_dim] * self.encoder_num_layers +
                       [latent_dim * input_dim * num_feats],
            prefix="encoder",
            norm="instance"
        )
        self.decoder = MLP(
            width_list=[(self.latent_dim - 2 * num_sins) * num_feats] +
                       [self.decoder_hidden_dim] * self.decoder_num_layers +
                       [input_dim * num_feats],
            prefix="decoder",
            norm="instance"
        )

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

    def forward(self, x):
        """
        The forward function breaks the time dimension into smaller chunks of size `self.input_dim`.
        It then generates the embedding as a linear combination of the observations within each chunk.
        The weights of the linear combinations are learned via an MLP. There are multiple set of such
        weight parameters --- one for each measurement function.

        :param x: Input tensor of size (batch, feat_dim, ts)
        :return:
        """

        embedding_cur, X_cur, V_cur = self.generate_embeddings(x)
        X_recon_cur = self.decoder(embedding_cur.flatten(start_dim=2))

        return V_cur, X_cur, \
               X_recon_cur, embedding_cur


class testKNF_Embedder(unittest.TestCase):
    def testShape(self):
        input_dim = 16
        ts = 1200
        u_window = 800
        batch = 32
        n_nodes = 50
        latent_dim = 24
        model = KNF_Embedder_Recon(input_dim, ts,
                                   256, 256, 3, 3, latent_dim=latent_dim,
                                   num_sins=2, num_poly=4, num_exp=2,
                                   num_feats=50)
        x = torch.FloatTensor(torch.randn(batch, n_nodes, ts))
        V_cur, X_cur, \
        X_recon_cur, embedding_cur = model(x, max_k=3)
        self.assertEqual(V_cur.shape, (batch, 75, 24, n_nodes))
        self.assertEqual(X_cur.shape, X_recon_cur.shape)
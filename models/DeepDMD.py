# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pytorch implementation of Koopman Neural Operator."""
import itertools
import math
import unittest

from layers.normalizer import RevIN
from layers.positional_encoder import PositionalEncoding3D
import numpy as np
import torch
from torch import nn
import time

from layers.mlp import MLP
from models.model_utils import varying_multiply

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DeepDMD(nn.Module):
    """Koopman Neural Forecaster.

    Attributes:
      input_dim: number of steps of historical observations encoded at every step
      input_length: input length of ts
      output_dim: number of output features
      num_steps: number of prediction steps every forward pass
      encoder_hidden_dim: hidden dimension of encoder
      decoder_hidden_dim: hidden dimension of decoder
      encoder_num_layers: number of layers in the encoder
      decoder_num_layers: number of layers in the decoder
      latent_dim: dimension of finite koopman space
      num_feats=1: number of features
      add_global_operator: whether to use a global operator
      add_control: whether to use a feedback module
      control_num_layers: number of layers in the control module
      control_hidden_dim: hidden dim in the control module
      use_RevIN: whether to use reversible normalization
      use_instancenorm: whether to use instance normalization on hidden states
      real_modes: number of real modes in local Koopman operator
      complex_modes: number of complex modes in local Koopman operator
      num_sins: number of pairs of sine and cosine measurement functions
      num_poly: the highest order of polynomial functions
      num_exp: number of exponential functions
      num_heads: Number of the head the transformer encoder
      transformer_dim: hidden dimension of tranformer encoder
      transformer_num_layers: number of layers in the transformer encoder
      omega_dim: hidden dimension of omega-net
      omega_num_layers: number of layers in omega-net
      dropout_rate: dropout rate of MLP modules
    """

    def __init__(self,
                 input_dim,
                 input_length,
                 output_dim,
                 num_steps,
                 encoder_hidden_dim,
                 decoder_hidden_dim,
                 encoder_num_layers,
                 decoder_num_layers,
                 latent_dim,
                 rank,
                 num_feats=1,
                 use_revin=True,
                 use_instancenorm=False,
                 add_global_operator=False,
                 num_sins=-1,
                 num_poly=-1,
                 num_exp=-1,
                 num_heads=1,
                 transformer_dim=128,
                 transformer_num_layers=2,
                 dropout_rate=0,
                 stride=1):

        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_length = input_length
        self.num_steps = num_steps
        self.latent_dim = latent_dim
        self.use_revin = use_revin
        self.use_instancenorm = use_instancenorm
        self.num_feats = num_feats
        self.stride = stride
        self.rank = rank
        self.add_global_operator = add_global_operator

        # num_poly/num_sins/num_exp = -1 means using default values
        if num_poly == -1:
            self.num_poly = 3
        else:
            self.num_poly = num_poly

        if num_sins == -1:
            self.num_sins = input_length // 2 - 1
        else:
            self.num_sins = num_sins

        if num_exp == -1:
            self.num_exp = 1
        else:
            self.num_exp = num_exp

        # we also use interation terms for multivariate time series
        # calculate the number of second-order interaction terms
        if self.num_feats > 1:
            self.len_interas = len(
                list(itertools.combinations(np.arange(0, self.num_feats), 2)))
        else:
            self.len_interas = 0
        # reversible instance normalization
        if use_revin:
            self.normalizer = RevIN(num_features=self.output_dim, axis=(1, 2))

        ### MLP Encoder: Learning coeffcients of measurement functions ###
        self.encoder = MLP(
            input_dim=input_dim * self.num_feats,
            output_dim=(latent_dim + self.num_sins * 2) * input_dim *
                       self.num_feats,  # we learn both the freq and magn for sin/cos,
            hidden_dim=encoder_hidden_dim,
            num_layers=encoder_num_layers,
            use_instancenorm=self.use_instancenorm,
            dropout_rate=dropout_rate)

        w = math.ceil((input_length - input_dim + 1) / self.stride)
        ### Transformer for Koopman Operator
        self.projU = nn.Linear(latent_dim * self.num_feats + self.len_interas, self.rank, bias=False)
        self.projV = nn.Linear(self.rank, latent_dim * self.num_feats + self.len_interas, bias=False)
        nn.init.orthogonal_(self.projU.weight)
        nn.init.orthogonal_(self.projV.weight)

        ### Global projection
        if self.add_global_operator:
            self.global_transform = nn.Parameter(torch.rand(
                latent_dim * self.num_feats + self.len_interas, latent_dim * self.num_feats + self.len_interas))

        self.encoder_layer_local = nn.TransformerEncoderLayer(
            d_model=w,
            nhead=num_heads,
            dim_feedforward=transformer_dim)
        self.transformer_encoder_local = nn.TransformerEncoder(
            self.encoder_layer_local, num_layers=transformer_num_layers)
        self.attention = nn.MultiheadAttention(
            embed_dim=w,
            num_heads=num_heads,
            batch_first=True)

        ### MLP Decoder: Reconstruct Observations from Measuremnets ###
        self.decoder = MLP(
            input_dim=latent_dim * self.num_feats + self.len_interas,
            output_dim=output_dim * self.num_feats,
            hidden_dim=decoder_hidden_dim,
            num_layers=decoder_num_layers,
            use_instancenorm=self.use_instancenorm,
            dropout_rate=dropout_rate)

    def single_forward(
            self,
            inps,  # input ts tensor
    ):
        ##################### Encoding ######################
        # the encoder learns the coefficients of basis functions
        encoder_outs = self.encoder(inps)
        encoder_outs = encoder_outs.reshape(
            inps.shape[0], inps.shape[1],
            (self.latent_dim + self.num_sins * 2),
            self.input_dim * self.num_feats)
        encoder_outs = encoder_outs.reshape(
            inps.shape[0], inps.shape[1],
            (self.latent_dim + self.num_sins * 2),
            self.input_dim, self.num_feats)
        inps = inps.reshape(inps.shape[0], inps.shape[1],
                            self.input_dim, self.num_feats)
        coefs = torch.einsum("blkdf, bldf -> blfk", encoder_outs, inps)

        ################ Calculate Meausurements ############
        embedding = torch.zeros(encoder_outs.shape[0], encoder_outs.shape[1],
                                self.num_feats, self.latent_dim).to(inps.device)

        for f in range(self.num_feats):
            # polynomials
            for i in range(self.num_poly):
                embedding[:, :, f, i] = coefs[:, :, f, i] ** (i + 1)

            # exponential function
            for i in range(self.num_poly, self.num_poly + self.num_exp):
                embedding[:, :, f, i] = torch.exp(coefs[:, :, f, i])

            # sine/cos functions
            for i in range(self.num_poly + self.num_exp,
                           self.num_poly + self.num_exp + self.num_sins):
                embedding[:, :, f,
                i] = coefs[:, :, f, self.num_sins * 2 + i] * torch.cos(
                    coefs[:, :, f, i])
                embedding[:, :, f, self.num_sins +
                                   i] = coefs[:, :, f, self.num_sins * 3 + i] * torch.sin(
                    coefs[:, :, f, self.num_sins + i])

            # the remaining ouputs are purely data-driven measurement functions.
            embedding[:, :, f, self.num_poly + self.num_exp +
                               self.num_sins * 2:] = coefs[:, :, f, self.num_poly +
                                                                    self.num_exp + self.num_sins * 4:]

        embedding = embedding.reshape(embedding.shape[0], embedding.shape[1], -1)
        # if there are multiple features,
        # second-order interaction terms should also be included
        if self.num_feats > 1:
            inter_lsts = list(itertools.combinations(np.arange(0, self.num_feats), 2))
            embedding_inter = torch.zeros(encoder_outs.shape[0],
                                          encoder_outs.shape[1],
                                          len(inter_lsts)).to(inps.device)
            for i, item in enumerate(inter_lsts):
                embedding_inter[Ellipsis, i] = (
                        coefs[:, :, item[0], 0] * coefs[:, :, item[1], 0]
                )
            embedding = torch.cat([embedding, embedding_inter], dim=-1)
        # Reconstruction
        reconstructions = self.decoder(embedding)

        embedding_lowrank = self.projU(embedding)
        # embedding_recon = self.projV(embedding_lowrank)
        trans_out = self.transformer_encoder_local(embedding_lowrank.transpose(1, 2))
        local_transform = self.attention(trans_out, trans_out, trans_out)[1]

        # Collect predicted measurements on the lookback window
        inp_embed_preds = []
        # Number of future step prediction from current time-point
        forward_iters = self.num_steps // self.input_dim
        if self.num_steps % self.input_dim > 0:
            forward_iters += 1

        for i in range(inps.shape[1] - forward_iters):
            forw = embedding_lowrank[:, i:i + 1]
            for k in range(forward_iters):
                if self.add_global_operator:
                    forw_g = torch.einsum("bnl, lh -> bnh", embedding[:, i:i + 1],
                                          self.global_transform)
                else:
                    forw_g = 0
                forw = torch.einsum("bnl, blh -> bnh", forw,
                                          local_transform)
                inp_embed_preds.append(self.projV(forw) + forw_g)
        embed_preds = torch.cat(inp_embed_preds, dim=1)

        # Reconstruction
        inp_preds = self.decoder(embed_preds).unfold(
            1, forward_iters, forward_iters)

        forw_preds = []
        forw = embedding_lowrank[:, -1:]
        # Forward predictions
        for i in range(forward_iters):
            if self.add_global_operator:
                forw_g = torch.einsum("bnl, lh -> bnh", embedding[:, -1:],
                                          self.global_transform)
            else:
                forw_g = 0
            forw = torch.einsum("bnl, blh -> bnh", forw,
                                      local_transform)
            forw_preds.append(self.projV(forw) + forw_g)
        forw_preds = torch.cat(forw_preds, dim=1)

        # Reconstruction
        forw_preds = self.decoder(forw_preds)

        #####################################################################
        return (reconstructions, inp_preds, forw_preds,
                embedding[:, 1:].unfold(1, forward_iters, 1),
                embed_preds.unfold(1, forward_iters, forward_iters))

    def forward(self, org_inps, tgts):
        # number of autoregressive step
        auto_steps = tgts.shape[1] // self.num_steps
        if tgts.shape[1] % self.num_steps > 0:
            auto_steps += 1

        if self.use_revin:
            denorm_outs = []
            norm_tgts = []
            norm_outs = []
            norm_inps = []
            norm_recons = []
            norm_inp_preds = []
            enc_embeds = []
            pred_embeds = []
            dmd_modes = []
            dmd_freq = []

            for i in range(auto_steps):
                try:
                    inps = org_inps.unfold(1, self.input_dim, self.stride).flatten(start_dim=2)
                except ValueError as valueerror:
                    raise ValueError(
                        "Input length is not divisible by input dim") from valueerror

                norm_inp = self.normalizer.forward(inps, mode="norm")
                norm_inps.append(norm_inp)

                single_forward_output = self.single_forward(norm_inp)
                (reconstructions, inp_preds, forw_preds, enc_embedding,
                 pred_embedding) = single_forward_output

                norm_recons.append(reconstructions)
                norm_inp_preds.append(inp_preds)
                enc_embeds.append(enc_embedding)
                pred_embeds.append(pred_embedding)

                forw_preds = forw_preds.reshape(forw_preds.shape[0], -1,
                                                self.num_feats)[:, :self.num_steps]
                norm_outs.append(forw_preds)

                # normalize tgts
                norm_tgts.append(
                    self.normalizer.normalize(tgts[:, i * self.num_steps:(i + 1) *
                                                                         self.num_steps]))

                # denormalize prediction and add back to the input
                denorm_outs.append(self.normalizer.forward(forw_preds, mode="denorm"))
                org_inps = torch.cat([org_inps[:, self.num_steps:], denorm_outs[-1]],
                                     dim=1)

            norm_outs = torch.cat(norm_outs, dim=1)
            norm_tgts = torch.cat(norm_tgts, dim=1)
            denorm_outs = torch.cat(denorm_outs, dim=1)

            norm_inps = torch.cat(norm_inps, dim=0)
            norm_inp_preds = torch.cat(norm_inp_preds, dim=0)
            norm_recons = torch.cat(norm_recons, dim=0)
            enc_embeds = torch.cat(enc_embeds, dim=0)
            pred_embeds = torch.cat(pred_embeds, dim=0)

            forward_output = [
                denorm_outs[:, :norm_tgts.shape[1]],
                [norm_outs[:, :norm_tgts.shape[1]], norm_tgts],
                [norm_recons, norm_inp_preds, norm_inps], [enc_embeds, pred_embeds],
                None, None
            ]

            return forward_output

        else:
            outs = []
            true_inps = []
            recons = []
            inputs_preds = []
            enc_embeds = []
            pred_embeds = []
            dmd_modes = []
            dmd_freq = []

            for i in range(auto_steps):
                try:
                    inps = org_inps.unfold(1, self.input_dim, self.stride).flatten(start_dim=2)
                except ValueError as valueerror:
                    raise ValueError(
                        "Input length is not divisible by input dim") from valueerror

                true_inps.append(inps)
                single_forward_output = self.single_forward(inps)
                (reconstructions, inp_preds, forw_preds, enc_embedding,
                 pred_embedding) = single_forward_output

                recons.append(reconstructions)
                inputs_preds.append(inp_preds)
                enc_embeds.append(enc_embedding)
                pred_embeds.append(pred_embedding)

                forw_preds = forw_preds.reshape(forw_preds.shape[0], -1,
                                                self.num_feats)[:, :self.num_steps]
                outs.append(forw_preds)

                org_inps = torch.cat([org_inps[:, self.num_steps:], outs[-1]], dim=1)

            outs = torch.cat(outs, dim=1)
            true_inps = torch.cat(true_inps, dim=0)
            inputs_preds = torch.cat(inputs_preds, dim=0)
            recons = torch.cat(recons, dim=0)
            enc_embeds = torch.cat(enc_embeds, dim=0)
            pred_embeds = torch.cat(pred_embeds, dim=0)

            forward_output = [
                outs[:, :tgts.shape[1]], [outs[:, :tgts.shape[1]], tgts],
                [recons, inputs_preds, true_inps], [enc_embeds, pred_embeds],
                None, None
            ]

            return forward_output


class DeepDMDtest(unittest.TestCase):
    def test_model_output(self):
        model = DeepDMD(
            input_dim=8,
            stride=4,
            input_length=128,
            output_dim=32,
            num_steps=32,
            encoder_hidden_dim=256,
            encoder_num_layers=3,
            decoder_num_layers=3,
            decoder_hidden_dim=256,
            latent_dim=32,
            num_feats=50,
            rank=8,
            num_sins=3,
            num_poly=2,
            num_exp=2
        )
        inp = torch.rand(size=(9, 128, 50))
        tgt = torch.rand(size=(9, 32, 50))
        output = model(inp, tgt)

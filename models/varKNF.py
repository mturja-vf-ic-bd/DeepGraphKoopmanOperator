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
import numpy as np
import torch
from torch import nn
import time

from layers.mlp import MLP
from models.model_utils import varying_multiply

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Koopman(nn.Module):
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
                 num_feats=1,
                 add_global_operator=False,
                 add_control=False,
                 control_num_layers=None,
                 control_hidden_dim=None,
                 use_revin=True,
                 use_instancenorm=False,
                 real_modes=1,
                 complex_modes=3,
                 num_sins=-1,
                 num_poly=-1,
                 num_exp=-1,
                 num_heads=1,
                 transformer_dim=128,
                 transformer_num_layers=3,
                 omega_dim=128,
                 omega_num_layers=3,
                 dropout_rate=0,
                 stride=1):

        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_length = input_length
        self.num_steps = num_steps
        self.latent_dim = latent_dim
        self.use_revin = use_revin
        self.real_modes = real_modes
        self.complex_modes = complex_modes
        self.use_instancenorm = use_instancenorm
        self.add_control = add_control
        self.add_global_operator = add_global_operator
        self.num_feats = num_feats
        self.stride = stride

        assert complex_modes % 2 == 0, \
            "Number of complex modes should be even"

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

        ### Global Linear Koopman Operator: A finite matrix ###
        if self.add_global_operator:
            self.global_linear_transform = nn.Linear(
                latent_dim * self.num_feats + self.len_interas,
                latent_dim * self.num_feats + self.len_interas,
                bias=False)

        ### Transformer Encoder: learning Local Koopman Eigen-functions
        w = math.ceil((input_length - input_dim + 1) / stride)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=w,
            nhead=num_heads,
            dim_feedforward=transformer_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=transformer_num_layers)
        self.eigen_func = nn.Linear(
            w, real_modes + complex_modes // 2)

        ### Omega-net: Learning time-specific frequency
        self.omega_net = MLP(
            input_dim=latent_dim * num_feats + self.len_interas,
            output_dim=real_modes + complex_modes,
            hidden_dim=omega_dim,
            num_layers=omega_num_layers,
            use_instancenorm=use_instancenorm,
            dropout_rate=dropout_rate)

        ### MLP Control/Feedback Module
        if self.add_control:
            # learn the adjustment to the koopman operator
            # based on the prediction error on the look back window.
            self.control = MLP(
                input_dim=(w - 1) * self.num_feats * self.input_dim,
                output_dim=latent_dim * self.num_feats + self.len_interas,
                hidden_dim=control_hidden_dim,
                num_layers=control_num_layers,
                use_instancenorm=self.use_instancenorm,
                dropout_rate=dropout_rate)

        ### MLP Decoder: Reconstruct Observations from Measuremnets ###
        self.decoder = MLP(
            input_dim=latent_dim * self.num_feats + self.len_interas,
            output_dim=output_dim * self.num_feats,
            hidden_dim=decoder_hidden_dim,
            num_layers=decoder_num_layers,
            use_instancenorm=self.use_instancenorm,
            dropout_rate=dropout_rate)

        self.regressor = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2*self.len_interas, 4))

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

        # Koopman Eigenfunctions and Eigenvals
        trans_out = self.transformer_encoder(embedding.transpose(1, 2))
        local_eig_func = self.eigen_func(trans_out)
        local_eig_func = local_eig_func / torch.norm(
            local_eig_func, dim=1, keepdim=True, p=2)
        local_eig_func_real = torch.cat([local_eig_func[:, -self.len_interas:, 0],
                                         local_eig_func[:, -self.len_interas:, self.real_modes]],
                                        dim=-1)
        local_eig_func_complex = torch.stack(
            [local_eig_func[:, :, -self.complex_modes // 2:],
             local_eig_func[:, :, -self.complex_modes // 2:]],
            dim=-1).flatten(start_dim=2)
        local_eig_func_aug = torch.cat(
            [local_eig_func[:, :, 0:self.real_modes], local_eig_func_complex],
            dim=-1)
        local_eig_vals = self.omega_net(embedding)

        # Collect predicted measurements on the lookback window
        inp_embed_preds = []
        # Number of future step prediction from current time-point
        forward_iters = self.num_steps // self.input_dim
        if self.num_steps % self.input_dim > 0:
            forward_iters += 1

        for i in range(inps.shape[1] - forward_iters):
            forw = embedding[:, i:i + 1]
            for k in range(forward_iters):
                if self.add_global_operator:
                    forw = self.global_linear_transform(forw)
                forw = varying_multiply(
                    forw,
                    local_eig_func_aug,
                    local_eig_vals[:, i:i + 1],
                    1,
                    self.real_modes,
                    self.complex_modes // 2,
                    k=1)
                inp_embed_preds.append(forw)
        embed_preds = torch.cat(inp_embed_preds, dim=1)

        # Reconstruction
        inp_preds = self.decoder(embed_preds).unfold(
            1, forward_iters, forward_iters)
        #########################################################

        ########## Generate Predictions on the Forecasting Window ##########
        # If the predictions on the lookback window deviates a lot from groud truth,
        # adjust the Koopman operator with the control module.
        # if self.add_control:
        #     pred_diff = inp_preds.reshape(
        #         inp_preds.shape[0], -1) - \
        #                 inps[:, 1:].unfold(1, forward_iters,
        #                                    forward_iters).reshape(
        #                     inp_preds.shape[0], -1)
            # linear_adj = self.control(pred_diff)
            # linear_adj = torch.stack(
            #     [torch.diagflat(linear_adj[i]) for i in range(len(linear_adj))])

        forw_preds = []
        forw = embedding[:, -1:]
        # Forward predictions
        for i in range(forward_iters):
            if self.add_global_operator:
                forw = self.global_linear_transform(forw)
            forw = varying_multiply(
                forw,
                local_eig_func_aug,
                local_eig_vals[:, -1:],
                1,
                self.real_modes,
                self.complex_modes // 2,
                k=1)
            # if self.add_control:
            #     forw_l = forw_l + torch.einsum("bnl, blh -> bnh", forw_l,
            #                                    linear_adj)
            forw_preds.append(forw)
        forw_preds = torch.cat(forw_preds, dim=1)

        # Reconstruction
        forw_preds = self.decoder(forw_preds)

        # Regression task
        y_pred = self.regressor(local_eig_func_real)
        #####################################################################
        return (reconstructions, inp_preds, forw_preds,
                embedding[:, 1:].unfold(1, forward_iters, 1),
                embed_preds.unfold(1, forward_iters, forward_iters), local_eig_func_aug, y_pred)

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

            for i in range(auto_steps):
                try:
                    inps = org_inps.unfold(1, self.input_dim, self.stride).flatten(start_dim=2)
                    # inps = org_inps.reshape(org_inps.shape[0], -1, self.input_dim,
                    #                         self.num_feats)
                    # inps = inps.reshape(org_inps.shape[0], -1,
                    #                     self.input_dim * self.num_feats)
                except ValueError as valueerror:
                    raise ValueError(
                        "Input length is not divisible by input dim") from valueerror

                norm_inp = self.normalizer.forward(inps, mode="norm")
                norm_inps.append(norm_inp)

                single_forward_output = self.single_forward(norm_inp)
                (reconstructions, inp_preds, forw_preds, enc_embedding,
                 pred_embedding, local_eig_func_aug, y_pred) = single_forward_output

                norm_recons.append(reconstructions)
                norm_inp_preds.append(inp_preds)
                enc_embeds.append(enc_embedding)
                pred_embeds.append(pred_embedding)
                dmd_modes.append(local_eig_func_aug)

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
            dmd_modes = torch.cat(dmd_modes, dim=0)

            forward_output = [
                denorm_outs[:, :norm_tgts.shape[1]],
                [norm_outs[:, :norm_tgts.shape[1]], norm_tgts],
                [norm_recons, norm_inp_preds, norm_inps], [enc_embeds, pred_embeds],
                dmd_modes, y_pred
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

            for i in range(auto_steps):
                try:
                    inps = org_inps.unfold(1, self.input_dim, self.stride).flatten(start_dim=2)
                except ValueError as valueerror:
                    raise ValueError(
                        "Input length is not divisible by input dim") from valueerror

                true_inps.append(inps)
                single_forward_output = self.single_forward(inps)
                (reconstructions, inp_preds, forw_preds, enc_embedding,
                 pred_embedding, local_eig_func_aug, y_pred) = single_forward_output

                recons.append(reconstructions)
                inputs_preds.append(inp_preds)
                enc_embeds.append(enc_embedding)
                pred_embeds.append(pred_embedding)
                dmd_modes.append(local_eig_func_aug)

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
            dmd_modes = torch.cat(dmd_modes, dim=0)

            forward_output = [
                outs[:, :tgts.shape[1]], [outs[:, :tgts.shape[1]], tgts],
                [recons, inputs_preds, true_inps], [enc_embeds, pred_embeds],
                dmd_modes, y_pred
            ]

            return forward_output


class varKNFtest(unittest.TestCase):
    def test_model_output(self):
        model = Koopman(
            input_dim=8,
            input_length=256,
            output_dim=32,
            num_steps=32,
            encoder_hidden_dim=256,
            encoder_num_layers=3,
            decoder_num_layers=3,
            decoder_hidden_dim=256,
            latent_dim=32,
            num_feats=50,
            real_modes=3,
            complex_modes=8,
            num_sins=3,
            num_poly=2,
            num_exp=2
        )
        inp = torch.rand(size=(2, 256, 50))
        tgt = torch.rand(size=(2, 32, 50))
        output = model(inp, tgt)
        print(output[-1].shape)

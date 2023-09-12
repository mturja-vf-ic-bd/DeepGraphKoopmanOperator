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

"""Training and evalution functions for one epoch."""

import numpy as np
from torch import nn


def get_lr(optimizer):
  for param_group in optimizer.param_groups:
    return param_group["lr"]


def train_epoch_koopman(train_loader,
                        model,
                        loss_fun,
                        optimizer,
                        rank=0):
  """Train the KNF model for one epoch.

  Args:
    train_loader: the dataloader of the training set
    model: KNF model
    loss_fun: loss function
    optimizer: Adam

  Returns:
    RMSE on the training set

  """
  train_loss = []
  train_loss_recon = []
  train_loss_pred = []
  train_loss_embedding = []

  for inps, tgts in train_loader:
    if len(inps.shape) == 2:
      inps = inps.unsqueeze(-1)
      tgts = tgts.unsqueeze(-1)
    if rank >= 0:
      inps = inps.to(rank)
      tgts = tgts.to(rank)

    (_, [norm_outs,
         norm_tgts], [norm_recons, norm_inp_preds,
                      norm_inps], [enc_embeds,
                                   pred_embeds], _) = model(inps, tgts)
    loss_recon = loss_fun(norm_recons, norm_inps)
    loss_pred = loss_fun(norm_inp_preds, norm_inps[:, 1:].unfold(
      1, norm_inp_preds.shape[-1], 1))
    loss_embedding = loss_fun(enc_embeds, pred_embeds)
    loss_forward_pred = loss_fun(norm_outs, norm_tgts)
    loss = loss_forward_pred + loss_recon + loss_pred + \
                                            loss_embedding
    train_loss_recon.append(loss_recon.item())
    train_loss_pred.append(loss_pred.item())
    train_loss_embedding.append(loss_embedding.item())
    train_loss.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
    optimizer.step()
  return np.sqrt(np.mean(train_loss)), np.sqrt(np.mean(train_loss_recon)), \
            np.sqrt(np.mean(train_loss_pred)), np.sqrt(np.mean(train_loss_embedding))


def eval_epoch_koopman(eval_loader, model, loss_fun, rank=0):
  """Evaluate the KNF model on the validation set.

  Args:
    eval_loader: the dataloader of the validation/test set
    model: KNF model
    loss_fun: MSE loss
  Returns:
    RMSE, predictions and ground truth on the evalution set

  """
  eval_loss = []
  all_preds = []
  all_trues = []
  all_recon = []
  all_recon_trues = []
  all_dmd_modes = []
  for inps, tgts in eval_loader:
    if len(inps.shape) == 2:
      inps = inps.unsqueeze(-1)
      tgts = tgts.unsqueeze(-1)
    if rank >= 0:
      inps = inps.to(rank)
      tgts = tgts.to(rank)

    (denorm_outs, [norm_outs,
                    norm_tgts], [norm_recons, norm_inp_preds,
                                 norm_inps], [enc_embeds,
                                              pred_embeds], dmd_modes) = model(inps, tgts)

    loss_recon = loss_fun(norm_recons, norm_inps)
    loss_pred = loss_fun(norm_inp_preds, norm_inps[:, 1:].unfold(
      1, norm_inp_preds.shape[-1], 1))
    loss_embedding = loss_fun(enc_embeds, pred_embeds)
    loss = loss_fun(norm_outs,
                    norm_tgts) + loss_recon + loss_pred + loss_embedding

    eval_loss.append(loss.item())
    all_preds.append(denorm_outs.cpu().data.numpy())
    all_trues.append(tgts.cpu().data.numpy())
    all_recon.append(norm_recons.cpu().data.numpy())
    all_recon_trues.append(norm_inps.cpu().data.numpy())
    all_dmd_modes.append(dmd_modes.cpu().data.numpy())
  return np.sqrt(np.mean(eval_loss)), np.concatenate(
      all_preds, axis=0), np.concatenate(
          all_trues, axis=0), np.concatenate(
            all_recon, axis=0), np.concatenate(
              all_recon_trues, axis=0), np.concatenate(
                all_dmd_modes, axis=0), loss_pred

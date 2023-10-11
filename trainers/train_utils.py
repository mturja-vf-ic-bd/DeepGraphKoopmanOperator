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
import torch


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def compute_reproducibility_loss(dmd_modes, sample_per_subject):
    dmd_modes = dmd_modes.view(-1, sample_per_subject, dmd_modes.shape[1], dmd_modes.shape[2])
    total_loss = 0
    for s in range(dmd_modes.shape[0]):
        loss = 0
        for i in range(0, dmd_modes.shape[3], 2):
            loss = loss + (1 - torch.mm(dmd_modes[s, :, :, i], dmd_modes[s, :, :, i].t())).mean()
        loss = loss / dmd_modes.shape[2]
        total_loss = total_loss + loss
    return total_loss / dmd_modes.shape[0]


def train_epoch_koopman(train_loader,
                        model,
                        loss_fun,
                        optimizer,
                        rank=0,
                        sample_per_subject=1):
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
    train_loss_repod = []

    for inps, tgts, y_true in train_loader:
        if len(inps.shape) == 2:
            inps = inps.unsqueeze(-1)
            tgts = tgts.unsqueeze(-1)
        if rank >= 0:
            inps = inps.to(rank)
            tgts = tgts.to(rank)

        (_, [norm_outs,
             norm_tgts], [norm_recons, norm_inp_preds,
                          norm_inps], [enc_embeds,
                                       pred_embeds], dmd_modes, _) = model(inps, tgts)
        loss_recon = loss_fun(norm_recons, norm_inps)
        loss_pred = loss_fun(norm_inp_preds, norm_inps[:, 1:].unfold(
            1, norm_inp_preds.shape[-1], 1))
        loss_embedding = loss_fun(enc_embeds, pred_embeds)
        loss_forward_pred = loss_fun(norm_outs, norm_tgts)
        loss_repod = compute_reproducibility_loss(dmd_modes, sample_per_subject)
        loss = loss_forward_pred + loss_recon + loss_pred + \
               loss_embedding + loss_repod
        train_loss_recon.append(loss_recon.item())
        train_loss_pred.append(loss_pred.item())
        train_loss_embedding.append(loss_embedding.item())
        train_loss_repod.append(loss_repod.item())
        train_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
        optimizer.step()
    return np.sqrt(np.mean(train_loss)), np.sqrt(np.mean(train_loss_recon)), \
           np.sqrt(np.mean(train_loss_pred)), np.sqrt(np.mean(train_loss_embedding)), \
           np.sqrt(np.mean(train_loss_repod))


def eval_epoch_koopman(eval_loader, model, loss_fun, rank=0, sample_per_subject=1):
    """Evaluate the KNF model on the validation set.

  Args:
    eval_loader: the dataloader of the validation/test set
    model: KNF model
    loss_fun: MSE loss
  Returns:
    RMSE, predictions and ground truth on the evalution set

  """
    eval_loss = []
    pred_loss = []
    repod_loss = []
    regress_loss = []
    all_preds = []
    all_trues = []
    all_recon = []
    all_recon_trues = []
    all_dmd_modes = []
    all_dmd_vals = []
    y_trues = []
    for inps, tgts, y_true in eval_loader:
        if len(inps.shape) == 2:
            inps = inps.unsqueeze(-1)
            tgts = tgts.unsqueeze(-1)
        if rank >= 0:
            inps = inps.to(rank)
            tgts = tgts.to(rank)
            y_true = y_true.to(rank)

        (denorm_outs, [norm_outs,
                       norm_tgts], [norm_recons, norm_inp_preds,
                                    norm_inps], [enc_embeds,
                                                 pred_embeds], dmd_modes, dmd_vals) = model(inps, tgts)

        loss_recon = loss_fun(norm_recons, norm_inps)
        loss_pred = loss_fun(norm_inp_preds, norm_inps[:, 1:].unfold(
            1, norm_inp_preds.shape[-1], 1))
        loss_embedding = loss_fun(enc_embeds, pred_embeds)
        loss_repod = compute_reproducibility_loss(dmd_modes, sample_per_subject)
        loss = loss_fun(norm_outs,
                        norm_tgts) + loss_recon + loss_pred + loss_embedding + loss_repod

        eval_loss.append(loss.item())
        pred_loss.append(loss_pred.cpu().data.numpy())
        repod_loss.append(loss_repod.cpu().data.numpy())
        all_preds.append(denorm_outs.cpu().data.numpy())
        all_trues.append(tgts.cpu().data.numpy())
        all_recon.append(norm_recons.cpu().data.numpy())
        all_recon_trues.append(norm_inps.cpu().data.numpy())
        all_dmd_modes.append(dmd_modes.cpu().data.numpy())
        all_dmd_vals.append(dmd_vals.cpu().data.numpy())
        y_trues.append(y_true.cpu().data.numpy())
    return np.sqrt(np.mean(eval_loss)), np.concatenate(
        all_preds, axis=0), np.concatenate(
        all_trues, axis=0), np.concatenate(
        all_recon, axis=0), np.concatenate(
        all_recon_trues, axis=0), np.concatenate(
        all_dmd_modes, axis=0), np.concatenate(
        all_dmd_vals, axis=0), np.sqrt(np.mean(pred_loss)), \
           np.sqrt(np.mean(repod_loss)), np.concatenate(y_trues, axis=0)

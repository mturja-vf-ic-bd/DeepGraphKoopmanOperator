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

"""Main script for training KNF."""
import os
import random
import time
from tqdm import tqdm
from collections import OrderedDict
import re

from CONSTANTS import CONSTANTS
from dataloaders.MegaTrawl import MegaTrawlDataset, HCPTaskfMRIDataset, M4Dataset
from models.DeepDMD import DeepDMD
from trainers.metrics import SMAPE
from trainers.train_utils import train_epoch_koopman, eval_epoch_koopman, get_lr

import numpy as np
import torch
from torch import nn
from torch.utils import data
from torch.utils.data import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp


def ddp_setup(rank: int, world_size: int):
    """
    Args:
       rank: Unique identifier of each process
      world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12375"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def main(rank, world_size, input_dim, input_length,
         latent_dim,
         train_output_length, test_output_length,
         num_steps, encoder_hidden_dim,
         decoder_hidden_dim, encoder_num_layers,
         decoder_num_layers, use_revin, use_instancenorm,
         add_global_operator,
         batch_size, num_epochs, learning_rate,
         dataset_name, seed, jumps,
         num_poly, num_sins, num_exp, decay_rate,
         transformer_dim, transformer_num_layers,
         num_feats, dropout_rate, min_epochs,
         mode, stride, freq, dmd_rank):
    ddp_setup(rank, world_size)
    random.seed(seed)  # python random generator
    np.random.seed(seed)  # numpy random generator

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    output_dim = input_dim
    if dataset_name == "megatrawl":
        train_set = MegaTrawlDataset(
            input_length=input_length,
            output_length=train_output_length,
            mode="train",
            jumps=jumps)
        valid_set = MegaTrawlDataset(
            input_length=input_length,
            output_length=train_output_length,
            mode="valid",
            jumps=jumps)
        test_set = MegaTrawlDataset(
            input_length=input_length,
            output_length=test_output_length,
            jumps=jumps,
            mode="test")
    elif dataset_name == "M4":
        data_dir = CONSTANTS.CODEDIR + "/data/M4/"
        direc = os.path.join(data_dir, "train.npy")
        direc_test = os.path.join(data_dir, "test.npy")
        train_set = M4Dataset(
            input_length=input_length,
            output_length=train_output_length,
            freq=freq,
            direc=direc,
            mode="train",
            jumps=jumps)
        valid_set = M4Dataset(
            input_length=input_length,
            output_length=train_output_length,
            freq=freq,
            direc=direc,
            mode="valid",
            jumps=jumps)
        test_set = M4Dataset(
            input_length=input_length,
            output_length=test_output_length,
            freq=freq,
            direc=direc,
            direc_test=direc_test,
            mode="test")
    else:
        train_set = HCPTaskfMRIDataset(
            input_length=input_length,
            output_length=train_output_length,
            mode="train",
            jumps=jumps,
            datapath=CONSTANTS.DATADIR)
        valid_set = HCPTaskfMRIDataset(
            input_length=input_length,
            output_length=train_output_length,
            mode="valid",
            jumps=jumps,
            datapath=CONSTANTS.DATADIR)
        test_set = HCPTaskfMRIDataset(
            input_length=input_length,
            output_length=test_output_length,
            jumps=jumps,
            mode="test",
            datapath=CONSTANTS.DATADIR)

    train_loader = data.DataLoader(
        train_set, batch_size=batch_size, shuffle=False,
        sampler=DistributedSampler(train_set)
    )
    valid_loader = data.DataLoader(
        valid_set, batch_size=batch_size, shuffle=False,
        sampler=DistributedSampler(valid_set)
    )
    test_loader = data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False
    )

    model_name = (
            "DDMD_"
            + str(dataset_name)
            + f"_seed{seed}_jumps{jumps}_poly{num_poly}_"
              f"sin{num_sins}_exp{num_exp}_"
              f"lr{learning_rate}_decay{decay_rate}_dim{input_dim}_"
              f"inp{input_length}_pred{train_output_length}_"
              f"num{num_steps}_latdim{latent_dim}"
              f"_RevIN{use_revin}_"
              f"insnorm{use_instancenorm}_globalK{add_global_operator}"
              f"_stride{stride}_rank{dmd_rank}"
    )
    model = DeepDMD(
        # number of steps of historical observations encoded at every step
        input_dim=input_dim,
        # input length of ts
        input_length=input_length,
        # number of output features
        output_dim=output_dim,
        # number of prediction steps every forward pass
        num_steps=num_steps,
        # hidden dimension of encoder
        encoder_hidden_dim=encoder_hidden_dim,
        # hidden dimension of decoder
        decoder_hidden_dim=decoder_hidden_dim,
        # number of layers in the encoder
        encoder_num_layers=encoder_num_layers,
        # number of layers in the decoder
        decoder_num_layers=decoder_num_layers,
        # number of feature
        num_feats=num_feats,
        # dimension of finite koopman space
        latent_dim=latent_dim,
        # rank of the Koopman Operator
        rank=dmd_rank,
        use_revin=use_revin,  # whether to use reversible normalization
        # whether to use instance normalization on hidden states
        use_instancenorm=use_instancenorm,
        # number of pairs of sine and cosine measurement functions
        num_sins=num_sins,
        # the highest order of polynomial functions
        num_poly=num_poly,
        # number of exponential functions
        num_exp=num_exp,
        # Number of the head the transformer encoder
        num_heads=1,
        # hidden dimension of tranformer encoder
        transformer_dim=transformer_dim,
        # number of layers in the transformer encoder
        transformer_num_layers=transformer_num_layers,
        # dropout rate of MLP modules
        dropout_rate=dropout_rate,
        stride=stride,
        add_global_operator=add_global_operator
    ).to(rank)
    results_dir = dataset_name + "_results/"
    if os.path.exists(results_dir + model_name + ".pth"):
        ckpt = torch.load(results_dir + model_name + ".pth",
                          map_location={"cuda:0": f"cuda:{rank}"})
        state_dict = ckpt["model"]
        model_dict = OrderedDict()
        pattern = re.compile('module.')
        for k, v in state_dict.items():
            if re.search("module", k):
                model_dict[re.sub(pattern, '', k)] = v
            else:
                model_dict[k] = v
        model.load_state_dict(model_dict)
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        last_epoch = ckpt["epoch"]
        learning_rate = ckpt["lr"]
        print("Resume Training")
        print("last_epoch:", last_epoch, "learning_rate:", learning_rate)
    else:
        last_epoch = 0
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        if rank == 0 and not os.path.exists(results_dir):
            os.mkdir(results_dir)
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        print("New model")

    best_model = model
    print("number of params:",
          sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=decay_rate
    )  # stepwise learning rate decay
    loss_fun = nn.MSELoss()

    all_train_rmses, all_eval_rmses = [], []
    best_eval_rmse = 1e6

    if mode == "train":
        for epoch in tqdm(range(last_epoch, num_epochs)):
            start_time = time.time()
            train_loader.sampler.set_epoch(epoch)
            train_rmse, train_recon, train_pred, train_embedding = train_epoch_koopman(
                train_loader,
                model,
                loss_fun,
                optimizer,
                rank=rank)
            eval_rmse, _, _, _, _, _, _, eval_pred = eval_epoch_koopman(
                valid_loader,
                model, loss_fun,
                rank=rank)

            if eval_rmse < best_eval_rmse and rank == 0:
                best_eval_rmse = eval_rmse
                best_model = model
                torch.save({"model": best_model.module.state_dict(),
                            "epoch": epoch, "lr": get_lr(optimizer)},
                           results_dir + model_name + ".pth")

            all_train_rmses.append(train_rmse)
            all_eval_rmses.append(eval_rmse)

            if np.isnan(train_rmse) or np.isnan(eval_rmse):
                raise ValueError("The model generate NaN values")

            # train the model at least 60 epochs and do early stopping
            if epoch > min_epochs and np.mean(all_eval_rmses[-10:]) > np.mean(
                    all_eval_rmses[-20:-10]):
                break

            epoch_time = time.time() - start_time
            scheduler.step()
            print(
                'Epoch {} | T: {:0.2f} | Train RMSE: {:0.3f}, Train Recon: {:0.3f}, Train Pred: {:0.3f}, '
                'Train Embedding: {:0.3f} | Valid RMSE: {:0.3f}, Valid Pred: {:0.3f}'
                .format(epoch + 1, epoch_time / 60, train_rmse,
                        train_recon, train_pred, train_embedding,
                        eval_rmse, eval_pred),
                flush=True)
    elif mode == "test":
        print(f"Test set length: {len(test_loader)}")
        test_rmse, test_preds, test_tgts, \
        test_recon, test_recon_true, test_dmd_modes, \
        test_dmd_vals, test_pred_rmse = eval_epoch_koopman(
            test_loader, best_model, loss_fun)
        print(f"Test pred shape: {test_preds.shape} | Test Pred: {test_pred_rmse:0.3f}")
        # Denormalize the predictions
        if dataset_name == "M4":
            test_preds = (test_preds * test_set.ts_stds.reshape(
                -1, 1, 1)) + test_set.ts_means.reshape(-1, 1, 1)
            test_tgts = (test_tgts * test_set.ts_stds.reshape(
                -1, 1, 1)) + test_set.ts_means.reshape(-1, 1, 1)
        eval_score = SMAPE(test_preds, test_tgts)
        print("SMAPE scores:", eval_score)
        if rank == 0:
            torch.save(
                {
                    "test_preds": test_preds,
                    "test_tgts": test_tgts,
                    "test_recon": test_recon,
                    "test_recon_true": test_recon_true,
                    "eval_score": eval_score,
                    "test_dmd_vals": test_dmd_vals
                }, results_dir + model_name + ".pt")
            np.save(results_dir + model_name + ".npy", test_dmd_modes)
    destroy_process_group()


def run_ddp_training():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--max_epochs", nargs="?", type=int, default=100)
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--ckpt", type=str, help="Checkpoint file for prediction")
    parser.add_argument("--from_ckpt", dest='from_ckpt', default=False, action='store_true')
    parser.add_argument("--exp_name", type=str, default="var_knf",
                        help="Name you experiment")
    parser.add_argument("--write_dir", type=str, default="log/varKNF")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_feats", type=int, default=50)
    parser.add_argument("--input_dim", type=int, default=8)
    parser.add_argument("--input_length", type=int, default=256)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--dmd_rank", type=int, default=16)
    parser.add_argument("--train_output_length", type=int, default=32)
    parser.add_argument("--test_output_length", type=int, default=32)
    parser.add_argument("--num_steps", type=int, default=32)
    parser.add_argument("--min_epochs", type=int, default=10)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--use_revin", action='store_true')
    parser.add_argument("--use_instancenorm", action='store_true')
    parser.add_argument("--add_global_operator", action='store_true')
    parser.add_argument("--seed", type=int, default=901)
    parser.add_argument("--jumps", type=int, default=128)
    parser.add_argument("--num_poly", type=int, default=3)
    parser.add_argument("--num_sins", type=int, default=5)
    parser.add_argument("--num_exp", type=int, default=2)
    parser.add_argument("--decay_rate", type=float, default=0.99)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="megatrawl")
    parser.add_argument("--data_freq", type=str, default="Weekly")
    parser.set_defaults(
        use_revin=False,
        use_instancenorm=False)

    args = parser.parse_args()
    world_size = torch.cuda.device_count()
    print("Hyper-parameters:")
    for k, v in vars(args).items():
        print("{} -> {}".format(k, v))
    mp.spawn(main,
             args=(world_size, args.input_dim, args.input_length,
                   args.latent_dim, args.train_output_length, args.test_output_length,
                   args.num_steps, args.hidden_dim,
                   args.hidden_dim, args.num_layers,
                   args.num_layers, args.use_revin, args.use_instancenorm,
                   args.add_global_operator,
                   args.batch_size, args.max_epochs, args.lr,
                   args.dataset, args.seed, args.jumps,
                   args.num_poly, args.num_sins, args.num_exp, args.decay_rate,
                   args.hidden_dim, args.num_layers,
                   args.num_feats, args.dropout_rate, args.min_epochs, args.mode,
                   args.stride, args.data_freq, args.dmd_rank),
             nprocs=world_size, join=True)


if __name__ == "__main__":
    start_time = time.time()
    run_ddp_training()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time elapsed {elapsed_time // 3600}h "
          f"{(elapsed_time % 3600) // 60}m "
          f"{((elapsed_time % 3600) % 60)}s")

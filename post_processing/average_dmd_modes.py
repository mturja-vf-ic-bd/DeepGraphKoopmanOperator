import unittest

import torch
import numpy as np
import os

from tqdm import tqdm

from CONSTANTS import CONSTANTS


def pearson(X):
    m = torch.mean(X, dim=-1, keepdim=True)
    s = torch.std(X, dim=-1, keepdim=True)
    X_norm = (X - m) / s
    corr = torch.matmul(X_norm, X_norm.transpose(-2, -1)) / (X.shape[-1] - 1)
    return corr


def compute_correlation_from_modes(X, n_nodes=50):
    """

    :param X: is a (B, T, K, d) tensor for a subject
    where T is the number of windows and K is the number
    of dmd modes, d is the dmd mode dimension
    :return:
    """

    n_edges = (n_nodes * (n_nodes - 1)) // 2
    X_avg = X.mean(dim=1)
    X_avg = X_avg[:, :, :-n_edges].view((X_avg.shape[0], X_avg.shape[1], n_nodes, -1))
    return pearson(X_avg)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=0.0003, type=float, help="learning rate")
    parser.add_argument("--num_feats", type=int, default=50)
    parser.add_argument("--input_dim", type=int, default=8)
    parser.add_argument("--input_length", type=int, default=128)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--real_modes", type=int, default=1)
    parser.add_argument("--complex_modes", type=int, default=32)
    parser.add_argument("--train_output_length", type=int, default=32)
    parser.add_argument("--test_output_length", type=int, default=32)
    parser.add_argument("--num_steps", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--use_revin", action='store_true')
    parser.add_argument("--use_instancenorm", action='store_true')
    parser.add_argument("--add_global_operator", action='store_true')
    parser.add_argument("--add_dmd_operator", action='store_true')
    parser.add_argument("--add_local_operator", action='store_true')
    parser.add_argument("--add_control", action='store_true')
    parser.add_argument("--seed", type=int, default=902)
    parser.add_argument("--jumps", type=int, default=128)
    parser.add_argument("--num_poly", type=int, default=3)
    parser.add_argument("--num_sins", type=int, default=5)
    parser.add_argument("--num_exp", type=int, default=2)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--dataset", type=str, default="megatrawl")
    parser.add_argument("--decay_rate", type=float, default=1.0)
    parser.add_argument("--data_freq", type=str, default="Weekly")
    args = parser.parse_args()

    model_name = (
            "SKNF_"
            + str(args.dataset)
            + f"_seed{args.seed}_jumps{args.jumps}_poly{args.num_poly}_"
              f"sin{args.num_sins}_exp{args.num_exp}_"
              f"lr{args.lr}_decay{args.decay_rate}_dim{args.input_dim}_"
              f"inp{args.input_length}_pred{args.train_output_length}_"
              f"num{args.num_steps}__latdim{args.latent_dim}_rm={args.real_modes}"
              f"_cm={args.complex_modes}_RevIN{args.use_revin}_"
              f"insnorm{args.use_instancenorm}_stride{args.stride}"
    )
    results_dir = CONSTANTS.DATADIR + args.dataset + "_results/"
    results = torch.load(results_dir + model_name + ".pt")
    dmd_modes = np.load(results_dir + model_name + ".npy")
    sample_per_subject = (2048 - args.input_length - args.input_dim) // args.jumps + 1
    dmd_modes = dmd_modes.reshape(-1, sample_per_subject, dmd_modes.shape[1], dmd_modes.shape[2])
    dmd_modes = torch.from_numpy(dmd_modes).transpose(-1, -2)

    print(f"DMD mode shape: {dmd_modes.shape}")

    corr = compute_correlation_from_modes(dmd_modes, 50)
    print(f"Corr tensor shape: {corr.shape}")



    triu_idx = torch.triu_indices(corr.shape[2], corr.shape[3], 1)
    write_dir = "./corr_mats"

    for mode_idx in tqdm(range(0, corr.shape[1], 2)):
        mat = corr[:, mode_idx, triu_idx[0], triu_idx[1]].cpu().numpy()
        np.savetxt(os.path.join(write_dir, f"netmats4_mode{mode_idx}.txt"), mat)
    print(f"Saved correlation matrices in {write_dir}")

    if results["test_dmd_vals"] is not None:
        dmd_vals = results["test_dmd_vals"]
        dmd_vals = dmd_vals.reshape(-1, sample_per_subject, dmd_vals.shape[1], dmd_vals.shape[2]).mean(axis=2).mean(axis=1)

    print(f"DMD vals shape: {dmd_vals.shape}")
    np.savetxt(os.path.join(write_dir, "freq.txt"), dmd_vals)




# class test_compute_correlation_from_modes(unittest.TestCase):
#     def test_shape(self):
#         X = torch.tensor(torch.rand(1, 15, 33, 4425))
#         corr = compute_correlation_from_modes(X, 50)
#         self.assertEqual((1, 33, 50, 50), corr.shape)
#         self.assertEqual((torch.abs(torch.ones((1, 33, 50)) - torch.diagonal(corr, dim1=2, dim2=3)) < 1e-5).all().item(), True)

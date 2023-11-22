from pydmd import FbDMD, DMD, EDMD
from deeptime.kernels import GaussianKernel
from deeptime.decomposition import KernelEDMD
import numpy as np
from torch.utils import data
import math
from CONSTANTS import CONSTANTS
import os

from tqdm import tqdm

from dataloaders.MegaTrawl import MegaTrawlDataset, M4Dataset, CrytosDataset, HCPTaskfMRIDataset

from trainers.metrics import SMAPE, WRMSE

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--write_dir", type=str, default="log/")
parser.add_argument("--num_feats", type=int, default=50)
parser.add_argument("--input_dim", type=int, default=8)
parser.add_argument("--input_length", type=int, default=128)
parser.add_argument("--output_length", type=int, default=32)
parser.add_argument("--stride", type=int, default=1)
parser.add_argument("--dataset", type=str, default="megatrawl")
parser.add_argument("--data_freq", type=str, default="Weekly")

args = parser.parse_args()

print("Hyper-parameters:")
for k, v in vars(args).items():
    print("{} -> {}".format(k, v))

input_length = args.input_length
output_length = args.output_length
mode = "all"
jumps = input_length
d = args.input_dim
T = 2048
sample_per_subject = math.ceil(
    (T - input_length - output_length) / jumps)

data_dict = {
    "M4": M4Dataset,
    "Cryptos": CrytosDataset,
    "Task": HCPTaskfMRIDataset,
    "megatrawl": MegaTrawlDataset
}

metric_dict = {
    "M4": SMAPE,
    "Cryptos": WRMSE,
    "Task": SMAPE,
    "megatrawl": SMAPE
}

data_dir = {
    "M4": "data/M4",
    "Cryptos": "data/Cryptos",
    "Task": "data/task",
    "megatrawl": "data/megatrawl"
}

if args.dataset == "megatrawl":
    test_set = MegaTrawlDataset(
        input_length=input_length,
        output_length=output_length,
        jumps=jumps,
        mode="all")
elif args.dataset == "M4" or args.dataset == "Cryptos":
    dataset = data_dict[args.dataset]
    data_dir = os.path.join(CONSTANTS.CODEDIR, data_dir[args.dataset])
    direc = os.path.join(data_dir, "train.npy")
    direc_test = os.path.join(data_dir, "test.npy")
    test_set = dataset(
        input_length=input_length,
        output_length=output_length,
        freq=args.data_freq,
        direc=direc,
        direc_test=direc_test,
        mode="test")

test_loader = data.DataLoader(
        test_set, batch_size=1, shuffle=False)

inp_preds = []
inp_true = []
eig_vals = []
dmd_modes = []
dmd_vals = []
print(f"Test loader shape: {len(test_loader)}, sample per subject: {sample_per_subject}")

for X, Y in tqdm(test_loader):
    # prepare data
    if len(X.shape) == 2:
        X = X.unsqueeze(-1)
        Y = Y.unsqueeze(-1)
    X = X.unfold(1, d, 1).flatten(start_dim=2)[0].cpu().numpy()
    Y = Y[0].cpu().numpy()

    # fit dmd
    sigma = 1  # kernel bandwidth
    kernel = GaussianKernel(sigma)

    n_eigs = 13  # number of eigenfunctions to be computed
    kedmd_estimator = KernelEDMD(kernel, n_eigs=n_eigs, epsilon=1e-3)
    kedmd_model = kedmd_estimator.fit((X[:-1], X[1:])).fetch_model()


    # predict
    Ypred = []
    Ycur = X[-1:, :]
    for i in range(Y.shape[0]):
        Ynext = kedmd_model.forward(Ycur)
        Ypred.append(Ynext.squeeze().reshape(-1, d)[:, -1:])
        Ycur = Ynext
    Ypred = np.concatenate(Ypred, axis=-1).T
    inp_true.append(Y)
    inp_preds.append(Ypred)

# evaluate prediction
inp_true = np.stack(inp_true, axis=0)
inp_preds = np.stack(inp_preds, axis=0)
if args.dataset == "M4":
    inp_preds = (inp_preds * test_set.ts_stds.reshape(
        -1, 1, 1)) + test_set.ts_means.reshape(-1, 1, 1)
    inp_true = (inp_true * test_set.ts_stds.reshape(
        -1, 1, 1)) + test_set.ts_means.reshape(-1, 1, 1)
# dmd_modes = np.stack(dmd_modes, axis=0)
# dmd_modes = dmd_modes.reshape(-1, sample_per_subject, dmd_modes.shape[1], dmd_modes.shape[2])
# dmd_vals = np.stack(dmd_vals, axis=0)
# dmd_vals = dmd_vals.reshape(-1, sample_per_subject, dmd_vals.shape[1])
eval_score = SMAPE(inp_preds, inp_true)
np.save(f"FbDMD_{args.dataset}_inputlength{args.input_length}_outputlength{args.output_length}_stride1_d{d}_pred.npy", inp_preds)
np.save(f"FbDMD_{args.dataset}_inputlength{args.input_length}_outputlength{args.output_length}_stride1_d{d}_true.npy", inp_true)
# np.save(f"FbDMD_{args.dataset}_inputlength{input_length}_outputlength{output_length}_stride1_d{d}_modes.npy", dmd_modes)
# np.save(f"FbDMD_{args.dataset}_inputlength{input_length}_outputlength{output_length}_stride1_d{d}_freqs.npy", dmd_vals)
print(f"SMAPE score: {eval_score}")

# for eig in eig_vals:
#     print(
#         "Eigenvalue {}: distance from unit circle {}, freq {}".format(
#             eig, np.abs(np.sqrt(eig.imag ** 2 + eig.real ** 2) - 1), np.angle(eig) / 0.72
#         )
#     )
#
#     # plot_eigs(dmd, show_axes=True, show_unit_circle=True)
# plt.hist([np.abs(np.angle(eig)) / 0.72 for eig in eig_vals])
# # plot prediction
# plt.figure(figsize=(10, 100))
# for i in range(50):
#     plt.subplot(50, 1, i+1)
#     plt.plot(Ypred[:, i], c='r')
#     plt.plot(Y[:, i], c='b')
# plt.tight_layout()
# plt.show()

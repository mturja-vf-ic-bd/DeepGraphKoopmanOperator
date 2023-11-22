from pydmd import FbDMD, DMD, EDMD
import numpy as np
from torch.utils import data
import math
from CONSTANTS import CONSTANTS
import os

from tqdm import tqdm

from dataloaders.MegaTrawl import MegaTrawlDataset, M4Dataset, CrytosDataset, HCPTaskfMRIDataset

from trainers.metrics import SMAPE, WRMSE
from baselines.utils import KernelRidgeRegression, polynomial_kernel

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--write_dir", type=str, default="log/")
parser.add_argument("--num_feats", type=int, default=50)
parser.add_argument("--input_dim", type=int, default=8)
parser.add_argument("--input_length", type=int, default=128)
parser.add_argument("--train_output_length", type=int, default=32)
parser.add_argument("--test_output_length", type=int, default=32)
parser.add_argument("--stride", type=int, default=1)
parser.add_argument("--dataset", type=str, default="megatrawl")
parser.add_argument("--data_freq", type=str, default="Weekly")
parser.add_argument("--mode", type=str, default="train")

args = parser.parse_args()

print("Hyper-parameters:")
for k, v in vars(args).items():
    print("{} -> {}".format(k, v))

input_length = args.input_length
mode = "all"
jumps = input_length
d = args.input_dim
T = 2048

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
    train_set = MegaTrawlDataset(
        input_length=input_length,
        output_length=args.train_output_length,
        mode="train",
        jumps=jumps)
    test_set = MegaTrawlDataset(
        input_length=input_length,
        output_length=args.test_output_length,
        jumps=jumps,
        mode="test")
elif args.dataset == "M4" or args.dataset == "Cryptos":
    dataset = data_dict[args.dataset]
    data_dir = os.path.join(CONSTANTS.CODEDIR, data_dir[args.dataset])
    direc = os.path.join(data_dir, "train.npy")
    direc_test = os.path.join(data_dir, "test.npy")
    train_set = dataset(
        input_length=input_length,
        output_length=args.train_output_length,
        freq=args.data_freq,
        direc=direc,
        mode="train",
        jumps=jumps)
    test_set = dataset(
        input_length=input_length,
        output_length=args.test_output_length,
        freq=args.data_freq,
        direc=direc,
        direc_test=direc_test,
        mode="test")

test_loader = data.DataLoader(
        test_set, batch_size=1, shuffle=False)

lambdas = [0.01, 0.1]
K = [0, 1, 2, 3, 4]
hparams = [(x, y) for x in lambdas for y in K]
score = []
min_score = 10000000

for l, k in hparams:
    inp_preds = []
    inp_true = []
    train_loader = data.DataLoader(
        train_set, batch_size=1, shuffle=False)
    for X, Y in tqdm(train_loader):
        # prepare data
        if len(X.shape) == 2:
            X = X.unsqueeze(-1)
            Y = Y.unsqueeze(-1)
        X = X.unfold(1, d, 1).flatten(start_dim=2)[0].cpu().numpy()
        X = polynomial_kernel(X.T, k).T
        Y = Y[0].cpu().numpy()

        # fit dmd
        krr = KernelRidgeRegression(l=l)
        krr.fit(X.T)

        # predict
        Ypred = []
        Ycur = X[-1:, :].T
        for i in range(Y.shape[0]):
            Ynext = krr.predict(Ycur)
            if k > 0:
                Ypred.append(Ynext.squeeze().reshape(-1, k+1)[:, 1].reshape(-1, d)[:, -1:])
            else:
                Ypred.append(Ynext.squeeze().reshape(-1, d)[:, -1:])
            Ycur = Ynext
        Ypred = np.concatenate(Ypred, axis=-1).T
        inp_true.append(Y)
        inp_preds.append(Ypred)

    # evaluate prediction
    inp_true = np.stack(inp_true, axis=0)
    inp_preds = np.stack(inp_preds, axis=0)
    train_score = SMAPE(inp_preds, inp_true)[-1]
    score.append(train_score)
    if min_score > train_score:
        min_score = train_score
        best_hparam = (l, k)

for i in range(len(hparams)):
    print(f"({hparams[i][0]}, {hparams[i][1]}) -> {score[i]}")
print(f"Best hparam: {best_hparam}")
# Test result
inp_preds = []
inp_true = []
l, k = best_hparam
for X, Y in tqdm(test_loader):
    # prepare data
    if len(X.shape) == 2:
        X = X.unsqueeze(-1)
        Y = Y.unsqueeze(-1)
    X = X.unfold(1, d, 1).flatten(start_dim=2)[0].cpu().numpy()
    X = polynomial_kernel(X.T, k).T
    Y = Y[0].cpu().numpy()

    # fit dmd
    krr = KernelRidgeRegression(l=l)
    krr.fit(X.T)

    # predict
    Ypred = []
    Ycur = X[-1:, :].T
    for i in range(Y.shape[0]):
        Ynext = krr.predict(Ycur)
        Ypred.append(Ynext.squeeze().reshape(-1, k+1)[:, 1].reshape(-1, d)[:, -1:])
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
eval_score = SMAPE(inp_preds, inp_true)
print(f"SMAPE Score: {eval_score}")
# np.save(f"kRRDMD_{args.dataset}_inputlength{args.input_length}_outputlength{args.output_length}_stride1_d{d}_pred.npy", inp_preds)
# np.save(f"kRRDMD_{args.dataset}_inputlength{args.input_length}_outputlength{args.output_length}_stride1_d{d}_true.npy", inp_true)
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

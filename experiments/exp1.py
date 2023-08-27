# Test if there is a difference between different input_length in megatrawl dataset
# Check for Train Pred error
import math
import os.path
import torch
from experiments.utils import parse_model_name
from matplotlib import pyplot as plt
import numpy as np

base = f"/Users/mturja/PycharmProjects/DeepGraphKoopmanOperator/megatrawl_results"
files = os.listdir(base)
loss = {}
loss_fn = torch.nn.MSELoss()
T = 2048

for f in files:
    if f.endswith("pth"):
        continue
    params = parse_model_name(f)
    input_dim = params["input_dim"]
    input_length = params["input_length"]
    idx = 0
    print(f)
    if params["input_dim"] == 16 and params["num_steps"] == 32 and params["lr"] == 0.0003:
        res = torch.load(base + "/" + f)
        sample_per_subject = math.ceil((T - params["input_length"] - params["num_steps"]) / params["jumps"])
        feat_dim = 50
        preds = np.concatenate(
            [res["test_preds"].reshape(-1, sample_per_subject, input_dim, feat_dim).reshape(
                -1, sample_per_subject * input_dim, feat_dim)[idx],
             res["test_preds"].reshape(-1, sample_per_subject, input_dim, feat_dim).reshape(
                 -1, sample_per_subject * input_dim, feat_dim)[idx]],
            axis=0)
        tgts = np.concatenate(
            [res["test_tgts"].reshape(-1, sample_per_subject, input_dim, feat_dim).reshape(
                -1, sample_per_subject * input_dim, feat_dim)[idx],
             res["test_tgts"].reshape(-1, sample_per_subject, input_dim, feat_dim).reshape(
                 -1, sample_per_subject * input_dim, feat_dim)[idx]],
            axis=0)

        N = 2
        fig, axs = plt.subplots(N, 1, figsize=(30, 5 * N))
        sub_title_font = {'fontsize': 20, 'fontweight': 'bold'}
        for i in range(0, N):
            plt.subplot(N, 1, i + 1)
            pred = preds[:, i]
            tgt = tgts[:, i]
            axs[i].set_title(f"Node:{i}", fontdict=sub_title_font)
            axs[i].plot(pred, c="r", label="Prediction")
            axs[i].plot(tgt, c="b", label="Ground Truth")
            axs[i].legend(prop={'size': 18}, loc="upper right")

        plt.suptitle(f"Window Length: {input_length}", fontsize=24)
        plt.show()
        if params["input_length"] not in loss.keys():
            loss[params["input_length"]] = [((res["test_preds"] - res["test_tgts"])**2).mean()]
        else:
            loss[params["input_length"]].append(
                ((res["test_preds"] - res["test_tgts"])**2).mean())

        print(loss[params["input_length"]])

x, y = [], []
for k, v in loss.items():
    for i in v:
        x.append(k)
        y.append(i)

y = sorted(y, key=lambda k: x[y.index(k)])
x.sort()
plt.ylim(0.6, 1.2)
plt.xlabel(f"Step Size")
plt.ylabel(f"Prediction Loss")
plt.xticks(x, x)
plt.scatter(x, y)

# Connect consecutive points with lines
for i in range(len(x) - 1):
    plt.plot([x[i], x[i+1]], [y[i], y[i+1]], color="b")
plt.tight_layout()
plt.show()

print(loss)



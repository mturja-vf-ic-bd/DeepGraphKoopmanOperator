# Test if there is a difference between different input_length in megatrawl dataset
# Check for Train Pred error
import math
import os.path
import torch
from experiments.utils import parse_model_name
from matplotlib import pyplot as plt
import numpy as np
from CONSTANTS import CONSTANTS

base = os.path.join(CONSTANTS.CODEDIR, "megatrawl_results")
files = os.listdir(base)
loss = {}
loss_fn = torch.nn.MSELoss()
T = 2048


def compute_cosine_distance(mat):
    print(f"mat shape: {mat.shape}")
    cosine_distance = 1 - np.matmul(mat, mat.swapaxes(1, 2))
    print(cosine_distance.shape)
    return np.mean(cosine_distance) * cosine_distance.shape[1] / (cosine_distance.shape[1] - 1)


for f in files:
    if f.endswith("pth"):
        continue
    params = parse_model_name(f)
    input_dim = params["input_dim"]
    input_length = params["input_length"]
    idx = 0
    print(f)
    if params["input_dim"] == 16 and params["num_steps"] == 32 and params["lr"] == 0.0003 and params["rm"] == 5 and params["decay"] == 0.9:
        if f.endswith("npy"):
            res = np.load(base + "/" + f)

            sample_per_subject = math.ceil((T - params["input_length"] - params["num_steps"]) / params["jumps"])
            dmd_modes = res.reshape(501, -1, 4425, 30)
            print(f"dmd_mode shape: {dmd_modes.shape}")
            cosine_dist = []
            for i in range(dmd_modes.shape[-1]):
                cosine_dist += [compute_cosine_distance(dmd_modes[:, :, :, i])]
            print(cosine_dist)
            print("Mean: ", np.mean(cosine_dist))
        # N = 2
        # fig, axs = plt.subplots(N, 1, figsize=(30, 5 * N))
        # sub_title_font = {'fontsize': 20, 'fontweight': 'bold'}
        # for i in range(0, N):
        #     plt.subplot(N, 1, i + 1)
        #     pred = preds[:, i]
        #     tgt = tgts[:, i]
        #     axs[i].set_title(f"Node:{i}", fontdict=sub_title_font)
        #     axs[i].plot(pred[-640:], c="r", label="Prediction")
        #     axs[i].plot(tgt[-640:], c="b", label="Ground Truth")
        #     axs[i].legend(prop={'size': 18}, loc="upper right")
        #     axs[i].set_xticks(np.arange(0, 700, 32))
        #
        # plt.suptitle(f"Window Length: {input_length}", fontsize=24)
        # plt.savefig(os.path.join(CONSTANTS.CODEDIR, "experiments", "plots",
        #                          f"ts_{params['input_dim']}_{params['input_length']}.png"))
        # if params["input_length"] not in loss.keys():
        #     loss[params["input_length"]] = [((res["test_preds"] - res["test_tgts"])**2).mean()]
        # else:
        #     loss[params["input_length"]].append(
        #         ((res["test_preds"] - res["test_tgts"])**2).mean())

        # print(loss[params["input_length"]])

plt.clf()

x, y = [], []
for k, v in loss.items():
    for i in v:
        x.append(k)
        y.append(i)

y = sorted(y, key=lambda k: x[y.index(k)])
x.sort()
plt.ylim(0.7, 1.1)
plt.xlabel(f"Step Size")
plt.ylabel(f"Prediction Loss")
plt.xticks(x, x)
plt.scatter(x, y)

# Connect consecutive points with lines
for i in range(len(x) - 1):
    plt.plot([x[i], x[i+1]], [y[i], y[i+1]], color="b")
plt.tight_layout()
plt.savefig(os.path.join(CONSTANTS.CODEDIR, "experiments", "plots",
                                 f"loss_progression.png"))

print(loss)



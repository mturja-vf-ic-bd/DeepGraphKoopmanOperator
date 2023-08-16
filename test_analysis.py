import torch
from matplotlib import pyplot as plt
import numpy as np

res = torch.load("./pred/test_pred_stride16_step48_2.pt")
print(res["test_preds"].shape)
print(res["test_tgts"].shape)
idx = 0
input_dim = 16
input_length = 256
T = 2048
jumps = 128
n_win = T // input_length - 1
sample_per_subject = (T - input_length - input_dim) // jumps + 1
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
recon = res["test_recon"].reshape(
    res["test_recon"].shape[0], -1, input_dim, feat_dim).reshape(
    res["test_recon"].shape[0] // sample_per_subject, sample_per_subject, -1,
    input_dim, feat_dim)[idx, 0].reshape(-1, feat_dim)
recon_true = res["test_recon_true"].reshape(
    res["test_recon_true"].shape[0], -1, input_dim, feat_dim).reshape(
    res["test_recon_true"].shape[0] // sample_per_subject, sample_per_subject, -1,
    input_dim, feat_dim)[idx, 0].reshape(-1, feat_dim)

N = 5
fig, axs = plt.subplots(N, 1, figsize=(20, 5*N))
sub_title_font = {'fontsize': 20, 'fontweight': 'bold'}
for i in range(0, N):
    plt.subplot(N, 1, i + 1)
    pred = preds[:, i]
    tgt = tgts[:, i]
    axs[i].set_title(f"Node:{i}", fontdict=sub_title_font)
    axs[i].plot(pred, c="r", label="Prediction")
    axs[i].plot(tgt, c="b", label="Ground Truth")
    axs[i].legend(prop={'size': 18}, loc="upper right")

plt.tight_layout()
plt.show()

fig, axs = plt.subplots(N, 1, figsize=(20, 5*N))
for i in range(0, N):
    pred = recon[:, i]
    tgt = recon_true[:, i]
    axs[i].set_title(f"Node:{i}")
    axs[i].plot(pred, c="r", label="Reconstructed")
    axs[i].plot(tgt, c="b", label="Ground Truth")
    axs[i].legend(prop={'size': 18}, loc="upper right")
plt.tight_layout()
plt.show()

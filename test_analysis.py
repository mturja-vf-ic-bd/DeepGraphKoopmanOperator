import torch
from matplotlib import pyplot as plt
import numpy as np

res = torch.load("./pred/Koopman_megatrawl_seed902_jumps128_poly3_sin5_exp2_lr0.0003_decay1.0_dim8_inp128_pred32_num32__latdim64_rm=1_cm=32_RevINTrue_insnormFalse_globalKFalse_dmdKTrue_localKTrue_contKFalse_stride4_sample_2.pt",
                 map_location=torch.device('cpu'))
dmd_modes = np.load("./pred/Koopman_megatrawl_seed902_jumps128_poly3_sin5_exp2_lr0.0003_decay1.0_dim8_inp128_pred32_num32__latdim64_rm=1_cm=32_RevINTrue_insnormFalse_globalKFalse_dmdKTrue_localKFalse_contKFalse_stride4.npy")
print(res["test_preds"].shape)
print(res["test_tgts"].shape)
idx = 16
input_dim = 8
input_length = 128
T = 2048
jumps = 128
n_win = T // input_length - 1
sample_per_subject = (T - input_length - input_dim) // jumps + 1
feat_dim = 50
n_real = 1
n_complex = 32
preds = res["test_preds"].reshape(-1, sample_per_subject, input_dim, feat_dim).reshape(
    -1, sample_per_subject * input_dim, feat_dim)[idx]
tgts = res["test_tgts"].reshape(-1, sample_per_subject, input_dim, feat_dim).reshape(
        -1, sample_per_subject * input_dim, feat_dim)[idx]
dmd_vals = res["test_dmd_vals"].reshape(-1, sample_per_subject, 31, n_real + n_complex)[idx]
recon = res["test_recon"].reshape(
    res["test_recon"].shape[0], -1, input_dim, feat_dim).reshape(
    res["test_recon"].shape[0] // sample_per_subject, sample_per_subject, -1,
    input_dim, feat_dim)[idx, 0].reshape(-1, feat_dim)
recon_true = res["test_recon_true"].reshape(
    res["test_recon_true"].shape[0], -1, input_dim, feat_dim).reshape(
    res["test_recon_true"].shape[0] // sample_per_subject, sample_per_subject, -1,
    input_dim, feat_dim)[idx, 0].reshape(-1, feat_dim)

N = 10
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

n_real = 1
delta = 1
X = []
Y = []
fig, ax = plt.subplots()
for i in range(n_real, res["test_dmd_vals"].shape[-1], 2):
    scale = np.where(res["test_dmd_vals"][idx, :, i+1] < 0, np.exp(res["test_dmd_vals"][idx, :, i+1]), 1)
    if scale.mean() > 0.95:
        print(f"stable mode: {i}, phi={res['test_dmd_vals'][idx, :, i].mean()}")
    x = (scale * np.cos(res["test_dmd_vals"][idx, :, i] * delta)).reshape(-1)
    x = np.concatenate([x, x], axis=0)
    y = (scale * np.sin(res["test_dmd_vals"][idx, :, i] * delta)).reshape(-1)
    y = np.concatenate([y, -y], axis=0)
    ax.scatter(x, y)
circle = plt.Circle((0, 0), 1.0, fill=False, color='blue')
ax.add_patch(circle)
circle = plt.Circle((0, 0), 0.9, fill=False, color='blue')
ax.add_patch(circle)
circle = plt.Circle((0, 0), 0.8, fill=False, color='blue')
ax.add_patch(circle)
plt.tight_layout()
plt.show()

stable_mode = dmd_modes.reshape((-1, sample_per_subject, dmd_modes.shape[1], n_real + n_complex))[idx]
print(stable_mode.shape)

corr_set = []
dmd_mode_avg = []
for i in range(stable_mode.shape[2]):
    s = stable_mode[:, -1225:, i]
    s /= np.linalg.norm(s, ord=2, axis=-1)[:, np.newaxis]
    corr = np.mean(np.matmul(s, s.T))
    corr_set.append(corr)
    dmd_mode_avg.append(s.mean(axis=0))
dmd_mode_avg = np.stack(dmd_mode_avg, axis=-1)
print(corr_set)
print(dmd_mode_avg.shape)

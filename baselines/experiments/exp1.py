import argparse
import numpy as np
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument("--write_dir", type=str, default="log/")
parser.add_argument("--num_feats", type=int, default=50)
parser.add_argument("--input_dim", type=int, default=4)
parser.add_argument("--input_length", type=int, default=128)
parser.add_argument("--output_length", type=int, default=32)
parser.add_argument("--stride", type=int, default=1)
parser.add_argument("--dataset", type=str, default="megatrawl")
parser.add_argument("--data_freq", type=str, default="Weekly")
args = parser.parse_args()

modes = np.load(f"../FbDMD_{args.dataset}_inputlength{args.input_length}_" \
                f"outputlength{args.output_length}_stride1_d{args.input_dim}_modes.npy")
freq = np.load(f"../FbDMD_{args.dataset}_"
               f"inputlength{args.input_length}_"
               f"outputlength{args.output_length}_"
               f"stride1_d{args.input_dim}_freqs.npy")
print(f"mode shape: ", modes.shape)
print(f"freq shape: ", freq.shape)


def select_dmd_indices(dmdmodes, dmdvals, freq_range, delT=0.72):
    l_freq, r_freq = freq_range
    dmdmodes = dmdmodes.swapaxes(1, 2).reshape(-1, dmdmodes.shape[1])
    dmdvals = dmdvals.reshape(-1)
    dmdvals_cont = np.abs(np.imag(np.log(dmdvals))) / delT
    # Find the indices within the frequency range
    freq_indices = np.where((dmdvals_cont >= l_freq) & (dmdvals_cont <= r_freq))

    # Use the indices to select entries from dmdmodes
    selected_dmdmodes = dmdmodes[freq_indices]
    filtered_repr = np.corrcoef(selected_dmdmodes.T)
    return np.real(filtered_repr)


modes = modes[:, :, 0:50, :]
freq_list = [(0, 0.005), (0.01, 0.05), (0.08, 0.12), (0.2, 0.3)]
all_modes = []

for s in range(modes.shape[0]):
    mode_list = [select_dmd_indices(modes[s], freq[s], freq_list[i]) for i in range(len(freq_list))]
    flat_modes = []
    for i, mode in enumerate(mode_list):
        mode = np.triu(mode, k=1)
        flat_mode = mode[np.where(mode != 0)]
        flat_modes.append(flat_mode)
    all_modes.append(flat_modes)

all_modes = np.array(all_modes)
for i in range(all_modes.shape[1]):
    np.savetxt(f"/Users/mturja/Downloads/HCP_PTN1200/netmats/3T_HCP1200_MSMAll_d50_ts2/netmats_EDMD_{freq_list[i][0]}_{freq_list[i][1]}.txt", all_modes[:, i])
#
# freq = np.imag(np.log(freq)) / 0.72
# freq = freq.reshape(-1)
# freq = freq[freq != 0]
# freq = np.abs(freq)

# from matplotlib import pyplot as plt
#
# plt.hist(freq, bins=1000)
# plt.title("Histograph of the frequency")
# plt.show()


# Assuming mode0, mode1, and mode2 are numpy arrays representing images
# plt.figure(figsize=(20, 5))
# for i in range(len(freq_list)):
#     plt.subplot(1, 4, i+1)
#     plt.imshow((mode_list[i] * 255).astype(int))
#     plt.title(f'Mode {i}')
#
# plt.tight_layout()
# plt.show()


import csv
import math
import os

import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
plt.ion()

from mne.io import read_raw_eeglab


def read_data(filename, channel_prefix):
    raw = read_raw_eeglab(filename)
    ch_names = raw.info['ch_names']
    sel_ch = []
    if isinstance(channel_prefix, list):
        for c in channel_prefix:
            sel_ch += [s for s in ch_names if s.startswith(c)]
    else:
        sel_ch = [s for s in ch_names if s.startswith(channel_prefix)]
    sel_ch.sort()
    data = raw.pick_channels(sel_ch).load_data()
    return data.load_data()


def filter_data(raw, l_freq, h_freq):
    return raw.filter(l_freq=l_freq, h_freq=h_freq)


def plot_spec():
    for prefix in ["'PFC", "'PPC", "'VC", "'LPl"]:
        data = read_data("../../data/lfp_1000fdA.set", prefix)
        fig = data.plot_psd(fmin=2, fmax=20, show=False)
        fig.subplots_adjust(top=0.8)
        fig.suptitle(f'{prefix[1:]}', size='xx-large', weight='bold')
        fig.show()


def plot_raw():
    for prefix in ["'PFC", "'PPC", "'VC", "'LPl"]:
        data = read_data("../../data/lfp_1000fdA.set", prefix)
        data = filter_data(data, 4, 7)
        fig = data.plot(start=698, duration=15, show=False)
        fig.subplots_adjust(top=0.9)
        fig.suptitle(f'{prefix[1:]}', size='xx-large', weight='bold')
        fig.show()


def split_data_by_trial(data, meta_data):
    trial_data = []
    trial_stim = []
    END_IDX = 8000
    START_IDX = 5000
    for index, row in meta_data.iterrows():
        init_idx = math.floor(row["Init"] * 1000)
        if init_idx + END_IDX >= data.shape[1]:
            break
        trial_data.append(data[:, init_idx - START_IDX: init_idx + END_IDX])
        trial_stim.append(
            [START_IDX,
             math.floor((row["StimOnset"] - row["Init"]) * 1000 + START_IDX),
             math.floor((row["Touch"] - row["Init"]) * 1000 + START_IDX)]
        )
    trial_data = np.stack(trial_data, axis=0)
    trial_stim = np.array(trial_stim)
    return trial_data, trial_stim


def save_filtered_data(input_dir, prefix, l_freq, h_freq):
    suffix = ""
    for s in prefix:
        suffix += "_" + s[1:]
    data = read_data(os.path.join(input_dir, "lfp", "lfp_1000fdA.set"), prefix)
    data = filter_data(data, l_freq, h_freq).get_data()
    meta_data = pd.read_csv(os.path.join(input_dir, "sessionMetaBehav_c23.csv"))
    valid_trials = meta_data[meta_data["keep"] == 1]
    trial_data, trial_stim = split_data_by_trial(data, valid_trials)
    np.save(f"{input_dir}/trial_data_{l_freq}_{h_freq}{suffix}_13s.npy", trial_data)
    np.save(f"{input_dir}/trial_stim.npy", trial_stim)


THETA_BAND = (4, 8)
ALPHA_BAND = (8, 12)
BETA_BAND = (12, 30)
GAMMA = (30, 80)
HIGH_GAMMA = (80, 120)


def check_shapes(directory, band, region):
    data = np.load(os.path.join(directory, f"trial_data_{band[0]}_{band[1]}_{region}_13s.npy"))
    return data.shape


def check_all_shapes(directory):
    band = [THETA_BAND, ALPHA_BAND, BETA_BAND, GAMMA, HIGH_GAMMA]
    regions = ["PFC", "PPC", "LPl", "VC"]
    shapes = []
    for b in band:
        for r in regions:
            shapes.append(check_shapes(directory, b, r))
            print(f"{b}-{r}-{shapes[-1]}")
    return shapes


if __name__ == '__main__':
    # data = read_data("../../data/lfp_1000fdA.set", ["'PFC", "'PPC"])
    # data = filter_data(data, 12, 18)
    # data.plot(start=697, duration=10)
    # plot_spec()
    # for file in os.listdir("/Users/mturja/Desktop/ferret_data/"):
    #     if file == ".DS_Store":
    #         continue
    #     for region in ["'PFC", "'PPC", "'LPl", "'VC"]:
    #         save_filtered_data(f"/Users/mturja/Desktop/ferret_data/{file}", [region], 4, 8)
    #         save_filtered_data(f"/Users/mturja/Desktop/ferret_data/{file}", [region], 8, 12)
    #         save_filtered_data(f"/Users/mturja/Desktop/ferret_data/{file}", [region], 12, 30)
    #         save_filtered_data(f"/Users/mturja/Desktop/ferret_data/{file}", [region], 30, 80)
    #         save_filtered_data(f"/Users/mturja/Desktop/ferret_data/{file}", [region], 80, 120)

    shapes = check_all_shapes("/Users/mturja/Desktop/ferret_data/20190305")
    print(shapes)



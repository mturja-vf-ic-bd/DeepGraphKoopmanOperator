import os
import torch
import numpy as np
import re
from numpy import genfromtxt
import pandas as pd


def load_task_fmri_new(root_dir='/home/turja/', normalize=False):
    data_file_ptr = re.compile(r"""TimeSeries(?P<id>\d+)\.csv""")
    label_file_ptr = re.compile(r"""timingLabels_(?P<id>\d+)\.csv""")
    data = {}
    labels = {}
    for f in os.listdir(root_dir):
        match_data = data_file_ptr.match(f)
        match_labels = label_file_ptr.match(f)
        if match_data is not None:
            with open(os.path.join(root_dir, f)) as csv_file:
                signal = genfromtxt(csv_file, delimiter=',')
                if np.isnan(signal).any():
                    continue
                if normalize:
                    signal -= signal.mean()
                    signal /= signal.std()
                data[match_data.group("id")] = torch.DoubleTensor(signal)
        if match_labels is not None:
            with open(os.path.join(root_dir, f)) as csv_file:
                labels[match_labels.group("id")] = torch.LongTensor(genfromtxt(csv_file, delimiter=','))
    return {"data": data, "labels": labels}


def write_data_in_numpy():
    data_dict = load_task_fmri_new(
        root_dir="/Users/mturja/NEWDATA_PAUL/FullData_Oct26/Scan1",
        normalize=True
    )
    data = []
    labels = []
    for k, v in data_dict["data"].items():
        data.append(data_dict["data"][k])
        labels.append(data_dict["labels"][k])

    data = np.stack(data, axis=0)
    labels = np.stack(labels, axis=0)

    import pickle
    pickle.dump(data, open("../../data/data_array.p", "wb"))
    pickle.dump(labels, open("../../data/label_array.p", "wb"))


def get_sees_data():
    sess_data = pd.read_csv("../../data/SessionMetaBehav.csv")
    return sess_data


def get_gt_index():
    sess_data = get_sees_data()
    sess_data = sess_data.loc[sess_data["TrialID"].isin(np.arange(33, 41))]
    base_time = sess_data.iloc[0]["Init"]
    sess_data["Init"] = ((sess_data["Init"] - base_time) * 1000).astype(int)
    sess_data["StimOnset"] = ((sess_data["StimOnset"] - base_time) * 1000).astype(int)
    sess_data["Touch"] = ((sess_data["Touch"] - base_time) * 1000).astype(int)
    return sess_data[["Init", "StimOnset", "Touch"]].to_numpy()


def load_eeg_data(data_dir):
    data_set = []
    import pickle
    for file in sorted(os.listdir(data_dir)):
        if file.endswith("csv"):
            full_path = os.path.join(data_dir, file)
            data_set.append(np.genfromtxt(full_path, delimiter=","))
            pickle.dump(data_set[-1], open(os.path.join("../../data/", file.split(".")[0] + ".p"), "wb"))
    return data_set


def get_data_matrix():
    import pickle
    DATA_DIR = "/Users/mturja/PycharmProjects/KVAE/data"
    lpl = torch.Tensor(pickle.load(open(os.path.join(DATA_DIR, "LPl_Data.p"), "rb")))
    pfc = torch.Tensor(pickle.load(open(os.path.join(DATA_DIR, "PFC_Data.p"), "rb")))
    ppc = torch.Tensor(pickle.load(open(os.path.join(DATA_DIR, "PPC_Data.p"), "rb")))
    vc = torch.Tensor(pickle.load(open(os.path.join(DATA_DIR, "VC_Data.p"), "rb")))
    data = torch.cat([lpl, pfc, ppc, vc], dim=1)
    return data


if __name__ == '__main__':
    import plotly.express as px
    event_indices = get_gt_index()
    dhruv_data = load_eeg_data("/Users/mturja/Dropbox/EEGData/Unfiltered")
    # dhruv_data
    from matplotlib import pyplot as plt

    plt.figure(figsize=(1000, 4))
    df = np.zeros((dhruv_data[0].shape[0], len(dhruv_data)))
    for i, d in enumerate(dhruv_data):
        d = d.mean(axis=1)
        d -= d.mean()
        d /= d.std()
        df[:, i] = d
        # plt.plot(d)
        # for j in range(event_indices.shape[1]):
        #     plt.vlines(x=event_indices[:, j], ymin=d.min(), ymax=d.max())
        # plt.savefig("fig_{}.png".format(str(i)))
    df = pd.DataFrame(data=df, columns=["LPl", "PFC", "PPC", "VC"])
    fig = px.line(df, x=df.index, y=df.VC)
    for i in range(event_indices.shape[0]):
        fig.add_vrect(x0=event_indices[i, 1], x1=event_indices[i, 1] + 100,
                      fillcolor="green", opacity=0.25, line_width=0, annotation_text="Stimulate")
    for i in range(event_indices.shape[0]):
        fig.add_vrect(x0=event_indices[i, 0], x1=event_indices[i, 0] + 100,
                      fillcolor="red", opacity=0.25, annotation_text="Init",
                      line_width=0)
    for i in range(event_indices.shape[0]):
        fig.add_vrect(x0=event_indices[i, 2], x1=event_indices[i, 2] + 100,
                      fillcolor="orange", opacity=0.25, annotation_text="Touch",
                      line_width=0)
    fig.show()



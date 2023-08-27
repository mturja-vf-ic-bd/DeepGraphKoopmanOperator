import unittest
from abc import ABC

import pytorch_lightning as pl
import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import pickle
from torch.utils.data import DataLoader, random_split, \
    TensorDataset, Dataset, DistributedSampler

from CONSTANTS import CONSTANTS


class TaskFMRIDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 2, val_size=0.2, test_size=0.2, seed=42, datapath=None):
        super(TaskFMRIDataModule, self).__init__()
        fmri_signal = pickle.load(open(os.path.join(datapath, "../data/HCP_task.p"), "rb"))
        fmri_signal = (fmri_signal - fmri_signal.mean(axis=-1)[:, :, np.newaxis]) / fmri_signal.std(axis=-1)[:, :,
                                                                                    np.newaxis]
        dataset = TensorDataset(torch.from_numpy(fmri_signal))
        length = [len(dataset) - int(len(dataset) * val_size) - int(len(dataset) * test_size),
                  int(len(dataset) * val_size),
                  int(len(dataset) * test_size)]
        self.train, self.val, self.test = random_split(
            dataset, length, generator=torch.Generator().manual_seed(seed))
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)


class MegaTrawlDataModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size: int = 2,
                 val_size=0.2,
                 test_size=0.2,
                 seed=42,
                 parcel=50):
        super(MegaTrawlDataModule, self).__init__()
        subjectIDs = np.loadtxt(os.path.join(CONSTANTS.HOME, "subjectIDs.txt"))
        fmri_signal = np.zeros((len(subjectIDs), parcel, 1200), dtype=float)
        for i, s in enumerate(subjectIDs):
            fmri_signal[i] = np.loadtxt(
                os.path.join(CONSTANTS.HOME,
                             "node_timeseries",
                             f"3T_HCP1200_MSMAll_d{parcel}_ts2",
                             f"{int(s)}.txt"))[:1200].T
        fmri_signal = (fmri_signal - fmri_signal.mean(axis=-1)[:, :, np.newaxis]) \
                      / fmri_signal.std(axis=-1)[:, :, np.newaxis]
        dataset = TensorDataset(torch.from_numpy(fmri_signal).float(), torch.LongTensor(subjectIDs))
        length = [len(dataset) - int(len(dataset) * val_size) - int(len(dataset) * test_size),
                  int(len(dataset) * val_size),
                  int(len(dataset) * test_size)]
        self.train, self.val, self.test = random_split(dataset, length,
                                                       generator=torch.Generator().manual_seed(seed))
        self.batch_size = batch_size
        self.dataset = dataset

    def train_dataloader(self):
        return DataLoader(dataset=self.train, batch_size=self.batch_size, shuffle=False,
                          sampler=DistributedSampler(self.train))

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=1000)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)


class MegaTrawlDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            input_length,  # num of input steps
            output_length,  # forecasting horizon
            mode="train",  # train, validation or test
            jumps=1,  # The number of skipped steps when generating samples
            parcel=50
    ):

        self.input_length = input_length
        self.output_length = output_length
        self.mode = mode

        subjectIDs = np.loadtxt(
            os.path.join(CONSTANTS.HOME, "subjectIDs.txt"))
        train_subID, test_subID = \
            train_test_split(
                subjectIDs,
                test_size=0.5,
                random_state=1,
                shuffle=True)
        if mode == "train" or "valid":
            ids = train_subID
        else:
            ids = test_subID
        T = 2048
        fmri_signal = np.zeros((len(ids), T, parcel), dtype=float)
        for i, s in enumerate(ids):
            fmri_signal[i] = np.loadtxt(
                os.path.join(CONSTANTS.HOME,
                             "node_timeseries",
                             f"3T_HCP1200_MSMAll_d{parcel}_ts2",
                             f"{int(s)}.txt"))[:T]
            if np.isnan(fmri_signal[i]).any():
                print(f"Subject {s} has NaN values")
        self.data_lsts = (fmri_signal - fmri_signal.mean(axis=1)[:, np.newaxis]) \
                         / fmri_signal.std(axis=-1)[:, :, np.newaxis]

        self.ts_indices = []
        for i, item in enumerate(self.data_lsts):
            for j in range(0,
                           len(item) - input_length - output_length,
                           jumps):
                self.ts_indices.append((i, j))

        if mode == "train" or "valid":
            np.random.seed(123)
            idx = np.arange(0, len(self.data_lsts))
            np.random.shuffle(idx)
            train_valid_split = int(len(idx) * 0.9)
            
            # np.random.shuffle(self.ts_indices)

            # 90%-10% train-validation split
            train_valid_split = int(len(self.ts_indices) * 0.9)
            if mode == "train":
                self.ts_indices = self.ts_indices[:train_valid_split]
            elif mode == "valid":
                self.ts_indices = self.ts_indices[train_valid_split:]
        print(f"{mode}-data count: {len(self.ts_indices)}")

    def __len__(self):
        return len(self.ts_indices)

    def __getitem__(self, index):
        i, j = self.ts_indices[index]
        x = self.data_lsts[i][j:j + self.input_length]
        y = self.data_lsts[i][j + self.input_length:j + self.input_length +
                                                    self.output_length]
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()


class HCPTaskfMRIDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            input_length,  # num of input steps
            output_length,  # forecasting horizon
            mode="train",  # train, validation or test
            jumps=1,  # The number of skipped steps when generating samples
            datapath=None
    ):
        super(HCPTaskfMRIDataset, self).__init__()
        self.datapath = datapath
        self.input_length = input_length
        self.output_length = output_length
        self.mode = mode
        self.jumps = jumps
        data = self.load_task_fmri(os.path.join(datapath, "NEWDATA_PAUL/FullData_Oct26", "Scan1"))
        subjectIDs = list(data["signal"].keys())
        train_subID, test_subID = \
            train_test_split(
                subjectIDs,
                test_size=0.5,
                random_state=1,
                shuffle=True)
        if mode == "train" or "valid":
            ids = train_subID
        else:
            ids = test_subID

        fmri_signal = np.zeros((len(ids), 393, 268), dtype=float)
        for i, s in enumerate(ids):
            fmri_signal[i] = data["signal"][s]
            if i == 0:
                self.labels = data["labels"][s]
        self.data_lsts = (fmri_signal - fmri_signal.mean(axis=1)[:, np.newaxis]) \
                         / fmri_signal.std(axis=-1)[:, :, np.newaxis]

        sel_idx = self.pick_state([2, 4, 7, 9, 11, 13, 15, 17])
        sel_nodes = get_DMN_Attn_ind()
        self.data_lsts = self.data_lsts[:, sel_idx]
        self.data_lsts = self.data_lsts[:, :, sel_nodes]
        self.ts_indices = []
        for i, item in enumerate(self.data_lsts):
            for j in range(0,
                           len(item) - input_length - output_length,
                           jumps):
                self.ts_indices.append((i, j))

        if mode == "train" or "valid":
            np.random.seed(1)
            np.random.shuffle(self.ts_indices)

            # 90%-10% train-validation split
            train_valid_split = int(len(self.ts_indices) * 0.9)
            if mode == "train":
                self.ts_indices = self.ts_indices[:train_valid_split]
            elif mode == "valid":
                self.ts_indices = self.ts_indices[train_valid_split:]
        print(f"{mode}-data count: {len(self.ts_indices)}")

    def load_task_fmri(self, datapath):
        import re
        from numpy import genfromtxt

        data_file_ptr = re.compile(r"""TimeSeries(?P<id>\d+)\.csv""")
        label_file_ptr = re.compile(r"""timingLabels_(?P<id>\d+)\.csv""")
        data = {}
        labels = {}
        for f in os.listdir(datapath):
            match_data = data_file_ptr.match(f)
            match_labels = label_file_ptr.match(f)
            if match_data is not None:
                with open(os.path.join(datapath, f)) as csv_file:
                    signal = genfromtxt(csv_file, delimiter=',').T
                    if np.isnan(signal).any():
                        print(f"{f} has NaN values")
                        continue
                    data[match_data.group("id")] = signal
            elif match_labels is not None:
                with open(os.path.join(datapath, f)) as csv_file:
                    labels[match_labels.group("id")] = genfromtxt(csv_file, delimiter=',')
            else:
                print("Skipping file: {}".format(f))
        return {"signal": data, "labels": labels}

    def pick_state(self, keepstate):
        idx = []
        for i, s in enumerate(keepstate):
            if i == 0:
                idx = (self.labels == s).nonzero()[0]
            else:
                idx = np.concatenate([idx, (self.labels == s).nonzero()[0]], axis=0)
        idx = np.sort(idx)
        return idx

    def __len__(self):
        return len(self.ts_indices)

    def __getitem__(self, index):
        i, j = self.ts_indices[index]
        x = self.data_lsts[i][j:j + self.input_length]
        y = self.data_lsts[i][j + self.input_length:j + self.input_length +
                                                    self.output_length]
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()


def get_DMN_ind():
    import csv
    with open(os.path.join(CONSTANTS.DATADIR, "NEWDATA_PAUL/FullData_Oct26", "DMN_ROI.csv")) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        node = 0
        ind = []
        for row in csv_reader:
            flag = int(row[0].strip())
            if flag == 1:
                ind.append(node)
            node = node + 1
    print(len(ind))
    return ind


def get_Attn_ind():
    import csv
    with open(os.path.join(CONSTANTS.DATADIR, 'NEWDATA_PAUL/FullData_Oct26', "AttentionROI.csv")) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        node = 0
        ind = []
        for row in csv_reader:
            flag = int(row[0].strip())
            if flag == 1:
                ind.append(node)
            node = node + 1
    print(len(ind))
    return ind


def get_DMN_Attn_ind():
    ind_dmn = get_DMN_ind()
    ind_attn = get_Attn_ind()
    return ind_dmn + ind_attn


class testMegaTrawlDataset(unittest.TestCase):
    def testMegaTrawlDatashape(self):
        dataset = MegaTrawlDataset(
            input_length=128, output_length=32,
            mode="train", jumps=128)
        dataloaders = DataLoader(dataset, batch_size=2, shuffle=True)
        src, tgt = next(iter(dataloaders))
        self.assertEqual(src.shape, (2, 128, 50))
        self.assertEqual(tgt.shape, (2, 32, 50))


class testHCPTaskfMRIDataset(unittest.TestCase):
    def testDataShape(self):
        datapath = "/Users/mturja/Downloads/"
        dataset = HCPTaskfMRIDataset(
            input_length=24,
            output_length=12,
            mode="train",
            jumps=36,
            datapath=datapath
        )
        dataloaders = DataLoader(dataset, batch_size=2, shuffle=True)
        src, tgt = next(iter(dataloaders))
        self.assertEqual(src.shape, (2, 24, 268))
        self.assertEqual(tgt.shape, (2, 12, 268))

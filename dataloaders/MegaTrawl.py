import unittest
from abc import ABC

import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pickle

from CONSTANTS import CONSTANTS


# class TaskFMRIDataModule(pl.LightningDataModule):
#     def __init__(self, batch_size: int = 2, val_size=0.2, test_size=0.2, seed=42, datapath=None):
#         super(TaskFMRIDataModule, self).__init__()
#         fmri_signal = pickle.load(open(os.path.join(datapath, "../data/HCP_task.p"), "rb"))
#         fmri_signal = (fmri_signal - fmri_signal.mean(axis=-1)[:, :, np.newaxis]) / fmri_signal.std(axis=-1)[:, :,
#                                                                                     np.newaxis]
#         dataset = TensorDataset(torch.from_numpy(fmri_signal))
#         length = [len(dataset) - int(len(dataset) * val_size) - int(len(dataset) * test_size),
#                   int(len(dataset) * val_size),
#                   int(len(dataset) * test_size)]
#         self.train, self.val, self.test = random_split(
#             dataset, length, generator=torch.Generator().manual_seed(seed))
#         self.batch_size = batch_size
#
#     def train_dataloader(self):
#         return DataLoader(self.train, batch_size=self.batch_size)
#
#     def val_dataloader(self):
#         return DataLoader(self.val, batch_size=self.batch_size)
#
#     def test_dataloader(self):
#         return DataLoader(self.test, batch_size=self.batch_size)
#
#     def predict_dataloader(self):
#         return DataLoader(self.test, batch_size=self.batch_size)
#
#
# class MegaTrawlDataModule(pl.LightningDataModule):
#     def __init__(self,
#                  batch_size: int = 2,
#                  val_size=0.2,
#                  test_size=0.2,
#                  seed=42,
#                  parcel=50):
#         super(MegaTrawlDataModule, self).__init__()
#         subjectIDs = np.loadtxt(os.path.join(CONSTANTS.HOME, "subjectIDs.txt"))
#         fmri_signal = np.zeros((len(subjectIDs), parcel, 1200), dtype=float)
#         for i, s in enumerate(subjectIDs):
#             fmri_signal[i] = np.loadtxt(
#                 os.path.join(CONSTANTS.HOME,
#                              "node_timeseries",
#                              f"3T_HCP1200_MSMAll_d{parcel}_ts2",
#                              f"{int(s)}.txt"))[:1200].T
#         fmri_signal = (fmri_signal - fmri_signal.mean(axis=-1)[:, :, np.newaxis]) \
#                       / fmri_signal.std(axis=-1)[:, :, np.newaxis]
#         dataset = TensorDataset(torch.from_numpy(fmri_signal).float(), torch.LongTensor(subjectIDs))
#         length = [len(dataset) - int(len(dataset) * val_size) - int(len(dataset) * test_size),
#                   int(len(dataset) * val_size),
#                   int(len(dataset) * test_size)]
#         self.train, self.val, self.test = random_split(dataset, length,
#                                                        generator=torch.Generator().manual_seed(seed))
#         self.batch_size = batch_size
#         self.dataset = dataset
#
#     def train_dataloader(self):
#         return DataLoader(dataset=self.train, batch_size=self.batch_size, shuffle=False,
#                           sampler=DistributedSampler(self.train))
#
#     def val_dataloader(self):
#         return DataLoader(self.val, batch_size=1000)
#
#     def test_dataloader(self):
#         return DataLoader(self.test, batch_size=self.batch_size)
#
#     def predict_dataloader(self):
#         return DataLoader(self.dataset, batch_size=self.batch_size)


class M4Dataset(torch.utils.data.Dataset):
  """Dataset class for M4 dataset."""

  def __init__(
      self,
      input_length,  # num of input steps
      output_length,  # forecasting horizon
      freq,  # The frequency of time series
      direc,  # path to numpy data files
      direc_test=None,  # for testing mode, we also need to load test data.
      mode="train",  # train, validation or test
      jumps=1  # The number of skipped steps when generating sliced samples
  ):

    self.input_length = input_length
    self.output_length = output_length
    self.mode = mode

    # Load training set
    self.train_data = np.load(direc, allow_pickle=True)
    self.data_lsts = self.train_data.item().get(freq)

    # First do global standardization
    self.ts_means, self.ts_stds = [], []
    for i, item in enumerate(self.data_lsts):
      if np.isnan(item).any():
        print(f"Subject {i} has NaN values")
      avg, std = np.mean(item), np.std(item)
      self.ts_means.append(avg)
      self.ts_stds.append(std)
      self.data_lsts[i] = (self.data_lsts[i] - avg) / std

    self.ts_means = np.array(self.ts_means)
    self.ts_stds = np.array(self.ts_stds)

    if mode == "test":
      self.test_lsts = np.load(direc_test, allow_pickle=True).item().get(freq)
      for i, item in enumerate(self.test_lsts):
        self.test_lsts[i] = (item - self.ts_means[i]) / self.ts_stds[i]
      self.ts_indices = list(range(len(self.test_lsts)))

    elif mode == "train" or mode == "valid":
      # shuffle slices before split
      np.random.seed(123)
      self.ts_indices = []
      for i, item in enumerate(self.data_lsts):
        for j in range(0, len(item)-input_length-output_length, jumps):
          self.ts_indices.append((i, j))
      np.random.shuffle(self.ts_indices)

      # 90%-10% train-validation split
      train_valid_split = int(len(self.ts_indices) * 0.9)
      if mode == "train":
        self.ts_indices = self.ts_indices[:train_valid_split]
      elif mode == "valid":
        self.ts_indices = self.ts_indices[train_valid_split:]
    else:
      raise ValueError("Mode can only be one of train, valid, test")

  def __len__(self):
    return len(self.ts_indices)

  def __getitem__(self, index):
    if self.mode == "test":
      x = self.data_lsts[index][-self.input_length:]
      y = self.test_lsts[index]

    else:
      i, j = self.ts_indices[index]
      x = self.data_lsts[i][j:j + self.input_length]
      y = self.data_lsts[i][j + self.input_length:j + self.input_length +
                            self.output_length]

    return torch.from_numpy(x).float(), torch.from_numpy(y).float()


class MegaTrawlDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            input_length,  # num of input steps
            output_length,  # forecasting horizon
            mode="train",  # train, validation or test
            jumps=1,  # The number of skipped steps when generating samples
            parcel=50,
            sample_per_subject=1
    ):

        self.input_length = input_length
        self.output_length = output_length
        self.mode = mode
        self.sample_per_subject = sample_per_subject

        # required for segment_generator
        self.subject_index = 0
        self.cur_sample = 0

        subjectIDs = np.loadtxt(
            os.path.join(CONSTANTS.HOME, "subjectIDs.txt"))
        train_subID, test_subID = \
            train_test_split(
                subjectIDs,
                test_size=0.5,
                random_state=1,
                shuffle=True)
        if mode == "train" or mode == "valid":
            ids = train_subID
        elif mode == "all":
            ids = subjectIDs
        else:
            ids = test_subID
        self.ids = ids
        print(f"Number of subjects: {len(ids)}")
        T = 2048
        self.total_length = T
        fmri_signal = np.zeros((len(ids), T, parcel), dtype=float)
        self.regressors = np.zeros((len(ids), 4))
        count = 0
        for i, s in enumerate(ids):
            signal = np.loadtxt(
                os.path.join(CONSTANTS.HOME,
                             "node_timeseries",
                             f"3T_HCP1200_MSMAll_d{parcel}_ts2",
                             f"{int(s)}.txt"))[:T]
            if np.isnan(signal).any():
                print(f"Subject {s} has NaN values. Skipping")
                continue

            fmri_signal[count] = signal
            count += 1

        print(f"Total subjects: {count}")
        fmri_signal = fmri_signal[0:count]
        self.total_subject = count
        self.mode = mode
        self.regressors = self.regressors[0:count]
        self.data_lsts = (fmri_signal - fmri_signal.mean(axis=1)[:, np.newaxis]) \
                         / fmri_signal.std(axis=1)[:, np.newaxis]

        self.ts_indices = []
        for i, item in enumerate(self.data_lsts):
            for j in range(0,
                           len(item) - input_length - output_length,
                           jumps):
                self.ts_indices.append((i, j))

        if mode == "train" or mode == "valid":
            np.random.seed(123)
            idx = np.arange(0, len(self.data_lsts))
            np.random.shuffle(idx)
            train_valid_split = int(len(idx) * 0.9)

            # 90%-10% train-validation split
            # train_valid_split = int(len(self.ts_indices) * 0.9)
            sample_per_sub = len(self.ts_indices) // len(idx)
            ts_indices = []
            if mode == "train":
                for ind in idx[:train_valid_split]:
                    ts_indices += self.ts_indices[ind * sample_per_sub: (ind + 1) * sample_per_sub]
                self.ts_indices = ts_indices
            elif mode == "valid":
                for ind in idx[train_valid_split:]:
                    ts_indices += self.ts_indices[ind * sample_per_sub: (ind + 1) * sample_per_sub]
                self.ts_indices = ts_indices
            np.random.shuffle(self.ts_indices)
        print(f"{mode}-data count: {len(self.ts_indices)}")

    def segment_generator(self):
        # Initialize counters
        while True:
            i = self.subject_index
            j = np.random.randint(0, self.total_length - self.input_length - self.output_length)
            self.cur_sample += 1
            # print(f"{i},{j}")

            # If we've accumulated the specified number of segments for this subject in the current epoch, move to the next subject and reset the epoch_index
            if self.cur_sample >= self.sample_per_subject:
                self.subject_index += 1
                self.cur_sample = 0

                # If we've reached the end of the subjects, wrap around to the first subject
                if self.subject_index >= self.total_subject:
                    self.subject_index = 0

            yield i, j

    def __len__(self):
        return len(self.ts_indices)
        # if self.mode == "test":
        #     return len(self.ts_indices)
        # else:
        #     return self.total_subject * self.sample_per_subject

    def __getitem__(self, index):
        i, j = self.ts_indices[index]
        # if self.mode == "test":
        #     i, j = self.ts_indices[index]
        # else:
        # i, j = next(self.segment_generator())
        x = self.data_lsts[i][j:j + self.input_length]
        y = self.data_lsts[i][j + self.input_length:j + self.input_length +
                                                    self.output_length]

        return torch.from_numpy(x).float(), torch.from_numpy(y).float()


class CrytosDataset(torch.utils.data.Dataset):
  """Dataset class for Cryptos data."""

  def __init__(
      self,
      input_length,  # num of input steps
      output_length,  # forecasting horizon
      direc,  # path to numpy data files
      direc_test=None,  # # for testing mode, we also need to load test data.
      mode="train",  # train, validation or test
      jumps=10,  # number of skipped steps between two sliding windows
      freq=None):

    self.input_length = input_length
    self.output_length = output_length
    self.mode = mode

    # First do global standardization
    # Load training set
    self.train_data = np.load(direc, allow_pickle=True)

    # First do global standardization
    self.ts_means, self.ts_stds = [], []
    for i, item in enumerate(self.train_data):
        avg, std = np.mean(
            item, axis=0, keepdims=True), np.std(
            item, axis=0, keepdims=True)
        self.ts_means.append(avg)
        self.ts_stds.append(std)
        self.train_data[i] = (self.train_data[i] - avg) / std

    self.ts_means = np.concatenate(self.ts_means, axis=0)
    self.ts_stds = np.concatenate(self.ts_stds, axis=0)

    if mode == "test":
      self.test_lsts = np.load(direc_test, allow_pickle=True)
      for i, item in enumerate(self.test_lsts):
        self.test_lsts[i] = (self.test_lsts[i] -
                             self.ts_means[i]) / self.ts_stds[i]

      # change the input length (< 100) will not affect the target output
      self.ts_indices = []
      for i, item in enumerate(self.test_lsts):
        for j in range(100, len(item) - output_length, output_length):
          self.ts_indices.append((i, j))

    elif mode == "train" or mode == "valid":
      # shuffle slices before split
      np.random.seed(123)
      self.ts_indices = []
      for i, item in enumerate(self.train_data):
        for j in range(0, len(item)-input_length-output_length, jumps):
          self.ts_indices.append((i, j))

      np.random.shuffle(self.ts_indices)

      # 90%-10% train-validation split
      train_valid_split = int(len(self.ts_indices) * 0.9)
      if mode == "train":
        self.ts_indices = self.ts_indices[:train_valid_split]
      elif mode == "valid":
        self.ts_indices = self.ts_indices[train_valid_split:]
    else:
      raise ValueError("Mode can only be one of train, valid, test")
    print(f"Cryptos Data Loaded!")

  def __len__(self):
    return len(self.ts_indices)

  def __getitem__(self, index):
    if self.mode == "test":
      i, j = self.ts_indices[index]
      x = self.test_lsts[i][j - self.input_length:j]
      y = self.test_lsts[i][j:j + self.output_length]
    else:
      i, j = self.ts_indices[index]
      x = self.train_data[i][j:j + self.input_length]
      y = self.train_data[i][j + self.input_length:j + self.input_length +
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
            mode="train", jumps=128, sample_per_subject=1)
        dataloaders = DataLoader(dataset, batch_size=4, shuffle=False, drop_last=True)
        for src, tgt in iter(dataloaders):
            self.assertEqual(src.shape, (4, 128, 50))
            self.assertEqual(tgt.shape, (4, 32, 50))

        correlation = []
        w = 16
        p = 28
        q = 29
        for i in range(w//2, 256-w//2 + 1, 1):
            a = np.corrcoef(src[0, i-w//2:i+w//2, p], src[0, i-w//2:i+w//2, q])[0, 1]
            correlation.append(a)
        from matplotlib import pyplot as plt
        plt.figure(figsize=(60, 20))
        plt.subplot(2, 1, 1)
        plt.xticks([])
        plt.yticks([])

        # plt.plot(src[0, w//2:-w//2, p], c='b', linewidth=10)
        plt.plot(src[0, w//2:-w//2, q], c='r', linewidth=10)
        plt.subplot(2, 1, 2)
        plt.xticks([])
        plt.yticks([])
        plt.plot(correlation, linewidth=4)
        plt.tight_layout()
        plt.show()


class testCryptosDataset(unittest.TestCase):
    def testCryptosDatashape(self):
        dataset = CrytosDataset(
            input_length=128, output_length=32, direc=f"{CONSTANTS.CODEDIR}/data/Cryptos/train.npy",
            mode="train", jumps=128)
        dataloaders = DataLoader(dataset, batch_size=4, shuffle=False, drop_last=True)
        for src, tgt in iter(dataloaders):
            self.assertEqual(src.shape, (4, 128, 8))
            self.assertEqual(tgt.shape, (4, 32, 8))


class EEGDataset(torch.utils.data.Dataset):
    def __init__(self,
                 input_length,  # num of input steps
                 output_length,  # forecasting horizon
                 mode="train",  # train, validation or test
                 jumps=1,  # The number of skipped steps when generating samples
                 datapath=None):
        super().__init__()
        self.input_length = input_length
        self.output_length = output_length
        self.mode = mode
        signal_lpl = torch.Tensor(
            pickle.load(open(os.path.join(datapath, "LPl_Data.p"), "rb"))
        ).mean(dim=1, keepdim=True)
        signal_pfc = torch.Tensor(
            pickle.load(open(os.path.join(datapath, "PFC_Data.p"), "rb"))
        ).mean(dim=1, keepdim=True)
        signal_ppc = torch.Tensor(
            pickle.load(open(os.path.join(datapath, "PPC_Data.p"), "rb"))
        ).mean(dim=1, keepdim=True)
        signal_vc = torch.Tensor(
            pickle.load(open(os.path.join(datapath, "VC_Data.p"), "rb"))
        ).mean(dim=1, keepdim=True)
        signal = torch.cat([signal_vc, signal_ppc, signal_pfc, signal_lpl], dim=1)
        signal = signal[0: -(signal.shape[0] % 128), :]
        signal = (signal - torch.min(signal, dim=0, keepdim=True)[0]) / \
                 (torch.max(signal, dim=0, keepdim=True)[0] - torch.min(signal, dim=0, keepdim=True)[0])
        self.data_lsts = signal.unsqueeze(0)
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

    def __len__(self):
        return len(self.ts_indices)

    def __getitem__(self, index):
        i, j = self.ts_indices[index]
        x = self.data_lsts[i][j:j + self.input_length]
        y = self.data_lsts[i][j + self.input_length:j + self.input_length +
                                                    self.output_length]
        return x, y


class testEEG(unittest.TestCase):
    def testDataShape(self):
        datapath = "/Users/mturja/PycharmProjects/KVAE/data/EEG"
        dataset = EEGDataset(
            input_length=512,
            output_length=64,
            mode="train",
            jumps=256,
            datapath=datapath
        )
        dataloaders = DataLoader(dataset, batch_size=2, shuffle=True)

        src, tgt = next(iter(dataloaders))
        self.assertEqual(src.shape, (2, 512, 4))
        self.assertEqual(tgt.shape, (2, 64, 4))

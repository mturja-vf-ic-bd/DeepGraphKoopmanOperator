import unittest

import pytorch_lightning as pl
import os
import numpy as np
import scipy.linalg
import torch
from scipy.integrate import odeint
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, random_split, \
    TensorDataset, Dataset, DistributedSampler

from CONSTANTS import CONSTANTS


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
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()


class testMegaTrawlDataset(unittest.TestCase):
    def testMegaTrawlDatashape(self):
        dataset = MegaTrawlDataset(
            input_length=128, output_length=32,
            mode="train", jumps=128)
        dataloaders = DataLoader(dataset, batch_size=2, shuffle=True)
        src, tgt = next(iter(dataloaders))
        self.assertEqual(src.shape, (2, 128, 50))
        self.assertEqual(tgt.shape, (2, 32, 50))
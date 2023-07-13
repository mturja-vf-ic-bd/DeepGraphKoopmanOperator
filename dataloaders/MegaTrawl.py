import pytorch_lightning as pl
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, TensorDataset

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
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=1000)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)
import plotly.express as px
import numpy as np
import pickle
import os
import torch

from src.utils.helper import pearson2
from src.utils.load_data import get_gt_index, load_eeg_data, get_data_matrix


def sliding_window_corr(w=100, stride=1, start_idx=0, end_idx=10000):
    data = get_data_matrix()[start_idx:end_idx]
    data = data.unfold(0, w, stride)
    _corr = pearson2(data)
    return _corr


if __name__ == '__main__':
    import plotly.express as px
    from matplotlib import pyplot as plt
    init_indices = get_gt_index()[:, 0]
    stim_idx = get_gt_index()[:, 1]
    torch_idx = get_gt_index()[:, 2]
    print(stim_idx/1000)
    print(torch_idx/1000)
    _corr = sliding_window_corr(w=1000, stride=1000,
                                start_idx=init_indices[1],
                                end_idx=init_indices[2])
    plt.figure(figsize=(50, 5))
    for i in range(_corr.shape[0]):
        plt.subplot(1, _corr.shape[0], i + 1)
        plt.imshow(_corr[i])
    plt.tight_layout()
    plt.show()




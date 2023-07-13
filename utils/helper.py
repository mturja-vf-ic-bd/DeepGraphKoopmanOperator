import numpy as np


def get_top_links(connectome, count=1, offset=0, weight=False):
    connectome = np.array(connectome)
    row, col = connectome.shape
    idx = np.argsort(connectome, axis=None)[::-1]
    idx_row = idx // col
    idx_col = idx % col
    if not weight:
        idx_coord = list(zip(idx_row, idx_col))
    else:
        idx_coord = list(zip(idx_row, idx_col, connectome[idx_row, idx_col]))

    return idx_coord[offset:count + offset]


def get_pos_neg_pair(count):
    comb = np.loadtxt('comb_func_inv.txt')
    # print((comb > .8).sum())
    # top_link = np.argsort(comb, axis=1)[:, ::-1]
    # print(np.take_along_axis(comb, top_link, axis=1))
    # print(top_link)
    top_link = get_top_links(comb, count=count)
    np.fill_diagonal(comb, 100000)
    bottom_link = get_top_links(-comb, count=count)

    return top_link, bottom_link


import torch


def inner_diff(X):
    X_sq = (X**2).sum(dim=-2, keepdim=True)
    X_sq = X_sq + X_sq.transpose(-2, -1)
    return X_sq - 2 * torch.matmul(X.transpose(-2, -1), X)


def pearson(X):
    m = torch.mean(X, dim=-1, keepdim=True)
    s = torch.std(X, dim=-1, keepdim=True)
    X_norm = (X - m) / s
    corr = torch.matmul(X_norm, X_norm.transpose(-2, -1)) / (X.shape[-1] - 1)
    return corr


def get_pos_neg_pair_(x, count, method='dtw'):
    if method == 'dtw':
        adj = dtw_corr(x.T)
    elif method == 'ecl':
        adj = inner_diff(x.T)
    # adj /= adj.sum(axis=1)[:, np.newaxis]
    ind_neg = np.array(get_top_links(adj, count))
    np.fill_diagonal(adj, 10000)
    ind_pos = np.array(get_top_links(-adj, count))

    return ind_pos, ind_neg


def get_node_wise_pos_neg(x, count):
    adj = dtw_corr(x.T)
    pos_ind = np.argsort(adj, axis=1)[:, 1:count+1]
    neg_ind = np.argsort(adj, axis=1)[:, -count:]
    pos_ind = np.stack([np.arange(len(pos_ind))[:, np.newaxis].repeat(pos_ind.shape[1], axis=1), pos_ind], axis=-1).reshape(-1, 2)
    neg_ind = np.stack([np.arange(len(neg_ind))[:, np.newaxis].repeat(neg_ind.shape[1], axis=1), neg_ind], axis=-1).reshape(-1, 2)
    return pos_ind, neg_ind


def get_pos_neg_all(dataset, count):
    pos_pair = []
    neg_pair = []
    for i in range(len(dataset)):
        print(i)
        x = dataset[i]["fmri"]
        pos, _ = get_pos_neg_pair_(x, count*116)
        _, neg = get_node_wise_pos_neg(x, count)
        pos_pair.append(pos)
        neg_pair.append(neg)

    return pos_pair, neg_pair


def pearson2(X):
    m = torch.mean(X, dim=-1, keepdim=True)
    s = torch.std(X, dim=-1, keepdim=True)
    X = X - m
    return torch.matmul(X, X.transpose(-1, -2)) / torch.matmul(s, s.transpose(-1, -2)) / (X.shape[-1] - 1)


def create_BV(fmri_t, window=10):
    fmri_slide = fmri_t.unfold(-1, window, 1).permute(0, 2, 3, 1)
    return pearson2(fmri_slide)


def bv_boundary(bv):
    bv = bv.sum(dim=(-1, -2)).mean(0)
    return bv


def process_bv(bv, window=36):
    bv = align_local_mean(bv)
    bv[bv > 0] = 0
    bv = -bv
    bv_expanded = torch.cat([bv, bv[-window+1:]], dim=-1)
    bv_slide = bv_expanded.unfold(-1, window, 1)
    bv = bv_slide.softmax(dim=-1)[:, 0]
    return bv


def align_local_mean(fmri, window=15):
    fmri_expanded = torch.cat([fmri, fmri[-window+1:]], dim=-1)
    local_mean = fmri_expanded.unfold(-1, window, 1).mean(-1)
    return fmri - local_mean





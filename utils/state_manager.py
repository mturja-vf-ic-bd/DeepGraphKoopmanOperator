import os

import numpy as np
import torch

_HOME = "/home/mturja/KVAE"
_HOME = "/Users/mturja/PycharmProjects/KVAE"
_DATA_PATH = os.path.join(_HOME, "data")


def remove_states(state, labels):
    keepstate = []
    for i in range(1, 18):
        if i not in state:
            keepstate.append(i)

    idx = []
    for i, s in enumerate(keepstate):
        new_idx = (labels == s).nonzero()
        if i == 0:
            idx = new_idx[0]
        else:
            idx = np.concatenate((idx, new_idx[0]), axis=0)
    # idx = idx.squeeze(1)
    idx = np.sort(idx)
    return idx


def pick_state(keepstate, labels):
    idx = []
    for i, s in enumerate(keepstate):
        if i == 0:
            idx = (labels == s).nonzero(as_tuple=False)
        else:
            idx = torch.cat((idx, (labels == s).nonzero(as_tuple=False)), 0)
    idx = torch.sort(idx)[0]
    return idx[:, 0]


def get_subnet_map():
    import csv
    map = {}
    with open(os.path.join(_DATA_PATH, "shen_268_parcellation_networklabels.csv")) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        count = 0
        for row in csv_reader:
            if count == 0:
                count = count + 1
                continue
            subnet_id = int(row[1].strip())
            node = int(row[0].strip()) - 1
            if subnet_id not in map.keys():
                map[subnet_id] = [node]
            else:
                map[subnet_id].append(node)
            count = count + 1
        print("Total lines: {}".format(count))
    for k, v in map.items():
        map[k] = torch.LongTensor(v)
    return map


def get_DMN_ind():
    import csv
    with open(os.path.join(_DATA_PATH, "DMN_ROI.csv")) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        node = 0
        ind = []
        for row in csv_reader:
            flag = int(row[0].strip())
            if flag == 1:
                ind.append(node)
            node = node + 1
    return torch.LongTensor(ind)


def get_Attn_ind():
    import csv
    with open(os.path.join(_DATA_PATH, "AttentionROI.csv")) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        node = 0
        ind = []
        for row in csv_reader:
            flag = int(row[0].strip())
            if flag == 1:
                ind.append(node)
            node = node + 1
    return torch.LongTensor(ind)


def get_DMN_Attn_ind():
    ind_dmn = get_DMN_ind()
    ind_attn = get_Attn_ind()
    return torch.cat([ind_dmn, ind_attn])


def get_subnetwork(raw_data, subnet):
    subnet_map = get_subnet_map()
    return raw_data[:, subnet_map[subnet], :]


def sort_based_on_subnet(data, skip_clusters=None):
    subnet_map = get_subnet_map()
    ind_list = []
    cluster_len = []
    for k, v in subnet_map.items():
        if skip_clusters is not None and k in skip_clusters:
            continue
        ind_list += v
        cluster_len.append(len(v))
    ind = np.array(ind_list)
    data = data[ind]
    data = data[:, ind]
    return data, cluster_len


if __name__ == '__main__':
    dmn_ind_1 = get_subnet_map()
    dmn_ind_2 = get_DMN_ind()
    attn_ind = get_Attn_ind()
    for k, v in dmn_ind_1.items():
        print(k, len(v), v)
    print(len(dmn_ind_2))
    print(len(attn_ind))


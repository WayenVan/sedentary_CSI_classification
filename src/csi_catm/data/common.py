from importlib.resources import path
import os
from socketserver import DatagramRequestHandler
import numpy as np
import scipy.io as scio
import torch.nn as nn
from random import shuffle
from typing import Tuple, List, Dict

"""----------util functions-------------------"""

def zero_padding(data, T_MAX):
    # data(list)=>data_pad(ndarray): [20,20,T1/T2/...]=>[20,20,T_MAX]
    data_pad = []
    for i in range(len(data)):
        t = np.array(data[i]).shape[2]
        data_pad.append(np.pad(data[i], ((0,0),(0,0),(T_MAX - t,0)), 'constant', constant_values = 0).tolist())
    return np.array(data_pad)


def load_data_BvP(path_to_dataset, file_name, T_MAX):
    file_path = os.path.join(path_to_dataset, file_name)

    data_1 = scio.loadmat(file_path)['velocity_spectrum_ro']
    label_1 = int(file_name.split('-')[1]) - 1

    data_1 = zero_padding([data_1], T_MAX)
    data_1 = data_1[0]

    return data_1, label_1

def load_data_catm(path_to_dataset, file_name, T_MAX):
    file_path = os.path.join(path_to_dataset, file_name)

    data_1 = scio.loadmat(file_path)['save_spect']
    label_1 = int(file_name.split('-')[1]) - 1
    
    return data_1, label_1


def load_data_timeData(path_to_dataset, file_name):
    file_path = os.path.join(path_to_dataset, file_name)

    data = scio.loadmat(file_path)['timesData']
    data = np.abs(data)
    label = int(file_name.split('-')[1]) - 1

    return data, label

def random_split_data_list(data_list: list, test_ratio):
    shuffle(data_list)
    test_size = round(len(data_list)*test_ratio)
    train_size = len(data_list) - test_size
    train_list = data_list[:train_size]
    test_list = data_list[train_size:]
    return train_list, test_list

def parse_catm_file_name(file_name: str) -> Tuple[int, int, int, int]:
    """parser a name from catm dataset, for example, "user1-2-1-12.m" will be pasered into:
    (user, action, channel, index), which is (1, 2, 1, 12)
    """
    splited = file_name.split('-')
    
    user = int(splited[0][4:])
    action = int(splited[1])
    channel = int(splited[2])
    index = int(splited[3].split('.')[0])
    
    return user, action, channel, index


def aggregate_3channel(file_list: List[str]) -> List[List]:
    """find 3 channel data for CATM dataset
    :return: [[channel1, channe2, ...]]
    """
    d: Dict[str, List[str]] = {}
    
    for name in file_list:
        user, action, channel,index = parse_catm_file_name(name)
        label = '{}-{}-{}'.format(user, action, index)
        if label not in d.keys():
            d[label] = []
        d[label].append(name)
    
    ret = list(d.values())
    _ret = []
    for names in ret:
        if len(names) == 3:
            item = sorted(names, key=lambda x: parse_catm_file_name(x)[-2])
            _ret.append(item)
    return _ret

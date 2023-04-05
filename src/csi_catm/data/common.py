from importlib.resources import path
import os
from socketserver import DatagramRequestHandler
import numpy as np
import scipy.io as scio
import torch.nn as nn
from random import shuffle


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


def collide_into_3channel(data_list: list):
    pass
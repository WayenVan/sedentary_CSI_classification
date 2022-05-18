import os
from socketserver import DatagramRequestHandler
import numpy as np
import scipy.io as scio
import torch.nn as nn


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



def print_parameters_grad(model: nn.Module):
    named_parameters = model.named_parameters()
    for name, parms in named_parameters:
        print("--->name:", name, '--->grad_requires:', parms.requires_grad, '--->grad_value:', parms.grad)

def print_parameters(model: nn.Module):
    named_parameters = model.named_parameters()
    for name, parms in named_parameters:
        print("--->name:", name, '--->grad_requires:', parms.requires_grad, '--->parms_value:', parms)



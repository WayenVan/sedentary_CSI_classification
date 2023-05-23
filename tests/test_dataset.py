import unittest
import torch
import torch.nn as nn

from csi_catm.data.common import parse_catm_file_name, aggregate_3channel
from csi_catm.data.dataset import Catm3ChannelDataset

import os

def test_3channel_dataset():
    file_list = os.listdir("dataset/CATM")
    file_list = aggregate_3channel(file_list)
    dataset = Catm3ChannelDataset("dataset/CATM", file_list, 100)   
    for data, label in dataset:
        print(data)

def test_softmax():
    a = torch.tensor([[1, 2, 3],
                    [3, 2, 4]], dtype=torch.float32)
    loss = nn.Softmax(dim=0) 
    a = loss(a)
    print(a)


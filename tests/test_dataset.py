import torch
import torch.nn as nn
import sys
sys.path.append('src')
from csi_catm.data.common import parse_catm_file_name, aggregate_3channel
from csi_catm.data.dataset import Catm3ChannelDataset, BvPDataset

import os
import matplotlib.pyplot as plt


def test_3channel_dataset():
    file_list = os.listdir("dataset/CATM")
    file_list = aggregate_3channel(file_list)
    dataset = Catm3ChannelDataset("dataset/CATM", file_list, 100)   
    for data, label in dataset:
        for image in data:
            plt.imshow(image[0])
            plt.show(block=False)
            plt.pause(0.1)
            plt.cla()

def test_softmax():
    a = torch.tensor([[1, 2, 3],
                    [3, 2, 4]], dtype=torch.float32)
    loss = nn.Softmax(dim=0) 
    a = loss(a)
    print(a)

def test_catm_bvpformat():
    file_list = os.listdir("dataset/CATM_bvpformat")
    dataset = BvPDataset("dataset/CATM_bvpformat", file_list, 30)   
    for image in dataset[0][0]:
        plt.imshow(image)
        plt.show(block=False)
        plt.pause(0.1)
        plt.cla()

if __name__ == '__main__':
    # test_catm_bvpformat()
    test_3channel_dataset()
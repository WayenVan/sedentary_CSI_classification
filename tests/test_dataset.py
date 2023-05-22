import unittest
from csi_catm.data.common import parse_catm_file_name, aggregate_3channel
from csi_catm.data.dataset import Catm3ChannelDataset

import os

def test_3channel_dataset():
    file_list = os.listdir("dataset/CATM")
    file_list = aggregate_3channel(file_list)
    dataset = Catm3ChannelDataset("dataset/CATM", file_list, 100)   
    for data, label in dataset:
        print(data)


if __name__ == "__main__":
    unittest.main()
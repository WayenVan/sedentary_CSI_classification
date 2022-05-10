import os
import torch
from torch.utils.data import Dataset
from torch import tensor
import scipy.io as scio
from common import normalize_data
from torch.nn import functional
import numpy as np

class CustomDataset(Dataset):
    
    def __init__(self, path_to_data, data_list, num_class) -> None:
        super().__init__()
       # self.data_list = data_list
        self.path_to_data = path_to_data
        self.data_list = data_list
        self.num_class = num_class
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):

        data_file_name = self.data_list[index]
        file_path = os.path.join(self.path_to_data,data_file_name)
        
        data_1 = scio.loadmat(file_path)['save_spect']
        label_1 = int(data_file_name.split('-')[1])

        data_norm_1 = normalize_data(data_1)
        # label_1 = onehot_encoding(label_1, self.num_class)
        label_1 = functional.one_hot(tensor(label_1), self.num_class)
        return tensor(data_norm_1), label_1


        

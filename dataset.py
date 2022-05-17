import os
import torch
from torch.utils.data import Dataset
from torch import tensor
import scipy.io as scio
from common import load_data,onehot_encoding,zero_padding,normalize_data
from torch.nn import functional
import numpy as np
import torchvision as tv
from einops import rearrange

class BvPDataset(Dataset):
    
    def __init__(self, path_to_data, data_list, num_class, T_MAX, img_size=(30, 30)) -> None:
        super().__init__()
       # self.data_list = data_list
        self.T_MAX = T_MAX
        self.path_to_data = path_to_data
        self.data_list = data_list
        self.num_class = num_class
        self.img_size = img_size
        self.resize = tv.transforms.Resize(img_size)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):

        data_file_name = self.data_list[index]
        file_path = os.path.join(self.path_to_data,data_file_name)
        
        data_1 = scio.loadmat(file_path)['velocity_spectrum_ro']
        # data_1 = scio.loadmat(file_path)['save_spect']
        label_1 = int(data_file_name.split('-')[1]) - 1
        
        data_1 = zero_padding([data_1], self.T_MAX)
        data_1 = data_1[0]
        

        data_1 = torch.tensor(data_1)
        data_1 = rearrange(data_1, '(c h) w s -> s c h w', c=1)
        data_1 = self.resize(data_1)
        data_1 = functional.normalize(data_1, dim=-3)
   
        label_1 = functional.one_hot(tensor(label_1), self.num_class).type(torch.float64)

        return data_1, label_1


        

class MyDataset(Dataset):
    
    def __init__(self, path_to_data, data_list, num_class, T_MAX, img_size=(30, 30)) -> None:
        super().__init__()
       # self.data_list = data_list
        self.T_MAX = T_MAX
        self.path_to_data = path_to_data
        self.data_list = data_list
        self.num_class = num_class
        self.img_size = img_size
        self.resize = tv.transforms.Resize(img_size)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):

        data_file_name = self.data_list[index]
        file_path = os.path.join(self.path_to_data,data_file_name)
        
        data_1 = scio.loadmat(file_path)['save_spect']
        label_1 = int(data_file_name.split('-')[1]) - 1
        
        data_1 = zero_padding([data_1], self.T_MAX)
        data_1 = data_1[0]
        

        data_1 = torch.tensor(data_1)
        data_1 = self.resize(data_1)
        data_1 = functional.normalize(data_1, dim=-3)
   
        label_1 = functional.one_hot(tensor(label_1), self.num_class).type(torch.float64)

        return data_1, label_1
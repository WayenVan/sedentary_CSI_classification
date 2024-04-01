import os
import torch
from torch.utils.data import Dataset
from torch import tensor
import scipy.io as scio
from .common import zero_padding, load_data_BvP
import numpy as np
import torchvision as tv
from einops import rearrange
from typing import List, Union
import torch.nn.functional as F

class BvPDataset(Dataset):
    
    def __init__(self, path_to_data, data_list, T_MAX, img_size=(20, 20)) -> None:
        super().__init__()
        # self.data_list = data_list
        self.T_MAX = T_MAX
        self.path_to_data = path_to_data
        self.data_list = data_list
        self.img_size = img_size
        self.resize = tv.transforms.Resize(img_size)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):

        data_file_name = self.data_list[index]
        data, label = load_data_BvP(self.path_to_data, data_file_name, self.T_MAX)
        data = rearrange(data, 'h w t -> t h w')
        data = torch.tensor(data, dtype=torch.float32)
        data = self.resize(data)
        data = F.normalize(data, dim=-1)
        
        return data, label

class CatmDataset(Dataset):
    
    def __init__(self, path_to_data, data_list, num_class, down_sample=(1, 1, 1), transform=None) -> None:
        """
        :param down_sample: down_sample for dimension(time, high, width) defaults to (1, 1, 1)
        """
        super().__init__()
        # self.data_list = data_list
        self.path_to_data = path_to_data
        self.data_list = data_list
        self.num_class = num_class
        self.down_sample = down_sample
        self.t = transform
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        """
        return: data_1 [seq, h, w], label index
        """
        data_file_name = self.data_list[index]
        file_path = os.path.join(self.path_to_data,data_file_name)
        
        data_1 = scio.loadmat(file_path)['save_spect']
        label_1 = int(data_file_name.split('-')[1]) - 1
        data_1: np.ndarray = data_1.astype('float32')
        
        if self.t is not None:
            data_1 = self.t(data_1)
        
        return data_1,np.array(label_1, dtype='int64')
    
    def _padding_t(self, data: np.ndarray, padding_length):
        pad = []
        for index, dim in enumerate(data.shape):
            if index == 0:
                pad.append((padding_length-dim, 0))
            else:
                pad.append((0,0))         
        return np.pad(data, pad, mode='constant', constant_values=0)

class TimeDataset(Dataset):
    
    def __init__(self, data_dir, data_list, num_classes, col_select = "dop_spec_ToF", norm = False) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.data_list = data_list
        self.num_classes = num_classes
        self.col_select = col_select
        self.norm = norm
    
    def __len__(self):
        return len(self.data_list)
    
    
    def __getitem__(self, index):
        
        data_file_name = self.data_list[index]
        file_path = os.path.join(self.data_dir, data_file_name)
        
        data = scio.loadmat(file_path)[self.col_select]
        label = int(data_file_name.split('-')[1]) - 1
            
        #[s, d]
        data = torch.tensor(data)
        if self.norm:
            data = F.normalize(data, dim = 0)
        #[]
        label = torch.tensor(label)
        
        return data, label #[s, d], []
        
class Catm3ChannelDataset(Dataset):
    
    def __init__(self, data_root: str, data_list: List[List[str]], t_padding: int, transform=None) -> None:
        super().__init__()
        self._data_root = data_root
        self._data_list:  List[List[str]] = data_list
        self._padding = t_padding
        self._transform = transform
        
    def __len__(self):
        return len(self._data_list)
        
    def __getitem__(self, index):
        file_names_3c = self._data_list[index]
        channels_data = []
        for idx, channel_file in enumerate(file_names_3c):
            assert channel_file.split('-')[-2] == str(idx + 1)
            file_path = os.path.join(self._data_root, channel_file)
            data: np.ndarray = scio.loadmat(file_path)['save_spect']

            #[t, h, w]
            data = data.astype(np.float32)
            data = self._padding_t(data, self._padding)
            channels_data.append(data)
            
        label = int(file_names_3c[0].split('-')[1]) - 1
        
        #[t, c, h, w]
        x = np.stack(channels_data, axis=1)
        
        if self._transform:
            x = self._transform(x)

        #standardization
        mean = np.mean(x, (-2, -1), keepdims=True)
        std = np.std(x, (-2, -1), keepdims=True)
        x = (x - std) / (mean + 1e-8)

        return x, label
    
    def _padding_t(self, data: np.ndarray, padding_length):
        pad = []
        for index, dim in enumerate(data.shape):
            if index == 0:
                pad.append((padding_length-dim, 0))
            else:
                pad.append((0,0))         
        return np.pad(data, pad, mode='constant', constant_values=0)
    
    
import os
import torch
from torch.utils.data import Dataset
from torch import tensor
import scipy.io as scio
from .common import zero_padding, load_data_BvP
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
        data_1, label_1 = load_data_BvP(self.path_to_data, data_file_name, self.T_MAX)

        data_1 = torch.tensor(data_1)
        data_1 = rearrange(data_1, '(c h) w s -> s c h w', c=1)
        data_1 = self.resize(data_1)
        data_1 = functional.normalize(data_1, dim=0)
   
        label_1 = functional.one_hot(tensor(label_1), self.num_class).type(torch.float64)

        return data_1, label_1


class CatmDataset(Dataset):
    
    def __init__(self, path_to_data, data_list, num_class, t_padding, down_sample=(1, 1, 1)) -> None:
        """
        :param down_sample: down_sample for dimension(time, high, width) defaults to (1, 1, 1)
        """
        super().__init__()
       # self.data_list = data_list
        self.t_padding = t_padding
        self.path_to_data = path_to_data
        self.data_list = data_list
        self.num_class = num_class
        self.down_sample = down_sample
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        """
        return: data_1 [seq, h, w], label index
        """
        data_file_name = self.data_list[index]
        file_path = os.path.join(self.path_to_data,data_file_name)
        
        data_1: np.ndarray = scio.loadmat(file_path)['save_spect']
        label_1 = int(data_file_name.split('-')[1]) - 1
        data_1 = data_1[::self.down_sample[0], ::self.down_sample[1], ::self.down_sample[2]]
        data_1 = self._padding_t(data_1, self.t_padding)
        
        data_1_tensor: torch.Tensor = torch.tensor(data_1, dtype=torch.float32)
        data_1_tensor = functional.normalize(data_1_tensor, dim=0)
   
        label_1 = functional.one_hot(tensor(label_1), self.num_class)

        return data_1_tensor, label_1
    
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
            data = functional.normalize(data, dim = 0)
        #[]
        label = torch.tensor(label)
        
        return data, label #[s, d], []
        
        
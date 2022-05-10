import os
from torch.utils.data import DataLoader,Dataset
from torch import tensor
import scipy.io as scio
from common import normalize_data
import numpy as np

class CustomDataset(Dataset):
    
    def __init__(self, path_to_data) -> None:
        super().__init__()
       # self.data_list = data_list
        self.path_to_data = path_to_data
        self.data_list = os.listdir(path_to_data)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        file_path = os.path.join(self.path_to_data, self.data_list[index])
        data_1 = scio.loadmat(file_path)['save_spect']
        label_1 = int(self.data_list[index].split('-')[1])
        data_normed_1 = normalize_data(data_1)
        
        return tensor(data_normed_1), tensor(label_1)


        

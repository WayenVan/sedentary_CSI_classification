import os
import torch
from torch.utils.data import Dataset
from torch import tensor
import scipy.io as scio
from common import normalize_data
from torch.nn import functional
import numpy as np
import torchvision as tv


class CustomDataset(Dataset):
    
    def __init__(self, path_to_data, data_list, num_class, img_size=(30, 30)) -> None:
        super().__init__()
       # self.data_list = data_list
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
        label_1 = int(data_file_name.split('-')[1]) - 1

        data_1 = torch.tensor(data_1)
        data_1 = torch.unsqueeze(data_1, 1).cuda()
        data_1 = self.resize(data_1)
        # data_1 = functional.normalize(data_1, dim=-3)
        data_1 = torch.movedim(data_1, 1, -1)
        
        # label_1 = onehot_encoding(label_1, self.num_class)
        label_1 = functional.one_hot(tensor(label_1), self.num_class)
        return data_1, label_1


        

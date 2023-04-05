from os import lseek
from turtle import forward

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from einops import rearrange
import math

import torchsnooper


class CNN(nn.Module):

    def __init__(self, d_model, n_classes, img_size, dropout=0.1) -> None:
        super().__init__()
        
        self._img_size = img_size
        
        self.conv = nn.Sequential(
            nn.Conv2d(img_size[0], 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(dropout)
        )

        self.flatten = nn.Flatten(start_dim=-3)
        
       
        self.fc = nn.Sequential(
            nn.Linear(self._linear_intput_size, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
            nn.Softmax(dim=-1)
        )


    def forward(self, x: torch.Tensor):
        """
        :param x [b, c, h, w]
        :return [b, emb]
        """
        x = x.type(torch.float32)
        x = x.contiguous()
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x 
    
    @property
    def _linear_intput_size(self) -> int:
        tmp_size = list(self._img_size)
        tmp_size.insert(0, 1)
        tmp: torch.Tensor = torch.zeros(tmp_size)
        tmp = self.conv(tmp)
        tmp = self.flatten(tmp)
        
        return tmp.size()[-1]
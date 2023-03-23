from os import lseek
from turtle import forward

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from einops import rearrange
import math

import torchsnooper

class CnnGru(nn.Module):
    
    def __init__(self, n_class, d_model, img_size, channel, gru_num_layers, dropout=0.1, **nn_attributes) -> None:
        super(CnnGru, self).__init__()

        channel = img_size[0]
        width = img_size[2]
        height = img_size[1]
        
        self.cnn = CNN(d_model, img_size, channel, dropout=dropout)

        self.gru_num_layers = gru_num_layers

        self.h0 = Parameter(torch.rand((gru_num_layers, d_model), requires_grad=True))
        self.gru = nn.GRU(d_model, d_model, num_layers=gru_num_layers, dropout=dropout, **nn_attributes)
        
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model, **nn_attributes),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_class, **nn_attributes),
            nn.Softmax(dim=-1)
        )

    # @torchsnooper.snoop()
    def forward(self, x: torch.Tensor):
        """
        [t, b, c, h, w]
        """
        batch_size = x.size()[1]

        x = rearrange(x, 't b c h w -> (t b) c h w')
        x = self.cnn(x)
        x = rearrange(x, '(t b) emb -> t b emb', b=batch_size)
        
        h0 = rearrange(self.h0, 'l (b emb) -> l b emb', b=1)
        h0 = h0.expand(-1, batch_size, -1).contiguous()
        
        _, hn= self.gru(x, h0)
        output = self.fc(hn[-1, :, :])
        return output


class CNN(nn.Module):

    def __init__(self, d_model, img_size, channel, dropout=0.1) -> None:
        super().__init__()
        
        self._img_size = img_size
        
        self.conv = nn.Sequential(
            nn.Conv2d(channel, 32, 3),
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
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )

    # @torchsnooper.snoop()
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
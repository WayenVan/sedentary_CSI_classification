import torch
from torch.nn import Module, Parameter
from torch import nn 
from einops import rearrange
import torch.nn.functional as F
from typing import List
from torch.nn import Module, Parameter
from torch.nn.modules import LayerNorm
from torch import float64

import numpy as np
import math
import copy
from ..csi_typing import ImageSize

class ResLSTM(nn.Module):
    
    def __init__(self, d_model: int, 
                 input_size: ImageSize,
                 n_class: int,
                 n_res_block: int, 
                 n_lstm_layer: int, 
                 channel_size: int, 
                 freezer = None,
                 kernel_size: int = 3,
                 bidirectional: bool = False,
                 dropout: float = 0.1,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        assert kernel_size % 2 == 1, "kernel size must be odd"
        cnn_list = [nn.Conv2d(input_size[0], channel_size, kernel_size, padding=(kernel_size-1)//2), nn.MaxPool2d((2, 2))]
        cnn_list = cnn_list + [ResBlock(channel_size, channel_size, channel_size, kernel_size) for i in range(n_res_block)]
        cnn_list = cnn_list + [
             nn.MaxPool2d((2, 2)),
            nn.Flatten(start_dim=-3, end_dim=-1),
            nn.Linear(math.prod([channel_size, input_size[1]//4, input_size[2]//4]), d_model),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        ]
        self.cnn = nn.Sequential(*cnn_list)
        
        self.lstm = nn.LSTM(d_model, d_model, n_lstm_layer, bidirectional = bidirectional, dropout=dropout, **kwargs)
        
        self.fc = nn.Sequential(
            nn.Linear(d_model, n_class),
            nn.Dropout(dropout),
            nn.Softmax(dim = -1)
        )
        
        d = 2 if bidirectional else 1
        self.c0 = Parameter(torch.zeros((d*n_lstm_layer, d_model), requires_grad=True))
        self.h0 = Parameter(torch.zeros((d*n_lstm_layer, d_model), requires_grad=True))

        if freezer is not None:
            freezer(self)
        

    def forward(self, x: torch.Tensor):
        
        batch_size = x.size()[1]

        x = rearrange(x, 's b c h w -> (s b) c h w')
        x = self.cnn(x)
        x = rearrange(x, '(s b) d_model -> s b d_model', b=batch_size)

        _h0 = torch.unsqueeze(self.h0, -2).broadcast_to((-1, batch_size, -1)).contiguous()
        _c0 = torch.unsqueeze(self.c0, -2).broadcast_to((-1, batch_size, -1)).contiguous()

        _, (hn, cn) = self.lstm(x, (_h0, _c0))
        output = hn[-1, :, :]
        output = self.fc(output)
        return F.log_softmax(output, dim=-1)



"""
Define an nn.Module class for a simple residual block with equal dimensions
"""
class ResBlock(nn.Module):

    """
    Iniialize a residual block with two convolutions followed by batchnorm layers
    """
    def __init__(self, in_size:int, hidden_size:int, out_size:int, kernel_size:int):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel size must be odd"
        
        self.conv1 = nn.Conv2d(in_size, hidden_size, kernel_size, padding=(kernel_size-1)//2)
        self.conv2 = nn.Conv2d(hidden_size, out_size, kernel_size, padding=(kernel_size-1)//2)
        self.batchnorm1 = nn.BatchNorm2d(hidden_size)
        self.batchnorm2 = nn.BatchNorm2d(out_size)

    def convblock(self, x):
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = F.relu(self.batchnorm2(self.conv2(x)))
        return x
   
    """
    Combine output with the original input
    """
    def forward(self, x): return x + self.convblock(x) # skip connection
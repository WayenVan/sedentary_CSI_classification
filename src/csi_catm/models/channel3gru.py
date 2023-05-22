import torch
import torch.nn as nn
import torch.nn.functional as f
from ..csi_typing import ImageSize
import math
from torch.nn import Parameter
import copy
from einops import rearrange

class ResRnn3C(nn.Module):
    
    def __init__(self, d_model: int, 
                image_size: ImageSize,
                n_class: int,
                n_res_block: int, 
                n_rnn_layer: int, 
                channel_size: int, 
                kernel_size: int = 3,
                bidirectional: bool = False,
                dropout: float = 0.1,
                *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        assert kernel_size % 2 == 1, "kernel size must be odd"
        
        cnn_list = [nn.Conv2d(image_size[0], channel_size, kernel_size, padding=(kernel_size-1)//2)]
        cnn_list = cnn_list + [ResBlock(channel_size, channel_size, channel_size, kernel_size) for i in range(n_res_block)]
        cnn_list = cnn_list + [
            nn.MaxPool2d((2, 2)),
            nn.Flatten(start_dim=-3, end_dim=-1),
            nn.Linear(math.prod([channel_size, image_size[1]//2, image_size[2]//2]), d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        ]
        self.cnn0 = nn.Sequential(*cnn_list)
        self.cnn1 = copy.deepcopy(self.cnn0)
        self.cnn2 = copy.deepcopy(self.cnn1)
        
        self.gru = nn.GRU(d_model*3, d_model, n_rnn_layer, bidirectional = bidirectional, dropout=dropout, **kwargs)
        
        self.fc = nn.Sequential(
            nn.Linear(d_model, n_class),
            nn.Dropout(dropout),
            nn.Softmax(dim = -1)
        )
        
        
        d = 2 if bidirectional else 1
        self.h0 = Parameter(torch.rand((d*n_rnn_layer, d_model), requires_grad=True))

    def forward(self, c0: torch.Tensor, c1: torch.Tensor, c2: torch.Tensor):
        """
        param: c channel with [time, batch, height, width]
        """
        batch_size = c0.shape[1]
        #change to [time, batch, img_channel, height, width], where img_channel=1
        _c0 = rearrange(c0, 't b (c h) w -> (t b) c h w', c=1)
        _c1 = rearrange(c1, 't b (c h) w -> (t b) c h w', c=1)
        _c2 = rearrange(c2, 't b (c h) w -> (t b) c h w', c=1)
        
        _c0 = self.cnn0(_c0)
        _c1 = self.cnn1(_c1)
        _c2 = self.cnn2(_c2)
        
        #c [(t b) d]
        x = torch.concatenate((_c0, _c1, _c2), dim=-1)
        x = rearrange(x, '(t b) d -> t b d', b=batch_size)
        
        h0 = torch.unsqueeze(self.h0, 1)
        h0 = h0.expand(-1, batch_size, -1)
        h0 = h0.contiguous()
        _, hn = self.gru(x, h0)
        return self.fc(hn[-1, :, :])
    
    
        

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
        x = f.relu(self.batchnorm1(self.conv1(x)))
        x = f.relu(self.batchnorm2(self.conv2(x)))
        return x

    """
    Combine output with the original input
    """
    def forward(self, x): return x + self.convblock(x) # skip connection
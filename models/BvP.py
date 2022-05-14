from turtle import forward, width
from numpy import dtype, float64, pad
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from einops import rearrange
import math

import torchsnooper

class BvP(nn.Module):
    
    def __init__(self, d_model, img_size, gru_num_layers, linear_emb = 64, gru_hidden_size=64, dropout=0.1) -> None:
        super(BvP, self).__init__()

        channel = img_size[0]
        height = img_size[1]
        width = img_size[2]

        conv_output_size = self._calculate_output_shape((height, width), (5, 5), (1, 1))
        conv_output_size = self._calculate_output_shape(conv_output_size, (2, 2), (2, 2))
    
        self.cnn = nn.Sequential(
            nn.Conv2d(channel, 16, (5, 5), dtype=torch.float64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Flatten(start_dim=-3),
            nn.Linear(int(math.prod(conv_output_size)*16), linear_emb, dtype=torch.float64),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(linear_emb, linear_emb, dtype=torch.float64),
            nn.Dropout(dropout),
            nn.ReLU()
        )

        self.h0 = Parameter(torch.rand((gru_num_layers, gru_hidden_size), requires_grad=True, dtype=torch.float64))
        self.gru = nn.GRU(linear_emb, gru_hidden_size, num_layers=gru_num_layers, dropout=dropout, dtype=torch.float64)
        
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden_size, d_model, dtype=torch.float64),
            nn.Dropout(dropout),
            nn.Softmax(dim=-1)
        )

    @torchsnooper.snoop()
    def forward(self, x: torch.Tensor):
        """
        [s, b, c, h, w]
        """
        batch_size = x.size()[1]
        dtype = x.dtype
        x.type(torch.float64)

        #time distributed
        x = rearrange(x, 's b c h w -> (s b) c h w')
        x = self.cnn(x)
        x = rearrange(x, '(s b) emb-> s b emb', b=batch_size)

        #create h0
        h0 = rearrange(self.h0, 'l (b emb) -> l b emb', b=1)
        h0 = h0.expand(-1, batch_size, -1).contiguous()

        # print(list(self.fc.parameters())[-2])
        _, hn = self.gru(x, h0)
        output = self.fc(hn[-1, :, :])
        output = output.type(dtype)
        print(self.h0.grad)
        return output


    def _calculate_output_shape(self, input, kernel_size, stride, padding=(0, 0), dilation=(1, 1)):

        h = input[0]
        w = input[1]

        return (h+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0]+1, \
        (w+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1]+1



from os import lseek

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from einops import rearrange
import math

import torchsnooper

class ImgGRU(nn.Module):
    
    def __init__(self, d_model, img_size, gru_num_layers, gru_hidden_size=64, dropout=0.1) -> None:
        super(ImgGRU, self).__init__()

        channel = img_size[0]
        width = img_size[2]
        height = img_size[1]

        self.gru_num_layers = gru_num_layers

        self.h0 = Parameter(torch.rand((gru_num_layers, gru_hidden_size), requires_grad=True, dtype=torch.float64))
        self.gru = nn.GRU(channel*width, gru_hidden_size, num_layers=gru_num_layers, dropout=dropout, dtype=torch.float64)
        
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden_size, gru_hidden_size, dtype=torch.float64),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(gru_hidden_size, d_model, dtype=torch.float64),
            nn.Dropout(dropout),
            nn.Softmax(dim=-1)
        )

    # @torchsnooper.snoop()
    def forward(self, x: torch.Tensor):
        """
        [b, c, h, w]
        """
        batch_size = x.size()[0]

        dtype = x.dtype
        x.type(torch.float64)

        x = rearrange(x, 'b c h w -> h b (c w)')

        #create h0
        h0 = rearrange(self.h0, 'l (b emb) -> l b emb', b=1)
        h0 = h0.expand(-1, batch_size, -1).contiguous()

        
        _, hn= self.gru(x, h0)
        output = self.fc(hn[-1, :, :])
        output = output.type(dtype)
        return output

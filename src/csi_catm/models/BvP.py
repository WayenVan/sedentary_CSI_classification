import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from einops import rearrange
import math

import torchsnooper

class BvP(nn.Module):
    
    def __init__(self, d_model, img_size, gru_num_layers, linear_emb = 64, gru_hidden_size=64, dropout=0.1) -> None:
        super(BvP, self).__init__()
    
        self.cnn = nn.Sequential(
            nn.Conv2d(img_size[0], 64, (3, 3), dtype=torch.float64),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), dtype=torch.float64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )

        self.flatten =  nn.Flatten(start_dim=-3)

        #find the fc shape:
        tmp = torch.rand([1, img_size[0], img_size[1], img_size[2]], dtype=torch.float64) 
        cnn_output = self.cnn(tmp)
        cnn_output = self.flatten(cnn_output)
        linear_input_shape = cnn_output.size()[-1]

        self.fc1 = nn.Sequential(
            nn.Linear(linear_input_shape, linear_emb, dtype=torch.float64),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(linear_emb, linear_emb, dtype=torch.float64),
            nn.Dropout(dropout),
            nn.ReLU()
        )

        self.h0 = Parameter(torch.rand((gru_num_layers, gru_hidden_size), requires_grad=True, dtype=torch.float64))
        self.gru = nn.GRU(linear_emb, gru_hidden_size, num_layers=gru_num_layers, dropout=dropout, dtype=torch.float64)
        
        self.fc2 = nn.Sequential(
            nn.Linear(gru_hidden_size, d_model, dtype=torch.float64),
            nn.Dropout(dropout),
            nn.Softmax(dim=-1)
        )

    # @torchsnooper.snoop()
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
        x = self.flatten(x)
        x = self.fc1(x)
        x = rearrange(x, '(s b) emb-> s b emb', b=batch_size)

        #create h0
        h0 = rearrange(self.h0, 'l (b emb) -> l b emb', b=1)
        h0 = h0.expand(-1, batch_size, -1).contiguous()
        
        
        _, hn= self.gru(x, h0)
        output = self.fc2(hn[-1, :, :])
        output = output.type(dtype)
        return output


    def _calculate_output_shape(self, input, kernel_size, stride, padding=(0, 0), dilation=(1, 1)):

        h = input[0]
        w = input[1]

        return (h+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0]+1, \
        (w+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1]+1



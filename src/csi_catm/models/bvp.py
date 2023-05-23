import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from einops import rearrange
import math

import torchsnooper

class BvP(nn.Module):
    
    def __init__(self, n_class, img_size, gru_num_layers, linear_emb = 64, gru_hidden_size=64, dropout=0.1) -> None:
        super(BvP, self).__init__()
    
        self.cnn = nn.Sequential(
            nn.Conv2d(img_size[0], 16, (3, 3), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 16, (3, 3), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )

        self.flatten =  nn.Flatten(start_dim=-3)

        #find the fc shape:
        tmp = torch.rand([1, img_size[0], img_size[1], img_size[2]]) 
        cnn_output = self.cnn(tmp)
        cnn_output = self.flatten(cnn_output)
        linear_input_shape = cnn_output.size()[-1]

        self.fc1 = nn.Sequential(
            nn.Linear(linear_input_shape, linear_emb),
            nn.Dropout(dropout),
            # nn.ReLU(),
            nn.Linear(linear_emb, linear_emb),
            nn.Dropout(dropout),
            nn.ReLU()
        )

        self.h0 = Parameter(torch.zeros(gru_num_layers ,gru_hidden_size))
        self.gru = nn.GRU(linear_emb, gru_hidden_size, num_layers=gru_num_layers, dropout=dropout)
        self.init_gru(self.gru)
        
        self.fc2 = nn.Sequential(
            nn.Linear(gru_hidden_size, n_class),
            nn.Dropout(dropout),
            nn.LogSoftmax(dim=-1)
        )

        self.apply(self.init_parameters)

    # @torchsnooper.snoop()
    def forward(self, x: torch.Tensor):
        """
        [s, b, c, h, w]
        """
        batch_size = x.size()[1]

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
        return output


    @staticmethod
    def init_parameters(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.constant_(module.bias, 0)
            
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)        
        
    @staticmethod
    def init_gru(module: nn.GRU):
        for name, param in module.named_parameters():
            if name.startswith('weight'):
                nn.init.xavier_uniform_(param)
            else:
                nn.init.zeros_(param)
        
    def _calculate_output_shape(self, input, kernel_size, stride, padding=(0, 0), dilation=(1, 1)):

        h = input[0]
        w = input[1]

        return (h+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0]+1, \
        (w+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1]+1



import torch
from torch.nn import Module, Parameter
from torch import nn 
from einops import rearrange

from nis import match
from tkinter import N
from torch.nn import Module, Parameter
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
from torch.nn.modules import LayerNorm
from torch import float64

import numpy as np
import typing
import math



class Vstm(Module):
    def __init__(self,
    d_model,
    d_emb,
    img_size,
    split_grid,
    nhead, 
    num_layer,
    LSTM_hidden_size,
    LSTM_num_layers,
    dim_feedforward=2048, 
    dropout=0.1, 
    activation='relu', 
    layer_norm_eps=1e-5,
    bidirectional=False, **kwargs) -> None:

    
        """Vit to lstm
        :param d_model: model dimension
        :param d_emb: the dimension of image flatten
        :param img_size: a tuple showing the size of image, (channel, height, width)
        :param split_grid: (row, col)
        :param nhead: the head number of multi-head attention
        :param num_layer: the number of transformer encodeer
        :param LSTM_hidden_size: the hidden dimension of LSTM hidden state
        :param LSTM_num_layers: the layers of LSTM chain
        :param dim_feedforward: the dimension of the feedforward network model, defaults to 2048
        :param dropout: the dropout value, defaults to 0.1
        :param activation: activation function, defaults to 'relu'
        :param layer_norm_eps: the epsilon of layer normalizatio , defaults to 1e-5
        :param bidirectional: if the LSTM is bidirectional, default to False
        :param **kwargs: other keyword that passed to my Vit and LSTM modules
        """
        super().__init__()

        self.bidirectional = bidirectional

        self.ViT = ViT(d_emb, d_emb, img_size, split_grid, nhead, num_layer, 
        dim_feedforward, dropout, activation, layer_norm_eps, **kwargs)

        self.LSTM = nn.LSTM(d_emb, LSTM_hidden_size, LSTM_num_layers, 
        dropout=dropout, bidirectional = bidirectional, **kwargs)

        self.fc = nn.Sequential(
            nn.Linear(LSTM_hidden_size, d_model),
            nn.Dropout(dropout),
            nn.Softmax(dim = -1)
        )
        

        d = 2 if bidirectional else 1
        self.c0 = Parameter(torch.zeros((d*LSTM_num_layers, LSTM_hidden_size), requires_grad=True))
        self.h0 = Parameter(torch.zeros((d*LSTM_num_layers, LSTM_hidden_size), requires_grad=True))

        
        

    def forward(self, x: torch.Tensor):
        """

        :param x: [sequence, batch, channel, height, width] 
        :return: [batchsize, category]
        """     
        
        origin_type = x.dtype
        batch_size = x.size()[1]

        x = rearrange(x, 's b c h w -> (s b) c h w')
        x = self.ViT(x)
        x = rearrange(x, '(s b) emb -> s b emb', b=batch_size)

        h0 = torch.unsqueeze(self.h0, -2).broadcast_to((-1, batch_size, -1)).contiguous()
        c0 = torch.unsqueeze(self.c0, -2).broadcast_to((-1, batch_size, -1)).contiguous()

        _, (hn, cn) = self.LSTM(x, (h0, c0))
        output = hn[-1, :, :]

        output = self.fc(output)

        return output.type(origin_type)


class ViT(Module):
    
    def __init__(self, 
    d_model,
    d_emb,
    img_size,
    split_grid,
    nhead, 
    num_layer,
    dim_feedforward=2048, 
    dropout=0.1, 
    activation='relu', 
    layer_norm_eps=1e-5) -> None:
        """my simple ViT

        :param d_model: model dimension
        :param d_emb: the dimension of image flatten
        :param img_size: a tuple showing the size of image, (channel, height, width)
        :param split_grid: (row, col)
        :param nhead: the head number of multi-head attention
        :param num_layer: the number of transformer encodeer
        :param dim_feedforward: the dimension of the feedforward network model, defaults to 2048
        :param dropout: the dropout value, defaults to 0.1
        :param activation: activation function, defaults to 'relu'
        :param layer_norm_eps: the epsilon of layer normalizatio , defaults to 1e-5
        :raises ValueError: 
        """
        super().__init__()

        #check if the image could be split properly
        assert img_size[1] % split_grid[0] == 0, "the row of img shoud be divisible by the split row"
        assert img_size[2] % split_grid[1] == 0, "the column of img should be divisible by teh split column"

        self.img_size = img_size
        self.split = split_grid        
        self.d_emb = d_emb


        encoder_layer = TransformerEncoderLayer(d_emb, nhead, dim_feedforward, dropout, activation, layer_norm_eps)
        layer_norm = LayerNorm(d_emb, eps=layer_norm_eps)
        self.encoder = TransformerEncoder(encoder_layer, num_layer, norm=layer_norm)

        self.class_emb = Parameter(torch.rand([d_emb], requires_grad=True))
       
        if not isinstance(img_size, typing.Tuple) and len(img_size) == 3:
            raise ValueError("The img_size must be a 3 dimension tuple, current value={}".format(img_size))

        flattend_size = int(np.prod([img_size[1]//self.split[0], img_size[2]//self.split[1], img_size[0]]))
        self.fc1 = nn.Sequential(   
            nn.Linear(flattend_size , d_emb),
            nn.Dropout(dropout),
            self._generate_activation(activation)
        )
       
        self.fc2 = nn.Sequential(
            nn.Linear(d_emb, d_model),
            nn.Dropout(dropout),
            self._generate_activation(activation)

        )
    
    def forward(self, x: torch.Tensor):
        """

        :param x: patch list [batch, channel, height, wideth]
        """
        origin_type = x.dtype

        batch = x.size()[0]
        row = x.size()[2]
        col = x.size()[3]
        channel = x.size()[1]

        assert (channel, row, col) == self.img_size, "input size should should match the model preset"

        x = rearrange(x, 'b c (sh h) (sw w) -> (sh sw) b (h w c)', sh=self.split[0], sw=self.split[1])

        x = self.fc1(x)

        #print(list(self.class_emb)[-1])
        class_emb = rearrange(self.class_emb, '(s b emb) -> s b emb', s=1, b=1)
        class_emb = class_emb.expand([-1, batch, -1])

        x = torch.cat((class_emb, x), dim=0)
        
        #transformer encoder
        pe = self._positionalencoding1d(self.d_emb, x.size(0)).cuda()
        pe = pe.unsqueeze(1)
        x = x + pe
        x = self.encoder(x)

        #MLP
        x = x[0, :, :]
        #print(list(self.fc2.parameters()))
        x = self.fc2(x)

        x = x.type(origin_type)
        return x
    
    

    def _positionalencoding1d(self, d_model, length):
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                            "odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                            -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        return pe

    def _generate_activation(self, func: str):
        ret = None
        if func == 'relu':
            ret = nn.ReLU()
        if func == 'tanh':
            ret = nn.Tanh()
        if func == 'softmac':
            ret = nn.Softmax()

        assert ret!=None, "must input a valiade activation function!"

        return ret


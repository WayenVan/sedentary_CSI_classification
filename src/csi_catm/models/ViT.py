from nis import match
from tkinter import N
import torch
from torch.nn import Module, Parameter
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
from torch.nn.modules import LayerNorm
from torch import float64
from torch import nn 

from einops import rearrange

import numpy as np
import typing
import math


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


        encoder_layer = TransformerEncoderLayer(d_emb, nhead, dim_feedforward, dropout, activation, layer_norm_eps, dtype=float64)
        layer_norm = LayerNorm(d_emb, eps=layer_norm_eps, dtype=float64)
        self.encoder = TransformerEncoder(encoder_layer, num_layer, norm=layer_norm)

        self.class_emb = Parameter(torch.rand([d_emb], requires_grad=True, dtype=torch.float64))
       
        if not isinstance(img_size, typing.Tuple) and len(img_size) == 3:
            raise ValueError("The img_size must be a 3 dimension tuple, current value={}".format(img_size))

        
        self.fc1 = nn.Sequential(
            nn.Linear(np.prod([img_size[1]//self.split[0], img_size[2]//self.split[1], img_size[0]]), d_emb, dtype=torch.float64),
            nn.Dropout(dropout),
            self._generate_activation(activation)
        )
       
        self.fc2 = nn.Sequential(
            nn.Linear(d_emb, d_model, dtype=torch.float64),
            nn.Dropout(dropout),
            self._generate_activation(activation)

        )
    
    def forward(self, x: torch.Tensor):
        """

        :param x: patch list [batch, channel, height, wideth]
        """
        origin_type = x.dtype
        x = x.type(torch.float64)

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


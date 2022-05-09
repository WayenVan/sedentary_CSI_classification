from ctypes import sizeof
import imp
from turtle import forward
import torch
from torch.nn import Module
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
from torch.nn.modules import LayerNorm
from torch import nn

import numpy as np
import typing
import math

class ViT(Module):
    
    def __init__(self, 
    d_model,
    d_emb,
    img_size,
    nhead, 
    num_layer,
    dim_feedforward=2048, 
    dropout=0.1, 
    activation='relu', 
    layer_norm_eps=1e-5) -> None:
        """my simple ViT

        :param d_model: model dimension
        :param d_emb: the dimension of image flatten
        :param img_size: a tuple showing the size of image, (length, width, channel)
        :param nhead: the head number of multi-head attention
        :param num_layer: the number of transformer encodeer
        :param dim_feedforward: the dimension of the feedforward network model, defaults to 2048
        :param dropout: the dropout value, defaults to 0.1
        :param activation: activation function, defaults to 'relu'
        :param layer_norm_eps: the epsilon of layer normalizatio , defaults to 1e-5
        :raises ValueError: 
        """
        super().__init__()

        self.d_emb = d_emb

        active_funcs = {
            'relu': nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            'tanh': nn.Tanh()
        }
     

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps)
        layer_norm = LayerNorm(d_model, eps=layer_norm_eps)
        self.encoder = TransformerEncoder(encoder_layer, num_layer, norm=layer_norm)

        self.class_emb = torch.rand([d_emb], dtype=torch.float).cuda()
        self.class_emb.requires_grad = True

        if not isinstance(img_size, typing.Tuple) and sizeof(img_size) == 3:
            raise ValueError("The img_size must be a 3 dimension tuple, current value={}".format(img_size))

        self.activation_func1 = active_funcs[activation]
        self.fc1 = nn.Linear(np.prod(img_size), d_emb)
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(d_model, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim = -1)

    
    def forward(self, x: torch.Tensor):
        """

        :param x: patch list [Seq, Batch, Length, Width, Channel]
        """

        x = torch.flatten(x, start_dim=2, end_dim=-1)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.activation_func1(x)

        class_emb = torch.unsqueeze(self.class_emb, 0)
        class_emb = torch.unsqueeze(class_emb, 0)
        class_emb = class_emb.expand([-1, x.size()[1], -1])
        x = torch.cat((class_emb, x), dim=0)
        
        #transformer encoder
        pe = self._positionalencoding1d(self.d_emb, x.size(0)).cuda()
        pe = pe.unsqueeze(1)
        x = x + pe
        x = self.encoder(x)

        #MLP
        x = x[0, :, :]
        x = self.fc2(x)
        self.dropout2(x)
        x = self.softmax(x)
        
        print(x.size())
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
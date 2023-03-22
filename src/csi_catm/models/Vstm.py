import torch
from torch.nn import Module, Parameter
from torch import nn 
from einops import rearrange

from .ViT import ViT



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
    bidirectional=False) -> None:

    
        """my simple ViT

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
        :raises ValueError: 
        """
        super().__init__()

        self.bidirectional = bidirectional

        self.ViT = ViT(d_emb, d_emb, img_size, split_grid, nhead, num_layer, 
        dim_feedforward, dropout, activation, layer_norm_eps)

        self.LSTM = nn.LSTM(d_emb, LSTM_hidden_size, LSTM_num_layers, 
        dropout=dropout, bidirectional = bidirectional, dtype=torch.float64)

        self.fc = nn.Sequential(
            nn.Linear(LSTM_hidden_size, d_model, dtype = torch.float64),
            nn.Dropout(dropout),
            nn.Softmax(dim = -1)
        )
        

        d = 2 if bidirectional else 1
        self.c0 = Parameter(torch.zeros((d*LSTM_num_layers, LSTM_hidden_size), requires_grad=True, dtype=torch.float64))
        self.h0 = Parameter(torch.zeros((d*LSTM_num_layers, LSTM_hidden_size), requires_grad=True, dtype=torch.float64))

        
        

    def forward(self,x:torch.Tensor):
        origin_type = x.dtype
        x = x.type(torch.float64)

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



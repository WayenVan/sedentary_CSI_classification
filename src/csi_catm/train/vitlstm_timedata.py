import click 
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Union, Tuple

from csi_catm.models.vstm import Vstm

@click.command()
@click.option('--data_root', default='data', type=str)
@click.option('--lr', default=1e-3, type=float)
@click.option('--d_model', default=2028, type=int)
@click.option('--d_emb', default=2048, type=int)
@click.option('--img_size', nargs=3, type=int, default=(20, 20, 3), help='channel height width')
@click.option('--split_grid', nargs=2,type=int, default=(1, 1), help='split grid shape of Vit, (row, col)' )
@click.option('--nhead', default=64, type=int)
@click.option('--num_encoder', default=16, type=int)
@click.option('--num_lstm_layers', default=16, type=int)
@click.option('--dropout', default=0.1, type=float)
@click.option('--load_dir', default=None, type=str)
def main(data_root,
         lr,
         d_model,
         d_emb,
         img_size,
         split_grid,
         nhead,
         num_encoder,
         num_lstm_layers,
         dropout,
         load_dir):
    """training using the"""
    
    
    create_model(d_model, d_emb, img_size, split_grid, nhead, num_encoder, d_model, num_lstm_layers,
                 load_dir=load_dir, dropout=dropout)

    
def load_data(data_dir: str) -> Tuple[Dataset, Dataset]:
    
    return None, None

def create_model(*model_attrs, load_dir: Union[None, str] =None, **kwattrs) -> Tuple[nn.Module, dict]:

    info = {}
    model = Vstm(*model_attrs, **kwattrs)
    if load_dir != None:
        model.load_state_dict(torch.load(load_dir))
    
    return model, info

def train(model: nn.Module , epoch: int):
    pass

if __name__ == '__main__':
    main()
   
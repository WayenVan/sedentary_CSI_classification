import click 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Union, Tuple
import os

from csi_catm.models.vitlstm import Vstm
from csi_catm.data.dataset import TimeDataset
from csi_catm.data.common import random_split_data_list

@click.command()
@click.option('--data_root', default='data', type=str)
@click.option('--lr', default=1e-3, type=float)
@click.option('--batch', default=64, type=int)
@click.option('--d_model', default=8, type=int)
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
    
    
    model = create_model(d_model, d_emb, img_size, split_grid, nhead, num_encoder, d_emb, num_lstm_layers,
            load_dir=load_dir, dropout=dropout)

    
def load_data(data_dir: str, col_select: str, test_ratio=0.2) -> Tuple[Dataset, Dataset]:
    list = os.listdir(data_dir)
    train_list, test_list = random_split_data_list(list, test_ratio)
    
    trainset = TimeDataset(data_dir, train_list, 8, col_select=col_select, norm=True)
    testset = TimeDataset(data_dir, test_list, 8, col_select=col_select, norm=True)
    
    temp = trainset[0][0]
    n_seq = temp.shape[-2]
    input_dim = temp.shape[-1]
    
    # assert (n_seq == args.n_seq) & (input_dim == args.input_dim), 'the input shape of dataset should match that of the model'
    return trainset, testset

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

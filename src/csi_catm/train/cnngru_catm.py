import click 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Union, Tuple
import os
from torchinfo import summary

from csi_catm.models.cnngru import CnnGru
from csi_catm.data.dataset import CatmDataset
from csi_catm.data.common import random_split_data_list

@click.command()
@click.option('--data_root', default='dataset/CATM', type=str)
@click.option('--lr', default=1e-3, type=float)
@click.option('--batch', default=64, type=int)
@click.option('--n_class', default=8, type=int)
@click.option('--d_model', default=128, type=int)
@click.option('--img_size', nargs=3, type=int, default=(3, 20, 20), help='channel height width')
@click.option('--down_sample', nargs=3, default=(1, 1, 1), help="down sample in certain dim: (time, height, width)")
@click.option('--channel', default=64)
@click.option('--n_gru_layers', default=4)
@click.option('--test_ratio', default=.2)
@click.option('--dropout', default=0.1, type=float)
@click.option('--load_dir', default=None, type=str)
def main(data_root,
         lr,
         n_class,
         d_model,
         img_size,
         channel,
         n_gru_layers,
         dropout,
         test_ratio,
         load_dir, *args, **kwargs):
    """training using the"""
    
    
    model, info = create_model(n_class,
                         d_model,
                         img_size,
                         channel,
                         n_gru_layers,
                         load_dir=load_dir, 
                         dropout=dropout)
    
    summary(model, input_size=(1, 1, 3, 20 ,20))
    # trainset, testset  = load_data(data_root, test_ratio)

    
def load_data(data_dir: str, test_ratio=0.2) -> Tuple[Dataset, Dataset]:
    list = os.listdir(data_dir)
    train_list, test_list = random_split_data_list(list, test_ratio)
    
    trainset = CatmDataset(data_dir, train_list, 8, 100)
    testset = CatmDataset(data_dir, test_list, 8, 100)
    
    temp = trainset[0][0]
    n_seq = temp.shape[-2]
    input_dim = temp.shape[-1]
    
    # assert (n_seq == args.n_seq) & (input_dim == args.input_dim), 'the input shape of dataset should match that of the model'
    return trainset, testset

def create_model(*model_attrs, load_dir: Union[None, str] =None, **kwattrs) -> Tuple[nn.Module, dict]:
    info = {}
    model = CnnGru(*model_attrs, **kwattrs)
    if load_dir != None:
        model.load_state_dict(torch.load(load_dir))
    return model, info

def train(model: nn.Module , epoch: int):
    pass

if __name__ == '__main__':
    main()
import click
import os
import sys
import json
import numpy as np
from torchinfo import summary

from csi_catm.data.common import aggregate_3channel, random_split_data_list
from csi_catm.data.dataset import Catm3ChannelDataset, CatmDataset
from csi_catm.models.bvp import BvP
from torch.utils.data import DataLoader

from tqdm import tqdm
import torch
import torchmetrics
from einops import rearrange
import torch.nn.functional as F

@click.command()
@click.option('--debug', default=0, type=int)
@click.option('--data_root', default='dataset/CATM', type=str)
@click.option('--lr', default=1e-5, type=float)
@click.option('--batch', default=16, type=int)
@click.option('--epochs', default=100, type=int)
@click.option('--test_ratio', default=0.3)
@click.option('--device', default='cuda')
@click.option('--model_save_dir', default='models/bvp')
@click.option('--down_sample', nargs=3, default=(1, 3, 3), help='down sample (height, width)')
@click.option('--t_padding', default=100, type=int)
@click.option('--img_size', nargs=3, type=int, default=(1, 30, 30), help='channel height width')
@click.option('--d_model', default=256, type=int)
@click.option('--n_rnn_layers', default=2, type=int)
@click.option('--dropout', default=0.1, type=float)
def main(
    debug,
    data_root, 
    lr,
    batch,
    epochs,
    t_padding,
    test_ratio,
    img_size,
    d_model,
    n_rnn_layers,
    dropout,
    device,
    model_save_dir,
    down_sample):
    info = dict(locals())
    # torch.set_printoptions(profile='full')

    file_list = os.listdir(data_root)
    
    train_list, test_list = random_split_data_list(file_list, test_ratio)
    
    train_set, test_set = CatmDataset(data_root, train_list,num_class=8, t_padding=t_padding, down_sample=down_sample), \
        CatmDataset(data_root, test_list, num_class=8, t_padding=t_padding, down_sample=down_sample)
                    
    train_loader, test_loader = DataLoader(train_set, batch_size=batch), DataLoader(test_set, batch_size=batch)
    
    """Create model
    """    
    
    model = BvP(n_class=8, img_size=img_size, gru_num_layers=n_rnn_layers, linear_emb=d_model, gru_hidden_size=d_model, dropout=dropout).to(device)
    summary(model, input_data=rearrange(train_set[0][0], 't (b c h) w -> t b c h w', b=1, c=1))
    
    """training
    """
    info['train_set'] = train_list
    info['test_set'] = test_list
    
    
    os.makedirs(model_save_dir, exist_ok=True)
    with open(os.path.join(model_save_dir, 'info.json'), 'w') as f:
        json.dump(info, f)
        
    log = open(os.path.join(model_save_dir, 'training.log'), 'w', 1)
    sys.stdout = log
    
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    best_accuracy = 0.
    for epoch in range(epochs):
        model.train(True)
        
        #train
        accuracy_train = torchmetrics.Accuracy(task='multiclass', num_classes=8).to(device)
        tbar = tqdm(train_loader, desc='epoch: '+str(epoch), file=sys.stdout)
        for index, batch_data in enumerate(tbar):           
            # opt.zero_grad()
            
            #[b, t, c, h, w]
            x, labels = batch_data
            x = rearrange(x, 'b t (c h) w -> t b c h w', c=1)
            x = x.to(device)
            # x = F.normalize(x, dim=0)
            
            labels = labels.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred, labels)
            loss.backward()
            
            if debug:
                for name, parms in model.named_parameters():
                    if 'weight' not in name:
                        continue
                    print("-->name: {} -->grad_requirs: {} -->grad_value: {}".format(name, parms.requires_grad, parms.grad))
            
            opt.step()
            acu = accuracy_train(y_pred, labels).item()
            tbar.set_postfix({
                'batch_accuracy': acu,
                'batch_loss': loss.item()
            })

        model.train(False)
        #test
        accuracy_test = torchmetrics.Accuracy(task='multiclass', num_classes=8).to(device)
        
        
        for index, batch_data in enumerate(test_loader):
            x, labels = batch_data
            x = rearrange(x, 'b t c h w -> c t b h w')
            x = x.to(device)
            labels = labels.to(device)
            y_pred = model(x[0], x[0], x[0])
            accuracy_test.update(y_pred, labels)
        
        accu_test = accuracy_test.compute().item()
        print('test_accuracy='+str(accu_test), end=', ')
        
        #save model
        if accu_test >= best_accuracy:
            best_accuracy = accu_test
            print('best model saved', end='')
            torch.save(model.state_dict(), os.path.join(model_save_dir, 'model'))
            
        
    sys.stdout.close()
            


if __name__ == '__main__':
    main()
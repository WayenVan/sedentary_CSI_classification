import click
import os
import sys
import json

from csi_catm.data.common import aggregate_3channel, random_split_data_list
from csi_catm.data.dataset import Catm3ChannelDataset
from csi_catm.models.channel3gru import ResRnn3C
from csi_catm.data.transform import DownSample
from torch.utils.data import DataLoader

from tqdm import tqdm
import torch
import torchmetrics
from einops import rearrange

@click.command()
@click.option('--data_root', default='dataset/CATM', type=str)
@click.option('--lr', default=1e-3, type=float)
@click.option('--batch', default=4, type=int)
@click.option('--epochs', default=100, type=int)
@click.option('--test_ratio', default=0.3)
@click.option('--device', default='cuda')
@click.option('--model_save_dir', default='models/channel3GRU')
@click.option('--down_sample', nargs=2, default=(3, 3), help='down sample (height, width)')
@click.option('--t_padding', default=100, type=int)
@click.option('--img_size', nargs=3, type=int, default=(1, 30, 30), help='channel height width')
@click.option('--d_model', default=8, type=int)
@click.option('--n_rnn_layers', default=4, type=int)
@click.option('--n_res_block', default=1, type=int)
@click.option('--dropout', default=0.1, type=float)
@click.option('--conv_channel', default=64)
def main(data_root, 
        lr,
        batch,
        epochs,
        t_padding,
        test_ratio,
        img_size,
        d_model,
        n_rnn_layers,
        dropout,
        n_res_block,
        device,
        model_save_dir,
        conv_channel,
        down_sample):
    info = dict(locals())

    file_list = os.listdir(data_root)
    file_list = aggregate_3channel(file_list)
    
    train_list, test_list = random_split_data_list(file_list, test_ratio)
    
    trans = DownSample(down_sample[0], down_sample[1])
    train_set, test_set = Catm3ChannelDataset(data_root, train_list, t_padding, transform=trans), \
        Catm3ChannelDataset(data_root, test_list, t_padding, transform=trans)
    train_loader, test_loader = DataLoader(train_set, batch_size=batch), DataLoader(test_set, batch_size=batch)
    
    """Create model
    """    
    
    model = ResRnn3C(d_model, img_size, 8, n_res_block, n_rnn_layers, conv_channel, dropout=dropout).to(device)
    
    """training
    """
    info['train_set'] = train_list
    info['test_set'] = test_list
    
    
    os.makedirs(model_save_dir, exist_ok=True)
    with open(os.path.join(model_save_dir, 'info.json'), 'w') as f:
        json.dump(info, f)
        
    log = open(os.path.join(model_save_dir, 'training.log'), 'w', 1)
    sys.stdout = log
    
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    best_accuracy = 0.
    for epoch in range(epochs):
        
        #train
        accuracy_train = torchmetrics.Accuracy(task='multiclass', num_classes=8).to(device)
        tbar = tqdm(train_loader, desc='epoch: '+str(epoch), file=sys.stdout)
        for index, batch_data in enumerate(tbar):           
            opt.zero_grad()
            
            #[b, t, c, h, w]
            x, labels = batch_data
            x = rearrange(x, 'b t c h w -> c t b h w')
            x = x.to(device)
            labels = labels.to(device)
            y_pred = model(x[0], x[1], x[2])
            loss = loss_fn(y_pred, labels)
            loss.backward()
            
            opt.step()
            acu = accuracy_train(y_pred, labels).item()
            tbar.set_postfix({
                'batch_accuracy': acu
            })

        #test
        accuracy_test = torchmetrics.Accuracy(task='multiclass', num_classes=8).to(device)
        
        for index, batch_data in enumerate(test_loader):
            x, labels = batch_data
            x = rearrange(x, 'b t c h w -> c t b h w')
            x = x.to(device)
            labels = labels.to(device)
            y_pred = model(x[0], x[1], x[2])
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
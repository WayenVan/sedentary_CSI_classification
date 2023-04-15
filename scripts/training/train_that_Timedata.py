import argparse
from cProfile import label
import os
import torch
import sys
import json
import torchmetrics

from einops import rearrange

from tqdm import tqdm

from torch.utils.data import DataLoader

from csi_catm.data.common import random_split_data_list
from csi_catm.data.dataset import TimeDataset

from csi_that.Model import HARTrans

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="training the THAT model with timedata dataset")
    
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--nclass', type=int, default=8)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--input_dim', type=int, default=61, help='the dim of model\'s inpu')
    parser.add_argument('--n_seq', type=int, default=300, help='the number of channel of the model\'s input')
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--sample', type=int, default=1)
    parser.add_argument('--hlayers', type=int, default=4)
    parser.add_argument('--vlayers', type=int, default=4)
    parser.add_argument('--hheads', type=int, default=4)
    parser.add_argument('--vheads', type=int, default=4)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--data_dir', default='dataset/timeData')
    parser.add_argument('--save_dir', default='models/THAT_Timedata')
    parser.add_argument('--log', type=bool, default=False, help='a bool value, if true, will print output to a log file, default=0')
    parser.add_argument('--col_select', default='dop_spec_ToF')
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
        
    
    """prepare data
    """
    list = os.listdir(args.data_dir)
    train_list, test_list = random_split_data_list(list, 0.2)
    
    trainset = TimeDataset(args.data_dir, train_list, 8, col_select=args.col_select, norm=True)
    testset = TimeDataset(args.data_dir, test_list, 8, col_select=args.col_select, norm=True)
    
    trainloader = DataLoader(trainset, batch_size=args.batch)
    testloader = DataLoader(testset, batch_size=args.batch)
    
    temp = trainset[0][0]
    n_seq = temp.shape[-2]
    input_dim = temp.shape[-1]
    
    assert (n_seq == args.n_seq) & (input_dim == args.input_dim), 'the input shape of dataset should match that of the model'
    
    """create model
    """
    model = HARTrans(args.K, args.sample, args.n_seq, args.input_dim , args.d_model, args.nclass ,args.hlayers, args.vlayers, args.hheads, args.vheads).to(device)
    
    
    """training
    """
    info = vars(args)
    info['train_set'] = train_list
    info['test_set'] = test_list
    
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, 'info.json'), 'w') as f:
        json.dump(info, f)
        
    if args.log:
        log = open(os.path.join(args.save_dir, 'training.log'), 'w', 1)
        sys.stdout = log
    
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    loss_fn = torch.nn.NLLLoss()
    
    best_accuracy = 0.
    for epoch in range(args.epoch):
        
        #train
        accuracy_train = torchmetrics.Accuracy(task='multiclass', num_classes=args.nclass).to(device)
        tbar = tqdm(trainloader, desc='epoch: '+str(epoch), file=sys.stdout)
        for index, batch_data in enumerate(tbar):           
            opt.zero_grad()
            
            x, labels = batch_data
            x = x.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.int64)
            
            y_pred = model(x)
            loss = loss_fn(y_pred, labels)
            loss.backward()
            
            opt.step()
            acu = accuracy_train(y_pred, labels).item()
            tbar.set_postfix({
                'batch_accuracy': acu
            })

        #test
        accuracy_test = torchmetrics.Accuracy(task='multiclass', num_classes=args.nclass).to(device)
        
        for index, batch_data in enumerate(testloader):
            x, labels = batch_data
            x = x.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.int64)
            y_pred = model(x)
            accuracy_test.update(y_pred, labels)
        
        accu_test = accuracy_test.compute().item()
        print('test_accuracy='+str(accu_test), end=', ')
        
        #save model
        if accu_test >= best_accuracy:
            best_accuracy = accu_test
            print('best model saved', end='')
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'model'))
            
        
    sys.stdout.close()
            
            
            
            
            
            
        
        
    

    
    
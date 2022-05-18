from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
from zmq import device
from dataset import BvPDataset, MyDataset

from torchinfo import summary

from models_tc.Vstm import Vstm
from models_tc.BvP import BvP
from models_tc.base import ImgGRU

import torch
import os

from einops import rearrange
from common import print_parameters_grad, print_parameters

# Parameters
model_select = 'bvp'
data_dir = r'dataset/BVP/6-link/user1'
# data_dir = r'dataset/DAM_nonToF/all0508'
model_dir = r'./saved_models'

ALL_MOTION = [0,1,2,3,4,5]
N_MOTION = len(ALL_MOTION)
T_MAX = 50
img_size = (1, 30, 30)

n_epochs = 100
n_batch_size = 32
n_test_batch_size = 64
f_learning_rate = 0.0001

fraction_for_test = 0.1
envrionment = (1,)
use_cuda = True
log_interval = 10
dry_run = False

device = torch.device("cuda" if use_cuda else "cpu")

def train(model, device, train_loader, loss_func, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        batch_size = len(data)
        data = torch.transpose(data, 0, 1)
        target = torch.argmax(target, -1)
        data = data.double()
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        loss.backward()
        # print_parameters_grad(model)
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if dry_run:
                break


def test(model, device, test_loader, loss_func):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            data = torch.transpose(data, 0, 1)
            target = torch.argmax(target, -1)

            data = data.double()
            output = model(data)
            test_loss += loss_func(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

#-----------------loading data-----------------#
data_list_origin = os.listdir(data_dir)

#select envioronment
data_list = data_list_origin
# for file_name in data_list_origin:
#     if int(file_name.split('-')[2]) in envrionment:
#         data_list.append(file_name)

data_len = len(data_list)
dataset = BvPDataset(data_dir, data_list, N_MOTION, T_MAX)
# Split train and test
test_number = round(data_len*fraction_for_test)
train_dataset, test_dataset = random_split(dataset, [data_len-test_number, test_number])

# Package the dataset
print('\nLoaded dataset of ' + str(len(train_dataset)) + ' samples')

train_dataloader = DataLoader(train_dataset, batch_size=n_batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=n_test_batch_size, shuffle=True)

#-----------load model-------------#
if model_select == "vstm":
    model = Vstm(d_model=N_MOTION, d_emb = 512, img_size=img_size,
        split_grid=(3,3), nhead=16, num_layer=4, LSTM_hidden_size=512,
         LSTM_num_layers=2, bidirectional=True).to(device)

if model_select == 'bvp':
    model = BvP(d_model=N_MOTION, img_size=img_size, gru_num_layers=8).to(device)

if model_select == 'img_gru':
    model = ImgGRU(d_model=N_MOTION, img_size=img_size, gru_num_layers=4, gru_hidden_size=128).to(device)


#----------train model-------------#
summary(model, input_data=torch.rand((T_MAX, 3, img_size[0], img_size[1], img_size[2]), dtype=torch.float64), device=device)
optimizer = torch.optim.Adam(model.parameters(), lr = f_learning_rate)

for epoch in range(1, n_epochs + 1):
    train(model, device, train_dataloader, F.cross_entropy, optimizer, epoch)
    test(model, device, test_dataloader, F.cross_entropy)

    

    


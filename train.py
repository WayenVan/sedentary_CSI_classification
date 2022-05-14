from pickletools import optimize
from tkinter import E
from torch import tensor
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from dataset import CustomDataset

from torchinfo import summary

from models.Vstm import Vstm
from models.BvP import BvP

import torch
import time
import os
import math

# Parameters
model_select = 'bvp'
fraction_for_test = 0.1
data_dir = r'dataset/BVP/6-link/user1'
# data_dir = r'dataset/DAM_nonToF/all0508'
model_dir = r'./saved_models'
ALL_MOTION = [0,1,2,3,4, 5]
N_MOTION = len(ALL_MOTION)
T_MAX = 50
n_epochs = 10
f_dropout_ratio = 0.5
n_batch_size = 32
f_learning_rate = 0.001
img_size = (1, 30, 30)
envrionment = (1,)


def train(model, dataloader, optimizer, lossFunc, in_cuda=False):

    for epoch in range(n_epochs):

        time_start = time.time()
        losses = []

        for i, data in enumerate (dataloader, 0):

            train, label = data
            train = torch.transpose(train, 0, 1)

            #[s, b, channel, height, width]
            if in_cuda:
                label = label.cuda()
                train = train.cuda()

            outputs = model(train)

            loss = lossFunc(outputs, label).requires_grad_()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(list(model.parameters())[-1][-1])
            #precision
            p = torch.argmax(outputs, dim=-1)
            l = torch.argmax(label, dim=-1)

            losses.append(loss)
            print("Epoch {:}/{:}, Batch {:}/{:} loss: {:}".format(epoch+1, n_epochs+1, i, 
            math.ceil(len(train_dataset)/n_batch_size),loss.item()))


data_list_origin = os.listdir(data_dir)

#select envioronment
data_list = data_list_origin
# for file_name in data_list_origin:
#     if int(file_name.split('-')[2]) in envrionment:
#         data_list.append(file_name)

data_len = len(data_list)

# Split train and test
test_number = round(data_len*fraction_for_test)
train_data_list, test_data_list = random_split(data_list, [data_len-test_number, test_number])

# Package the dataset
# dataset = CustomDataset(data_train, label_train)
train_dataset = CustomDataset(data_dir, train_data_list, N_MOTION, T_MAX)
test_dataset = CustomDataset(data_dir, test_data_list, N_MOTION, T_MAX)
print('\nLoaded dataset of ' + str(len(train_dataset)) + ' samples')


train_dataloader = DataLoader(train_dataset, batch_size=n_batch_size, shuffle=True)


#-----------load model-------------#
if model_select == "vstm":
    model = Vstm(d_model=N_MOTION, d_emb = 512, img_size=img_size,
        split_grid=(3,3), nhead=16, num_layer=4, LSTM_hidden_size=512,
         LSTM_num_layers=2, bidirectional=True).cuda()

if model_select == 'bvp':
    model = BvP(d_model=N_MOTION, img_size=img_size, gru_num_layers=4)


#----------train model-------------#
summary(model, input_data=torch.rand((T_MAX, 3, img_size[0], img_size[1], img_size[2]), dtype=torch.float64))
lossFunc = nn.CrossEntropyLoss(reduction="mean")
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
train(model, train_dataloader, optimizer, lossFunc)

    


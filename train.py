from torch import tensor
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from dataset import CustomDataset
from ViT import ViT
import torch
from common import load_data,onehot_encoding,zero_padding,normalize_data
import time
import numpy as np
import os

# Parameters
use_existing_model = False
fraction_for_test = 0.1
data_dir = r'./dataset/tmp'
model_dir = r'./saved_models'
ALL_MOTION = [1,2,3,4,5,6,7,8]
N_MOTION = len(ALL_MOTION)
T_MAX = 100
n_epochs = 1
f_dropout_ratio = 0.5
n_batch_size = 4
f_learning_rate = 0.001


data_list = os.listdir(data_dir)
data_len = len(data_list)

# Split train and test
test_number = round(data_len*fraction_for_test)
train_data_list, test_data_list = random_split(data_list, [data_len-test_number, test_number])

# Package the dataset
# dataset = CustomDataset(data_train, label_train)
train_dataset = CustomDataset(data_dir, train_data_list, N_MOTION)
test_dataset = CustomDataset(data_dir, test_data_list, N_MOTION)
print('\nLoaded dataset of ' + str(len(train_dataset)) + ' samples')

train_dataloader = DataLoader(train_dataset, batch_size=n_batch_size,shuffle=True)


# Load or fabricate model
if use_existing_model:
    model = torch.load('0510model.pth')

    print(model)
else:
    model = ViT(d_model=8, d_emb = 2048, img_size=(90,90,1),
        split_grid=(3,3), nhead=4, num_layer=3).cuda()
    print(model)
    model.train()
    lossFunc = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        time_start = time.time()

        for i, data in enumerate (train_dataloader, 0):
            train, label = data
            trainset = torch.unsqueeze(train, -1)
            Size = trainset.size()
            trainset = torch.reshape(trainset, (Size[0]*Size[1], Size[2], Size[3], Size[4])).cuda()
            label = label.cuda()
            label = label.type(torch.double)
            outputs = model.forward(trainset)
            outputs = outputs[:label.size()[0], :]
            loss = lossFunc(outputs, label)
            print(loss)
            loss.backward()

    
    
    # print('Saving trained model...')
    # model.save(os.path.join(model_dir,'model_widar3_trained.h5'))
    torch.save(model, os.path.join(model_dir, '0510model.pth'))

# x = torch.rand((10, 1500, 30, 30, 3), dtype=torch.float).cuda()
# model = ViT(2048, 2048, (30, 30, 3), 8, 12).cuda()
# x = model.forward(x)
# print(x)
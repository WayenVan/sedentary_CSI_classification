
from pickletools import optimize
from torch import tensor
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from dataset import CustomDataset
from models import Vstm
import torch
from common import load_data,onehot_encoding,zero_padding,normalize_data
import time
import os
import math
# Parameters
use_existing_model = False
fraction_for_test = 0.1
data_dir = r'dataset/BVP/6-link/user1'
model_dir = r'./saved_models'
ALL_MOTION = [0,1,2,3,4,5,6,7]
N_MOTION = len(ALL_MOTION)
T_MAX = 100
n_epochs = 10
f_dropout_ratio = 0.5
n_batch_size = 32
f_learning_rate = 0.001
envrionment = (1,)


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
train_dataset = CustomDataset(data_dir, train_data_list, N_MOTION)
test_dataset = CustomDataset(data_dir, test_data_list, N_MOTION)
print('\nLoaded dataset of ' + str(len(train_dataset)) + ' samples')


train_dataloader = DataLoader(train_dataset, batch_size=n_batch_size,shuffle=True)


# Load or fabricate model
if use_existing_model:
    model = torch.load('0510model.pth')

    print(model)
else:
    model = Vstm(d_model=8, d_emb = 512, img_size=(30,30,1),
        split_grid=(3,3), nhead=4, num_layer=4, LSTM_hidden_size=512, LSTM_num_layers=2).cuda()
    #print(model)
    model.train()
    lossFunc = nn.CrossEntropyLoss(reduction="sum")
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

    for epoch in range(n_epochs):
        time_start = time.time()

        losses = []
        for i, data in enumerate (train_dataloader, 0):
            optimizer.zero_grad()
            train, label = data
            train = torch.transpose(train, 0, 1)
            # Size = trainset.size()
            label = label.cuda()
            label = label .type(torch.float64)
            outputs = model.forward(train)
            # a = torch.sum(outputs, -1)
            loss = lossFunc(outputs, label)
            losses.append(loss)
            loss.backward()
            optimizer.step()
            print(list(model.parameters())[0])
            print("Epoch {:}/{:}, Batch {:}/{:} loss: {:}".format(epoch+1, n_epochs+1, i, math.ceil(len(train_dataset)/n_batch_size),loss.item()))
        
        

    
    
    # print('Saving trained model...')
    # model.save(os.path.join(model_dir,'model_widar3_trained.h5'))
    torch.save(model, os.path.join(model_dir, '0510model.pth'))

# x = torch.rand((10, 1500, 30, 30, 3), dtype=torch.float).cuda()
# model = ViT(2048, 2048, (30, 30, 3), 8, 12).cuda()
# x = model.forward(x)
# print(x)
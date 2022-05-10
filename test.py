import torch
from dataset import CustomDataset
from torch.utils.data import DataLoader
import numpy as np
data_dir = r'C:\Users\11853\Desktop\CSI bvp data\MVP\tmp'

def collate_fn(batch):
    res = list(zip(*batch))
    res1 = torch.stack(res[0])
    res2 = torch.stack(res[1])
    return res1, res2

dataset = CustomDataset(data_dir)
print(len(dataset))

train_dataloader = DataLoader(dataset, batch_size=32,shuffle=True, collate_fn=collate_fn)

# for data, label in dataset:
 #   print(data.size())
 #   print(label.size())
 

#local_data,local_label = next(iter(train_dataloader))
count = 0

for _,_1 in train_dataloader:
    print(_.size())
    print(_1.size())

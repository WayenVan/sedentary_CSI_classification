from csi_catm.models.reslstm import ResLSTM
from csi_catm.models.vitlstm import Vstm
from csi_catm.models.bvp import BvP
from csi_catm.models.cnn import CNN
from csi_that.Model import HARTrans
from csi_catm.data.dataset import CatmDataset
from csi_catm.data.common import random_split_data_list

from typing import Tuple
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary

import os
from einops import rearrange
import torch



data = torch.ones(size=(100, 1, 1, 30, 30))
data_bvp = torch.ones(size=(30, 1, 1, 20, 20))
data_that = torch.ones(size=(1, 3000, 90))
data_cnn = torch.ones(size=(1, 1, 3000, 60))

reslstm = ResLSTM(d_model=128, input_size=(1, 30, 30), n_class=8, n_res_block=1, n_lstm_layer=4, channel_size=32, kernel_size=3)
bvp = BvP(n_class=8, img_size=(1, 20, 20), gru_num_layers=4, gru_hidden_size=128)
that = HARTrans(K=10, sample=1, n_seq=3000, input_dim=90, d_model=128, n_class=8, hlayers=4, vlayers=4, hheads=4, vheads=4)
cnn = CNN(128, n_classes=8, img_size=(1, 3000, 60))


summary(reslstm, input_data=data)
summary(bvp, input_data=data_bvp)
summary(that, input_data=data_that)
summary(cnn, input_data=data_cnn)
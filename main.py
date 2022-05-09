import imp
from torch import tensor
from ViT import ViT
import torch
x = torch.rand((10, 1500, 30, 30, 3), dtype=torch.float).cuda()
model = ViT(2048, 2048, (30, 30, 3), 8, 12).cuda()
x = model.forward(x)
print(x)
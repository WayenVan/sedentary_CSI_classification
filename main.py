
from ViT import ViT
import torch
x = torch.rand((1500, 30, 30, 3), dtype=torch.float).cuda()
print(x.size())
model = ViT(2048, 2048, (30, 30, 3), (3, 3), 8, 12).cuda()
x = model.forward(x)
print(x.size())

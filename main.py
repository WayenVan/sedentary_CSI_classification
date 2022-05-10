
from ViT import ViT
import torch
from torchinfo import summary

x = torch.rand((1500, 30, 30, 3), dtype=torch.float).cuda()
print(x.size())
model = ViT(2048, 2048, (30, 30, 3), (3, 3), 8, 12).cuda()

summary(model, input_data=x)

x = model.forward(x)
print(x.size())

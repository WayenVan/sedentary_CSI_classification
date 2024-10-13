import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange, repeat
import torchvision.models as models
from torchvision.models import ResNet18_Weights


class Resnet(nn.Module):
    def __init__(self, drop_prob=0.1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.dropout = nn.Dropout2d(drop_prob)

    def forward(self, x):
        T = int(x.size(2))
        x = rearrange(x, "n c t h w -> (n t) c h w")
        x = self._forward_impl(x)
        x = rearrange(x, "(n t) c -> n c t", t=T)
        return x

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.dropout(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)

        return x


if __name__ == "__main__":
    model = Resnet()
    x = torch.randn(2, 3, 10, 224, 224)
    print(model(x).shape)

from torch import nn
from torch.nn import functional as F


class Loss(nn.Module):
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, input, target):
        input = F.log_softmax(input, dim=-1)
        return F.nll_loss(input, target, reduction='mean')
        
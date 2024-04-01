from torch import nn
from einops import rearrange
import torch


class BaseModel(nn.Module):
    
    
    def __init__(
        self, 
        visual_backbone: nn.Module,
        sequence_backbone: nn.Module,
        header: nn.Module,
        freeze_visual=False,
        freeze_sequence=False,
        freeze_header=False,
        *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.v_backbone = visual_backbone
        self.s_backbone = sequence_backbone
        self.header = header
        
        self.f_v = freeze_visual
        self.f_s = freeze_sequence
        self.f_h = freeze_header
    
    def forward(self, x):
        """
        [b, c, t, h, w]
        """

        #time distributed
        x = self.v_backbone(x)
        x = rearrange(x, 'n c t-> t n c')
        x = self.s_backbone(x)
        x = self.header(x)
        #b n_class

        return x

    def train(self, mode: bool = True):
        super().train(mode)

        for p in self.v_backbone.parameters():
            p.requires_grad = not self.f_v
        for p in self.s_backbone.parameters():
            p.requires_grad = not self.f_s
        for p in self.header.parameters():
            p.requires_grad = not self.f_h

from typing import Any
import torch
import torch.nn as nn

class Freezer:
    
    def __init__(self) -> None:
        pass
    
    def __call__(self, model: nn.Module) -> Any:
        for name, module in model.named_modules():
            if name == 'lstm':
                module.requires_grad_(False)
        return


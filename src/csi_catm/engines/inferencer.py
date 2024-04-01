import torchmetrics
from einops import rearrange, repeat
import torch.nn as nn
import torch
from tqdm import tqdm
from csi_catm.utils.misc import info, warn, is_debugging
from torchmetrics.functional.classification import accuracy

class Inferencer:

    def __init__(self, device, logger) -> None:
        self.device = device
        if logger is not None:
            self.logger = logger.getChild(str(__class__.__name__))
        else:
            self.logger = None
        
    def do_inference(self, model: nn.Module, loader):
        model.eval()
        
        """on begin of test"""
        accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=8).to(self.device)
        preds = []
        targets = []
        for index, batch_data in enumerate(tqdm(loader)):
            x, labels = batch_data
            x = repeat(x, 'b t h w -> b c t h w', c=3)
            x = x.to(self.device)
            labels = labels.to(self.device)
            with torch.inference_mode():
                with torch.no_grad():
                    y_pred = model(x)
                    preds.append(y_pred)
                    targets.append(labels)
        
        accu = accuracy(torch.concat(preds, dim=0), torch.concat(targets, dim=0)).item()
        return accu
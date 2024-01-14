import torchmetrics
from einops import rearrange
import torch.nn as nn
import torch
from tqdm import tqdm

class Inferencer:

    def __init__(self, device, logger) -> None:
        self.device = device
        self.logger = logger.getChild(str(__class__))
        
    def do_inference(self, model: nn.Module, loader):

        model.eval()
        
        """on begin of test"""
        accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=8).to(self.device)
        for index, batch_data in tqdm(enumerate(loader)):
            x, labels = batch_data
            x = rearrange(x, 'b t (c h) w -> t b c h w', c=1)
            x = x.to(self.device)
            labels = labels.to(self.device)
            with torch.no_grad():
                y_pred = model(x)
            accuracy.update(y_pred, labels)
        
        accu = accuracy.compute().item()
        return accu
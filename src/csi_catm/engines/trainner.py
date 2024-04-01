import torch
import torchmetrics
import numpy as np
from tqdm import tqdm
from einops import rearrange, repeat
from logging import Logger
from csi_catm.utils.misc import info, warn, is_debugging

class Trainner:
    
    def __init__(self, device, logger: Logger, debug=False) -> None:
        self.device = device
        self.debug = debug
        
        if logger is not None:
            self.logger = logger.getChild(str(__class__.__name__))
        else:
            self.logger = None
    
    def do_train(self, model, loss_fn, train_loader, opt):
        model.to(self.device)
        model.train(True)
        
        """on begin of train"""
        accuracy_train = torchmetrics.Accuracy(task='multiclass', num_classes=8).to(self.device)
        tbar = tqdm(train_loader)
        losses = []
        for index, batch_data in enumerate(tbar):           
            
            #[b, t, h, w]
            x, labels = batch_data
            x = repeat(x, 'b t h w -> b c t h w', c=3)

            x = x.to(self.device)
            labels = labels.to(self.device)
            y_pred = model(x)
            loss = loss_fn(y_pred, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            if self.debug:
                for name, parms in model.named_parameters():
                    if 'weight' not in name:
                        continue
                    warn(self.logger, "-->name: {} -->grad_requirs: {} -->grad_value: {}".format(name, parms.requires_grad, parms.grad))
            
            acu = accuracy_train(y_pred, labels).item()
            info(self.logger, f'batch index {index}, loss {loss.item()},  accuracy {acu}')
            losses.append(loss.item())
        return np.mean(losses)
    
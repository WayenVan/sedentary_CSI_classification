import torch
import torchmetrics
import numpy as np
from tqdm import tqdm
from einops import rearrange
from logging import Logger

class Trainner:
    
    def __init__(self, device, loss, logger: Logger, debug=False) -> None:
        self.device = device
        self.loss_fn = loss
        self.debug = debug
        self.logger = logger.getChild(str(__class__))
    
    def do_train(self, model, train_loader, opt):
        model.to(self.device)
        model.train(True)
        
        
        """on begin of train"""
        accuracy_train = torchmetrics.Accuracy(task='multiclass', num_classes=8).to(self.device)
        tbar = tqdm(train_loader)
        losses = []
        for index, batch_data in enumerate(tbar):           
            
            #[b, t, c, h, w]
            x, labels = batch_data
            x = rearrange(x, 'b t (c h) w -> t b c h w', c=1)
            x = x.to(self.device)
            
            labels = labels.to(self.device)
            y_pred = model(x)
            loss = self.loss_fn(y_pred, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            if self.debug:
                for name, parms in model.named_parameters():
                    if 'weight' not in name:
                        continue
                    print("-->name: {} -->grad_requirs: {} -->grad_value: {}".format(name, parms.requires_grad, parms.grad))
            
            acu = accuracy_train(y_pred, labels).item()
            self.logger.info(f'batch index {index}, loss {loss.item()},  accuracy {acu}')
            losses.append(loss.item)
        return np.mean(losses)
    
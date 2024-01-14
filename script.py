import hydra
from omegaconf import DictConfig
import torch
import os
import logging
from torch.utils.data.dataloader import DataLoader
from torch.nn import Module
from torch.optim import Optimizer
import shutil
from hydra.utils import instantiate
from csi_catm.engines.trainner import Trainner
from csi_catm.engines.inferencer import Inferencer
from csi_catm.data.common import aggregate_3channel, random_split_data_list
import sys
sys.path.append('src')
logger = logging.getLogger('main')

@hydra.main(version_base=None, config_path='../configs', config_name='train.yaml')
def main(cfg: DictConfig):
    # torch.cuda.manual_seed_all(cfg.seed)
    # torch.manual_seed(cfg.seed)
    script = os.path.abspath(__file__)
    
    shutil.copyfile(script, 'script.py')
    logger.info('building model and dataloaders')
    
    file_list = os.listdir(cfg.data_root)
    train_list, test_list = random_split_data_list(file_list, cfg.val_ratio)
    
    train_loader: DataLoader = instantiate(cfg.data.train_loader, dataset={'data_list': train_list})
    val_loader: DataLoader = instantiate(cfg.data.val_loader, dataset={'data_list': test_list})
    
    model: Module = instantiate(cfg.model)
    opt: Optimizer = instantiate(cfg.optimizer, model.parameters())
    loss = instantiate(cfg.loss_fn)
    
    logger.info('building trainner and inferencer')
    trainer: Trainner = instantiate(cfg.trainner, logger=logger)
    inferencer: Inferencer = instantiate(cfg.inferencer, logger=logger) 
    logger.info('training loop start')
    
    best_accu = 0.
    for epoch in range(cfg.epoch):
        logger.info(f'epoch {epoch}')
        mean_loss = trainer.do_train(model, train_loader, opt)
        logger.info(f'training finished, mean loss: {mean_loss}')
        accu = inferencer.do_inference(model, val_loader)
        if accu > best_accu:
            best_accu = accu
            torch.save(model, 'model.pth')
            logger.info(f'best model saved')
        logger.info(f'finish one epoch')
        
if __name__ == '__main__':
    main()
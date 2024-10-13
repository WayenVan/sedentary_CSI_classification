#! /usr/bin/env python3

from omegaconf import OmegaConf, DictConfig
import time
import sys
import logging
sys.path.append('src')
from hydra.utils import instantiate
import uuid
import torch
from torch.utils.data.dataloader import DataLoader
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler

import hydra
import os
import shutil
from torch import nn
import torch.distributed as dist
from torch.utils.data import DistributedSampler
from datetime import datetime
logger = logging.getLogger('main')
local_rank = int(os.getenv('LOCAL_RANK'))
device = f'cuda:{local_rank}'
assert local_rank is not None

from csi_catm.utils.git import save_git_hash, save_git_diff_to_file
from csi_catm.utils.misc import info, warn, is_debugging
from csi_catm.engines.inferencer import Inferencer
from csi_catm.engines.trainner import Trainner
from csi_catm.data.common import aggregate_3channel, random_split_data_list

@hydra.main(version_base='1.3.2', config_path='../configs', config_name='runs/train_cnn_lstm')
def main(cfg: DictConfig):
    global logger
    setup()

    is_main_rank = (local_rank == 0)
    print(is_main_rank)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.manual_seed(cfg.seed)

    if is_main_rank:
        current_time = datetime.now()
        file_name = os.path.basename(__file__)
        save_dir = os.path.join('outputs', file_name[:-3], current_time.strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(save_dir, exist_ok=False)
        if is_debugging():
            with open(os.path.join(save_dir, 'debug'), 'w'):
                pass
        info(logger, 'saving git info')
        save_git_hash(os.path.join(save_dir, 'git_version.bash'))
        save_git_diff_to_file(os.path.join(save_dir, 'changes.patch'))
        _setup_logger(logger, save_dir)
    else:
        logger = None
        f = open(os.devnull, 'w')
        sys.stdout = f

    info(logger, 'building model and dataloaders')
    
    #initialize data 
    model, loss_fn, train_loader, val_loader = build_model_and_data(cfg)

    #initialize record list
    metas = []
    train_id = uuid.uuid1()

    #load checkpoint
    if cfg.load_weights:
        info(logger, 'loading checkpoint')
        metas = load_checkpoints(cfg, model)
        _log_history(metas, logger)

    #!important, this train will set the parameter states in the model.
    model.train()
    opt, lr_scheduler, trainer, inferencer = build_engines(cfg, model)

    best_accu = metas[-1]['accuracy'] if len(metas) > 0 else 1000.
    for i in range(cfg.epoch):
        real_epoch = i
        #train
        train_loader.sampler.set_epoch(real_epoch)

        lr = lr_scheduler.get_last_lr()
        info(logger, f'epoch {real_epoch}, lr={lr}')

        start_time = time.time()
        mean_loss = trainer.do_train(model, loss_fn, train_loader, opt)
        train_time = time.time() - start_time
        info(logger, f'training finished, mean loss: {mean_loss},  total time: {train_time}')

        lr_scheduler.step()
        info(logger, f'finish one epoch')

        #validation
        if is_main_rank:
            pass
            accu = inferencer.do_inference(model.module, val_loader)
            info(logger, f'validation finished, accuracy: {accu}')

            #save essential informations 
            metas.append(dict(
                accuracy=accu,
                lr = lr,
                train_loss=mean_loss.item(),
                epoch=real_epoch,
                train_time=train_time,
                train_id=train_id
            ))
            
            
            if accu < best_accu:
                best_accu = accu
                torch.save({
                    'model_state': model.module.state_dict(),
                    'meta': metas
                    }, os.path.join(save_dir, 'checkpoint.pt'))
                info(logger, f'best checkpoint saved')
    
    cleanup()

def setup():
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group("gloo")

def cleanup():
    dist.destroy_process_group()

def build_model_and_data(cfg):
    #initialize data 
    file_list = os.listdir(cfg.data_root)
    print(file_list)
    train_list, test_list = random_split_data_list(file_list, cfg.val_ratio)

    train_set = instantiate(cfg.data.dataset.train, data_list=train_list)
    val_set = instantiate(cfg.data.dataset.val, data_list=test_list)
    
    train_sampler = DistributedSampler(train_set)

    train_loader: DataLoader = instantiate(cfg.data.loader.train, dataset=train_set, sampler=train_sampler)
    val_loader: DataLoader = instantiate(cfg.data.loader.val, dataset=val_set)

    #initialize trainning essential
    model: Module = instantiate(cfg.model).to(device)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(
        model, 
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True
        )
    
    loss_fn: nn.Module = instantiate(cfg.loss)

    return model, loss_fn, train_loader, val_loader

def load_checkpoints(cfg, model):
    info(logger, 'loading checkpoint')
    checkpoint = torch.load(cfg.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    metas = checkpoint['meta']
    return metas

def build_engines(cfg, model):
    opt: Optimizer = instantiate(cfg.optimizer, params=filter(lambda p: p.requires_grad, model.parameters()))
    lr_scheduler: LRScheduler = instantiate(cfg.lr_scheduler, opt)
    trainer: Trainner = instantiate(cfg.engines.trainner, logger=logger, device=device)
    inferencer: Inferencer = instantiate(cfg.engines.inferencer, logger=logger, device=device) 
    return opt, lr_scheduler, trainer, inferencer
    

def _log_history(metas, logger: logging.Logger):
    info(logger, '-----------showing training history--------------')
    for info in metas:
        info(logger, f"train id: {info['train_id']}")
        info(logger, f"epoch: {info['epoch']}")
        info(logger, "lr: {}, train loss: {}, train wer: {}, val wer: {}".format(info['lr'], info['train_loss'], info['train_wer'], info['val_wer']))
    info(logger, '-----------finish history------------------------')
    
def _setup_logger(l, save_dir):
    handler = logging.FileHandler(os.path.join(save_dir, 'train.log'))
    l.addHandler(handler)

    
if __name__ == '__main__':
    main()
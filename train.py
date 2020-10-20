# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Created Time: 7/12/2020 11:29 PM

import os
import argparse
import collections

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import model.pick as pick_arch_module
from data_utils import pick_dataset as pick_dataset_module

from data_utils.pick_dataset import BatchCollateFn
from parse_config import ConfigParser
from trainer import Trainer

import torch.nn as nn
import torch.optim as optim

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config: ConfigParser, local_master: bool, logger=None):
    # setup dataset and data_loader instances
    train_dataset = config.init_obj('train_dataset', pick_dataset_module)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) \
        if config['distributed'] else None

    is_shuffle = False if config['distributed'] else True
    train_data_loader = config.init_obj('train_data_loader', torch.utils.data.dataloader,
                                        dataset=train_dataset,
                                        sampler=train_sampler,
                                        shuffle=is_shuffle,
                                        collate_fn=BatchCollateFn())

    val_dataset = config.init_obj('validation_dataset', pick_dataset_module)
    val_data_loader = config.init_obj('val_data_loader', torch.utils.data.dataloader,
                                      dataset=val_dataset,
                                      collate_fn=BatchCollateFn())
    logger.info(f'Dataloader instances created. Train datasets: {len(train_dataset)} samples '
                f'Validation datasets: {len(val_dataset)} samples.') if local_master else None

    # build model architecture
    pick_model = config.init_obj('model_arch', pick_arch_module)
    logger.info(f'Model created, trainable parameters: {pick_model.model_parameters()}.') if local_master else None

    # build optimizer, learning rate scheduler.
    optimizer = config.init_obj('optimizer', torch.optim, pick_model.parameters())
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    logger.info('Optimizer and lr_scheduler created.') if local_master else None

    # print training related information
    logger.info('Max_epochs: {} Log_per_step: {} Validation_per_step: {}.'.
                format(config['trainer']['epochs'],
                       config['trainer']['log_step_interval'],
                       config['trainer']['val_step_interval'])) if local_master else None

    logger.info('Training start...') if local_master else None
    trainer = Trainer(pick_model, optimizer,
                      config=config,
                      data_loader=train_data_loader,
                      valid_data_loader=val_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()
    logger.info('Training end...') if local_master else None


def entry_point(config: ConfigParser):
    '''
    entry-point function for a single worker, distributed training
    '''

    local_world_size = config['local_world_size']

    # check distributed environment cfgs
    if config['distributed']:  # distributed gpu mode
        # check gpu available
        if torch.cuda.is_available():
            if torch.cuda.device_count() < local_world_size:
                raise RuntimeError(f'the number of GPU ({torch.cuda.device_count()}) is less than '
                                   f'the number of processes ({local_world_size}) running on each node')
            local_master = (config['local_rank'] == 0)
        else:
            raise RuntimeError('CUDA is not available, Distributed training is not supported.')
    else:  # one gpu or cpu mode
        if config['local_world_size'] != 1:
            raise RuntimeError('local_world_size must set be to 1, if distributed is set to false.')
        config.update_config('local_rank', 0)
        local_master = True
        config.update_config('global_rank', 0)

    logger = config.get_logger('train') if local_master else None
    if config['distributed']:
        logger.info('Distributed GPU training model start...') if local_master else None
    else:
        logger.info('One GPU or CPU training mode start...') if local_master else None

    if config['distributed']:
        # these are the parameters used to initialize the process group
        env_dict = {
            key: os.environ[key]
            for key in ('MASTER_ADDR', 'MASTER_PORT', 'RANK', 'WORLD_SIZE')
        }
        logger.info(f'[Process {os.getpid()}] Initializing process group with: {env_dict}') if local_master else None

        # init process group
        dist.init_process_group(backend='nccl', init_method='env://')
        config.update_config('global_rank', dist.get_rank())
        # info distributed training cfg
        logger.info(
            f'[Process {os.getpid()}] world_size = {dist.get_world_size()}, '
            + f'rank = {dist.get_rank()}, backend={dist.get_backend()}'
        ) if local_master else None

    # start train
    main(config, local_master, logger if local_master else None)

    # tear down the process group
    dist.destroy_process_group()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch PICK Distributed Training')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to be available (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags default type target help')  # CustomArgs.flags, CustomArgs.default
    options = [
        # CustomArgs(['--lr', '--learning_rate'], default=0.0001, type=float, target='optimizer;args;lr',
        #            help='learning rate (default: 0.0001)'),
        CustomArgs(['--bs', '--batch_size'], default=2, type=int, target='train_data_loader;args;batch_size',
                   help='batch size (default: 2)'),
        # CustomArgs(['--ng', '--n_gpu'], default=2, type=int, target='n_gpu',
        #            help='num of gpu (default: 2)'),
        CustomArgs(['-dist', '--distributed'], default='true', type=str, target='distributed',
                   help='run distributed training. (true or false, default: true)'),
        CustomArgs(['--local_world_size'], default=1, type=int, target='local_world_size',
                   help='the number of processes running on each node, this is passed in explicitly '
                        'and is typically either $1$ or the number of GPUs per node. (default: 1)'),
        CustomArgs(['--local_rank'], default=0, type=int, target='local_rank',
                   help='this is automatically passed in via torch.distributed.launch.py, '
                        'process will be assigned a local rank ID in [0, local_world_size-1]. (default: 0)')

    ]
    config = ConfigParser.from_args(args, options)
    # The main entry point is called directly without using subprocess, call by torch.distributed.launch.py
    entry_point(config)

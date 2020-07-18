# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Created Time: 7/12/2020 11:29 PM

import argparse
import collections

import numpy as np
import torch

import model.pick as pick_arch_module
from data_utils import pick_dataset as pick_dataset_module

from data_utils.pick_dataset import BatchCollateFn
from parse_config import ConfigParser
from trainer import Trainer

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger('train')

    # setup dataset and data_loader instances
    train_dataset = config.init_obj('train_dataset', pick_dataset_module)
    val_dataset = config.init_obj('validation_dataset', pick_dataset_module)
    train_data_loader = config.init_obj('train_data_loader', torch.utils.data.dataloader,
                                        dataset=train_dataset,
                                        collate_fn=BatchCollateFn())
    val_data_loader = config.init_obj('val_data_loader', torch.utils.data.dataloader,
                                      dataset=val_dataset,
                                      collate_fn=BatchCollateFn())

    # build model architecture
    pick_model = config.init_obj('model_arch', pick_arch_module)
    logger.info('Model trainable parameters: {}'.format(pick_model.model_parameters()))

    # build optimizer, learning rate scheduler.
    optimizer = config.init_obj('optimizer', torch.optim, pick_model.parameters())
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    # print training related information
    logger.info('Train datasets: {} samples Validation datasets: {} samples Max_epochs: {} Log_per_step: {} '
                'Validation_per_step: {}'.format(len(train_dataset), len(val_dataset),
                                                 config['trainer']['epochs'], config['trainer']['log_step_interval']
                                                 , config['trainer']['val_step_interval']))
    logger.info('Training start...')
    trainer = Trainer(pick_model, optimizer,
                      config=config,
                      data_loader=train_data_loader,
                      valid_data_loader=val_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()
    logger.info('Training end...')

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch PICK Training')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to be available (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='train_data_loader;args;batch_size'),
        CustomArgs(['--ng', '--n_gpu'], type=int, target='n_gpu')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)

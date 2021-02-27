# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import open3d as o3d  # prevent loading error

import sys
import os
import json
import logging
import torch
import yaml
# from omegaconf import OmegaConf

from easydict import EasyDict as edict

from lib.ddp_data_loaders import make_data_loader
import lib.multiprocessing as mpu
# import hydra

from lib.ddp_trainer import HardestContrastiveLossTrainer, PointNCELossTrainer

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format='%(asctime)s %(message)s', datefmt='%m/%d %H:%M:%S', handlers=[ch])

torch.manual_seed(0)
torch.cuda.manual_seed(0)

logging.basicConfig(level=logging.INFO, format="")

def get_trainer(trainer):
  if trainer == 'HardestContrastiveLossTrainer':
    return HardestContrastiveLossTrainer
  elif trainer == 'PointNCELossTrainer':
    return PointNCELossTrainer
  else:
    raise ValueError(f'Trainer {trainer} not found')

def load_config(config_path):
    if '.yaml' in config_path or '.yml' in config_path:
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    config = edict(config)
    return config


# @hydra.main(config_path='config', config_name='defaults.yaml')
def main(config):
  logger = logging.getLogger()
  # if config.misc.config:
  #   resume_config = OmegaConf.load(config.misc.config)
  #   if config.misc.weight:
  #     weight = config.misc.weight
  #     config = resume_config
  #     config.misc.weight = weight
  #   else:
  #     config = resume_config

  logging.info('===> Configurations')
  logging.info(config)

  # Convert to dict
  if config.misc.num_gpus > 1:
      mpu.multi_proc_run(config.misc.num_gpus,
              fun=single_proc_run, fun_args=(config,))
  else:
      single_proc_run(config)

def single_proc_run(config):
  train_loader = make_data_loader(
      config,
      config.trainer.batch_size,
      num_threads=config.misc.train_num_thread)

  Trainer = get_trainer(config.trainer.trainer)
  trainer = Trainer(
      config=config,
      data_loader=train_loader,
  )
  trainer.train()


if __name__ == "__main__":
  config = load_config('./config/defaults.yaml')
  main(config)

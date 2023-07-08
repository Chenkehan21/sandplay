import os
from os import mkdir
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

import torch.nn.functional as F
import torch.nn as nn
import sys
sys.path.append('../')
from data import build_dataloader
from engine import do_train
from model import build_model
from optimizer import build_optimizer
from utils.logger import setup_logger


def train(cfg, logger):
    model = build_model(cfg)
    optimizer = build_optimizer(cfg, model)
    train_loader = build_dataloader(cfg, is_train=True, is_test=False)
    val_loader = build_dataloader(cfg, is_train=False, is_test=False)

    do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        nn.CrossEntropyLoss(),
        logger
    )


@hydra.main(version_base=None, config_path="/raid/ckh/sandplay_homework/configs", config_name="train_sandplay.yaml")
def main(cfg: DictConfig):
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    # import pdb;pdb.set_trace()
    output_dir = cfg.checkpoint.state_dict_dir
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)

    logger = setup_logger("sandplay", save_dir=None, distributed_rank=0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info("Running with config:\n{}".format(cfg))

    train(cfg, logger)


if __name__ == '__main__':
    main()
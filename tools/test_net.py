import torch
import sys; sys.path.append("../")
import hydra
from omegaconf import DictConfig
from data import build_dataloader
from engine import inference
from model import build_model
from ignite.handlers import Checkpoint


@hydra.main(version_base=None, config_path="/raid/ckh/sandplay_homework/configs", config_name="train_sandplay.yaml")
def main(cfg: DictConfig):
    model = build_model(cfg)
    checkpoint = torch.load(cfg.test.weight)
    Checkpoint.load_objects(to_load={"model": model}, checkpoint=checkpoint)
    val_loader = build_dataloader(cfg, is_train=False, is_test=True)

    inference(cfg, model, val_loader)


if __name__ == '__main__':
    main()
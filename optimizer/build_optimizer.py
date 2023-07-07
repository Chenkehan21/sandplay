import torch


def build_optimizer(cfg, model):
    optimizer = getattr(torch.optim, cfg.optimizer.name)(params=model.parameters(), lr=cfg.optimizer.lr)
    
    return optimizer
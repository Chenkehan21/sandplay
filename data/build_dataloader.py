from torch.utils.data import DataLoader
from .transforms.build_transforms import build_transforms
from .datasets.sandplay_dataset import SandplayDataset


def build_dataset(cfg, is_train, is_test=False):
    transforms = build_transforms(cfg)
    img_dir_path = cfg.dataset.img_dir_path
    img_names_path = cfg.dataset.img_names_path
    label_path = cfg.dataset.label_path
    partition = cfg.dataset.partition
    dataset = SandplayDataset(img_dir_path, img_names_path, label_path, partition, transforms, is_train, is_test)
    
    return dataset


def build_dataloader(cfg, is_train=True, is_test=False):
    if is_train:
        batch_size = cfg.train.batch_size
        shuffle = True
    else:
        batch_size = cfg.train.batch_size
        shuffle = False
    dataset = build_dataset(cfg, is_train, is_test)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=cfg.dataset.num_workers)
    
    return dataloader
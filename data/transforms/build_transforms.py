from torchvision.transforms import Compose
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from torchvision.models import ResNet18_Weights
import clip


def build_transforms(cfg):
    model_name = cfg.model.name
    cnn_input_shape = cfg.model.cnn_input_shape
    if model_name == "vit":
        _, transforms = clip.load("ViT-B/16")
    elif model_name == "resnet18":
        transforms = ResNet18_Weights.DEFAULT.transforms(antialias=True)
    elif model_name == "cnn":
        transforms = Compose([
                T.Resize(tuple(cnn_input_shape[1:])),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                T.ToTensor(), 
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return transforms
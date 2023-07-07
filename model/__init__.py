from .models import Vit, Resnet18, CNN, KNN, KMeans
import torch


def build_model(cfg):
    device = torch.device(cfg.model.device if torch.cuda.is_available() else "cpu")
    model_name = cfg.model.name
    num_classes = cfg.model.num_classes
    pretrained = cfg.model.pretrained
    cnn_input_shape = tuple(cfg.model.cnn_input_shape)
    if model_name == "resnet18":
        model = Resnet18(pretrained, num_classes)
    elif model_name == "vit":
        model = Vit(num_classes)
    elif model_name == "cnn":
        model = CNN(input_shape=cnn_input_shape, num_classes=num_classes)
    elif model_name == "knn":
        model = KNN(num_classes)
    elif model_name == "kmeans":
        model = KMeans(num_classes)
    
    return model.to(device)
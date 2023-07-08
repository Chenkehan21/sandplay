import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
import clip
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans


class Vit(nn.Module):
    def __init__(self, num_classes=6) -> None:
        super().__init__()
        clip_model_name = "ViT-B/16"
        clip_model, _ = clip.load(clip_model_name)
        self.img_encode = clip_model
        for param in self.img_encode.parameters():
            param.requires_grad = False
        self.linear = nn.Linear(512, num_classes)
    
    def forward(self, imgs):
        out = self.img_encode.encode_image(imgs).float()
        out = F.relu(out)
        out = self.linear(out)
        
        return out
    

class Resnet18(nn.Module):
    def __init__(self, pretrained=True, num_classes=6) -> None:
        super().__init__()
        if pretrained:
            self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
            for param in self.resnet.parameters():
                param.requires_grad = False
        else:
            self.resnet = resnet18(weights=None)
        num_features = self.resnet.fc.in_features
        self.linear = nn.Linear(num_features, num_classes)
        self.resnet.fc = self.linear
    
    def forward(self, imgs):
        return self.resnet(imgs)
    
    
class CNN(nn.Module):
    def __init__(self, input_shape=(3, 224, 224), num_classes=6) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.conv = nn.Sequential(
            nn.Conv2d(3, 5, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),
            nn.Flatten()
        )
        shape = self.conv_forward()
        self.linear = nn.Linear(shape, num_classes)
        
    def conv_forward(self):
        x = torch.randn(self.input_shape).unsqueeze(0)
        res = self.conv(x)
        
        return res.numel()
    
    def forward(self, x):
        return self.linear(self.conv(x))
import torch.nn as nn
from torchvision.models import resnet50


class ResNet50Binary(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet50(weights="IMAGENET1K_V2")
        # 替换最后全连接层为 2 分类
        self.model.fc = nn.Linear(2048, 2)

    def forward(self, x):
        return self.model(x)
import torch.nn.functional as F
import torch

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1.5, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce)  # pt 越小表示越难样本
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce
        return focal_loss.mean()

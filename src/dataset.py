import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def load_data(batch_size=16):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225])
    ])

    train_set = datasets.ImageFolder('data/2-MedImage-TrainSet', transform=transform)
    test_set = datasets.ImageFolder('data/2-MedImage-TestSet', transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


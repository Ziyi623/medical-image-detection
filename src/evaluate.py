import torch
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from dataset import load_data
from model import ResNet50Binary
import os
import numpy as np


def evaluate():
    _, test_loader = load_data(batch_size=8)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    model = ResNet50Binary().to(device)
    model.load_state_dict(torch.load('outputs/models/resnet50_medimg.pth', map_location=device))
    model.eval()


    all_labels = []
    all_preds = []


    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # 报告
    print(classification_report(all_labels, all_preds, target_names=['normal','disease']))

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['normal','disease'], yticklabels=['normal','disease'])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    if not os.path.exists('outputs/img'):
        os.makedirs('outputs/img')
    plt.savefig('outputs/img/confusion_matrix.png')
    plt.close()
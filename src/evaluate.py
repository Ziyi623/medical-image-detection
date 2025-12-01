import torch
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
from dataset import load_data
from model import ResNet50Binary
import os
import numpy as np
import torch.nn.functional as F


def plot_roc_curve(labels, probs, save_path):
    """绘制 ROC 曲线并保存"""
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")

    plt.savefig(save_path)
    plt.close()

    print(f"[ROC] AUC = {roc_auc:.4f}")
    print(f"[ROC] Curve saved to {save_path}")


def evaluate(if_plot_roc=1):
    _, test_loader = load_data(batch_size=8)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ResNet50Binary().to(device)
    model.load_state_dict(torch.load('outputs/models/resnet50_medimg.pth', map_location=device))
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []   # disease 的概率

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)

            # softmax 取 disease 类的概率（类索引=1）
            probs = F.softmax(outputs, dim=1)[:, 1]

            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # 分类报告
    print(classification_report(all_labels, all_preds, target_names=['normal', 'disease']))

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['normal', 'disease'],
        yticklabels=['normal', 'disease']
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    if not os.path.exists('outputs/img'):
        os.makedirs('outputs/img')
    plt.savefig('outputs/img/confusion_matrix.png')
    plt.close()

    # ===========================
    # ROC / AUC（开关控制）
    # ===========================
    if if_plot_roc == 1:
        roc_save_path = "outputs/img/roc_curve.png"
        plot_roc_curve(np.array(all_labels), np.array(all_probs), roc_save_path)

import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from dataset import load_data
from model import ResNet50Binary
import os
from FocalLoss import FocalLoss

def train_model():
    train_loader, test_loader = load_data(batch_size=8)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    model = ResNet50Binary().to(device)
    # 假设 normal=0, disease=1
    criterion = FocalLoss(alpha=2.0, gamma=2).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)


    num_epochs = 10
    loss_history = []


    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0


        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)


            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


            running_loss += loss.item()


        epoch_loss = running_loss / len(train_loader)
        loss_history.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}")

    # 保存曲线
    if not os.path.exists('outputs/img'): os.makedirs('outputs/img')
    plt.plot(loss_history)
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('outputs/img/loss_curve.png')
    plt.close()
    # 保存模型
    if not os.path.exists('outputs/models'): os.makedirs('outputs/models')
    torch.save(model.state_dict(), 'outputs/models/resnet50_medimg.pth')
    print("Training completed.")


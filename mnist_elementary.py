import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np

# --------------- 数据集定义 -----------------
class MNISTDataset(Dataset):
    """
    自定义的 PyTorch Dataset，用于加载 MNIST 数据。
    如果 is_test=True，则只加载图片，不加载标签（用于测试集）。
    """
    def __init__(self, df, is_test=False):
        self.is_test = is_test
        if is_test:
            # 测试集没有标签，只包含图片像素
            self.images = df.values.astype(np.float32).reshape(-1, 1, 28, 28) / 255.0  # 归一化到[0,1]
            self.labels = None
        else:
            # 训练/验证集包含标签
            self.labels = df['label'].values.astype(np.int64)
            self.images = df.drop(columns=['label']).values.astype(np.float32).reshape(-1, 1, 28, 28) / 255.0

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.is_test:
            return torch.tensor(self.images[idx])
        else:
            return torch.tensor(self.images[idx]), torch.tensor(self.labels[idx])

# --------------- 卷积神经网络模型定义 -----------------
class SimpleCNN(nn.Module):
    """
    一个简单的卷积神经网络，用于手写数字识别。
    包括两层卷积 + ReLU激活 + 最大池化 + Dropout，后接全连接层。
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # 输入1通道，输出32通道，卷积核3x3，padding=1
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 继续卷积，输出64通道
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                           # 2x2最大池化，尺寸减半
            nn.Dropout(0.25)                              # 防止过拟合
        )
        self.fc = nn.Sequential(
            nn.Flatten(),                                 # 拉平成一维向量
            nn.Linear(64*14*14, 128),                     # 全连接到128维
            nn.ReLU(),
            nn.Dropout(0.5),                              # 再次Dropout
            nn.Linear(128, 10)                            # 输出10类概率
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# --------------- 训练函数 -----------------
def train(model, device, train_loader, optimizer, criterion, epoch):
    """
    单个epoch的训练流程。
    """
    model.train()  # 设置为训练模式
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()           # 梯度清零
        outputs = model(data)           # 前向传播
        loss = criterion(outputs, target) # 计算损失
        loss.backward()                 # 反向传播
        optimizer.step()                # 更新参数
        running_loss += loss.item()
    print(f"Epoch {epoch}, Loss: {running_loss/len(train_loader):.4f}")

# --------------- 验证函数 -----------------
def validate(model, device, val_loader):
    """
    在验证集上评估模型准确率。
    """
    model.eval()   # 设置为评估模式
    correct = 0
    total = 0
    with torch.no_grad():  # 不计算梯度
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, preds = torch.max(outputs, 1)   # 取概率最大的类别
            correct += (preds == target).sum().item()
            total += target.size(0)
    acc = correct / total
    print(f"Validation Accuracy: {acc:.4f}")
    return acc

# --------------- 主流程 -----------------
def main():
    # 加载Kaggle的训练和测试数据
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    # 划分部分训练集为验证集（常用做法，便于调参）
    from sklearn.model_selection import train_test_split
    train_data, val_data = train_test_split(train_df, test_size=0.1, random_state=42)

    # 创建Dataset和DataLoader
    train_dataset = MNISTDataset(train_data)
    val_dataset = MNISTDataset(val_data)
    test_dataset = MNISTDataset(test_df, is_test=True)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # 实例化模型、损失函数与优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()           # 多分类交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器

    # 训练模型
    epochs = 100
    best_acc = 0.0
    for epoch in range(1, epochs+1):
        train(model, device, train_loader, optimizer, criterion, epoch)
        acc = validate(model, device, val_loader)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "./weights/elementary_cnn.pth")  # 保存最优模型

    # 用最佳模型对测试集预测
    model.load_state_dict(torch.load("./weights/elementary_cnn.pth"))
    model.eval()
    predictions = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())

    # 生成提交文件
    submission = pd.DataFrame({
        "ImageId": np.arange(1, len(predictions)+1),
        "Label": predictions
    })
    submission.to_csv("./data/elementary.csv", index=False)
    print("预测结果已保存到 elementary.csv")

if __name__ == "__main__":
    main()
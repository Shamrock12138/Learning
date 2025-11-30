import torch as tc
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class SimpleCNN(nn.Module):
  def __init__(self):
    super().__init__()
    # 定义卷积层：输入1通道，输出32通道，卷积核大小3x3
    self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1)
    # 定义卷积层：输入32通道，输出64通道
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
    # 定义全连接层
    self.fc1 = nn.Linear(64*7*7, 128)
    self.fc2 = nn.Linear(128, 10)
  def forward(self, x):
    x = F.relu(self.conv1(x))      # 卷积层1 + ReLU激活
    x = F.max_pool2d(x, 2)         # 2x2最大池化
    x = F.relu(self.conv2(x))      # 卷积层2 + ReLU激活
    x = F.max_pool2d(x, 2)         # 2x2最大池化
    x = x.view(-1, 64*7*7)         # 展平张量
    x = F.relu(self.fc1(x))        # 全连接层1 + ReLU激活
    x = self.fc2(x)                # 全连接层2
    return x
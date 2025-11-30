import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset

# print(f'Using device: {device}')

class BasicBlock(nn.Module):
  def __init__(self, inchannels, outchannels, stride=1):
    super().__init__()
    self.left_path = nn.Sequential(
      nn.Conv2d(inchannels, outchannels, kernel_size=3, stride=stride, padding=1, bias=False),
      nn.BatchNorm2d(outchannels),
      nn.ReLU(inplace=True),
      nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(outchannels)
    )
    if stride != 1 or inchannels != outchannels:
      self.shortcut = nn.Sequential(
        nn.Conv2d(inchannels, outchannels, kernel_size=1, stride=stride, bias=False),   # 1x1卷积
        nn.BatchNorm2d(outchannels)
      )
    else:
      self.shortcut = nn.Sequential()

  def forward(self, x):
    out = self.left_path(x)
    out += self.shortcut(x)
    out = nn.ReLU(inplace=True)(out)
    return out
    
class Bottleneck(nn.Module):
  def __init__(self, inchannels, outchannels, stride=1):
    super().__init__()
    self.left_path = nn.Sequential(
      nn.Conv2d(inchannels, int(outchannels/4), kernel_size=1, stride=stride, padding=0, bias=False),  # 压缩维度
      nn.BatchNorm2d(int(outchannels/4)),
      nn.ReLU(inplace=True),
      nn.Conv2d(int(outchannels/4), int(outchannels/4), kernel_size=3, stride=1, padding=1, bias=False),  # 卷积处理
      nn.BatchNorm2d(int(outchannels/4)),
      nn.ReLU(inplace=True),
      nn.Conv2d(int(outchannels/4), outchannels, kernel_size=1, stride=1, bias=False),  # 恢复维度
      nn.BatchNorm2d(outchannels)
    )
    if stride != 1 or inchannels != outchannels:
      self.shortcut = nn.Sequential(
        nn.Conv2d(inchannels, outchannels, kernel_size=1, stride=stride, bias=False),   # 1x1卷积
        nn.BatchNorm2d(outchannels)
      )
    else:
      self.shortcut = nn.Sequential()

  def forward(self, x):
    out = self.left_path(x)
    out += self.shortcut(x)
    out = nn.ReLU(inplace=True)(out)
    return out
  
class ResNet(nn.Module):
  def __init__(self, block, num_blocks, num_classes=1000):
    super().__init__()
    self.in_channels = 64

    self.conv1 = nn.Sequential(
      nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=3, bias=False),
      nn.BatchNorm2d(self.in_channels),
      nn.ReLU(inplace=True),
    )
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    self.layer1 = self._make_layer(block, self.in_channels*2, num_blocks[0], stride=1)
    self.layer2 = self._make_layer(block, self.in_channels*2, num_blocks[1], stride=2)
    self.layer3 = self._make_layer(block, self.in_channels*2, num_blocks[2], stride=2)
    self.layer4 = self._make_layer(block, self.in_channels*2, num_blocks[3], stride=2)
    
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(self.in_channels, num_classes)
    self._init_weights()

  def _make_layer(self, block, outchannels, num_blocks, stride):
    strides = [stride]+[1]*(num_blocks-1)   # 第一个block可能需要下采样，后续的block不需要
    layers = []
    for stride in strides:
      layers.append(block(self.in_channels, outchannels, stride))
      self.in_channels = outchannels
    return nn.Sequential(*layers)
  
  def _init_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
      
  def forward(self, x):
    # x: [batch_size, 3, 224, 224]
    x = self.conv1(x)       # kernel=7x7, stride=2, padding=3 -> [batch_size, 64, 112, 112]
    x = self.maxpool(x)     # -> [batch_size, 64, 56, 56]

    x = self.layer1(x)      # -> [batch_size, 128, 56, 56]
    x = self.layer2(x)      # -> [batch_size, 256, 28, 28]
    x = self.layer3(x)      # -> [batch_size, 512, 14, 14]
    x = self.layer4(x)      # -> [batch_size, 1024, 7, 7]

    x = self.avgpool(x)     # -> [batch_size, 512, 1, 1]
    x = torch.flatten(x, 1) # -> [batch_size, 512]
    x = self.fc(x)          # -> [batch_size, num_classes]
    return x

# 论文中的ResNet变体
def ResNet18(num_classes=1000):
  return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)
def ResNet34(num_classes=1000):
  return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)
def ResNet50(num_classes=1000):
  return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)
def ResNet101(num_classes=1000):
  return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)
def ResNet152(num_classes=1000):
  return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)

from tqdm import tqdm
import matplotlib.pyplot as plt
import time

# CIFAR10数据集
class CIFAR10Dataset(Dataset):
  def __init__(self, root, transform=None, train=True):
    self.dataset = datasets.CIFAR10(root=root, train=train, 
                                    download=True, transform=transform)

  def __len__(self):
    return len(self.dataset)
  
  def __getitem__(self, idx):
    return self.dataset[idx]

class DataLoaderFactory:
  def __init__(self, batch_size=32, num_workers=2):
    self.batch_size = batch_size
    self.num_workers = num_workers
  
  def get_transform(self):
    train_transform = transforms.Compose([
      # transforms.RandomResizedCrop(224),
      transforms.Resize((224, 224)),
      transforms.RandomHorizontalFlip(p=0.5),
      # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
      transforms.Resize((224, 224)),
      # transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform
  
  def get_dataloaders(self, root):
    train_transform, val_transform = self.get_transform()
    train_dataset = CIFAR10Dataset(root=root, transform=train_transform, train=True)
    val_dataset = CIFAR10Dataset(root=root, transform=val_transform, train=False)
    train_loader = DataLoader(
      train_dataset,
      batch_size=self.batch_size,
      shuffle=True,
      num_workers=self.num_workers
    )
    val_loader = DataLoader(
      val_dataset,
      batch_size=self.batch_size,
      shuffle=False,
      num_workers=self.num_workers
    )
    return train_loader, val_loader

class Trainer:
  def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
               criterion: nn.Module, optimizer: optim.Optimizer, scheduler=None):
    self.model = model
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.criterion = criterion
    self.optimizer = optimizer
    self.scheduler = scheduler
    self.train_losses = []
    self.val_losses = []
    self.train_accuracies = []
    self.val_accuracies = []
    self.best_val_accuracy = 0.0
  
  def train_epoch(self):
    self.model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(self.train_loader, desc='Training', unit='batch')
    for batch_idx, (inputs, targets) in enumerate(pbar):
      inputs, targets = inputs.to(device), targets.to(device)

      self.optimizer.zero_grad()
      outputs = self.model(inputs)
      loss = self.criterion(outputs, targets)
      loss.backward()
      self.optimizer.step()

      running_loss += loss.item()*inputs.size(0)
      _, predicted = outputs.max(1)
      total += targets.size(0)
      correct += predicted.eq(targets).sum().item()
      pbar.set_postfix({'loss': running_loss/total, 'accuracy': 100.*correct/total})
    
    epoch_loss = running_loss / total
    epoch_accuracy = 100.*correct/total
    return epoch_loss, epoch_accuracy

  def validate_epoch(self):
    self.model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
      pbar = tqdm(self.val_loader, desc='Validating', unit='batch')
      for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)

        running_loss += loss.item()*inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        pbar.set_postfix({'loss': running_loss/total, 'accuracy': 100.*correct/total})

    epoch_loss = running_loss / total
    epoch_accuracy = 100.*correct/total
    return epoch_loss, epoch_accuracy

  def train(self, num_epochs):
    for epoch in range(num_epochs):
      print(f'Epoch [{epoch+1}/{num_epochs}]')
      train_loss, train_accuracy = self.train_epoch()
      val_loss, val_accuracy = self.validate_epoch()

      self.train_losses.append(train_loss)
      self.val_losses.append(val_loss)
      self.train_accuracies.append(train_accuracy)
      self.val_accuracies.append(val_accuracy)

      # if val_accuracy > self.best_val_accuracy:
      #   self.best_val_accuracy = val_accuracy
      #   torch.save(self.model.state_dict(), 'best_resnet_model.pth')
      #   print('Best model saved!')

      if self.scheduler:
        self.scheduler.step()
      
      print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
      print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

  def evaluate(self, test_loader):
    self.model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
      for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = self.model(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    accuracy = 100.*correct/total
    return accuracy

def plot_train_history(trainer):
  epochs = range(1, len(trainer.train_losses)+1)

  plt.figure(figsize=(12, 5))

  plt.subplot(1, 2, 1)
  plt.plot(epochs, trainer.train_losses, 'b-', label='Training Loss')
  plt.plot(epochs, trainer.val_losses, 'r-', label='Validation Loss')
  plt.title('Loss over Epochs')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()

  plt.subplot(1, 2, 2)
  plt.plot(epochs, trainer.train_accuracies, 'b-', label='Training Accuracy')
  plt.plot(epochs, trainer.val_accuracies, 'r-', label='Validation Accuracy')
  plt.title('Accuracy over Epochs')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy (%)')
  plt.legend()

  plt.show()

if __name__ == "__main__":
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  torch.manual_seed(42)
  num_classes = 10
  batch_size = 64
  learning_rate = 0.01
  momentum = 0.9
  weight_decay = 5e-4
  epochs = 30
  step_size = 30
  gamma = 0.1
  num_workers = 2

  print('loading data...')
  data_loader_factory = DataLoaderFactory(batch_size=batch_size, num_workers=num_workers)
  train_loader, val_loader = data_loader_factory.get_dataloaders(root='./data')

  print('initializing model...')
  model = ResNet18(num_classes=num_classes).to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(
    model.parameters(),
    lr=learning_rate,
    momentum=momentum,
    # weight_decay=weight_decay
  )
  scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=step_size,
    gamma=gamma
  )

  print('starting training...')
  trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler
  )
  start_time = time.time()
  trainer.train(num_epochs=epochs)
  end_time = time.time()
  print(f'Training completed in: {end_time - start_time:.2f} seconds')
  plot_train_history(trainer)



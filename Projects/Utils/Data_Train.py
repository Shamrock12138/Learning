#                         数据处理&训练模块
#                           2025/10/24
#                            shamrock

import json, os
config_path = os.path.join(os.path.dirname(__file__), 'config.json')
print(config_path)

with open(config_path, 'r', encoding='utf-8') as f:
  config = json.load(f)
data_path = config['data_path']

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
import matplotlib.pyplot as plt

class Utils_TransformdDatasetFactory:
  '''
    支持动态 transform 的 Dataset 包装器，对 dataset 进行 transform
  '''
  def __init__(self, dataset, transform=None):
    self.dataset = dataset
    self.transform = transform
  
  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    X, y = self.dataset[idx]
    if self.transform:
      X = self.transform(X)
    return X, y

class Utils_DatasetFactory(Dataset):
  '''
    Dataset封装函数
  '''
  def __init__(self):
    super().__init__()
    self.X, self.y = [], []

  def load_data(self):
    '''
      用户实现加载原始数据到 self.X self.y 中
    '''
    print('\033[91mUtils_DatasetFactory `load_data` is NULL\033[0m')
    raise NotImplementedError('Utils_DatasetFactory must ensure `load_data`.')

  def load_rawData(self):
    '''
      用户实现加载原始数据到 self.raw_data 中
    '''
    print('\033[91mUtils_DatasetFactory `load_data` is NULL\033[0m')
    raise NotImplementedError('Utils_DatasetFactory must ensure `load_data`.')

  def divide_rawData(self, train_ratio=0.8):
    '''
      将 self.data_raw 划分为训练数据和测试数据，返回训练、测试数据 train_X, train_y, eval_X, eval_y
    '''
    train_size = int(len(self.X)*train_ratio)
    self.train_X, self.eval_X = self.X[:train_size], self.X[train_size:]
    self.train_y, self.eval_y = self.y[:train_size], self.y[train_size:]
    return self.train_X, self.train_y, self.eval_X, self.eval_y
  
  def get_dataloader(self, batch_size=32, shuffle=True, num_workers=0):
    train_dataset = TensorDataset(
      torch.tensor(self.train_X, dtype=torch.float32).unsqueeze(-1),
      torch.tensor(self.train_y, dtype=torch.float32)
    )
    eval_dataset = TensorDataset(
      torch.tensor(self.eval_X, dtype=torch.float32).unsqueeze(-1),
      torch.tensor(self.eval_y, dtype=torch.float32)
    )
    self.train_loader = DataLoader(train_dataset, batch_size, shuffle, num_workers=num_workers)
    self.eval_loader = DataLoader(eval_dataset, batch_size, False, num_workers=num_workers)
    return self.train_loader, self.eval_loader

  def __len__(self):
    return len(self.X)

  def __getitem__(self, idx):
    return self.X[idx], self.y[idx]

# class Utils_DataLoaderFactory:
#   '''
#     （弃用）
#     数据加载器工厂，封装DataLoader
#   '''
#   def __init__(self):
#     pass

#   def get_dataloders(self, train_data:Dataset, eval_data:Dataset, 
#                      batch_size=8, num_workers=0,
#                      ) -> tuple[DataLoader, DataLoader]:
#     '''
#       根据 trian_data, eval_data 创建 train_loader, eval_loader
#         params:
#           train_data - 训练集数据
#           eval_data - 测试集数据
#           batch_size - 一批的数据量
#           num_workers - 多线程
#         return:
#           train_loader, eval_loader 
#     '''
#     train_loader = DataLoader(
#       train_data,
#       batch_size,
#       True,
#       num_workers=num_workers,
#     )
#     eval_loader = DataLoader(
#       eval_data,
#       batch_size,
#       False,
#       num_workers=num_workers
#     )
#     return train_loader, eval_loader
  
#   def get_datasets(self, data, train_ratio=0.8):
#     '''
#       将 data 按 train_ratio 分为训练集和测试集
#         params:
#           data - 最终的全部数据
#           train_ratio - 训练集比例
#         return:
#           train_subset - 训练集
#           eval_subset - 测试集
#     '''
#     total_size = len(data)
#     train_size = int(total_size * train_ratio)
#     eval_size = total_size - train_size
#     train_subset, eval_subset = random_split(data, [train_size, eval_size])
#     return train_subset, eval_subset

class Utils_Trainer:
  '''
    训练器，封装训练函数，以及简单显示损失
  '''
  def __init__(self, model:nn.Module, criterion:nn.Module, 
               optimizer:optim.Optimizer, scheduler:optim.lr_scheduler=None):
    self.model = model
    self.criterion = criterion
    self.optimizer = optimizer
    self.scheduler = scheduler
    self.train_dataloader, self.evaluate_dataloader = None, None

  def train_epoch(self) -> tuple[list, list]:
    '''
    用户实现一趟训练，返回 train_loss, train_accuracy，self.train_dataloader是数据
    '''
    print('\033[91mUtils_Trainer `train_epoch` is NULL\033[0m')
    raise NotImplementedError('Utils_Trainer `train_epoch` is NULL')

  def evaluate_epoch(self) -> tuple[list, list]:
    '''
    用户实现一趟测试，返回 evaluate_loss, evaluate_accuracy，self.evaluate_dataloader是数据
    '''
    print('\033[91mUtils_Trainer `evaluate_epoch` is NULL\033[0m')
    raise NotImplementedError('Utils_Trainer `evaluate_epoch` is NULL')

  def train(self, epochs, dataloader) -> tuple[list, list]:
    '''
      进行训练，获取loss和accuracy
    '''
    self.train_dataloader = dataloader
    loss, accuracy = [], []
    for epoch in range(epochs):
      print(f'train: {epoch+1}/{epochs}')
      self.model.train()
      l, a = self.train_epoch()
      loss.append(sum(l)/len(l))
      accuracy.append(sum(a)/len(a))
      if self.scheduler:
        self.scheduler.step()
    return loss, accuracy
  
  def evaluate(self, epochs, dataloader) -> tuple[list, list]:
    '''
      进行评测，获取loss和accuracy
    '''
    self.evaluate_dataloader = dataloader
    loss, accuracy = [], []
    for epoch in range(epochs):
      print(f'evaluate: {epoch+1}/{epochs}')
      self.model.eval()
      l, a = self.evaluate_epoch()
      loss.append(sum(l)/len(l))
      accuracy.append(sum(a)/len(a))
    return loss, accuracy
  
  def predict(self, dataloader):
    '''
      用户实现进行预测，返回predictions和actuals
    '''
    print('\033[91mUtils_Trainer `predict` is NULL\033[0m')
    raise NotImplementedError('Utils_Trainer `predict` is NULL')
  
  def train_eval(self, epochs, train_dataloader, 
                 eval_dataloader) -> tuple[list, list, list, list]:
    '''
      进行训练、评测，获取train_loss、train_accuracy、eval_loss、eval_accurary
    '''
    self.train_dataloader = train_dataloader
    self.evaluate_dataloader = eval_dataloader
    t_loss, t_accu = [], []
    e_loss, e_accu = [], []
    for epoch in range(epochs):
      print(f'train&evaluate: {epoch+1}/{epochs}')
      self.model.train()
      l, a = self.train_epoch()
      t_loss.append(sum(l)/len(l))
      t_accu.append(sum(a)/len(a))
      self.model.eval()
      l, a = self.evaluate_epoch()
      e_loss.append(sum(l)/len(l))
      e_accu.append(sum(a)/len(a))
      if self.scheduler:
        self.scheduler.step()
    return t_loss, t_accu, e_loss, e_accu

  def plot(self, t_t1, t_t2, e_t1, e_t2,
           T1:str='Loss', T2:str='Accuracy'):
    '''
      绘制T1、T2数据指标的图表
        params：
          t_t1 - 训练时的T1指标数组
          t_t2 - 训练时的T2指标数组
          e_t1 - 测试时的T1指标数组
          e_t2 - 测试时的T2指标数组
    '''
    t_epochs = range(1, len(t_t1)+1)
    e_epochs = range(1, len(e_t1)+1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(t_epochs, t_t1, 'b-', label=f'Training {T1}')
    plt.plot(t_epochs, e_t1, 'r-', label=f'Validation {T1}')
    plt.title(f'{T1} over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel(f'{T1}')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(e_epochs, t_t2, 'b-', label=f'Training {T2}')
    plt.plot(e_epochs, e_t2, 'r-', label=f'Validation {T2}')
    plt.title(f'{T2} over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel(f'{T2}')
    plt.legend()

    plt.show()

if __name__ == "__main__":
  pass
  # import numpy as np

  # class MyDataset(Utils_DatasetFactory):
  #   def load_data(self):
  #     np.random.seed(42)
  #     self.X = np.random.randn(200, 5).astype(np.float32)
  #     self.y = np.random.randint(0, 2, size=200)

  # class MyTrainer(Utils_Trainer):
  #   def train_epoch(self):
  #     losses, accs = [], []
  #     for X, y in self.train_dataloader:
  #       losses.append(0.5)
  #       accs.append(0.8)
  #     return losses, accs

  #   def evaluate_epoch(self):
  #     losses, accs = [], []
  #     for X, y in self.evaluate_dataloader:
  #       losses.append(0.4)
  #       accs.append(0.82)
  #     return losses, accs

  # data = MyDataset()
  # loader_factory = Utils_DataLoaderFactory()
  # model = nn.Linear(5, 2)
  # criterion = nn.CrossEntropyLoss()
  # optimizer = optim.SGD(model.parameters(), lr=0.01)
  # trainer = MyTrainer(model, criterion, optimizer)

  # t_loss, t_acc, e_loss, e_acc = Utils_Run(data, loader_factory, None, None, 
  #           16, 0, trainer, 3, 'Loss', 'Accu')

  # # 打印结果
  # print("\n✅ 训练损失:", t_loss)
  # print("✅ 训练准确率:", t_acc)
  # print("✅ 验证损失:", e_loss)
  # print("✅ 验证准确率:", e_acc)

#         ,--.                                                 ,--.     
#  ,---.  |  ,---.   ,--,--. ,--,--,--. ,--.--.  ,---.   ,---. |  |,-.  
# (  .-'  |  .-.  | ' ,-.  | |        | |  .--' | .-. | | .--' |     /  
# .-'  `) |  | |  | \ '-'  | |  |  |  | |  |    ' '-' ' \ `--. |  \  \  
# `----'  `--' `--'  `--`--' `--`--`--' `--'     `---'   `---' `--'`--' 
                                                                      

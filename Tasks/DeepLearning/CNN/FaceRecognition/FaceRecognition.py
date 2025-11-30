import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob, os
from PIL import Image

# CelebA数据集加载器
class CelebADataset(Dataset):
  _select_identities = None
  _identity_to_idx = None
  identity_images = None
  identity_map = None
  split_map = None

  def __init__(self, root_dir, transform=None, train_ratio=0.8, train=True):
    self.transform = transform
    self.images, self.labels = [], []
    self.class_names = []
    self._load_celeba_dataset(root_dir, train_ratio, train)
  
  def _load_celeba_dataset(self, root_dir, train_ratio, train):
    img_dir = os.path.join(root_dir, 'img_align_celeba')
    identity_file = os.path.join(root_dir, 'identity_CelebA.txt')
    split_file = os.path.join(root_dir, 'list_eval_partition.txt')

    if CelebADataset.identity_map is None:
      CelebADataset.identity_map = {}
      with open(identity_file, 'r') as f:
        for line in f:
          parts = line.strip().split()
          if len(parts) >= 2:
            CelebADataset.identity_map[parts[0]] = int(parts[1])

    if CelebADataset.split_map is None:
      CelebADataset.split_map = {}
      with open(split_file, 'r') as f:
        for line in f:
          parts = line.strip().split()
          if len(parts) >= 2:
            CelebADataset.split_map[parts[0]] = int(parts[1])  # 0: train, 1: val, 2: test
    
    if CelebADataset.identity_images is None:
      CelebADataset.identity_images = {}
      for img_name, identity in CelebADataset.identity_map.items():
        if img_name in CelebADataset.split_map:
          split = CelebADataset.split_map[img_name]
          if (train and split == 0) or (not train and split in [1, 2]):
            img_path = os.path.join(img_dir, img_name)
            if identity not in CelebADataset.identity_images:
              CelebADataset.identity_images[identity] = []
            CelebADataset.identity_images[identity].append(img_path)

    target_images_per_identity = 20
    max_identities = 5

    import random
    random.seed(42)
    if CelebADataset._select_identities is None:
      qualified_identities = [identity for identity, images in CelebADataset.identity_images.items()
                              if len(images) >= target_images_per_identity]
      qualified_identities.sort(key=lambda x: len(CelebADataset.identity_images[x]), reverse=True)
      CelebADataset._select_identities = qualified_identities[:max_identities]
      CelebADataset._identity_to_idx = {identity: idx for idx, identity in enumerate(CelebADataset._select_identities)}
      print(f"Selected {len(CelebADataset._select_identities)} identities for CelebA dataset.")
      
    # 只处理选中的身份
    original_images = []
    original_labels = []
    
    for identity in CelebADataset._select_identities:
      if identity in CelebADataset.identity_images:
        images = CelebADataset.identity_images[identity]
        random.shuffle(images)
        split_index = int(len(images)*train_ratio)
        selected_images = images[:split_index] if train else images[split_index:]
        
        if len(selected_images) > target_images_per_identity:
          selected_images = random.sample(selected_images, target_images_per_identity)
        
        original_images.extend(selected_images)
        original_labels.extend([identity]*len(selected_images))

    # 应用标签映射
    self.images = original_images
    self.labels = [CelebADataset._identity_to_idx[label] for label in original_labels]
    self.class_names = [f'ID_{uid}' for uid in CelebADataset._select_identities]
    
    print(f"CelebA {'Training' if train else 'Testing'}: {len(self.images)} images")

  def __len__(self):
    return len(self.images)
  
  def __getitem__(self, index):
    img_path = self.images[index]
    label = self.labels[index]
    try:
      image = Image.open(img_path)
      if image.mode != 'L':
        image = image.convert('L')    # 'L'代表Luminance,即灰度图
    except Exception as e:
      print(f"Error loading image {img_path}: {e}")
      image = Image.new('L', (178, 218))  # 创建一个空白图像以防止崩溃
    if self.transform:
      image = self.transform(image)
    return image, label

# ORL数据集加载器
class ORLDataset(Dataset):
  def __init__(self, root_dir, transform=None, train_ratio=0.8, train=True):
    self.transform = transform
    self.images, self.labels = [], []
    self.class_names = []
    self._load_orl_dataset(root_dir, train_ratio, train)

  def _load_orl_dataset(self, root_dir, train_ratio, train):
    pgm_files = glob.glob(os.path.join(root_dir, '**/*.pgm'), recursive=True)
    print(f"Found {len(pgm_files)} image files")

    subject_dict = {}
    for file_path in pgm_files:
      subject_id = os.path.basename(os.path.dirname(file_path))
      if subject_id not in subject_dict:
        subject_dict[subject_id] = []
      subject_dict[subject_id].append(file_path)

    self.class_names = sorted(subject_dict.keys())

    for subject_id, subject_name in enumerate(self.class_names):
      images = sorted(subject_dict[subject_name])
      split_index = int(len(images)*train_ratio)
      if train:
        selected_images = images[:split_index]
      else:
        selected_images = images[split_index:]
      for img_path in selected_images:
        self.images.append(img_path)
        self.labels.append(subject_id)
    print(f"{'Training' if train else 'Testing'} set: {len(self.images)} images")
  
  def __len__(self):
    return len(self.images)
  
  def __getitem__(self, index):
    img_path = self.images[index]
    label = self.labels[index]
    try:
      image = Image.open(img_path)
      if image.mode != 'L':
        image = image.convert('L')    # 'L'代表Luminance,即灰度图
    except Exception as e:
      print(f"Error loading image {img_path}: {e}")
      image = Image.new('L', (92, 112))  # 创建一个空白图像以防止崩溃
    if self.transform:
      image = self.transform(image)
    return image, label

# MNIST数据集加载器
class MNISTDataset(Dataset):
  def __init__(self, root_dir, transform=None, train=True):
    self.transform = transform
    self.class_names = [str(i) for i in range(10)]
    self.dataset = datasets.MNIST(root=root_dir, train=train, download=True)
    
  def __len__(self):
    return len(self.dataset)
  
  def __getitem__(self, index):
    image, label = self.dataset[index]
    if image.mode != 'L':
      image = image.convert('L')    # 'L'代表Luminance,即灰度图
    if self.transform:
      image = self.transform(image)
    return image, label

# 数据加载器
class FaceRecognitionDataLoader(Dataset):
  '''
  通用数据加载器 适用于不同数据集
  '''
  def __init__(self, dataset, transform=None):
    self.dataset = dataset
    self.transform = transform
    self.classes = dataset.class_names if hasattr(dataset, 'class_names') else []
    self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)} if self.classes else {}
    self.images = dataset.images if hasattr(dataset, 'images') else []
    self.labels = dataset.labels if hasattr(dataset, 'labels') else []
    self.class_names = dataset.class_names if hasattr(dataset, 'class_names') else []
  
  def __len__(self):
    return len(self.dataset)
  
  def __getitem__(self, index):
    img, label = self.dataset[index]
    return img, label

# CNN模型
class FaceRecognitionCNN(nn.Module):
  def __init__(self, num_classes, input_size=(112, 92)):
    super(FaceRecognitionCNN, self).__init__()
    self.features = nn.Sequential(
      # 第一层卷积
      nn.Conv2d(1, 32, kernel_size=3, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2),
      
      # 第二层卷积
      nn.Conv2d(32, 64, kernel_size=3, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2),
      
      # 第三层卷积
      nn.Conv2d(64, 128, kernel_size=3, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2),

      # 第四层卷积块
      nn.Conv2d(128, 256, kernel_size=3, padding=1),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2),
    )
    
    '''
    卷积层输出：[batch_size, 128, 14, 11]
    '''
    with torch.no_grad():
      dummy_input = torch.zeros(1, 1, *input_size)
      dummy_output = self.features(dummy_input)
      flattened_size = dummy_output.view(1, -1).size(1)
      print(f'Flattened feature size: {flattened_size}')

    self.classifier = nn.Sequential(
      nn.Dropout(0.5),
      nn.Linear(flattened_size, 1024),
      nn.ReLU(inplace=True),
      nn.BatchNorm1d(1024),
      nn.Dropout(0.5),
      nn.Linear(1024, 512),
      nn.ReLU(inplace=True),
      nn.Dropout(0.3),
      nn.Linear(512, num_classes)
    )
  
  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    return x

# 训练器
class FaceRecognitionTrainer:
  def __init__(self, dataset_type='orl', data_path='./data', **dataset_kwargs):
    self.dataset_type = dataset_type.lower()
    self.dataset_kwargs = dataset_kwargs
    self.number_of_classes = 0
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {self.device}, Dataset: {self.dataset_type}')
    self.data_path = data_path
    self.train_losses = []
    self.val_losses = []
    self.train_accuracies = []
    self.val_accuracies = []
    self.data_preprocess()
    self.load_data()

  def setup(self, epochs=30):
    self.init_model()
    self.train_model(epochs)

  def data_preprocess(self):
    '''
    目的 
      创建self.transform和self.test_transform
    '''
    if self.dataset_type == 'orl':
      self.transform = transforms.Compose([
        transforms.Resize((112, 92)),             # 调整图像大小为112x92
        transforms.RandomHorizontalFlip(p=0.3),   # 30%概率水平翻转图像
        transforms.RandomRotation(5),             # 随机旋转图像，角度范围为±5度
        transforms.ToTensor(),                    # 将图像转换为张量，像素值从[0, 255]缩放到[0.0, 1.0] 对于灰度图：(112, 92) → (1, 112, 92)
        transforms.Normalize(mean=[0.5], std=[0.5]) # (x-0.5)/0.5 → [-1, 1]
      ])
      self.test_transform = transforms.Compose([
        transforms.Resize((112, 92)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
      ])
    elif self.dataset_type == 'celeba':
      target_size = (128, 128)
      self.transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        # 表示亮度在 [1 - 0.2, 1 + 0.2] = [0.8, 1.2] 范围内随机调整
        # 对比度在 [1 - 0.2, 1 + 0.2] = [0.8, 1.2] 范围内随机调整
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
      ])
      self.test_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
      ])
    elif self.dataset_type == 'mnist':
      self.transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),   # 平移
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
      ])
      self.test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
      ])
    else:
      raise ValueError(f"Unsupported dataset type: {self.dataset_type}. Supported types: 'orl', 'celeba'.")

  def load_data(self):
    print(f'Loading {self.dataset_type} data... Dataset path: {self.data_path}')
    if not os.path.exists(self.data_path):
      raise FileNotFoundError(f"Data path {self.data_path} does not exist.")
    if self.dataset_type == 'orl':
      train = ORLDataset(self.data_path, transform=self.transform, train=True)
      test = ORLDataset(self.data_path, transform=self.test_transform, train=False)
      batch_size = 8
    elif self.dataset_type == 'celeba':
      train = CelebADataset(self.data_path, transform=self.transform, train=True)
      test = CelebADataset(self.data_path, transform=self.test_transform, train=False)
      batch_size = 32
    elif self.dataset_type == 'mnist':
      train = MNISTDataset(self.data_path, transform=self.transform, train=True)
      test = MNISTDataset(self.data_path, transform=self.test_transform, train=False)
      batch_size = 128
    else:
      raise ValueError(f"Unsupported dataset type: {self.dataset_type}. Supported types: 'orl', 'celeba', 'mnist'.")
    self.train_dataset = FaceRecognitionDataLoader(train)
    self.test_dataset = FaceRecognitionDataLoader(test)
    self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
    self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
    self.class_names = train.class_names
    self.number_of_classes = len(train.class_names)
    if self.dataset_type.lower() == 'celeba' and self.number_of_classes > 1000:
      print("Warning: CelebA has many identities, consider using attribute recognition instead.")

  def init_model(self):
    if self.dataset_type == 'orl':
      input_size = (112, 92)
    elif self.dataset_type == 'celeba':
      input_size = (128, 128)
    elif self.dataset_type == 'mnist':
      input_size = (32, 32)
    self.model = FaceRecognitionCNN(self.number_of_classes, input_size=input_size).to(self.device)

  def train_model(self, num_epochs=30):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    print('Starting training...')
    for epoch in range(num_epochs):
      print(f'Epoch {epoch+1}/{num_epochs}')
      self._train_one_epoch(criterion, optimizer, scheduler)

  def evaluate_model(self):
    self.model.eval()
    correct, test_loss = 0, 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
      for images, labels in self.test_loader:
        images, labels = images.to(self.device), labels.to(self.device)
        outputs = self.model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return 100*correct/total, test_loss/len(self.test_loader)

  def _train_one_epoch(self, criterion, optimizer, scheduler):
    self.model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    progress_bar = tqdm(self.train_loader, desc='Training ')
    for batch_idx, (images, labels) in enumerate(progress_bar):
      images, labels = images.to(self.device), labels.to(self.device)
      outputs = self.model(images)
      loss = criterion(outputs, labels)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      running_loss += loss.item()
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
      progress_bar.set_postfix({
        'Loss': f'{running_loss/(batch_idx+1):.4f}', 
        'Accuracy': f'{100 * correct / total:.2f}%'
      })
    scheduler.step()
    train_accuracy = 100 * correct / total
    avg_train_loss = running_loss / len(self.train_loader)
    self.train_losses.append(running_loss / len(self.train_loader))
    self.train_accuracies.append(train_accuracy)
    # 验证
    test_accuracy, test_loss = self.evaluate_model()
    self.val_accuracies.append(test_accuracy)
    self.val_losses.append(test_loss)
    
    print(f'  Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%')
    print(f'  Validation Loss: {test_loss:.4f}, Validation Accuracy: {test_accuracy:.2f}%')

# 可视化工具
class FaceRecognitionPlotter:
  def __init__(self, trainer):
    self.trainer = trainer

  def plot_metrics(self):
    epochs = range(1, len(self.trainer.train_losses) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, self.trainer.train_losses, 'b', label='Training loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, self.trainer.train_accuracies, 'b', label='Training Accuracy')
    plt.plot(epochs, self.trainer.val_accuracies, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

# 测试器
class FaceRecognitionTester:
  def __init__(self, model, device, class_names):
    self.class_names = class_names
    self.model = model
    self.device = device

  def test_single_image(self, image_path, transform):
    image_pil = Image.open(image_path)
    if image_pil.mode != 'L':
      image_pil = image_pil.convert('L')
    image_display = image_pil

    input_tensor = transform(image_pil).unsqueeze(0).to(self.device)
    self.model.eval()
    with torch.no_grad():
      output = self.model(input_tensor)
      probabilities = nn.functional.softmax(output[0], dim=0)
      predicted_class = torch.argmax(probabilities).item()
      confidence = probabilities[predicted_class].item()
    self._display_result(image_display, predicted_class, confidence, image_path)
    return predicted_class, confidence
  
  def _display_result(self, image, predicted_class, confidence, image_path):
      plt.figure(figsize=(10, 8))
      
      if isinstance(image, Image.Image):
        plt.imshow(image, cmap='gray')
      else:
        plt.imshow(image, cmap='gray')
      
      plt.title(f'image: {os.path.basename(image_path)}\n'
                f'prediction: {self.class_names[predicted_class]} (ID: {predicted_class})\n'
                f'confidence: {confidence:.4f}',
                fontsize=12, pad=20)
      plt.axis('off')
      plt.tight_layout()
      plt.show()

# 模型加载器
class FaceRecognitionModelLoader:
  def __init__(self, model_path):
    self.model_path = model_path
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.model = None
  
  def load_model(self, num_classes, input_size):
    if not os.path.exists(self.model_path):
      print(f"Model path {self.model_path} does not exist.")
      return None
    self.model = FaceRecognitionCNN(num_classes, input_size).to(self.device)
    self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
    self.model.eval()
    print(f'Model loaded from {self.model_path}')
    return self.model
  
  def save_model(self, model):
    if model:
      torch.save(model.state_dict(), self.model_path)
      print(f'Model saved to {self.model_path}')
    else:
      print('No model to save.')

if __name__ == "__main__":
  dataset_type = 'orl'  # 'orl' 'celeba' 'mnist'

  if dataset_type == 'orl':
    model_path = './orl_model.pth'
    data_path = './data/orl_faces'
    epochs = 20
    test_image_path = './data/orl_faces/s1/2.pgm'
    input_size = (112, 92)
  elif dataset_type == 'celeba':
    model_path = './celeba_model.pth'
    data_path = './data/celeba'
    epochs = 30
    test_image_path = 'data/celeba/img_align_celeba/002279.jpg'
    input_size = (128, 128)
  elif dataset_type == 'mnist':
    model_path = './mnist_model.pth'
    data_path = './data'
    epochs = 3
    test_image_path = './data/MNIST/raw/mnist_test_image.png'
    input_size = (32, 32)
  trainer = FaceRecognitionTrainer(dataset_type=dataset_type, data_path=data_path)

  loader = FaceRecognitionModelLoader(model_path=model_path)
  model = loader.load_model(trainer.number_of_classes, input_size=input_size)
  if model is None:
    trainer.setup(epochs)
    model = trainer.model
    plotter = FaceRecognitionPlotter(trainer)
    plotter.plot_metrics()
    loader.save_model(trainer.model)
  else:
    tester = FaceRecognitionTester(model, trainer.device, trainer.class_names)
    tester.test_single_image(test_image_path, trainer.test_transform)

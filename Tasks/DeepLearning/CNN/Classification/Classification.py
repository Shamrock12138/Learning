import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time

# 设置随机种子保证可重复性
torch.manual_seed(42)
np.random.seed(42)

class AdvancedCNN(nn.Module):
    """
    高级CNN模型，包含批量归一化、Dropout等现代技术
    适用于CIFAR-10图像分类任务
    """
    
    def __init__(self, num_classes=10):
        """
        初始化CNN模型
        
        Args:
            num_classes: 分类类别数，CIFAR-10为10
        """
        super(AdvancedCNN, self).__init__()
        
        # 第一个卷积块: 提取低级特征 (边缘、颜色)
        self.conv_block1 = nn.Sequential(
            # 卷积层1: 3个输入通道(RGB), 64个输出通道, 3x3卷积核
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            # 批量归一化: 加速训练，提高稳定性
            nn.BatchNorm2d(64),
            # ReLU激活函数: 引入非线性
            nn.ReLU(inplace=True),   # inplace直接在输入张量上修改，节省内存
            # 卷积层2: 64->64通道
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 最大池化: 2x2窗口，步长2，特征图尺寸减半
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Dropout: 随机失活，防止过拟合
            nn.Dropout(0.25)    # 25%的神经元失活
        )
        
        # 第二个卷积块: 提取中级特征 (纹理、形状)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        
        # 第三个卷积块: 提取高级特征 (物体部件)
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        
        # 第四个卷积块: 进一步提取特征
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        
        # 分类器部分: 全连接层进行最终分类
        self.classifier = nn.Sequential(
            # 全连接层1: 将特征图展平后连接
            nn.Linear(512 * 2 * 2, 1024),  # 经过4次池化后尺寸: 32x32 -> 2x2
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            # 全连接层2
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            # 输出层: 10个类别
            nn.Linear(512, num_classes)
        )
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 卷积层使用He初始化，适合ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # 批量归一化层初始化
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 全连接层初始化
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播过程
        
        Args:
            x: 输入张量 [batch_size, 3, 32, 32]
            
        Returns:
            output: 分类结果 [batch_size, num_classes]
        """
        # 卷积特征提取
        x = self.conv_block1(x)  # [batch, 64, 16, 16]
        x = self.conv_block2(x)  # [batch, 128, 8, 8]
        x = self.conv_block3(x)  # [batch, 256, 4, 4]
        x = self.conv_block4(x)  # [batch, 512, 2, 2]
        
        # 展平特征图
        x = x.view(x.size(0), -1)  # [batch, 512*2*2]
        
        # 分类
        x = self.classifier(x)    # [batch, num_classes]
        
        return x

class CNNVisualizer:
    """CNN可视化工具类"""
    
    @staticmethod
    def visualize_feature_maps(model, image_tensor, layer_name='conv_block1'):
        """
        可视化指定层的特征图
        
        Args:
            model: CNN模型
            image_tensor: 输入图像张量
            layer_name: 要可视化的层名称
        """
        model.eval()
        activations = {}
        
        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach()
            return hook
        
        # 注册钩子获取中间层输出
        if hasattr(model, layer_name):
            layer = getattr(model, layer_name)
            hook = layer.register_forward_hook(get_activation(layer_name))
        else:
            print(f"Layer {layer_name} not found!")
            return
        
        # 前向传播
        with torch.no_grad():
            _ = model(image_tensor.unsqueeze(0))
        
        # 可视化特征图
        feature_maps = activations[layer_name][0]  # 取第一个batch
        num_features = min(feature_maps.size(0), 16)  # 最多显示16个特征图
        
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        fig.suptitle(f'Feature Maps from {layer_name}', fontsize=16)
        
        for i, ax in enumerate(axes.flat):
            if i < num_features:
                ax.imshow(feature_maps[i].cpu(), cmap='viridis')
                ax.set_title(f'Channel {i}')
                ax.axis('off')
            else:
                ax.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # 移除钩子
        hook.remove()
    
    @staticmethod
    def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies):
        """绘制训练历史曲线"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        ax1.plot(train_losses, label='Training Loss')
        ax1.plot(val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot(train_accuracies, label='Training Accuracy')
        ax2.plot(val_accuracies, label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

class CIFAR10Trainer:
    """CIFAR-10训练器"""
    
    def __init__(self, batch_size=128, learning_rate=0.001):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # 数据预处理
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        # 加载数据
        self._load_data()
        
        # 初始化模型
        self.model = AdvancedCNN().to(self.device)
        
        # 损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        
        # 训练历史记录
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def _load_data(self):
        """加载CIFAR-10数据集"""
        print("Loading CIFAR-10 dataset...")
        
        # 训练集
        self.train_set = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=self.transform_train)
        self.train_loader = DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=2)
        
        # 测试集
        self.test_set = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=self.transform_test)
        self.test_loader = DataLoader(
            self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=2)
        
        # 类别名称
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck')
        
        print(f"Training set: {len(self.train_set)} samples")
        print(f"Test set: {len(self.test_set)} samples")
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.3f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    def validate(self, loader, desc="Validation"):
        """验证模型"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(loader, desc=desc)
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                pbar.set_postfix({
                    'Loss': f'{loss.item():.3f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
        
        epoch_loss = running_loss / len(loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    def train(self, epochs=50):
        """完整训练过程"""
        print(f"Starting training for {epochs} epochs...")
        start_time = time.time()
        
        best_acc = 0  # 保存最佳准确率
        
        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            print('-' * 50)
            
            # 训练
            train_loss, train_acc = self.train_epoch()
            
            # 验证
            val_loss, val_acc = self.validate(self.test_loader)
            
            # 学习率调度
            self.scheduler.step()
            
            # 记录历史
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # 保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), 'best_cnn_model.pth')
                print(f"New best model saved with accuracy: {best_acc:.2f}%")
            
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'Best Acc: {best_acc:.2f}%')
            print(f'Learning Rate: {self.optimizer.param_groups[0]["lr"]:.6f}')
        
        training_time = time.time() - start_time
        print(f'\nTraining completed in {training_time//60:.0f}m {training_time%60:.0f}s')
        print(f'Best validation accuracy: {best_acc:.2f}%')
    
    def evaluate(self):
        """最终评估模型"""
        print("\nFinal Evaluation:")
        self.model.load_state_dict(torch.load('best_cnn_model.pth'))
        test_loss, test_acc = self.validate(self.test_loader, "Testing")
        print(f'Test Accuracy: {test_acc:.2f}%')
        
        # 绘制训练历史
        CNNVisualizer.plot_training_history(
            self.train_losses, self.val_losses, 
            self.train_accuracies, self.val_accuracies
        )
    
    def visualize_sample_predictions(self, num_samples=12):
        """可视化样本预测结果"""
        self.model.eval()
        self.model.load_state_dict(torch.load('best_cnn_model.pth'))
        
        # 获取一些测试样本
        dataiter = iter(self.test_loader)
        images, labels = next(dataiter)
        images, labels = images.to(self.device), labels.to(self.device)
        
        # 预测
        with torch.no_grad():
            outputs = self.model(images)
            _, predicted = outputs.max(1)
        
        # 反标准化用于显示
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1).to(self.device)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1).to(self.device)
        images = images * std + mean
        images = images.clamp(0, 1)
        
        # 绘制结果
        fig, axes = plt.subplots(3, 4, figsize=(12, 9))
        fig.suptitle('Sample Predictions on Test Set', fontsize=16)
        
        for i, ax in enumerate(axes.flat):
            if i < num_samples:
                img = images[i].cpu().permute(1, 2, 0).numpy()
                ax.imshow(img)
                
                true_label = self.classes[labels[i]]
                pred_label = self.classes[predicted[i]]
                
                color = 'green' if labels[i] == predicted[i] else 'red'
                ax.set_title(f'True: {true_label}\nPred: {pred_label}', color=color)
                ax.axis('off')
        
        plt.tight_layout()
        plt.show()

def main():
    """主函数"""
    # 初始化训练器
    trainer = CIFAR10Trainer(batch_size=128, learning_rate=0.001)
    
    # 打印模型结构
    print("\nModel Architecture:")
    print(trainer.model)
    
    # 计算参数数量
    total_params = sum(p.numel() for p in trainer.model.parameters())
    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    # 训练模型
    trainer.train(epochs=50)
    
    # 评估模型
    trainer.evaluate()
    
    # 可视化预测样本
    trainer.visualize_sample_predictions()
    
    # 可视化特征图（可选）
    # 获取一个样本图像
    sample_image, _ = trainer.test_set[0]
    sample_image = sample_image.to(trainer.device)
    
    # 可视化第一个卷积块的特征图
    CNNVisualizer.visualize_feature_maps(trainer.model, sample_image, 'conv_block1')

if __name__ == "__main__":
    main()
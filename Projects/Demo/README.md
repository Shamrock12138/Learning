# 无人机充电系统强化学习训练项目

## 项目概述
本项目实现了使用强化学习训练充电无人机为任务无人机充电的系统。通过DDQN算法，充电无人机学会在任务机电量低时及时为其充电，同时也能在基站为自身充电。

## 核心文件

### 训练文件
- `behavior_based_charging_demo.py`: 基于行为树的充电无人机训练器（推荐使用）
- `deep_charging_training.py`: 深度训练版本（1000轮训练，含收敛分析）

### 测试与演示
- `animated_model_test.py`: 动画测试脚本，用于测试训练好的模型

### 配置文件
- `simple_charging_config.json`: 训练配置文件

### 预训练模型
- `model/BehaviorBasedCharging_DDQN_299.pt`: 推荐使用的预训练模型
- `model/DeepCharging_DDQN_999.pt`: 深度训练模型

### 文档
- `FINAL_REPORT.md`: 项目最终总结报告
- `README.md`: 本文件

## 使用说明

### 1. 训练模型
```bash
python behavior_based_charging_demo.py  # 推荐的基础训练
# 或
python deep_charging_training.py        # 深度训练（1000轮）
```

### 2. 测试模型
```bash
python animated_model_test.py           # 动画测试已训练的模型
```

## 训练结果
- 任务机存活率从0%提升至100%
- 充电无人机学会有效的追踪和充电策略
- 训练过程稳定收敛

## 技术栈
- Python 3.11
- PyTorch
- NumPy
- Matplotlib
- Tqdm

## 算法
- Double Deep Q-Network (DDQN)
- 经验回放
- 目标网络更新

## 项目亮点
1. **混合策略**: 结合规则和强化学习，确保训练稳定性
2. **行为状态**: 专注于距离和电量关系的状态表示
3. **奖励优化**: 针对充电行为优化的奖励函数
4. **可视化**: 支持实时动画演示训练过程和结果
5. **收敛分析**: 包含训练收敛性分析功能
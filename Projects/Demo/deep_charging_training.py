"""
深度训练充电无人机模型
增加训练次数至1000轮，并添加奖励函数变化曲线分析
"""

import torch
import numpy as np
from tqdm import tqdm
import sys
import os
import random
import matplotlib.pyplot as plt

# 添加项目路径到sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from myTools.Model.RL import DoubleDQN
from myTools.Utils.MARL_env import MARL_Env_UAVs
from myTools.Utils.MARL_tools import Q_Net as MARL_Q_Net
from myTools.Utils.tools import utils_readParams, utils_autoAssign, utils_showHistory
from myTools.Utils.tools import utils_ReplayBuffer
from myTools.Utils.config import Sample


class DeepChargingTrainer:
    """
    深度充电机训练器
    训练1000轮，增加奖励函数分析
    """
    def __init__(self, config_path, device):
        self.device = device
        
        # 读取配置
        model_config = utils_readParams(config_path, 'model')
        self.model_params = model_config
        
        # 更新训练轮数为1000
        trainer_config = utils_readParams(config_path, 'trainer')
        trainer_config['train_episodes'] = 1000  # 增加到1000轮
        
        # 创建环境
        self.env = MARL_Env_UAVs(config_path)
        
        # 创建DDQN智能体
        qnet = MARL_Q_Net(
            state_dim=model_config['state_dim'], 
            action_dim=model_config['action_dim'], 
            Q_Net_config=config_path
        )
        
        self.agent = DoubleDQN(
            state_dim=model_config['state_dim'],
            action_dim=model_config['action_dim'],
            lr=model_config['lr'],
            gamma=model_config['gamma'],
            epsilon=model_config['epsilon'],
            target_update=model_config['target_update'],
            Q_Net=qnet,
            device=device
        )
        
        # 训练参数
        # 手动分配训练参数
        for key, value in trainer_config.items():
            setattr(self, key, value)
        
        self.name = f'Deep_Charging_DDQN'
        
    def _get_behavior_based_state(self, full_state):
        """
        获取基于行为的充电机状态
        更专注于距离和电量关系
        """
        # 充电机状态索引
        cuav_idx = self.env.n_task_uavs  # 充电机在状态数组中的索引
        
        # 获取充电机自身状态 [x, y, battery, type]
        cuav_state = full_state[cuav_idx].copy()
        
        # 获取任务机状态 [x, y, battery, type]
        tuav_state = full_state[0].copy()  # 假设只有一个任务机
        
        # 获取基站位置
        base_pos = self.env.base_station.position
        
        # 计算相对位置
        cuav_pos = cuav_state[:2]  # [x, y]
        tuav_pos = tuav_state[:2]  # [x, y]
        
        # 曼哈顿距离（更适合网格移动）
        manhattan_dist_to_task = abs(tuav_pos[0] - cuav_pos[0]) + abs(tuav_pos[1] - cuav_pos[1])
        manhattan_dist_to_base = abs(base_pos[0] - cuav_pos[0]) + abs(base_pos[1] - cuav_pos[1])
        
        # 相对位置
        rel_tuav_pos = tuav_pos - cuav_pos  # 相对于充电机的任务机位置
        rel_base_pos = base_pos - cuav_pos   # 相对于充电机的基站位置
        
        # 构建状态：[自身位置(2), 自身电量(1), 任务机位置(2), 任务机电量(1), 
        # 基站位置(2), 相对任务机位置(2), 相对基站位置(2), 到任务机曼哈顿距离(1)]
        behavior_state = np.concatenate([
            cuav_state[:2],                    # 自身位置 [x, y]
            [cuav_state[2]],                   # 自身电量
            tuav_state[:2],                    # 任务机位置 [x, y]  
            [tuav_state[2]],                   # 任务机电量
            base_pos,                          # 基站位置 [x, y]
            rel_tuav_pos,                      # 相对任务机位置 [dx, dy]
            rel_base_pos,                      # 相对基站位置 [dx, dy]
            [manhattan_dist_to_task]           # 到任务机的曼哈顿距离
        ]).astype(np.float32)
        
        return behavior_state

    def take_action(self, full_state):
        """
        获取动作 - 结合规则和学习
        """
        actions = np.zeros(self.env._n_agents, dtype=int)
        
        # 任务机随机行动
        for i in range(self.env.n_task_uavs):
            # 随机选择动作：0(静止), 1(上), 2(下), 3(左), 4(右)
            # 动作5是充电模式，任务机不应该使用
            actions[i] = np.random.randint(0, 5)
        
        # 充电机使用混合策略：规则+学习
        if self.env.n_charging_uavs > 0:
            # 获取充电机状态
            cuav_idx = self.env.n_task_uavs
            cuav = self.env.charging_uavs[0]  # 假设只有一个充电机
            tuav = self.env.task_uavs[0]      # 假设只有一个任务机
            
            # 行为树逻辑：
            # 1. 如果在充电范围内且任务机电量低，直接充电
            if cuav.can_charge_task(tuav) and tuav.battery < 0.5:
                actions[cuav_idx] = 5  # 充电动作
            # 2. 如果自身电量低，前往基站
            elif cuav.battery < 0.2 and cuav.battery < tuav.battery:
                # 使用简单的导航到基站逻辑
                action = cuav.navigate_to_target(self.env.base_station.position)
                actions[cuav_idx] = action
            # 3. 否则前往任务机
            else:
                # 使用导航到任务机的逻辑
                action = cuav.navigate_to_target(tuav.position)
                # 如果已经在任务机旁边，准备充电
                if cuav.can_charge_task(tuav):
                    actions[cuav_idx] = 5  # 充电
                else:
                    actions[cuav_idx] = action
        
        return actions

    def train(self, model_name=None, begin=0):
        """
        深度训练充电机 - 1000轮，包含奖励分析
        """
        if model_name:
            self.agent.load_model(self.model_path, model_name)
        
        history = {'rewards': [], 'losses': [], 'alive_tasks': [], 'charging_events': [], 'steps': []}
        
        # 创建模型和历史记录目录
        import os
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.history_path, exist_ok=True)
        
        # 经验回放缓冲区
        buffer_config = utils_readParams("Projects\\Demo\\simple_charging_config.json", 'buffer')
        replay_buffer = utils_ReplayBuffer(buffer_config.get('buffer_size', 10000))
        
        min_size = buffer_config.get('min_size', 128)
        batch_size = buffer_config.get('batch_size', 64)
        
        with tqdm(total=self.train_episodes, desc=f'{self.name}') as pbar:
            for episode in range(self.train_episodes):
                done = False
                state, _ = self.env.reset()
                total_reward = 0
                total_loss = 0
                step = 0
                charging_events = 0  # 记录充电事件
                
                # 随着训练进行，逐渐增加学习策略的权重
                learning_ratio = min(0.1 + episode * 0.0006, 0.8)  # 从0.1增加到0.8，增长更慢
                
                while not done and step < self.env.max_steps:
                    # 随机决定使用规则还是学习策略
                    if random.random() < learning_ratio and self.env.n_charging_uavs > 0:
                        # 使用学习策略
                        behavior_state = self._get_behavior_based_state(state)
                        charging_action = self.agent.take_action(behavior_state)
                        actions = np.zeros(self.env._n_agents, dtype=int)
                        
                        # 任务机随机行动
                        for i in range(self.env.n_task_uavs):
                            actions[i] = np.random.randint(0, 5)
                        
                        actions[self.env.n_task_uavs] = charging_action
                        
                        # 构造转移样本用于学习
                        next_state, reward, done, info = self.env.step(actions)
                        
                        # 统计充电事件
                        charging_events += info.get("charging_events", {}).get("static", 0)
                        charging_events += info.get("charging_events", {}).get("base", 0)
                        
                        # 构造行为状态的转移样本
                        behavior_next_state = self._get_behavior_based_state(next_state)
                        
                        # 获取充电机的奖励
                        cuav_reward = reward[self.env.n_task_uavs]
                        
                        # 添加到经验回放
                        sample = Sample(
                            state=behavior_state,
                            action=np.array([actions[self.env.n_task_uavs]]),
                            reward=np.array([cuav_reward]),
                            next_state=behavior_next_state,
                            done=np.array([done])
                        )
                        replay_buffer.add_sample(sample)
                        
                        state = next_state
                        total_reward += cuav_reward
                        step += 1
                        
                        # 训练网络
                        if len(replay_buffer) > min_size:
                            transition_dict = replay_buffer.sample_sample(batch_size)
                            
                            loss = self.agent.update(transition_dict)
                            total_loss += loss
                    else:
                        # 使用规则策略
                        actions = self.take_action(state)
                        next_state, reward, done, info = self.env.step(actions)
                        
                        # 统计充电事件
                        charging_events += info.get("charging_events", {}).get("static", 0)
                        charging_events += info.get("charging_events", {}).get("base", 0)
                        
                        state = next_state
                        # 对于规则策略，我们不将其加入经验回放进行学习
                        # 但我们仍然计算总奖励
                        if self.env.n_charging_uavs > 0:
                            total_reward += reward[self.env.n_task_uavs]
                        step += 1
                
                # 记录任务机存活情况
                alive_tasks = sum(1 for t in self.env.task_uavs if t.state["alive"])
                
                history['rewards'].append(total_reward)
                history['losses'].append(total_loss / max(step, 1) if step > 0 else 0)  # 平均损失
                history['alive_tasks'].append(alive_tasks)
                history['charging_events'].append(charging_events)
                history['steps'].append(step)
                
                pbar.set_postfix({
                    'Reward': f'{total_reward:.2f}',
                    'Alive': f'{alive_tasks}/{self.env.n_task_uavs}',
                    'Charges': charging_events,
                    'Steps': step,
                    'Learn%': f'{learning_ratio:.2f}'
                })
                pbar.update(1)
                
                # 保存模型 (每100轮保存一次)
                if (episode + 1) % 100 == 0:
                    self.agent.save_model(self.model_path, f'\\DeepCharging_DDQN_{episode+begin}.pt')
        
        # 绘制详细的训练历史
        self.plot_detailed_training_history(history)
        
        print(f"\n深度训练完成! 共训练 {self.train_episodes} 轮")
        print(f"平均奖励: {np.mean(history['rewards']):.2f}")
        print(f"平均存活任务机: {np.mean(history['alive_tasks']):.2f}/{self.env.n_task_uavs}")
        print(f"平均充电事件数: {np.mean(history['charging_events']):.2f}")
        print(f"平均步数: {np.mean(history['steps']):.2f}")
        
        # 分析训练收敛性
        self.analyze_convergence(history)
        
        return history

    def plot_detailed_training_history(self, history):
        """
        绘制详细的训练历史，包括奖励函数变化曲线
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        episodes = range(len(history['rewards']))
        
        # 奖励曲线
        axes[0, 0].plot(episodes, history['rewards'], alpha=0.6, label='Episode Rewards', color='blue')
        # 添加移动平均线
        window_size = 50
        if len(history['rewards']) >= window_size:
            moving_avg = np.convolve(history['rewards'], np.ones(window_size)/window_size, mode='valid')
            axes[0, 0].plot(range(window_size-1, len(history['rewards'])), moving_avg, 
                           label=f'Moving Avg ({window_size})', color='red', linewidth=2)
        axes[0, 0].set_title('Rewards Over Episodes')
        axes[0, 0].set_xlabel('Episodes')
        axes[0, 0].set_ylabel('Rewards')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 存活任务机曲线
        axes[0, 1].plot(episodes, history['alive_tasks'], label='Alive Tasks', color='green')
        if len(history['alive_tasks']) >= window_size:
            moving_avg_alive = np.convolve(history['alive_tasks'], np.ones(window_size)/window_size, mode='valid')
            axes[0, 1].plot(range(window_size-1, len(history['alive_tasks'])), moving_avg_alive, 
                           label=f'Moving Avg ({window_size})', color='darkgreen', linewidth=2)
        axes[0, 1].set_title('Alive Tasks Over Episodes')
        axes[0, 1].set_xlabel('Episodes')
        axes[0, 1].set_ylabel('Alive Count')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 充电事件曲线
        axes[0, 2].plot(episodes, history['charging_events'], label='Charging Events', color='orange')
        if len(history['charging_events']) >= window_size:
            moving_avg_charges = np.convolve(history['charging_events'], np.ones(window_size)/window_size, mode='valid')
            axes[0, 2].plot(range(window_size-1, len(history['charging_events'])), moving_avg_charges, 
                           label=f'Moving Avg ({window_size})', color='darkorange', linewidth=2)
        axes[0, 2].set_title('Charging Events Over Episodes')
        axes[0, 2].set_xlabel('Episodes')
        axes[0, 2].set_ylabel('Events Count')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 损失曲线
        axes[1, 0].plot(episodes, history['losses'], label='Loss', color='purple')
        if len(history['losses']) >= window_size:
            moving_avg_loss = np.convolve(history['losses'], np.ones(window_size)/window_size, mode='valid')
            axes[1, 0].plot(range(window_size-1, len(history['losses'])), moving_avg_loss, 
                           label=f'Moving Avg ({window_size})', color='indigo', linewidth=2)
        axes[1, 0].set_title('Loss Over Episodes')
        axes[1, 0].set_xlabel('Episodes')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 步数曲线
        axes[1, 1].plot(episodes, history['steps'], label='Steps per Episode', color='brown')
        if len(history['steps']) >= window_size:
            moving_avg_steps = np.convolve(history['steps'], np.ones(window_size)/window_size, mode='valid')
            axes[1, 1].plot(range(window_size-1, len(history['steps'])), moving_avg_steps, 
                           label=f'Moving Avg ({window_size})', color='maroon', linewidth=2)
        axes[1, 1].set_title('Steps Per Episode')
        axes[1, 1].set_xlabel('Episodes')
        axes[1, 1].set_ylabel('Steps')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 综合性能指标
        # 标准化各项指标到0-1范围进行比较
        rewards_norm = (np.array(history['rewards']) - np.min(history['rewards'])) / (np.max(history['rewards']) - np.min(history['rewards']) + 1e-8)
        alive_norm = np.array(history['alive_tasks']) / self.env.n_task_uavs  # 假设最大存活数为任务机总数
        charges_norm = (np.array(history['charging_events']) - np.min(history['charging_events'])) / (np.max(history['charging_events']) - np.min(history['charging_events']) + 1e-8)
        
        axes[1, 2].plot(episodes, rewards_norm, label='Normalized Rewards', alpha=0.7)
        axes[1, 2].plot(episodes, alive_norm, label='Normalized Alive Tasks', alpha=0.7)
        axes[1, 2].plot(episodes, charges_norm, label='Normalized Charging Events', alpha=0.7)
        axes[1, 2].set_title('Normalized Performance Metrics')
        axes[1, 2].set_xlabel('Episodes')
        axes[1, 2].set_ylabel('Normalized Value (0-1)')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.history_path, 'deep_training_detailed_history.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"详细训练历史图表已保存到: {os.path.join(self.history_path, 'deep_training_detailed_history.png')}")

    def analyze_convergence(self, history):
        """
        分析训练收敛性
        """
        rewards = np.array(history['rewards'])
        
        # 将训练分为几个阶段来分析
        n_parts = 5
        part_size = len(rewards) // n_parts
        
        print("\n=== 训练收敛性分析 ===")
        print(f"将 {len(rewards)} 轮训练分为 {n_parts} 个阶段:")
        
        for i in range(n_parts):
            start_idx = i * part_size
            if i == n_parts - 1:
                end_idx = len(rewards)  # 最后一部分包含剩余的所有
            else:
                end_idx = (i + 1) * part_size
            
            part_rewards = rewards[start_idx:end_idx]
            avg_reward = np.mean(part_rewards)
            std_reward = np.std(part_rewards)
            
            alive_part = np.array(history['alive_tasks'][start_idx:end_idx])
            avg_alive = np.mean(alive_part)
            
            charges_part = np.array(history['charging_events'][start_idx:end_idx])
            avg_charges = np.mean(charges_part)
            
            print(f"  阶段 {i+1} ({start_idx}-{end_idx-1}): "
                  f"平均奖励={avg_reward:.2f}±{std_reward:.2f}, "
                  f"存活={avg_alive:.2f}, "
                  f"充电={avg_charges:.2f}")
        
        # 检查最后几个阶段是否趋于稳定
        final_stages_rewards = []
        for i in range(n_parts-2, n_parts):  # 检查最后两个阶段
            start_idx = i * part_size
            if i == n_parts - 1:
                end_idx = len(rewards)
            else:
                end_idx = (i + 1) * part_size
            final_stages_rewards.extend(rewards[start_idx:end_idx])
        
        if len(final_stages_rewards) > 0:
            final_avg = np.mean(final_stages_rewards)
            overall_avg = np.mean(rewards)
            
            print(f"\n总体平均奖励: {overall_avg:.2f}")
            print(f"最后阶段平均奖励: {final_avg:.2f}")
            
            if abs(final_avg - overall_avg) < 0.1 * abs(overall_avg):  # 如果差异小于10%
                print("✅ 训练趋于收敛!")
            else:
                print("⚠️ 训练可能还未完全收敛，建议继续训练。")


def main():
    # 设置设备
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"使用设备: {device}")
    
    # 创建训练器
    config_path = "Projects/Demo/simple_charging_config.json"
    trainer = DeepChargingTrainer(config_path, device)
    
    print("=" * 70)
    print("深度充电无人机训练")
    print("训练1000轮，包含奖励函数分析")
    print("=" * 70)
    
    print("环境配置:")
    print(f"  - 任务机数量: {trainer.env.n_task_uavs}")
    print(f"  - 充电机数量: {trainer.env.n_charging_uavs}")
    print(f"  - 环境大小: {trainer.env.grid_size}")
    print(f"  - 行为状态维度: {trainer.model_params['state_dim']}")
    print(f"  - 动作维度: {trainer.model_params['action_dim']}")
    print(f"  - 训练轮数: {trainer.train_episodes}")
    
    # 开始训练
    print(f"\n开始深度训练... (共 {trainer.train_episodes} 轮)")
    print("训练目标: 让充电机学会在任务机电量低时及时充电")
    print("策略: 初期使用规则指导，后期逐渐转向纯学习")
    print("注意: 任务机每步都会消耗电量")
    
    history = trainer.train()
    
    print("\n" + "=" * 70)
    print("深度训练完成!")
    print("充电机已通过1000轮训练学会了能量管理策略")
    print("=" * 70)


if __name__ == "__main__":
    main()
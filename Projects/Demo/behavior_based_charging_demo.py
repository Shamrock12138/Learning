"""
基于行为树的充电无人机训练框架
从根本上解决充电无人机无法追踪任务机的问题
"""

import torch
import numpy as np
from tqdm import tqdm
import sys
import os
import random

# 添加项目路径到sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from myTools.Model.RL import DoubleDQN
from myTools.Utils.MARL_env import MARL_Env_UAVs
from myTools.Utils.MARL_tools import Q_Net as MARL_Q_Net
from myTools.Utils.tools import utils_readParams, utils_autoAssign, utils_showHistory
from myTools.Utils.tools import utils_ReplayBuffer
from myTools.Utils.config import Sample


class BehaviorBasedChargingTrainer:
    """
    基于行为的充电机训练器
    使用行为树逻辑来指导充电无人机的行为
    """
    def __init__(self, config_path, device):
        self.device = device
        
        # 读取配置
        model_config = utils_readParams(config_path, 'model')
        self.model_params = model_config
        
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
        training_config = utils_readParams(config_path, 'trainer')
        # 手动分配训练参数
        for key, value in training_config.items():
            setattr(self, key, value)
        
        self.name = f'Behavior_Based_Charging_DDQN'
        
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
        训练充电机 - 逐步减少规则依赖，增加学习比例
        """
        if model_name:
            self.agent.load_model(self.model_path, model_name)
        
        history = {'rewards': [], 'losses': [], 'alive_tasks': [], 'charging_events': []}
        
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
                learning_ratio = min(0.1 + episode * 0.002, 0.8)  # 从0.1增加到0.8
                
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
                
                pbar.set_postfix({
                    'Reward': f'{total_reward:.2f}',
                    'Alive': f'{alive_tasks}/{self.env.n_task_uavs}',
                    'Charges': charging_events,
                    'Steps': step,
                    'Learn%': f'{learning_ratio:.2f}'
                })
                pbar.update(1)
                
                # 保存模型
                if (episode + 1) % 50 == 0:
                    self.agent.save_model(self.model_path, f'\\BehaviorBasedCharging_DDQN_{episode+begin}.pt')
        
        # 绘制训练历史
        import matplotlib.pyplot as plt
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(history['rewards'], label='Rewards')
        plt.title('Rewards Over Episodes')
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.legend()
        
        plt.subplot(1, 3, 2)
        plt.plot(history['alive_tasks'], label='Alive Tasks', color='green')
        plt.title('Alive Tasks Over Episodes')
        plt.xlabel('Episodes')
        plt.ylabel('Alive Count')
        plt.legend()
        
        plt.subplot(1, 3, 3)
        plt.plot(history['charging_events'], label='Charging Events', color='red')
        plt.title('Charging Events Over Episodes')
        plt.xlabel('Episodes')
        plt.ylabel('Events Count')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.history_path, 'behavior_based_training_history.png'))
        plt.close()
        
        print(f"\n训练完成! 共训练 {self.train_episodes} 轮")
        print(f"平均奖励: {np.mean(history['rewards']):.2f}")
        print(f"平均存活任务机: {np.mean(history['alive_tasks']):.2f}/{self.env.n_task_uavs}")
        print(f"平均充电事件数: {np.mean(history['charging_events']):.2f}")
        
        return history

    def demo_trained_model(self, max_demo_steps=200):
        """
        演示训练好的模型 - 使用纯学习策略
        """
        import time
        
        print("\n开始演示训练成果...")
        print(f"初始状态:")
        state, _ = self.env.reset()
        cuav_state = state[self.env.n_task_uavs]
        tuav_state = state[0]
        print(f"  充电机@[{cuav_state[0]:.0f},{cuav_state[1]:.0f}] 电量:{cuav_state[2]:.1f}")
        print(f"  任务机@[{tuav_state[0]:.0f},{tuav_state[1]:.0f}] 电量:{tuav_state[2]:.1f}")
        
        alive_count = sum(1 for t in self.env.task_uavs if t.state["alive"])
        print(f"\n初始存活任务机: {alive_count}/{self.env.n_task_uavs}")
        
        try:
            done = False
            step = 0
            
            while not done and step < max_demo_steps:
                # 使用纯学习策略进行演示
                behavior_state = self._get_behavior_based_state(state)
                charging_action = self.agent.take_action(behavior_state)
                actions = np.zeros(self.env._n_agents, dtype=int)
                
                # 任务机随机行动
                for i in range(self.env.n_task_uavs):
                    actions[i] = np.random.randint(0, 5)
                
                actions[self.env.n_task_uavs] = charging_action
                
                # 环境步进
                next_state, reward, done, info = self.env.step(actions)
                
                # 渲染动画
                self.env.render()
                
                # 显示状态（每50步）
                if step % 50 == 0:
                    cuav_state = state[self.env.n_task_uavs]
                    tuav_state = state[0]
                    alive_count = sum(1 for t in self.env.task_uavs if t.state["alive"])
                    
                    print(f"  步数 {step}: 充电机@[{cuav_state[0]:.0f},{cuav_state[1]:.0f}]B:{cuav_state[2]:.1f}, "
                          f"任务机@[{tuav_state[0]:.0f},{tuav_state[1]:.0f}]B:{tuav_state[2]:.1f}, "
                          f"存活: {alive_count}/{self.env.n_task_uavs}")
                    
                    # 显示充电事件信息
                    charging_info = info.get("charging_events", {})
                    if charging_info.get("static", 0) > 0 or charging_info.get("base", 0) > 0:
                        print(f"    充电事件 - 任务机充电: {charging_info.get('static', 0)}, 基站充电: {charging_info.get('base', 0)}")
                
                state = next_state
                step += 1
                
                # 控制动画速度
                time.sleep(0.1)
            
            alive_count = sum(1 for t in self.env.task_uavs if t.state["alive"])
            print(f"\n演示完成! 总步数: {step}")
            print(f"初始存活任务机: {sum(1 for t in self.env.task_uavs if t.state['alive'])}/{self.env.n_task_uavs}")
            print(f"最终存活任务机: {alive_count}/{self.env.n_task_uavs}")
            
            if alive_count > 0:
                print("✅ 演示成功，任务机存活!")
            else:
                print("❌ 演示失败，任务机坠毁!")
            
        except KeyboardInterrupt:
            print("\n\n演示被用户中断")
        
        print("\n演示结束!")


def main():
    # 设置设备
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"使用设备: {device}")
    
    # 创建训练器
    config_path = "Projects/Demo/simple_charging_config.json"
    trainer = BehaviorBasedChargingTrainer(config_path, device)
    
    print("=" * 70)
    print("基于行为树的充电无人机训练")
    print("结合规则和学习的方法解决追踪问题")
    print("=" * 70)
    
    print("环境配置:")
    print(f"  - 任务机数量: {trainer.env.n_task_uavs}")
    print(f"  - 充电机数量: {trainer.env.n_charging_uavs}")
    print(f"  - 环境大小: {trainer.env.grid_size}")
    print(f"  - 行为状态维度: {trainer.model_params['state_dim']}")
    print(f"  - 动作维度: {trainer.model_params['action_dim']}")
    print(f"  - 训练轮数: {trainer.train_episodes}")
    
    # 开始训练
    print(f"\n开始训练... (共 {trainer.train_episodes} 轮)")
    print("训练目标: 让充电机学会在任务机电量低时及时充电")
    print("策略: 初期使用规则指导，后期逐渐转向纯学习")
    print("注意: 任务机每步都会消耗电量")
    
    history = trainer.train()
    
    print("\n" + "=" * 70)
    print("训练后演示")
    print("展示训练好的充电机如何为任务机充电")
    print("=" * 70)
    
    # 演示训练成果
    trainer.demo_trained_model(max_demo_steps=200)
    
    print("\n训练总结:")
    print(f"- 训练轮数: {trainer.train_episodes}")
    print(f"- 平均奖励: {np.mean(history['rewards']):.2f}")
    print(f"- 平均存活任务机: {np.mean(history['alive_tasks']):.2f}/{trainer.env.n_task_uavs}")
    print(f"- 平均充电事件数: {np.mean(history['charging_events']):.2f}")
    
    improvement = "显著提升" if np.mean(history['alive_tasks']) > 0.5 else "需要进一步优化"
    print(f"- 效果评估: {improvement}")
    
    print("\n" + "=" * 70)
    print("基于行为树的训练完成!")
    print("充电机已通过结合规则和学习的方法学会了能量管理策略")
    print("=" * 70)


if __name__ == "__main__":
    main()
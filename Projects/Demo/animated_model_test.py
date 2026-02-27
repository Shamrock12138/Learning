"""
深度训练模型的动画测试
使用训练好的模型进行动画演示
"""

import torch
import numpy as np
import sys
import os
import time
import matplotlib.pyplot as plt

# 添加项目路径到sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from myTools.Model.RL import DoubleDQN
from myTools.Utils.MARL_env import MARL_Env_UAVs
from myTools.Utils.MARL_tools import Q_Net as MARL_Q_Net
from myTools.Utils.tools import utils_readParams


def animate_trained_model(model_path=None):
    """
    动画演示训练好的模型
    """
    # 设置设备
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"使用设备: {device}")
    
    # 加载配置
    config_path = "Projects/Demo/simple_charging_config.json"
    model_config = utils_readParams(config_path, 'model')
    
    # 创建环境
    env = MARL_Env_UAVs(config_path)
    
    # 创建DDQN智能体
    qnet = MARL_Q_Net(
        state_dim=model_config['state_dim'], 
        action_dim=model_config['action_dim'], 
        Q_Net_config=config_path
    )
    
    agent = DoubleDQN(
        state_dim=model_config['state_dim'],
        action_dim=model_config['action_dim'],
        lr=model_config['lr'],
        gamma=model_config['gamma'],
        epsilon=model_config['epsilon'],
        target_update=model_config['target_update'],
        Q_Net=qnet,
        device=device
    )
    
    # 如果没有指定模型路径，则使用最新的模型
    if model_path is None:
        # 尝试加载训练1000轮后的模型
        model_path = "Projects/Demo/model/DeepCharging_DDQN_999.pt"
        try:
            agent.load_model("Projects/Demo/model/", "DeepCharging_DDQN_999.pt")
            print("✅ 加载模型: DeepCharging_DDQN_999.pt")
        except:
            # 如果没有1000轮的模型，尝试加载行为树模型
            try:
                agent.load_model("Projects/Demo/model/", "BehaviorBasedCharging_DDQN_299.pt")
                model_path = "Projects/Demo/model/BehaviorBasedCharging_DDQN_299.pt"
                print("✅ 加载模型: BehaviorBasedCharging_DDQN_299.pt")
            except:
                print("❌ 无法加载任何预训练模型")
                return
    
    print("=" * 70)
    print("深度训练模型动画测试")
    print("使用训练好的模型进行动画演示")
    print("=" * 70)
    
    print("环境配置:")
    print(f"  - 任务机数量: {env.n_task_uavs}")
    print(f"  - 充电机数量: {env.n_charging_uavs}")
    print(f"  - 环境大小: {env.grid_size}")
    print(f"  - 状态维度: {model_config['state_dim']}")
    print(f"  - 动作维度: {model_config['action_dim']}")
    print(f"  - 模型路径: {model_path}")
    
    # 演示多个回合
    num_demos = 5
    results = {
        'success_count': 0,
        'total_steps': [],
        'charging_events': [],
        'survival_rates': []
    }
    
    for demo_num in range(num_demos):
        print(f"\n--- 动画演示回合 {demo_num + 1}/{num_demos} ---")
        
        # 重置环境
        state, _ = env.reset()
        cuav_state = state[env.n_task_uavs]
        tuav_state = state[0]
        print(f"初始状态:")
        print(f"  充电机@[{cuav_state[0]:.0f},{cuav_state[1]:.0f}] 电量:{cuav_state[2]:.1f}")
        print(f"  任务机@[{tuav_state[0]:.0f},{tuav_state[1]:.0f}] 电量:{tuav_state[2]:.1f}")
        
        alive_count = sum(1 for t in env.task_uavs if t.state["alive"])
        print(f"  初始存活任务机: {alive_count}/{env.n_task_uavs}")
        
        done = False
        step = 0
        max_demo_steps = 300  # 增加演示步数
        
        # 记录充电事件
        charging_events = 0
        base_charging_events = 0
        charging_log = []  # 记录充电事件详情
        
        while not done and step < max_demo_steps:
            # 使用训练好的模型获取动作
            # 构建行为状态
            cuav_state = state[env.n_task_uavs].copy()
            tuav_state = state[0].copy()
            base_pos = env.base_station.position
            
            cuav_pos = cuav_state[:2]  # [x, y]
            tuav_pos = tuav_state[:2]  # [x, y]
            
            # 曼哈顿距离
            manhattan_dist_to_task = abs(tuav_pos[0] - cuav_pos[0]) + abs(tuav_pos[1] - cuav_pos[1])
            
            # 相对位置
            rel_tuav_pos = tuav_pos - cuav_pos
            rel_base_pos = base_pos - cuav_pos
            
            # 构建行为状态
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
            
            # 使用模型选择动作
            charging_action = agent.take_action(behavior_state)
            
            # 构建完整动作
            actions = np.zeros(env._n_agents, dtype=int)
            # 任务机随机行动
            for i in range(env.n_task_uavs):
                actions[i] = np.random.randint(0, 5)
            actions[env.n_task_uavs] = charging_action
            
            # 环境步进
            next_state, reward, done, info = env.step(actions)
            
            # 渲染动画
            env.render()
            
            # 统计充电事件
            prev_charging_events = charging_events
            prev_base_events = base_charging_events
            charging_events += info.get("charging_events", {}).get("static", 0)
            base_charging_events += info.get("charging_events", {}).get("base", 0)
            
            # 记录充电事件详情
            if charging_events > prev_charging_events:
                charging_log.append(f"步骤 {step}: 任务机充电 +{charging_events - prev_charging_events}")
            if base_charging_events > prev_base_events:
                charging_log.append(f"步骤 {step}: 基站充电 +{base_charging_events - prev_base_events}")
            
            # 显示状态（每100步）
            if step % 100 == 0:
                cuav_state = state[env.n_task_uavs]
                tuav_state = state[0]
                alive_count = sum(1 for t in env.task_uavs if t.state["alive"])
                
                print(f"  步数 {step}: 充电机@[{cuav_state[0]:.0f},{cuav_state[1]:.0f}]B:{cuav_state[2]:.1f}, "
                      f"任务机@[{tuav_state[0]:.0f},{tuav_state[1]:.0f}]B:{tuav_state[2]:.1f}, "
                      f"存活: {alive_count}/{env.n_task_uavs}")
                
                if info.get("charging_events", {}).get("static", 0) > 0:
                    print(f"    ⚡ 任务机充电事件: +{info['charging_events']['static']}")
                if info.get("charging_events", {}).get("base", 0) > 0:
                    print(f"    🔋 基站充电事件: +{info['charging_events']['base']}")
            
            state = next_state
            step += 1
            
            # 控制动画速度
            time.sleep(0.03)  # 减慢一点以便观察
        
        final_alive_count = sum(1 for t in env.task_uavs if t.state["alive"])
        print(f"\n动画演示回合 {demo_num + 1} 完成!")
        print(f"  总步数: {step}")
        print(f"  最终存活任务机: {final_alive_count}/{env.n_task_uavs}")
        print(f"  本回合充电事件数: {charging_events}")
        print(f"  本回合基站充电事件: {base_charging_events}")
        
        # 输出充电事件日志
        if charging_log:
            print("  充电事件详情:")
            for log_entry in charging_log[-5:]:  # 只显示最后5个事件
                print(f"    {log_entry}")
        
        if final_alive_count > 0:
            print("  ✅ 任务机存活!")
            results['success_count'] += 1
        else:
            print("  ❌ 任务机坠毁!")
        
        results['total_steps'].append(step)
        results['charging_events'].append(charging_events)
        results['survival_rates'].append(final_alive_count)
    
    # 总结结果
    print("\n" + "=" * 70)
    print("动画测试总结")
    print("=" * 70)
    print(f"演示回合数: {num_demos}")
    print(f"成功回合数: {results['success_count']} ({results['success_count']/num_demos*100:.1f}%)")
    print(f"平均步数: {np.mean(results['total_steps']):.1f}")
    print(f"平均充电事件数: {np.mean(results['charging_events']):.1f}")
    print(f"平均存活率: {np.mean(results['survival_rates']):.1f}/{env.n_task_uavs}")
    print(f"存活率: {np.mean(results['survival_rates']):.1f}/{env.n_task_uavs} ({np.mean(results['survival_rates'])/env.n_task_uavs*100:.1f}%)")
    
    if results['success_count'] > num_demos * 0.7:  # 如果成功率超过70%
        print("\n🎉 模型表现优秀！任务机存活率高。")
    elif results['success_count'] > num_demos * 0.4:  # 如果成功率超过40%
        print("\n👍 模型表现良好，有一定成功率。")
    else:
        print("\n⚠️  模型需要进一步优化，任务机存活率较低。")
    
    print("\n动画测试完成!")
    print("=" * 70)


def main():
    animate_trained_model()


if __name__ == "__main__":
    main()
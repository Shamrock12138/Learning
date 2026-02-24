#                        MARL相关工具函数
#                           2026/2/21
#                            shamrock

from collections import defaultdict
import torch.nn.functional as F

from myTools.Utils.tools import *

#---------------------- 多无人机充电 -------------------------
#                          2026/2/21

class Charging_UAVs_Rewards:
  '''
    充电无人机奖励的计算器
  '''
  def __init__(self, n_charging_uavs, 
               emergency_threshold, low_battery_threshold,
               reward_weights_path:str) -> None:
    self.n = n_charging_uavs
    self.emergency_threshold = emergency_threshold
    self.low_battery_threshold = low_battery_threshold
    self.weights = utils_readParams(reward_weights_path, 'cuav_reward_weights')

    self.last_cuav_batteries = None
    
    # 调试信息
    self.reward_history = []

  def reset(self):
    self.last_cuav_batteries = None
    self.reward_history = []

  def compute_rewards(self,
    is_out_of_bounds: np.ndarray,
    cuav_states: np.ndarray,         # shape: (n, 4) [x, y, battery, type]
    target_tasks: np.ndarray,        # shape: (n, 5) [x, y, battery, type, task_id] | task_id=-1表示无目标
    base_position: np.ndarray,       # shape: (2,)
    alive_task_mask: np.ndarray,     # shape: (n_task_uavs,) 布尔数组
    charging_mask: np.ndarray,       # shape: (n,) 布尔数组：本步是否成功充电
    charged_amount: np.ndarray,      # shape: (n,) 本步充电量
    dead_task_count: int = 0,        # 本步坠毁任务机数量
    step_count: int = 0,             # 当前步数（用于调试）) -> np.ndarray:
  ) -> np.ndarray:
    '''
      计算 rewards
        params:
          cuav_states: 充电机当前状态 [x, y, battery, type]
          target_tasks: 目标任务机状态 [x, y, battery, type, task_id]（task_id=-1表示无目标）
          base_position: 基站位置
          alive_task_mask: 任务机存活掩码
          charging_mask: 本步充电成功掩码
          charged_amount: 本步充电量
          dead_task_count: 本步坠毁任务机数量
          step_count: 当前步数
        returns:
          rewards - shape=(n,) 每个充电机的综合奖励
    '''
    n = self.n
    rewards = np.zeros(n)

    # ===== 0. 边界判定（首要安全检查）=====

    if np.any(is_out_of_bounds):
      print('out bounds')
      rewards[is_out_of_bounds] -= self.weights['out_of_bounds']
    
    # ===== 1. 自身电量充沛奖励（首要目标）=====
    cuav_batteries = cuav_states[:, 2]
    
    # 电量高时给予正奖励（首要目标）
    high_battery = (cuav_batteries > 0.8)
    rewards[high_battery] += self.weights['high_battery']  # 0.5
    
    # 电量中等时中性（0）
    # 电量低时给予惩罚（见第2部分）
    
    # ===== 2. 自身电量安全惩罚（强化首要目标）=====
    # 定义安全阈值
    safety_threshold = 0.7  # 70% 以上视为安全
    critical_threshold = 0.2  # 20% 以下为极度危险
    
    # 电量危险系数 (0~1)：越低越危险
    battery_danger = np.maximum(0, (safety_threshold - cuav_batteries) / safety_threshold)
    
    # 大幅增加低电量惩罚（惩罚强度与危险程度成正比）
    # 电量 70% → 0, 50% → 0.3, 30% → 0.6, 20% → 0.7, 10% → 0.85, 0% → 1.0
    rewards -= self.weights['battery_danger'] * battery_danger
    
    # ===== 3. 基站充电奖励（强化返航行为）=====
    current_pos = cuav_states[:, :2]
    at_base = np.all(np.abs(current_pos - base_position) < 1e-5, axis=1)
    
    # 1. 低电量时返航奖励（鼓励及时返航）
    low_battery = (cuav_batteries < 0.3)
    return_base_reward = at_base & low_battery
    if np.any(return_base_reward):
        # 电量越低，奖励越大
        urgency = (0.3 - cuav_batteries[return_base_reward]) / 0.3
        rewards[return_base_reward] += self.weights['return_base'] * (0.8 + 0.2 * urgency)
    
    # 2. 充电完成奖励（完成充电后）
    if self.last_cuav_batteries is not None:
        # 之前电量低 (<30%)，现在电量高 (>70%)，且在基站
        charged_at_base = at_base & \
                         (cuav_batteries > 0.7) & \
                         (self.last_cuav_batteries < 0.3)
        rewards[charged_at_base] += self.weights['charge_complete']
    
    # 更新缓存
    self.last_cuav_batteries = cuav_batteries.copy()
    
    # ===== 4. 任务机充电奖励（次要目标）=====
    valid_target_mask = (target_tasks[:, 4] != -1)
    
    # 1. 充电成功奖励（小幅奖励，避免过度关注任务机）
    charge_event = valid_target_mask & charging_mask
    if np.any(charge_event):
        rewards[charge_event] += self.weights['charge_success']  # 0.2 (降低！)
    
    # 2. 低电量任务机惩罚（小幅惩罚，避免完全忽视任务机）
    if np.any(valid_target_mask):
        target_batteries = target_tasks[:, 2]
        critical_mask = valid_target_mask & (target_batteries < 0.1)
        if np.any(critical_mask):
            rewards[critical_mask] -= self.weights['task_low_battery']  # 0.1 (小幅)
    
    # ===== 5. 其他惩罚 =====
    # 1. 时间步惩罚（小幅）
    rewards -= self.weights['time_step']  # 0.005 (降低)
    
    # 2. 任务机坠毁惩罚（保持）
    if dead_task_count > 0:
        team_penalty = (dead_task_count * self.weights['team_safe']) / n
        rewards -= team_penalty
    
    # ===== 6. 奖励裁剪（防止极端值）=====
    rewards = np.clip(rewards, -1.5, 1.5)
    
    return rewards

class Task_UAVs_Rewards:
  '''
    任务无人机奖励的计算器
  '''
  def __init__(self) -> None:
    pass  

  def reset(self):
    pass

  def compute_rewards(self):
    pass

class Q_Net(torch.nn.Module):
  '''
    用于DDQN的网络
  '''
  def __init__(self, state_dim, action_dim, Q_Net_config:str):
    super().__init__()
    config = utils_readParams(Q_Net_config, 'Q_Net')
    self.fc1 = torch.nn.Linear(state_dim, config['fc1_dim'])
    self.fc2 = torch.nn.Linear(config['fc1_dim'], config['fc2_dim'])
    self.fc3 = torch.nn.Linear(config['fc2_dim'], config['fc3_dim'])
    self.fc4 = torch.nn.Linear(config['fc3_dim'], action_dim)
    # self.bn1 = torch.nn.BatchNorm1d(config['fc1_dim'])
    # self.dropout = torch.nn.Dropout(0.2)

  def forward(self, x):
    # x = F.relu(self.bn1(self.fc1(x)))
    # x = self.dropout(x)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    return self.fc4(x)

#         ,--.                                                 ,--.     
#  ,---.  |  ,---.   ,--,--. ,--,--,--. ,--.--.  ,---.   ,---. |  |,-.  
# (  .-'  |  .-.  | ' ,-.  | |        | |  .--' | .-. | | .--' |     /  
# .-'  `) |  | |  | \ '-'  | |  |  |  | |  |    ' '-' ' \ `--. |  \  \  
# `----'  `--' `--'  `--`--' `--`--`--' `--'     `---'   `---' `--'`--' 

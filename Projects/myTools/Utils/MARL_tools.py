#                        MARL相关工具函数
#                           2026/2/21
#                            shamrock

from collections import defaultdict

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

    # 状态缓存（用于路径效率和探索奖励）
    self.last_positions = None          # shape: (n, 2)
    self.discovered_low_battery = [set() for _ in range(self.n)]  # 每个充电机的发现记录
    self.charging_counts = np.zeros(self.n, dtype=np.int32)       # 历史充电次数
    
    # 调试信息
    self.reward_history = []

  def reset(self):
    self.last_positions = None
    self.discovered_low_battery = [set() for _ in range(self.n)]
    self.charging_counts = np.zeros(self.n, dtype=np.int32)
    self.reward_history = []

  def compute_rewards(self,
    cuav_states: np.ndarray,         # shape: (n, 4) [x, y, battery, type]
    target_tasks: np.ndarray,        # shape: (n, 5) [x, y, battery, type, task_id] | task_id=-1表示无目标
    base_position: np.ndarray,       # shape: (2,)
    alive_task_mask: np.ndarray,     # shape: (n_task_uavs,) 布尔数组
    charging_mask: np.ndarray,       # shape: (n,) 布尔数组：本步是否成功充电
    charged_amount: np.ndarray,      # shape: (n,) 本步充电量
    dead_task_count: int = 0,        # 本步坠毁任务机数量
    step_count: int = 0              # 当前步数（用于调试）) -> np.ndarray:
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
    breakdown = defaultdict(lambda: np.zeros(n))
    
    # ===== 1. 充电奖励（向量化）=====
    charging_reward = np.zeros(n)
    
    # 有效目标掩码（task_id != -1）
    valid_target_mask = (target_tasks[:, 4] != -1)
    
    # 紧急程度计算（基于目标任务机电量）
    target_batteries = target_tasks[:, 2]
    emergency_bonus = np.zeros(n)
    emergency_bonus[valid_target_mask & (target_batteries < self.emergency_threshold)] = 0.7
    emergency_bonus[valid_target_mask & (target_batteries >= self.emergency_threshold) & 
                    (target_batteries < self.low_battery_threshold)] = 0.3
    
    # 充电效率（考虑传输损耗）
    efficiency_bonus = np.zeros(n)
    efficiency_bonus[charging_mask] = 0.2 * 0.9  # 固定效率0.9
    
    # 基础充电奖励
    base_charge_reward = np.zeros(n)
    base_charge_reward[charging_mask] = 0.8
    
    charging_reward = base_charge_reward + emergency_bonus + efficiency_bonus
    breakdown['charging'] = charging_reward
    
    # 更新历史充电计数（环境应在调用后更新，此处仅记录）
    if np.any(charging_mask):
        self.charging_counts += charging_mask.astype(np.int32)
    
    # ===== 2. 路径效率奖励（向量化）=====
    path_reward = np.zeros(n)
    if self.last_positions is not None:
        # 计算到目标的距离（仅对有效目标）
        current_pos = cuav_states[:, :2]  # (n, 2)
        target_pos = target_tasks[:, :2]  # (n, 2)
        
        # 当前距离和上一步距离
        current_dist = np.sum(np.abs(current_pos - target_pos), axis=1)  # (n,)
        last_dist = np.sum(np.abs(self.last_positions - target_pos), axis=1)  # (n,)
        
        # 距离减少量（仅对有效目标）
        dist_improvement = np.zeros(n)
        valid_move_mask = valid_target_mask & (last_dist > 0)
        dist_improvement[valid_move_mask] = last_dist[valid_move_mask] - current_dist[valid_move_mask]
        
        # 接近奖励（上限0.2）
        path_reward = np.minimum(0.1 * dist_improvement, 0.2)
        
        # 进入充电范围奖励（从>1到≤1）
        entered_range_mask = valid_move_mask & (last_dist > 1) & (current_dist <= 1)
        path_reward[entered_range_mask] += 0.3
    
    breakdown['path_efficiency'] = path_reward
    
    # 更新历史位置
    self.last_positions = cuav_states[:, :2].copy()
    
    # ===== 3. 能量管理奖励（向量化）=====
    energy_reward = np.zeros(n)
    batteries = cuav_states[:, 2]
    
    # 基站位置判断（曼哈顿距离=0）
    at_base = np.all(np.abs(cuav_states[:, :2] - base_position) < 1e-5, axis=1)
    energy_reward[at_base & (batteries < 100)] = 0.05
    
    # 电量区间奖励
    safe_mask = (batteries >= 40) & (batteries <= 90)
    high_mask = batteries > 90
    energy_reward[safe_mask] += 0.02
    energy_reward[high_mask] -= 0.01
    
    breakdown['energy_management'] = energy_reward
    
    # ===== 4. 团队贡献奖励（向量化）=====
    team_reward = np.zeros(n)
    alive_count = np.sum(alive_task_mask)
    
    # 基础团队奖励（所有充电机平分）
    base_team_reward = alive_count * 0.03
    team_reward += base_team_reward
    
    # 个人历史贡献奖励（基于历史充电次数）
    personal_bonus = np.minimum(self.charging_counts * 0.01, 0.2)
    team_reward += personal_bonus
    
    # 紧急救援额外奖励（基于历史紧急充电次数）
    # 注意：此处简化，实际可维护单独的紧急充电计数
    emergency_bonus_team = np.zeros(n)
    team_reward += emergency_bonus_team
    
    breakdown['team_contribution'] = team_reward
    
    # ===== 5. 探索奖励（需循环，但n小）=====
    exploration_reward = np.zeros(n)
    for i in range(n):
        if valid_target_mask[i] and target_batteries[i] < self.low_battery_threshold:
            task_id = int(target_tasks[i, 4])
            if task_id not in self.discovered_low_battery[i]:
                self.discovered_low_battery[i].add(task_id)
                exploration_reward[i] = 0.2  # 首次发现奖励
    
    breakdown['exploration'] = exploration_reward
    
    # ===== 6. 惩罚项（向量化）=====
    penalty = np.zeros(n)
    
    # 低电量惩罚
    low_battery_mask = batteries < 20
    penalty[low_battery_mask] = -(20 - batteries[low_battery_mask]) * 0.02
    
    # 任务机坠毁惩罚（团队失败，平均分摊）
    if dead_task_count > 0:
      penalty -= dead_task_count * 0.5 / n
    
    # 步数惩罚（鼓励高效）
    penalty -= 0.01
    
    breakdown['penalty'] = penalty
    
    # ===== 7. 加权组合 =====
    total_reward = (
      charging_reward * self.weights['charging'] +
      path_reward * self.weights['path_efficiency'] +
      energy_reward * self.weights['energy_management'] +
      team_reward * self.weights['team_contribution'] +
      exploration_reward * self.weights['exploration'] +
      penalty * self.weights['penalty']
    )
    
    # ===== 8. 调试记录 =====
    if step_count % 100 == 0:
      self.reward_history.append({
        'step': step_count,
        'mean_reward': np.mean(total_reward),
        'breakdown': {k: np.mean(v) for k, v in breakdown.items()}
      })
    
    return total_reward

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

#         ,--.                                                 ,--.     
#  ,---.  |  ,---.   ,--,--. ,--,--,--. ,--.--.  ,---.   ,---. |  |,-.  
# (  .-'  |  .-.  | ' ,-.  | |        | |  .--' | .-. | | .--' |     /  
# .-'  `) |  | |  | \ '-'  | |  |  |  | |  |    ' '-' ' \ `--. |  \  \  
# `----'  `--' `--'  `--`--' `--`--`--' `--'     `---'   `---' `--'`--' 

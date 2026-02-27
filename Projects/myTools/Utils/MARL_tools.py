#                        MARL相关工具函数
#                           2026/2/21
#                            shamrock

from collections import defaultdict
import torch.nn.functional as F

from myTools.Utils.tools import *
from myTools.Utils.MARL_config import *

#---------------------- 多无人机充电 -------------------------
#                          2026/2/21

class Charging_BaseStation(BaseStation):
  '''
    充电基站类
  '''
  def __init__(self, position: Tuple[int], charging_rate:float):
    super().__init__(position)
    self.charging_rate = charging_rate

  def can_charge(self, cuav:'Charging_UAV') -> bool:
    '''
      检查充电机是否可以充电
    '''
    cuav_position = cuav.position
    # print(cuav_position, self.position)
    # print(np.array_equal(np.round(cuav_position).astype(int), 
    #                   np.round(self.position).astype(int)))
    return np.array_equal(np.round(cuav_position).astype(int), 
                      np.round(self.position).astype(int))
  
  def provide_charging(self, cuav:'Charging_UAV') -> Tuple[float, float]:
    """
      为充电机提供充电服务，返回(充电量, 新电量)
        params:
          cuav_battery - uav当前电量
    """
    charge_amount = min(self.charging_rate, 100.0-cuav.battery)
    new_battery = cuav.battery+charge_amount
    cuav.battery = new_battery
    return charge_amount, new_battery
  
  def get_info(self) -> Dict[str, Any]:
    return {
      "position": tuple(self.position),
      "charging_rate": self.charging_rate
    }

class Charging_UAV(UAV):
  '''
    充电无人机类
  '''
  def __init__(self, uav_id:int, position:np.ndarray, battery:float,
               charging_rate:float=10.0, movement_cost:float=1.0,
               charging_cost_rate:float=0.9, distance:float=1.0) -> None:
    super().__init__(uav_id, position, battery, uav_type=1)
    utils_autoAssign(self)

  def can_charge_task(self, task_uav:'Task_UAV') -> bool:
    if not task_uav.state['alive']:
      return False
    distance = np.sum(np.abs(self.position-task_uav.position))
    if distance > self.distance:
      return False
    if self.battery < 10:
      return False
    if task_uav.battery >= 100:
      return False
    return True
  
  def charge_task_uav(self, task_uav:'Task_UAV') -> Tuple[float, float]:
    '''
      为任务机充电，返回(充电机消耗电量, 任务机获得电量)
    '''
    # print('yes')
    if not self.can_charge_task(task_uav):
      return -1.0, -1.0
    self.consume_battery(self.charging_rate)
    actual_charge = task_uav.charge_battery(self.charging_rate*self.charging_cost_rate)
    self.state["charging"] = True
    task_uav.state["charging"] = True
    return self.charging_rate, actual_charge
  
  def navigate_to_target(self, target_position:np.ndarray) -> int:
    '''
      导航到目标位置
    '''
    tx, ty = target_position
    cx, cy = self.position
    if cx < tx:
      return 4  # 右
    elif cx > tx:
      return 3  # 左
    elif cy < ty:
      return 1  # 上
    elif cy > ty:
      return 2  # 下
    else:
      return 0  # 已到达

class Task_UAV(UAV):
  '''
    任务无人机类
  '''
  def __init__(self, uav_id:int, position:np.ndarray, battery:float,
               movement_cost:float=0.5) -> None:  # 默认移动消耗0.5电量
    super().__init__(uav_id, position, battery, uav_type=0)
    utils_autoAssign(self)

  def random_action(self) -> int:
    '''
      生成随机动作
    '''
    return random.randint(0, 4)

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

    cuav_batteries = cuav_states[:, 2]
    cuav_positions = cuav_states[:, :2]

    dist_to_base = np.linalg.norm(cuav_positions - base_position, axis=1)
    at_base = dist_to_base < 1.5  # 距离基站 1.5 格内视为到达

    # ===== 0. 边界判定（首要安全检查）=====
    if np.any(is_out_of_bounds):
      rewards[is_out_of_bounds] -= self.weights['out_of_bounds']
    
    # ===== 1. 自身电量充沛奖励（首要目标）=====

    
    # ===== 2. 自身电量安全惩罚（强化首要目标）=====
    # 2.1 高电量奖励（鼓励保持电量充足）
    high_battery_mask = (cuav_batteries > 0.8)
    rewards[high_battery_mask] += self.weights['high_battery']  # +0.5
    
    # 2.2 低电量惩罚（随危险程度递增）
    low_battery_mask = (cuav_batteries < 0.3)
    battery_danger = np.maximum(0, (0.3 - cuav_batteries) / 0.3)  # 0~1
    rewards[low_battery_mask] -= self.weights['battery_danger'] * battery_danger  # 最大 -1.0
    
    # 2.3 电量耗尽惩罚（最严重）
    dead_mask = (cuav_batteries <= 0)
    rewards[dead_mask] -= self.weights['battery_dead']  # -50.0
    
    # ===== 3. 基站充电奖励（强化返航行为）=====
    # 3.1 低电量时靠近基站奖励（引导返航）
    need_charge_mask = (cuav_batteries < 0.4) & at_base
    if np.any(need_charge_mask):
        urgency = (0.4 - cuav_batteries[need_charge_mask]) / 0.4  # 电量越低越紧急
        rewards[need_charge_mask] += self.weights['return_base'] * (1.0 + urgency)  # +1.0~2.0
    
    # 3.2 成功在基站充电奖励（使用缓存对比）
    if self.last_cuav_batteries is not None:
        # 之前低电量 (<30%)，现在高电量 (>70%)，且在基站 = 成功充电
        charged_success = cuav_batteries > self.last_cuav_batteries
        if np.any(charged_success):
            print("yes")
            rewards[charged_success] += self.weights['charge_complete']  # +5.0
    
    # 更新电量缓存（用于下次判断是否完成充电）
    self.last_cuav_batteries = cuav_batteries.copy()
    
    # 3.3 距离基站越近奖励（连续引导）
    # 只在需要充电时生效，避免一直待在基站
    if np.any(need_charge_mask):
        dist_reward = -dist_to_base * self.weights['dist_to_base']  # -0.01 * dist
        rewards[need_charge_mask] += dist_reward
    
    # # ===== 4. 任务机充电奖励（次要目标）=====
    # valid_target_mask = (target_tasks[:, 4] != -1)
    
    # # 1. 充电成功奖励（小幅奖励，避免过度关注任务机）
    # charge_event = valid_target_mask & charging_mask
    # if np.any(charge_event):
    #     rewards[charge_event] += self.weights['charge_success']  # 0.2 (降低！)
    
    # # 2. 低电量任务机惩罚（小幅惩罚，避免完全忽视任务机）
    # if np.any(valid_target_mask):
    #     target_batteries = target_tasks[:, 2]
    #     critical_mask = valid_target_mask & (target_batteries < 0.1)
    #     if np.any(critical_mask):
    #         rewards[critical_mask] -= self.weights['task_low_battery']  # 0.1 (小幅)
    
    # ===== 4. 任务机充电奖励（关键行为奖励）=====
    # 4.1 任务机充电成功奖励（大幅奖励，鼓励充电行为）
    if np.any(charging_mask):
        rewards[charging_mask] += self.weights['charge_success']  # +5.0
    
    # 4.2 低电量任务机接近奖励（鼓励前往低电量任务机）
    for i, cuav_state in enumerate(cuav_states):
        if not alive_task_mask.any():  # 如果所有任务机都死亡，跳过此部分
            continue
            
        # 查找最近的低电量任务机
        for j, task_alive in enumerate(alive_task_mask):
            if not task_alive:
                continue
            task_state = target_tasks[j]
            if task_state[2] < 0.4:  # 任务机电量低于40%
                # 计算充电机与低电量任务机的距离
                dist_to_task = np.linalg.norm(cuav_positions[i] - task_state[:2])
                if dist_to_task < 5.0:  # 扩大影响范围到5格
                    proximity_bonus = (5.0 - dist_to_task) * 0.3  # 距离越近奖励越大，调整系数
                    rewards[i] += proximity_bonus
    
    # ===== 5. 其他惩罚 =====
    # 1. 时间步惩罚（小幅）
    rewards -= self.weights['time_step']  # 0.001 (降低)
    
    # 2. 任务机坠毁惩罚（严厉惩罚，因为保护任务机是主要目标）
    if dead_task_count > 0:
        team_penalty = (dead_task_count * self.weights['team_safe']) / n
        rewards -= team_penalty
    
    # ===== 6. 奖励裁剪（防止极端值）=====
    # rewards = np.clip(rewards, -5.0, 5.0)
    
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
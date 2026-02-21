#                         多智能体强化学习环境
#                           2026/1/22
#                            shamrock

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from myTools.Utils.MARL_config import *
from myTools.Utils.MARL_tools import *
from myTools.Utils.tools import *

#---------------------- 多无人机充电场景 -------------------------
#                         2026/2/18

class MARL_Env_UAVs(MARL_EnvConfig):
  '''
    多无人机充电场景。
      包含两类无人机：任务机（Task UAVs）、充电机（Charging UAVs）
        1. 任务机 type=0：随机移动，电量消耗快（每步-2）
        2. 充电机 type=1：电量充足（初始100），可给相邻任务机充电（每步-1移动，充电-10）
      终止条件：任一任务机电量≤0 或 达到最大步数
      状态空间 - [x, y, battery, type]
      动作空间 - 0静止 1上 2下 3左 4右 5充电模式
  '''
  def __init__(self, env_config:str,
               grid_size: Tuple[int, int]=(15, 15)):
    env_config = utils_readParams(env_config, 'env')
    utils_setAttr(self, env_config)
    utils_autoAssign(self)
    self.n_agents = self.n_task_uavs+self.n_charging_uavs
    super().__init__(self.n_agents, self.state_dim, self.action_dim)

    self.cuav_rewards_computer = Charging_UAVs_Rewards(self.n_charging_uavs, 20.0, 40.0,
                                                       'Knowledge\MutilAgentReinforcementLearning\MultiUAVsCharging\config.json')
    self.tuav_rewards_computer = Task_UAVs_Rewards()
    
    self._init_env()
    self.step_count = 0
    self._done = False

  def _generate_unique_position(self, n:int) -> list:
    '''
      生成 n 个不重叠的位置
    '''
    positions = []
    while len(positions) < n:
      pos = (
        random.randint(0, self.grid_size[0]-1),
        random.randint(0, self.grid_size[1]-1)
      )
      if pos not in positions:
        positions.append(pos)
    return positions

  def _init_env(self):
    '''
      初始化基站和无人机
    '''
    base_pos = (self.grid_size[0]//2, self.grid_size[1]//2)
    self.base_station = Charging_BaseStation(position=base_pos, charging_rate=20.0)

    uav_position = self._generate_unique_position(self.n_agents)
    self.task_uavs:List[Task_UAV] = []
    for i in range(self.n_task_uavs):
      pos = uav_position[i]
      battery = 60.0+random.uniform(-5, 5)
      task_uav = Task_UAV(
        uav_id=i,
        position=np.array(pos, dtype=np.float32),
        battery=battery,
        movement_cost=2.0
      )
      self.task_uavs.append(task_uav)
    
    self.charging_uavs:List[Charging_UAV] = []
    for i in range(self.n_charging_uavs):
      pos = uav_position[self.n_task_uavs+i]
      battery = 90.0 + random.uniform(-5, 5)
      charging_uav = Charging_UAV(
        uav_id=self.n_task_uavs+i,
        position=np.array(pos, dtype=np.float32),
        battery=battery,
        charging_rate=10.0,
        movement_cost=1.0,
        charging_cost_rate=0.9
      )
      self.charging_uavs.append(charging_uav)

  def _get_obs(self) -> np.ndarray:
    '''
      返回全局状态 [(x, y, B, type), ..., ] 前为tasks UAVs，
    '''
    obs = []
    for task_uav in self.task_uavs:
      obs.append(np.array([
        task_uav.position[0],
        task_uav.position[1],
        task_uav.battery,
        task_uav.type
      ], dtype=np.float32))
    for cuav in self.charging_uavs:
      obs.append(np.array([
        cuav.position[0],
        cuav.position[1],
        cuav.battery,
        cuav.type
      ], dtype=np.float32))
    return np.array(obs, dtype=np.float32)

  def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
    positions = self._generate_unique_position(self.n_agents+1)
    base_pos = positions[0]
    uav_positions = positions[1:]
    self.base_station.position = np.array(base_pos, dtype=np.float32)

    for i, task_uav in enumerate(self.task_uavs):
      pos = uav_positions[i]
      battery = 60.0+random.uniform(-5, 5)
      task_uav.reset(
        position=np.array(pos, dtype=np.float32),
        battery=battery
      )
    for i, charging_uav in enumerate(self.charging_uavs):
      pos = uav_positions[self.n_task_uavs+i]
      battery = 90.0+random.uniform(-5, 5)
      charging_uav.reset(
        position=np.array(pos, dtype=np.float32),
        battery=battery
      )
    self.step_count = 0
    self.done = False
    self.charged_tasks = set()

    self.tuav_rewards_computer.reset()
    self.cuav_rewards_computer.reset()
    
    return self._get_obs(), {"base_station": tuple(self.base_station.position)}

  def _handle_charging_mode(self, cuav:Charging_UAV) -> Tuple[bool, float]:
    '''
      向可充电范围内特定的uav充电
    '''
    target_list = self._find_charging_target(cuav)
    target_idx = None
    for idx in target_list:
      task_uav = self.task_uavs[idx]
      distance = np.sum(np.abs(cuav.position-task_uav.position))
      if distance <= 1:
        target_idx = idx
        break
    if target_idx:
      cost, charge = cuav.charge_task_uav(self.task_uavs[target_idx])
      return True, charge
    return False, 0.0

  def _find_charging_target(self, cuav:Charging_UAV) -> list:
    '''
      寻找 cuav 此时的充电目标，返回需要充电的uav下标
    '''
    target_idx = []
    for idx, task_uav in enumerate(self.task_uavs):
      if not task_uav.state['alive']:
        continue
      if task_uav.battery < 40:
        target_idx.append(idx)
    return target_idx
  
  def _compute_rewards(self, charging_mask, charged_amount, info:dict) -> np.ndarray:
    '''
      返回 每个无人机的奖励rewards
    '''
    rewards = np.zeros(self.n_agents)

    cuav_states = np.array([
      [cuav.position[0], cuav.position[1], cuav.battery, cuav.type]
      for cuav in self.charging_uavs
    ])

    # Charging UAVs

    target_tasks = np.full((self.n_charging_uavs), -1.0)
    for i, cuav in enumerate(self.charging_uavs):
      targets = self._find_charging_target(cuav)
      if targets:
        task_idx = targets[0]
        task_uav = self.task_uavs[task_idx]
        target_tasks[i] = [
          task_uav.position[0], task_uav.position[1], 
          task_uav.battery, task_uav.type, task_idx
        ]

    cuav_rewards = self.cuav_rewards_computer(
      cuav_states=cuav_states,
      target_tasks=target_tasks,
      base_position=self.base_station.position,
      alive_task_mask=np.array([t.state["alive"] for t in self.task_uavs]),
      charging_mask=charging_mask,
      charged_amount=charged_amount,
      dead_task_count=info["dead_uavs"],
      step_count=self.step_count
    )

    for i in range(self.n_charging_uavs):
      global_idx = self.n_task_uavs+i
      rewards[global_idx] = cuav_rewards[i]

    # Task UAVs
    # TODO

    return rewards

  def _check_termination(self) -> bool:
    '''
      检查终止条件
    '''
    any_dead = any(not t.state['alive'] for t in self.task_uavs)
    max_steps_reached = self.step_count >= self.max_steps
    return any_dead or max_steps_reached

  def step(self, actions:np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, Dict[str, Any]]:
    '''
      所有 UAVs 执行 actions 动作
        params:
          actions - 前面是tasks UAVs的动作，后面是charging UAVs的动作
        return:
          states - 
          rewards - [n_agents, ] 前面是tasks UAVs的rewards
          dones - 
          info - dict
    '''
    self.step_count += 1
    self.charged_tasks = set()
    # rewards = np.zeros(self.n_agents)
    info = {
      "charging_events": {"static": 0, "base": 0},
      "dead_uavs": 0,
      "team_reward": 0.0
    }
    charging_mask = np.zeros(self.n_charging_uavs, dtype=bool)
    charged_amount = np.zeros(self.n_charging_uavs)

    # 处理充电机
    for i in range(self.n_charging_uavs):
      cuav:Charging_UAV = self.charging_uavs[i]
      global_idx = self.n_task_uavs+i
      action = int(actions[global_idx])
      at_base = self.base_station.can_charge(cuav)
      # 基站充电
      if at_base and action == 0:
        charge_amount, new_battery = self.base_station.provide_charging(cuav.battery)
        cuav.battery = new_battery
        # rewards[global_idx] += 0.5
        charging_mask[i] = True
        charged_amount[i] = charge_amount
        info["charging_events"]["base"] += 1
        continue
      # 充电模式
      if action == 5:
        is_charging, charge_amount = self._handle_charging_mode(cuav)
        if is_charging:
          charging_mask[i] = True
          charged_amount[i] = charge_amount
        continue
      # 正常移动
      if action != 0:
        cuav.move(action, self.grid_size)
        cuav.consume_battery(cuav.movement_cost)
    
    # 处理任务机
    for i in range(self.n_task_uavs):
      task_uav:Task_UAV = self.task_uavs[i]
      if not task_uav.state["alive"]:
        continue
      if i not in self.charged_tasks:
        action = int(actions[i])
        if action != 0:
          task_uav.move(action, self.grid_size)
          task_uav.consume_battery(task_uav.movement_cost)

    rewards = self._compute_rewards(charging_mask, charged_amount, info)
    self.done = self._check_termination()
    return self._get_obs(), rewards, self.done, info

  def _render_init(self):
    matplotlib.use('TkAgg')
    plt.ion()
    self._fig, self._ax = plt.subplots(figsize=(6, 6.5))
    self._fig.canvas.manager.set_window_title('Multi-UAV Charging Environment')
    self._fig.tight_layout(pad=1.5)
    plt.show(block=False)
    plt.pause(0.1)

  def render(self) -> None:
    '''
      基站：金色+充电标识
      任务无人机：蓝色
      充电无人机：红色
    '''
    if not hasattr(self, '_fig') or self._fig is None:
      self._render_init()

    ax = self._ax
    ax.clear()

    # 设置坐标轴
    ax.set_xlim(-0.5, self.grid_size[0] - 0.5)
    ax.set_ylim(-0.5, self.grid_size[1] - 0.5)
    ax.set_aspect('equal')

    ax.grid(True, alpha=0.15, linestyle=':', linewidth=0.3, color='gray')

    # 移除坐标轴刻度标签（保持刻度线但不显示数字）
    ax.set_xticks(range(self.grid_size[0]))
    ax.set_yticks(range(self.grid_size[1]))
    ax.set_xticklabels([])  # 移除X轴数字
    ax.set_yticklabels([])  # 移除Y轴数字
    
    # 淡化坐标轴边框
    for spine in ax.spines.values():
      spine.set_alpha(0.3)
      spine.set_linewidth(0.5)
    
    # === 标题（更简洁）===
    alive_count = sum(1 for t in self.task_uavs if t.state["alive"])
    ax.set_title(
      f'Step {self.step_count}/{self.max_steps} | '
      f'Alive: {alive_count}/{self.n_task_uavs}',
      fontsize=10, fontweight='bold', pad=10, color='#333333'
    )
    
    # 绘制基站（金色星形）
    bx, by = self.base_station.position.astype(int)
    ax.plot(bx, by, marker='*', markersize=12, color='#FFD700', 
            markeredgecolor='#8B7500', markeredgewidth=1.0, label='BS')
    # ax.text(bx, by-0.8, 'BASE', ha='center', fontsize=9, fontweight='bold', color='#8B7500')
    
    # === 绘制任务无人机（修改版）===
    for i, uav in enumerate(self.task_uavs):
      if not uav.state["alive"]:
        continue
      
      x, y = uav.position
      battery_level = uav.battery / 100.0
      
      # 五级颜色渐变
      if battery_level > 0.8:
        color = (0.0, 0.6, 0.0, 0.95)  # 深绿
      elif battery_level > 0.6:
        color = (0.0, 0.8, 0.0, 0.95)  # 翠绿
      elif battery_level > 0.4:
        color = (0.8, 0.8, 0.0, 0.95)  # 深黄
      elif battery_level > 0.2:
        color = (0.9, 0.4, 0.0, 0.95)  # 深橙
      else:
        color = (0.8, 0.0, 0.0, 0.95)  # 深红
      
      # 无人机主体（圆圈）
      circle = plt.Circle((x, y), 0.25, ec='black', linewidth=1.0)
      ax.add_patch(circle)
      
      # 电量数字（白色+黑色描边）
      battery_text = f'{int(uav.battery)}'
      ax.text(
        x, y, battery_text,
        ha='center', va='center',
        fontsize=8,       # 适当增大
        fontweight='bold',
        color='black',    # 纯黑字
        bbox=dict(
          boxstyle='round,pad=0.25',  # 圆角+内边距
          facecolor='white',          # 纯白背景
          alpha=0.95,                 # 高不透明度
          edgecolor='black',          # 黑色边框
          linewidth=1.0
        )
      )

      # 标签（上移，带背景框）
      ax.text(
        x, y + 0.35, f'T{i}',
        ha='center', va='center',
        fontsize=5,
        fontweight='bold',
        color=color,
        bbox=dict(
          boxstyle='round,pad=0.1',
          facecolor='white',
          alpha=0.7,
          edgecolor='none'
        )
      )

    # 绘制充电无人机（红色）
    for i, uav in enumerate(self.charging_uavs):
      x, y = uav.position
      battery_level = uav.battery / 100.0
      
      # 充电机颜色（红色系）
      color = (1.0, 0.3 * (1 - battery_level), 0.3 * (1 - battery_level), 0.9)
      
      # 无人机主体（六边形，区别于任务机）
      hexagon = patches.RegularPolygon(
          (x, y), numVertices=6, radius=0.28, 
          orientation=0, color=color, ec='black', linewidth=1.2
      )
      ax.add_patch(hexagon)
      
      # 充电标识（闪电符号）
      ax.plot([x-0.10, x+0.10], [y+0.10, y-0.10], 'yellow', linewidth=1.5)
      ax.plot([x-0.10, x], [y+0.10, y+0.03], 'yellow', linewidth=1.5)
      
      # 电量条
      bar_width = 0.5
      bar_height = 0.12
      bar = plt.Rectangle(
        (x - bar_width/2, y - 0.48),
        bar_width * battery_level,
        bar_height,
        color='gold',
        ec='black',
        linewidth=0.7,
        alpha=0.9
      )
      ax.add_patch(bar)
      ax.add_patch(plt.Rectangle(
        (x - bar_width/2, y - 0.48),
        bar_width,
        bar_height,
        fill=False,
        ec='black',
        linewidth=0.7
      ))
      
      # 标签
      ax.text(x, y - 0.68, f'C{i}',  # 移除电量百分比
              ha='center', va='center', fontsize=6, fontweight='bold', color='white')
    
    # === 底部状态栏（紧凑版）===
    status_text = (
      f"Grid: {self.grid_size[0]}x{self.grid_size[1]} | "
      f"Tasks: {self.n_task_uavs} | "
      f"Chargers: {self.n_charging_uavs}"
    )
    ax.text(
      self.grid_size[0]/2, -0.8,
      status_text,
      ha='center', va='center',
      fontsize=7, color='#555555',
      bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='none')
    )
    
    # 实时更新
    self._fig.canvas.draw()
    self._fig.canvas.flush_events()
    plt.pause(0.01)  # 微小延迟确保渲染完成
    
#         ,--.                                                 ,--.     
#  ,---.  |  ,---.   ,--,--. ,--,--,--. ,--.--.  ,---.   ,---. |  |,-.  
# (  .-'  |  .-.  | ' ,-.  | |        | |  .--' | .-. | | .--' |     /  
# .-'  `) |  | |  | \ '-'  | |  |  |  | |  |    ' '-' ' \ `--. |  \  \  
# `----'  `--' `--'  `--`--' `--`--`--' `--'     `---'   `---' `--'`--' 


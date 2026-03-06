#                         多智能体强化学习环境
#                           2026/1/22
#                            shamrock

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, RegularPolygon

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
               grid_size: Tuple[int, int]=(10, 10)):
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

  def _normalize_state(self, state:np.ndarray):
    '''
      归一化 state
    '''
    state_normalized = state.copy()
    state_normalized[0] = state[0] / self.grid_size[0]      # x 归一化
    state_normalized[1] = state[1] / self.grid_size[1]      # y 归一化
    state_normalized[2] = state[2] / 100.0                  # battery 归一化
    return state_normalized

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
        movement_cost=0.5  # 任务机每步移动消耗0.5电量
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
        charging_cost_rate=0.9,
        distance=self.distance
      )
      self.charging_uavs.append(charging_uav)

  def _get_obs(self) -> np.ndarray:
    '''
      返回全局状态 [(x, y, B, type), ..., ] 前为tasks UAVs，
    '''
    obs = []
    for task_uav in self.task_uavs:
      state = np.array([
        task_uav.position[0],
        task_uav.position[1],
        task_uav.battery,
        task_uav.type
      ], dtype=np.float32)
      nor_state = self._normalize_state(state)
      obs.append(nor_state)
    for cuav in self.charging_uavs:
      state = np.array([
        cuav.position[0],
        cuav.position[1],
        cuav.battery,
        cuav.type
      ], dtype=np.float32)
      nor_state = self._normalize_state(state)
      obs.append(nor_state)
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
    for idx in target_list:
      task_uav = self.task_uavs[idx]
      cost, charge = cuav.charge_task_uav(task_uav)
      if cost >= 0.0:
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
      if task_uav.battery <= 60 and cuav.can_charge_task(task_uav):
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

    target_tasks = np.full((self.n_charging_uavs, 5), -1.0)
    for i, cuav in enumerate(self.charging_uavs):
      targets = self._find_charging_target(cuav)
      if targets:
        task_idx = targets[0]
        task_uav = self.task_uavs[task_idx]
        target_tasks[i] = [
          task_uav.position[0], task_uav.position[1], 
          task_uav.battery, task_uav.type, float(task_idx)
        ]

    cuav_rewards = self.cuav_rewards_computer.compute_rewards(
      is_out_of_bounds=info['is_out_of_bounds'],
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
    any_dead |= any(not c.state['alive'] for c in self.charging_uavs)
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
      "team_reward": 0.0, 
      "is_out_of_bounds": np.zeros(self.n_charging_uavs, dtype=bool),
    }
    charging_mask = np.zeros(self.n_charging_uavs, dtype=bool)
    charged_amount = np.zeros(self.n_charging_uavs)

    # 处理充电机
    for i in range(self.n_charging_uavs):
      cuav:Charging_UAV = self.charging_uavs[i]
      if not cuav.state['alive']:
        info['dead_uavs'] += 1
        continue
      global_idx = self.n_task_uavs+i
      action = int(actions[global_idx])
      # print(action)
      # 充电模式 - 优先处理充电任务
      if action == 5:
        is_charging, charge_amount = self._handle_charging_mode(cuav)
        if is_charging:
          charging_mask[i] = True
          charged_amount[i] = charge_amount
        continue
      # 基站充电
      elif self.base_station.can_charge(cuav) and action == 0:
        # print('no')
        charge_amount, new_battery = self.base_station.provide_charging(cuav)
        cuav.battery = new_battery
        info["charging_events"]["base"] += 1
        continue
      # 正常移动
      elif action != 0:
        _, info["is_out_of_bounds"][i] = cuav.move(action, self.grid_size)
        cuav.consume_battery(cuav.movement_cost)
    
    # 处理任务机
    for i in range(self.n_task_uavs):
      task_uav:Task_UAV = self.task_uavs[i]
      if not task_uav.state['alive']:
        info['dead_uavs'] += 1
        continue
      if i not in self.charged_tasks:
        action = int(actions[i])
        if action != 0:
          task_uav.move(action, self.grid_size)
          # 移动时消耗电量
          task_uav.consume_battery(task_uav.movement_cost)
        # 无论是否移动，每步都消耗基础电量
        task_uav.consume_battery(0.2)  # 每步消耗0.2电量，降低消耗率

    rewards = self._compute_rewards(charging_mask, charged_amount, info)
    self.done = self._check_termination()
    return self._get_obs(), rewards, self.done, info

  def _render_init(self):
    # matplotlib.use('TkAgg')
    plt.ion()
    self._fig, self._ax = plt.subplots(figsize=(6, 6.5))
    self._fig.canvas.manager.set_window_title('Multi-UAV Charging Environment')
    self._fig.tight_layout(pad=1.5)
    plt.show(block=False)
    plt.pause(0.1)

  def render(self) -> None:
      '''
        可视化渲染（美化版 - 电量突出显示）
        基站：金色星形
        任务无人机：蓝色圆圈 + 大电量显示（最上层）
        充电无人机：红色六边形 + 闪电图标
        充电连接：绿色动态连线
      '''
      if not hasattr(self, '_fig') or self._fig is None:
          self._render_init()

      ax = self._ax
      ax.clear()

      # ===== 1. 画布基础设置 =====
      ax.set_xlim(-1, self.grid_size[0])
      ax.set_ylim(-1, self.grid_size[1])
      ax.set_aspect('equal')
      ax.set_facecolor('#F5F7FA')  # 浅蓝灰背景，更柔和

      # 网格（淡化）
      ax.grid(True, alpha=0.08, linestyle='--', linewidth=0.5, color='#BBBBBB')

      # 隐藏刻度标签
      ax.set_xticks(range(self.grid_size[0]))
      ax.set_yticks(range(self.grid_size[1]))
      ax.set_xticklabels([])
      ax.set_yticklabels([])

      # 边框淡化
      for spine in ax.spines.values():
          spine.set_alpha(0.15)
          spine.set_linewidth(0.5)
          spine.set_color('#888888')

      # ===== 2. 标题与状态栏 =====
      alive_count = sum(1 for t in self.task_uavs if t.state.get("alive", True))
      ax.set_title(
          f'Step {self.step_count}/{self.max_steps}  |  '
          f'Alive: {alive_count}/{self.n_task_uavs}  |  '
          f'Charging: {sum(1 for c in self.charging_uavs if c.state.get("is_charging", False))}',
          fontsize=11, fontweight='bold', pad=12, color='#1A252F'
      )

      # ===== 3. 绘制基站（金色星形 + 光晕效果）=====
      bx, by = self.base_station.position.astype(int)
      # 光晕
      ax.plot(bx, by, marker='o', markersize=20, color='#FFD700', alpha=0.25, zorder=1)
      # 主体
      ax.plot(bx, by, marker='*', markersize=16, color='#FFD700',
              markeredgecolor='#B8860B', markeredgewidth=2.0, zorder=10, label='Base Station')
      ax.text(bx, by - 1.0, 'BASE', ha='center', fontsize=9, fontweight='bold', color='#B8860B')

      # ===== 4. 绘制充电连接线（在无人机下方，zorder=3）=====
      for i, uav in enumerate(self.charging_uavs):
          x, y = uav.position
          target_id = uav.state.get("target_task_id", -1)
          is_charging = uav.state.get("is_charging", False)

          if is_charging and target_id >= 0 and target_id < len(self.task_uavs):
              target_uav = self.task_uavs[target_id]
              if target_uav.state.get("alive", True):
                  tx, ty = target_uav.position
                  # 多层线模拟能量流动
                  ax.plot([x, tx], [y, ty],
                          color='#00FF00', linewidth=4.0, alpha=0.3, zorder=3)
                  ax.plot([x, tx], [y, ty],
                          color='#00CC00', linewidth=2.5, alpha=0.7, zorder=4)
                  # 流动点动画
                  flow_pos = (self.step_count % 20) / 20.0
                  fx = x + (tx - x) * flow_pos
                  fy = y + (ty - y) * flow_pos
                  ax.plot(fx, fy, marker='o', markersize=6, color='#FFFF00', 
                          alpha=0.9, zorder=5)

      # ===== 5. 绘制任务无人机（美化版 - 电量突出）=====
      for i, uav in enumerate(self.task_uavs):
          if not uav.state.get("alive", True):
              # 坠毁标记（灰色 X）
              x, y = uav.position
              ax.plot(x, y, marker='x', markersize=12, color='#999999', 
                      alpha=0.6, linewidth=3, zorder=2)
              ax.text(x, y - 0.5, f'T{i}', ha='center', fontsize=6, 
                      color='#999999', alpha=0.5, zorder=2)
              continue

          x, y = uav.position
          battery_level = uav.battery / 100.0

          # 电量颜色映射（高对比度）
          if battery_level > 0.75:
              face_color = '#2ECC71'  # 翠绿
              edge_color = '#27AE60'
              battery_text_color = '#FFFFFF'
              battery_bg_color = '#27AE60'
          elif battery_level > 0.50:
              face_color = '#F39C12'  # 橙色
              edge_color = '#D68910'
              battery_text_color = '#FFFFFF'
              battery_bg_color = '#D68910'
          elif battery_level > 0.25:
              face_color = '#E74C3C'  # 红色
              edge_color = '#C0392B'
              battery_text_color = '#FFFFFF'
              battery_bg_color = '#C0392B'
          else:
              face_color = '#9B59B6'  # 紫色（紧急）
              edge_color = '#8E44AD'
              battery_text_color = '#FFFFFF'
              battery_bg_color = '#8E44AD'
              # 紧急状态添加闪烁效果
              if self.step_count % 10 < 5:
                  face_color = '#FF0000'
                  edge_color = '#CC0000'

          # 无人机主体（带阴影的圆圈，zorder=6）
          circle = plt.Circle((x, y), 0.35,
                              facecolor=face_color,
                              ec=edge_color,
                              linewidth=2.0,
                              alpha=0.95,
                              zorder=6)
          ax.add_patch(circle)

          # ⭐ 电量显示（最上层，zorder=20，大而明显）⭐
          # 大背景框
          ax.text(x, y, f'{int(uav.battery)}%',
                  ha='center', va='center',
                  fontsize=14,          # 大字体
                  fontweight='extra bold',
                  color=battery_text_color,
                  bbox=dict(boxstyle='round,pad=0.4',  # 大内边距
                            facecolor=battery_bg_color,
                            alpha=1.0,                  # 完全不透明
                            edgecolor='#FFFFFF',        # 白色边框
                            linewidth=2.5),             # 粗边框
                  zorder=20)                            # 最上层

          # 电量低时添加警告图标
          if battery_level <= 0.25:
              ax.text(x, y + 0.55, '!',
                      ha='center', va='center',
                      fontsize=12,
                      zorder=21)

          # 标签（上方，小一些）
          ax.text(x, y + 0.70, f'T{i}',
                  ha='center', va='center',
                  fontsize=8, fontweight='bold',
                  color='#1A252F',
                  bbox=dict(boxstyle='round,pad=0.2',
                            facecolor='white',
                            alpha=0.9,
                            edgecolor=edge_color,
                            linewidth=1.2),
                  zorder=15)

      # ===== 6. 绘制充电无人机（美化版）=====
      for i, uav in enumerate(self.charging_uavs):
          x, y = uav.position
          battery_level = uav.battery / 100.0
          is_charging = uav.state.get("is_charging", False)

          # 充电机颜色（红色系）
          base_red = 0.85
          darken = 0.3 * (1 - battery_level)
          face_color = (base_red, darken, darken, 0.95)
          edge_color = (0.5, 0.1, 0.1, 1.0)

          # 无人机主体（六边形，zorder=7）
          hexagon = patches.RegularPolygon(
              (x, y), numVertices=6, radius=0.38,
              orientation=np.deg2rad(30),
              facecolor=face_color,
              ec=edge_color,
              linewidth=2.0,
              zorder=7
          )
          ax.add_patch(hexagon)

          # 闪电图标（充电标识）
          if is_charging:
              flash_alpha = 0.6 + 0.4 * np.sin(self.step_count * 0.6)
              ax.plot([x - 0.15, x + 0.10], [y + 0.18, y - 0.08],
                      color='#FFFF00', linewidth=3.0, alpha=flash_alpha, zorder=8)
              ax.plot([x - 0.15, x - 0.02], [y + 0.18, y + 0.06],
                      color='#FFFF00', linewidth=3.0, alpha=flash_alpha, zorder=8)

          # 电量条（底部）
          bar_width = 0.60
          bar_height = 0.12
          bar_y = y - 0.60
          ax.add_patch(plt.Rectangle(
              (x - bar_width / 2, bar_y),
              bar_width, bar_height,
              fill=True, facecolor='#2C3E50',
              ec=edge_color, linewidth=1.0, alpha=0.9, zorder=8
          ))
          ax.add_patch(plt.Rectangle(
              (x - bar_width / 2 + 0.03, bar_y + 0.02),
              (bar_width - 0.06) * battery_level,
              bar_height - 0.04,
              color='#00FF00' if battery_level > 0.3 else '#FF4444',
              ec='none', alpha=0.95, zorder=9
          ))

          # 标签（下方）
          ax.text(x, y - 0.85, f'C{i}',
                  ha='center', va='center',
                  fontsize=8, fontweight='bold',
                  color='white',
                  bbox=dict(boxstyle='round,pad=0.2',
                            facecolor=edge_color,
                            alpha=0.95,
                            edgecolor='none'),
                  zorder=15)

      # ===== 7. 图例（右上角）=====
      legend_elements = [
          Line2D([0], [0], marker='*', color='w', markerfacecolor='#FFD700',
                markersize=14, markeredgecolor='#B8860B', label='Base Station'),
          Circle((0, 0), 0.35, facecolor='#2ECC71', ec='#27AE60', linewidth=2, label='Task (High)'),
          Circle((0, 0), 0.35, facecolor='#E74C3C', ec='#C0392B', linewidth=2, label='Task (Low)'),
          RegularPolygon((0, 0), 6, radius=0.38, facecolor='#E74C3C', ec='#800000', linewidth=2, label='Charger'),
          Line2D([0], [0], color='#00CC00', linewidth=3, label='Charging Link')
      ]
      ax.legend(
          handles=legend_elements,
          loc='upper right',
          fontsize=8,
          framealpha=0.95,
          edgecolor='#CCCCCC',
          facecolor='white',
          bbox_to_anchor=(0.99, 0.99),
      )

      # ===== 8. 底部状态栏 =====
      status_text = (
          f"Grid: {self.grid_size[0]}×{self.grid_size[1]}  |  "
          f"Tasks: {self.n_task_uavs}  |  "
          f"Chargers: {self.n_charging_uavs}  |  "
          f"Dead: {self.n_task_uavs - alive_count}"
      )
      ax.text(
          self.grid_size[0] / 2, -1.0,
          status_text,
          ha='center', va='center',
          fontsize=9, color='#444444',
          bbox=dict(boxstyle='round,pad=0.35',
                    facecolor='white',
                    alpha=0.9,
                    edgecolor='#CCCCCC',
                    linewidth=1),
          zorder=100
      )

      # ===== 9. 实时更新 =====
      self._fig.canvas.draw()
      self._fig.canvas.flush_events()
      plt.pause(0.01)
    
#         ,--.                                                 ,--.     
#  ,---.  |  ,---.   ,--,--. ,--,--,--. ,--.--.  ,---.   ,---. |  |,-.  
# (  .-'  |  .-.  | ' ,-.  | |        | |  .--' | .-. | | .--' |     /  
# .-'  `) |  | |  | \ '-'  | |  |  |  | |  |    ' '-' ' \ `--. |  \  \  
# `----'  `--' `--'  `--`--' `--`--`--' `--'     `---'   `---' `--'`--'
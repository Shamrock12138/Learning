#                  Reinforcement Learning 环境函数
#                           2025/11/30
#                            shamrock

import numpy as np
from .RL_config import ENV_INFO, MDP
import gymnasium as gym
import pygame

def _pos_to_state(width, pos: tuple) -> int:
  """将 (row, col) 转为 0 ~ states_num-1 的整数状态编号"""
  r, c = pos
  return r * width + c

def _state_to_pos(width, state: int) -> tuple:
  """将状态编号转回 (row, col)"""
  return divmod(state, width)

class Env_CliffWalking(ENV_INFO):
  '''
    悬崖漫步，使用MDP的方式
      action: up right down left
  '''
  def __init__(self, height=4, width=12):
    super().__init__()
    self.matrix = MDP(states_num=height*width, actions_num=4)
    self.name = 'CliffWalking'
    self.height = height
    self.width = width
    self._states_num = height*width
    self._actions_num = 4

    self._cliff_cols = list(range(1, width-1))
    self._goal_pos = (height-1, width-1)
    self._start_pos = (height-1, 0)

    self._cliff_sta = []
    self._start_sta = _pos_to_state(width, self._start_pos)
    self._goal_sta = _pos_to_state(width, self._goal_pos)
    self._cliff_sta = [
      (height - 1) * width + c for c in range(1, width - 1)
    ]
    # self._cliff_sta.append((height-2)*width+width/2)
    # self._cliff_sta.append((height-3)*width+width/2)
    # self._cliff_sta.append((height-3)*width+width-1)
    # self._cliff_sta.append((height-3)*width+width-2)
    self.r_cliff = -100
    self.r_path = 0
    self.r_goal = 10
    self._build_matrix()

    self._pos = None

  def _build_matrix(self):
    self.matrix.P = [
      [[0.0 for _ in range(self._states_num)] for _ in range(self._actions_num)]
      for _ in range(self._states_num)
    ]
    self.matrix.R_E = [.0]*self._states_num
    self.matrix.done = [False]*self._states_num

    nS, nA = self._states_num, self._actions_num
    width, height = self.width, self.height

    # 预计算每个 state 的「上下左右」邻居（纯 state 编号）
    # 若越界或撞墙，则 stay in place
    def get_next_state(s: int, action: int) -> int:
      r, c = divmod(s, width)  # 仅在此内部转换，不暴露给外部
      if action == 0:   # ↑
        nr, nc = r - 1, c
      elif action == 1: # →
        nr, nc = r, c + 1
      elif action == 2: # ↓
        nr, nc = r + 1, c
      elif action == 3: # ←
        nr, nc = r, c - 1
      else:
        raise ValueError("Invalid action")
      # 撞墙检查：stay
      if not (0 <= nr < height and 0 <= nc < width):
        return s
      return nr * width + nc

    # Step 1: 初始化 done 和 R_E
    for s in range(nS):
      if s in self._cliff_sta:
        self.matrix.done[s] = True
        self.matrix.R_E[s] = self.r_cliff
      elif s == self._goal_sta:
        self.matrix.done[s] = True
        self.matrix.R_E[s] = self.r_goal   # Sutton & Barto 原版用 0（与每步 -1 一致）
      else:
        self.matrix.R_E[s] = self.r_path

    # Step 2: 填充 P[s][a][s_next]
    for s in range(nS):
      for a in range(nA):
        if self.matrix.done[s]:
          # 终止态：stay
          self.matrix.P[s][a][s] = 1.0
        else:
          s_next = get_next_state(s, a)
          self.matrix.P[s][a][s_next] = 1.0

    self.matrix.test()
  
  def reset(self, seed=None, options=None):
    if seed is not None:
      import random
      random.seed(seed)
    self._pos = self._start_pos
    self._done = False
    initial_state = _pos_to_state(self.width, self._pos)
    self._state = initial_state
    self._info = {'done': False}
    return initial_state, self._info.copy()
  
  def step(self, action:int):
    if self._done:
      return self._state, 0.0, True, {'done': True, 'warning': 'env already done'}
    r, c = self._pos
    if action == 0:   # 上
      r = max(0, r - 1)
    elif action == 1: # 右
      c = min(self.width - 1, c + 1)
    elif action == 2: # 下
      r = min(self.height - 1, r + 1)
    elif action == 3: # 左
      c = max(0, c - 1)
    else:
      raise ValueError(f"Invalid action: {action}")
    next_pos = (r, c)
    next_state = _pos_to_state(self.width, next_pos)
    self._done = self.matrix.done[next_state]
    reward = self.matrix.R_E[next_state]
    self._pos = next_pos
    self._info = {
      'done': self._done,
      'position': self._pos,
      'next_state': next_state
    }
    self._state = next_state
    return next_state, reward, self._done, self._info.copy()

  def render(self, pi):
    arrows = ['↑', '→', '↓', '←']
    print("=" * (self.width * 2 + 1))
    
    for r in range(self.height):
      row_str = "|"
      for c in range(self.width):
        s = r * self.width + c
        ch = '.'
        if (r, c) == self._start_pos:
          ch = '\033[34mS\033[0m'
        elif (r, c) == self._goal_pos:
          ch = '\033[32mG\033[0m'
        elif s in self._cliff_sta:
          ch = '\033[31mX\033[0m'
        else:
          if not self.matrix.done[s]:
            probs = pi[s]
            if np.allclose(probs, np.ones(4)/4):
              ch = '?'  # 随机策略
            else:
              best_a = int(np.argmax(probs))
              ch = arrows[best_a]
          else:
            ch = 'X'
        row_str += f"{ch} "
      row_str = row_str.rstrip() + "|"
      print(row_str)
    print("=" * (self.width * 2 + 1))

class Env_FrozenLake(ENV_INFO):
  '''
    冰湖环境，MDP，尺寸 4x4
    s 1 2 3 
    4 x 6 x 
    8 9 10 x 
    x 13 14 o
      action: left down right up
  '''
  def __init__(self):
    super().__init__()
    self._env = gym.make('FrozenLake-v1', is_slippery=True, render_mode=None)
    self._states_num = 16
    self._actions_num = 4
    self.name = 'FrozkenLake'

    self.matrix = MDP(states_num=16, actions_num=4)
    self._build_matrix()

  def _build_matrix(self):
    nS, nA = 16, 4
    self.matrix.P = [[[0.0]*nS for _ in range(nA)] for _ in range(nS)]
    self.matrix.R_E = [0.0]*nS
    self.matrix.done = [False]*nS

    # Hole states: 5,7,11,12; Goal: 15
    hole_states = [5, 7, 11, 12]
    goal_state = 15

    for s in range(nS):
      if s in hole_states:
        self.matrix.R_E[s] = -10.0
        self.matrix.done[s] = True
      elif s == goal_state:
        self.matrix.R_E[s] = 5.0
        self.matrix.done[s] = True

    # gym 的 env.P[s][a] = [(prob, next_state, reward, done), ...]
    for s in range(nS):
      for a in range(nA):
        if self.matrix.done[s]:
          self.matrix.P[s][a][s] = 1.0
        else:
          for prob, ns, r, _ in self._env.unwrapped.P[s][a]:
            self.matrix.P[s][a][ns] += prob
            # 注意：R_E 是进入 ns 的期望奖励，而 gym 返回的是 (s,a,ns) 的即时奖励
            # 因 FrozenLake 奖励仅依赖 ns，故 R_E[ns] 已设定，此处无需更新 R_E
    self.matrix.test()

  def reset(self, seed=None, options=None):
    self._state = 0
    self._done = False
    self._info = {'done': False}
    return self._state, self._info.copy()

  def step(self, action):
    scores, target = .0, np.random.random()
    for ns in range(self._states_num):
      scores += self.matrix.P[self._state][action][ns]
      if scores >= target:
        self._state = ns
        break
    reward = self.matrix.R_E[self._state]
    self._done = self.matrix.done[self._state]
    self._info = {'done': self._done, 'next_state': self._state}
    return self._state, reward, self._done, self._info.copy()

  def render(self):
    pass

class Env_CartPole(ENV_INFO):
  '''
    车杆环境，状态连续，动作离散；
    控制小车平衡杆子，每坚持一帧，智能体能获得分数为 1 的奖励
      state(dims): position_car speed_car angle_pole speed_pole_top
      action: left right
  '''
  def __init__(self):
    super().__init__()
    self.env = None
    self.env_train = gym.make('CartPole-v1', render_mode=None)
    self.env_eval = gym.make('CartPole-v1', render_mode='human')
    self.train()
    self._states_num = self.env.observation_space.shape[0]  # states 的维数
    self._actions_num = self.env.action_space.n             # actions
    self.matrix = None
    self.name = 'CartPole'

  def eval(self):
    self.env = self.env_eval

  def train(self):
    self.env = self.env_train

  def reset(self):
    obs, info = self.env.reset()
    self._state = obs.astype(np.float32)
    self._done = False
    self._info = {
      'done': False,
      'gym_info': info
    }
    return self._state, self._info.copy()
  
  def step(self, action):
    '''
    env 执行一步 action 的结果
      return:
        next_state, reward, done, info
    '''
    if self._done:
      return self._state, 0.0, True, {'done': True, 'warning': 'env already done'}
    obs, reward, terminated, truncated, info = self.env.step(action)
    # CartPole 中，terminated（失败）或 truncated（超时）均视为 episode 结束
    self._done = terminated or truncated
    self._state = obs.astype(np.float32)
    self._info = {
      'done': self._done,
      'terminated': terminated,
      'truncated': truncated,
      'gym_info': info,
      'next_state': self._state
    }
    return self._state, float(reward), self._done, self._info.copy()

  def render(self):
    '''
      渲染一帧的动画
    '''
    return self.env.render()

class Env_Pendulum(ENV_INFO):
  '''
    倒立摆环境，状态连续，动作连续；
    控制摆杆使其竖直向上，角度越接近0，奖励越高（最大为0，越负表示表现越差）
      state(dims): cos_theta sin_theta angular_velocity
      action: torque (continuous value in [-2.0, 2.0])
  '''
  def __init__(self):
    super().__init__()
    self.env = None
    self.env_train = gym.make('Pendulum-v1', render_mode=None)
    self.env_eval = gym.make('Pendulum-v1', render_mode='human')
    self.train()
    self._states_num = self.env.observation_space.shape[0]
    self._actions_num = self.env.action_space.shape[0]
    # self._actions_num = 12
    self.name = 'Pendulum'

    # self.c = 4.0/(self._actions_num-1)

  def train(self):
    self.env = self.env_train

  def eval(self):
    self.env = self.env_eval

  def reset(self):
    obs, info = self.env.reset()
    self._state = obs.astype(np.float32)
    self._done = False
    self._info = {
      'done': False,
      'gym_info': info
    }
    return self._state, self._info.copy()

  def step(self, action):
    if self._done:
      return self._state, 0.0, True, {'done': True, 'warning': 'env already done'}
    
    # Pendulum环境动作需要是数组形式
    # action = -2.0+self.c*action
    # print(action)
    if isinstance(action, (int, float)):
      action = np.array([action], dtype=np.float32)
    elif isinstance(action, list):
      action = np.array(action, dtype=np.float32)
    action = np.clip(action, -2.0, 2.0)
    
    obs, reward, terminated, truncated, info = self.env.step(action)
    # Pendulum环境中，通常不会自然终止，但可能有最大步数限制
    self._done = terminated or truncated
    self._state = obs.astype(np.float32)
    self._info = {
      'done': self._done,
      'terminated': terminated,
      'truncated': truncated,
      'gym_info': info,
      'next_state': self._state,
      'raw_reward': float(reward)  # 保存原始奖励，Pendulum的奖励范围是[-16.27, 0]
    }
    return self._state, float(reward), self._done, self._info.copy()

  def render(self):
    return self.env.render()

class Env_AimBall(ENV_INFO):
  '''
  类似 CS 的“小球瞄准训练”环境：
    - 屏幕中随机出现圆形目标
    - 智能体控制准星（十字）移动
    - RL 控制 ↑→↓← 四方向移动准星
    - 目标：快速将准星移至目标中心
  使用示例：
    env = AimTrainerEnv(seed=42)
    state, info = env.reset()
    print("Initial state:", state)

    for _ in range(50):
      action = np.random.randint(4)  # 随机策略
      next_state, reward, info = env.step(action)
      print(f"Action: {action}, Reward: {reward:.3f}, Hit: {info['hit']}")
      env.render()
      if info['done']:
        break
  '''
  def __init__(
    self,
    screen_width: int = 800,
    screen_height: int = 600,
    cursor_speed: float = 10.0,        # 像素/步
    target_radius: float = 15.0,       # 像素
    max_steps: int = 100,
  ):
    super().__init__()
    self.width = screen_width
    self.height = screen_height
    self.cursor_speed = cursor_speed
    self.target_radius = target_radius
    self.max_steps = max_steps

    self._states_num = 4
    self._actions_num = 4    # 0:↑, 1:→, 2:↓, 3:←
    self.name = 'AimBall'

  def _get_state(self):
    # 归一化到 [0,1]
    return np.array([
      self.cursor_x / self.width,
      self.cursor_y / self.height,
      self.target_x / self.width,
      self.target_y / self.height
    ], dtype=np.float32)

  def _reset_target(self):
    margin = self.target_radius + 50
    self.target_x = np.random.uniform(margin, self.width - margin)
    self.target_y = np.random.uniform(margin, self.height - margin)

  def train(self):
    return super().train()
  
  def eval(self):
    return super().eval()

  def reset(self):
    # 初始化准星居中
    self.cursor_x = self.width / 2.0
    self.cursor_y = self.height / 2.0
    # 随机生成目标（避开边缘）
    self._reset_target()
    self.steps = 0
    self._done = False
    self._state = self._get_state()
    self._info = {
      'done': False,
      'steps': self.steps,
      'hit': False
    }
    return self._state, self._info.copy()

  def step(self, action: int):
    if self._done:
      return self._state, 0.0, {'done': True}

    # 移动准星
    dx, dy = 0.0, 0.0
    if action == 0:   # ↑
      dy = -self.cursor_speed
    elif action == 1: # →
      dx = self.cursor_speed
    elif action == 2: # ↓
      dy = self.cursor_speed
    elif action == 3: # ←
      dx = -self.cursor_speed
    else:
      raise ValueError(f"Invalid action: {action}")

    self.cursor_x = np.clip(self.cursor_x + dx, 0, self.width)
    self.cursor_y = np.clip(self.cursor_y + dy, 0, self.height)

    dist = np.sqrt(
      (self.cursor_x - self.target_x)**2 +
      (self.cursor_y - self.target_y)**2
    )
    hit = dist <= self.target_radius

    norm_dist = dist / np.sqrt(self.width**2 + self.height**2)
    reward = -norm_dist*10
    if 'distance' in self._info:
      prev_dist = self._info['distance']
      delta = prev_dist - dist
      reward += 0.1 * delta    # 每像素 +0.1 → 靠近10px = +1.0
    if hit:
      reward += 10.0
    reward -= 0.01  # 每步微小时间惩罚，防拖延

    self.steps += 1
    timeout = self.steps >= self.max_steps

    self._done = hit or timeout
    self._state = self._get_state()
    self._info = {
      'done': self._done,
      'steps': self.steps,
      'hit': hit,
      'distance': dist
    }

    return self._state, float(reward), self._done, self._info.copy()

  def render(self):
    if not hasattr(self, '_screen') or self._screen is None:
      pygame.init()
      pygame.display.set_caption("AimBall Dynamic")
      self._screen = pygame.display.set_mode((self.width, self.height))
      self._clock = pygame.time.Clock()

    # 背景
    self._screen.fill((30, 30, 30))

    # 目标球（加阴影提升可视性）
    pygame.draw.circle(self._screen, (200, 50, 50), (int(self.target_x), int(self.target_y)), int(self.target_radius))
    pygame.draw.circle(self._screen, (255, 100, 100), (int(self.target_x), int(self.target_y)), int(self.target_radius)-3)

    # 准星（更清晰的十字+圆环）
    cx, cy = int(self.cursor_x), int(self.cursor_y)
    pygame.draw.line(self._screen, (0, 255, 100), (cx-12, cy), (cx+12, cy), 2)
    pygame.draw.line(self._screen, (0, 255, 100), (cx, cy-12), (cx, cy+12), 2)
    pygame.draw.circle(self._screen, (0, 200, 100, 100), (cx, cy), 8, 1)

    # 实时信息
    font = pygame.font.SysFont(None, 20)
    info_text = f"Steps: {self.steps} | Dist: {self._info.get('distance', 0):.1f}"
    if self._info.get('hit', False):
        info_text += " | HIT!"
    text = font.render(info_text, True, (255, 255, 255))
    self._screen.blit(text, (10, 10))

    pygame.display.flip()
    self._clock.tick(60)  # 更高帧率更流畅

class Env_AimBallDynamic(ENV_INFO):
  '''
  动态小球瞄准训练环境：
    - 小球以一定速度随机移动（或匀速、弹跳）
    - 智能体控制准星移动
    - 鼓励快速命中 + 持续跟踪
  '''
  def __init__(
      self,
      screen_width: int = 800,
      screen_height: int = 600,
      cursor_speed: float = 10.0,
      target_radius: float = 15.0,
      max_steps: int = 100,
      target_speed: float = 2.0,              # 小球移动速度（像素/步）
      target_move_mode: str = 'random_walk',  # 'uniform', 'random_walk', 'teleport'
      seed: int = None,
  ):
      super().__init__()
      self.width = screen_width
      self.height = screen_height
      self.cursor_speed = cursor_speed
      self.target_radius = target_radius
      self.max_steps = max_steps
      self.target_speed = target_speed
      self.target_move_mode = target_move_mode

      if seed is not None:
          np.random.seed(seed)

      self._states_num = 6  # [cx, cy, tx, ty, tvx, tvy] 或 [cx, cy, tx, ty, vx, vy]
      self._actions_num = 4
      self.name = 'AimBall-Dynamic'

  def _get_state(self) -> np.ndarray:
      # 归一化：位置 ∈ [0,1]；速度 ∈ [-1,1] → 归一化到 [-target_speed_norm, +target_speed_norm]
      speed_norm = max(self.width, self.height) / 2.0
      return np.array([
          self.cursor_x / self.width,
          self.cursor_y / self.height,
          self.target_x / self.width,
          self.target_y / self.height,
          self.target_vx / speed_norm,
          self.target_vy / speed_norm
      ], dtype=np.float32)

  def _reset_target(self):
      margin = self.target_radius + 50
      self.target_x = np.random.uniform(margin, self.width - margin)
      self.target_y = np.random.uniform(margin, self.height - margin)
      # 初始化速度
      angle = np.random.uniform(0, 2 * np.pi)
      self.target_vx = self.target_speed * np.cos(angle)
      self.target_vy = self.target_speed * np.sin(angle)

  def reset(self):
      self.cursor_x = self.width / 2.0
      self.cursor_y = self.height / 2.0
      self._reset_target()
      self.steps = 0
      self._done = False
      self._state = self._get_state()
      self._info = {
          'done': False,
          'steps': self.steps,
          'hit': False,
          'distance': np.linalg.norm([self.cursor_x - self.target_x, self.cursor_y - self.target_y])
      }
      return self._state, self._info.copy()

  def _update_target(self):
      if self.target_move_mode == 'uniform':
          # 匀速直线运动 + 边界反弹
          self.target_x += self.target_vx
          self.target_y += self.target_vy

          # 边界反弹（弹性反射）
          if self.target_x <= self.target_radius or self.target_x >= self.width - self.target_radius:
              self.target_vx *= -1
              self.target_x = np.clip(self.target_x, self.target_radius, self.width - self.target_radius)
          if self.target_y <= self.target_radius or self.target_y >= self.height - self.target_radius:
              self.target_vy *= -1
              self.target_y = np.clip(self.target_y, self.target_radius, self.height - self.target_radius)

      elif self.target_move_mode == 'random_walk':
          # 随机扰动方向（每步小角度偏转）
          angle_noise = np.random.normal(0, np.pi / 8)  # ~22.5° 标准差
          speed = np.sqrt(self.target_vx**2 + self.target_vy**2)
          current_angle = np.arctan2(self.target_vy, self.target_vx)
          new_angle = current_angle + angle_noise
          self.target_vx = speed * np.cos(new_angle)
          self.target_vy = speed * np.sin(new_angle)

          self.target_x += self.target_vx
          self.target_y += self.target_vy

          # 碰壁反弹（同上）
          if self.target_x <= self.target_radius or self.target_x >= self.width - self.target_radius:
              self.target_vx *= -1
              self.target_x = np.clip(self.target_x, self.target_radius, self.width - self.target_radius)
          if self.target_y <= self.target_radius or self.target_y >= self.height - self.target_radius:
              self.target_vy *= -1
              self.target_y = np.clip(self.target_y, self.target_radius, self.height - self.target_radius)

      elif self.target_move_mode == 'teleport':
          # 每隔若干步（如每 30 步）随机跳转
          if self.steps % 30 == 0 and self.steps > 0:
              margin = self.target_radius + 50
              self.target_x = np.random.uniform(margin, self.width - margin)
              self.target_y = np.random.uniform(margin, self.height - margin)
              # 重置速度方向
              angle = np.random.uniform(0, 2 * np.pi)
              self.target_vx = self.target_speed * np.cos(angle)
              self.target_vy = self.target_speed * np.sin(angle)
          else:
              self.target_x += self.target_vx
              self.target_y += self.target_vy
              # 边界限制（不反弹，仅 clamp）
              self.target_x = np.clip(self.target_x, self.target_radius, self.width - self.target_radius)
              self.target_y = np.clip(self.target_y, self.target_radius, self.height - self.target_radius)

  def step(self, action: int):
      if self._done:
          return self._state, 0.0, True, self._info.copy()

      # 移动准星
      dx, dy = 0.0, 0.0
      if action == 0:   # ↑
          dy = -self.cursor_speed
      elif action == 1: # →
          dx = self.cursor_speed
      elif action == 2: # ↓
          dy = self.cursor_speed
      elif action == 3: # ←
          dx = -self.cursor_speed
      else:
          raise ValueError(f"Invalid action: {action}")

      self.cursor_x = np.clip(self.cursor_x + dx, 0, self.width)
      self.cursor_y = np.clip(self.cursor_y + dy, 0, self.height)

      # 更新目标位置（关键改动！）
      self._update_target()

      # 计算距离
      dist = np.sqrt(
          (self.cursor_x - self.target_x)**2 +
          (self.cursor_y - self.target_y)**2
      )
      hit = dist <= self.target_radius

      norm_dist = dist / np.sqrt(self.width**2 + self.height**2)
      reward = -norm_dist*10
      if 'distance' in self._info:
        prev_dist = self._info['distance']
        delta = prev_dist - dist
        reward += 0.1 * delta    # 每像素 +0.1 → 靠近10px = +1.0
      if hit:
        reward += 10.0
      reward -= 0.01  # 每步微小时间惩罚，防拖延

      self.steps += 1
      timeout = self.steps >= self.max_steps
      self._done = timeout or hit

      self._state = self._get_state()
      self._info = {
          'done': self._done,
          'steps': self.steps,
          'hit': hit,
          'distance': dist,
          'target_pos': (self.target_x, self.target_y),
          'cursor_pos': (self.cursor_x, self.cursor_y)
      }

      return self._state, float(reward), self._done, self._info.copy()

  def render(self):
      if not hasattr(self, '_screen') or self._screen is None:
          pygame.init()
          pygame.display.set_caption("AimBall Dynamic")
          self._screen = pygame.display.set_mode((self.width, self.height))
          self._clock = pygame.time.Clock()

      # 背景
      self._screen.fill((30, 30, 30))

      # 目标球（加阴影提升可视性）
      pygame.draw.circle(self._screen, (200, 50, 50), (int(self.target_x), int(self.target_y)), int(self.target_radius))
      pygame.draw.circle(self._screen, (255, 100, 100), (int(self.target_x), int(self.target_y)), int(self.target_radius)-3)

      # 准星（更清晰的十字+圆环）
      cx, cy = int(self.cursor_x), int(self.cursor_y)
      pygame.draw.line(self._screen, (0, 255, 100), (cx-12, cy), (cx+12, cy), 2)
      pygame.draw.line(self._screen, (0, 255, 100), (cx, cy-12), (cx, cy+12), 2)
      pygame.draw.circle(self._screen, (0, 200, 100, 100), (cx, cy), 8, 1)

      # 实时信息
      font = pygame.font.SysFont(None, 20)
      info_text = f"Steps: {self.steps} | Dist: {self._info.get('distance', 0):.1f}"
      if self._info.get('hit', False):
          info_text += " | HIT!"
      text = font.render(info_text, True, (255, 255, 255))
      self._screen.blit(text, (10, 10))

      pygame.display.flip()
      self._clock.tick(60)  # 更高帧率更流畅

  def close(self):
      if hasattr(self, '_screen') and self._screen:
          pygame.quit()
          self._screen = None

  def train(self):
    return super().train()
  
  def eval(self):
    return super().eval()

#         ,--.                                                 ,--.     
#  ,---.  |  ,---.   ,--,--. ,--,--,--. ,--.--.  ,---.   ,---. |  |,-.  
# (  .-'  |  .-.  | ' ,-.  | |        | |  .--' | .-. | | .--' |     /  
# .-'  `) |  | |  | \ '-'  | |  |  |  | |  |    ' '-' ' \ `--. |  \  \  
# `----'  `--' `--'  `--`--' `--`--`--' `--'     `---'   `---' `--'`--' 

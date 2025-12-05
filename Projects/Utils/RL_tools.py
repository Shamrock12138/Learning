#                  Reinforcement Learning 工具函数
#                           2025/11/30
#                            shamrock
import numpy as np

#---------------------- 环境 -------------------------
#                      2025/11/30

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
        self.matrix.R_E[s] = -100.0
      elif s == self._goal_sta:
        self.matrix.done[s] = True
        self.matrix.R_E[s] = 0.0   # Sutton & Barto 原版用 0（与每步 -1 一致）
      else:
        self.matrix.R_E[s] = -1.0

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

  def render(self):
    pass

class Env_FrozenLake(ENV_INFO):
  '''
    冰湖环境，MDP，尺寸 4x4
      s 1 2 3 
      4 x 6 x 
      8 9 10 x 
      x 13 14 o
  '''
  def __init__(self):
    super().__init__()
    # 封装官方环境
    self._env = gym.make('FrozenLake-v1', is_slippery=True, render_mode=None)
    self._states_num = 16
    self._actions_num = 4   # left down right up

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
    cursor_speed: float = 20.0,        # 像素/步
    target_radius: float = 15.0,       # 像素
    max_steps: int = 200,
    # seed: int = None
  ):
    super().__init__()
    self.width = screen_width
    self.height = screen_height
    self.cursor_speed = cursor_speed
    self.target_radius = target_radius
    self.max_steps = max_steps

    # 状态/动作维度
    self._states_num = None  # 连续状态，不预设总数
    self._actions_num = 4    # 0:↑, 1:→, 2:↓, 3:←
    self.matrix = None       # 如需 MDP，需离散化后构建（见 _build_matrix）

    # if seed is not None:
    #   random.seed(seed)
    #   np.random.seed(seed)

    self.reset()

  def _get_state(self):
    # 归一化到 [0,1]
    return np.array([
      self.cursor_x / self.width,
      self.cursor_y / self.height,
      self.target_x / self.width,
      self.target_y / self.height
    ], dtype=np.float32)

  def reset(self):
    # 初始化准星居中
    self.cursor_x = self.width / 2.0
    self.cursor_y = self.height / 2.0
    # 随机生成目标（避开边缘）
    margin = self.target_radius + 50
    self.target_x = np.random.uniform(margin, self.width - margin)
    self.target_y = np.random.uniform(margin, self.height - margin)
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

    # 计算距离
    dist = np.sqrt(
      (self.cursor_x - self.target_x)**2 +
      (self.cursor_y - self.target_y)**2
    )

    # 奖励设计（推荐组合）
    reward = -dist / np.sqrt(self.width**2 + self.height**2)  # 归一化负距离
    hit = dist <= self.target_radius
    if hit:
      reward += 1.0
      # 可选：立即重置目标 or 等待 episode 结束
      # self._reset_target()

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

    return self._state, float(reward), self._info.copy()

  def _reset_target(self):
    margin = self.target_radius + 50
    self.target_x = np.random.uniform(margin, self.width - margin)
    self.target_y = np.random.uniform(margin, self.height - margin)

  def render(self, mode='human'):
    if not hasattr(self, '_screen'):
      pygame.init()
      self._screen = pygame.display.set_mode((self.width, self.height))
      self._clock = pygame.time.Clock()
    
    self._screen.fill((0, 0, 0))
    # 画目标
    pygame.draw.circle(self._screen, (255, 0, 0), (int(self.target_x), int(self.target_y)), int(self.target_radius))
    # 画准星（十字）
    cx, cy = int(self.cursor_x), int(self.cursor_y)
    pygame.draw.line(self._screen, (0, 255, 0), (cx-10, cy), (cx+10, cy), 2)
    pygame.draw.line(self._screen, (0, 255, 0), (cx, cy-10), (cx, cy+10), 2)
    pygame.display.flip()
    self._clock.tick(30)

  def _build_matrix(self, grid_size: int = 10):
    '''
    将连续状态离散化为 grid_size x grid_size x grid_size x grid_size
    仅用于小规模验证（如 5x5），实际 RL 推荐直接用连续状态 + DQN/PPO
    '''
    n = grid_size
    nS = n**4
    nA = 4
    self._states_num = nS
    self.matrix = MDP(nS, nA)

    # 初始化
    self.matrix.P = [[[0.0]*nS for _ in range(nA)] for _ in range(nS)]
    self.matrix.R_E = [0.0]*nS
    self.matrix.done = [False]*nS

    # 离散化映射：s = ix + iy*n + itx*n^2 + ity*n^3
    def encode(ix, iy, itx, ity):
      return ix + iy*n + itx*n*n + ity*n*n*n

    for ix in range(n):
      for iy in range(n):
        for itx in range(n):
          for ity in range(n):
            s = encode(ix, iy, itx, ity)
            cx = (ix + 0.5) / n * self.width
            cy = (iy + 0.5) / n * self.height
            tx = (itx + 0.5) / n * self.width
            ty = (ity + 0.5) / n * self.height
            dist = np.sqrt((cx-tx)**2 + (cy-ty)**2)
            hit = dist <= self.target_radius

            if hit:
              self.matrix.done[s] = True
              self.matrix.R_E[s] = 1.0
            else:
              self.matrix.R_E[s] = -dist / np.sqrt(self.width**2 + self.height**2)

            for a in range(nA):
              # 模拟移动
              dx, dy = [0, self.cursor_speed, 0, -self.cursor_speed][a], \
                        [-self.cursor_speed, 0, self.cursor_speed, 0][a]
              ncx = np.clip(cx + dx, 0, self.width)
              ncy = np.clip(cy + dy, 0, self.height)
              # 目标假设不变（简化）
              nix = min(n-1, int(ncx / self.width * n))
              niy = min(n-1, int(ncy / self.height * n))
              ns = encode(nix, niy, itx, ity)
              self.matrix.P[s][a][ns] = 1.0
    self.matrix.test()

# env = AimTrainerEnv(seed=42)
# state, info = env.reset()
# print("Initial state:", state)

# for _ in range(50):
#     action = np.random.randint(4)  # 随机策略
#     next_state, reward, info = env.step(action)
#     print(f"Action: {action}, Reward: {reward:.3f}, Hit: {info['hit']}")
#     env.render()
#     if info['done']:
#         break

#---------------------- 采取动作的策略 -------------------------
#                         2025/12/4

def RTools_epsilon(epsilon, n_actions:int, argmax_action:int):
  '''
    采用 ε-greedy 的方式，epsilon越大，越随机，反之，越容易选择argmax_action。
      params:
        epsilon: float - 
        n_actions - 动作个数
        argmax_action: int - 较优action
  '''
  if np.random.random() < epsilon:
    action = np.random.randint(n_actions)
  else:
    action = argmax_action
  return action

#---------------------- 探索方式 -------------------------
#                      2025/11/29

def RTools_Sample(MDP, INFO, pi, timestep_max, number):
  '''
    对INFO环境下的MDP进行采样，返回采样得到的所有“路径”[s, a, r, s_next]
      params:
        MDP - 描述环境马尔可夫决策过程 提供(S, A, P, R, gamma)
          S: list [states_num] 所有状态
          A: list [actions_num] 所有动作  
          P: list [states_num, actions_num, states_num] s采取了a动作后的next_s的所有概率
          R: list [states_num, actions_num] s采取了a动作后的奖励
        INFO - 提供环境信息 (end: list, states_num: int, actions_num: int)
        pi: dict - 策略
        timestep_max - 单条轨迹的最大时间步数
        number - 要生成的 episode 总数
      return:
        episodes - number 次 episode 的探索列表 [[s, a, r, s_next], ...]
  '''
  S, A, P, R, gamma = MDP
  end, states_num, actions_num = INFO
  episodes = []
  for _ in range(number):
    episode, timestep = [], 0
    s = S[np.random.randint(states_num)]
    while s not in end and timestep <= timestep_max:
      timestep += 1
      rand, temp = np.random.rand(), 0
      for a_opt in A:
        # temp += pi.get(s+' '+a_opt, 0)
        temp += pi[s][a_opt]
        if temp > rand:
          a = a_opt
          # r = R.get(s+' '+a, 0)
          r = R[s][a]
          break
      rand, temp = np.random.rand(), 0
      for s_opt in S:
        # temp += P.get(join(join(s, a), s_opt), 0)
        temp += P[s][a][s_opt]
        if temp > rand:
          s_next = s_opt
          break
      episode.append((s, a, r, s_next)) 
      s = s_next
    episodes.append(episode)
  return episodes

#---------------------- 评价指标 -------------------------
#                      2025/11/29

def RTools_OccupancyMeasure(episodes, s, a, timestep_max, gamma):
  '''
    计算 (s, a) 的占用度量，即：(s, a) 在环境交互过程的频率
      params:
        episodes - 轨迹列表，格式[(s₀, a₀, r₀, s₁), (s₁, a₁, r₁, s₂), ...]
        s, a - 要统计的状态和动作
        timestep_max - 最大步长
        gamma - 折扣因子
      return:
        rho - (s, a) 在环境交互过程的频率
  '''
  rho = 0
  total_times = np.zeros(timestep_max)  # total_times[i] - 总步长>=i的次数
  occur_times = np.zeros(timestep_max)  # occur_times[i] - 第i步出现(s, a)的次数
  for episode in episodes:
    for i in range(len(episode)):
      (s_opt, a_opt, r, s_next) = episode[i]
      total_times[i] += 1
      if s == s_opt and a == a_opt:
        occur_times[i] += 1
  for i in reversed(range(timestep_max)):
    if total_times[i]:
      rho += gamma**i*occur_times[i]/total_times[i]
  return (1-gamma)*rho

#---------------------- 计算状态价值（V） -------------------------
#                         2025/11/30

def RTools_MonteCorlo(episodes, gamma, first_visit=True):
  '''
    用 Monte Corlo 的方式，计算 episodes 中每个 state 的 V
      params:
        episodes - 探索列表 [[s, a, r, s_next], ...]
        gamma - 折扣因子
        first_visit - 是否使用首次访问 MC（未实现）
      return:
        V: dir - {s1: v1, ...} S 状态的 V
  '''
  V, N = {}, {}
  for episode in episodes:
    G = 0.0
    for i in range(len(episode)-1, -1, -1):
      (s, a, r, next_s) = episode[i]
      G = r+gamma*G
      if s not in V:
        V[s] = 0.0
        N[s] = 0
      N[s] += 1
      V[s] += (G-V[s])/N[s]
  return V

if __name__ == '__main__':
  env = Env_CliffWalking()
  # print(env.matrix.P)
  # state, info = env.reset(seed=42)
  # print("Initial state:", state, "info:", info)

  # for step in range(15):
  #   action = 0  # 一直向右走（会掉崖）
  #   next_state, reward, info = env.step(action)
  #   print(f"Step {step}: action={action} → state={next_state}, reward={reward}, done={info['done']}")
  #   if info["done"]:
  #     print("Episode ended.")
  #     break

#         ,--.                                                 ,--.     
#  ,---.  |  ,---.   ,--,--. ,--,--,--. ,--.--.  ,---.   ,---. |  |,-.  
# (  .-'  |  .-.  | ' ,-.  | |        | |  .--' | .-. | | .--' |     /  
# .-'  `) |  | |  | \ '-'  | |  |  |  | |  |    ' '-' ' \ `--. |  \  \  
# `----'  `--' `--'  `--`--' `--`--`--' `--'     `---'   `---' `--'`--' 

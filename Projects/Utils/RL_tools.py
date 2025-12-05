#                  Reinforcement Learning 工具函数
#                           2025/11/30
#                            shamrock
import numpy as np

#---------------------- 环境 -------------------------
#                      2025/11/30
# Model Free:
# Model Base: Env_CliffWalking Env_FrozenLake

from .RL_config import ENV_INFO, MDP
import gymnasium as gym

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

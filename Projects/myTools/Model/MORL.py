#                         多目标强化学习模型
#                           2026/1/22
#                            shamrock

from myTools.Utils.MORL_config import *
from myTools.Utils.MORL_tools import *
from myTools.Utils.tools import *
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque
import copy

#---------------------- Envelope Q Learning -------------------------
#                        2026/1/22

class EQL(MORL_ModelConfig):
  def __init__(self, env:MORL_EnvConfig, model:EQL_Network, buffer:utils_prioritReplayBuffer, 
               agent_params:dict, device):
    super().__init__()
    utils_autoAssign(self)
    utils_setAttr(self, agent_params)

    self.model = model               # 当前网络
    self.target_model = copy.deepcopy(model) # 目标网络
    self.epsilon_delta = (self.epsilon-0.05)/self.episode_num

    self.beta_init = self.beta
    self.beta_uplim = 1.00
    self.tau = 1000.
    self.beta_expbase = float(np.power(self.tau*(self.beta_uplim-self.beta), 1./self.episode_num))
    self.beta_delta = self.beta_expbase / self.tau

    self.trans = buffer.Transition

    if self.optimizer == 'Adam':
      self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    self.w_kept = None
    self.update_count = 0

  def take_action(self, state:np.ndarray, preference:Optional[torch.Tensor]=None):
    if preference is None:
      if self.w_kept is None:
        self.w_kept = torch.randn(self.model.reward_size)
        self.w_kept = torch.abs(self.w_kept)/torch.norm(self.w_kept, p=1)
      preference = self.w_kept

    state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
    preference_tensor = preference.unsqueeze(0)
    with torch.no_grad():
      _, q_values = self.model(state_tensor, preference_tensor)
    q_values = q_values.squeeze(0)
    scalarized_q = torch.matmul(q_values, preference)
    greedy_action = scalarized_q.argmax().item()
    if self.is_train == 1:
      # 检查是否强制探索（经验池未满或随机探索）
      force_explore = (
        self.buffer.size() < self.batch_size or 
        torch.rand(1, device=self.device).item() < self.epsilon
      )
      if force_explore:
        # 随机选择动作
        return np.random.randint(self.model.action_size)
    return greedy_action
  
  def memorize(self, state, action, next_state, reward, done):
    state = torch.from_numpy(state).float().to(self.device)
    next_state = torch.from_numpy(next_state).float().to(self.device)
    reward = torch.from_numpy(reward).float().to(self.device)
    # action = torch.tensor(action, dtype=torch.long)

    with torch.no_grad():
      preference = torch.randn(self.model.reward_size)
      preference = torch.abs(preference)/torch.norm(preference, p=1)

      _, q = self.model(
        state.unsqueeze(0),
        preference.unsqueeze(0)
      )
      current_q = q[0, action]

      scalar_q = torch.dot(preference, current_q)   # q = wTQ
      scalar_reward = torch.dot(preference, reward) # r = wTR

      if done != 0:
        next_q, _ = self.model(
          next_state.unsqueeze(0),
          preference.unsqueeze(0)
        )
        next_q = next_q[0]
        scalar_next_q = torch.dot(preference, next_q)
        td_error = scalar_reward+self.gamma*scalar_next_q-scalar_q
      else:
        td_error = scalar_reward-scalar_q
        self.w_kept = None
        # 探索率衰减
        if self.epsilon_decay:
          self.epsilon -= self.epsilon_delta
        # 同伦优化
        # if self.homotopy:
        self.beta += self.beta_delta    # 同伦优化，实现beta的指数增长
        self.beta_delta = (self.beta-self.beta_init)*self.beta_expbase+self.beta_init-self.beta
      priority = torch.abs(td_error)+1e-5
    self.buffer.add(priority=priority, state=state, action=action, next_state=next_state, 
                    reward=reward, done=done)

  def sample(self, batch_size):
    return self.buffer.sample(batch_size)
  
  def actmask(self, num_dim, index):
    '''
    生成one_hot编码，生成一个长度为num_dim的二进制掩码，只有index位置为1，其余为0
    例如：actmsk(5, 2) 会生成 [[0, 0, 1, 0, 0]]
    '''
    mask = torch.zeros(num_dim, dtype=torch.bool, device=self.device)
    mask[index] = True
    return mask.unsqueeze(0)

  def nontmlinds(self, terminal_batch):
    '''
      获取"非终止状态的索引"（non-terminal indices）
    '''
    terminal_tensor = torch.tensor(terminal_batch, dtype=torch.bool, device=self.device)
    non_terminal_mask = ~terminal_tensor
    return torch.nonzero(non_terminal_mask, as_tuple=True)[0]

  def update(self):
    action_size = self.model.action_size
    reward_size = self.model.reward_size

    self.update_count += 1
    minibatch, _ = self.sample(self.batch_size)

    batchify = lambda x: list(x)*self.weight_num
    state_batch = batchify(map(lambda x: x.state.unsqueeze(0), minibatch))
    action_batch = batchify(map(lambda x: torch.tensor([x.action], dtype=torch.long, device=self.device), minibatch))
    reward_batch = batchify(map(lambda x: x.reward.unsqueeze(0), minibatch))  # 关键：保持奖励维度
    next_state_batch = batchify(map(lambda x: x.next_state.unsqueeze(0), minibatch))
    terminal_batch = batchify(map(lambda x: torch.tensor([x.done], dtype=torch.bool, device=self.device), minibatch))  # 添加 unsqueeze 保持维度

    w_batch = np.random.randn(self.weight_num, reward_size)
    w_batch = np.abs(w_batch)/np.linalg.norm(w_batch, ord=1, axis=1, keepdims=True)
    w_batch = torch.from_numpy(w_batch.repeat(self.batch_size, axis=0)).float().to(self.device)
    # [batch_size * weight, reward_size]

    state_cat = torch.cat(state_batch, dim=0)
    next_state_cat = torch.cat(next_state_batch, dim=0)
    # [batch_size * weight, state_size]

    _, Q = self.model(state_cat, w_batch, w_num=self.weight_num)
    with torch.no_grad():
      _, target_Q = self.target_model(next_state_cat, w_batch)

      # Double DQN
      _, online_next_Q = self.model(next_state_cat, w_batch)
      w_ext = w_batch.unsqueeze(1).expand(-1, action_size, -1).reshape(-1, reward_size)
      online_next_Q_flat = online_next_Q.reshape(-1, reward_size)
      scalarized_Q = torch.bmm(
        w_ext.unsqueeze(1),
        online_next_Q_flat.unsqueeze(2)
      ).squeeze().view(-1, action_size)
      # [batch_size * weight_num, action_size]
      best_actions = scalarized_Q.argmax(dim=1)

      HQ = target_Q.gather(
        1, 
        best_actions.view(-1, 1, 1).expand(target_Q.size(0), 1, target_Q.size(2))
      ).squeeze()
      # [batch_size * weight_num, reward_size]
      nontml_mask = self.nontmlinds(terminal_batch)
      target_values = torch.zeros(self.batch_size*self.weight_num, reward_size,
                                  device=self.device, dtype=torch.float32)
      target_values[nontml_mask] = self.gamma*HQ[nontml_mask]
      # print(target_values.shape, torch.cat(reward_batch, dim=0).shape)
      target_values += torch.cat(reward_batch, dim=0)
      # [batch_size * weight_num, reward_size]

    actions = torch.cat(action_batch, dim=0)
    current_Q = Q.gather(
      1,
      actions.view(-1, 1, 1).expand(Q.size(0), 1, Q.size(2))
    ).view(-1, reward_size)
    wQ = torch.bmm(w_batch.unsqueeze(1), current_Q.unsqueeze(2)).squeeze()
    wTQ = torch.bmm(w_batch.unsqueeze(1), target_values.unsqueeze(2)).squeeze()

    loss = self.beta*F.mse_loss(wQ, wTQ)+(1-self.beta)*F.mse_loss(current_Q, target_values)

    self.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # More efficient gradient clipping
    self.optimizer.step()

    # if self.update_count % self.update_freq == 0:
    #   self.load_state_dict(self.model.state_dict())

    return loss.item()

  def reset(self):
    self.w_kept = None
    if self.epsilon_decay:
      self.epsilon -= self.epsilon_delta
    if self.homotopy:
      self.beta += self.beta_delta
      self.beta_delta = (self.beta-self.beta_init)*self.beta_expbase+self.beta_init-self.beta

  def predict(self, probe):
    with torch.no_grad():
      # Create a dummy state [0, 0] for prediction
      dummy_state = torch.zeros(1, self.model.state_size, device=self.device)
      probe_tensor = probe.unsqueeze(0).to(self.device)
      return self.model(dummy_state, probe_tensor)
    
  def find_preference(self, w_batch: np.ndarray, target_batch: np.ndarray, 
                      pref_param: np.ndarray) -> np.ndarray:
    """
    Find the optimal preference parameter using gradient ascent.
    
    Args:
      w_batch: Sampled preference vectors
      target_batch: Target values for each preference
      pref_param: Initial preference parameter guess
    
    Returns:
      Updated preference parameter after optimization step
    """
    # Convert to tensors
    w_batch_tensor = torch.tensor(w_batch, dtype=torch.float32, device=self.device)
    target_batch_tensor = torch.tensor(target_batch, dtype=torch.float32, device=self.device)
    pref_param_tensor = torch.tensor(pref_param, dtype=torch.float32, device=self.device, requires_grad=True)
    
    # Create normal distribution with small variance
    sigmas = torch.full_like(pref_param_tensor, 0.001)
    dist = torch.distributions.normal.Normal(pref_param_tensor, sigmas)
    
    # Compute log probability and weight by targets
    pref_loss = dist.log_prob(w_batch_tensor).sum(dim=1) * target_batch_tensor
    
    # Optimize preference parameters
    loss = pref_loss.mean()
    loss.backward()
    
    # Gradient ascent step
    eta = 1e-3
    with torch.no_grad():
        updated_pref = pref_param_tensor + eta * pref_param_tensor.grad
    
    # Project back to simplex constraint
    return self.simplex_proj(updated_pref.cpu().numpy())
  
  def simplex_proj(x: np.ndarray) -> np.ndarray:
    """Project vector x onto the unit simplex."""
    y = -np.sort(-x)
    sum_val = 0.0
    rho = 0
    for j in range(len(x)):
      sum_val += y[j]
      if j == len(x) - 1 or y[j+1] + (1 - sum_val) / (j + 1) <= 0:
        rho = j
        break
    delta = (1 - y[:rho+1].sum()) / (rho + 1)
    return np.clip(x + delta, 0, 1)

  @utils_timer
  def train(self, episodes_num, probe):
    '''
      params:
        episodes_num - 训练次数
        probe - 用户想要的偏好，例：[0.5, 0.5]
    '''
    loss_history = []
    reward_history = []
    for episode in tqdm(range(episodes_num), desc=self.name+' Iteration'):
      done = False
      tot_reward = 0
      loss, cnt = 0, 0
      state, _ = self.env.reset()
      while not done:
        action = self.take_action(state)
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        self.memorize(state, action, next_state, reward, float(done))
        if self.buffer.size() > self.batch_size:
          loss += self.update()
        tot_reward += (probe.cpu().numpy().dot(reward))*np.power(self.gamma, cnt)
        cnt += 1
        if cnt > 100:
          done = True
          self.reset()
        state = next_state
      reward_history.append(tot_reward)
      loss_history.append(loss)
    return loss_history, reward_history
      



#         ,--.                                                 ,--.     
#  ,---.  |  ,---.   ,--,--. ,--,--,--. ,--.--.  ,---.   ,---. |  |,-.  
# (  .-'  |  .-.  | ' ,-.  | |        | |  .--' | .-. | | .--' |     /  
# .-'  `) |  | |  | \ '-'  | |  |  |  | |  |    ' '-' ' \ `--. |  \  \  
# `----'  `--' `--'  `--`--' `--`--`--' `--'     `---'   `---' `--'`--' 


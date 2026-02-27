#                       多智能体强化学习模型
#                           2026/2/14
#                            shamrock

from tqdm import tqdm

from myTools.Utils.tools import *
from myTools.Utils.config import *
from myTools.Utils.RL_config import *
from myTools.Utils.MARL_config import *
from myTools.Utils.MARL_tools import *

#---------------------- Independent 框架 -------------------------
#                          2026/2/14

class Independent_Trainer(RL_TrainerConfig):
  '''
    Independent 框架的训练器，RL算法为每一个agent单独计算
  '''
  def __init__(self, rl:RL_Model, config:str, agents_num, 
               env:MARL_EnvConfig, device):
    '''
      params:
        rl - 传统的RL算法
        config - 参数配置文件路径(.json)
        agents_num - 需要agents的数量
    '''
    model_params = utils_readParams(config, 'model')
    buffer_params = utils_readParams(config, 'buffer')
    trainer_params = utils_readParams(config, 'trainer')
    utils_setAttr(self, buffer_params)
    utils_setAttr(self, trainer_params)
    utils_autoAssign(self)
    qnet = Q_Net(model_params['state_dim'], model_params['action_dim'], config)
    self.agent:RL_Model = rl(**model_params, Q_Net=qnet, device=device)
    # self.agents = [rl(**model_params) for _ in range(n_agents)]
    self.name = f'Independent {self.agent.name}'

  def update(self, transition_dict):
    print('update')
    # n_agents = transition_dict['states'].shape[1]
    n_cuavs = self.env.n_charging_uavs
    cuav_losses = np.zeros(n_cuavs, dtype=float)
    for i in range(n_cuavs):
      idx = i+n_cuavs
      single_transition = {
        'states': transition_dict['states'][:, idx, :],         # (B, state_dim)
        'actions': transition_dict['actions'][:, idx],          # (B,)
        'next_states': transition_dict['next_states'][:, idx, :],  # (B, state_dim)
        'rewards': transition_dict['rewards'][:, idx],          # (B,)
        'dones': transition_dict['dones']                             # (B,) 全局done
      }
      cuav_losses[i] += self.agent.update(single_transition)
    return np.sum(cuav_losses)

  def take_action(self, state) -> np.ndarray:
    '''
      任务机随机移动，充电机根据DQN提供的take action移动
    '''
    actions = np.zeros(self.agents_num, dtype=int)
    for i in range(self.env.n_task_uavs):
      actions[i] = np.random.randint(0, 5)
    for i in range(self.env.n_charging_uavs):
      actions[i+self.env.n_task_uavs] = self.agent.take_action(state[i])
    return actions

  def train(self, replay_buffer:utils_ReplayBuffer=None, model_name=None, begin=0):
    '''
      训练 episodes_num 次，
        params:
          is_on_policy - 使用的 RL 算法是否是 on-policy
    '''
    if model_name:
      self.agent.load_model(self.model_path, model_name)
    history = {'rewards': [], 'loss': []}
    if self.is_on_policy:
      transition_dict = {
        'states': [], 
        'actions': [], 
        'next_states': [], 
        'rewards': [], 
        'dones': []
        }
      transitions = [transition_dict]*self.agents_num
      with tqdm(total=self.train_episodes, desc=f'{self.name}') as pbar:
        for episode in range(self.train_episodes):
          done = False
          state, _ = self.env.reset()
          total_rewards = np.zeros(self.agents_num)
          while not done:
            actions = np.array([self.agent.take_action(state[i])
              for i in range(self.agents_num)])
            next_state, reward, done, info = self.env.step(actions)
            for agent_i in range(self.agents_num):
              transitions[agent_i]['states'].append(state[agent_i])
              transitions[agent_i]['actions'].append(actions[agent_i])
              transitions[agent_i]['next_states'].append(next_state[agent_i])
              transitions[agent_i]['rewards'].append(reward[agent_i])
              transitions[agent_i]['dones'].append(done[agent_i])
            state = next_state
            total_rewards += reward
          history['rewards'].append(total_rewards)
          for i in range(self.agents_num):
            self.agent.update(transitions[i])
          pbar.update(1)
    else:
      with tqdm(total=self.train_episodes, desc=f'{self.name}') as pbar:
        for episode in range(self.train_episodes):
          done = False
          state, _ = self.env.reset()
          total_rewards = np.zeros(self.agents_num)
          loss = 0
          while not done:
            actions = self.take_action(state)
            next_state, reward, done, _ = self.env.step(actions)
            n_charging_uavs = self.env.n_charging_uavs
            replay_buffer.add_sample(Sample(state, actions, reward, next_state, done))
            state = next_state
            total_rewards += reward
            if len(replay_buffer) > self.min_size:
              transition_dict = replay_buffer.sample_sample(self.batch_size)
              loss += self.update(transition_dict)
          history['loss'].append(loss)
          history['rewards'].append(total_rewards)
          pbar.update(1)
          if (episode+1)%100 == 0:
            print('\nmean rewards: ', np.mean(history['rewards'][-100:]))
            print('mean losss: ', np.mean(history['loss'][-100:]))
            self.agent.save_model(self.model_path, f'\{self.agent.name}_{episode+begin}.pt')
    utils_showHistory(history, list(history.keys()), 'test', 'eps', 
                      'reward', self.history_path+f'\{self.agent.name}.png')
    return history

  def eval(self, model_name=None) -> Dict[str, Any]:
    '''
      测试
        params:
          delay - 每帧的渲染延迟（秒）
    '''
    if model_name:
      self.agent.load_model(self.model_path, model_name)
    if hasattr(self.agent, 'eval'):
      self.agent.eval()
    eval_results = {
      'total_rewards': [],
      'steps': [],
      'alive_tasks': []
    }

    try:
      for ep in range(self.eval_episodes):
        state, _ = self.env.reset()
        total_reward = np.zeros(self.agents_num)
        done = False
        step = 0
        
        print(f"\n{'='*60}")
        print(f"Evaluation Episode {ep+1}/{self.eval_episodes}")
        print(f"{'='*60}")
        
        while not done:
          # 生成动作（任务机随机，充电机DQN）
          actions = self.take_action(state)
          
          # 环境步进
          next_state, reward, done, info = self.env.step(actions)
          
          # 渲染动画（关键：展示UAVs运行动画）
          self.env.render()
          time.sleep(self.delay)  # 控制动画速度
          
          # 累计奖励
          total_reward += reward
          state = next_state
          step += 1
          
          # 实时显示关键信息
          if step % 20 == 0:
            alive_count = sum(1 for t in self.env.task_uavs if t.state["alive"])
            print(f"  Step {step}: Alive Tasks={alive_count}/{self.env.n_task_uavs} | "
                  f"Mean Reward={np.mean(reward):.4f}")
        
        # 记录本轮结果
        alive_count = sum(1 for t in self.env.task_uavs if t.state["alive"])
        eval_results['total_rewards'].append(np.sum(total_reward))
        eval_results['steps'].append(step)
        eval_results['alive_tasks'].append(alive_count)
        
        # 显示本轮结果
        print(f"\nEpisode {ep+1} Result:")
        print(f"  Steps: {step}")
        print(f"  Total Reward: {np.sum(total_reward):.2f}")
        print(f"  Alive Task UAVs: {alive_count}/{self.env.n_task_uavs}")
        print(f"  Success: {'✓ YES' if alive_count == self.env.n_task_uavs else '✗ NO'}")
        print(f"{'='*60}\n")
    
    finally:
      # 确保切换回训练模式
      if hasattr(self.agent, 'train'):
        self.agent.train()

    return {
      'mean_total_reward': np.mean(eval_results['total_rewards']),
      'mean_steps': np.mean(eval_results['steps']),
      'success_rate': np.mean([
          1.0 if alive == self.env.n_task_uavs else 0.0 
          for alive in eval_results['alive_tasks']
      ]),
      'mean_alive_tasks': np.mean(eval_results['alive_tasks']),
      'raw_results': eval_results  # 保留原始数据供进一步分析
    }

#         ,--.                                                 ,--.     
#  ,---.  |  ,---.   ,--,--. ,--,--,--. ,--.--.  ,---.   ,---. |  |,-.  
# (  .-'  |  .-.  | ' ,-.  | |        | |  .--' | .-. | | .--' |     /  
# .-'  `) |  | |  | \ '-'  | |  |  |  | |  |    ' '-' ' \ `--. |  \  \  
# `----'  `--' `--'  `--`--' `--`--`--' `--'     `---'   `---' `--'`--'
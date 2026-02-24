from myTools.Utils.MARL_env import MARL_Env_UAVs
# from myTools.Utils.MARL_tools import Q_Net
from myTools.Model.MARL import *
from myTools.Model.RL import *

device = utils_getDevice()

config_path = 'Knowledge\MutilAgentReinforcementLearning\MultiUAVsCharging\config.json'

env = MARL_Env_UAVs(config_path, (5, 5))

env_dir = utils_readParams(config_path, 'env')
agents_num = env_dir['n_task_uavs']+env_dir['n_charging_uavs']
agent = Independent_Trainer(DoubleDQN, config_path, agents_num, env, device)

replay_buffer = utils_ReplayBuffer(10000)

if __name__ == "__main__":
  # TODO 1.表格记录有问题 2.render不动
  history = agent.train(replay_buffer=replay_buffer)
  utils_showHistory(history, list(history.keys()), 'test', 'eps', 'reward')
  agent.eval()
  # num = 0
  # agent.eval(f'\Double_DQN_{num}.pt')

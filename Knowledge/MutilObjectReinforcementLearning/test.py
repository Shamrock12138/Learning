from myTools.Utils.MORL_env import *
from myTools.Utils.MORL_tools import *
from myTools.Model.MORL import *

device = utils_getDevice()

args_path = 'F:\\GraduateStudent\\Code\\Git_Learning\\Knowledge\\MutilObjectReinforcementLearning\\Learning\\EnvelopeQLearning\\config.json'
buffer_params = utils_readParams(args_path, 'buffer')
agent_params = utils_readParams(args_path, 'agent')
trainer_params = utils_readParams(args_path, 'trainer')

env = Env_DeepSeaTreasure()

model = EQL_Network(env._states_size, env._actions_size, env._rewards_size)

transition_type = ('state', 'action', 'next_state', 'reward', 'done')
buffer = utils_prioritReplayBuffer(
  buffer_params['mem_size'], 
  transition_type
  )

agent = EQL(env, model, buffer, agent_params, device)
probe = torch.tensor([0.8, 0.2], dtype=torch.float32, device=device)
history = agent(trainer_params['episodes_num'], probe)
utils_showHistory(history, 'EQL', 'episodes', 'loss')

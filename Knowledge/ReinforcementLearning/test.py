import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from Projects.Model.RL import *
from Projects.Utils.RL_tools import *
from Projects.Utils.tools import *

device = utils_getDevice()

# env = Env_CliffWalking(height=5, width=5)
# env = Env_FrozenLake()
env = Env_CartPole()
# env = Env_AimBall()
# env = Env_AimBallDynamic(target_move_mode='uniform')

# agent = DP_ValueIteration(env, 0.001, 0.9)
# agent = DP_PolicyIteration(env, 0.001, 0.9)
# agent = SARSA_nstep(env, 5, 0.1, 0.1, 0.9)
# agent = SARSA(env, 0.1, 0.1, 0.9)
# agent = Q_Learning(env, 0.3, 0.1, 0.9)
# agent = Dyna_Q(env, 0.6, 0.1, 0.9, 3)
agent = DQN(env, env._states_num, 256, env._actions_num, 
            2e-3, 0.98, 0.01, 30, device)

model_save_path = r'F:\\GraduateStudent\\Code\\Git_Learning\\Knowledge\\ReinforcementLearning\\Learning\\Deep_Q_Network\\model\\'
history_save_path = r'F:\\GraduateStudent\\Code\\Git_Learning\\Knowledge\\ReinforcementLearning\\Learning\\Deep_Q_Network\\history\\'

agent(300)
agent.show_history(history_save_path, 'target_update_30.png')
agent.save_model(model_save_path, 'target_update_30.pt')
# agent.load_model(model_save_path, 'target_update_30.pt')
# agent.render(10)


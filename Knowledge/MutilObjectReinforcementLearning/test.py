from myTools.Utils.MORL_env import *
from myTools.Utils.MORL_tools import *
from myTools.Model.MORL import *

model = EQL_Network()

env = Env_DeepSeaTreasure()

buffer = utils_prioritReplayBuffer()

agent = EQL(env, model, buffer, )

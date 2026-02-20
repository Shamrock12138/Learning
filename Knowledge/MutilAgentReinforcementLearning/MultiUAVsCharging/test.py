from myTools.Utils.MARL_env import MARL_Env_UAVs
from myTools.Model.MARL import *

env = MARL_Env_UAVs('Knowledge\MutilAgentReinforcementLearning\MultiUAVsCharging\config.json')

if __name__ == "__main__":
  # 重置环境
  obs, info = env.reset()
  print(f"Environment initialized! Base station at: {info['base_station']}")

  # 单步交互测试
  actions = np.zeros(env.n_agents, dtype=int)
  actions[0] = 1  # 任务机0向上移动
  actions[env.n_task_uavs] = 5  # 充电机0执行充电模式

  next_obs, rewards, done, info = env.step(actions)
  env.render()  # ← 渲染当前状态

  print(f"Step completed. Done: {done}, Rewards: {rewards}")
  input("Press Enter to close window...")  # 保持窗口打开

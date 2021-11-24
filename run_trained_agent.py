import ray
import time
import ray.rllib.agents.dqn as dqn
from ray.tune.logger import pretty_print
from vacuum_robot_discrete_v0 import DiscreteVaccumRobotV0
from vacuum_robot_discrete_v1 import DiscreteVaccumRobotV1

ray.init(num_gpus=1)
config = dqn.DEFAULT_CONFIG.copy()
config["env_config"] = {'room_file': 'room2.csv', 'max_moves': 300, 'window_size': 5}
#config["env_config"] = {'room_file': 'room1.csv', 'max_moves': 300, 'window_size': 5}
config["num_gpus"] = 0
config["num_workers"] = 1
config["framework"] = 'torch'
config["double_q"] = True
config["dueling"] = True
config["num_atoms"] = 1
config["noisy"] = False
config["prioritized_replay"] = False
config["n_step"] = 1
config["target_network_update_freq"] = 8000
config["lr"] = 0.0000625
config["adam_epsilon"] = 0.00015
config["hiddens"] = [512]
config["learning_starts"] = 20000
config["buffer_size"] = 1000000
config["rollout_fragment_length"] = 4
config["train_batch_size"] = 32
config["exploration_config"] ={"epsilon_timesteps": 200000, "final_epsilon": 0.01}
config["prioritized_replay_alpha"] = 0.5
config["final_prioritized_replay_beta"] = 1.0
config["prioritized_replay_beta_annealing_timesteps"] = 2000000
config["timesteps_per_iteration"] = 10000
agent = dqn.DQNTrainer(config=config, env=DiscreteVaccumRobotV1)

# Can optionally call trainer.restore(path) to load a checkpoint.

agent.restore('/home/marcelo/ray_results/DQN_DiscreteVaccumRobotV1_2021-11-23_22-09-41ictjd0gp/checkpoint_000015/checkpoint-15')
#agent.restore('/home/marcelo/ray_results/DQN_DiscreteVaccumRobotV1_2021-11-23_22-09-41ictjd0gp/checkpoint_000001/checkpoint-1')

env = DiscreteVaccumRobotV1(config['env_config'])

for i in range(1000):
   episode_reward = 0
   terminal = False
   obs = env.reset()
   counter = 0
   while terminal is False:
      action = agent.compute_action(obs)
      obs, reward, terminal, info = env.step(action)
      episode_reward += reward
      env.render()
      print(reward)
      time.sleep(0.01)
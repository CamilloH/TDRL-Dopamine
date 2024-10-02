import os
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import TimeEnv1
import TimeEnv2
import TimeEnv3
import Learning
import parameters
import plotter

from stable_baselines3.common.env_checker import check_env



#  update rules possible to use 

env = TimeEnv1.TimeEnvironment()

#check whether the environment satisfies gym rules
check_env(env)
obs, _ = env.reset()

# define the q_table 
q_table = np.zeros((env.observation_space.n, env.action_space.n))
value_table = np.zeros(env.observation_space.n)

# biased action 
biased_action = random.randint(0,1)
# introduce bias by pre-setting the qvalues for a certain action a little above the other side 
# action/reward state is state nr.5 (if we dont have a transition state)
for i in range(4):
    q_table[i + 1, biased_action] = 0.5
print(q_table)
#print(biased_action)


print("First rule")
fraction_correct_session, rpe_all_trials, response_bias, rewards_all_trials, q_table, env1_first_sess_percentages, env1_last_sess_percentages, rpe_only_reward, rpe_no_reward = Learning.learn(env, q_table, parameters.SARSA)
print("Q Table")
print(np.round(q_table, 3))
plotter.plot_results(fraction_correct_session, rpe_all_trials, response_bias, rewards_all_trials, q_table, "Location", rpe_only_reward, rpe_no_reward)

#TD learning
# value_table = Learning.learn(env, value_table, biased_action, False, "td")
# print("State Values:")
# print(value_table)


env = TimeEnv2.TimeEnvironment()
print("second rule:")
fraction_correct_session, rpe_all_trials, response_bias, rewards_all_trials, q_table,  env2_first_sess_percentages, env2_last_sess_percentages, rpe_only_reward, rpe_no_reward = Learning.learn(env, q_table, parameters.SARSA)
print("Q Table")
print(np.round(q_table, 3))
plotter.plot_results(fraction_correct_session, rpe_all_trials, response_bias, rewards_all_trials, q_table, "Frequency", rpe_only_reward, rpe_no_reward)

# TD learning
# value_table = Learning.learn(env, value_table, biased_action, False, "td")
# print("State Values:")
# print(value_table)

env = TimeEnv3.TimeEnvironment()
print("third rule:")
fraction_correct_session, rpe_all_trials, response_bias, rewards_all_trials, q_table,  env3_first_sess_percentages, env3_last_sess_percentages, rpe_only_reward, rpe_no_reward = Learning.learn(env, q_table, parameters.SARSA)
print("Q Table")
print(np.round(q_table, 3))


plotter.plot_results(fraction_correct_session, rpe_all_trials, response_bias, rewards_all_trials, q_table, "Frequency reversed", rpe_only_reward, rpe_no_reward)

plotter.plot_ruleswitches(env1_last_sess_percentages, env2_first_sess_percentages,env2_last_sess_percentages, env3_first_sess_percentages )
# rule switch 1 no more bias
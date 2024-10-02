from cProfile import label
import os
import random
from turtle import color
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import Learning
import parameters

def plot_results(fraction_correct_session, rpe_all_trials, response_bias, rewards_all_trials, q_table, rule_type, rpe_only_reward, rpe_no_reward):
    #rewards_per_session = np.split(np.array(rewards_all_trials[3,:]), parameters.num_sessions)
    #count = 1
   # print("********** Average  reward per session **********\n")

    #for r in rewards_per_session:
     #       print(count, ": ", str(sum(r / parameters.num_trials_per_session)))
      #      count += 1
    
    # Print updated Q-table
   # print("\n\n********** Q-table **********\n")
    #print(q_table)

    rpe_no_reward = np.array(rpe_no_reward)
    rpe_only_reward = np.array(rpe_only_reward)
    #no reward trials 
    first_trials_average_no_reward = np.true_divide(np.sum(rpe_no_reward[0:50 , :], axis = 0), 50)
    middle_trials_average_nor_reward = np.true_divide(np.sum(rpe_no_reward[490 : 545, :], axis = 0), 50)
    end_trials_average_no_reward = np.true_divide(np.sum(rpe_no_reward[2500 : 2550, :], axis = 0), 50)
    
    #reward trials
    first_trials_average = np.true_divide(np.sum(rpe_only_reward[0:30 , :], axis = 0), 30)
    middle_trials_average = np.true_divide(np.sum(rpe_only_reward[250 : 280, :], axis = 0), 30)
    end_trials_average = np.true_divide(np.sum(rpe_only_reward[500 : 530, :], axis = 0), 30)
    #print(rpe_only_reward[]


    #plotting
    fig, axs = plt.subplots(2,2, figsize = (9,5))
    axs[0,0].set(ylabel = "RPE random trials ")
    axs[0,0].plot(rpe_all_trials[10, :], color = "red", label="trial 10")
    axs[0,0].plot(rpe_all_trials[500, :], color = "blue", label="trial 500")
    axs[0,0].plot(rpe_all_trials[1000, :], color = "green", label="trial 1000")
    axs[0,0].set_ylim(-1,1)

    axs[0,1].set(ylabel = "Average RPE per 10 reward trials")
    axs[0,1].plot(first_trials_average, color = "red", label="trial 0-10")
    axs[0,1].plot(middle_trials_average, color = "blue", label="trial 250-260")
    axs[0,1].plot(end_trials_average, color = "green", label="trial 500-510")
    axs[0,1].set_ylim(-1,1)
    axs[0,1].legend()

    axs[1,0].set(ylabel = "Fraction correct/session")
    axs[1,0].plot(fraction_correct_session)
    axs[1,0].set_ylim(0,1)

    axs[1,1].set(ylabel = "Absolute Response Bias")
    axs[1,1].plot(response_bias)
    axs[1,1].set_ylim(0,0.5)

    fig.suptitle(rule_type)
    fig.tight_layout()
    plt.show()      




'''
Sound types:
0 = low frequency from left 
1 = high frequency from left 
2 = low frequency from right 
3 = high frequency from right
'''

def plot_ruleswitches(env1_last_sess_percentages, env2_first_sess_percentages, env2_last_sess_percentages, env3_first_sess_percentages):
        rules1_2 = np.zeros(4)
        rules1_2[0] = (env1_last_sess_percentages[0] + env1_last_sess_percentages[3]) / 2
        rules1_2[1] = (env2_first_sess_percentages[0] + env2_first_sess_percentages[3]) / 2
        rules1_2[2] = (env1_last_sess_percentages[1] + env1_last_sess_percentages[2]) / 2
        rules1_2[3] = (env2_first_sess_percentages[1] + env2_first_sess_percentages[2]) / 2

        rules2_3 = np.zeros(2)
        rules2_3[0] = np.sum(env2_last_sess_percentages)/4
        rules2_3[1] = np.sum(env3_first_sess_percentages)/4
        
        fig, axs = plt.subplots(2, figsize = (9,5))
        axs[0].set(ylabel = "Fraction correct")
        axs[0].set_title("Location to frequency")
        axs[0].plot((0, rules1_2[0]), (1, rules1_2[1]), color = 'black', label = 'Stay conditions')
        axs[0].plot((0,rules1_2[2]), (1, rules1_2[3]), color = 'purple', label = 'Switch conditions')
        axs[0].legend()

        axs[1].set(ylabel = "Fraction correct")
        axs[1].set_title("Frequency to frequency reversed")
        axs[1].plot((0,rules2_3[0]), (1, rules2_3[1]), color = 'purple', label = 'Switch conditions')
        axs[1].legend()
        fig.suptitle("Rule switches")
        fig.tight_layout()
        plt.show()





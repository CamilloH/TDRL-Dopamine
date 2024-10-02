import os
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import parameters
import learning_update

'''
parameters: 
env: used environment/currently active rule
q_table: empty q_table with size: state x actions
biased_action: biased action the mouse has at the start of a new rule 
algorithm: update rule used
'''

def learn(env, q_table, update_rule):
    
    exploration_rate = parameters.exploration_rate
    # bias = parameters.bias

    #lists for different plots
    rewards_all_trials = np.zeros(( parameters.num_sessions * parameters.num_trials_per_session, 5))
    rpe_all_trials = np.zeros((parameters.num_sessions * parameters.num_trials_per_session, 5))
    rpe_only_reward = []
    rpe_no_reward = []

    rpe_all_trials_neg = [] #np.zeros((5, parameters.num_sessions * parameters.num_trials_per_session))
    predicted_values = [] # #np.zeros((5, parameters.num_sessions * parameters.num_trials_per_session))

    fraction_correct_session = [] # np.zeros(num_sessions) #for plot d
    fraction_correct_left = []  #for plot e
    fraction_correct_right = [] #for plot e 

    response_bias = []
    # biases = []
    first_sess_percentages = []   
    last_sess_percentages = []  

    '''
    if the performance exceeds 80 percent in a session this will increase
    if it does twice, learning the current rule is a success and learning will be terminated 
    '''
    end_criterion_count = 0


    for session in range(parameters.num_sessions): 
        session_right_counter = 0
        session_left_counter = 0
        session_correct = 0
        session_correct_left = 0 
        session_correct_right = 0 

        low_left_counter = 0
        high_left_counter = 0
        low_right_counter = 0
        high_right_counter = 0

        low_left_right_counter = 0
        high_left_right_counter = 0
        low_right_right_counter = 0
        high_right_right_counter = 0

        for trial in range(parameters.num_trials_per_session):
            
            state, _ = env.reset() 
            # Exploration - exploitation trade-off
            if update_rule == "td":
                action = chooseAction_td(exploration_rate, q_table, state, env) #bias, with_bias, biased_action if would like to trs the bias decay again 
            else: # sarsa or q-learning
                action = chooseAction(exploration_rate, q_table, state, env)
            
            done = False

            counter = 0
            check_if_reward_trial = False
            while (not done):   
                rpe_array = np.zeros(5)
                new_state, reward, done,_, info = env.step(action)
                #state counters 
                if state == 1:
                    low_left_counter += 1
                #check for being in the reward state
                if state == 9:
                    low_left_right_counter += 1
                elif state == 2:
                    high_left_counter += 1
                #check for being in the reward state
                if state == 11:
                    high_left_right_counter += 1
                elif state == 3:
                    low_right_counter += 1 
                #check for being in the reward state
                if state == 13:
                    low_right_right_counter += 1
                elif state == 4:
                    high_right_counter += 1
                #check for being in the reward state
                if state == 13:
                    high_right_right_counter += 1

                if (state in (5,6,7,8) and action == 0): 
                    session_left_counter += 1
                elif(state in (5,6,7,8) and action == 1): 
                    session_right_counter += 1

                if(new_state in (9,11,13,15)):
                    session_correct += 1
                    #predicted_values.append(current_value)
                    if (action == 0): 
                        session_correct_left += 1
                    elif(action == 1): 
                        session_correct_right += 1 
                # value that reflects dopamine encoding
                # rpe_all_trials.append(rpe) 

                # Update q_table for Q(s,a) with q learning
                if update_rule == parameters.QLEARNING:
                    current_value = q_table[state, action]
                    max_q_value_state = np.max(q_table[new_state,:])
                    q_table[state, action] =  learning_update.q_learning(current_value, max_q_value_state, reward)
                    rpe = reward - current_value
                    #calc rpe value
                    rpe_array[counter] = rpe
                    if reward == 1: 
                        check_if_reward_trial = True
                    if (check_if_reward_trial and counter == 3): 
                        rpe_only_reward.append(rpe_array)
                    elif(check_if_reward_trial == False and counter == 3 ): 
                        rpe_no_reward.append(rpe_array)
                    rpe_all_trials[(session * parameters.num_trials_per_session) + trial, counter] = rpe 
                    rewards_all_trials[(session * parameters.num_trials_per_session) + trial, counter] = reward

                    action = chooseAction(exploration_rate, q_table, new_state, env)
                # Update q_table for Q(s,a) with SARSA
                
                elif update_rule == parameters.SARSA:
                    current_value = q_table[state, action]
                    action2 = chooseAction(exploration_rate, q_table, new_state, env)
                    next_q_value = (q_table[new_state, action2])
                    debug = learning_update.sarsa(current_value, next_q_value, reward)
                    q_table[state, action] =  learning_update.sarsa(current_value, next_q_value, reward)
                    rpe = reward - current_value
                    #calc rpe value
                    rpe_array[counter] = rpe
                    if reward == 1: 
                        check_if_reward_trial = True
                    if (check_if_reward_trial and counter == 3): 
                        rpe_only_reward.append(rpe_array)
                    elif(check_if_reward_trial == False and counter == 3 ): 
                        rpe_no_reward.append(rpe_array)
                    rpe_all_trials[(session * parameters.num_trials_per_session) + trial, counter] = rpe 
                    rewards_all_trials[(session * parameters.num_trials_per_session) + trial, counter] = reward
                    # have to use action which was used within the update rule 
                    action = action2
                elif update_rule == "td": 
                    state_value = q_table[state]
                    next_state_value = q_table[new_state]
                    q_table[state] = learning_update.td(state_value, next_state_value, reward)
                    action = chooseAction_td(exploration_rate, q_table, new_state, env)
                state = new_state
                counter = counter + 1


                

                
                # count occurence of each sound plus choosing right at occurence  
                
                
                
            # Exploration rate decay
            exploration_rate = parameters.min_exploration_rate + \
                (parameters.max_exploration_rate - parameters.min_exploration_rate) * np.exp(-parameters.exploration_decay_rate * (((session) * parameters.num_trials_per_session) + trial))
            #bias decay to be at 0 ~ session 7
            # bias_decay_value = (((session) * parameters.num_trials_per_session) + trial) / (3000)
            # bias = np.exp(-(bias_decay_value**2)/0.2) 
            # biases.append(bias)

            ###
            ###
            ##
            ##
            ##
            ##
            # Data acquisition for the different plots
            #rewards_all_trials.append(reward)
            #rpe_all_trials_neg.append(rpe)


        #lists for different fractions of correct actions
        fraction_correct_session.append(session_correct / parameters.num_trials_per_session)
        if(session_left_counter == 0):
            fraction_correct_left.append(0)
        elif(session_left_counter != 0):
            fraction_correct_left.append(session_correct_left / session_left_counter)
            
        if(session_right_counter == 0):
            fraction_correct_right.append(0)
        elif(session_right_counter != 0):
            fraction_correct_right.append(session_correct_right / session_right_counter)

        last_session = False
        if (session_correct / parameters.num_trials_per_session >= 0.8): 
            end_criterion_count += 1
         # learning success or last session
        if end_criterion_count == 2 or session == (parameters.num_sessions - 1):
            last_session = True
        # for the rule switch plot calculate the percentage of correctly choosing each sound 
        if session == 0:
            first_sess_percentages.append(low_left_right_counter/low_left_counter)
            first_sess_percentages.append(high_left_right_counter/high_left_counter)
            first_sess_percentages.append(low_right_right_counter/low_right_counter)
            first_sess_percentages.append(high_right_right_counter/high_right_counter)
        if last_session == True:
            last_sess_percentages.append(low_left_right_counter/low_left_counter)
            last_sess_percentages.append(high_left_right_counter/high_left_counter)
            last_sess_percentages.append(low_right_right_counter/low_right_counter)
            last_sess_percentages.append(high_right_right_counter/high_right_counter)
            break
    # calculate absolute response bias 
    for i in range(len(fraction_correct_left)): 
        response_bias.append(abs((fraction_correct_left[i] / (fraction_correct_left[i] + fraction_correct_right [i])) - 0.5))

    
    return fraction_correct_session, rpe_all_trials, response_bias, rewards_all_trials, q_table, first_sess_percentages, last_sess_percentages, rpe_only_reward, rpe_no_reward

#TODO different policy for the beginning 

def chooseAction(exploration_rate, q_table, state, env): 
    if random.uniform(0,1) > exploration_rate: 
        action = np.argmax(q_table[state,:])
    else:
        action = env.action_space.sample()
    # biased action 
    #if ((random.uniform(0,1) < bias) and with_bias): 
    #    action = biased_action
    return action

def chooseAction_td(exploration_rate, q_table, state, env): 
    if random.uniform(0,1) > exploration_rate: 
    # TODO
    # doesnt work
        action = np.argmax(q_table[state])
    else:
        action = env.action_space.sample()
    # biased action 
    #if ((random.uniform(0,1) < bias) and with_bias): 
    #    action = biased_action
    return action
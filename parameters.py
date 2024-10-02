SARSA = "sarsa"
QLEARNING = "qLearning"



num_sessions = 30 
num_trials_per_session = 300

learning_rate = 0.01   
discount_rate = 0.5

#strong bias in the beginning
# bias = 1 
# max_bias = 1 
# min_bias = 0.01 

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01

exploration_decay_rate = 0.001 #if we decrease it, will learn slower if qlearning is used


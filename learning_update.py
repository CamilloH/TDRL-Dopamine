import parameters
import numpy as np

def q_learning(q_value, max_q_value_state, reward): 
    return (1 - parameters.learning_rate) * q_value + (parameters.learning_rate * (reward + parameters.discount_rate * max_q_value_state - q_value))


def sarsa(q_value, next_q_value, reward): 
    return (1 - parameters.learning_rate) * q_value + (parameters.learning_rate * (reward + parameters.discount_rate * next_q_value - q_value))


def td(state_value, next_state_value, reward): 
    return (1 - parameters.learning_rate) * state_value + (parameters.learning_rate * (reward + parameters.discount_rate * next_state_value - state_value))

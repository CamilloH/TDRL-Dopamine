import os
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class TimeEnvironment(gym.Env):
  """
  This envirnment represents the second rule
  """
  # Because of google colab, we cannot implement the GUI ('human' render mode)
  metadata = {'render.modes': ['human']}
  # Define constants for clearer code
  LICK_LEFT = 0
  LICK_RIGHT = 1
  DO_NOTHING = 2
  #States

  INITIAL_STATE = 0 

  SOUND_LEFT_LOW = 1
  SOUND_LEFT_HIGH = 2
  SOUND_RIGHT_LOW = 3
  SOUND_RIGHT_HIGH = 4

  ACTION_STATE_STIMULUS1 = 5
  ACTION_STATE_STIMULUS2 = 6
  ACTION_STATE_STIMULUS3 = 7
  ACTION_STATE_STIMULUS4 = 8

  REWARD_STATE_STIMULUS1 = 9 
  NO_REWARD_STATE_STIMULUS1 = 10
  REWARD_STATE_STIMULUS2 = 11
  NO_REWARD_STATE_STIMULUS2 = 12
  REWARD_STATE_STIMULUS3 = 13
  NO_REWARD_STATE_STIMULUS3 = 14
  REWARD_STATE_STIMULUS4 = 15
  NO_REWARD_STATE_STIMULUS4 = 16

  FINAL_STATE = 17

  def __init__(self):
    super(TimeEnvironment, self).__init__()


    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions, we have three: left, right and do nothing
    n_actions = 3
    self.action_space = spaces.Discrete(n_actions)
    
    # How many different stimuli can be presented (4 in our case)
    self.num_stimuli = 4 #num_stimuli
    # Observation is the sound type can be box or discrete 
    # self.observation_space = spaces.Box(low=0, high=self.num_stimuli - 1,
    #                                  shape=(1,), dtype=np.float32)

    self.observation_space = spaces.Discrete(18)
    '''
    Sound types:
    1 = low frequency from left 
    2 = high frequency from left 
    3 = low frequency from right 
    4 = high frequency from right
    '''
    # Initial sound direction and frequency (our states) will always be random
    self.state = self.INITIAL_STATE
    self.sound_type = random.randint(1,4)


  def reset(self, seed=None):
    """
    Important: the observation must be a numpy array
    :return: (np.array) 
    """
    # reset environment and present a new random sound type
  
    self.sound_type = random.randint(1,4)
    #start state every action yields 0 reward 
    self.state = self.INITIAL_STATE 
    self.info = {}
    
    return self.state, self.info

  def step(self, action):
    # done will be set to true if were in the final state
    done = False

    # first either reward assignment
    # if were not in reward states (numbers 9, 11, 13 ,15) the reward is always zero
    if self.state in (9, 11, 13, 15): 
        reward = 1
    else: 
        reward = 0
    #state transitions: 

    # the action is only relevant in the action states 
    # only the state transitions from action to reward state are defined here 

    if self.state == self.INITIAL_STATE: 
        #if we are in the initial state the second state will be random and will reflect a stimulus 
        self.state = self.sound_type
    # 1 2 3 4 are the soundtypes/stimulus states
    elif self.state in (1,2,3,4): 
        # Stim 1
        if self.state == self.SOUND_LEFT_LOW: 
            self.state = self.ACTION_STATE_STIMULUS1
        # Stim 2
        elif self.state == self.SOUND_LEFT_HIGH: 
            self.state = self.ACTION_STATE_STIMULUS2
        # Stim 3
        elif self.state == self.SOUND_RIGHT_LOW: 
            self.state = self.ACTION_STATE_STIMULUS3
        # Stim 4
        elif self.state == self.SOUND_RIGHT_HIGH: 
            self.state = self.ACTION_STATE_STIMULUS4

    #lick left for reward 
    elif self.state == self.ACTION_STATE_STIMULUS1: 
        if action == self.LICK_LEFT:  
            self.state = self.REWARD_STATE_STIMULUS1
        elif action == self.LICK_RIGHT:  
            self.state = self.NO_REWARD_STATE_STIMULUS1
        elif action == self.DO_NOTHING:
            self.state = self.NO_REWARD_STATE_STIMULUS1
    #lick right for reward
    elif self.state == self.ACTION_STATE_STIMULUS2: 
        if action == self.LICK_LEFT:  
            self.state = self.NO_REWARD_STATE_STIMULUS2
        elif action == self.LICK_RIGHT:  
            self.state = self.REWARD_STATE_STIMULUS2
        elif action == self.DO_NOTHING:
            self.state = self.NO_REWARD_STATE_STIMULUS2
        
    #lick left for reward
    elif self.state == self.ACTION_STATE_STIMULUS3: 
        if action == self.LICK_LEFT:  
            self.state = self.REWARD_STATE_STIMULUS3
        elif action == self.LICK_RIGHT:  
            self.state = self.NO_REWARD_STATE_STIMULUS3
        elif action == self.DO_NOTHING:
            self.state = self.NO_REWARD_STATE_STIMULUS3
        
    #lick right for reward
    elif self.state == self.ACTION_STATE_STIMULUS4: 
        if action == self.LICK_LEFT:  
            self.state = self.NO_REWARD_STATE_STIMULUS4
        elif action == self.LICK_RIGHT:  
            self.state = self.REWARD_STATE_STIMULUS4
        elif action == self.DO_NOTHING:
            self.state = self.NO_REWARD_STATE_STIMULUS4
    
    elif self.state in (9, 10, 11, 12, 13, 14, 15, 16):
        self.state = self.FINAL_STATE
        done = True
    
    #TODO 
    #maybe introduce small negative reward?  else nothing happens



    # else:
    #  raise ValueError("Received invalid action={} which is not part of the action space".format(action))

    # State transitions
    
    

    # Optionally we can pass additional info, we are not using that for now
    info = {}
    truncated = False

    
    return self.state, reward, done, truncated, info


  def render(self, mode='console'):
    if mode != 'console':
      raise NotImplementedError()
    # agent is represented as a cross, rest as a dot
    if(self.sound_type in (0, 1)):
      print("Sound coming from the left frequency doesn't matter")
    elif(self.sound_type in (2, 3)):
      print("Sound coming from the right frequency doesn't matter")
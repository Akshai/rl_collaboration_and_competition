from unityagents import UnityEnvironment
import numpy as np
#from maddpg import MADDPG
import torch
import numpy as np
import os
#from utilities import transpose_list, transpose_to_tensor
import time
from multi_ddpg_agent import MADDPG 
from collections import deque
import matplotlib.pyplot as plt

env = UnityEnvironment(file_name="Tennis.app")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]


# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents 
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])


scores_deque = deque(maxlen=100)
scores = []
scores_avg = []
episodes = 1        

PRINT_EVERY = 1

print(state_size)
print('action_size: ', action_size)
maddpgagent = MADDPG(2,state_size, action_size, 0) 
maddpgagent.load_weights()

for e in range(episodes):                                    # play game for 5 episodes
    env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
    states = env_info.vector_observations                  # get the current state (for each agent)
    score = np.zeros((num_agents,))
    
    while True:
        actions = maddpgagent.act(states)
        env_info = env.step(actions)[brain_name]           # send all actions to tne environment
        
        next_states = env_info.vector_observations
        rewards = env_info.rewards 
        re = np.asarray(rewards)
        dones = env_info.local_done          
        score += np.array(re)
       
        maddpgagent.step(states, actions, rewards, next_states, dones)
        
        states = next_states
        
        if any(dones):                                 
            break
        





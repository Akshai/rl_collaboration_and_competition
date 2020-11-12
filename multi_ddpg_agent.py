import numpy as np

import time

import torch
from replay_buffer import ReplayBuffer
from ddpg_agent import DDPGAgent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MADDPG:
    '''
    MultiAgent DDPG implementation
    '''

    def __init__(self, 
                 number_of_agents, 
                 state_size, 
                 action_size, 
                 rand_seed,
                 BUFFER_SIZE = int(1e5),
                 BATCH_SIZE  = 250, pretrained=False):
        '''
        Creates a new instance of MultiAgent DDPG agent
        
        agent_qty: indicates the number of agents.
        state_size: size of the observation space
        action_size: size of the actions space
        '''    

        self.number_of_agents = number_of_agents
        self.batch_size = BATCH_SIZE 
        self.buffer_size = BUFFER_SIZE
       
        # Creating Replay Buffer for multiple agents
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size, rand_seed, self.number_of_agents)
     
        self.action_size = action_size
        self.state_size = state_size 
        print('r seed is: ', rand_seed)
        self.agents = [DDPGAgent(i, self.state_size, self.action_size, rand_seed, self) for i in range(self.number_of_agents)]

    def step(self, states, actions, rewards, next_states, dones):
        """ Add into memory the experience. It will include all agents information """

        self.memory.add(states, actions, rewards, next_states, dones)

        # For each agent perfoms a step 
        for agent in self.agents:
            agent.step()

    def act(self, states, add_noise=True):
        """ Returns the action to take """
        na_rtn = np.zeros([self.number_of_agents, self.action_size])
        for agent in self.agents:
            idx = agent.agent_id
            na_rtn[idx, :] = agent.act(states[idx], add_noise)
        return na_rtn

    def save_weights(self):
        """ Save agents weights"""
        for agent in self.agents:
            torch.save(agent.actor_local.state_dict(), 'agent{}_checkpoint_actor.pth'.format(agent.agent_id+1))
            torch.save(agent.critic_local.state_dict(), 'agent{}_checkpoint_critic.pth'.format(agent.agent_id+1))
    def load_weights(self):
        for agent in self.agents:
            agent.actor_local.load_state_dict(torch.load('checkpoints/agent{}_checkpoint_actor.pth'.format(agent.agent_id+1), map_location='cpu'))
            agent.critic_local.load_state_dict(torch.load('checkpoints/agent{}_checkpoint_critic.pth'.format(agent.agent_id+1), map_location='cpu'))
    
    def reset(self):
        """ Just reset all agents """
        for agent in self.agents:
            agent.reset()

    def __len__(self):
        return self.number_of_agents

    def __getitem__(self, key):
        return self.agents[key]
            
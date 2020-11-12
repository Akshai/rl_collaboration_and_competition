from collections import namedtuple, deque
import random
import numpy as np
import torch

device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed, number_of_agents):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["states", "actions", "rewards", "next_states", "dones"])
        self.number_of_agents = number_of_agents
        
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        
        experiences = random.sample(self.memory, k=self.batch_size)
        
        list_of_states = list()
        list_of_actions = list()
        list_of_next_states = list()

        
        for i in range(self.number_of_agents):
            index = np.array([i])
            states = torch.from_numpy(np.vstack([e.states[index] for e in experiences if e is not None])).float().to(device)
            actions = torch.from_numpy(np.vstack([e.actions[index] for e in experiences if e is not None])).float().to(device)
            next_states = torch.from_numpy(np.vstack([e.next_states[index] for e in experiences if e is not None])).float().to(device)
            
            #print ('Memory states', states.shape)
            list_of_states.append(states)
            list_of_actions.append(actions)
            list_of_next_states.append(next_states)
        
        rewards = torch.from_numpy(np.vstack([e.rewards for e in experiences if e is not None])).float().to(device)        
        dones = torch.from_numpy(np.vstack([e.dones for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        #print ('Memory rewards', rewards.shape )
        #print ('Memory dones', dones.shape )
        

        return (list_of_states, list_of_actions, rewards, list_of_next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
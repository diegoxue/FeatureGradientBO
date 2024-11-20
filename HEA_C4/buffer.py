import numpy as np
import random
import torch
from collections import deque, namedtuple

from environment import Environment

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, device, env: Environment):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.device = device
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.env = env
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """ Randomly sample a batch of experiences from memory. """
        if len(self.memory) < self.batch_size:
            raise ValueError('Not enough experiences in memory to sample')

        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.stack([e.state.repr() for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.stack([e.next_state.repr() for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        # TODO check 
        next_states_obj = [e.next_state for e in experiences if e is not None]
        next_act_idx_bounds = [ns.get_action_idx_limits() if not ns.done() else (0, 0) for ns in next_states_obj]

        return (states, actions, rewards, next_states, dones, next_act_idx_bounds)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    '''
        NOTE: Random collection of sample experiences is done in utils.
    '''
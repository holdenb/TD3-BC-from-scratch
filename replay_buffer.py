import torch
import numpy as np
from typing_extensions import Self

from utils import DEVICE, D4RLDatasetKeys, StateUtils


class ReplayBuffer(object):
    def __init__(
        self,
        max_steps: int,
        state_dim: int,
        action_dim: int,
        norm_epsilon: float,
    ):
        self.state = np.zeros((max_steps, state_dim))
        self.action = np.zeros((max_steps, action_dim))
        self.next_state = np.zeros((max_steps, state_dim))
        self.reward = np.zeros((max_steps, 1)).reshape(-1, 1)
        self.not_done = np.zeros((max_steps, 1)).reshape(-1, 1)
        self.norm_epsilon = norm_epsilon
        self.size = 0
        self.mean = 0
        self.std = 0

    def random_sample(self, batch_size: int):
        rand_index = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[rand_index]).to(DEVICE),
            torch.FloatTensor(self.action[rand_index]).to(DEVICE),
            torch.FloatTensor(self.next_state[rand_index]).to(DEVICE),
            torch.FloatTensor(self.reward[rand_index]).to(DEVICE),
            torch.FloatTensor(self.not_done[rand_index]).to(DEVICE),
        )

    def init_states_from_D4RL_dataset(self, dataset) -> Self:
        self.state = dataset[D4RLDatasetKeys.STATE]
        self.action = dataset[D4RLDatasetKeys.ACTIONS]
        self.next_state = dataset[D4RLDatasetKeys.NEXT_STATE]
        self.reward = dataset[D4RLDatasetKeys.REWARD].reshape(-1, 1)
        self.not_done = 1.0 - dataset[D4RLDatasetKeys.NOT_DONE].reshape(-1, 1)
        self.size = self.state.shape[0]

        # Always normalize states
        (self.mean, self.std) = StateUtils.normalize(
            self.state, self.next_state
        )

        return self

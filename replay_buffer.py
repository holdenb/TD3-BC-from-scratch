import torch
import numpy as np

import utils


class __D4RLDatasetKeys:
    __STATE = "observations"
    __ACTIONS = "actions"
    __NEXT_STATE = "next_observations"
    __REWARD = "rewards"
    __NOT_DONE = "terminals"


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
        # Buffer position will allow us to round-robin
        # through the replay states
        self.buffer_position = 0
        self.size = 0

    def random_sample(self, batch_size: int):
        rand_index = np.random.randint(0, self.size, size=batch_size)
        sample = [
            torch.FloatTensor(self.state[rand_index]),
            torch.FloatTensor(self.action[rand_index]),
            torch.FloatTensor(self.next_state[rand_index]),
            torch.FloatTensor(self.reward[rand_index]),
            torch.FloatTensor(self.not_done[rand_index]),
        ]
        utils.move_to_device_ol_list(sample)

        return sample

    def norm(self) -> (float, float):
        mean = self.state.mean(0, keepdims=True)
        # epis is a small normalization constant. This is commonly
        # used in many deep RL algorithms
        std = self.state.std(0, keepdims=True) + self.norm_epsilon
        self.state = (self.state - mean) / std

        return (mean, std)

    def states_from_D4RL_dataset(self, dataset: dict[str, np.ndarray]) -> None:
        self.state = dataset[__D4RLDatasetKeys.__STATE]
        self.action = dataset[__D4RLDatasetKeys.__ACTIONS]

        self.next_state = dataset[__D4RLDatasetKeys.__NEXT_STATE]

        self.reward = dataset[__D4RLDatasetKeys.__REWARD].reshape(-1, 1)
        self.not_done = 1.0 - dataset[__D4RLDatasetKeys.__NOT_DONE].reshape(
            -1, 1
        )

        self.size = self.state.shape[0]

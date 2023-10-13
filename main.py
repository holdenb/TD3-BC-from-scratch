import os
from pathlib import Path
import gym
import numpy as np
import torch
import d4rl

from utils import DefaultParams
from replay_buffer import ReplayBuffer


def set_seeds(env_name: str, seed: int) -> gym.Env:
    env: gym.Env = gym.make(env_name)
    env.seed(seed)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    return env


# TODO add a flag to capture the duration of each training & output .npy


def main() -> None:
    # Sets the env seeds for gym/torch/numpy
    env: gym.Env = set_seeds(DefaultParams.GYM_ENV, DefaultParams.SEED)

    results = Path("./results")
    models = Path("./models")

    # Create output directories if they do not already exist
    if not Path("./results").exists():
        os.makedirs(results)

    if not Path("./models").exists():
        os.makedirs(models)

    # Environment state/action space
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # TODO optional model loader

    replay_buffer: ReplayBuffer = ReplayBuffer(
        DefaultParams.MAX_TIMESTEPS,
        state_dim,
        action_dim,
        DefaultParams.NORM_EPSILON,
    ).states_from_D4RL_dataset(d4rl.qlearning_dataset(env))

    # TODO optionally normalize states
    mean, std = replay_buffer.norm()

    env.close()
    return 0


if __name__ == "__main__":
    main()

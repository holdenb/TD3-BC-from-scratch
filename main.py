import os
from pathlib import Path
import gym
import numpy as np
import torch
import d4rl

from utils import DefaultParams
from replay_buffer import ReplayBuffer
from TD3_BC import TD3_BC


def set_seeds(env_name: str, seed: int) -> gym.Env:
    env: gym.Env = gym.make(env_name)
    env.seed(seed)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    return env


def fmt_output(eval_episodes, avg_reward, d4rl_score) -> None:
    print("---------------------------------------")
    print(
        f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f},"
        + f"D4RL score: {d4rl_score:.3f}"
    )
    print("---------------------------------------")


def eval_policy(
    policy_, env_name, seed, mean_, std_, seed_offset=100, eval_episodes=10
) -> float:
    eval_env = gym.make(env_name)
    # fixed seed for the environment
    eval_env.seed(seed + seed_offset)

    avg_reward = 0.0
    # evaluate over the given number of episodes
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            state = (np.array(state).reshape(1, -1) - mean_) / std_
            action = policy_.select_action(state)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes
    d4rl_score = eval_env.get_normalized_score(avg_reward) * 100

    fmt_output(eval_episodes, avg_reward, d4rl_score)

    return d4rl_score


def train_policy(
    policy: TD3_BC, replay_buffer: ReplayBuffer, mean, std
) -> None:
    results_filename = f"TD3_BC_{DefaultParams.GYM_ENV}_{DefaultParams.SEED}"

    evaluations = []
    for ts in range(int(DefaultParams.MAX_TIMESTEPS)):
        policy.train(replay_buffer, DefaultParams.BATCH_SIZE)
        # Evaluate episode at each interval according to the
        # evaluation frequency
        if (ts + 1) % DefaultParams.EVAL_FREQ == 0:
            print(f"Time steps: {ts+1}")
            evaluations.append(
                eval_policy(
                    policy,
                    DefaultParams.GYM_ENV,
                    DefaultParams.SEED,
                    mean,
                    std,
                )
            )
            np.save(f"./results/{results_filename}", evaluations)


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

    replay_buffer: ReplayBuffer = ReplayBuffer(
        DefaultParams.MAX_TIMESTEPS,
        state_dim,
        action_dim,
        DefaultParams.NORM_EPSILON,
    ).states_from_D4RL_dataset(d4rl.qlearning_dataset(env))

    # Always normalize states
    mean, std = replay_buffer.norm()

    # Init TD3+BC policy with default params
    policy = TD3_BC(
        **DefaultParams.to_td3_bc_kwargs(state_dim, action_dim, max_action)
    )

    train_policy(policy)

    env.close()
    return 0


if __name__ == "__main__":
    main()

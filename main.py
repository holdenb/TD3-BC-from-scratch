import os
import torch
import d4rl
import gym
import numpy as np
from pathlib import Path
from utils import DefaultParams
from replay_buffer import ReplayBuffer
from TD3_BC import TD3_BC


# TODO save models for each env
RESULTS_DIR = "./results"


def set_env_and_seed(env_name: str, seed: int) -> gym.Env:
    env: gym.Env = gym.make(env_name)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    return env


def setup_output_dirs():
    results = Path(RESULTS_DIR)
    if not Path(RESULTS_DIR).exists():
        os.makedirs(results)


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
    eval_env.reset(seed=seed + seed_offset)

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


def train_policy(policy: TD3_BC, replay_buffer: ReplayBuffer, args) -> None:
    results_filename = f"TD3_BC_{args.env}_{args.seed}"

    evaluations = []
    for step in range(int(DefaultParams.MAX_TIMESTEPS)):
        policy.train(replay_buffer, DefaultParams.BATCH_SIZE)
        # Evaluate episode at each interval according to the
        # evaluation frequency
        if (step + 1) % DefaultParams.EVAL_FREQ == 0:
            print(f"Time steps: {step+1}")
            evaluations.append(
                eval_policy(
                    policy,
                    args.env,
                    args.seed,
                    replay_buffer.mean,
                    replay_buffer.std,
                )
            )

            np.save(Path(RESULTS_DIR, results_filename), evaluations)


def extract_state_action_space(env: gym.Env):
    # Environment state/action space
    return (
        env.observation_space.shape[0],
        env.action_space.shape[0],
        float(env.action_space.high[0]),
    )


def main(args) -> None:
    setup_output_dirs()
    env: gym.Env = set_env_and_seed(args.env, args.seed)
    (state_dim, action_dim, max_action) = extract_state_action_space(env)

    replay_buffer: ReplayBuffer = ReplayBuffer(
        int(DefaultParams.MAX_TIMESTEPS),
        state_dim,
        action_dim,
        DefaultParams.NORM_EPSILON,
    ).init_states_from_D4RL_dataset(d4rl.qlearning_dataset(env))

    # Init TD3+BC policy with default params
    policy = TD3_BC(
        **DefaultParams.to_td3_bc_kwargs(state_dim, action_dim, max_action)
    )

    train_policy(policy, replay_buffer, args)

    env.close()
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=DefaultParams.GYM_ENV, type=str)
    parser.add_argument("--seed", default=DefaultParams.SEED, type=int)
    args = parser.parse_args()

    main(args)

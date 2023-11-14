#!/bin/bash

# Suppress failed 'flow' import
export D4RL_SUPPRESS_IMPORT_ERROR=1

envs=(
    "halfcheetah-random-v0"
    "hopper-random-v0"
    "walker2d-random-v0"
    "halfcheetah-medium-v0"
    "hopper-medium-v0"
    "walker2d-medium-v0"
    "halfcheetah-expert-v0"
    "hopper-expert-v0"
    "walker2d-expert-v0"
    "halfcheetah-medium-expert-v0"
    "hopper-medium-expert-v0"
    "walker2d-medium-expert-v0"
    "halfcheetah-medium-replay-v0"
    "hopper-medium-replay-v0"
    "walker2d-medium-replay-v0"
    # TODO upgrade to use v2 environments and compare against paper v0 baselines
    # "halfcheetah-random-v2"
    # "hopper-random-v2"
    # "walker2d-random-v2"
    # "halfcheetah-medium-v2"
    # "hopper-medium-v2"
    # "walker2d-medium-v2"
    # "halfcheetah-expert-v2"
    # "hopper-expert-v2"
    # "walker2d-expert-v2"
    # "halfcheetah-medium-expert-v2"
    # "hopper-medium-expert-v2"
    # "walker2d-medium-expert-v2"
    # "halfcheetah-medium-replay-v2"
    # "hopper-medium-replay-v2"
    # "walker2d-medium-replay-v2"
    # TODO utilize Gymnasium-Robotics and Minari for updated offline RL baselines
    # https://github.com/Farama-Foundation/Minari
    # DR4L is planning to support Minari envs
    # List of supported tasks here: https://github.com/Farama-Foundation/d4rl/wiki/Tasks
    # Note: will need to replicate for Minari
    # https://github.com/Farama-Foundation/D4RL/blob/71a9549f2091accff93eeff68f1f3ab2c0e0a288/d4rl/offline_env.py#L71
)

for ((i = 0; i < 5; i += 1)); do
    for env in ${envs[*]}; do
        python3 main.py \
            --env $env \
            --seed $i
    done
done

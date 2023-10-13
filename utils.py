import torch
import torch.nn as nn
from collections import deque


__DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DefaultParams:
    # Gym/torch/numpy seeds
    SEED = 0
    # Default Gym environment
    GYM_ENV = "hopper-medium-v0"
    # How often we intent to evaluate
    EVAL_FREQ = 5e3
    # Max timesteps to run the Gym environment
    MAX_TIMESTEPS = 1e6
    # Std of Gaussian exploration noise
    EXPLORATION_NOISE = 0.1
    # Default Batch size
    BATCH_SIZE = 256
    # Discount factor
    DISCOUNT_FACTOR = 0.99
    # Target network update rate
    TAU = 0.005
    # Noise added to target policy during critic update
    POLICY_NOISE = 0.2
    # Range to clip target policy noise
    NOISE_CLIP_RANGE = 0.5
    # Frequency of delayed policy updates
    POLICY_UPDATE_FREQ = 2

    # TD3+BC specific parameters
    ALPHA = 2.5
    NORMALIZE = True
    NORM_EPSILON = 1e-3


def move_to_device_ol_list(
    tensors: [torch.FloatTensor] | [torch.Tensor],
) -> None:
    deque(map(lambda x: move_to_device(x), tensors), maxlen=0)


def move_to_device(obj: torch.FloatTensor | torch.Tensor | nn.Module) -> None:
    obj.to(__DEVICE)

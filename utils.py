import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    @staticmethod
    def to_td3_bc_kwargs(state_dim, action_dim, max_action):
        return {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "max_action": max_action,
            "discount": DefaultParams.DISCOUNT_FACTOR,
            "tau": DefaultParams.TAU,
            # TD3
            "policy_noise": DefaultParams.POLICY_NOISE * max_action,
            "noise_clip": DefaultParams.NOISE_CLIP_RANGE * max_action,
            "policy_freq": DefaultParams.POLICY_UPDATE_FREQ,
            # TD3 + BC
            "alpha": DefaultParams.ALPHA,
        }

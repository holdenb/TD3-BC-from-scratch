import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class D4RLDatasetKeys:
    STATE = "observations"
    ACTIONS = "actions"
    NEXT_STATE = "next_observations"
    REWARD = "rewards"
    NOT_DONE = "terminals"


class DefaultParams:
    # Gym/torch/numpy seeds
    SEED = 0
    # Default Gym environment
    GYM_ENV = "hopper-medium-v0"
    # How often we intent to evaluate
    EVAL_FREQ = 5e3
    # Max timesteps to run the Gym environment
    MAX_TIMESTEPS = 1e6
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
    NORM_EPSILON = 1e-3
    ACTOR_OPT_LR = 3e-4
    CRITIC_OPT_LR = 3e-4

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


class StateUtils:
    @staticmethod
    def normalize(
        state, next_state, norm_epsilon=DefaultParams.NORM_EPSILON
    ) -> (float, float):
        mean = state.mean(0, keepdims=True)
        # Epsilon is a small normalization constant. This is commonly
        # used in many deep RL algorithms.
        std = state.std(0, keepdims=True) + norm_epsilon
        state = (state - mean) / std
        next_state = (next_state - mean) / std
        return (mean, std)

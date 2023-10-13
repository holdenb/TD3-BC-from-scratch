import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        # Note: 256 is used here to match actor in original TD3
        # (online) algorithm
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        # acton_dim is used here because the goal is to attempt
        # to choose the appropriate action before being scored
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        state = F.relu(self.l1(state))
        state = F.relu(self.l2(state))
        return self.max_action * torch.tanh(self.l3(state))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        # Note: 256 is used here to match critic in original TD3
        # (online) algorithm
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        state_action_pair = torch.cat([state, action], dim=1)

        q1 = F.relu(self.l1(state_action_pair))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(state_action_pair))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return (q1, q2)

    def q1(self, state, action):
        state_action_pair = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(state_action_pair))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3_BC(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        alpha=2.5,
        actor_opt_lr=3e-4,
        critic_opt_lr=3e-4,
    ):
        # Note: state_dim = env.observation_space
        # Note: action_dim = env.action_space
        # Note: max_action = maximum values for each dimension of the action
        # space
        self.actor = utils.move_to_device(
            Actor(state_dim, action_dim, max_action)
        )
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_opt_lr
        )

        self.critic = utils.move_to_device(Critic(state_dim, action_dim))
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_opt_lr
        )

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha

        self.total_it = 0

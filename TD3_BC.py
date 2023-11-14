import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import DEVICE
from replay_buffer import ReplayBuffer


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        # Note: 400 & 300 are used here to match the original TD3
        # (online) algorithm
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        # acton_dim is used here because the goal is to attempt
        # to choose the appropriate action before being scored
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, state):
        state = F.relu(self.l1(state))
        state = F.relu(self.l2(state))
        return self.max_action * torch.tanh(self.l3(state))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        # Note: 400 & 300 are used here to match the original TD3
        # (online) algorithm
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

    def forward(self, state, action):
        state_action_pair = torch.cat([state, action], dim=1)

        q1 = F.relu(self.l1(state_action_pair))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(state_action_pair))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return (q1, q2)

    def Q1(self, state, action):
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

        # Both actor & critic use Adam optimizer as per the original
        # paper/implementation

        # Init actor & actor target
        self.actor = Actor(state_dim, action_dim, max_action).to(DEVICE)
        # Note: Deepcopy will also put the copy onto the device
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_opt_lr
        )

        # Init critic & critic target
        self.critic = Critic(state_dim, action_dim).to(DEVICE)
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

    def __update_target_model_frozen_params(self, model, model_target):
        for param, target_param in zip(
            model.parameters(), model_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def select_action(self, state):
        # Need to temporarily move to GPU before sending to the actor network
        return (
            self.actor(torch.FloatTensor(state.reshape(1, -1)).to(DEVICE))
            .cpu()
            .data.numpy()
            .flatten()
        )

    def select_next_action(self, next_state, current_action):
        return (
            self.actor_target(next_state) + self.compute_noise(current_action)
        ).clamp(-self.max_action, self.max_action)

    def compute_noise(self, action):
        return (torch.randn_like(action) * self.policy_noise).clamp(
            -self.noise_clip, self.noise_clip
        )

    def update_critic_frozen_params(self):
        self.__update_target_model_frozen_params(
            self.critic, self.critic_target
        )

    def update_actor_frozen_params(self):
        self.__update_target_model_frozen_params(self.actor, self.actor_target)

    def optimize_critic(self, critic_loss):
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def optimize_actor(self, actor_loss):
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    # TODO try upping the batch_size after original experiments
    def train(self, replay_buffer: ReplayBuffer, batch_size=256):
        self.total_it += 1

        # Sample the replay buffer using the mini-batch size,
        # see: https://arxiv.org/abs/1812.02900 - Off-policy batch constrained
        # RL for restricting action space
        (
            state,
            action,
            next_state,
            reward,
            not_done,
        ) = replay_buffer.random_sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            # and compute the target Q value
            target_Q1, target_Q2 = self.critic_target(
                next_state, self.select_next_action(next_state, action)
            )
            target_Q = torch.min(target_Q1, target_Q2)
            # Target Q is shifted by reward value and scaled by gamma to
            # account for a decay in future states. Discount is also scaled
            # by how close we are to the end of the dataset i.e the end of the
            # action space
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss & optimize
        self.optimize_critic(
            F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        )

        # Delayed policy updates: see https://arxiv.org/abs/1802.09477
        # Only update the actor & target critic network every d iterations
        # Default of 2 is used as indicated in the paper
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            # policy (pi)
            pi = self.actor(state)
            Q = self.critic.Q1(state, pi)
            # Note: we do not care about the
            lmbda = self.alpha / Q.abs().mean().detach()

            # Compute actor loss & optimize
            self.optimize_actor(-lmbda * Q.mean() + F.mse_loss(pi, action))

            # Update the frozen target models
            self.update_critic_frozen_params()
            self.update_actor_frozen_params()

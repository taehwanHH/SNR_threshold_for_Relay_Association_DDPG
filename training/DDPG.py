import numpy as np
import torch
import torch.nn as nn

from training.MLP import MultiLayerPerceptron as MLP


class OrnsteinUhlenbeckProcess:
    """
    OU process; The original implementation is provided by minimalRL.
    https://github.com/seungeunrho/minimalRL/blob/master/ddpg.py
    """

    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.1, 0.01, 0.1
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x


class Actor(nn.Module):

    def __init__(self):
        super(Actor, self).__init__()
        self.mlp = MLP(1, 1,
                       num_neurons=[128, 64],
                       hidden_act='ReLU',
                       out_act='Identity')

    def forward(self, state):
        # Action space is [0  40]
        return self.mlp(state).clamp(0.0, 40.0)


class Critic(nn.Module):

    def __init__(self):
        super(Critic, self).__init__()
        self.state_encoder = MLP(1, 64,
                                 num_neurons=[],
                                 out_act='ReLU')  # single layer model
        self.action_encoder = MLP(1, 64,
                                  num_neurons=[],
                                  out_act='ReLU')  # single layer model
        self.q_estimator = MLP(128, 1,
                               num_neurons=[32],
                               hidden_act='ReLU',
                               out_act='Identity')

    def forward(self, x, a):
        emb = torch.cat([self.state_encoder(x), self.action_encoder(a)], dim=-1)
        return self.q_estimator(emb)


class DDPG(nn.Module):

    def __init__(self,
                 critic: nn.Module,
                 critic_target: nn.Module,
                 actor: nn.Module,
                 actor_target: nn.Module,
                 lr_critic: float = 0.0005,
                 lr_actor: float = 0.001,
                 gamma: float = 0.99):
        """
        :param critic: critic network
        :param critic_target: critic network target
        :param actor: actor network
        :param actor_target: actor network target
        :param lr_critic: learning rate of critic
        :param lr_actor: learning rate of actor
        :param gamma:
        """

        super(DDPG, self).__init__()
        self.critic = critic
        self.actor = actor
        self.lr_critic = lr_critic
        self.lr_actor = lr_actor
        self.gamma = gamma

        # setup optimizers
        self.critic_opt = torch.optim.Adam(params=self.critic.parameters(),
                                           lr=lr_critic)

        self.actor_opt = torch.optim.Adam(params=self.actor.parameters(),
                                          lr=lr_actor)

        # setup target networks
        critic_target.load_state_dict(critic.state_dict())
        self.critic_target = critic_target
        actor_target.load_state_dict(actor.state_dict())
        self.actor_target = actor_target

        self.criteria = nn.SmoothL1Loss()

    def get_action(self, state):
        with torch.no_grad():
            a = self.actor(state).clip(-0.2,4)
        return a

    def update(self, state, action, reward, next_state, done):
        s, a, r, ns = state, action, reward, next_state

        # compute critic loss and update the critic parameters
        with torch.no_grad():
            critic_target = r + self.gamma * self.critic_target(ns, self.actor_target(ns)) * (1 - done)
        critic_loss = self.criteria(self.critic(s, a), critic_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # compute actor loss and update the actor parameters
        actor_loss = -self.critic(s, self.actor(s)).mean()  # !!!! Impressively simple
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()


def prepare_training_inputs(sampled_exps, device='cpu'):
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []
    for sampled_exp in sampled_exps:
        states.append(sampled_exp[0])
        actions.append(sampled_exp[1])
        rewards.append(sampled_exp[2])
        next_states.append(sampled_exp[3])
        dones.append(sampled_exp[4])

    states = torch.cat(states, dim=0).float().to(device)
    actions = torch.cat(actions, dim=0).to(device)
    rewards = torch.cat(rewards, dim=0).float().to(device)
    next_states = torch.cat(next_states, dim=0).float().to(device)
    dones = torch.cat(dones, dim=0).float().to(device)
    return states, actions, rewards, next_states, dones

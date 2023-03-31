import torch
import numpy as np
from RNs_env import Relay_Association_ENV

from DDPG import DDPG, Actor, Critic, prepare_training_inputs
from DDPG import OrnsteinUhlenbeckProcess as OUProcess
from memory import ReplayMemory
from train_utils import to_tensor
from target_update import soft_update

env = Relay_Association_ENV
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# HYPERPARAMETERS
lr_actor = 0.005
lr_critic = 0.001
gamma = 0.99
batch_size = 256
memory_size = 50000
tau = 0.001 # polyak parameter for soft target update
sampling_only_until = 2000

# SETTING
actor, actor_target = Actor(), Actor()
critic, critic_target = Critic(), Critic()

agent = DDPG(critic=critic,
             critic_target=critic_target,
             actor=actor,
             actor_target=actor_target).to(DEVICE)

memory = ReplayMemory(memory_size)

total_eps = 200
print_every = 10

# EPISODE
for n_epi in range(total_eps):
    ou_noise = OUProcess(mu=np.zeros(1))
    s = env.reset()
    cum_r = 0

    while True:
        s = to_tensor(s, size=(1, 3)).to(DEVICE)
        a = agent.get_action(s).cpu().numpy() + ou_noise()[0]
        ns, r, done, info = env.step(a)

        experience = (s,
                      torch.tensor(a).view(1, 1),
                      torch.tensor(r).view(1, 1),
                      torch.tensor(ns).view(1, 3),
                      torch.tensor(done).view(1, 1))
        memory.push(experience)

        s = ns
        cum_r += r

        if len(memory) >= sampling_only_until:
            # train agent
            sampled_exps = memory.sample(batch_size)
            sampled_exps = prepare_training_inputs(sampled_exps, device=DEVICE)
            agent.update(*sampled_exps)
            # update target networks
            soft_update(agent.actor, agent.actor_target, tau)
            soft_update(agent.critic, agent.critic_target, tau)

        if done:
            break

    if n_epi % print_every == 0:
        msg = (n_epi, cum_r) # ~ -100 cumulative reward = "solved"
        print("Episode : {} | Cumulative Reward : {} |".format(*msg))

torch.save(agent.state_dict(), 'ddpg_cartpole_user_trained.ptb')

import torch
import numpy as np
# from env.RNs_env import CommunicationEnv
from env.temp_env import CommunicationEnv
from training.DDPG import DDPG, Actor, Critic, prepare_training_inputs
from training.DDPG import OrnsteinUhlenbeckProcess as OUProcess
from training.memory import ReplayMemory
from training.train_utils import to_tensor
from training.target_update import soft_update

env = CommunicationEnv()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# HYPERPARAMETER
lr_actor = 0.01 ## 0.005
lr_critic = 0.005 ## 0.001
gamma = 0.9 ## 0.99
batch_size = 16 ## 256
memory_size = 50000
tau = 0.01 # polyak parameter for soft target update 0.001
sampling_only_until = 20 ## 2000

# SETTING
actor, actor_target = Actor(), Actor()
critic, critic_target = Critic(), Critic()

agent = DDPG(critic=critic,
             critic_target=critic_target,
             actor=actor,
             actor_target=actor_target).to(DEVICE)

memory = ReplayMemory(memory_size)

total_eps = 200
print_every = 1 ## 10

# EPISODE
for n_epi in range(total_eps):
    ou_noise = OUProcess(mu=np.zeros(1))
    s = env.reset()
    cum_r = 0
    while True:
        tmp_s = s
        env.eta = tmp_s.reshape((1,1))
        s = to_tensor(s, size=(1, 1)).to(DEVICE)
        # env.eta = s.reshape((1,1)) ##
        a = agent.get_action(s).cpu().numpy() + ou_noise()[0]
        ns, r, done, info = env.step(a)

        experience = (s,
                      torch.tensor(a).view(1, 1),
                      torch.tensor(r).view(1, 1),
                      torch.tensor(ns).view(1, 1),
                      torch.tensor(done).view(1, 1))
        memory.push(experience)

        temp_s = s
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


        print(experience)
        if done:
            break


    if n_epi % print_every == 0:
        msg = (n_epi, cum_r,temp_s ) # ~ -100 cumulative reward = "solved"
        print("Episode : {} | Cumulative Reward : {} | eta : {}".format(*msg))

# torch.save(agent.state_dict(), 'ddpg_cartpole_user_trained.ptb')

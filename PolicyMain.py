import argparse
from itertools import count

import os, sys, random
import numpy as np

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from tensorboardX import SummaryWriter
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from Utils.Utils import BuildState,ActionRestrict

class Replay_buffer():
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''
    def __init__(self, capacity):
        self.storage = []
        self.max_size = capacity
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, x):
        x = torch.tensor(x).to(device)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action, args):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.replay_buffer = Replay_buffer(args.capacity)
        self.writer = SummaryWriter(directory)

        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def select_action(self, state, avail_a_n, epsilon, current_epslion):
        eps= current_epslion/epsilon
        avail_a_n = torch.tensor(avail_a_n, dtype=torch.int)  # avail_a_n.shape=(N, action_dim)
        avail_a_n_itr = torch.chunk(avail_a_n, chunks=1, dim=0)
        if np.random.uniform() <= eps:  # epsilon-greedy
            a_n = [np.random.choice(np.nonzero(avail_a).flatten()) for avail_a in avail_a_n]
        else:
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self):

        for it in range(args.update_iteration):
            # Sample replay buffer
            x, y, u, r, d = self.replay_buffer.sample(args.batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(1 - d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (done * args.gamma * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)
            self.writer.add_scalar('Loss/critic_loss', critic_loss, global_step=self.num_critic_update_iteration)
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()
            self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1

    def save(self):
        torch.save(self.actor.state_dict(), directory + 'actor.pth')
        torch.save(self.critic.state_dict(), directory + 'critic.pth')
        # print("====================================")
        # print("Model has been saved...")
        # print("====================================")

    def load(self):
        self.actor.load_state_dict(torch.load(directory + 'actor.pth'))
        self.critic.load_state_dict(torch.load(directory + 'critic.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")

def action_transfer_Unity(actions):
    unity_action=[]
    for i in actions:
        if i == 0:
            unity_action.append([2,0])
        if i == 1:
            unity_action.append([-2,0])
        if i == 2:
            unity_action.append([0,2])
        if i == 3:
            unity_action.append([0,-2])
        if i == 4:
            unity_action.append([0, 0])
    return np.array(unity_action)

def main(args, env):
    agent = DDPG(state_dim, action_dim, max_action, args)
    ep_r = 0
    if args.mode == 'test':
        agent.load()
        for i in range(args.test_iteration):
            state = env.reset()
            for t in count():
                action = agent.select_action(state, avail_a, epsilon)
                next_state, reward, done, info = env.step(np.float32(action))
                ep_r += reward
                env.render()
                if done or t >= args.max_length_of_trajectory:
                    print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_r, t))
                    ep_r = 0
                    break
                state = next_state

    elif args.mode == 'train':
        env.reset()
        behavior_names = list(env.behavior_specs.keys())
        behavior_value = list(env.behavior_specs.values())
        current_epslion = args.epslion
        if args.load: agent.load()
        total_step = 0
        for i in range(args.max_episode):
            total_reward = 0
            step = 0
            env.reset()
            Groups_done =  [False] * args.N
            behavior_names = list(env.behavior_specs.keys())
            behavior_value = list(env.behavior_specs.values())
            DecisionSteps, TerminalSteps = env.get_steps(behavior_names[0])
            ObsN = DecisionSteps.obs[0]
            State, Exit_index, Exit_Distance, Fire_Distance = BuildState(ObsN, args.N)
            avail_a_n = np.ones((args.N, action_dim))
            for i in range(args.N):
                avail_a_n[i][4] = 0
            for t in count():
                avail_a_n = ActionRestrict(ObsN, avail_a_n)
                action = agent.select_action(State, avail_a_n, args.epslion, current_epslion)
                #Update strategy
                if (current_epslion/args.epslion) > 0.1:
                    current_epslion-=1
                else:
                    current_epslion = 0.1 * args.epslion
                continuous_actions = action_transfer_Unity(action)
                discrete_actions = None
                actions = ActionTuple(continuous_actions, discrete_actions)
                env.set_actions(self.behavior_names[0], actions)
                env.step()

                DecisionSteps_next, TerminalSteps_next = self.env.get_steps(self.behavior_names[0])
                ObsN_next = DecisionSteps_next.obs[0]
                State_next, Exit_index, Exit_Distance_Next, Fire_Distance_Next = BuildState( ObsN_next, args.N)

                reward = (Exit_Distance - Exit_Distance_Next) * 2
                reward_list.append(reward)

                # judge individual agent arrive at exits
                for i in range(self.args.N):
                    if Exit_Distance_Next[i] < 6.0:
                        if Groups_done[i] == False:
                            reward[i] += 5
                            avail_a_n[i][0:4] = 0
                            avail_a_n[i][4] = 1
                            Groups_done[i] = True
                        else:
                            reward[i] += 0
                agent.replay_buffer.push((state, next_state, action, reward, np.float(Groups_done)))
                state = next_state
                if done:
                    break
                step += 1
                total_reward += reward
            total_step += step + 1
            print("Total T:{} Episode: \t{} Total Reward: \t{:0.2f}".format(total_step, i, total_reward))
            agent.update()
            # "Total T: %d Episode Num: %d Episode T: %d Reward: %f

            if i % args.log_interval == 0:
                agent.save()
    else:
        raise NameError("mode wrong!!!")

if __name__ == '__main__':

    '''
    Implementation of Deep Deterministic Policy Gradients (DDPG) with pytorch 
    riginal paper: https://arxiv.org/abs/1509.02971
    Not the author's implementation !
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', type=str)  # mode = 'train' or 'test'
    # OpenAI gym environment name, # ['BipedalWalker-v2', 'Pendulum-v0'] or any continuous environment
    # Note that DDPG is feasible about hyper-parameters.
    # You should fine-tuning if you change to another environment.
    parser.add_argument("--env_name", default="Pendulum-v0")
    parser.add_argument('--tau', default=0.005, type=float)  # target smoothing coefficient
    parser.add_argument('--target_update_interval', default=1, type=int)
    parser.add_argument('--N', default=1, type=int)
    parser.add_argument('--test_iteration', default=10, type=int)

    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--gamma', default=0.99, type=int)  # discounted factor
    parser.add_argument('--capacity', default=1000000, type=int)  # replay buffer size
    parser.add_argument('--batch_size', default=100, type=int)  # mini batch size
    parser.add_argument('--seed', default=False, type=bool)
    parser.add_argument('--random_seed', default=9527, type=int)
    parser.add_argument('--epslion',default=5000, type =int)
    # optional parameters
    parser.add_argument('--sample_frequency', default=2000, type=int)
    parser.add_argument('--render', default=False, type=bool)  # show UI or not
    parser.add_argument('--log_interval', default=50, type=int)  #
    parser.add_argument('--load', default=False, type=bool)  # load model
    parser.add_argument('--render_interval', default=100, type=int)  # after render_interval, the env.render() will work
    parser.add_argument('--exploration_noise', default=0.1, type=float)
    parser.add_argument('--max_episode', default=100000, type=int)  # num of games
    parser.add_argument('--print_log', default=5, type=int)
    parser.add_argument('--update_iteration', default=200, type=int)
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    script_name = os.path.basename(__file__)
    # env = gym.make(args.env_name)
    env_name = './EnvironmentOneAgentRandom/UnityEnvironment'
    channel = EngineConfigurationChannel()
    env = UnityEnvironment(file_name=env_name, seed=1, side_channels=[channel])
    channel.set_configuration_parameters(time_scale=10.0)

    if args.seed:
        # env.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)

    state_dim = 4
    action_dim = 5
    max_action = 1
    min_Val = torch.tensor(1e-7).float().to(device)  # min value

    directory = './exp' + script_name + args.env_name + './'

    main(args, env)
import argparse
import pickle
from collections import namedtuple
from itertools import count

import os, time
import numpy as np
import matplotlib.pyplot as plt
from mlagents_envs.environment import UnityEnvironment,ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter
from Utils.Utils import BuildState,ArtificialPotentialField,ActionRestrict
from Aggregattion.aggregation import aggragate,aggragate_ppo, reward_aggragate_ppo, personalized_aggragate_ppo
from Utils.PathRecorder import Pathrecord
# Parameters
gamma = 0.99
render = False
seed = 1
log_interval = 10
env_name = './Environment32AgentRandom/UnityEnvironment'
writer = SummaryWriter('./Result/ppo_result_8_agent_episode96_5e-4_pps')
channel = EngineConfigurationChannel()
env = UnityEnvironment(file_name=env_name, seed=1, side_channels=[channel])
channel.set_configuration_parameters(time_scale=10.0)
env.reset()
# Load environment of configuration
behavior_names = list(env.behavior_specs.keys())
behavior_value = list(env.behavior_specs.values())
DecisionSteps, TerminalSteps = env.get_steps(behavior_names[0])
# The number of agents
num_agent = len(DecisionSteps.agent_id)
num_observation = behavior_value[0].observation_specs[0].shape[0]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Heuristic observation dim
# num_observation = 4
num_state = 5
# Action space contains up, down, rigth, left 2*2 + stop 1
num_action = behavior_value[0].action_spec[0] * 2 + 1
# self.args.state_dim = self.args.obs_dim * self.args.N  # The dimensions of global state space
num_action = 5
num_episode = 96
evaluate_times = 32
# env = gym.make('CartPole-v0').unwrapped
# num_state = env.observation_space.shape[0]
# num_action = env.action_space.n
torch.manual_seed(seed)
# env.seed(seed)
Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])

def calculate_reward(reward, Exit_Distance_Next, Groups_done):
    # Groups_done = [False] * num_agent
    for i in range(num_agent):
        if Exit_Distance_Next[i] < 6.0:
            Groups_done[i] = True
            # reward += 5
        # else:
        #     Groups_done[i] = False
    done = all(Groups_done)
    return reward,Groups_done,done

def action_transfer_Unity(actions):
    unity_action = []
    for i in actions:
        if i == 0:
            unity_action.append([2, 0])
        if i == 1:
            unity_action.append([-2, 0])
        if i == 2:
            unity_action.append([0, 2])
        if i == 3:
            unity_action.append([0, -2])
        if i == 4:
            unity_action.append([0, 0])

    return np.array(unity_action)

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_state, 32)
        self.action_head = nn.Linear(32, num_action)
        self.soft_layer=nn.Softmax()
    def forward(self, x):
        # x = F.layer_norm(x)
        x = F.relu(self.fc1(x))
        self.test_a=x
        x =  F.tanh(self.action_head(x))
        self.test_b=x
        action_prob = F.softmax(x,dim=1)
        self.test_c=action_prob
        return action_prob


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_state, 100)
        self.state_value = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        value = self.state_value(x)
        return value


class PPO():
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_update_time = 1
    buffer_capacity = 1000
    batch_size = 32

    def __init__(self):
        super(PPO, self).__init__()
        self.actor_net = Actor()
        self.actor_net= self.actor_net.to(device)
        self.critic_net = Critic()
        self.critic_net=self.critic_net.to(device)
        self.buffer = []
        self.counter = 0
        self.training_step = 0


        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), 5e-4)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), 3e-3)
        if not os.path.exists('../param'):
            os.makedirs('../param/net_param')
            os.makedirs('../param/img')

    def select_action(self, state, avail_a_n):
        try:
            state = torch.from_numpy(state).float().unsqueeze(0)
            state = state.to(device)
            with torch.no_grad():
                action_prob = self.actor_net(state)
            avail_a_n = torch.tensor(avail_a_n, dtype=torch.int).reshape(1,-1) # avail_a_n.shape=(N, action_dim)
            avail_a_n_itr = torch.chunk(avail_a_n, chunks=1, dim=0)
            action_prob[avail_a_n == 0] = 0
            c = Categorical(action_prob)
            action = c.sample()
            return action.item(), action_prob[:, action.item()].item()
        except ValueError:
            print("May be no choice")


    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    def save_param(self):
        torch.save(self.actor_net.state_dict(), '../param/net_param/actor_net' + str(time.time())[:10], +'.pkl')
        torch.save(self.critic_net.state_dict(), '../param/net_param/critic_net' + str(time.time())[:10], +'.pkl')

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1

    def update(self, i_ep, agent_index):
        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float).to(device)
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1, 1).to(device)
        reward = [t.reward for t in self.buffer]
        # update: don't need next_state
        # reward = torch.tensor([t.reward for t in self.buffer], dtype=torch.float).view(-1, 1)
        # next_state = torch.tensor([t.next_state for t in self.buffer], dtype=torch.float)
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1).to(device)

        R = 0
        Gt = []
        for r in reward[::-1]:
            R = r + gamma * R
            Gt.insert(0, R)
        Gt = torch.tensor(Gt, dtype=torch.float)
        # print("The agent is updateing....")
        for i in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False):
                # if self.training_step % 1000 == 0:
                #     print('I_ep {} ，train {} times'.format(i_ep, self.training_step))
                # with torch.no_grad():
                Gt_index = Gt[index].view(-1, 1).to(device)
                V = self.critic_net(state[index])
                delta = Gt_index - V
                advantage = delta.detach()
                # epoch iteration, PPO core!!!
                action_prob = self.actor_net(state[index]).gather(1, action[index])  # new policy

                ratio = (action_prob / old_action_log_prob[index])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                # update actor network
                action_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent
                writer.add_scalar('loss/agent_{}_action_loss'.format(agent_index), action_loss, global_step=self.training_step)
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # update critic network
                value_loss = F.mse_loss(Gt_index, V)
                writer.add_scalar('loss/agent_{}_value_loss'.format(agent_index), value_loss, global_step=self.training_step)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()
                self.training_step += 1

        del self.buffer[:]  # clear experience

def evaluate(agents,i_epoch):
    # evaluate_times=32
    accumulate_reward = np.zeros(num_agent)
    evacuation_times = 0
    evacuation_nums=0
    evacuation_total_steps=0
    paths=[]

    for evaluate_index in range(evaluate_times):
        paths_oneepisode=[]
        for i in range(num_agent):
            paths_oneepisode.append([])
        env.reset()
        active = np.ones(num_agent)
        evacuation_individual_step=np.zeros(num_agent)
        avail_a_n = np.ones((num_agent, num_action))
        for z in range(num_agent):
            avail_a_n[z][4] = 0
        Groups_done = [False] * num_agent
        for t in count():
            DecisionSteps, TerminalSteps = env.get_steps(behavior_names[0])
            ObsN = DecisionSteps.obs[0]
            for index in range(num_agent):
                paths_oneepisode[index].append(ObsN[index][6])
                paths_oneepisode[index].append(ObsN[index][8])
            State, Exit_index, Exit_Distance, Fire_Distance = BuildState(ObsN, num_agent,t)
            # State = State.reshape(-1)
            avail_a_n = ActionRestrict(ObsN, avail_a_n, Groups_done)
            actions = []
            actions_prob = []
            for j in range(num_agent):
                action, action_prob = agents[j].select_action(State[j], avail_a_n[j])
                actions.append(action)
                actions_prob.append(action_prob)
            continuous_actions = action_transfer_Unity(actions)
            discrete_actions = None
            actions = ActionTuple(continuous_actions, discrete_actions)
            # Execute Action
            env.set_actions(behavior_names[0], actions)
            env.step()
            # Get Next State
            DecisionSteps_next, TerminalSteps_next = env.get_steps(behavior_names[0])
            ObsN_next = DecisionSteps_next.obs[0]
            State_next, Exit_index, Exit_Distance_Next, Fire_Distance_Next = BuildState(ObsN_next, num_agent,t)
            # State_next = State_next.reshape(-1)
            # next_state, reward, done, _ = env.step(action)
            # Calculate Reward
            reward = (Exit_Distance - Exit_Distance_Next)
            reward, Groups_done, done = calculate_reward(reward, Exit_Distance_Next,Groups_done )

            for k in range(num_agent):
                if Groups_done[k]==True and active[k]==1:
                    evacuation_individual_step[k]=t
                    active[k]=0

            accumulate_reward += reward
            State = State_next
            if done:
                evacuation_times+=1
                # print("Evcuation Success")
                break
            if t > num_episode:
                done = True
            if done:
                break
        for cal in range(num_agent):
            if Groups_done[cal]:
                evacuation_nums+=1
                evacuation_total_steps += evacuation_individual_step[cal]
        for k in range(num_agent):
            paths_oneepisode[k].append(Groups_done[k])
        paths.extend(paths_oneepisode)
    Pathrecord(paths,num_agent,i_epoch)
    print("Evacuation Num:", evacuation_nums, "Evacuation Rate:", evacuation_nums/(evaluate_times*num_agent),
          "Evacuation total Rate:",evacuation_times/evaluate_times," Accumulate Reawrd:", accumulate_reward/evaluate_times,
          "Evacuation Average Step:",evacuation_total_steps/evacuation_nums)
    writer.add_scalar('evacuation_rate',evacuation_nums/(evaluate_times*num_agent),global_step=i_epoch)
    writer.add_scalar('num', evacuation_nums, global_step=i_epoch)
    writer.add_scalar('evacuation average step', evacuation_total_steps/evacuation_nums,global_step=i_epoch )
    for i in range(num_agent):
        writer.add_scalar('Accumulate_Reward/agent{}_accumulate_reward'.format(i), accumulate_reward[i]/evaluate_times, global_step=i_epoch)

def evaluate_escapetime(agents,i_epoch):
    # evaluate_times=32
    accumulate_reward = 0
    evacuation_times = 0
    evacuation_nums=0
    evacuation_total_time=0
    for i in range(evaluate_times):
        env.reset()
        avail_a_n = np.ones((num_agent, num_action))
        Groups_done = [False] * num_agent
        for t in count():
            DecisionSteps, TerminalSteps = env.get_steps(behavior_names[0])
            ObsN = DecisionSteps.obs[0]
            State, Exit_index, Exit_Distance, Fire_Distance = BuildState(ObsN, num_agent,t)
            # State = State.reshape(-1)
            avail_a_n = ActionRestrict(ObsN, avail_a_n, Groups_done)
            actions = []
            actions_prob = []
            for j in range(num_agent):
                action, action_prob = agents[j].select_action(State[j], avail_a_n[j])
                actions.append(action)
                actions_prob.append(action_prob)
            continuous_actions = action_transfer_Unity(actions)
            discrete_actions = None
            actions = ActionTuple(continuous_actions, discrete_actions)
            # Execute Action
            env.set_actions(behavior_names[0], actions)
            env.step()
            # Get Next State
            DecisionSteps_next, TerminalSteps_next = env.get_steps(behavior_names[0])
            ObsN_next = DecisionSteps_next.obs[0]
            State_next, Exit_index, Exit_Distance_Next, Fire_Distance_Next = BuildState(ObsN_next, num_agent,t)
            # State_next = State_next.reshape(-1)
            # next_state, reward, done, _ = env.step(action)
            # Calculate Reward
            reward = (Exit_Distance - Exit_Distance_Next)
            reward, Groups_done, done = calculate_reward(reward, Exit_Distance_Next,Groups_done )
            accumulate_reward += reward
            State = State_next
            if done:
                evacuation_times+=1
                break
            if done:
                break
        evacuation_total_time+=t
        for i in Groups_done:
            if i: evacuation_nums+=1
    print("Evacuation Time", evacuation_total_time/evaluate_times,"Evacuation Num:", evacuation_nums, "Evacuation Num:", evacuation_nums/(evaluate_times*num_agent), "Evacuation total Rate:",evacuation_times/evaluate_times," Accumulate Reawrd:", accumulate_reward/evaluate_times)
    writer.add_scalar('evacuation_rate',evacuation_nums/(evaluate_times*num_agent),global_step=i_epoch)
    writer.add_scalar('num', evacuation_nums, global_step=i_epoch)
    for i in range(num_agent):
        writer.add_scalar('Accumulate_Reward/agent{}_accumulate_reward'.format(i), accumulate_reward[i], global_step=i_epoch)

def main():
    agents = []
    for i in range(num_agent):
        agents.append(PPO())
    aggragate_model=PPO()
    for i_epoch in range(5000):
        Exit_index_list=[]
        env.reset()
        accumulate_reward = np.zeros(num_agent)
        avail_a_n = np.ones((num_agent, num_action))
        if i_epoch % 20 ==0:
            # print(agents[0].training_step)
            # evaluate_escapetime(agents,i_epoch)
            evaluate(agents, i_epoch)
        Groups_done = [False] * num_agent
        for i in range(num_agent):
            avail_a_n[i][4] = 0
        for t in count():
            # Get Current State
            DecisionSteps, TerminalSteps = env.get_steps(behavior_names[0])
            ObsN = DecisionSteps.obs[0]
            State, Exit_index, Exit_Distance, Fire_Distance = BuildState(ObsN, num_agent,t)
            # Exit_index_list.append(Exit_index)
            # State = State.reshape(-1)
            avail_a_n = ActionRestrict(ObsN, avail_a_n, Groups_done)
            actions = []
            actions_prob = []

            for j in range(num_agent):
                action, action_prob = agents[j].select_action(State[j], avail_a_n[j])
                actions.append(action)
                actions_prob.append(action_prob)
            continuous_actions = action_transfer_Unity(actions)
            discrete_actions = None
            unity_actions = ActionTuple(continuous_actions, discrete_actions)
                # Execute Action
            env.set_actions(behavior_names[0], unity_actions)
            env.step()
            # Get Next State
            DecisionSteps_next, TerminalSteps_next = env.get_steps(behavior_names[0])
            ObsN_next = DecisionSteps_next.obs[0]
            State_next, Exit_index, Exit_Distance_Next, Fire_Distance_Next = BuildState(ObsN_next, num_agent, t)
            # Exit_index_list.append(int(Exit_index))
            # State_next =State_next.reshape(-1)
            # next_state, reward, done, _ = env.step(action)
            #Calculate Reward
            reward = (Exit_Distance - Exit_Distance_Next)
            reward, Groups_done, done = calculate_reward(reward,Exit_Distance_Next,Groups_done)
            accumulate_reward += reward
            if done:
                # print("Evacuation Successfully!")
                break
            for i in range(num_agent):
                trans = Transition(State[i], actions[i], actions_prob[i], reward[i], State_next[i])
                agents[i].store_transition(trans)

            State = State_next

            for i in range(num_agent):
                if len(agents[i].buffer) >= agents[i].batch_size: agents[i].update(i_epoch,i)

            if t > num_episode:
                # print("Exits_index_list", Exit_index_list)
                done = True
                # print("Evacuation False! Still in Fire Hazard, num:",i_epoch)
            if done:
                if len(agents[i].buffer) >= agents[i].batch_size: agents[i].update(i_epoch,i)
                # writer.add_scalar('Accumulate_Reward/reward', t, global_step=i_epoch)
                break
        # if i_epoch % 40 == 0:
        #     # print("AFL")
        #     # w = torch.tensor(accumulate_reward)
        #     weights= torch.softmax(torch.tensor(accumulate_reward), dim=0)
        #     # aggragate_net = aggragate_ppo(num_agent, agents, aggragate_model)
        #     # aggragate_net = reward_aggragate_ppo(num_agent, agents, aggragate_model, weights)
        #     aggragate_net = personalized_aggragate_ppo(num_agent, agents, aggragate_model, 4)
        #     for i in range(num_agent):
        #         agents[i].actor_net.load_state_dict(aggragate_net.state_dict())
        #     print("!!!!!After aggregate---result!!!!!")

if __name__ == '__main__':
    main()
    print("end")
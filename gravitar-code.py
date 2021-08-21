# -*- coding: utf-8 -*-
#  [1]  Code for this task is based on the sample code supplied by the coordinator:
#       Willcocks, Chris
#       https://gist.github.com/cwkx/e674744c30296b238bd70214dff1962e#file-rl-assignment-ipynb
#
#  [2]  https://github.com/higgsfield/RL-Adventure/blob/master/4.prioritized%20dqn.ipynb

# run in notebooks to see full features

import gym
import math
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Prioritised buffer adapted from [2]

# Priority experience replay used to assign priority to transitions in memory.
class NaivePrioritizedBuffer(object):
    def __init__(self, max_size, probability_alpha=0.6):
        self.buffer = []
        self.position = 0
        self.max_size = max_size
        self.probability_alpha = probability_alpha
        self.priorities = np.zeros((max_size,), dtype=np.float32)

    def put(self, transition):

        max_priority = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.max_size:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.max_size

    def sample(self, batch_amount, beta=0.4):
        if len(self.buffer) == self.max_size:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]

        probabilities = priorities ** self.probability_alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_amount, p=probabilities)
        mini_batch = [self.buffer[idx] for idx in indices]

        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            done_mask_lst.append(done_mask)

        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        return torch.tensor(s_lst, dtype=torch.float).to(device), torch.tensor(a_lst).to(device), \
               torch.tensor(r_lst).to(device), torch.tensor(s_prime_lst, dtype=torch.float).to(device), \
               torch.tensor(done_mask_lst).to(device), indices, torch.tensor(weights, dtype=torch.float).to(device)

    def update_priorities(self, batch_indices, batch_priorities):
        for i in range(len(batch_indices)):
            id = batch_indices[i]
            priority = batch_priorities[i]
            self.priorities[id] = priority

    def size(self):
        return len(self.buffer)


# DQ Network incorporated from [2] following individual experimentation with using the code from my Discriminator
# as a DQN in this code. Despite initial success, it became apparent the model was not learning as epsilon decreased and
# episode scores fell to zero. I wish I had more time to experiment and invent my own DQ
# however I had to resort to using this simple convolutional model.
class CnnDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CnnDQN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions

        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

    def sample_action(self, state, epsilon):
        if random.random() > epsilon:
            q_value = self.forward(state)
            action = q_value.max(1)[1].data[0]
        else:
            action = random.randrange(env.action_space.n)
        return action


# Vital code for use of convolutions on observations data in DQN as without this transformation,
# the atari screen dimensions do not well suit convolutions.
class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                dtype=np.uint8)

    def observation(self, observation):
        return np.swapaxes(observation, 2, 0)


def wrap_pytorch(env):
    return ImageToPyTorch(env)


# uses variable naming conventions defined in [1] but loss function of [2]
def train(q, q_target, memory, optimizer, beta):
    for i in range(10):
        s, a, r, s_prime, done_mask, indices, weights = memory.sample(batch_size, beta)
        q_out = q(s)
        q_a = q_out.gather(1, a.unsqueeze(1)).squeeze(1)
        max_q_prime = q_target(s_prime).max(1)[0]
        # calculates predicted score
        target = r + gamma * max_q_prime * (1 - done_mask)
        # uses prediction to calculate loss
        loss = (q_a - target.detach()).pow(2) * weights
        # uses loss to calculate priorities
        priorities = loss + 1e-5
        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        memory.update_priorities(indices, priorities.data.cpu().numpy())
        optimizer.step()
    return loss

# calculates beta for every iteration as described in [2] to influence memory sampling
def beta_by_iter(iter):
    return min(1.0, beta_start + iter * (1.0 - beta_start) / beta_frames)


# hyperparameters using conventions laid out in [1]
video_every = 10
print_every = 10
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 4000
learning_rate = 0.00025
buffer_limit = 75000
batch_size = 32
gamma = 0.99
initial_buffer = 2000
beta_start = 0.4
beta_frames = 667
ndf = 32

env = gym.make('Gravitar-v0')
env = gym.wrappers.Monitor(env, "./video", video_callable=lambda episode_id: (episode_id % video_every) == 0,
                           force=True)
env = wrap_pytorch(env)
seed = 742
torch.manual_seed(seed)
env.seed(seed)
random.seed(seed)
np.random.seed(seed)
env.action_space.seed(seed)
# initialises both QNs, one to track the predicted score and space, the other for the current space
q = CnnDQN(env.observation_space.shape, env.action_space.n).to(device)
q_target = CnnDQN(env.observation_space.shape, env.action_space.n).to(device)
q_target.load_state_dict(q.state_dict())
# initialises memory buffer
memory = NaivePrioritizedBuffer(buffer_limit)

losses = []
all_rewards = []
score = 0.0
marking = []
optimizer = optim.Adam(q.parameters(), lr=learning_rate)

# arbitrary number of episodes
for n_episode in range(int(20000)):
    # calculate new epsilon to reduce likelihood of random actions
    epsilon = epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * n_episode / epsilon_decay)
    s = env.reset()
    done = False
    score = 0.0

    while True:
        # while the episode runs, repeatedly choose actions and log results in memory
        a = q.sample_action(torch.from_numpy(s).float().unsqueeze(0).to(device), epsilon)
        s_prime, r, done, info = env.step(a)
        done_mask = 0.0 if done else 1.0
        memory.put((s, a, r, s_prime, done_mask))
        s = s_prime
        score += r
        if done:
            break

    if memory.size() > initial_buffer:
        b = beta_by_iter(n_episode)
        l = train(q, q_target, memory, optimizer, b)

    # do not change lines 44-48 here, they are for marking the submission log
    marking.append(score)
    if n_episode % 100 == 0:
        print("marking, episode: {}, score: {:.1f}, mean_score: {:.2f}, std_score: {:.2f}".format(
            n_episode, score, np.array(marking).mean(), np.array(marking).std()))
        marking = []

    if n_episode % print_every == 0 and n_episode != 0:
        q_target.load_state_dict(q.state_dict())
        print("episode: {}, score: {:.1f}, epsilon: {:.2f}".format(n_episode, score, epsilon))

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import *
env = gym.make('MountainCar-v0', render_mode='rgb_array',goal_velocity=0.1)
class ReinforceAgent(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ReinforceAgent, self).__init__()
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, out_dim)
        self.log_std = nn.Parameter(torch.zeros(out_dim))
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def get_action(self, state):
        state = torch.tensor(state[0], dtype=torch.float32).unsqueeze(0)
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)
def train(env,agent,num_episodes=1000, gamma=0.99):
    optimizer = optim.Adam(agent.parameters(), lr=0.001)
    for episode in range(num_episodes):
        state = env.reset()
        log_probs = []
        rewards = []
        done = False
        while not done:
            action,log_prob = agent.get_action(state)
            state, reward, done, info,_= env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            
        returns = []
        R = 0
        for r in rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        log_probs = torch.tensor(log_probs)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        policy_gradient = -log_probs * returns
        optimizer.zero_grad()
        policy_gradient.sum().backward()
        optimizer.step()
        print(f"Episode {episode+1}\tTotal Reward: {sum(rewards):.2f}")
def evaluate(env, agent, num_episodes=10):
    for episode in range(num_episodes):
        state,_ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = agent.get_action(state)
            state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            env.render()
        print(f"Episode {episode+1}\tTotal Reward: {total_reward:.2f}")
    env.close()
print(env.reset())
print(env.step(0))
agent = ReinforceAgent(env.observation_space.shape[0], env.action_space.n)
print("starting training..../n")
train(env, agent)
evaluate(env, agent)

from __future__ import annotations

import random
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

import gymnasium as gym

class Policy(nn.Module):
    def __init__(self, obs_dim:int, act_dim:int, hidden_size=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim)
        )
        
    def forward(self, x):
        logits = self.model(x.float())
        return logits
class REINFORCE:
    def __init__(self,obs_dim:int, act_dim:int, lr=0.001, gamma=0.99):
        self.policy = Policy(obs_dim, act_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.eps = 1
        
        self.probs = []
        self.rewards = []
    def sample_action(self, state: np.ndarray) -> int:
        state = torch.tensor(state, dtype=torch.float32)
        logits = self.policy(state)
        action_dist = torch.distributions.Categorical(logits=logits)
        action = action_dist.sample()
        
        self.probs.append(action_dist.log_prob(action))
        return action.item()
    def update_policy(self):
        returns = []
        R = 0
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        if self.probs != []:
            probs = torch.stack(self.probs)
            loss = -(probs * returns).sum()
            loss.backward()
        self.optimizer.zero_grad()
        
        self.optimizer.step()
    
        self.probs = []
        self.rewards = []
env = gym.make('MountainCar-v0', render_mode=None)
wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

agent = REINFORCE(obs_dim, act_dim, lr=0.001, gamma=0.99)
total_episodes = 5000
for episode in range(total_episodes):
    obs,info = wrapped_env.reset()
    done = False
    while not done:
        if random.random() < agent.eps:
            action = env.action_space.sample()
        else:
            action = agent.sample_action(obs)
        obs, reward, terminated, truncated ,info = wrapped_env.step(action)
        agent.rewards.append(reward)
        done = truncated or terminated
    agent.update_policy()
    if episode % 1000 == 0:
        avg_reward = int(np.mean(wrapped_env.return_queue))
        print(f"Episode: {episode}, Average Reward: {avg_reward}")
def evaluate(agent,num_episodes=10):
    env = gym.make('MountainCar-v0', render_mode='human')
    for episode in range(num_episodes):
        obs,info = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.sample_action(obs)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
        print(f"Episode {episode+1}\tTotal Reward: {total_reward:.2f}")
evaluate(agent)
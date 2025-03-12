import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
def run(episodes=1000, isTraining=True,render = False, retrain = False):
    env = gym.make('MountainCar-v0', render_mode='human' if render else None)
    alpha = 0.9
    gamma = 0.9
    pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20)
    vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20)
    best_reward = -1000
    if isTraining and not retrain:
        q= np.zeros((len(pos_space), len(vel_space), env.action_space.n))
    else:
        f = open('mountain_car_q_learning_best.pkl', 'rb')
        q = pickle.load(f)
        f.close()
    rewards_per_episode = []
    epsilon = 1
    
    epsilon_decay = 2 / episodes
    
    rng = np.random.default_rng()
    for i in range(episodes):
        state = env.reset()[0]
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)
        done = False   
        rewards = 0
        
        while (not done and rewards > -1000):
            
            
            if isTraining and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state_p, state_v, :])
            new_state, reward, done,_,_ = env.step(action)
            new_state_p = np.digitize(state[0], pos_space)
            new_state_v = np.digitize(state[1], vel_space)
            if isTraining:
                q[state_p, state_v, action] = q[state_p, state_v, action] + alpha * (reward + gamma * np.max(q[new_state_p, new_state_v, :]) - q[state_p, state_v, action])
            state = new_state
            state_p = new_state_p
            state_v = new_state_v
            
            
            rewards += reward
        if isTraining:
            if rewards > best_reward:
                best_reward = rewards
                print(f'Episode {i} reward: {rewards}')
                f = open('mountain_car_q_learning_best.pkl', 'wb')
                pickle.dump(q, f)
                f.close()
        epsilon = max(epsilon - epsilon_decay,0)
        rewards_per_episode.append(rewards)
    env.close()
    mean_rewards = np.zeros(episodes)
    for t in range(episodes):
        mean_rewards[t] = np.mean(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(mean_rewards)
    if isTraining:
        plt.savefig('mountain_car_q_learning.png')
if __name__ == '__main__':
    #run(5000,isTraining=True, render=False)
    run(10,isTraining=False, render=True)
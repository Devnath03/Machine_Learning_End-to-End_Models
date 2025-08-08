#Import Libraries
import gym
import random

env = gym.make('CartPole-v0', render_mode='human')
state = env.observation_space.shape[0]
actions = env.action_space.n

episodes = 100

for episode in range(episodes+1):
    state = env.reset()
    score = 0
    done = False

    
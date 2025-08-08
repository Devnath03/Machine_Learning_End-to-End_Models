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

    while not done:
        action = random.randint(0, actions-1)
        next_state, reward, done, info = env.step(action)
        state = next_state
        score += reward

    print("Episode:", episode, "Score:", score)
    env.close()
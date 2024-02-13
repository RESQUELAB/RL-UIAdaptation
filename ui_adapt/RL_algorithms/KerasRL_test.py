import gym
import random
import numpy as np
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adam

import matplotlib
# Decimos a matplotlib que use TkAgg para que muestre bien los plots.
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from ui_adapt.envs.uiadaptationenv import UIAdaptationEnv
import os
import pickle
import datetime

# Define the environment
env = UIAdaptationEnv()

states = np.prod(env.observation_space.nvec)

print('States', states)
actions = env.action_space.n
print('Actions', actions)

episodes = 10

for episode in range(1,episodes+1):
    # At each begining reset the game 
    state = env.reset()
    # set done to False
    done = False
    # set score to 0
    score = 0
    # while the game is not finished
    while not done:
        # visualize each step
        env.render()
        # choose a random action
        action = env.action_space.sample()
        # action = random.choice([0,1])
        # execute the action
        n_state, reward, done, info = env.step(action)
        # keep track of rewards
        score+=reward
    print('episode {} score {}'.format(episode, score))

def agent(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape = (1, states)))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model
  
model = agent(env.observation_space.shape[0], env.action_space.n)


from rl.agents import SARSAAgent
from rl.policy import EpsGreedyQPolicy

policy = EpsGreedyQPolicy()

sarsa = SARSAAgent(model = model, policy = policy, nb_actions = env.action_space.n)

sarsa.compile('adam', metrics = ['mse'])

sarsa.fit(env, nb_steps = 500, visualize = False, verbose = 1)

scores = sarsa.test(env, nb_episodes = 100, visualize= False)
print('Average score over 100 test games:{}'.format(np.mean(scores.history['episode_reward'])))

sarsa.save_weights('sarsa_weights.h5f', overwrite=True)

_ = sarsa.test(env, nb_episodes = 2, visualize= True)

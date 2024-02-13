import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

import matplotlib.pyplot as plt

from uiadaptationenv import UIAdaptationEnv

# Get the environment and extract the number of actions.
env = UIAdaptationEnv()

nb_actions = env.action_space.n
states = np.prod(env.observation_space.nvec)

# Next, we build a very simple model.
model = Sequential()
# model.add(Flatten(input_shape=(states,)))
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
# model.add(Flatten(input_shape=env.observation_space.shape))
model.add(Dense(12))
model.add(Activation('relu'))
model.add(Dense(12))
model.add(Activation('relu'))
model.add(Dense(12))
model.add(Activation('relu'))
model.add(Dense(nb_actions, activation='linear'))
# model.add(Activation('linear'))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=20000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, 
               nb_steps_warmup=10, target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
train_history = dqn.fit(env, nb_steps=100000, visualize=False, 
                        verbose=2, nb_max_episode_steps=10)

# After training is done, we save the final weights.
dqn.save_weights('dqn_{}_weights.h5f'.format('test'), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
scores = dqn.test(env, nb_episodes=20, visualize=False, nb_max_episode_steps= 20)
print("scores: ")
print(scores)

train_rewards = train_history.history['episode_reward']
nb_steps = train_history.history['nb_steps']
nb_episode_steps = train_history.history['nb_episode_steps']


# Create a single figure with two subplots
fig, ax = plt.subplots(2, 2)

# Plot the data in the first subplot
ax[0][0].plot(train_rewards)
ax[0][0].set_title("Training_reward evolution")

# Plot the data in the second subplot
ax[0][1].plot(nb_steps)
ax[0][1].set_title("Number of steps evolution")

# Plot the data in the third subplot
ax[1][0].plot(nb_episode_steps)
ax[1][0].set_title("Episode steps evolution")

# Show the figure
plt.show()

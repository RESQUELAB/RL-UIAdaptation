import numpy as np
import random
import gym
import matplotlib
# Decimos a matplotlib que use TkAgg para que muestre bien los plots.
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import pickle
import datetime

import ui_adapt    

env_name = "CartPole-v1"

# Define the environment

env = gym.make(env_name)
# env = gym.make("Taxi-v3").env

# FOR UI ADAPT: nvec
# observation_space_size = env.observation_space.nvec
observation_space_size = env.observation_space.n

action_space_size = env.action_space.n

# Define the Q-table

q_table = np.zeros((np.prod(observation_space_size), action_space_size))
# q_table = np.zeros([env.observation_space.n, env.action_space.n])

# define hyperparameters ----------
alpha = 0.7           # Learning rate
gamma = 0.3                  # Discounting rate

# Exploration parameters
epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.05            # Minimum exploration probability
decay_rate = 0.00005          # Exponential decay rate for exploration prob

# Define the num of episodes and steps per episode
episodes = 100000
steps = 15

# Define the metric list to plot
mean_reward_per_episode = []
total_reward_per_episode = []
number_of_steps_to_complete = []

actions_history = []
epsilon_history = []
total_done = []

for episode in range(episodes):
    if episode % 10 == 0 :
        print(f"Episode {episode}/{episodes}")
    # Reset the environment and get the initial state
    state = env.reset()
    state_idx = np.ravel_multi_index(state, env.all_combinations)
    # state_idx = state[0]

    rewards_episode = []
    rewards = []
    actions_episode = []
    
    epsilon_history.append(epsilon)

    for step in range(steps):
        # Choose an action using an epsilon-greedy policy
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state_idx, :])
        
        actions_episode.append(action)

        # Take the action and get the next state, reward, and done flag        
        next_state, reward, done, _= env.step(action)
        # next_state, reward, done, _, _= env.step(action)
        next_state_idx = np.ravel_multi_index(next_state, observation_space_size)
        # next_state_idx = next_state

        # Update the Q-table
        q_table[state_idx][action] = (1 - alpha) * q_table[state_idx][action] + alpha * (reward + gamma * np.max(q_table[next_state_idx]))
        
        # Update the state
        state = next_state
        
        rewards_episode.append(reward)
        # Check if the episode is done
        if done:
            break

    
    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)

    rewards.append(reward)
    
    actions_history.append(actions_episode)

    total_done.append(done)

    mean_reward_per_episode.append(np.max(rewards_episode))
    total_reward_per_episode.append(np.sum(rewards_episode))
    number_of_steps_to_complete.append(step)

counter = 0
for q in q_table:
    if q.any():
        print(q)
    else:
        counter+=1
print(str(counter) + " zeros.")

# Create a single figure with two subplots
fig, ax = plt.subplots(2, 2)

# Plot the data in the first subplot
ax[0][0].plot(number_of_steps_to_complete)
ax[0][0].set_title("number_of_steps_to_complete")

# Plot the data in the second subplot
ax[0][1].plot(mean_reward_per_episode)
ax[0][1].set_title("Mean reward per episode")

# Plot the data in the second subplot
ax[1][0].plot(epsilon_history)
ax[1][0].set_title("E-Decay ")

# Plot the data in the second subplot
ax[1][1].plot(total_reward_per_episode)
ax[1][1].set_title("Total rewards per episode?")


# Show the figure
plt.show()

current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
models_dir = 'RL_algorithms/models'
filename = str(current_time) + "_" + env_name + "_QLearning" + ".pickle"
file_path = models_dir+"/"+filename
# Check if the directory for models exists
if not os.path.exists(models_dir):
    os.makedirs(models_dir)


# Save the metrics and the model to disk
with open(file_path, 'wb') as file:
    pickle_data = {'mean_reward_per_episode': mean_reward_per_episode,
                   'total_reward_per_episode': total_reward_per_episode,
                   'number_of_steps_to_complete': number_of_steps_to_complete,
                   'q_table': q_table}
    pickle.dump(pickle_data, file)


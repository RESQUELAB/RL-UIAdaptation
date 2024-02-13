import numpy as np
import gym
import pickle
import matplotlib.pyplot as plt

import pandas as pd

import ui_adapt


# Create bins and Q table
def load_q_table():
    obsSpaceSize = env.observation_space.nvec
    qTable = saved_data['q_table']
    return obsSpaceSize, qTable

# Given a state of the enviroment, return its descreteState index in qTable
def get_discrete_state(state, obsSpaceSize):
    state_idx = np.ravel_multi_index(state, obsSpaceSize)
    return state_idx

# Load the saved Q-table and metrics
file_path = 'RL_algorithms/models/QLearning_Sigma_0.5_UIAdaptation-v0.pickle'

SIGMA = 0.5

env_name = "UIAdaptation-v0"

# Create the testing environment
env = gym.make(env_name)



with open(file_path, 'rb') as file:
    saved_data = pickle.load(file)

import csv
metrics_learn = saved_data['metrics']
# Get the keys (column headers) and values (rows)
keys = metrics_learn.keys()
rows = zip(*metrics_learn.values())
file_path = "QLearning_LEARN_SIGMA" + str(SIGMA) + "_" + env_name + ".csv"
# Write to CSV
with open(file_path, 'w', newline='') as file:
    writer = csv.writer(file)
    
    # Write the header
    writer.writerow(keys)
    
    # Write the rows
    writer.writerows(list(rows))

print(f'The dictionary has been saved to {file_path}.')


EPISODES = 1000
test_steps = 20
UPDATE_EVERY = 30
MAX_STEPS = 20  # Max number of steps to consider it to be a failure
MAX_STEPS_TARGET = 5  # Max number of steps to consider it to be a failure

# Extract the Q-table from saved data
obsSpaceSize, q_table = load_q_table()

previousStep = []  # array of all counts over runs
previousReward = []  # array of all scores over runs
previousAlignment = []  # array of all scores over runs

metrics = {'ep': [], 'avg_step': [], 'min_step': [], 'max_step': [],
           'avg_rew': [], 'min_rew': [], 'max_rew': [],
           'avg_alig': []}  # metrics recorded for graph

max_reward = -10000
max_reward_state = -1000

firstEpisode = True


# Initialize a list to store episode scores
episode_scores = []

# Initialize a list to store rolling averages
rolling_averages = []
rolling_averages_eps = []

rolling_window_size = 150

for episode in range(EPISODES):
    discreteState = get_discrete_state(env.reset(), obsSpaceSize)
    done = False  # has the enviroment finished?
    step = 0  # how may movements cart has made

    while not done:
        step += 1
        action = np.argmax(q_table[discreteState])
        newState, reward, done, info = env.step(action, sigma=SIGMA)  # perform action on enviroment
        newDiscreteState = get_discrete_state(newState, obsSpaceSize)

        alignment = info["alignment"]
        # too many time, penalize.
        if step > MAX_STEPS:
            break
        elif done and step <= MAX_STEPS_TARGET:
            # additional reward for completing the episode within the target number of steps
            reward += 1
        discreteState = newDiscreteState
            
    previousStep.append(step)
    previousReward.append(round(reward, 4))
    previousAlignment.append(round(alignment, 4))

    
    # Append the episode score to the list
    episode_scores.append(reward)

    # Calculate the rolling average using Pandas with a window of size 150
    if episode >= rolling_window_size - 1:
        rolling_average = pd.Series(episode_scores).rolling(window=rolling_window_size).mean().values[-1]
        rolling_averages.append(rolling_average)
        rolling_averages_eps.append(episode)


    # Add new metrics for graph
    if episode % UPDATE_EVERY == 0:
        if firstEpisode:
            firstEpisode = False
            continue
        latestEpisodes = previousStep[-UPDATE_EVERY:]
        latestEpisodes_reward = previousReward[-UPDATE_EVERY:]
        latestEpisodes_alignment = previousAlignment[-UPDATE_EVERY:]
        averageStep = round(sum(latestEpisodes) / len(latestEpisodes), 4)
        averageRew = round(sum(latestEpisodes_reward) / len(latestEpisodes_reward), 4)
        averageAlignment = round(sum(latestEpisodes_alignment) / len(latestEpisodes_alignment), 4)
        metrics['ep'].append(episode)
        metrics['avg_step'].append(averageStep)
        metrics['min_step'].append(min(latestEpisodes))
        metrics['max_step'].append(max(latestEpisodes))
        metrics['avg_rew'].append(averageRew)
        metrics['min_rew'].append(min(latestEpisodes_reward))
        metrics['max_rew'].append(max(latestEpisodes_reward))
        metrics['avg_alig'].append(averageAlignment)
        
        print("Episode:", episode, "/",str(EPISODES) ,"\n\tSteps - Average:", averageStep, "Min:", min(latestEpisodes), "Max:", max(latestEpisodes))
        print("\tReward - Average:", averageRew, "Min:", min(latestEpisodes_reward), "Max:", max(latestEpisodes_reward))
        print("\tAlignment - Average:", averageAlignment, "Min:", min(latestEpisodes_alignment), "Max:", max(latestEpisodes_alignment))

env.close()


# Plot the rolling averages
plt.plot(rolling_averages, label=f'Rolling Average (Window {rolling_window_size})')
plt.xlabel('Episode')
plt.ylabel('Rolling Average')
plt.legend()
plt.show()


# Plot the testing results
fig, ax = plt.subplots(2, 2)


ax[0][1].plot(metrics['ep'], metrics['avg_step'], label="average steps")
ax[0][1].set_title("Testing - Average number of steps")

ax[1][0].plot(metrics['ep'], metrics['avg_rew'], label="average rewards")
ax[1][0].set_title("Testing - Average Reward")

ax[1][1].plot(metrics['ep'], metrics['avg_alig'], label="average alignment")
ax[1][1].set_title("Testing - Average Alignment per Episode")

# Show the figure
plt.show()

import os
# import pickle
import datetime
import csv


current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
models_dir = 'RL_algorithms/evaluation'
filename = "QLearning_EVAL_SIGMA" + str(SIGMA) + "_" + env_name + ".csv"
file_path = models_dir+"/"+filename
# Check if the directory for models exists
if not os.path.exists(models_dir):
    os.makedirs(models_dir)


# Get the keys (column headers) and values (rows)
keys = metrics.keys()
rows = zip(*metrics.values())

print(f"rows: {metrics.values()}")

# Write to CSV
with open(file_path, 'w', newline='') as file:
    writer = csv.writer(file)
    
    # Write the header
    writer.writerow(keys)
    
    # Write the rows
    writer.writerows(list(rows))

print(f'The dictionary has been saved to {file_path}.')

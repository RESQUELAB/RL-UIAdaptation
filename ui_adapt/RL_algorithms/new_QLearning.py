import gym
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

import ui_adapt
# env = gym.make('CartPole-v0')
env_name = "UIAdaptation-v0"
env = gym.make(env_name)

SIGMA = 1

# How much new info will override old info. 0 means nothing is learned, 1 means only most recent is considered, old knowledge is discarded
LEARNING_RATE = 0.85
# Between 0 and 1, mesue of how much we carre about future reward over immedate reward
DISCOUNT = 0.8
EPISODES = 60000  # Number of iterations run
SHOW_EVERY = 2000  # How oftern the current solution is rendered
UPDATE_EVERY = 150  # How oftern the current progress is recorded
MAX_STEPS = 20  # Max number of steps to consider it to be a failure
MAX_STEPS_TARGET = 3  # Max number of steps to consider it to be a failure

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)
MIN_EPSILON = 0.1

# Create bins and Q table
def create_q_table():
    obsSpaceSize = env.observation_space.nvec
    action_space_size = env.action_space.n

    qTable = np.zeros((np.prod(obsSpaceSize), action_space_size))

    return obsSpaceSize, qTable


# Given a state of the enviroment, return its descreteState index in qTable
def get_discrete_state(state, obsSpaceSize):
    state_idx = np.ravel_multi_index(state, obsSpaceSize)
    return state_idx


obsSpaceSize, qTable = create_q_table()

previousStep = []  # array of all counts over runs
previousReward = []  # array of all scores over runs
previousEpsilon = []  # array of all epsilons over runs
previousAction = []  # array of all epsilons over runs
metrics = {'ep': [], 'avg_step': [], 'min_step': [], 'max_step': [],
           'avg_rew': [], 'min_rew': [], 'max_rew': [],
           'epsilon': [], 'action': [],
           'rolling_averages_eps': [],
           'rolling_averages': []}  # metrics recorded for graph

epsilon_plot = []
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
        if episode % SHOW_EVERY == 0:
            # env.render()  # if running RL comment this out
            pass
                
        step += 1
        # Get action from Q table
        if np.random.random() > epsilon:
            action = np.argmax(qTable[discreteState])
        # Get random action
        else:
            action = np.random.randint(0, env.action_space.n)
        newState, reward, done, _ = env.step(action, sigma=SIGMA)  # perform action on enviroment

        newDiscreteState = get_discrete_state(newState, obsSpaceSize)

        maxFutureQ = np.max(qTable[newDiscreteState])  # estimate of optiomal future value
        currentQ = qTable[discreteState][action]  # old value
        
        # too many time, penalize.
        if step > MAX_STEPS:
            break
        elif done and step <= MAX_STEPS_TARGET:
            # additional reward for completing the episode within the target number of steps
            reward += 1
        
        # formula to caculate all Q values
        newQ = (1 - LEARNING_RATE) * currentQ + LEARNING_RATE * (reward + DISCOUNT * maxFutureQ)
        qTable[discreteState][action] = newQ  # Update qTable with new Q value
        
        discreteState = newDiscreteState
        
    
    previousStep.append(step)
    previousReward.append(reward)
    previousEpsilon.append(epsilon)
    previousAction.append(action)

    epsilon_plot.append(epsilon)
    print(epsilon)

    # Decaying is being done every run if run number is within decaying range
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value
    if epsilon <= MIN_EPSILON:
        epsilon = MIN_EPSILON
    

    # Append the episode score to the list
    episode_scores.append(reward)

    # Calculate the rolling average using Pandas with a window of size 150
    if episode >= rolling_window_size - 1:
        rolling_average = pd.Series(episode_scores).rolling(window=rolling_window_size).mean().values[-1]
        rolling_averages.append(rolling_average)
        rolling_averages_eps.append(episode)

        # Print or use the rolling average as needed
        # print(f"Episode: {episode + 1}, Step: {step}, Rolling Average (Window {rolling_window_size}): {rolling_average}")


    # Add new metrics for graph
    if episode % UPDATE_EVERY == 0:
        if firstEpisode:
            firstEpisode = False
            continue
        latestEpisodes = previousStep[-UPDATE_EVERY:]
        latestEpisodes_reward = previousReward[-UPDATE_EVERY:]
        latestEpisodes_epsilon = previousEpsilon[-UPDATE_EVERY:]
        latestEpisodes_action = previousAction[-UPDATE_EVERY:]
        averageStep = sum(latestEpisodes) / len(latestEpisodes)
        averageRew = sum(latestEpisodes_reward) / len(latestEpisodes_reward)
        averageAction = sum(latestEpisodes_action) / len(latestEpisodes_action)
        metrics['ep'].append(episode)
        metrics['action'].append(averageAction)
        metrics['avg_step'].append(averageStep)
        metrics['min_step'].append(min(latestEpisodes))
        metrics['max_step'].append(max(latestEpisodes))
        metrics['avg_rew'].append(averageRew)
        metrics['min_rew'].append(min(latestEpisodes_reward))
        metrics['max_rew'].append(max(latestEpisodes_reward))
        metrics['epsilon'].append(np.mean(latestEpisodes_epsilon))
        print("Episode:", episode, "/",str(EPISODES) ,"\n\tSteps - Average:", averageStep, "Min:", min(latestEpisodes), "Max:", max(latestEpisodes))
        print("\tReward - Average:", averageRew, "Min:", min(latestEpisodes_reward), "Max:", max(latestEpisodes_reward))

metrics["rolling_averages"] = rolling_averages
metrics["rolling_averages_eps"] = rolling_averages_eps

env.close()


# Plot the Epsilon:
plt.plot(epsilon_plot, label=f'Epsilon over time')
plt.xlabel('Episode')
plt.ylabel('Rolling Average')
plt.legend()
plt.show()




# Plot the rolling averages
plt.plot(rolling_averages, label=f'Rolling Average (Window {rolling_window_size})')
plt.xlabel('Episode')
plt.ylabel('Rolling Average')
plt.legend()
plt.show()


# Plot graph
plt.plot(metrics['ep'], metrics['avg_step'], label="average steps")
# plt.plot(metrics['ep'], metrics['min_step'], label="min step")
# plt.plot(metrics['ep'], metrics['max_step'], label="max step")

plt.plot(metrics['ep'], metrics['avg_rew'], label="average rewards")
# plt.plot(metrics['ep'], metrics['min_rew'], label="min rewards")
# plt.plot(metrics['ep'], metrics['max_rew'], label="max rewards")

# plt.plot(metrics['ep'], metrics['action'], label="average action")

import os
import pickle
import datetime

current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
models_dir = 'RL_algorithms/models'
filename = "QLearning_Sigma_" + str(SIGMA) + "_" + env_name + ".pickle"
file_path = models_dir+"/"+filename
# Check if the directory for models exists
if not os.path.exists(models_dir):
    os.makedirs(models_dir)


# Save the metrics and the model to disk
with open(file_path, 'wb') as file:
    pickle_data = {'metrics': metrics,
                    'q_table': qTable}
    pickle.dump(pickle_data, file)




plt.legend(loc=0)
plt.show()

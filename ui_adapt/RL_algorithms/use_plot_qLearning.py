import pickle
import matplotlib.pyplot as plt

model_path = "models/q_table_2023-02-14_14-02.pickle"
# Load the metrics and the model from disk
with open(model_path, 'rb') as file:
    pickle_data = pickle.load(file)
    mean_reward_per_episode = pickle_data['mean_reward_per_episode']
    total_reward_per_episode = pickle_data['total_reward_per_episode']
    number_of_steps_to_complete = pickle_data['number_of_steps_to_complete']
    q_table = pickle_data['q_table']


print(total_reward_per_episode)

def plot_metric():
    # Create a single figure with two subplots
    fig, ax = plt.subplots(1, 2)

    # Plot the data in the first subplot
    ax[0].plot(total_reward_per_episode)
    ax[0].set_title("Total (acumulated) reward per Episode")

    # Plot the data in the second subplot
    ax[1].plot(number_of_steps_to_complete)
    ax[1].set_title("Steps to complete episode")

    # Show the figure
    plt.show()
 
| OS  | Version | Working |
| :---: | :---: | :---:|
| <img src="https://user-images.githubusercontent.com/25181517/186884153-99edc188-e4aa-4c84-91b0-e2df260ebc33.png" alt="drawing" width="32"/>  | Ubuntu 18.04 | :heavy_check_mark: |
| <img src="https://user-images.githubusercontent.com/25181517/186884150-05e9ff6d-340e-4802-9533-2c3f02363ee3.png" alt="drawing" width="32"/>  | Windows 10  | :heavy_check_mark: |

Python version: 3.6.3 :heavy_check_mark:


# Reinforcement Learning User Interface Adaptation - Open AI GYM environment

This repository will be used to provide the implementation of a training environment for RL agents for the UI adaptation problem.

## Installation

It is recommended to create and/or activate your own Conda environment:
```bash
conda create -n UIrlhf python==3.6.3
conda activate UIrlhf
```

Then install

```bash
cd  RL-based-UIAdaptation
pip install -e .
```

Then you can use it in your files by importing it

```python
import gym
import ui_adapt
env = gym.make("UIAdaptation-v0")
```

## Trying the environment

To try the environment, we are going to import it first, then we are going to render the env after each step.

```python
import gym
import ui_adapt
env = gym.make("UIAdaptation-v0")
env.render() # You should see the UI-Context created
env.step(0, verbose=True) # No-operate
env.render() # You should see the same as before (since the action was no-operate)
env.step(1, verbose=True) # Changing to list
env.render() # You should see that the layout changed to 'List'
```


#### Modify the environment:

The 3 main aspects of the RL agent are the **`state`**, the **`actions`** and the **`reward`**. Everything should be changed in the **`config.json`** file. 

* State: This JSON file MUST have a `USER`, `PLATFORM`, `ENVIRONMENT` and `UIDESIGN` keys in order to represent the **state**.

* Actions: It also MUST have an `ACTIONS` key in order to define what adaptations you have implemented. The actions must have a `name`, a `target` and a `value` representing what is going to be changed (target) and with which property (value).
  
* Reward: the `compute_reward()` function in the `UIAdaptationEnv` class should be modified.


If you have an adaptive app, you can connect it to this environment through API calls. The API configuration should be defined in the `API_CONNECTION` key. More information about how to connect the API below (_Still work in progress_).


#### Rules for the configuration file


1. Format: 
2. TODO...
3. TODO...

#### "Connecting" your adaptive app to this environment

TODO - formatting of the API... (The config file allows some personalization for the API calls but not entirely.)


#### Running RL algorithms with this enviroment:
First, install dependencies:

```bash
pip install tensorflow==1.10
pip install keras==2.1.6,
pip install keras-rl2
```

Then,

1. Go to `src` path: `cd src`
2. Test with algorithms:
  2.1 `python QLearning.py` - This will stop automatically
  2.2 `python KerasRL_DQN.py` - This will stop when you hit "Ctrl+C"

#### How to user other algorithms:

You can install libraries such as Keras-RL2, stable-baselines, RL-coach, etc. Then, create your own python program that uses an algorithm from one of these libraries and import the UIAdaptation Environment like in the #Installation section


#### TODO

There are still some work in progress. This is a prototype.

The reward function now only works with preferences + emotions. Possible future work:
* Get the usability from UI Design. 
* Get Emotions from sources such as facial expressions, etc.
* Use Reinforcement Learning with Human Feedback. Check [`rl-teacher-ui-adapt`](https://github.com/Dagasfi/rl-teacher-ui-adapt) to see how to connect both and how to use it.



import gym
from gym import spaces
import numpy as np

import ui_adapt.utils as utils
from ui_adapt.utils import Config
from ui_adapt.envs.reward_predictor import RewardPredictor

class UIAdaptationEnv(gym.Env):
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    _max_episode_steps = 20

    def __init__(self,render_mode=None):
        self.name = 'uiadaptation'
        
        self.config = Config()
        self.reward_predictor = RewardPredictor("model.pckl")

        self.user = utils.get_random_user(self.config)
        self.uidesign = utils.get_random_ui(self.config)
        self.platform = utils.get_random_platform(self.config)
        self.environment = utils.get_random_environment(self.config)

        self.state = self.get_observation()
        self.actions = utils.get_actions(self.config)

        self.action_space = spaces.Discrete(len(self.actions))

        self.all_combinations = (self.uidesign.combinations + 
                            self.user.combinations +
                           self.platform.combinations + 
                           self.environment.combinations)
        self.observation_space = gym.spaces.MultiDiscrete(self.all_combinations)
        self.observation_space_size = np.prod(self.observation_space.nvec)
        self.reward_collected = 0
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def get_image():
        pass

    def render(self, render_mode='human'):
        '''
        Returns: None
        Show the current environment state e.g., the graphical window
        This method must be implemented, but it's ok to have an empty implementation if
        rendering is not important
        '''
        self.uidesign.render()
        self.user.info()
        self.environment.info()
        self.platform.info()

    def close (self):
        '''
        Return: None
        This method is optional. Used to cleanup all resources (threads, GUI, etc)
        '''
        print("CLOSING")
        pass


    def step(self, action, verbose=False, sigma=0.5):
        err_msg = f"Action {action!r} ({type(action)}) invalid. Does not exist in ACTION_SPACE."
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."        

        initial_state = self.get_observation()
        done = False
        info = {}
        reward = 0
        
        penalize_flag = False

        action_data = self.actions.get(action, {})
        if not action_data:
            print(f"Action {action} is not defined.")
            return

        name = action_data.get("name", "")
        target = action_data.get("target", "")
        value = action_data.get("value", "")
        api_call = action_data.get("api_call", "")

        if target == "pass":
            # No operation
            pass
        else:
            self.uidesign.update(target, value, api_call=api_call)
            # if initial_state == self.get_observation():
                # We took an action different than "pass" and nothing changed.
                # penalize_flag = True

        self.state = self.get_observation()
        
        alignment = self.reward_predictor.get_alignment(
                    self.user, self.uidesign)
        info["alignment"] = alignment


        '''if penalize_flag:
            reward = -5
        else:
            reward = self.compute_reward()         
            # reward = alignment
        '''
        reward = self.compute_reward(sigma=sigma)
        
        self.reward_collected += reward

        
        done = bool(reward>=0.7)

        if verbose:
            print(f"Performed action: {name}, Target: {target}, Value: {value}, Reward: {reward}," 
                  f"Collected Reward: {self.reward_collected}, Done: {done}")

        return self.state, reward, done, info


    def reset(self, *, seed = None, options = None):
        '''
        We first clean everithing and then we create a new Context, app, etc
        '''
        # self.close()

        self.user = utils.get_random_user(self.config)
        self.platform = utils.get_random_platform(self.config)
        self.environment = utils.get_random_environment(self.config)
        self.uidesign = utils.get_random_ui(self.config)
        self.state = self.get_observation()
        self.reward_collected = 0
        return self.state

    def state_as_array(self, state, npArray=False):
        state_array = []
        for a in state:
            if type(state[a]) is dict:
                for b in state[a]:
                    if type(state[a][b]) is dict:
                        for c in state[a][b]:
                            state_array.append(state[a][b][c])
                    else:
                        state_array.append(state[a][b])
            else:
                state_array.append(state[a])
        if npArray:
            return np.array(state_array)
        return state_array
    
    def get_observation(self):
        """
            This method traduces the representation of the state into an observation
            that the gym can work with.
        """
        uidesign_state = self.uidesign.get_state()
        user_state = self.user.get_state()
        environment_state = self.environment.get_state()
        platform_state = self.platform.get_state()

        self.state = {
            **uidesign_state,
            **user_state, 
            **platform_state,
            **environment_state
            }
        return self.state_as_array(self.state, npArray=False)
        
    def compute_reward(self, sigma=1):
        '''
        Add here the reward function.
        '''
        reward = (1 - sigma) * self.individual_reward() + sigma * self.general_reward()
        return reward
    
    def general_reward(self):
        return self.reward_predictor.predict(self.uidesign)[0]
    
    def individual_reward(self):
        return self.reward_predictor.get_alignment(self.user, self.uidesign)
    

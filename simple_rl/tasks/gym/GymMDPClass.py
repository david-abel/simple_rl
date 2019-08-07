'''
GymMDPClass.py: Contains implementation for MDPs of the Gym Environments.
'''

# Python imports.
import random
import sys
import os
import random
from collections import defaultdict

# Other imports.
import gym
from simple_rl.mdp.MDPClass import MDP
from simple_rl.tasks.gym.GymStateClass import GymState

class GymMDP(MDP):
    ''' Class for Gym MDPs '''

    def __init__(self, env_name='CartPole-v0', render=False, render_every_n_episodes=0):
        '''
        Args:
            env_name (str)
            render (bool): If True, renders the screen every time step.
            render_every_n_epsiodes (int): @render must be True, then renders the screen every n episodes.
        '''
        # self.render_every_n_steps = render_every_n_steps
        self.render_every_n_episodes = render_every_n_episodes
        self.episode = 0
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.render = render
        MDP.__init__(self, range(self.env.action_space.n), self._transition_func, self._reward_func, init_state=GymState(self.env.reset()))
    
    def get_parameters(self):
        '''
        Returns:
            (dict) key=param_name (str) --> val=param_val (object).
        '''
        param_dict = defaultdict(int)
        param_dict["env_name"] = self.env_name
   
        return param_dict

    def _reward_func(self, state, action, next_state):
        '''
        Args:
            state (AtariState)
            action (str)

        Returns
            (float)
        '''
        return self.prev_reward

    def _transition_func(self, state, action):
        '''
        Args:
            state (AtariState)
            action (str)

        Returns
            (State)
        '''
        obs, reward, is_terminal, info = self.env.step(action)

        if self.render and (self.render_every_n_episodes == 0 or self.episode % self.render_every_n_episodes == 0):
            self.env.render()

        self.prev_reward = reward
        self.next_state = GymState(obs, is_terminal=is_terminal)

        return self.next_state

    def reset(self):
        self.env.reset()
        self.episode += 1

    def __str__(self):
        return "gym-" + str(self.env_name)

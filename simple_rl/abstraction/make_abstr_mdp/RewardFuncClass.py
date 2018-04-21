# Python imports.
import random
from collections import defaultdict

class RewardFunc(object):

    def __init__(self, reward_func_lambda, state_space, action_space):
        self.reward_dict = make_dict_from_lambda(reward_func_lambda, state_space, action_space)

    def reward_func(self, state, action):
        return self.reward_dict[state][action]

def make_dict_from_lambda(reward_func_lambda, state_space, action_space, sample_rate=1):
    reward_dict = defaultdict(lambda:defaultdict(float))
    for s in state_space:
        for a in action_space:
            for i in range(sample_rate):
                reward_dict[s][a] = reward_func_lambda(s, a) / sample_rate

    return reward_dict

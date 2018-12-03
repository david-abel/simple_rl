# Python imports.
from __future__ import print_function
import random
from collections import defaultdict

# Other imports.
from simple_rl.abstraction.action_abs.PolicyClass import Policy

class PolicyFromDict(Policy):

    def __init__(self, policy_dict):
        self.policy_dict = policy_dict

    def get_action(self, state):
        return self.policy_dict[state]

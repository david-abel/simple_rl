# Python imports.
from collections import defaultdict
from os import path
import sys

# Other imports.
parent_dir = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(parent_dir)
from simple_rl.mdp.StateClass import State
from simple_rl.mdp.MDPClass import MDP
from action_abs.ActionAbstractionClass import ActionAbstraction

class ActionAbstractionStack(ActionAbstraction):

    def __init__(self, list_of_aa, prim_actions, level=0):
        '''
        Args:
            list_of_aa (list)
        '''
        self.list_of_aa = list_of_aa
        self.level = level
        self.prim_actions = prim_actions
        ActionAbstraction.__init__(self, options=self.get_actions(level), prim_actions=prim_actions)

    def get_level(self):
        return self.level

    def get_num_levels(self):
        return len(self.list_of_aa)

    def get_aa_list(self):
        return self.list_of_aa
    
    def get_actions(self, level=None):
        if level is None:
            level = self.level
        elif level == -1:
            level = self.get_num_levels()
        elif level == 0:
            # If we're at level 0, let the agent act with primitives.
            return self.prim_actions

        return self.list_of_aa[level - 1].get_actions()

    def set_level(self, new_level):
        self.level = new_level

    def act(self, agent, state_abstr_stack, ground_state, reward, level=None):
        '''
        Args:
            agent (Agent)
            abstr_state (State)
            lower_state (State): One level down from abstr_state.
            reward (float)

        Returns:
            (str)
        '''
        if level is None:
            level = self.level
        elif level == -1:
            level = self.get_num_levels()
        elif level == 0:
            # If we're at level 0, let the agent act with primitives.
            agent.actions = self.prim_actions
            return agent.act(ground_state, reward)

        abstr_state = state_abstr_stack.phi(ground_state, level)
        lower_state = state_abstr_stack.phi(ground_state, level - 1)
        # Calls agent update.
        lower_option = self.list_of_aa[level - 1].act(agent, abstr_state, lower_state, reward)
        level -= 1

        # Descend via options.
        while level > 0:
            lower_state = state_abstr_stack.phi(ground_state, level - 1)
            lower_option = lower_option.act(lower_state)
            level -= 1

        return lower_option

    def add_aa(self, new_aa):
        self.list_of_aa.append(new_aa)

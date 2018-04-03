# Python imports.
from collections import defaultdict
from os import path
import sys

# Other imports.
from simple_rl.mdp.MDPClass import MDP
parent_dir = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(parent_dir)
from state_abs.StateAbstractionClass import StateAbstraction
from HierarchyStateClass import HierarchyState


'''
NOTE: Level 0 is the ground state space. Level 1 is the first abstracted, and so on.
    --> Therefore, the "level 0" state abstraction is the identity.
'''

class StateAbstractionStack(StateAbstraction):

    def __init__(self, list_of_phi, level=0):
        '''
        Args:
            list_of_phi (list)
        '''
        self.list_of_phi = list_of_phi # list where each item is a dict, key:state, val:int. (int represents an abstract state).
        cur_phi = {} if len(self.list_of_phi) == 0 else self.list_of_phi[level]
        self.level = 0
        StateAbstraction.__init__(self, phi=cur_phi)

    def get_num_levels(self):
        return len(self.list_of_phi)

    def get_level(self):
        return self.level

    def set_level_to_max(self):
        self.level = self.get_num_levels()

    def set_level(self, new_level):
        if new_level > self.get_num_levels() or new_level < 0:
            raise ValueError("StateAbstractionStack Error: given level (" + str(new_level) + ") is invalid. Must be between" + \
                "0 and the number of levels in the stack (" + str(self.get_num_levels()) + ").")
        self.level = new_level

    def phi(self, lower_state, level=None):
        '''
        Args:
            lower_state (simple_rl.State)
            level (int)

        Returns:
            (simple_rl.State)

        Notes:
            level:
                0 --> Ground
                1 --> First abstract layer, and so on.
        '''

        # Get the level to raise the state to.
        if level == None:
            # Defaults to whatever it's set to.
            level = self.level
        elif level == -1:
            # Grab the last one.
            level = self.get_num_levels() - 1

        if self.get_num_levels() == 0 or level == 0:
            # If there are no more levels, identity function.
            return lower_state

        # Suppose level = 1. Now we'll grab the phi in the first slot and abstract it.
        # Suppose level = 2. We abstract once, cur_level=1, abstract again, cur_level=2 (DONE).

        # Get the current state's level.
        if isinstance(lower_state, HierarchyState):
            cur_level = lower_state.get_level()
        else:
            cur_level = 0


        # if cur_level < level:
        # # Get the immediate abstracted state (one lvl higher).
        # print "cur_level:", cur_level, level
        # s_a = self.list_of_phi[cur_level][lower_state]
        # cur_level += 1

        s_a = lower_state
        # Iterate until we're at the right level.
        while cur_level < level:
            s_a = self.list_of_phi[cur_level][s_a]
            cur_level += 1

        return s_a

    def get_abs_states(self):
        # For each ground state, get its abstract state.
        if self.level == 0:
            # If we're at level 0, identity.
            return self.get_ground_states()

        return set([abs_state for abs_state in set(self.list_of_phi[self.level - 1].values())])

    def get_ground_states_in_abs_state(self, abs_state):
        '''
        Args:
            abs_state (State)

        Returns:
            (list): Contains all ground states in the cluster.
        '''
        return [s_g for s_g in self.get_ground_states() if self.phi(s_g, level=abs_state.get_level()) == abs_state]

    def get_lower_states_in_abs_state(self, abs_state):
        '''
        Args:
            abs_state (State)

        Returns:
            (list): Contains all ground states in the cluster.
        '''
        return [s_g for s_g in self.get_lower_states(level=abs_state.get_level()) if self.phi(s_g, level=abs_state.get_level()) == abs_state]


    def get_lower_states(self, level=None):
        if level == None:
            # Defaults to whatever it's set to.
            level = self.level
        elif level == -1:
            # Grab the last one.
            level = self.get_num_levels()
        elif level == 0:
            return self.get_ground_states()

        return self.list_of_phi[level - 1].keys()

    def get_ground_states(self):
        return self.list_of_phi[0].keys()

    def add_phi(self, new_phi):
        self.list_of_phi.append(new_phi)

    def remove_last_phi(self):
        self.list_of_phi = self.list_of_phi[:-1]

    def print_state_space_sizes(self):
        print("State Space Sizes:")
        print("\t0", len(self.get_ground_states()))

        for i, phi in enumerate(self.list_of_phi):
            print("\t", i + 1, len(set(phi.values())))

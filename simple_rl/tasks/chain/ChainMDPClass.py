''' ChainMDPClass.py: Contains the ChainMDPClass class. '''

# Python imports.
from __future__ import print_function

# Other imports.
from simple_rl.mdp.MDPClass import MDP
from simple_rl.tasks.chain.ChainStateClass import ChainState

class ChainMDP(MDP):
    ''' Implementation for a standard Chain MDP '''

    ACTIONS = ["forward", "reset"]

    def __init__(self, num_states=5, reset_val=0.01, gamma=0.99):
        '''
        Args:
            num_states (int) [optional]: Number of states in the chain.
        '''
        MDP.__init__(self, ChainMDP.ACTIONS, self._transition_func, self._reward_func, init_state=ChainState(1), gamma=gamma)
        self.num_states = num_states
        self.reset_val = reset_val

    def get_parameters(self):
        '''
        Returns:
            (dict) key=param_name (str) --> val=param_val (object).
        '''
        param_dict = defaultdict(int)
        param_dict["num_states"] = self.num_states
        param_dict["reset_val"] = self.reset_val
   
        return param_dict

    def _reward_func(self, state, action, next_state=None):
        '''
        Args:
            state (State)
            action (str)
            next_state (State)

        Returns
            (float)
        '''
        if action == "forward" and state.num == self.num_states:
            return 1
        elif action == "reset":
            return self.reset_val
        else:
            return 0

    def _transition_func(self, state, action):
        '''
        Args:
            state (State)
            action (str)

        Returns
            (State)
        '''
        if action == "forward":
            if state < self.num_states:
                return state + 1
            else:
                return state
        elif action == "reset":
            return ChainState(1)
        else:
            raise ValueError("(simple_rl Error): Unrecognized action! (" + action + ")")

    def __str__(self):
        return "chain-" + str(self.num_states)

''' ChainMDPClass.py: Contains the ChainMDPClass class. '''

# Python imports.
from __future__ import print_function

# Other imports.
from simple_rl.mdp.MDPClass import MDP
from simple_rl.tasks.chain.ChainStateClass import ChainState

class ComboLockMDP(MDP):
    ''' Imeplementation for a standard Chain MDP '''

    ACTIONS = []

    def __init__(self, combo, num_actions=3, num_states=None, reset_val=0.01, gamma=0.99):
        '''
        Args:
            num_states (int) [optional]: Number of states in the chain.
        '''
        ComboLockMDP.ACTIONS = [str(i) for i in range(1, num_actions + 1)]
        self.num_states = len(combo) if num_states is None else num_states
        self.num_actions = num_actions
        self.combo = combo

        if len(combo) != self.num_states:
            raise ValueError("(simple_rl.ComboLockMDP Error): Combo length (" + str(len(combo)) + ") must be the same as num_states (" + str(self.num_states) + ").")
        elif max(combo) > num_actions:
            raise ValueError("(simple_rl.ComboLockMDP Error): Combo (" + str(combo) + ") must only contain values less than or equal to @num_actions (" + str(num_actions) +").")

        MDP.__init__(self, ComboLockMDP.ACTIONS, self._transition_func, self._reward_func, init_state=ChainState(1), gamma=gamma)

    def get_parameters(self):
        '''
        Returns:
            (dict) key=param_name (str) --> val=param_val (object).
        '''
        param_dict = defaultdict(int)
        param_dict["combo"] = self.combo
        param_dict["num_actions"] = self.num_actions
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
        if state.num == self.num_states and int(action) == self.combo[state.num - 1]:
            return 1
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
        # print(state.num, self.num_states, action, self.combo[state.num])
        if int(action) == self.combo[state.num - 1]:
            if state < self.num_states:
                return state + 1
            else:
                # At end of chain.
                return state
        else:
            return ChainState(1)

    def __str__(self):
        return "combolock-" + str(self.num_states)

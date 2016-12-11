''' ChainMDPClass.py: Contains the ChainMDPClass class. '''

# Local imports.
from ...mdp.MDPClass import MDP
from ChainStateClass import ChainState

class ChainMDP(MDP):
    ''' Imeplementation for a standard Chain MDP '''

    ACTIONS = ["forward", "reset"]

    def __init__(self, num_states=5):
        '''
        Args:
            num_states (int) [optional]: Number of states in the chain.
        '''
        MDP.__init__(self, ChainMDP.ACTIONS, self._transition_func, self._reward_func, init_state=ChainState(1))
        self.num_states = num_states

    def _reward_func(self, state, action):
        '''
        Args:
            state (State)
            action (str)
            statePrime

        Returns
            (float)
        '''
        if action == "forward" and state.num == self.num_states:
            return 1
        elif action == "reset":
            return 0.01
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
            #print "Error: Unrecognized action! (" + action + ")"
            quit()

    def __str__(self):
        return "chainmdp-" + str(self.num_states)

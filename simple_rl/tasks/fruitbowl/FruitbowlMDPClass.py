''' FruitbowlMDPClass.py: Contains the FruitbowlMDPClass class. '''

# Local imports.
from ...mdp.MDPClass import MDP
from FruitbowlStateClass import FruitbowlState

class FruitbowlMDP(MDP):
    ''' Imeplementation of a Fruitbowl MDP '''

    ACTIONS = ["add", "continue"]

    def __init__(self, num_fruits=5, reward_str="11111"):
        '''
        Args:
            num_fruits (int) [optional]: Number of fruits that could be in the bowl.
        '''
        if len(reward_str) != num_fruits:
            print "Error: parameter reward_str must have the same length as the value of the parameter num_fruits."
            quit()

        self.reward_str = reward_str
        MDP.__init__(self, FruitbowlMDP.ACTIONS, self._transition_func, self._reward_func, init_state=ChainState(1))
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

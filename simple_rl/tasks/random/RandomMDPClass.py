''' RandomMDPClass.py: Contains the RandomMDPClass class. '''

# Python imports.
import random
import numpy as np

# Local imports.
from ...mdp.MDPClass import MDP
from RandomStateClass import RandomState

class RandomMDP(MDP):
    ''' Imeplementation for a standard Random MDP '''

    ACTIONS = [str(i) for i in range(3)]

    def __init__(self, num_states=5, num_rand_trans=5):
        '''
        Args:
            num_states (int) [optional]: Number of states in the Random MDP.
            num_rand_trans (int) [optional]: Number of possible next states.

        Summary:
            Each state-action pair picks @num_rand_trans possible states and has a uniform distribution
            over them for transitions. Rewards are also chosen randomly.
        '''
        MDP.__init__(self, RandomMDP.ACTIONS, self._transition_func, self._reward_func, init_state=RandomState(1))
        assert(num_rand_trans <= num_states)
        self.num_rand_trans = num_rand_trans
        self.num_states = num_states
        self._reward_s_a = (random.choice(range(self.num_states)), random.choice(RandomMDP.ACTIONS))
        self._transitions = {}

    def _reward_func(self, state, action):
        '''
        Args:
            state (State)
            action (str)
            statePrime

        Returns
            (float)
        '''

        if (state.data, action) == self._reward_s_a:
            return 1.0
        else:
            return 0.0

    def _transition_func(self, state, action):
        '''
        Args:
            state (State)
            action (str)

        Returns
            (State)
        '''
        if (state, action) not in self._transitions:
            # Chooses @self.num_rand_trans from range(self.num_states)
            self._transitions[(state, action)] = np.random.choice(self.num_states, self.num_rand_trans, replace=False)

        state_id = np.random.choice(self._transitions[(state, action)])
        return RandomState(state_id)

    def __str__(self):
        return "RandomMDP-" + str(self.num_states)



def main():
    _gen_random_distr()

if __name__ == "__main__":
    main()
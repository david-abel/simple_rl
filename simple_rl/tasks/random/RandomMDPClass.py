''' RandomMDPClass.py: Contains the RandomMDPClass class. '''

# Python imports.
import random
import numpy as np
from collections import defaultdict

# Other imports.
from simple_rl.mdp.MDPClass import MDP
from simple_rl.tasks.random.RandomStateClass import RandomState

class RandomMDP(MDP):
    ''' Imeplementation for a standard Random MDP '''

    ACTIONS = []

    def __init__(self, num_states=5, num_rand_trans=5, num_actions=3, gamma=0.99):
        '''
        Args:
            num_states (int) [optional]: Number of states in the Random MDP.
            num_rand_trans (int) [optional]: Number of possible next states.

        Summary:
            Each state-action pair picks @num_rand_trans possible states and has a uniform distribution
            over them for transitions. Rewards are also chosen randomly.
        '''
        RandomMDP.ACTIONS = [str(i) for i in range(num_actions)]
        MDP.__init__(self, RandomMDP.ACTIONS, self._transition_func, self._reward_func, init_state=RandomState(1), gamma=gamma)
        # assert(num_rand_trans <= num_states)
        self.num_rand_trans = num_rand_trans
        self.num_states = num_states
        self._reward_s_a = (random.choice(range(self.num_states)), random.choice(RandomMDP.ACTIONS))
        self._transitions = defaultdict(lambda: defaultdict(str))


    def get_parameters(self):
        '''
        Returns:
            (dict) key=param_name (str) --> val=param_val (object).
        '''
        param_dict = defaultdict(int)
        param_dict["num_states"] = self.num_states
        param_dict["num_rand_trans"] = self.num_rand_trans
        param_dict["num_actions"] = self.num_actions
   
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
        if self.num_states == 1:
            return state

        if (state, action) not in self._transitions:
            # Chooses @self.num_rand_trans from range(self.num_states)
            self._transitions[state][action] = np.random.choice(self.num_states, self.num_rand_trans, replace=False)

        state_id = np.random.choice(self._transitions[state][action])
        return RandomState(state_id)

    def __str__(self):
        return "RandomMDP-" + str(self.num_states)



def main():
    _gen_random_distr()

if __name__ == "__main__":
    main()
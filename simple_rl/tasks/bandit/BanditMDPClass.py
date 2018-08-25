''' BanditMDPClass.py: Contains the BanditMDPClass class. '''

# Python imports.
from __future__ import print_function
from collections import defaultdict
import numpy as np

# Other imports.
from simple_rl.mdp.MDPClass import MDP
from simple_rl.mdp.StateClass import State

class BanditMDP(MDP):
    ''' Imeplementation for a standard Bandit MDP.

        Note: Assumes gaussians with randomly initialized mean and variance
        unless payout_distributions is set.
    '''

    ACTIONS = []

    def __init__(self, num_arms=10, distr_family=np.random.normal, distr_params=None):
        '''
        Args:
            num_arms (int): Number of arms.
            distr_family (lambda): A function from numpy which, when given
                entities from @distr_params, samples from the distribution family.
            distr_params (dict): If None is given, default mu/sigma for normal
                distribution are initialized randomly.
        '''
        BanditMDP.ACTIONS = [str(i) for i in range(1, num_arms + 1)]
        MDP.__init__(self, BanditMDP.ACTIONS, self._transition_func, self._reward_func, init_state=State(1), gamma=1.0)
        self.num_arms = num_arms
        self.distr_family = distr_family
        self.distr_params = self.init_distr_params() if distr_params is None else distr_params    

    def get_parameters(self):
        '''
        Returns:
            (dict) key=param_name (str) --> val=param_val (object).
        '''
        param_dict = defaultdict(int)
        param_dict["num_arms"] = self.num_arms
        param_dict["distr_family"] = self.distr_family
        param_dict["distr_params"] = self.distr_params
   
        return param_dict

    def init_distr_params(self):
        '''
        Summary:
            Creates default distribution parameters for each of
                the @self.num_arms arms. Defaults to Gaussian bandits
                with each mu ~ Unif(-1,1) and sigma ~ Unif(0,2).

        Returns:
            (dict)
        '''
        distr_params = defaultdict(lambda: defaultdict(list))
        
        for i in range(self.num_arms):
            next_mu = np.random.uniform(-1.0, 1.0)
            next_sigma = np.random.uniform(0, 2.0)
            distr_params[str(i)] = [next_mu, next_sigma]

        return distr_params

    def _reward_func(self, state, action):
        '''
        Args:
            state (State)
            action (str)
            statePrime

        Returns
            (float)
        '''
        # Samples from the distribution associated with @action.
        return self.distr_family(*self.distr_params[action])

    def _transition_func(self, state, action):
        '''
        Args:
            state (State)
            action (str)

        Returns
            (State)

        Notes:
            Required to fit naturally with the rest of simple_rl, but obviously
            doesn't do anything.
        '''
        return state

    def __str__(self):
        return str(self.num_arms) + "_Armed_Bandit" 

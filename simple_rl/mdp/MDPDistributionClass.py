''' MDPDistributionClass.py: Contains the MDP Distribution Class. '''

# Python imports.
from __future__ import print_function
import numpy as np
from collections import defaultdict

class MDPDistribution(object):
    ''' Class for distributions over MDPs. '''

    def __init__(self, mdp_prob_dict, horizon=0):
        '''
        Args:
            mdp_prob_dict (dict):
                Key (MDP)
                Val (float): Represents the probability with which the MDP is sampled.

        Notes:
            @mdp_prob_dict can also be a list, in which case the uniform distribution is used.
        '''
        if type(mdp_prob_dict) == list or len(mdp_prob_dict.values()) == 0:
            # Assume uniform if no probabilities are provided.
            mdp_prob = 1.0 / len(mdp_prob_dict.keys())
            new_dict = defaultdict(float)
            for mdp in mdp_prob_dict:
                new_dict[mdp] = mdp_prob
            mdp_prob_dict = new_dict

        self.horizon = horizon
        self.mdp_prob_dict = mdp_prob_dict

    def get_parameters(self):
        '''
        Returns:
            (dict) key=param_name (str) --> val=param_val (object).
        '''
        param_dict = {}
        param_dict["mdp_prob_dict"] = self.mdp_prob_dict
        param_dict["horizon"] = self.horizon

        return param_dict

    def remove_mdps(self, mdp_list):
        '''
        Args:
            (list): Contains MDP instances.

        Summary:
            Removes each mdp in @mdp_list from self.mdp_prob_dict and recomputes the distribution.
        '''
        for mdp in mdp_list:
            try:
                self.mdp_prob_dict.pop(mdp)
            except KeyError:
                raise ValueError("(simple-rl Error): Trying to remove MDP (" + str(mdp) + ") from MDP Distribution that doesn't contain it.")

        self._normalize()

    def remove_mdp(self, mdp):
        '''
        Args:
            (MDP)

        Summary:
            Removes @mdp from self.mdp_prob_dict and recomputes the distribution.
        '''
        try:
            self.mdp_prob_dict.pop(mdp)
        except KeyError:
            raise ValueError("(simple-rl Error): Trying to remove MDP (" + str(mdp) + ") from MDP Distribution that doesn't contain it.")

        self._normalize()

    def _normalize(self):
        total = sum(self.mdp_prob_dict.values())
        for mdp in self.mdp_prob_dict.keys():
            self.mdp_prob_dict[mdp] = self.mdp_prob_dict[mdp] / total

    def get_all_mdps(self, prob_threshold=0):
        '''
        Args:
            prob_threshold (float)

        Returns:
            (list): Contains all mdps in the distribution with Pr. > @prob_threshold.
        '''
        return [mdp for mdp in self.mdp_prob_dict.keys() if self.mdp_prob_dict[mdp] > prob_threshold]

    def get_horizon(self):
        return self.horizon

    def get_actions(self):
        return list(self.mdp_prob_dict.keys())[0].get_actions()

    def get_num_state_feats(self):
        return list(self.mdp_prob_dict.keys())[0].get_num_state_feats()

    def get_gamma(self):
        '''
        Notes:
            Not all MDPs in the distribution are guaranteed to share gamma.
        '''
        return list(self.mdp_prob_dict.keys())[0].get_gamma()

    def get_reward_func(self, avg=True):
        if avg:
            self.get_average_reward_func()
        else:
            self.get_all_mdps()[0].get_reward_func()

    def get_average_reward_func(self):
        def _avg_r_func(s, a):
            r = 0.0
            for m in self.mdp_prob_dict.keys():
                r += m.reward_func(s, a) * self.mdp_prob_dict[m]
            return r
        return _avg_r_func

    def get_init_state(self):
        '''
        Notes:
            Not all MDPs in the distribution are guaranteed to share init states.
        '''
        return list(self.mdp_prob_dict.keys())[0].get_init_state()

    def get_num_mdps(self):
        return len(self.mdp_prob_dict.keys())

    def get_mdps(self):
        return self.mdp_prob_dict.keys()

    def get_prob_of_mdp(self, mdp):
        if mdp in self.mdp_prob_dict.keys():
            return self.mdp_prob_dict[mdp]
        else:
            return 0.0

    def set_gamma(self, new_gamma):
        for mdp in self.mdp_prob_dict.keys():
            mdp.set_gamma(new_gamma)

    def sample(self, k=1):
        '''
        Args:
            k (int)

        Returns:
            (List of MDP): Samples @k mdps without replacement.
        '''

        sampled_mdp_id_list = np.random.multinomial(k, list(self.mdp_prob_dict.values())).tolist()
        indices = [i for i, x in enumerate(sampled_mdp_id_list) if x > 0]

        if k == 1:
            return list(self.mdp_prob_dict.keys())[indices[0]]

        mdps_to_return = []

        for i in indices:
            for copies in range(sampled_mdp_id_list[i]):
                mdps_to_return.append(list(self.mdp_prob_dict.keys())[i])

        return mdps_to_return
        
    def __str__(self):
        '''
        Notes:
            Not all MDPs are guaranteed to share a name (for instance, might include dimensions).
        '''
        return "lifelong-" + str(list(self.mdp_prob_dict.keys())[0])

def main():
    # Simple test code.
    from simple_rl.tasks import GridWorldMDP

    mdp_distr = {}
    height, width = 8, 8
    prob_list = [0.0, 0.1, 0.2, 0.3, 0.4]

    for i in range(len(prob_list)):
        next_mdp = GridWorldMDP(width=width, height=width, init_loc=(1, 1), goal_locs=r.sample(zip(range(1, width + 1), [height] * width), 2), is_goal_terminal=True)

        mdp_distr[next_mdp] = prob_list[i]

    m = MDPDistribution(mdp_distr)
    m.sample()

if __name__ == "__main__":
    main()

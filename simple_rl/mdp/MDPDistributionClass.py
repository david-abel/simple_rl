''' MDPDistributionClass.py: Contains the MDP Distribution Class. '''

# Python imports.
import numpy as np
import random as rnd
from collections import defaultdict

class MDPDistribution(object):
    ''' Class for distributions over MDPs. '''

    def __init__(self, mdp_prob_dict):
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

        self.mdp_prob_dict = mdp_prob_dict

    def get_all_mdps(self, prob_threshold=0):
        '''
        Args:
            prob_threshold (float)

        Returns:
            (list): Contains all mdps in the distribution with Pr. > @prob_threshold.
        '''
        return [mdp for mdp in self.mdp_prob_dict.keys() if self.mdp_prob_dict[mdp] > prob_threshold]

    def get_actions(self):
        return self.mdp_prob_dict.keys()[0].get_actions()

    def get_gamma(self):
        '''
        Notes:
            Not all MDPs in the distribution are guaranteed to share gamma.
        '''
        return self.mdp_prob_dict.keys()[0].get_gamma()

    def get_init_state(self):
        '''
        Notes:
            Not all MDPs in the distribution are guaranteed to share init states.
        '''
        return self.mdp_prob_dict.keys()[0].get_init_state()

    def get_num_mdps(self):
        return len(self.mdp_prob_dict.keys())

    def get_mdps(self):
        return self.mdp_prob_dict.keys()

    def sample(self, k=1):
        '''
        Args:
            k (int)

        Returns:
            (List of MDP): Samples @k mdps without replacement.
        '''

        sampled_mdp_id_list = np.random.multinomial(k, self.mdp_prob_dict.values()).tolist()
        indices = [i for i, x in enumerate(sampled_mdp_id_list) if x > 0]

        if k == 1:
            return self.mdp_prob_dict.keys()[indices[0]]

        mdps_to_return = []

        for i in indices:
            for copies in xrange(sampled_mdp_id_list[i]):
                mdps_to_return.append(self.mdp_prob_dict.keys()[i])

        return mdps_to_return
        

    def __str__(self):
        '''
        Notes:
            Not all MDPs are guaranteed to share a name (for instance, might include dimensions).
        '''
        return "multitask-" + str(self.mdp_prob_dict.keys()[0])

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
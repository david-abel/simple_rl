# Python imports.
import numpy as np

# Other imports.
from simple_rl.mdp.StateClass import State
from simple_rl.abstraction.state_abs.StateAbstractionClass import StateAbstraction

class ProbStateAbstraction(StateAbstraction):

    def __init__(self, abstr_dist):
        '''
        Args:
            abstr_dist (dict): Represents Pr(s_phi | phi)
                Key: state
                Val: dict
                    Key: s_phi (simple_rl.State)
                    Val: probability (float)
        '''
        self.abstr_dist = abstr_dist

    def phi(self, state):
        '''
        Args:
            state (State)

        Returns:
            state (State)
        '''

        sampled_s_phi_index = np.random.multinomial(1, self.abstr_dist[state].values()).tolist().index(1)
        abstr_state = self.abstr_dist[state].keys()[sampled_s_phi_index]

        return abstr_state

def convert_prob_sa_to_sa(prob_sa):
    '''
    Args:
        prob_sa (simple_rl.state_abs.ProbStateAbstraction)

    Returns:
        (simple_rl.state_abs.StateAbstraction)
    '''
    new_phi = {}

    for s_g in prob_sa.abstr_dist.keys():
        new_phi[s_g] = prob_sa.abstr_dist[s_g].keys()[prob_sa.abstr_dist[s_g].values().index(max(prob_sa.abstr_dist[s_g].values()))]

    return StateAbstraction(new_phi)

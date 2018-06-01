# Python imports.
import numpy as np

# Other imports.
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

        # abstr_state = State(self._phi[state])
        # abstr_state.set_terminal(state.is_terminal())

        return abstr_state

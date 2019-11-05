# Python imports.
from collections import defaultdict

# Other imports.
from simple_rl.mdp.StateClass import State

class BeliefState(State):
    '''
     Abstract class defining a belief state, i.e a probability distribution over states.
    '''
    def __init__(self, belief_distribution):
        '''
        Args:
            belief_distribution (defaultdict)
        '''
        self.distribution = belief_distribution
        State.__init__(self, data=list(belief_distribution.values()))

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'BeliefState::' + str(self.distribution)

    def belief(self, state):
        '''
        Args:
            state (State)
        Returns:
            belief[state] (float): probability that agent is in state
        '''
        return self.distribution[state]

    def sample(self, sampling_method='max'):
        '''
        Returns:
            sampled_state (State)
        '''
        if sampling_method == 'max':
            return max(self.distribution, key=self.distribution.get)
        raise NotImplementedError('Sampling method {} not implemented yet'.format(sampling_method))

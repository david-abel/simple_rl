from collections import defaultdict
from simple_rl.planning.ValueIterationClass import ValueIteration

class BeliefUpdater(object):
    ''' Wrapper class for different methods for belief state updates in POMDPs. '''

    def __init__(self, mdp, transition_func, reward_func, observation_func, updater_type='discrete'):
        '''
        Args:
            mdp (POMDP)
            transition_func: T(s, a) --> s'
            reward_func: R(s, a) --> float
            observation_func: O(s, a) --> z
            updater_type (str)
        '''
        self.reward_func = reward_func
        self.updater_type = updater_type

        # We use the ValueIteration class to construct the transition and observation probabilities
        self.vi = ValueIteration(mdp, sample_rate=500)

        self.transition_probs = self.construct_transition_matrix(transition_func)
        self.observation_probs = self.construct_observation_matrix(observation_func, transition_func)

        if updater_type == 'discrete':
            self.updater = self.discrete_filter_updater
        elif updater_type == 'kalman':
            self.updater = self.kalman_filter_updater
        elif updater_type == 'particle':
            self.updater = self.particle_filter_updater
        else:
            raise AttributeError('updater_type {} did not conform to expected type'.format(updater_type))

    def discrete_filter_updater(self, belief, action, observation):
        def _compute_normalization_factor(bel):
            return sum(bel.values())

        def _update_belief_for_state(b, sp, T, O, a, z):
            return O[sp][z] * sum([T[s][a][sp] * b[s] for s in b])

        new_belief = defaultdict()
        for sprime in belief:
            new_belief[sprime] = _update_belief_for_state(belief, sprime, self.transition_probs, self.observation_probs, action, observation)

        normalization = _compute_normalization_factor(new_belief)

        for sprime in belief:
            if normalization > 0: new_belief[sprime] /= normalization

        return new_belief

    def kalman_filter_updater(self, belief, action, observation):
        pass

    def particle_filter_updater(self, belief, action, observation):
        pass

    def construct_transition_matrix(self, transition_func):
        '''
        Create an MLE of the transition probabilities by sampling from the transition_func
        multiple times.
        Args:
            transition_func: T(s, a) -> s'

        Returns:
            transition_probabilities (defaultdict): T(s, a, s') --> float
        '''
        self.vi._compute_matrix_from_trans_func()
        return self.vi.trans_dict

    def construct_observation_matrix(self, observation_func, transition_func):
        '''
        Create an MLE of the observation probabilities by sampling from the observation_func
        multiple times.
        Args:
            observation_func: O(s) -> z
            transition_func: T(s, a) -> s'

        Returns:
            observation_probabilities (defaultdict): O(s, z) --> float
        '''
        def normalize_probabilities(odict):
            norm_factor = sum(odict.values())
            for obs in odict:
                odict[obs] /= norm_factor
            return odict

        obs_dict = defaultdict(lambda:defaultdict(float))
        for state in self.vi.get_states():
            for action in self.vi.mdp.actions:
                for sample in range(self.vi.sample_rate):
                    observation = observation_func(state, action)
                    next_state = transition_func(state, action)
                    obs_dict[next_state][observation] += 1. / self.vi.sample_rate
        for state in self.vi.get_states():
            obs_dict[state] = normalize_probabilities(obs_dict[state])
        return obs_dict

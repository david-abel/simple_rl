# Belief state and distribution

from simple_rl.mdp.StateClass import State
from simple_rl.planning.PlannerClass import Planner
import numpy as np
import random
import sys

EPSILON = 1e-6

# Difference between belief state and belief distribution
# is that one can iterate over all the states in a
# belief state, while that is not taken care of in a
# belief distribution. 

class BeliefState(State):
    '''
     Abstract class defining a belief state,
    i.e a probability distribution over states.
    '''
    def __init__(self, belief_distribution):
        '''
        Args:
            belief_distribution (defaultdict)
        '''
        self.distribution = belief_distribution
        State.__init__(self, data=[])

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'BeliefState::' + str(self.distribution)

    def update(self, *params, **kwargs):
        self.distribution = self.distribution.update(*params, **kwargs)

    def sample(self, sampling_method='max'):
        '''
        Returns:
            sampled_state (State)
        '''
        if sampling_method == 'random':  # random uniform
            return self.distribution.random()
        elif sampling_method == 'max':
            return self.distribution.mpe()
        raise NotImplementedError('Sampling method {} not implemented yet'.format(sampling_method))


class BeliefDistribution:
    def __getitem__(self, state):
        pass
    def __setitem__(self, state, value):
        pass
    def __hash__(self):
        pass
    def __eq__(self, other):
        pass
    def __str__(self):
        pass
    def mpe(self):
        pass
    def random(self):
        # Sample a state based on the underlying belief distribution
        pass
    def add(self, state):
        pass
    def get_histogram(self):
        """Returns a dictionary from state to probability"""
        pass
    def update(self, real_action, real_observation, pomdp, **kwargs):
        pass
        
class BeliefDistribution_Particles(BeliefDistribution):
    def __init__(self, particles):
        self._particles = particles  # each particle is a state
        
    @property
    def particles(self):
        return self._particles

    def __str__(self):
        hist = self.get_histogram()
        hist = [(k,hist[k]) for k in list(reversed(sorted(hist, key=hist.get)))[:5]]
        return str(hist)

    def __len__(self):
        return len(self._particles)
    
    def __getitem__(self, state):
        belief = 0
        for s in self._particles:
            if s == state:
                belief += 1
        return belief / len(self._particles)
    
    def __setitem__(self, state, value):
        """Assume that value is between 0 and 1"""
        particles = [s for s in self._particles if s != state]
        len_not_state = len(particles)
        amount_to_add = value * len_not_state / (1 - value)
        for i in range(amount_to_add):
            particles.append(state)
        self._particles = particles
        
    def __hash__(self):
        if len(self._particles) == 0:
            return hash(0)
        else:
            # if the state space is large, a random particle would differentiate enough
            indx = random.randint(0, len(self._particles-1))
            return hash(self._particles[indx])
        
    def __eq__(self, other):
        if not isinstance(other, BeliefDistribution_Particles):
            return False
        else:
            if len(self._particles) != len(other.praticles):
                return False
            state_counts_self = {}
            state_counts_other = {}
            for s in self._particles:
                if s not in state_counts_self:
                    state_counts_self[s] = 0
                state_counts_self[s] += 1
            for s in other.particles:
                if s not in state_counts_self:
                    return False
                if s not in state_counts_other:
                    state_counts_other[s] = 0
                state_counts_other[s] += 1
            return state_counts_self == state_counts_other

    def mpe(self):
        max_counts = 0
        mpe_state = None
        state_counts_self = {}
        for s in self._particles:
            if s not in state_counts_self:
                state_counts_self[s] = 0
            state_counts_self[s] += 1
            if state_counts_self[s] > max_counts:
                max_counts = state_counts_self[s]
                mpe_state = s
        return mpe_state

    def random(self):
        if len(self._particles) > 0:
            return random.choice(self._particles)
        else:
            return None

    def add(self, particle):
        self._particles.append(particle)

    def get_histogram(self):
        state_counts_self = {}
        for s in self._particles:
            if s not in state_counts_self:
                state_counts_self[s] = 0
            state_counts_self[s] += 1
        for s in state_counts_self:
            state_counts_self[s] = state_counts_self[s] / len(self._particles)
        return state_counts_self


class BeliefDistribution_Histogram:
    def __init__(self, histogram):
        """Histogram is a dictionary mapping from state to probability"""
        if not (isinstance(histogram, dict)):
            raise ValueError("Unsupported histogram representation! %s" % str(type(histogram)))
        self._histogram = histogram
        # ticks = (np.arange(self._histogram.shape[i]) for i in range(len(self._histogram.shape)))
        # indices = np.meshgrid(*ticks)
        # self._vertices = np.stack([*indices], axis=-1).reshape(-1, len(self._histogram.shape))

    @property
    def histogram(self):
        return self._histogram

    def __str__(self):
        if isinstance(self._histogram, dict):
            return str([(k,self._histogram[k])
                        for k in list(reversed(sorted(self._histogram, key=self._histogram.get)))[:5]])

    def __len__(self):
        return len(self._histogram)

    def __getitem__(self, state):
        if state in self._histogram:
            return self._histogram[state]
        else:
            return 0

    def __setitem__(self, state, value):
        self._histogram[state] = value

    def __hash__(self):
        if len(self._histogram) == 0:
            return hash(0)
        else:
            # if the state space is large, a random state would differentiate enough
            state = self.random()
            return hash(self._histogram[state])

    def __iter__(self):
        return iter(self._histogram)

    def mpe(self):
        state = max(self._histogram, key=self._histogram.get)
        return state

    def random(self):
        """Randomly sample a state based on the probability in the histogram"""
        candidates = list(self._histogram.keys())
        prob_dist = []
        for state in candidates:
            prob_dist.append(self._histogram[state])
        if sys.version_info[0] >= 3 and sys.version_info[1] >= 6:
            # available in Python 3.6+
            random.choices(candidates, weights=prob_dist, k=1)
        else:
            return np.random.choice(candidates, 1, p=prob_dist)[0]

    def is_normalized(self):
        """Returns true if this distribution is normalized"""
        prob_sum = sum(self._histogram[state] for state in self._histogram)
        return abs(1.0-prob_sum) < EPSILON

    def get_histogram(self):
        return self._histogram


class RandomPlanner(Planner):

    class DummyDistribution(BeliefDistribution):
        def __init__(self):
            super().__init__()
        def mpe(self):
            return None
        def random(self):
            return None
        def get_histogram(self):
            return {}
        def update(self, real_action, real_observation, pomdp, **kwargs):
            return self
        def __str__(self):
            return "DummyDistribution"
    
    def __init__(self, pomdp, name="random"):
        super().__init__(pomdp, name=name, init_default_fields=False)
        self._pomdp = pomdp

    def plan_and_execute_next_action(self):
        action = random.choice(self._pomdp.actions)
        return self.execute_next_action(action)

    def execute_next_action(self, action):
        reward, observation = self._pomdp.execute_agent_action_update_belief(action)
        return action, reward, observation

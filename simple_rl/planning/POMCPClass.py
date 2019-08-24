# POMCP implementation

from collections import defaultdict
from simple_rl.planning.PlannerClass import Planner

import sys
import time
import random
import math
import copy


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
        

class POMCP_Particles(BeliefDistribution_Particles):
    """Particle representation of the belief with update algorithm
    described in the POMCP paper"""
    def __init__(self, particles):
        super().__init__(particles)
        
    def update(self, real_action, real_observation, pomdp, num_particles):
        """
        Update the belief, given real action, real observation and a generator.

        generator (G)
        num_particles (int) number of particles wanted in the updated belief distribution.
        """
        def generator(state, action):
            '''
            (s', o', r) ~ G(s, a)
            '''
            next_state = pomdp.transition_func(state, action)
            observation = pomdp.observation_func(next_state, action)
            reward = pomdp.reward_func(state, action, next_state)
            return next_state, observation, reward
        
        # Update the belief
        particles = []
        for state in self._particles:
            next_state, observation, reward = generator(state, real_action)
            if observation == real_observation:
                particles.append(next_state)
        if len(particles) == num_particles:
            self._particles = particles
            return

        if len(particles) > num_particles:
            import pdb; pdb.set_trace()  # This is unexpected to happen.

        if len(particles) == 0:
            # No state supports the real observation. Reinvigorate the particles
            # so that the sampled states support the real_observation.
            print("Belief is empty. Particle depletion.")
            print("Resampling particles to support real observation")
            # TODO: this is not smart
            for state in pomdp.cur_belief:
                next_state, observation, reward = generator(state, real_action)
                if observation == real_observation:
                    particles.append(next_state)
                    break

            if len(particles) == 0:  # If still not able to help, fail.
                raise ValueError("No help. Particle depletion.")
                
        # If not enough particles, introduce artificial noise to existing particles (reinvigoration)
        print("Particle reinvigoration for %d particles" % (num_particles - len(particles)))
        reinvigorated_particles = list(particles)
        for i in range(num_particles - len(particles)):
            # need to make a copy otherwise the transform affects states in 'particles'
            state = copy.deepcopy(random.choice(particles))
            pomdp.add_transform(state)
            reinvigorated_particles.append(state)
        assert len(reinvigorated_particles) == num_particles
        self._particles = reinvigorated_particles

        print(self)

        
class POMCP(Planner):

    def print_tree_helper(self, root, depth, max_depth=None):
        if max_depth is not None and depth >= max_depth:
            return
        print("%s%s" % ("    "*depth, str(root)))
        for c in root.children:
            self.print_tree_helper(root[c], depth+1)
            
    def print_tree(self, max_depth=None):
        self.print_tree_helper(self._tree, 0, max_depth=max_depth)


    class TreeNode:
        def __init__(self):
            self.children = {}
        def __getitem__(self, key):
            return self.children.get(key,None)
        def __setitem__(self, key, value):
            self.children[key] = value
        def __contains__(self, key):
            return key in self.children

    class QNode(TreeNode):
        def __init__(self, action, num_visits, value):
            self.num_visits = num_visits
            self.value = value
            self.action = action
            self.children = {}  # o -> VNode
        def __str__(self):
            return "QNode(%.3f, %.3f | %s)->%s" % (self.num_visits, self.value, str(self.children.keys()), str(self.action))
        def __repr__(self):
            return self.__str__()

    class VNode(TreeNode):
        def __init__(self, num_visits, value, belief):
            self.num_visits = num_visits
            self.value = value
            self.belief = belief
            self.children = {}  # a -> QNode
            self.parent_ao = None  # (action, observation)
        def __str__(self):
            return "VNode(%.3f, %.3f, %d | %s)" % (self.num_visits, self.value, len(self.belief),
                                              str(self.children.keys()))
        def __repr__(self):
            return self.__str__()
    
    def __init__(self, pomdp,
                 num_particles=1000, max_depth=5, max_time=3.,
                 gamma=0.99, exploration_const=math.sqrt(2),
                 num_visits_init=1, value_init=0, rollout_policy=None):
        # The tree is a chained
        self._pomdp = pomdp
        self._tree = None
        self._num_particles = num_particles # K
        self._max_depth = max_depth
        self._max_time = max_time
        self._num_visits_init = num_visits_init
        self._value_init = value_init
        self._rollout_policy = rollout_policy
        self._gamma = gamma
        self._exploration_const = exploration_const
        self._history = ()

        # Initialize the tree for empty history; (adding K particles in it)
        belief = POMCP_Particles([])
        while len(belief) < self._num_particles:
            state = self._pomdp.init_belief.sample(sampling_method='random')
            belief.add(state)
        self._tree = POMCP.VNode(self._num_visits_init, self._value_init, belief)
        self._expand_vnode(self._tree)

    @property
    def gamma(self):
        return self._gamma

    def _expand_vnode(self, vnode):
        for action in self._pomdp.actions:
            if vnode[action] is None:
                history_action_node = POMCP.QNode(action, self._num_visits_init, self._value_init)
                vnode[action] = history_action_node

    def _simulate(self, state, root, parent, observation, depth):
        if depth > self._max_depth:
            return 0
        if root is None:
            root = POMCP.VNode(self._num_visits_init, self._value_init, POMCP_Particles([]))
            if parent is not None:
                parent[observation] = root
            self._expand_vnode(root)
            return self._rollout(state, root, depth)
        action = self._ucb(root)
        next_state, observation, reward = self._sample_generative_model(state, action)
        R = reward + self._gamma*self._simulate(next_state, root[action][observation], root[action], observation, depth+1)
        root.belief.add(state)
        root.num_visits += 1
        root[action].num_visits += 1
        root[action].value = root[action].value + (R - root[action].value) / (root[action].num_visits)
        return R
        
    def _rollout(self, state, root, depth):
        if depth > self._max_depth:
            return 0
        if self._rollout_policy is None:
            action = random.choice(self._pomdp.actions)
        else:
            action = self._rollout_policy(root, self._pomdp.actions)
        next_state, observation, reward = self._sample_generative_model(state, action)
        if root[action] is None:
            history_action_node = POMCP.QNode(action, self._num_visits_init, self._value_init)
            root[action] = history_action_node
        if observation not in root[action]:
            root[action][observation] = POMCP.VNode(self._num_visits_init, self._value_init, POMCP_Particles([]))
            root[action][observation].parent_ao = (action, observation)
            self._expand_vnode(root[action][observation])
        return reward + self._gamma * self._rollout(next_state, root[action][observation], depth+1)

    def _ucb(self, root):
        best_action, best_value = None, float('-inf')
        for action in self._pomdp.actions:
            if action in root:
                val = root[action].value + \
                    self._exploration_const * math.sqrt(math.log(root.num_visits) / root[action].num_visits)
                if val > best_value:
                    best_action = action
                    best_value = val
        return best_action

    def search(self, history):
        """It is assumed that the given history corresponds to self._tree.
        Meaning that self._tree is updated (i.e. tree truncated) as history
        progresses."""
        start_time = time.time()
        while time.time() - start_time < self._max_time:
            if len(history) == 0:
                state = self._pomdp.init_belief.sample(sampling_method='max')
            else:
                state = self._tree.belief.random()
            self._simulate(state, self._tree, None, None, 0)
            
        best_action, best_value = None, float('-inf')            
        for action in self._pomdp.actions:
            if self._tree[action] is not None:
                if self._tree[action].value > best_value:
                    best_value = self._tree[action].value
                    best_action = action
        return best_action

    def _sample_generative_model(self, state, action):
        '''
        (s', o, r) ~ G(s, a)
        '''
        next_state = self._pomdp.transition_func(state, action)
        observation = self._pomdp.observation_func(next_state, action)
        reward = self._pomdp.reward_func(state, action, next_state)
        return next_state, observation, reward

    def plan_and_execute_next_action(self):
        """This function is supposed to plan an action, execute that action,
        and update the belief"""
        action = self.search(self._history)
        # execute action and update belief
        return self.execute_next_action(action)

    def execute_next_action(self, action):
        """Execute the given action, and update the belief"""
        reward, observation = self._pomdp.execute_agent_action(action)
        self._pomdp.update_belief(action, observation, num_particles=self._num_particles)
        self._history += ((action, observation),)
        # Truncate the tree
        self._tree = self._tree[action][observation]
        if self._tree is None:
            # observation was never encountered in simulation.
            self._tree = POMCP.VNode(self._num_visits_init, self._value_init, POMCP_Particles([]))
            self._expand_vnode(self._tree)
        # Sync POMDP's belief to tree 
        self._tree.belief = copy.deepcopy(self._pomdp.cur_belief.distribution)
        return action, reward, observation        

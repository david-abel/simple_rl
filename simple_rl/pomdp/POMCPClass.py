# POMCP implementation

from collections import defaultdict
from simple_rl.planning.PlannerClass import Planner

from simple_rl.pomdp.shared import BeliefDistribution_Particles

# from simple_rl.planning.POMCPClass import POMCP
import sys
import time
import random
import math
import copy


class POMCP_Particles(BeliefDistribution_Particles):
    """Particle representation of the belief with update algorithm
    described in the POMCP paper"""
    def __init__(self, particles):
        super().__init__(particles)
        
    def update(self, real_action, real_observation, pomdp, num_particles,
               observation_based_resampling=True):
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
        new_belief = POMCP_Particles([])
        for state in self._particles:
            next_state, observation, reward = generator(state, real_action)
            if observation == real_observation:
                new_belief.add(next_state)
        if len(new_belief) == num_particles:
            return new_belief

        if len(new_belief) > num_particles:
            import pdb; pdb.set_trace()  # This is unexpected to happen.

        if len(new_belief) == 0:
            # No state supports the real observation. Reinvigorate the particles
            # so that the sampled states support the real_observation.
            print("Belief is empty. Particle depletion.")
            print("Resampling particles to support real observation")
            if observation_based_resampling:
                # TODO: this is not smart
                for state in pomdp.cur_belief:
                    next_state, observation, reward = generator(state, real_action)
                    if observation == real_observation:
                        new_belief.add(next_state)
                        break
            else:
                # random
                print("Randomly resample all particles.")
                for i in range(num_particles):
                    new_belief.add(pomdp.cur_belief.sample(sampling_method='random'))

            if len(new_belief) == 0:  # If still not able to help, fail.
                raise ValueError("No help. Particle depletion.")
                
        # If not enough particles, introduce artificial noise to existing particles (reinvigoration)
        num_reinvigorate = num_particles - len(new_belief)
        reinvigorated_belief = POMCP_Particles(list(new_belief.particles))
        print("Particle reinvigoration for %d particles" % (num_reinvigorate))
        for i in range(num_reinvigorate):
            # need to make a copy otherwise the transform affects states in 'particles'
            state = copy.deepcopy(random.choice(new_belief.particles))
            pomdp.add_transform(state)
            reinvigorated_belief.add(state)
        assert len(reinvigorated_belief) == num_particles
        return reinvigorated_belief

        
class POMCP(Planner):

    def print_tree_helper(self, root, depth, max_depth=None):
        if max_depth is not None and depth >= max_depth:
            return
        print("%s%s" % ("    "*depth, str(root)))
        for c in root.children:
            if root[c].num_visits > 1:
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
                 num_visits_init=1, value_init=0, rollout_policy=None,
                 observation_based_resampling=True):
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
        self._observation_based_resampling = observation_based_resampling
        self._history = ()

        # if not isinstance(self._pomdp.cur_belief.distribution, POMCP_Particles):
        #     raise ValueError("The belief representation of the POMDP must be POMCP_Particles.")

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
                print("action %s: %.3f" % (str(action), self._tree[action].value))
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
        reward, observation = self._pomdp.execute_agent_action_update_belief(action, num_particles=self._num_particles,
                                                                             observation_based_resampling=self._observation_based_resampling)
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

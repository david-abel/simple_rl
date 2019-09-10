# OOPOMCP implementation

from simple_rl.planning.PlannerClass import Planner
from simple_rl.pomdp.shared import BeliefState, BeliefDistribution_Histogram
import math
import time
import copy

class OOPOMCP_Histogram(BeliefDistribution_Histogram):
    """Represents the distribution of ONE SINGLE object"""
    def __init__(self, objid, histogram):
        self.objid = objid
        super().__init__(histogram)
        
    def update(self, real_action, real_observation, **kwargs):
        """Update bo(s), the belief distribution for SINGLE object o"""
        observation_model = kwargs.get("observation_model", None)
        transition_model = kwargs.get("transition_model", None)
        robot_state = kwargs.get("robot_state", None)
        next_robot_state = kwargs.get("next_robot_state", None)

        new_distribution = OOPOMCP_Histogram(self.objid, copy.deepcopy(self._histogram))
        total_prob = 0
        for next_object_state in new_distribution.get_histogram():
            observation_prob = observation_model(self.objid,
                                                 real_observation,
                                                 next_object_state,
                                                 next_robot_state,
                                                 real_action)
            transition_prob = 0
            for object_state in self._histogram: 
                transition_prob += transition_model(self.objid,
                                                    next_object_state,
                                                    next_robot_state,
                                                    object_state,
                                                    robot_state,
                                                    real_action) * self._histogram[object_state]
            new_distribution[next_object_state] = observation_prob * transition_prob
            total_prob += new_distribution[next_object_state]
        # Normalize
        for state in new_distribution:
            if total_prob > 0:
                new_distribution[state] /= total_prob
        
        return new_distribution

        
class OOPOMCP(Planner):

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
        def __init__(self, num_visits, value):
            self.num_visits = num_visits
            self.value = value
            self.children = {}  # a -> QNode
            self.parent_ao = None  # (action, observation)
        def __str__(self):
            return "VNode(%.3f, %.3f | %s)" % (self.num_visits, self.value, str(self.children.keys()))
                                              
        def __repr__(self):
            return self.__str__()

    def __init__(self, oopomdp, max_depth=5, max_time=3.,
                 num_visits_init=1, value_init=0, rollout_policy=None,
                 gamma=0.99, gamma_epsilon=1e-6, exploration_const=math.sqrt(2)):
        self._oopomdp = oopomdp
        self._max_depth = max_depth
        self._max_time = max_time
        self._num_visits_init = num_visits_init
        self._value_init = value_init                 
        self._rollout_policy = rollout_policy
        self._gamma = gamma
        self._gamma_epsilon = gamma_epsilon
        self._exploration_const = exploration_const
        self._history = ()

        self._tree = OOPOMCP.VNode(self._num_visits_init, self._value_init)
        self._expand_vnode(self._tree)

    @property
    def gamma(self):
        return self._gamma

    def _expand_vnode(self, vnode):
        for action in self._oopomdp.actions:
            if vnode[action] is None:
                history_action_node = OOPOMCP.QNode(action, self._num_visits_init, self._value_init)
                vnode[action] = history_action_node

    def _simulate(self, state, root, parent, observation, depth):
        if self._gamma**depth < self._gamma_epsilon or depth > self._max_depth:
            return 0
        if root is None:
            root = OOPOMCP.VNode(self._num_visits_init, self._value_init)
            if parent is not None:
                parent[observation] = root
            self._expand_vnode(root)
            return self._rollout(state, root, depth)
        action = self._ucb(root)
        # print("SIMULATE %d" % depth)
        next_state, observation, reward = self._sample_generative_model(state, action)
        R_future = self._simulate(next_state, root[action][observation], root[action], observation, depth+1)
        R = reward + self._gamma*R_future
        
        root.num_visits += 1
        root[action].num_visits += 1
        root[action].value = root[action].value + (R - root[action].value) / (root[action].num_visits)
        return R
        
    def _rollout(self, state, root, depth):
        if self._gamma**depth < self._gamma_epsilon or depth > self._max_depth:
            return 0
        if self._rollout_policy is None:
            action = random.choice(self._oopomdp.actions)
        else:
            action = self._rollout_policy(root, self._oopomdp.actions)
            
        next_state, observation, reward = self._sample_generative_model(state, action)
        if root[action] is None:
            history_action_node = OOPOMCP.QNode(action, self._num_visits_init, self._value_init)
            root[action] = history_action_node
        if observation not in root[action]:
            root[action][observation] = OOPOMCP.VNode(self._num_visits_init, self._value_init)
            root[action][observation].parent_ao = (action, observation)
            self._expand_vnode(root[action][observation])
        return reward + self._gamma * self._rollout(next_state, root[action][observation], depth+1)

    def _ucb(self, root):
        best_action, best_value = None, float('-inf')
        for action in self._oopomdp.actions:
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
        # Instead of specifying "number of simulations" as in the paper, just do time-out.
        start_time = time.time()
        while time.time() - start_time < self._max_time:
        # for i in range(100):
            state = self._oopomdp.cur_belief.sample(sampling_method='random')
            self._simulate(state, self._tree, None, None, 0)

        best_action, best_value = None, float('-inf')            
        for action in self._oopomdp.actions:
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
        next_state = self._oopomdp.transition_func(state, action)
        observation = self._oopomdp.observation_func(next_state, action)
        reward = self._oopomdp.reward_func(state, action, next_state)
        return next_state, observation, reward

    def plan_and_execute_next_action(self):
        """This function is supposed to plan an action, execute that action,
        and update the belief"""
        action = self.search(self._history)
        # execute action and update belief
        return self.execute_next_action(action)

    def execute_next_action(self, action):
        """Execute the given action, and update the belief"""
        reward, observation = self._oopomdp.execute_agent_action_update_belief(action)
        self._history += ((action, observation),)
        # Truncate the tree
        self._tree = self._tree[action][observation]
        if self._tree is None:
            # observation was never encountered in simulation.
            self._tree = OOPOMCP.VNode(self._num_visits_init, self._value_init)
            self._expand_vnode(self._tree)
        return action, reward, observation        

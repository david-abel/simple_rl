# Python imports.
from __future__ import print_function
from collections import defaultdict
import random
import numpy as np
import pdb
import time

# Other imports.
from simple_rl.planning.PlannerClass import Planner
from simple_rl.mdp.StateClass import State
from simple_rl.pomdp.BeliefStateClass import BeliefState
from simple_rl.pomdp.POMDPClass import POMDP

class POMCP(Planner):
    def __init__(self, pomdp, max_time_duration=3., max_rollout_depth=20, exploration_param=np.sqrt(2),
                 init_visits=0, init_value=0.):
        '''
        Args:
            pomdp (POMDP)
            max_time_duration (float)
            max_rollout_depth (int)
            exploration_param (float)
            init_visits (int)
            init_value (float)
        '''
        self.initial_belief = pomdp.init_belief

        self.search_tree = defaultdict()

        self.epsilon = 0.01
        self.max_time_duration = max_time_duration
        self.max_rollout_depth = max_rollout_depth
        self.exploration_param = exploration_param
        self.init_visits = init_visits
        self.init_value = init_value

        Planner.__init__(self, pomdp, name='pomcp')

    def run(self, verbose=True):
        discounted_sum_rewards = 0.0
        num_iter = 0
        history = ()
        policy = defaultdict()
        self.mdp.reset()
        while not self.mdp.is_in_goal_state():
            print('Calling _search() from history = {}'.format(history))
            action = self._search(history)
            reward, observation = self.mdp.execute_agent_action(action)
            policy[history] = action
            if verbose: print('From history {}, took action {}'.format(history, action))
            history = history + ((action, observation),)
            discounted_sum_rewards += ((self.gamma ** num_iter) * reward)
            num_iter += 1
        return discounted_sum_rewards, policy

    def enable_online_mode(self):
        self._history = ()
        self._policy = defaultdict()
        self._discounted_sum_rewards = 0
        self._num_iter = 0
        self.mdp.reset()
        self._online_mode = True

    def plan_and_execute_next_action(self):
        """This function is supposed to plan an action, execute that action,
        and update the belief"""
        if hasattr(self, "_online_mode") and self._online_mode is True:
            action = self._search(self._history)

            reward, observation = self.mdp.execute_agent_action(action)
            self._policy[self._history] = action
            self._history = self._history + ((action, observation),)
            self._discounted_sum_rewards += ((self.gamma ** self._num_iter) * reward)
            self._num_iter += 1
            return action, reward, observation

    def belief(self, state, history):
        '''
        bel_hat(s, h) = (1/K) * sum_(i=1, K){KroneckerDelta(s, B_i)}
        POMCP tracks the belief state by aggregating over the set of particles associated with the current history
        Args:
            state (State): the state over which we want to compute the belief probability
            history (tuple): the (action, observation) sequence observed so far

        Returns:
            belief_state (float): probability that we are in `state` given that we have seen `history`
        '''
        particles = self.search_tree[history][2] # type: list
        return particles.count(state) / float(len(particles))

    def _search(self, history):
        '''
        Main method in the POMCP algorithm.
        Args:
            history (tuple)

        Returns:
            action (str)
        '''
        start_time = time.time()
        while time.time() - start_time < self.max_time_duration:
            if len(history) == 0:
                state = self.initial_belief.sample(sampling_method='random')
            else:
                particles = self.search_tree[history][2]
                state = random.choice(particles)
            self._simulate(state, history, 0)

        return self._greedy_action(history)

    def _rollout(self, state, history, depth):
        '''
        Args:
            state (State)
            history (tuple)
            depth (int)

        Returns:
            value (float): Resulting discounted rewards from running MC simulations from state
        '''
        def _rollout_policy(h, available_actions):
            def _random_rollout_policy(actions):
                return random.choice(actions)
            return _random_rollout_policy(available_actions)

        if self.gamma ** depth < self.epsilon\
           or (isinstance(state, State) and state.is_terminal()):
            return 0.

        action = _rollout_policy(history, self.mdp.actions)
        next_state, observation, reward = self._sample_generative_model(state, action)

        new_history = history + ((action, observation),)
        return reward + (self.gamma * self._rollout(next_state, new_history, depth+1))

    def _simulate(self, state, history, depth):
        '''
        Conduct an MC simulation from (state, history) until we reach some terminal state/condition.
        Args:
            state (State)
            history (tuple)
            depth (int)

        Returns:
            value (float): Resulting discounted rewards from running MC simulations from state
        '''
        if self.gamma ** depth < self.epsilon or depth >= self.max_rollout_depth:
            return 0.

        if history not in self.search_tree:
            # T(h) --> <N(h), V(h), B(h)>
            self.search_tree[history] = [self.init_visits, self.init_value, list()]
            for action in self.mdp.actions:
                history_action = history + ((action,),)
                # T(ha) --> <N(ha), Q(ha)>
                self.search_tree[history_action] = [self.init_visits, self.init_value]
            return self._rollout(state, history, depth)

        action = self._ucb_action(history)
        next_state, observation, reward = self._sample_generative_model(state, action)

        history_action = history + ((action,),)
        next_history = history + ((action, observation),)
        discounted_reward = reward + (self.gamma * self._simulate(next_state, next_history, depth+1))

        self.search_tree[history][2].append(state)
        self.search_tree[history][0] += 1
        self.search_tree[history_action][0] += 1
        self.search_tree[history_action][1] += (discounted_reward - self.search_tree[history_action][1]) / self.search_tree[history_action][0]

        return discounted_reward

    def _prune_search_tree(self, history, new_action, new_observation):
        '''
        Using _search(), we have determined the action to take from `history`. Copy over the tree starting from
        (history, (new_action, new_observation)) and ignore everything else
        Args:
            history (tuple)
            new_action (str)
            new_observation (str)
        '''
        pass

    def _ucb_action(self, history):
        '''
        Choose action based on the UCB algorithm.
        Args:
            history (tuple)

        Returns:
            action (str)
        '''
        augmented_qvalues = defaultdict()
        for action in self.mdp.actions:
            history_action = history + ((action,),)
            if self.search_tree[history_action][0] == 0:
                return action
            exploration_part = np.sqrt(np.log(self.search_tree[history][0]) / self.search_tree[history_action][0])
            augmented_qvalues[action] = self.search_tree[history_action][1] + (self.exploration_param * exploration_part)
        return max(augmented_qvalues, key=augmented_qvalues.get)

    def _greedy_action(self, history):
        '''
        Return action with the highest qvalue in the current history state
        Args:
            history (tuple)

        Returns:
            action (str)
        '''
        max_val = 0.
        best_action = self.mdp.actions[0]
        for action in self.mdp.actions:
            history_action = history + ((action,),)
            current_value = self.search_tree[history_action][1]
            if current_value > max_val:
                max_val = current_value
                best_action = action
        return best_action

    def _sample_generative_model(self, state, action):
        '''
        (s', o, r) ~ G(s, a)
        Args:
            state (State)
            action (str)

        Returns:
            next_state (State)
            observation (str)
            reward (float)
        '''
        next_state = self.mdp.transition_func(state, action)
        observation = self.mdp.observation_func(state, action)
        reward = self.mdp.reward_func(state, action, next_state)

        return next_state, observation, reward

if __name__ == '__main__':
    from simple_rl.tasks.maze_1d.Maze1DPOMDPClass import Maze1DPOMDP
    import time
    maze = Maze1DPOMDP()
    p = POMCP(maze, max_time_duration=3., max_rollout_depth=20)
    r, pi = p.run(verbose=True)

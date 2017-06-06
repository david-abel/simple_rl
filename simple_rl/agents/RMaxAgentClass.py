''' RMaxAgentClass.py: Class for an RMaxAgent from [Brafman and Tennenholtz 2003].

Notes:
    - Assumes WLOG reward function codomain is [0,1] (so RMAX is 1.0)
'''

# Python imports.
import random as r
from collections import defaultdict

# Local classes.
from AgentClass import Agent

class RMaxAgent(Agent):
    '''
    Implementation for an R-Max Agent [Brafman and Tennenholtz 2003]
    '''

    def __init__(self, actions, gamma=0.95, horizon=4, s_a_threshold=10):
        Agent.__init__(self, name="rmax-h" + str(horizon), actions=actions, gamma=gamma)
        self.rmax = 1.0
        self.horizon = horizon
        self.s_a_threshold = s_a_threshold
        self.reset()

    def reset(self):
        '''
        Summary:
            Resets the agent back to its tabula rasa config.
        '''

        self.rewards = defaultdict(list) # keys are (s, a) pairs, value is int.
        self.transitions = defaultdict(lambda : defaultdict(int)) # key is (s, a) pair, val is a dict of <k:states,v:count>
        self.r_s_a_counts = defaultdict(int) # key is (s, a) pair, value is list of ints
        self.t_s_a_counts = defaultdict(int) # key is (s, a) pair, value is list of ints
        self.prev_state = None
        self.prev_action = None

    def get_num_known_sa(self):
        return sum([self.is_known(s,a) for s,a in self.r_s_a_counts.keys()])

    def is_known(self, s, a):
        return self.r_s_a_counts[(s,a)] >= self.s_a_threshold and self.t_s_a_counts[(s,a)] >= self.s_a_threshold

    def act(self, state, reward):
        # Update given s, a, r, s' : self.prev_state, self.prev_action, reward, state
        self.update(self.prev_state, self.prev_action, reward, state)

        # Compute best action.
        action = self.get_max_q_action(state)

        # Update pointers.
        self.prev_action = action
        self.prev_state = state

        return action

    def update(self, state, action, reward, next_state):
        '''
        Args:
            state (State)
            action (str)
            reward (float)
            next_state (State)

        Summary:
            Updates T and R.
        '''
        if state != None and action != None:
            if self.r_s_a_counts[(state, action)] <= self.s_a_threshold:
                # Add new data points if we haven't seen this s-a enough.
                self.rewards[(state, action)] += [reward]
                self.r_s_a_counts[(state, action)] += 1

            if self.t_s_a_counts[(state, action)] <= self.s_a_threshold:
                self.transitions[(state, action)][next_state] += 1
                self.t_s_a_counts[(state, action)] += 1

        # unique_states = set([])
        # for s,a in self.r_s_a_counts.keys():
        #     if s not in unique_states:
        #         unique_states.add(s)

        # print "Num unique states:", len(unique_states)

    def _compute_max_qval_action_pair(self, state, horizon=None):
        '''
        Args:
            state (State)
            horizon (int): Indicates the level of recursion depth for computing Q.

        Returns:
            (tuple) --> (float, str): where the float is the Qval, str is the action.
        '''
        # If this is the first call, use the default horizon.
        if horizon is None:
            horizon = self.horizon 

        # Grab random initial action in case all equal
        best_action = r.choice(self.actions)
        max_q_val = self.get_q_value(state, best_action, horizon)

        # Find best action (action w/ current max predicted Q value)
        for action in self.actions:
            q_s_a = self.get_q_value(state, action, horizon)
            if q_s_a > max_q_val:
                max_q_val = q_s_a
                best_action = action

        return max_q_val, best_action

    def get_max_q_action(self, state, horizon=None):
        '''
        Args:
            state (State)
            horizon (int): Indicates the level of recursion depth for computing Q.

        Returns:
            (str): The string associated with the action with highest Q value.
        '''

        # If this is the first call, use the default horizon.
        if horizon is None:
            horizon = self.horizon 
        return self._compute_max_qval_action_pair(state, horizon)[1]

    def get_max_q_value(self, state, horizon=None):
        '''
        Args:
            state (State)
            horizon (int): Indicates the level of recursion depth for computing Q.

        Returns:
            (float): The Q value of the best action in this state.
        '''

        # If this is the first call, use the default horizon.
        if horizon is None:
            horizon = self.horizon 
        return self._compute_max_qval_action_pair(state, horizon)[0]

    def get_q_value(self, state, action, horizon=None):
        '''
        Args:
            state (State)
            action (str)
            horizon (int): Indicates the level of recursion depth for computing Q.

        Returns:
            (float)
        '''

        # If this is the first call, use the default horizon.
        if horizon is None:
            horizon = self.horizon

        if horizon <= 0 or state.is_terminal():
            # If we're not looking any further.
            return self._get_reward(state, action)

        # Compute future return.
        expected_future_return = self.gamma*self._compute_exp_future_return(state, action, horizon)

        q_val = self._get_reward(state, action) + expected_future_return# self.q_func[(state, action)] = self._get_reward(state, action) + expected_future_return

        return q_val # self.q_func[(state, action)]

    def _compute_exp_future_return(self, state, action, horizon=None):
        '''
        Args:
            state (State)
            action (str)
            horizon (int): Recursion depth to compute Q

        Return:
            (float): Discounted expected future return from applying @action in @state.
        '''

        # If this is the first call, use the default horizon.
        if horizon is None:
            horizon = self.horizon

        next_state_dict = self.transitions[(state, action)]

        denominator = float(sum(next_state_dict.values()))
        state_weights = defaultdict(float)
        for next_state in next_state_dict.keys():
            count = next_state_dict[next_state]
            state_weights[next_state] = (count / denominator)

        weighted_future_returns = [self.get_max_q_value(next_state, horizon-1) * state_weights[next_state] for next_state in next_state_dict.keys()]

        return sum(weighted_future_returns)

    def _get_reward(self, state, action):
        '''
        Args:
            state (State)
            action (str)

        Returns:
            Believed reward of executing @action in @state. If R(s,a) is unknown
            for this s,a pair, return self.rmax. Otherwise, return the MLE.
        '''

        if self.r_s_a_counts[(state, action)] >= self.s_a_threshold:
            # Compute MLE if we've seen this s,a pair enough.
            rewards_s_a = self.rewards[(state, action)]
            return float(sum(rewards_s_a)) / len(rewards_s_a)
        else:
            # Otherwise return rmax.
            return self.rmax

    def _reset_reward(self):
        self.rewards = defaultdict(list) # keys are (s, a) pairs, value is int.
        self.r_s_a_counts = defaultdict(int) # key is (s, a) pair, value is list of ints


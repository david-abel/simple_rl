'''
RMaxAgentClass.py: Class for an RMaxAgent from [Strehl, Li and Littman 2009].

Notes:
    - Assumes WLOG reward function codomain is [0,1] (so RMAX is 1.0)
'''

# Python imports.
import random
import numpy as np
from collections import defaultdict

# Local classes.
from simple_rl.agents.AgentClass import Agent

class RMaxAgent(Agent):
    '''
    Implementation for an R-Max Agent [Strehl, Li and Littman 2009]
    '''

    def __init__(self, actions, gamma=0.95, s_a_threshold=2, epsilon_one=0.99, max_reward=1.0, name="RMax", custom_q_init=None):
        self.name = name 
        Agent.__init__(self, name=name, actions=actions, gamma=gamma)
        self.rmax = max_reward
        self.s_a_threshold = s_a_threshold
        self.custom_q_init = custom_q_init
        self.reset()
        self.custom_q_init = custom_q_init
        self.gamma = gamma
        self.epsilon_one = epsilon_one

        if self.custom_q_init:
            self.q_func = self.custom_q_init
        else:
            self.q_func = defaultdict(lambda: defaultdict(lambda: self.rmax))

    def reset(self):
        '''
        Summary:
            Resets the agent back to its tabula rasa config.
        '''
        self.rewards = defaultdict(lambda : defaultdict(list)) # S --> A --> reward
        self.transitions = defaultdict(lambda : defaultdict(lambda : defaultdict(int))) # S --> A --> S' --> counts
        self.r_s_a_counts = defaultdict(lambda : defaultdict(int)) # S --> A --> #rs
        self.t_s_a_counts = defaultdict(lambda : defaultdict(int)) # S --> A --> #ts
        self.prev_state = None
        self.prev_action = None

        if self.custom_q_init:
            self.q_func = self.custom_q_init
        else:
            self.q_func = defaultdict(lambda: defaultdict(lambda: self.rmax))

    def get_num_known_sa(self):
        return sum([self.is_known(s,a) for s,a in self.r_s_a_counts.keys()])

    def is_known(self, s, a):
        return self.r_s_a_counts[s][a] >= self.s_a_threshold and self.t_s_a_counts[s][a] >= self.s_a_threshold

    def act(self, state, reward):
        # Update given s, a, r, s' : self.prev_state, self.prev_action, reward, state
        self.update(self.prev_state, self.prev_action, reward, state)

        # Compute best action by argmaxing over Q values of all possible s,a pairs
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
            if self.r_s_a_counts[state][action] <= self.s_a_threshold or self.t_s_a_counts[state][action] <= self.s_a_threshold:
                # Add new data points if we haven't seen this s-a enough.
                self.rewards[state][action] += [reward]
                self.r_s_a_counts[state][action] += 1
                self.transitions[state][action][next_state] += 1
                self.t_s_a_counts[state][action] += 1

                if self.r_s_a_counts[state][action] == self.s_a_threshold:
                    # Start updating Q values for subsequent states
                    lim = int(np.log(1/(self.epsilon_one * (1 - self.gamma))) / (1 - self.gamma))
                    for i in range(1, lim):
                        for curr_state in self.rewards.keys():
                            for curr_action in self.actions:
                                if self.r_s_a_counts[curr_state][curr_action] >= self.s_a_threshold:
                                    self.q_func[curr_state][curr_action] = self._get_reward(curr_state, curr_action) + (self.gamma * self.get_transition_q_value(curr_state, curr_action))

    def get_transition_q_value(self, state, action):
        '''
        Args: 
            state
            action 

        Returns:
            empirical transition probability 
        '''
        return sum([(self._get_transition(state, action, next_state) * self.get_max_q_value(next_state)) for next_state in self.q_func.keys()])


    def get_value(self, state):
        '''
        Args:
            state (State)

        Returns:
            (float)
        '''
        return self.get_max_q_value(state)

    def _compute_max_qval_action_pair(self, state):
        '''
        Args:
            state (State)

        Returns:
            (tuple) --> (float, str): where the float is the Qval, str is the action.
        '''
        # Grab random initial action in case all equal
        best_action = random.choice(self.actions)
        max_q_val = self.get_q_value(state, best_action)

        # Find best action (action w/ current max predicted Q value) 
        for action in self.actions:
            q_s_a = self.get_q_value(state, action)
            if q_s_a > max_q_val:
                max_q_val = q_s_a
                best_action = action
        return max_q_val, best_action

    def get_max_q_action(self, state):
        '''
        Args:
            state (State)

        Returns:
            (str): The string associated with the action with highest Q value.
        '''
        return self._compute_max_qval_action_pair(state)[1]

    def get_max_q_value(self, state):
        '''
        Args:
            state (State)

        Returns:
            (float): The Q value of the best action in this state.
        '''
        return self._compute_max_qval_action_pair(state)[0]

    def get_q_value(self, state, action):
        '''
        Args:
            state (State)
            action (str)
        
        Returns:
            (float)
        '''

        return self.q_func[state][action]

    def _get_reward(self, state, action):
        '''
        Args:
            state (State)
            action (str)

        Returns:
            Believed reward of executing @action in @state. If R(s,a) is unknown
            for this s,a pair, return self.rmax. Otherwise, return the MLE.
        '''

        if self.r_s_a_counts[state][action] >= self.s_a_threshold:
            # Compute MLE if we've seen this s,a pair enough.
            rewards_s_a = self.rewards[state][action]
            return float(sum(rewards_s_a)) / len(rewards_s_a)
        else:
            # Otherwise return rmax.
            return self.rmax
    
    def _get_transition(self, state, action, next_state):
        '''
        Args: 
            state (State)
            action (str)
            next_state (str)

            Returns:
                Empirical probability of transition n(s,a,s')/n(s,a) 
        '''

        return self.transitions[state][action][next_state] / self.t_s_a_counts[state][action]

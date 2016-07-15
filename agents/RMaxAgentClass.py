''' RMaxAgentClass.py: Class for an RMaxAgent from [Brafman and Tennenholtz 2003].

Notes:
    - Currently assumes deterministic transitions.
    - Assumes WLOG reward function codomain is [0,1] (so RMAX is 1.0)
'''

# Python libs.
import random
from collections import defaultdict

# Local classes.
from simple_rl.agents.AgentClass import Agent

class RMaxAgent(Agent):
    '''
    Implementation for an R-Max Agent [Brafman and Tennenholtz 2003]
    '''

    def __init__(self, actions, gamma=0.95):
        Agent.__init__(self, name="rmax", actions=actions, gamma=gamma)
        self.rmax = 1.0
        self.reset()

    def reset(self):
        '''
        Summary:
            Resets the agent back to its tabula rasa config.
        '''
        self.rewards = defaultdict(lambda: self.rmax) # keys are (s, a) pairs, value is int.
        self.transitions = defaultdict(lambda: None) # key is (s, a) pair, val is a state (DETERMINISTIC for now)
        self.s_a_counts = defaultdict(int) # key is (s, a) pair, value is int (default 0).
        self.prev_state = None
        self.prev_action = None

    def act(self, state, reward):
        
        if self.prev_state != None and self.prev_action != None:
            # s, a, r, s' : self.prev_state, self.prev_action, reward, state
            self.s_a_counts[(self.prev_state, self.prev_action)] += 1
            self.rewards[(self.prev_state, self.prev_action)] = reward
            self.transitions[(self.prev_state, self.prev_action)] = state

        # Compute best action.
        action = self.get_max_q_action(state)

        self.prev_action = action
        self.prev_state = state


        return action

    def get_max_q_action(self, state):
        '''
        Args:
            state (State)

        Returns:
            (str): the string associated with the action with highest Q value.
        '''
        max_q = float("-inf")
        best_action = None

        # Find Max Q Action
        for action in self.actions:
            q_s_a = self.get_q_value(state, action)

            if q_s_a > max_q:
                max_q = q_s_a
                best_action = action

        return best_action

    def compute_q_value_of_state(self, state, horizon=7):
        '''
        Args:
            state (State)

        Returns:
            (float): max Q value for this state
        '''

        if state is None:
            return (self.rmax * horizon)

        # Grab random initial action in case all equal
        best_action = None
        max_q_val = float("-inf")
        shuffled_action_list = self.actions[:]
        random.shuffle(shuffled_action_list)

        # Find best action (action w/ current max predicted Q value)
        for action in shuffled_action_list:
            q_s_a = self.get_q_value(state, action, horizon)
            if q_s_a > max_q_val:
                max_q_val = q_s_a
                best_action = action

        return max_q_val

    def get_q_value(self, state, action, horizon=7):
        '''
        Args:
            state (State)
            action (str)
            horizon (int): indicates the level of recursion depth for computing Q.

        Returns:
            (float)
        '''

        # If we've hashed a Q value for this already.
        if (state, action) in self.q_func:
            return self.q_func[(state, action)]

        if horizon == 0:
            return self.rewards[(state, action)]

        next_state = self.transitions[(state, action)]

        q_val = self.rewards[(state, action)] + self.compute_q_value_of_state(next_state, horizon=horizon-1)

        return q_val

'''
DoubleQAgentClass.py: Class for an RL Agent acting according to Double Q Learning from:

    Hasselt, H. V. (2010). Double Q-learning.
    In Advances in Neural Information Processing Systems (pp. 2613-2621).

Author: David Abel
'''

# Python imports.
import random
from collections import defaultdict

# Other imports
from simple_rl.agents.QLearningAgentClass import QLearningAgent
from simple_rl.agents.AgentClass import Agent

class DoubleQAgent(QLearningAgent):
    ''' Class for an agent using Double Q Learning. '''

    def __init__(self, actions, name="Double-Q", alpha=0.05, gamma=0.99, epsilon=0.1, explore="uniform", anneal=False):
        '''
        Args:
            actions (list): Contains strings denoting the actions.
            name (str): Denotes the name of the agent.
            alpha (float): Learning rate.
            gamma (float): Discount factor.
            epsilon (float): Exploration term.
            explore (str): One of {softmax, uniform}. Denotes explore policy.
        '''
        QLearningAgent.__init__(self, actions, name=name, alpha=alpha, gamma=gamma, epsilon=epsilon, explore=explore, anneal=anneal)

        # Make two q functions.
        self.q_funcs = {"A":defaultdict(lambda : defaultdict(lambda: self.default_q)), \
                        "B":defaultdict(lambda : defaultdict(lambda: self.default_q))}

    def act(self, state, reward):
        '''
        Args:
            state (State)
            reward (float)

        Summary:
            The central method called during each time step.
            Retrieves the action according to the current policy
            and performs updates.
        '''
        self.update(self.prev_state, self.prev_action, reward, state)
        
        if self.explore == "softmax":
            # Softmax exploration
            action = self.soft_max_policy(state)
        else:
            # Uniform exploration
            action = self.epsilon_greedy_q_policy(state)

        self.prev_state = state
        self.prev_action = action
        self.step_number += 1

        # Anneal params.
        if self.anneal:
            self._anneal()

        return action

    def update(self, state, action, reward, next_state):
        '''
        Args:
            state (State)
            action (str)
            reward (float)
            next_state (State)

        Summary:
            Updates the internal Q Function according to the Double Q update:


        '''
        # If this is the first state, just return.
        if state is None:
            self.prev_state = next_state
            return

        # Randomly choose either "A" or "B".
        which_q_func = "A" if bool(random.getrandbits(1)) else "B"
        other_q_func = "B" if which_q_func is "A" else "A"

        # Update the Q Function.

        # Get max q action of the chosen Q func.
        max_q_action = self.get_max_q_action(next_state, q_func_id=which_q_func)
        prev_q_val = self.get_q_value(state, action, q_func_id=which_q_func)

        # Update
        self.q_funcs[which_q_func][state][action] = (1 - self.alpha) * prev_q_val + self.alpha * (reward + self.gamma * self.get_q_value(next_state, max_q_action, q_func_id=other_q_func))

    def get_max_q_action(self, state, q_func_id=None):
        '''
        Args:
            state (State)
            q_func_id (str): either "A" or "B"

        Returns:
            (str): denoting the action with the max q value in the given @state.
        '''
        return self._compute_max_qval_action_pair(state, q_func_id)[1]

    def get_max_q_value(self, state, q_func_id=None):
        '''
        Args:
            state (State)
            q_func_id (str): either "A" or "B"

        Returns:
            (float): denoting the max q value in the given @state.
        '''
        return self._compute_max_qval_action_pair(state, q_func_id)[0]
    
    def _compute_max_qval_action_pair(self, state, q_func_id=None):
        '''
        Args:
            state (State)
            q_func_id (str): either "A", "B", or None. If None, computes avg of A and B.

        Returns:
            (tuple) --> (float, str): where the float is the Qval, str is the action.
        '''
        # Grab random initial action in case all equal
        best_action = random.choice(self.actions)
        max_q_val = float("-inf")
        shuffled_action_list = self.actions[:]
        random.shuffle(shuffled_action_list)

        # Find best action (action w/ current max predicted Q value)
        for action in shuffled_action_list:
            q_s_a = self.get_q_value(state, action, q_func_id)
            if q_s_a > max_q_val:
                max_q_val = q_s_a
                best_action = action

        return max_q_val, best_action

    def get_q_value(self, state, action, q_func_id=None):
        '''
        Args:
            state (State)
            action (str)
            q_func_id (str): either "A", "B", or defaults to taking the average.

        Returns:
            (float): denoting the q value of the (@state, @action) pair relative to
                the specified q function.
        '''
        if q_func_id is None:
            return self.get_avg_q_value(state, action)
        else:
            return self.q_funcs[q_func_id][state][action]

    def reset(self):
        self.step_number = 0
        self.episode_number = 0
        self.q_funcs = {"A":defaultdict(lambda : defaultdict(lambda: self.default_q)), \
                        "B":defaultdict(lambda : defaultdict(lambda: self.default_q))}
        Agent.reset(self)

    # ---- DOUBLE Q NEW ----

    def get_avg_q_value(self, state, action):
        '''
        Args:
            state (State)
            action (str)

        Returns:
            (float): denoting the avg. q value of the (@state, @action) pair.
        '''
        return (self.q_funcs["A"][state][action] + self.q_funcs["B"][state][action]) / 2.0

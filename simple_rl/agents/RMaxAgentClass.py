'''
RMaxAgentClass.py: Class for an RMaxAgent from [Strehl, Li and Littman 2009].

Notes:
    - Assumes WLOG reward function codomain is [0,1] (so RMAX is 1.0)
'''
import math
import random
from collections import defaultdict

import numpy as np

from simple_rl.agents.AgentClass import Agent


class RMaxAgent(Agent):
    '''
    Implementation for an R-Max Agent [Strehl, Li and Littman 2009]
    https://jmlr.org/papers/volume10/strehl09a/strehl09a.pdf
    pseudocode described in the PDF
    '''

    def __init__(self, 
                states, 
                actions, 
                gamma=0.95, 
                s_a_threshold=2, 
                epsilon_one=0.99, 
                max_reward=1.0, 
                name="RMax", 
                custom_q_init=None):
        """
        Args:
            states: a list of states in the MDP. Simply range(n_states) would suffice
            actions: a list of actions in the MDP. Simply range(n_actions) would suffice
            gamma: discount factor
            s_a_threshold: the number of (s, a) transitions, first seen, to build the model of the MDP
            epsilon_one: param used to decide how many iterations of value iterations to do
            max_reward: the expected maximum reward of the MDP
            name: name of the agent
            custom_q_init: a (n_states, n_actions) numpy array specifying the initial Q values to use
        """
        self.name = name 
        self.states = list(states)
        Agent.__init__(self, name=name, actions=actions, gamma=gamma)
        self.rmax = max_reward
        self.s_a_threshold = s_a_threshold
        self.custom_q_init = custom_q_init
        self.epsilon_one = epsilon_one
        self.max_num_value_iter = math.ceil(np.log(1/(self.epsilon_one * (1 - self.gamma))) / (1 - self.gamma))

        self.reset()

    def reset(self):
        '''
        Summary:
            Resets the agent back to its tabular asa config.
        '''
        self.rewards = np.zeros((len(self.states), len(self.actions)))  # rewards are assumed to be state-action based R(s, a)
        self.transitions = np.zeros((len(self.states), len(self.actions), len(self.states)))
        self.s_a_counts = np.zeros((len(self.states), len(self.actions)))
        self.prev_state = None
        self.prev_action = None

        if self.custom_q_init:
            self.q_func = self.custom_q_init
        else:
            self.q_func = np.ones((len(self.states), len(self.actions))) * self.rmax * 1/(1-self.gamma)

    def act(self, state, reward):
        # Compute best action by argmaxing over Q values of all possible s,a pairs
        action = self.get_max_q_action(state)

        # Update given s, a, r, s' : self.prev_state, self.prev_action, reward, state
        self.update(self.prev_state, self.prev_action, reward, state)

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
            if self.s_a_counts[state][action] < self.s_a_threshold:
                # Add new data points if we haven't seen this s-a enough.
                self.rewards[state][action] += reward
                self.s_a_counts[state][action] += 1
                self.transitions[state][action][next_state] += 1

                if self.s_a_counts[state][action] == self.s_a_threshold:
                    self.value_iteration()

    def value_iteration(self):
        '''
        Do some iterations of value iteration to compute the q values
        Only update the (s, a) pairs that have enough experiences seen
        Q(s, a) = R(s, a) + gamma * \sum_s' T(s, a, s') * max_a' Q(s', a')
        '''
        # mask for update
        mask = self.s_a_counts >= self.s_a_threshold
        pseudo_count = np.where(self.s_a_counts == 0, 1, self.s_a_counts)  # avoid divide by zero

        # build the reward model
        empirical_reward_mat = self.rewards / pseudo_count

        # build the transition model: assume self-loop if there's not enough data
        # assume a self-loop if there's not enough data
        empirical_transition_mat = self.transitions / pseudo_count[:, :, None]
        # only masked positions should be trusted, otherwise self transition

        empirical_transition_mat[~mask] = self._self_transition_mat()[~mask]
        assert np.all(empirical_transition_mat.sum(axis=-1) == 1)

        # compute the update for every (s, a), but only apply the ones that needed with a mask
        for i in range(self.max_num_value_iter):
            v = np.max(self.q_func, axis=-1)
            new_q = empirical_reward_mat + self.gamma * np.einsum("san,n->sa", empirical_transition_mat, v)
            if np.all(np.abs(self.q_func[mask] - new_q[mask]) < 1e-4):
                break
            self.q_func[mask] = new_q[mask]

    def _self_transition_mat(self):
        '''
        create a transition matrix where each state just transition to itself
        '''
        self_transition_mat = np.zeros_like(self.transitions)
        self_transition_mat[np.arange(len(self.states)), :, np.arange(len(self.states))] = 1
        return self_transition_mat

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
        if np.all(self.q_func[state] == self.q_func[state, 0]):
            best_action = random.choice(self.actions)
        else:
            best_action =np.argmax(self.q_func[state])
        max_q_val = self.q_func[state][best_action]

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

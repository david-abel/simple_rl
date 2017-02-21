'''
LinearApproxSarsaAgentClass.py

Contains implementation for a Q Learner with a Linear Function Approximator.
'''

# Local classes
from LinearApproxQLearnerAgentClass import LinearApproxQLearnerAgent

# Python imports.
import numpy as np
import math

class LinearApproxSarsaAgent(LinearApproxQLearnerAgent):
    '''
    Sarsa Agent with a linear function approximator for the Q Function.
    '''

    def __init__(self, actions, name="sarsa-linear", alpha=0.05, gamma=0.95, epsilon=0.01, explore="uniform", rbf=False, anneal=True):
        name = name + "-rbf" if (name == "sarsa-linear" and rbf) else name
        LinearApproxQLearnerAgent.__init__(self, actions=list(actions), name=name, alpha=alpha, gamma=gamma, epsilon=epsilon, explore=explore, anneal=anneal)
        self.num_features = 0
        self.rbf = rbf
        self.weights = None

    def act(self, state, reward):
        '''
        Args:
            state (State)
            reward (float)

        Summary:
            The central update for SARSA.
        '''
        
        if self.weights is None:
            action = np.random.choice(self.actions)
        elif self.explore == "softmax":
            # Softmax exploration
            action = self.soft_max_policy(state)
        else:
            # Uniform exploration
            action = self.epsilon_greedy_q_policy(state)

        self.update(self.prev_state, self.prev_action, reward, state, action)

        self.prev_state = state
        self.prev_action = action
        self.step_number += 1

        # Anneal params.
        if self.anneal and self.step_number % 1000 == 0:
            self._anneal()

        return action

    def update(self, state, action, reward, next_state, next_action):
        '''
        Args:
            state (State)
            action (str)
            reward (float)
            next_state (State)

        Summary:
            Updates the internal Q Function according to the Bellman Equation. (Classic Q Learning update)
        '''
        # If this is the first state, initialize state-relevant data and return.
        if state is None:
            if self.num_features == 0:
                self.num_features = len(next_state.features())
                self.weights = np.zeros(self.num_features*len(self.actions))
            self.prev_state = next_state
            return

        self._update_weights(reward, next_state, next_action)

    def _update_weights(self, reward, curr_state, curr_action):
        '''
        Args:
            reward (float)
            curr_state (State)

        Summary:
            Updates according to:

            [Eq. 1] delta = r + gamma * max_b(Q(s_curr,b)) - Q(s_prev, a_prev)

            For each weight:
                w_i = w_i + alpha * phi(s,a)[i] * delta

            Where phi(s,a) maps the state action pair to a feature vector (see <QLearningAgent>._phi(s,a))
        '''

        # Compute temporal difference [Eq. 1]
        best_q_curr_state = self.get_q_value(curr_state, curr_action)
        prev_q_val = self.get_q_value(self.prev_state, self.prev_action)
        self.most_recent_loss = reward + self.gamma * best_q_curr_state - prev_q_val

        # Update each weight
        phi = self._phi(self.prev_state, self.prev_action)
        active_feats_index = self.actions.index(self.prev_action) * self.num_features

        # Sparsely update the weights (only update weights associated with the action we used).
        for i in xrange(active_feats_index, active_feats_index + self.num_features):
            self.weights[i] = self.weights[i] + self.alpha * phi[i] * self.most_recent_loss

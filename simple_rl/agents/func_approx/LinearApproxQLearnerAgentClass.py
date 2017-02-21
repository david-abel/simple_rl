'''
LinearApproxQLearnerAgentClass.py

Contains implementation for a Q Learner with a Linear Function Approximator.
'''

# Local classes
from ..QLearnerAgentClass import QLearnerAgent

# Python imports.
import numpy
import math

class LinearApproxQLearnerAgent(QLearnerAgent):
    '''
    QLearnerAgent with a linear function approximator for the Q Function.
    '''

    def __init__(self, actions, name="ql-linear", alpha=0.05, gamma=0.95, epsilon=0.01, explore="uniform", rbf=False, anneal=True):
        name = name + "-rbf" if (name == "ql-linear" and rbf) else name
        QLearnerAgent.__init__(self, actions=list(actions), name=name, alpha=alpha, gamma=gamma, epsilon=epsilon, explore=explore, anneal=anneal)
        self.num_features = 0
        self.rbf = rbf

    def update(self, state, action, reward, next_state):
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
                self.weights = numpy.zeros(self.num_features*len(self.actions))
            self.prev_state = next_state
            return

        self._update_weights(reward, next_state)

    def _phi(self, state, action):
        '''
        Args:
            state (State): The abstract state object.
            action (str): A string representing an action. See namespaceAIX.

        Returns:
            (numpy array): A state-action feature vector representing the current State and action.

        Notes:
            The resulting feature vector multiplies the state vector by |A| (size of action space), and only the action passed in retains
            the original vector, all other values are set to namespaceAIX.EMPTYFEATURE
        '''
        blank_vec = [0 for i in xrange(self.num_features * (len(self.actions) - 1))]
        act_index = self.actions.index(action)

        basis_feats = list(state.features())

        if self.rbf:
            basis_feats = [_rbf(f) for f in basis_feats]

        return numpy.array(blank_vec[:act_index*self.num_features] + basis_feats + blank_vec[(act_index)*self.num_features:])

    def _update_weights(self, reward, curr_state):
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
        max_q_curr_state = self.get_max_q_value(curr_state)
        prev_q_val = self.get_q_value(self.prev_state, self.prev_action)
        self.most_recent_loss = reward + self.gamma * max_q_curr_state - prev_q_val

        # Update each weight
        phi = self._phi(self.prev_state, self.prev_action)
        active_feats_index = self.actions.index(self.prev_action) * self.num_features

        # Sparsely update the weights (only update weights associated with the action we used).
        for i in xrange(active_feats_index, active_feats_index + self.num_features):
            self.weights[i] = self.weights[i] + self.alpha * phi[i] * self.most_recent_loss

    def get_q_value(self, state, action):
        '''
        Args:
            state (State): A State object containing the abstract state representation
            action (str): A string representing an action. See namespaceAIX.

        Returns:
            (float): denoting the q value of the (@state,@action) pair.
        '''

        # Return linear approximation of Q value
        sa_feats = self._phi(state, action)
        return numpy.dot(self.weights, sa_feats)

def _rbf(x):
    return math.exp(-(x)**2)
